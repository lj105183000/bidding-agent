"""Exa MCP 客户端：语义联网搜索 + 网页正文抓取。

对接 Exa 的**托管 MCP 服务** `https://mcp.exa.ai/mcp`（无需自建进程）。

★ 为什么这条路存在
----------------
`src/tools/rag_tools.py` 的 `search_exa` 工具**从项目早期就预留了本模块的接口**
（`exa_search_client` 单例 + `search(query, num_results=N)`），
只是一直没落地，所以那个工具始终返回"暂未就绪"。
本模块把插槽填上——`rag_tools.py` **一行都不用改**。

★ 为什么用原生 HTTP 而不是 `mcp` SDK
----------------------------------
实测确认：不装 SDK、手写 JSON-RPC 也能完整走通（见下）。
而 `mcp` SDK 会新增一条依赖链，这个项目对外部依赖一向克制
（`pyproject.toml` 里每个依赖都有明确用途）。
真要用 SDK 也很简单：`uv pip install mcp` 后把 `_call_tool` 换掉即可，
本模块的对外接口不变。

★ 实测踩过的两个协议细节（不写会连不上）
------------------------------------
**① 必须先握手拿 session id**：

    initialize            → 响应头带 `mcp-session-id`
    notifications/initialized  （协议要求的通知，无 id）
    tools/call            → **必须带头** `mcp-session-id`，否则失败

**② 响应是 SSE 包装，不是裸 JSON**：

    event: message
    data: {"jsonrpc":"2.0","id":2,"result":{...}}

直接 `response.json()` 会抛异常。故有 `_parse_sse_json`。

★ 为什么匿名（keyless）就够用
--------------------------
Exa 托管服务提供 keyless 模式（免费限速）。实测 `initialize` 与
`tools/call` 都返回 200，无需任何 key。
需要更高配额时在 `.env` 配 `EXA_API_KEY`，本模块会自动改用 `x-api-key` 头。
`EXA_MCP_URL` 可覆盖服务地址（自建 server 或换镜像时用）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from src.http_client import get_http_client

logger = logging.getLogger(__name__)

#: 托管 MCP 服务地址。可用 `EXA_MCP_URL` 覆盖。
DEFAULT_MCP_URL = "https://mcp.exa.ai/mcp"

#: 结果缓存：
#:
#: ★ 为什么 Exa 比 Tavily/智谱**更**需要缓存
#: ------------------------------------
#: Tavily 与智谱用付费 key，重复调用只是多花钱；而 Exa 走的是
#: `mcp.exa.ai` 的**免费限速**服务——实测**5 次调用内就触限**（见 docs/38）。
#: 没有缓存意味着同一轮问答里模型重复用同一句话调 `search_exa`，
#: 会直接把额度打光；触限后进入 300 秒冷却，该数据源彻底不可用。
#:
#: 参数与 `src/web_search.py` / `src/web_search_zhipu.py` 保持一致
#: （10 分钟 / 512 条），便于统一理解与调整。
CACHE_TTL_SECONDS = 600.0
CACHE_MAX_ITEMS = 512

#: 单次调用超时（秒）。
#:
#: ★ 为什么必须显式设，且**小于全局的 60s**
# ----------------------------------------
#: `http_client.HTTP_TIMEOUT` 的 read 是 60 秒，那是为"外网检索可能很慢"设的兜底。
#: 但 `web_search.SEARCH_TIMEOUT_SECONDS = 25` —— 项目对**单次搜索**的预算是 25 秒。
#: 若这里用全局值，一次挂住的 Exa 调用会把整轮问答拖到 60 秒以上。
#: 故按项目既有预算取 25 秒（与 Tavily 路径一致）。
CALL_TIMEOUT_SECONDS = 25.0

#: 握手超时（秒）。握手只传协议元信息，很快；单独设短超时以便快速失败。
HANDSHAKE_TIMEOUT_SECONDS = 10.0

#: 连续失败后暂停调用的时长（秒）。
#:
#: ★ 为什么需要它
#: -----------
#: Exa keyless 是**免费限速**的，撞到 429 后若继续每个问题都重试，
#: 每次都要白等一次超时（25 秒），把问答拖慢且刷满日志。
#: 故失败后进入冷却期，期间直接返回失败（由 `search_exa` 转成可读文案）。
#: 取值 300 秒：足够让限速窗口过去，又不至于长时间不恢复。
FAILURE_COOLDOWN_SECONDS = 300.0

#: MCP 协议版本。握手时声明，服务端据此决定响应格式。
PROTOCOL_VERSION = "2025-06-18"

#: 单次返回条数上限。与 `rag_tools.WEB_MAX_RESULTS` 对齐，避免调用方传超。
MAX_RESULTS = 10


class ExaMCPError(RuntimeError):
    """Exa MCP 调用失败（网络 / 协议 / 限速）。"""


def _parse_sse_json(text: str) -> dict | None:
    """从 SSE 包装里剥出 JSON-RPC 载荷。

    streamable HTTP 传输的响应形如：

        event: message
        data: {"jsonrpc":"2.0","id":2,"result":{...}}

    也兼容裸 JSON（部分实现或非流式响应）。

    逐块扫描而不是简单 `split("\\n")`：一次响应里可能有多个 event 块，
    且 `data:` 可能是多行拼接（SSE 规范允许）。
    """
    if not text:
        return None
    # 裸 JSON 先试一次（最省事，也覆盖非流式实现）
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    data_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif not line and data_lines:
            # 空行 = 一个 event 结束，尝试解析已累积的 data
            try:
                obj = json.loads("\n".join(data_lines))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            data_lines = []
    if data_lines:
        try:
            obj = json.loads("\n".join(data_lines))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


class ExaMCPClient:
    """Exa MCP 客户端（进程内单例，见文件末尾 `exa_search_client`）。

    线程安全：工具层是 4 线程并行执行（ToolRunner / react_loop），
    同一个单例会被并发调用，故 session 与冷却状态的读写全部持锁。
    **HTTP 调用在锁外进行**——否则一次慢搜索会把所有并发搜索串行化
    （与 `web_search.WebSearchClient` 同一考虑）。
    """

    def __init__(self, url: str = "", api_key: str = "") -> None:
        self._url = url or os.getenv("EXA_MCP_URL", "").strip() or DEFAULT_MCP_URL
        self._api_key = api_key or os.getenv("EXA_API_KEY", "").strip()
        self._lock = threading.Lock()
        self._session_id: str = ""
        # 冷却截止时间戳（0 表示不在冷却）
        self._cooldown_until: float = 0.0
        # 握手失败原因，仅用于日志去重
        self._last_error: str = ""
        # 结果缓存：key → (过期时间戳, 载荷)。与 `web_search` 同构。
        self._cache: dict[tuple[str, int], tuple[float, dict]] = {}
        # 命中/未命中计数（供诊断；与检索缓存的口径一致）
        self._cache_hits = 0
        self._cache_misses = 0

    # --- 缓存 ---

    def _evict_locked(self, now: float) -> None:
        """清理过期项；仍满则丢掉最早过期的。

        与 `web_search._evict_locked` 同一策略。调用方必须已持锁。
        """
        expired = [k for k, (expire_at, _) in self._cache.items() if expire_at <= now]
        for k in expired:
            self._cache.pop(k, None)
        if len(self._cache) < CACHE_MAX_ITEMS:
            return
        # 只丢到上限的八成，避免每次插入都触发一次全表排序
        for k, _ in sorted(self._cache.items(), key=lambda kv: kv[1][0])[
            : len(self._cache) - int(CACHE_MAX_ITEMS * 0.8)
        ]:
            self._cache.pop(k, None)

    def _cache_get(self, key: tuple[str, int]) -> dict | None:
        """命中且未过期时返回**深拷贝**，否则 None。

        深拷贝不是可选项：`rag_tools._rerank_web_results` 会往结果里写
        `score` / `rerank_score`，共享同一份对象会把改写带进缓存，
        后续命中就拿到被污染的数据。
        """
        import copy

        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._cache_misses += 1
                return None
            expire_at, payload = entry
            if expire_at <= now:
                self._cache.pop(key, None)
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return copy.deepcopy(payload)

    def _cache_put(self, key: tuple[str, int], payload: dict) -> None:
        """只缓存**成功**结果。

        失败结果不进缓存：Exa 是免费限速服务，把"限速/超时"缓存 10 分钟
        等于把故障固化，反而更难恢复（与 `web_search` 的取舍一致）。
        """
        import copy

        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            self._cache[key] = (now + CACHE_TTL_SECONDS, copy.deepcopy(payload))

    def clear_cache(self) -> None:
        """清空结果缓存（诊断与"强制刷新"用）。"""
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def clear_cooldown(self) -> None:
        """解除失败冷却。

        ★ 为什么需要它
        -----------
        冷却（300 秒）在**失败之后**生效，于是失败路径之后无法再测任何东西——
        第二次调用会被冷却拦住，看到的不是"缓存有没有生效"而是"冷却在生效"。
        实测写缓存测试时就被这一点绊住（断言 `calls == 2` 却得到 1）。

        生产上也有用：排障时确认限速已过、想立刻重试而不等 5 分钟。
        """
        with self._lock:
            self._cooldown_until = 0.0
            self._session_id = ""

    def cache_info(self) -> dict[str, Any]:
        """缓存状态（供诊断接口展示）。"""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            return {
                "size": len(self._cache),
                "max_items": CACHE_MAX_ITEMS,
                "ttl_seconds": CACHE_TTL_SECONDS,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": round(self._cache_hits / total, 4) if total else 0.0,
            }

    # --- 对外 ---

    def available(self) -> bool:
        """是否可用（供健康检查/诊断用）。不做网络请求。

        只反映**本地状态**：是否在冷却中、有无凭据需求。
        真正的可用性以一次 `search` 调用的结果为准。
        """
        with self._lock:
            return time.monotonic() >= self._cooldown_until

    def search(self, query: str, num_results: int = 5) -> dict[str, Any]:
        """语义联网搜索。

        返回 `{"results": [{title, url, content}], "answer": str}`。

        ★ `answer` 恒为空串：Exa 的 `web_search_exa` 工具不返回 answer 字段
        （Tavily 才返回）。保留该键是为了满足 `rag_tools.search_exa` 约定的
        **统一返回形状**——它按 `payload.get("answer", "")` 取值，
        键不存在与值为空串对它是等价的，但显式给出更利于日后排查。

        失败抛 `ExaMCPError`（由调用方 `search_exa` 转成可读文案）。
        """
        q = (query or "").strip()
        if not q:
            raise ExaMCPError("查询为空")

        # ⚠️ 不能用 `num_results or 5`：`0` 是假值，会被当成"未提供"而变成 5，
        # 于是**钳位本意失效**（调用方传 0 想要 1 条，却拿到 5 条）。
        # 实测由单测抓到。这里区分 None（未提供→默认）与 0（显式→钳到下限）。
        default = 5
        try:
            raw = default if num_results is None else int(num_results)
        except (TypeError, ValueError):
            raw = default
        limit = max(1, min(raw, MAX_RESULTS))

        # ★ 缓存键必须含 `limit`（与 Tavily/智谱**不同**）
        # ------------------------------------------
        # Tavily 的键只有 query，因为它把条数交给服务端默认；
        # 而这里的 `numResults` 直接决定返回几条。若键里不含它，
        # 先查 3 条再查 5 条会命中同一项，**拿到 3 条却以为要的是 5 条**——
        # 这类 bug 不会报错，只会静默少给数据。
        #
        # `q` 已经 strip 过（L184），故这里再 lower 即可覆盖大小写变体。
        cache_key = (q.lower(), limit)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Exa 搜索命中缓存：%s（%d 条）", q, limit)
            return cached

        self._ensure_not_cooling()

        # ⚠️ `_extract_results` 必须留在 try 内。
        #
        # 实测踩过的 bug：第一版只把 `_call_tool` 包在 try 里，而**限速是在
        # 解析阶段被识别的**（Exa 把限速文案当正常结果返回，`_tool_text` 才认出它）。
        # 于是异常在 `return {... _extract_results(payload) ...}` 处抛出，
        # **绕过了下面的 except，冷却永远不生效**——表现为"限速后每个问题
        # 都要重新等一次超时"，且 `available()` 恒为 True。
        #
        # 教训：守护块要覆盖"整条可能抛错链"，而不只是最外层那次网络调用。
        try:
            payload = self._call_tool("web_search_exa", {"query": q, "numResults": limit})
            records = self._extract_results(payload)
        except ExaMCPError:
            self._enter_cooldown()
            raise
        except Exception as exc:  # 网络层等未预期异常
            self._enter_cooldown()
            raise ExaMCPError(f"Exa MCP 调用失败: {type(exc).__name__}: {exc}") from exc

        # 只在这里写缓存：上面所有失败路径都已经 raise 出去了。
        # 顺序很重要——若把 `_cache_put` 放在 try 内或提前，失败结果会被缓存，
        # 而 Exa 的失败（限速）一旦被固化 10 分钟就特别难恢复。
        result = {"results": records, "answer": ""}
        self._cache_put(cache_key, result)
        return result

    def fetch(self, urls: list[str], max_characters: int = 8000) -> dict[str, Any]:
        """抓取网页**全文**（clean markdown）。

        ★ 为什么提供它（实测量出的缺口）
        -----------------------------
        Tavily `search` 返回的 content 被 `CONTENT_TRUNCATE_CHARS = 500` 截断，
        实测平均仅 **482 字/条**；而招标公告全文常达数千字，
        **资格条件、评分办法、工期都在正文中后段**——摘要看不到。
        实测 Tavily `extract` 抓到 8597 字/条（17.8x）。

        本方法暂未挂到 Agent 工具（那需要一个新工具），先作为能力就位，
        供后续按需接入。

        返回 `{"results": [{url, content}]}`。
        """
        targets = [str(u).strip() for u in (urls or []) if str(u).strip()]
        if not targets:
            raise ExaMCPError("未提供 URL")
        self._ensure_not_cooling()
        try:
            payload = self._call_tool(
                "web_fetch_exa", {"urls": targets, "maxCharacters": max_characters}
            )
            records = self._extract_fetched(payload)
        except ExaMCPError:
            self._enter_cooldown()
            raise
        except Exception as exc:
            self._enter_cooldown()
            raise ExaMCPError(f"Exa fetch 失败: {type(exc).__name__}: {exc}") from exc
        return {"results": records}

    # --- 冷却 ---

    def _ensure_not_cooling(self) -> None:
        with self._lock:
            remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            raise ExaMCPError(
                f"Exa MCP 处于失败冷却期（约 {remaining:.0f} 秒后恢复）"
            )

    def _enter_cooldown(self) -> None:
        with self._lock:
            self._cooldown_until = time.monotonic() + FAILURE_COOLDOWN_SECONDS
            # session 可能已失效，下次重新握手
            self._session_id = ""

    # --- 结果提取 ---

    @staticmethod
    def _tool_text(payload: dict | None) -> str:
        """取出 tools/call 返回的文本内容。

        MCP 的 `tools/call` 结果形如：
            {"result": {"content": [{"type": "text", "text": "..."}], "isError": false}}

        文本本身可能是**人类可读的排版文本**而非 JSON——Exa 的
        `web_search_exa` 返回的就是带标题/URL/正文的 markdown 式文本。

        ★ 限速文案也走这里，必须显式识别（实测踩坑）
        -----------------------------------------
        Exa 的免费限速**不返回 JSON-RPC error，也不返回 HTTP 429**，
        而是把提示当正常结果返回：

            "You've hit Exa's free MCP rate limit. To continue using without
             limits, create your own Exa API key. ..."

        第一版把它当普通文本走了"兜底：整段作为一条结果"的分支，
        于是**限速被伪装成一条搜索结果**（标题"Exa 搜索结果"、正文是限速提示）。
        更糟的是连续调用会拿到同一段文本，看起来像"结果重复"，
        把排查方向引到解析逻辑上（实测因此白查一轮）。
        """
        result = (payload or {}).get("result") or {}
        if result.get("isError"):
            raise ExaMCPError(f"Exa 工具返回错误: {str(result)[:200]}")
        parts = [
            str(c.get("text") or "")
            for c in (result.get("content") or [])
            if isinstance(c, dict)
        ]
        text = "\n".join(p for p in parts if p)
        lowered = text.lower()
        if "rate limit" in lowered or "exa api key" in lowered:
            raise ExaMCPError(f"Exa MCP 免费额度已触限：{text[:160]}")
        return text

    @staticmethod
    def _dedupe(items: list[dict]) -> list[dict]:
        """按 `(title, url)` 去重，保留**正文最长**的那条。

        ★ 为什么必须去重（实测发现）
        -------------------------
        Exa 返回的文本里，同一条结果可能**出现多次**——它把网页正文也一并带回，
        而正文开头常重复一次标题/URL，于是解析出两条内容高度重叠的块。

        实测：`search('安徽省 安装工程 招标公告')` 的第 1、2 条 title/url/content
        **完全相同**，在结果里并列显示两次。

        去重放在解析后而不是解析中：解析阶段无法判断"是重复还是同一页面的
        不同片段"，而保留"正文更长的那条"能自动留下信息量更大的版本。
        """
        best: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("title") or "").strip(), str(item.get("url") or "").strip())
            if not any(key):
                continue
            if key not in best:
                best[key] = item
                order.append(key)
                continue
            if len(str(item.get("content") or "")) > len(str(best[key].get("content") or "")):
                best[key] = item
        return [best[k] for k in order]

    def _extract_results(self, payload: dict | None) -> list[dict]:
        """把 `web_search_exa` 的返回转成 `[{title, url, content}]`。

        ★ 为什么做多形态兜底
        -----------------
        MCP 只约定"content 是文本块"，**不约定文本内部的格式**。
        实测 Exa 返回的是形如

            Title: xxx
            URL: https://...
            Published: N/A
            Highlights:
            <正文>

            ---

        的排版文本；但不同版本可能返回 JSON 数组或纯 markdown 链接列表。
        故按优先级依次尝试：JSON 数组 → 结构化文本块 → 正则兜底。
        **任一形态解析失败都不抛异常**，最多返回空列表——
        让 `search_exa` 走"没有返回结果"的可读文案，而不是崩掉。
        """
        text = self._tool_text(payload)
        if not text:
            return []

        # 形态一：JSON（数组或 {"results": [...]}）
        stripped = text.strip()
        if stripped[:1] in "[{":
            try:
                obj = json.loads(stripped)
                items = obj.get("results") if isinstance(obj, dict) else obj
                if isinstance(items, list):
                    out = [self._normalize_item(it) for it in items if isinstance(it, dict)]
                    out = self._dedupe([x for x in out if x])
                    if out:
                        return out
            except json.JSONDecodeError:
                pass

        # 形态二/三：排版文本。正则一次抓出 (title, url) 对，
        # 正文取到下一个标题之前。
        import re

        pattern = re.compile(
            r"^\s*(?:Title|标题)\s*[:：]\s*(?P<title>.+?)\s*$"
            r"|^\s*(?:URL|Url|Link|链接)\s*[:：]\s*(?P<url>https?://\S+)\s*$",
            re.MULTILINE,
        )
        blocks: list[dict] = []
        current: dict = {}
        last_end = 0
        for m in pattern.finditer(text):
            if current:
                body = text[last_end : m.start()].strip()
                current["content"] = body
                blocks.append(current)
            last_end = m.end()
            if m.group("title"):
                current = {"title": m.group("title").strip(), "url": "", "content": ""}
            elif m.group("url") and current:
                current["url"] = m.group("url").strip()
        if current:
            current["content"] = text[last_end:].strip()
            blocks.append(current)

        out = self._dedupe([b for b in (self._normalize_item(b) for b in blocks) if b])
        if out:
            return out

        # 最后兜底：整段作为一条结果。宁可给一条"没有标题的结果"，
        # 也不要让工具说"没有返回结果"——内容其实拿到了。
        urls = re.findall(r"https?://[^\s\)\]\"'，,]+", text)
        return [{
            "title": "Exa 搜索结果",
            "url": urls[0] if urls else "",
            "content": text,
        }]

    @staticmethod
    def _normalize_item(item: dict) -> dict | None:
        """统一成 `{title, url, content}`；无内容无链接的丢弃。"""
        if not isinstance(item, dict):
            return None
        title = str(
            item.get("title") or item.get("name") or item.get("heading") or ""
        ).strip()
        url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
        content = str(
            item.get("content")
            or item.get("text")
            or item.get("snippet")
            or item.get("highlights")
            or item.get("summary")
            or ""
        ).strip()
        if not (title or url or content):
            return None
        # `_web_sources` 要求至少有 url 或 title 之一才收
        if not url and not title:
            return None
        return {"title": title, "url": url, "content": content}

    def _extract_fetched(self, payload: dict | None) -> list[dict]:
        """把 `web_fetch_exa` 的返回转成 `[{url, content}]`。"""
        text = self._tool_text(payload)
        if not text:
            return []
        stripped = text.strip()
        if stripped[:1] in "[{":
            try:
                obj = json.loads(stripped)
                items = obj.get("results") if isinstance(obj, dict) else obj
                if isinstance(items, list):
                    out = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        url = str(it.get("url") or it.get("id") or "").strip()
                        content = str(
                            it.get("content") or it.get("text") or it.get("raw_content") or ""
                        ).strip()
                        if url or content:
                            out.append({"url": url, "content": content})
                    if out:
                        return out
            except json.JSONDecodeError:
                pass
        import re

        urls = re.findall(r"https?://[^\s\)\]\"'，,]+", text)
        return [{"url": urls[0] if urls else "", "content": text}]

    # --- MCP 传输 ---

    def _headers(self, with_session: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # 两个 Accept 都要给：streamable HTTP 可能回 SSE 也可能回 JSON，
            # 只声明一个会被服务端拒绝。
            "Accept": "application/json, text/event-stream",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if with_session:
            with self._lock:
                sid = self._session_id
            if sid:
                headers["mcp-session-id"] = sid
        return headers

    def _post(self, body: dict, timeout: float) -> tuple[int, str, dict]:
        client = get_http_client()
        resp = client.post(self._url, json=body, headers=self._headers(), timeout=timeout)
        return resp.status_code, resp.text, dict(resp.headers)

    def _ensure_session(self) -> None:
        """确保已握手。已握手则直接返回。"""
        with self._lock:
            if self._session_id:
                return
        self._handshake()

    def _handshake(self) -> None:
        """走 `initialize` → `notifications/initialized`，保存 session id。

        为什么不复用已有 session：服务端可能已回收（session 有生命周期），
        而**仅凭本地记录无法判断**。故失败时重新握手一次（见 `_call_tool` 的重试）。
        """
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "bidding-agent", "version": "0.1.0"},
            },
        }
        status, text, headers = self._post(init, HANDSHAKE_TIMEOUT_SECONDS)
        if status != 200:
            raise ExaMCPError(f"握手失败 HTTP {status}: {text[:200]}")
        sid = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id") or ""
        with self._lock:
            self._session_id = sid
        if not sid:
            # 无 session 的实现也要能用（部分 server 是无状态的）
            logger.info("Exa MCP 握手未返回 session id，按无状态模式调用")

        # 协议要求的通知（无 id，故无响应）。失败不影响后续调用，仅记日志。
        try:
            self._post(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                HANDSHAKE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.debug("发送 initialized 通知失败（不影响调用）: %s", exc)

    def _call_tool(self, name: str, arguments: dict) -> dict | None:
        """调用 MCP 工具。会话失效时自动重新握手并重试一次。"""
        self._ensure_session()
        body = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        status, text, _ = self._post(body, CALL_TIMEOUT_SECONDS)

        # 会话过期：服务端返回 404 或 400。重新握手一次再试——
        # 这是 MCP 的常规恢复路径，不重试会让第一次调用之后的请求全失败。
        if status in (400, 404):
            logger.info("Exa MCP 会话可能已失效（HTTP %s），重新握手后重试", status)
            with self._lock:
                self._session_id = ""
            self._handshake()
            status, text, _ = self._post(body, CALL_TIMEOUT_SECONDS)

        if status == 429:
            raise ExaMCPError("Exa MCP 触发限速（HTTP 429）")
        if status != 200:
            raise ExaMCPError(f"tools/call 失败 HTTP {status}: {text[:200]}")

        payload = _parse_sse_json(text)
        if payload is None:
            raise ExaMCPError(f"响应无法解析为 JSON-RPC: {text[:200]}")
        if payload.get("error"):
            raise ExaMCPError(f"JSON-RPC 错误: {str(payload['error'])[:200]}")
        msg = f"Exa MCP {name} 调用成功"
        with self._lock:
            if self._last_error:
                logger.info("%s（此前失败已恢复）", msg)
                self._last_error = ""
        return payload


#: 模块级单例。`rag_tools.search_exa` 按约定从本模块导入它。
exa_search_client = ExaMCPClient()


__all__ = [
    "CALL_TIMEOUT_SECONDS",
    "DEFAULT_MCP_URL",
    "FAILURE_COOLDOWN_SECONDS",
    "ExaMCPClient",
    "ExaMCPError",
    "exa_search_client",
]
