"""Tavily 联网搜索客户端（关键词搜索，带进程内缓存）。

用途
----
为 `search_web` 工具提供原始联网结果。本模块**只负责把数据拿回来**——不做格式化、
不认识 sources、不知道工具名的存在；把结果翻译成给 LLM 的文本是 `src/tools/rag_tools.py` 的事。
这条边界让缓存、超时、出站策略能独立测试，也让格式化器可以脱离网络单测。

为什么直连 REST 而不用 `tavily-python`
--------------------------------------
`tavily-python` 的同步客户端内部自建 `requests.Session`，会绕过本项目统一的
`trust_env=False` 出站策略（详见 `src/http_client.py` 顶部关于系统代理的说明）；
它还支持"无密钥免密模式"——`api_key` 为空时不报错，而是静默切到限流的匿名通道，
让"没配密钥"表现为"搜得到但结果很少"，与 `.env` 缺失即不可用的约定冲突。
直连 REST 只有一个 POST，换取的是出站策略统一、密钥缺失显式失败、缓存与截断全可控。

缓存为什么必须返回深拷贝
------------------------
`_rerank_web_results`（rag_tools）会往结果的 `score` 字段写归一化后的分数。若把缓存里的
对象直接交出去，重排结果会被写回缓存，后续命中同一 query 的请求会拿到"上一轮重排过的分数"。
"""

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# 顶层导入是安全的：src.http_client 只依赖 httpx 与标准库，不反向依赖任何业务模块。
# 出站策略（trust_env=False / 超时 / 连接池 / UA）只在那一个地方定义，本模块不另起一套。
from src.http_client import get_http_client

# 与 src/database/*.py 同一处理：本模块要能被脚本、`python -c`、测试单独使用，
# 而集中加载 .env 的 src/config.py 尚未落地。load_dotenv 默认不覆盖已有环境变量，
# 故不影响显式传参或 CI 里预设的变量。
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

# --- 配置 ---

API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# 搜索超时必须**短于工具层 30s 总超时**（src/agent/constants.py 的 TOOL_TIMEOUT_SECONDS）：
# 留出余量才能在超时前自己失败并给出可读文案，否则外层先掐断，用户只看到"工具执行超时"，
# 分不清是网络慢还是关键词太难。连接单独给 10s，不可达的域名要快速失败。
SEARCH_TIMEOUT_SECONDS = 25.0
CONNECT_TIMEOUT_SECONDS = 10.0

# --- 缓存 ---

# TTL 与容量取自文档（组件工作机制.md §16）：时效性信息 10 分钟内复用在招投标场景够用，
# 512 条足以覆盖一轮演示的全部提问，又不至于让进程内存无限增长。
CACHE_TTL_SECONDS = 600.0
CACHE_MAX_ITEMS = 512

# --- 结果裁剪 ---

# 单条正文与 AI 摘要都要截断：工具结果会原样进 LLM 上下文，不截断时 5 条长网页
# 就可能吃掉整个上下文预算（组件工作机制.md §16 要求单条截断 500 字符）。
CONTENT_TRUNCATE_CHARS = 500
ANSWER_TRUNCATE_CHARS = 1000

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10

# 只有这两档。advanced 更全但更慢更贵，默认 basic——招投标问答里绝大多数查询
# 用关键词就够，需要深度时由 LLM 显式指定。
ALLOWED_DEPTHS = ("basic", "advanced")
DEFAULT_DEPTH = "basic"


class WebSearchError(Exception):
    """联网搜索失败（密钥缺失 / 网络错误 / 非 200 / 响应无法解析）。

    只抛这一种类型，让调用方（工具执行器）能用一次 except 收敛所有失败分支，
    并转成用户可见的中文文案——原始异常与响应体只进日志。
    """


def is_configured() -> bool:
    """是否配置了 Tavily 密钥。

    每次调用都读环境变量而不是在构造时缓存：测试用 monkeypatch.setenv 注入，
    部署时也可能在进程启动后才补上 .env，读缓存会让两种情况都失效。
    """
    return bool(os.getenv(API_KEY_ENV, "").strip())


def _truncate(text: str, limit: int) -> str:
    """按字符截断并补省略号。中文按字符计，不做字节换算。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _safe_max_results(value) -> int:
    """把调用方（多半是 LLM）给的条数收敛成合法整数：非法退回默认值，超限截断。

    与 postgresql_client._safe_limit 同一套路：外部数值不可信——字符串或浮点会被
    上游当成合法参数一路带下来，在真正发请求前挡掉比拿 400 响应更省事。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS
    return max(1, min(number, MAX_RESULTS_LIMIT))


def _normalize_items(raw) -> list[dict]:
    """把 Tavily 的 results 归一成固定形状。

    只保留本工具真正要用的四个字段，且**只在下游需要的键上做类型兜底**：
    Tavily 的字段偶尔缺失（例如某些站点没有 content），格式化器不该为此写一堆判空。
    """
    items: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        title = str(entry.get("title") or "").strip()
        # 既没链接也没标题的条目无法引用，丢掉比留在结果里更诚实
        if not url and not title:
            continue
        items.append(
            {
                "title": title or url,
                "url": url,
                "content": _truncate(str(entry.get("content") or ""), CONTENT_TRUNCATE_CHARS),
                "score": entry.get("score"),
            }
        )
    return items


class WebSearchClient:
    """Tavily 搜索客户端：进程内缓存 + 统一出站 + 失败只抛 WebSearchError。

    线程安全：工具层是 4 线程并行执行（ToolRunner / react_loop），同一个单例会被并发调用，
    因此缓存的读写全部持锁。锁只护内存字典，HTTP 调用在锁外进行——否则一次慢搜索
    会把所有并发搜索串行化。
    """

    def __init__(self) -> None:
        # key = query.strip().lower()，value = (过期时间戳, 载荷)
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    # --- 缓存 ---

    def _evict_locked(self, now: float) -> None:
        """清过期项；仍然满则按时间戳丢掉最旧的一半。调用方必须已持锁。

        为什么"删一半"而不是逐条 LRU：搜索结果的访问分布很集中（同一个问题在一轮
        问答里被反复问），一次腾出较大空间比每次插入都做一次淘汰更省事，也避免
        刚好在容量边界上抖动。
        """
        for key in [k for k, (expire_at, _) in self._cache.items() if expire_at <= now]:
            self._cache.pop(key, None)
        if len(self._cache) < CACHE_MAX_ITEMS:
            return
        oldest = sorted(self._cache.items(), key=lambda kv: kv[1][0])
        for key, _ in oldest[: max(1, len(oldest) // 2)]:
            self._cache.pop(key, None)

    def _cache_get(self, key: str) -> dict | None:
        """命中且未过期时返回**深拷贝**（理由见模块 docstring），否则返回 None。"""
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expire_at, payload = entry
            if expire_at <= now:
                self._cache.pop(key, None)
                return None
            return copy.deepcopy(payload)

    def _cache_put(self, key: str, payload: dict) -> None:
        """只缓存成功结果。失败结果缓存会在故障期间把错误固化 10 分钟，反而更难恢复。"""
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            self._cache[key] = (now + CACHE_TTL_SECONDS, copy.deepcopy(payload))

    def clear_cache(self) -> None:
        """清空缓存（测试与排障用）。"""
        with self._lock:
            self._cache.clear()

    # --- 状态 ---

    def is_configured(self) -> bool:
        """是否有可用密钥（实例方法，便于调用方不关心模块级函数）。"""
        return is_configured()

    def health(self) -> dict:
        """给健康检查用的只读状态：**不触发任何网络请求**。

        为什么不像数据库客户端那样真探活：联网搜索是可选数据源，健康检查若去调 Tavily
        会消耗配额、也会让 /api/health 的耗时取决于外网。这里只报告"配没配密钥 + 缓存条数"。
        """
        with self._lock:
            cached = len(self._cache)
        configured = is_configured()
        return {
            "ok": configured,
            "configured": configured,
            "cached_queries": cached,
            "error": "" if configured else f"未配置 {API_KEY_ENV}",
        }

    # --- 搜索 ---

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        depth: str = DEFAULT_DEPTH,
        include_answer: bool = True,
    ) -> dict:
        """关键词搜索，返回 `{"results": [{title, url, content, score}], "answer": str}`。

        - 空 query 返回空结果而不是抛异常：没有关键词就没有可搜的东西，属于"无结果"而非"故障"。
        - 密钥缺失 / 网络错误 / 非 200 / 响应无法解析 → 抛 `WebSearchError`。
          调用方据此区分"搜不到"（空列表）与"搜不了"（异常），给出不同文案。
        """
        query = (query or "").strip()
        if not query:
            return {"results": [], "answer": ""}

        key = query.lower()
        cached = self._cache_get(key)
        if cached is not None:
            logger.info("联网搜索命中缓存：%s", query)
            return cached

        if not is_configured():
            # 显式失败而不是走 tavily-python 那种免密匿名通道：让"没配密钥"
            # 在日志和用户可见文案里都表现为"不可用"，而不是"结果偏少"
            raise WebSearchError(f"未配置 {API_KEY_ENV}")

        payload = {
            "query": query,
            "search_depth": depth if depth in ALLOWED_DEPTHS else DEFAULT_DEPTH,
            "max_results": _safe_max_results(max_results),
            "include_answer": include_answer,
        }
        headers = {
            "Authorization": f"Bearer {os.getenv(API_KEY_ENV, '').strip()}",
            "Content-Type": "application/json",
        }

        try:
            # 复用全局出站客户端（trust_env=False 等策略集中在那里）。逐请求覆盖超时，
            # 见 SEARCH_TIMEOUT_SECONDS 的说明。
            response = get_http_client().post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(SEARCH_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            # 原始异常可能带 URL 与网络细节，只进日志
            logger.warning("Tavily 请求失败（%s）：%s", type(exc).__name__, exc)
            raise WebSearchError("联网搜索请求失败") from exc

        if response.status_code != 200:
            # 响应体可能含账号信息，只记状态码
            logger.warning("Tavily 返回非 200：%s", response.status_code)
            raise WebSearchError(f"联网搜索服务返回 {response.status_code}")

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Tavily 响应无法解析为 JSON")
            raise WebSearchError("联网搜索响应无法解析") from exc
        if not isinstance(body, dict):
            logger.warning("Tavily 响应不是 JSON 对象（%s）", type(body).__name__)
            raise WebSearchError("联网搜索响应格式异常")

        result = {
            "results": _normalize_items(body.get("results")),
            "answer": _truncate(str(body.get("answer") or ""), ANSWER_TRUNCATE_CHARS),
        }
        self._cache_put(key, result)
        logger.info("联网搜索完成：%s（%d 条）", query, len(result["results"]))
        return result


# 模块级单例：与 CLAUDE.md 的单例清单（settings / embedder / web_search_client ...）同名。
# 这里可以**直接构造**而不必懒加载加锁——构造函数只建一个空字典和一把锁，不做任何 I/O；
# 真正昂贵的资源（httpx 连接池、密钥）分别在首次请求和每次调用时才解析。
web_search_client = WebSearchClient()
