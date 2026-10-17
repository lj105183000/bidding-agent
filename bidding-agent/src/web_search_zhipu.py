"""智谱 Web Search 客户端（中文关键词搜索，带进程内缓存）。

为什么需要它（与 Tavily 的互补关系）
------------------------------------
项目原有的 `search_web`（Tavily）强在**英文语料 + AI 摘要**，但对中文招投标场景有两个短板：

    1. 招投标公告大量发布在省市政府采购网、地方公共资源交易中心等**中文站点**，
       Tavily 的索引对这类站点的覆盖与时延都不理想；
    2. 它不返回**发布时间**。而招投标问答里"这条公告是不是最新的"经常是判断题的关键，
       没有日期就只能靠正文里碰运气。

智谱 Web Search 恰好补上这两点：返回中文结果，且**每条都带 `publish_date` 与 `media`（来源媒体）**，
另有 `search_intent`（意图分析，能看出服务端把查询理解成了什么关键词）。

两者不是替代而是互补，故本模块**独立于 `src/web_search.py`**，作为第二个搜索数据源并存。

架构约定（与 src/web_search.py 保持一致）
----------------------------------------
- 本模块**只负责把数据拿回来**，不做给 LLM 的文本格式化、不认识 sources —— 那是 rag_tools 的事。
- 失败只抛 `ZhipuSearchError` 一种异常，调用方一次 except 收敛所有失败分支。
- 未配置密钥时**显式失败**，不走任何免密匿名通道（避免"没配密钥"表现为"结果偏少"）。
- 缓存必须返回**深拷贝**：下游 `_rerank_web_results` 会往结果的 `score` 字段写归一化分数，
  直接交出缓存对象会让重排结果被写回缓存。

出站策略
--------
复用 `src.http_client.get_http_client()`（`trust_env=False`），与 Tavily 客户端同一套出站策略，
不另起连接池。原因见 `src/http_client.py` 顶部关于系统代理的说明。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.http_client import get_http_client

# 与 src/database/*.py 同一处理：本模块要能被脚本、`python -c`、测试单独使用。
# load_dotenv 默认不覆盖已有环境变量，故不影响显式传参或 CI 里预设的变量。
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

# --- 配置 ---

API_KEY_ENV = "ZHIPU_API_KEY"
SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"

# 超时必须**短于工具层 30s 总超时**（src/agent/constants.py 的 TOOL_TIMEOUT_SECONDS）：
# 实测智谱 Web Search 约 2s 返回，25s 是给网络抖动留的余量。
SEARCH_TIMEOUT_SECONDS = 25.0
CONNECT_TIMEOUT_SECONDS = 10.0

# --- 搜索引擎档位 ---
# search_std   ：标准版，便宜快，适合绝大多数时效性查询（默认）
# search_pro   ：专业版，结果更全，单价更高
# search_pro_sogou / search_pro_quark：接入特定引擎
ALLOWED_ENGINES = ("search_std", "search_pro", "search_pro_sogou", "search_pro_quark")
DEFAULT_ENGINE = "search_std"

# --- 缓存（与 Tavily 客户端同参数，便于统一预期）---
CACHE_TTL_SECONDS = 600.0
CACHE_MAX_ITEMS = 512

# --- 结果裁剪 ---
# 工具结果会原样进 LLM 上下文，不截断时几条长公告就能吃掉整个上下文预算。
# 智谱返回的 content 往往是一整篇公告（实测可达数千字），截断是必须的。
CONTENT_TRUNCATE_CHARS = 500

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10
# 智谱该接口一次最多返回 50 条，但本项目工具层只展示前 N 条，取 10 足够。
API_MAX_RESULTS = 50


class ZhipuSearchError(Exception):
    """智谱联网搜索失败（密钥缺失 / 网络错误 / 非 200 / 响应无法解析）。

    只抛这一种类型，让调用方（工具执行器）能用一次 except 收敛所有失败分支，
    并转成用户可见的中文文案 —— 原始异常与响应体只进日志。
    """


def is_configured() -> bool:
    """是否配置了智谱密钥。

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
    """把智谱的 `search_result` 归一成与 Tavily 同形的字段集，另加两项智谱特有信息。

    统一成 `{title, url, content, score, publish_date, media}` 的形状，好处是
    `rag_tools._fmt_web` / `_web_sources` / `_rerank_web_results` 三个函数
    **一行都不用改**就能同时服务两个搜索引擎 —— 下游只认这几个键。

    `publish_date` 与 `media` 是智谱独有的增量信息，保留下来供格式化层展示，
    它们是判断"信息时效性"的直接依据（这是引入第二个搜索源的核心理由之一）。
    """
    items: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("link") or entry.get("url") or "").strip()
        title = str(entry.get("title") or "").strip()
        # 既没链接也没标题的条目无法引用，丢掉比留在结果里更诚实
        if not url and not title:
            continue
        items.append(
            {
                "title": title or url,
                "url": url,
                "content": _truncate(str(entry.get("content") or ""), CONTENT_TRUNCATE_CHARS),
                # 智谱不返回相关性分数：留 None 让下游 _norm_score 走它的回退分支，
                # 而不是伪造一个 0 分把结果排到最后
                "score": None,
                "publish_date": str(entry.get("publish_date") or "").strip(),
                "media": str(entry.get("media") or "").strip(),
            }
        )
    return items


class ZhipuSearchClient:
    """智谱 Web Search 客户端：进程内缓存 + 统一出站 + 失败只抛 ZhipuSearchError。

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
        """清过期项；仍然满则按时间戳丢掉最旧的一半。调用方必须已持锁。"""
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
        """给健康检查用的只读状态：**不触发任何网络请求**（不消耗配额）。"""
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
        engine: str = DEFAULT_ENGINE,
    ) -> dict:
        """中文关键词搜索，返回 `{"results": [...], "answer": str, "intent": str}`。

        - 空 query 返回空结果而不是抛异常：没有关键词就没有可搜的东西，属于"无结果"而非"故障"。
        - 密钥缺失 / 网络错误 / 非 200 / 响应无法解析 → 抛 `ZhipuSearchError`。
          调用方据此区分"搜不到"（空列表）与"搜不了"（异常），给出不同文案。
        - `answer` 恒为空串：智谱该接口不提供 AI 摘要（这正是 Tavily 的互补点），
          保留字段是为了让下游格式化函数可以无差别复用。
        - `intent` 是智谱返回的意图分析关键词，供日志与排错观察服务端如何理解查询。
        """
        query = (query or "").strip()
        if not query:
            return {"results": [], "answer": "", "intent": ""}

        key = query.lower()
        cached = self._cache_get(key)
        if cached is not None:
            logger.info("智谱联网搜索命中缓存：%s", query)
            return cached

        if not is_configured():
            # 显式失败：让"没配密钥"在日志与用户可见文案里都表现为"不可用"
            raise ZhipuSearchError(f"未配置 {API_KEY_ENV}")

        payload = {
            "search_engine": engine if engine in ALLOWED_ENGINES else DEFAULT_ENGINE,
            "search_query": query,
            "count": min(_safe_max_results(max_results) * 2, API_MAX_RESULTS),
        }
        headers = {
            "Authorization": f"Bearer {os.getenv(API_KEY_ENV, '').strip()}",
            "Content-Type": "application/json",
        }

        try:
            # 复用全局出站客户端（trust_env=False 等策略集中在那里）。逐请求覆盖超时。
            response = get_http_client().post(
                SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(SEARCH_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            logger.warning("智谱搜索请求失败（%s）：%s", type(exc).__name__, exc)
            raise ZhipuSearchError("联网搜索请求失败") from exc

        if response.status_code != 200:
            # 响应体可能含账号信息，只记状态码
            logger.warning("智谱搜索返回非 200：%s", response.status_code)
            raise ZhipuSearchError(f"联网搜索服务返回 {response.status_code}")

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("智谱搜索响应无法解析为 JSON")
            raise ZhipuSearchError("联网搜索响应无法解析") from exc
        if not isinstance(body, dict):
            logger.warning("智谱搜索响应不是 JSON 对象（%s）", type(body).__name__)
            raise ZhipuSearchError("联网搜索响应格式异常")

        intents = body.get("search_intent")
        intent = ""
        if isinstance(intents, list) and intents and isinstance(intents[0], dict):
            intent = str(intents[0].get("keywords") or intents[0].get("query") or "").strip()

        result = {
            "results": _normalize_items(body.get("search_result")),
            "answer": "",  # 智谱不提供 AI 摘要，见方法 docstring
            "intent": intent,
        }
        self._cache_put(key, result)
        logger.info("智谱联网搜索完成：%s（%d 条）", query, len(result["results"]))
        return result


# 模块级单例：与 web_search_client 同风格。构造函数只建空字典与锁，不做 I/O；
# 真正昂贵的资源（httpx 连接池、密钥）在首次请求和每次调用时才解析。
zhipu_search_client = ZhipuSearchClient()
