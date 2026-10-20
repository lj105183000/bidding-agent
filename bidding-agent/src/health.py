"""健康检查：把"进程活着"、"依赖可用"、"功能可用"分成独立的问题回答。

★ 为什么要重做（现有实现的缺口）
-----------------------------
现有 `/api/health` 的探测本身是对的（真实探测、3 秒线程超时、
`points_count` 失败为 -1 而非 0）。缺的是**语义**：

    ① 没有 liveness / readiness 之分
       容器编排的存活探针会**杀掉正在冷启动的进程**。
       `_preload_models` 的注释已经记过这件事：预热要 100 秒
       （嵌入模型 20s + 精排模型 80s），而预热期间三个数据源其实都正常，
       把整个服务判成"不健康"是错的。

    ② 不返回 HTTP 状态码
       编排器（Docker HEALTHCHECK / K8s probe / 负载均衡）**只看状态码**，
       不会解析 JSON body。永远 200 等于"永远健康"。

    ③ 不探 LLM provider
       `/api/chat` 是核心接口，而它依赖 LLM。密钥失效 / 模型名配错时
       现有健康检查**全绿**，但问答必然失败。这是最误导的一种状态。

    ④ 没有配置可见性
       生产里最常见的故障不是"服务挂了"，而是"连到了错的 collection /
       模型名拼错 / 环境变量没生效"。健康检查应该让运维**一眼核对**
       实际生效的配置，而不是去翻 .env。

    ⑤ 探测是串行的
       三个依赖各 3 秒超时 → 最坏 9 秒。并发后接近最慢的那个。

★ 三层深度，因为"探一下"的代价差别巨大
-----------------------------------
    live    不探任何外部依赖，只回答"进程还在吗"（微秒级，可高频调用）
    ready   探本地基础设施（Qdrant / Neo4j / Postgres），毫秒级、免费
    deep    额外探"外部依赖**可用性**"（LLM 密钥是否真的能通过鉴权）

    **deep 绝不能是默认**：它会对 LLM 产生真实 API 调用。
    Docker HEALTHCHECK 默认每 30 秒一次，一天 2880 次计费调用——
    健康检查不该烧钱。故 deep 只在显式请求时跑。

★ 状态语义
--------
    ok        全部正常
    degraded  核心可用，但有可选组件不可用（如 Neo4j 挂了，问答仍能答但图谱页不可用）
    warming   正在预热本地模型（**不是故障**，只是还没到最佳状态）
    down      核心组件不可用（问答必然失败）

`degraded` 与 `down` 的区别是刻意设计的：把"可选组件坏了"也报成 down，
会让运维在 Neo4j 挂掉时误以为整个服务不可用而重启——重启解决不了上游服务的故障。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: 单个依赖探测的超时（秒）。
#:
#: 取 5 秒（**原为 3 秒，实测不够**）：
#:
#: 本项目的 Qdrant 是 **Qdrant Cloud（AWS eu-central-1 法兰克福）**，
#: 实测单次往返：
#:
#:     count(exact=True)   1524.7ms → 978.5ms → 233.4ms
#:     client.info()        947.7ms → 232.6ms → 237.9ms
#:
#: 即热态约 233ms，但**冷连接（TLS 握手 + 建连）要 1.5 秒**，
#: 叠加四个探测线程争 GIL，实测首次探测落在 **2.6~3.0 秒**——
#: 正好卡在原 3 秒阈值上抖动。连续三次实测：
#:
#:     2619ms 通过 / 3003ms 超时 / 3012ms 超时
#:
#: 表现为"健康检查时好时坏"，而依赖其实完全正常。
#: 跨洲 WAN 必须留余量；5 秒仍远小于编排器探针超时（通常 10~30 秒）。
PROBE_TIMEOUT_SECONDS = 5.0

#: 外部可用性探测（deep）的超时。LLM 要建连 + 推理，给更宽。
DEEP_PROBE_TIMEOUT_SECONDS = 15.0

#: 进程启动时刻，用于算 uptime。
_STARTED_AT = time.time()

#: 预热状态。由 `mark_warming_started` / `mark_warmup_done` 维护。
_warmup_lock = threading.Lock()
_warmup: dict[str, Any] = {
    "in_progress": False,
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
}


def mark_warming_started() -> None:
    """标记"本地模型预热开始"。

    有了这个标记，健康检查才能回答"为什么第一次问答很慢" ——
    否则运维只能看到"接口慢"，看不出是预热尚未完成。
    """
    with _warmup_lock:
        _warmup["in_progress"] = True
        _warmup["started_at"] = time.time()
        _warmup["finished_at"] = 0.0
        _warmup["error"] = ""


def mark_warmup_done(error: str = "") -> None:
    """标记预热结束（成功或失败）。"""
    with _warmup_lock:
        _warmup["in_progress"] = False
        _warmup["finished_at"] = time.time()
        _warmup["error"] = error


def warmup_state() -> dict[str, Any]:
    """预热状态快照。

    ★ `elapsed_ms` 的三态（实测踩过）
    ------------------------------
        -1      从未预热过（服务刚起、还没触发 `_preload_models`）
        >=0     正在预热（已耗时）或已完成（总耗时）

    第一版写成 `elif snap["finished_at"]:` 之外还允许 `started_at` 为 0，
    于是"从未预热但被调用过 `mark_warmup_done()`"会算出
    `now - 0` —— **17 亿毫秒**（实测数值 1791899572374）。
    这种数字比 -1 危险得多：看起来像个真实的耗时值。
    """
    with _warmup_lock:
        snap = dict(_warmup)
    started_at = snap.pop("started_at", 0.0)
    finished_at = snap.pop("finished_at", 0.0)

    if snap["in_progress"]:
        snap["elapsed_ms"] = int((time.time() - started_at) * 1000)
    elif finished_at and started_at:
        snap["elapsed_ms"] = int((finished_at - started_at) * 1000)
    else:
        # 从未真正预热过。-1 与"耗时 0ms"区分开：
        # 后者会让人以为"预热瞬间完成了"。
        snap["elapsed_ms"] = -1
    return snap


# ---------------------------------------------------------------------------
# 探测原语
# ---------------------------------------------------------------------------


def probe(
    fn: Callable[[], Any],
    timeout: float = PROBE_TIMEOUT_SECONDS,
    preload: Callable[[], None] | None = None,
) -> dict:
    """跑一个探测，返回结构化结果 `{ok, ms, error}`。

    `preload` 在**计时开始之前**执行一次，用来把模块 import 移出计时窗口。

    ★ 为什么需要 preload（实测）
    --------------------------
    首次探测报 `Timeout(>3.0s)`，而单独测同一操作只要 866ms。
    差别在于 `from src.rag.pipeline import RAGPipeline` 本身要 **1880ms**
    （句子转换器那串依赖），且它在探测线程内部执行 —— 被测的成了
    "import + 操作"。

    更糟的是四个探测并发时都争 GIL，**本来快的依赖被报成超时**。
    健康检查报假超时最误导：让人去查网络，而其实只是导入慢。

    ★ 为什么用线程而不是 asyncio
    -------------------------
    这些依赖的客户端全是**同步**的（qdrant-client / neo4j / psycopg）。
    在 async 端点里直接调会阻塞事件循环。直接起 daemon 线程最简单，
    超时后线程自行结束，不阻塞健康检查返回。

    超时返回 `ok=False` 且 `error="Timeout"` —— 与"探测函数抛异常"
    区分开：前者是慢，后者是错，应对方式不同。
    """
    if preload is not None:
        try:
            preload()
        except Exception:  # noqa: BLE001 - 预加载失败不改变探测结果
            pass

    t0 = time.perf_counter()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
            box["ok"] = True
        except Exception as exc:  # noqa: BLE001 - 任何异常都算探测失败
            box["ok"] = False
            box["error"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=run, daemon=True, name="health-probe")
    th.start()
    th.join(timeout)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if th.is_alive():
        return {"ok": False, "ms": elapsed_ms, "error": f"Timeout(>{timeout}s)"}
    if box.get("ok"):
        return {"ok": True, "ms": elapsed_ms, "value": box.get("value")}
    return {
        "ok": False,
        "ms": elapsed_ms,
        # 截断：有些驱动会把整个连接串/响应体拼进异常消息
        "error": str(box.get("error", "unknown"))[:300],
    }


def _component(name: str, kind: str, required: bool, result: dict,
               detail: str = "") -> dict:
    """构造一个组件状态条目。"""
    item: dict[str, Any] = {
        "name": name,
        "kind": kind,          # infra / data / llm / optional
        "required": required,  # 不可用时是否导致核心功能失效
        "ok": bool(result.get("ok")),
        "ms": result.get("ms", -1),
    }
    # ★ 透传结构化数值字段（`probe_qdrant` 的 `count`）
    #
    # 为什么必须透传：`api/server.py` 的 `points_count` 兼容字段需要这个数字。
    # 此前 `_component` 只拷 ok/ms/error，`count` 被丢掉，于是 server 只能
    # 去**解析 detail 显示字符串**（`detail.split("=")[-1]`）——
    # 而 detail 从 `points=565` 改成 `points≈565` 后解析就失败了，
    # 导致 `/api/health` 返回 `points_count = -1`、前端显示「未连接向量库」。
    # 结论：数值走结构化字段，绝不解析给人看的字符串。
    if "count" in result:
        item["count"] = result["count"]
    if not item["ok"] and result.get("error"):
        item["error"] = result["error"]
    if detail:
        item["detail"] = detail
    return item


# ---------------------------------------------------------------------------
# 各组件的探测
# ---------------------------------------------------------------------------


def probe_qdrant() -> dict:
    """向量库：连通性 + 真实点数。

    ★ 点数为 0 与"探测失败"必须分开表达。
    早期实现把 `points_count` 写成字面量 0，监控会以为向量库是空的。
    这里 `count` 为 -1 表示探测失败，0 表示连通但集合为空。

    ★ 为什么用 `preload` 把 import 移出计时窗口（实测）
    ------------------------------------------------
    首次探测实测报 `Timeout(>3.0s)`，而单独测 `count()` 只要 866ms。
    原因是 `from src.rag.pipeline import RAGPipeline` 本身要 **1880ms**
    （句子转换器那一大串依赖），且是在探测线程内部做的——
    于是被测的是"import + count"，而不是"count"。

    更糟的是四个探测线程并发时都持有/等待 GIL，导入的 CPU 时间互相挤占，
    于是**本来很快的依赖被报成超时**。健康检查报假超时会让人去查网络，
    方向完全错。

    故：把 import 放在计时之前的 `preload` 里。
    """
    def run() -> int:
        from src.rag.pipeline import RAGPipeline

        # ★ 用近似计数，不用 `store.count()`（后者是 `exact=True`）
        # ----------------------------------------------------
        # 健康检查要回答的是"通不通、集合里有没有东西"，
        # **不是精确点数**。`exact=True` 会让服务端扫描全部点数，
        # 在 Qdrant Cloud（跨洲）上实测 233ms~1.5s；
        # 近似计数走集合元数据，代价低一个量级。
        #
        # 精确点数另有出处：`/api/health` 的兼容字段会在
        # 探测成功时给出近似值，需要精确值时请直接查 Qdrant。
        return int(RAGPipeline().store.client.get_collection(
            RAGPipeline().store.collection
        ).points_count or 0)

    res = probe(run, preload=_preload_rag)
    if res["ok"]:
        res["count"] = res.pop("value")
    else:
        res["count"] = -1
    return res


def _preload_rag() -> None:
    """预加载 RAG 相关模块（不计入探测耗时）。失败不抛：探测自己会报。"""
    try:
        from src.rag.pipeline import RAGPipeline  # noqa: F401
    except Exception:
        pass


def probe_neo4j(graph_data: Any = None) -> dict:
    """Neo4j 连通性。

    ★ 单例从调用方传入，不在探测里 import
    ----------------------------------
    第一版写的是 `from src.database.graph_data import graph_data` —— 那是
    **模块**不是实例（`graph_data.py` 只有 `graph_available()` 函数），
    于是报 `ImportError`，探测恒为 False。

    真正的全局在 `api.server` 里。而从 `src.health` 反向 import `api.server`
    会形成循环导入，故由调用方把实例传进来。
    """
    def run() -> bool:
        return bool(graph_data is not None and graph_data.graph_available())

    return probe(run)


def probe_postgres(getter: Callable[[], Any] | None = None) -> dict:
    """PostgreSQL 连通性。`getter` 由调用方传入（同上，避免循环导入）。"""
    def run() -> bool:
        if getter is None:
            return False
        return getter() is not None

    return probe(run, preload=_preload_psycopg)


def _preload_psycopg() -> None:
    try:
        import psycopg  # noqa: F401
    except Exception:
        pass


def probe_llm(agent: Any = None) -> dict:
    """LLM provider 可用性。

    ★ 为什么这一项最重要
    ------------------
    `/api/chat` 是核心接口，而它依赖 LLM。密钥失效 / 模型名配错时
    旧版健康检查**全绿**，但问答必然失败 —— 这是最误导的状态。

    ★ 这里只做**本地判定**，不发请求
    -----------------------------
    `client.available()` 检查凭据齐全、模型名非空，纯本地。
    真实连通性放在 deep 层（会消耗额度）。

    候选链信息也要报出来：多 provider 自动切换时，
    "主 provider 不可用但有兜底" 与 "全都不可用" 是完全不同的状态。
    """
    def run() -> dict:
        if agent is None:
            raise RuntimeError("Agent 未初始化")
        client = agent._get_llm_client("")
        return {
            "available": bool(client.available()),
            "chain": list(getattr(client, "chain", []) or []),
            "last_used": getattr(client, "last_used_provider", "") or "",
            "provider": getattr(client, "provider", ""),
        }

    res = probe(run, preload=_preload_llm_clients)
    if res["ok"]:
        res["llm"] = res.pop("value")
        # available() 为 False 也算探测失败：链里一个能用的都没有
        if not res["llm"].get("available"):
            res["ok"] = False
            res["error"] = "候选链中没有可用的 provider（检查 API Key / 模型名）"
    return res


def _preload_llm_clients() -> None:
    try:
        from src.clients.llm_factory import get_llm_client  # noqa: F401
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 配置可见性
# ---------------------------------------------------------------------------


def config_snapshot() -> dict[str, Any]:
    """**实际生效**的关键配置。

    ★ 为什么健康检查要暴露配置
    ------------------------
    生产里最常见的故障不是"服务挂了"，而是"连到了错的 collection"、
    "模型名拼错"、"环境变量没生效"。让运维能一眼核对，比让他去翻 .env、
    再对照进程实际读到什么，要可靠得多。

    **不暴露任何密钥值**，只报"是否已设置"。
    """
    from src.config import settings

    def has(value: Any) -> bool:
        return bool(str(value or "").strip())

    return {
        "qdrant": {
            "url": getattr(settings, "qdrant_url", ""),
            "collection": getattr(settings, "qdrant_collection", ""),
            "vector_size": getattr(settings, "qdrant_vector_size", None),
            "api_key_set": has(getattr(settings, "qdrant_api_key", "")),
        },
        "postgres": {
            "host": getattr(settings, "pg_host", ""),
            "port": getattr(settings, "pg_port", None),
            "database": getattr(settings, "pg_database", ""),
            "password_set": has(getattr(settings, "pg_password", "")),
        },
        "neo4j": {
            "uri": getattr(settings, "neo4j_uri", ""),
            "user": getattr(settings, "neo4j_user", ""),
            "password_set": has(getattr(settings, "neo4j_password", "")),
        },
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "") or "deepseek(默认)",
            "fallback_chain": os.getenv("LLM_FALLBACK_PROVIDERS", "(默认 deepseek,zhipu)"),
            "deepseek_key_set": has(os.getenv("DEEPSEEK_API_KEY", "")),
            "zhipu_key_set": has(os.getenv("ZHIPU_API_KEY", "")),
        },
        "search": {
            "tavily_key_set": has(os.getenv("TAVILY_API_KEY", "")),
            "exa_key_set": has(os.getenv("EXA_API_KEY", "")),
        },
        "features": {
            "planning": _truthy(os.getenv("AGENT_PLANNING", "0")),
            "memory": _truthy(os.getenv("AGENT_MEMORY", "1")),
            "compaction": _truthy(os.getenv("AGENT_COMPACT", "1")),
            "rerank": bool(getattr(settings, "rerank_enabled", False)),
        },
        "logging": {
            "format": os.getenv("LOG_FORMAT", "") or "auto",
            "level": os.getenv("LOG_LEVEL", "") or "INFO",
        },
    }


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

#: 哪些组件属于"核心"：不可用则问答必然失败。
#:
#: 默认核心组件：Qdrant 与 LLM —— 本地知识库检索与生成都离不开。
#: Neo4j（知识图谱页）与 Postgres（会话/反馈/中标记录）默认是**可选**：
#: 它们挂了，问答仍能基于知识库工作，只是相应页面不可用。
_DEFAULT_CORE = ("qdrant", "llm")

#: 环境变量：覆盖核心组件集合（逗号分隔，如 `qdrant,llm,neo4j`）。
#:
#: ★ 为什么必须可配（局限 1）
#: ----------------------
#: "Neo4j 是可选的"是**本项目的定位判断**，不是普遍事实。若某个部署把
#: 知识图谱做成业务必需（例如图谱页是主入口），那 Neo4j 挂了就该被判
#: `down`（503）让编排器摘流量 —— 而写死的 `_CORE` 只会报 `degraded`(200)，
#: **编排器不会摘流量**，故障就一直挂着。
#:
#: 反之也不该一律把 Neo4j 算核心：大部分部署里它挂了问答照常，
#: 报 down 会触发无谓的重启（重启解决不了上游故障）。
#:
#: 故：**默认保守（可选），部署按环境覆盖**。
HEALTH_CORE_ENV = "HEALTH_CORE_COMPONENTS"


def core_components() -> tuple[str, ...]:
    """当前应视为"核心"的组件名元组。

    每次读环境变量（便于测试与运行时改配置）。非法值（空、未知组件名）
    **忽略并退回默认** —— 健康检查不能因为一个拼错的配置就改变语义。
    """
    raw = (os.getenv(HEALTH_CORE_ENV) or "").strip()
    if not raw:
        return _DEFAULT_CORE
    known = {"qdrant", "llm", "neo4j", "postgres"}
    picked = tuple(
        p.strip().lower() for p in raw.split(",") if p.strip().lower() in known
    )
    if not picked:
        # 全是不认识的名字 → 退回默认，而不是"没有核心组件"（那会让永远 ok）
        return _DEFAULT_CORE
    return picked


#: 向后兼容：旧代码可能按名字引用 `_CORE`。它是一个**快照**，
#: 不随环境变量变化 —— 新代码请用 `core_components()`。
_CORE = _DEFAULT_CORE


def overall_status(components: list[dict], warm: dict) -> str:
    """按组件状态推导总体状态。

    顺序刻意如此：
        ① 核心组件不可用 → down（**优先于 warming**：核心坏了就是坏了，
           不能因为"还在预热"就把 down 说成 warming）
        ② 预热中 → warming（不是故障）
        ③ 有可选组件不可用 → degraded
        ④ 否则 ok
    """
    core = core_components()
    core_bad = [c for c in components if c["name"] in core and not c["ok"]]
    if core_bad:
        return "down"
    if warm.get("in_progress"):
        return "warming"
    if warm.get("error"):
        return "degraded"
    optional_bad = [c for c in components if not c["ok"]]
    if optional_bad:
        return "degraded"
    return "ok"


def status_code(status: str) -> int:
    """状态 → HTTP 状态码。

    ★ 为什么必须区分（编排器只看状态码）
    ---------------------------------
        ok / warming  → 200   可用（warming 只是还没到最佳状态，不该被重启）
        degraded      → 200   核心可用，编排器**不该**重启它
        down          → 503   核心不可用，应当被摘除流量

    `degraded` 给 200 是刻意的：Neo4j 挂了不该触发容器重启——
    重启解决不了上游服务的故障，只会让服务反复重启、更难排查。
    """
    return 503 if status == "down" else 200


# ---------------------------------------------------------------------------
# 探测结果短 TTL 缓存（局限 2）
# ---------------------------------------------------------------------------
#: `(cache_key) -> (monotonic_ts, payload, code)`
_PROBE_CACHE: dict[tuple, tuple[float, dict, int]] = {}
_PROBE_CACHE_LOCK = __import__("threading").Lock()

#: 环境变量：缓存秒数。0 / 未设 → 关闭。
HEALTH_PROBE_CACHE_ENV = "HEALTH_PROBE_CACHE_SECONDS"


def clear_probe_cache() -> None:
    """清空探测缓存（测试与"改了依赖配置后立即生效"时用）。"""
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE.clear()


def _cache_get(key: tuple, ttl: float) -> tuple[dict, int] | None:
    if ttl <= 0:
        return None
    with _PROBE_CACHE_LOCK:
        hit = _PROBE_CACHE.get(key)
    if not hit:
        return None
    ts, payload, code = hit
    if time.monotonic() - ts > ttl:
        return None
    return payload, code


def _cache_put(key: tuple, payload: dict, code: int) -> None:
    with _PROBE_CACHE_LOCK:
        # 简单的容量保护：条目不会多（depth × 少量依赖组合），但仍设上限
        if len(_PROBE_CACHE) > 64:
            _PROBE_CACHE.clear()
        _PROBE_CACHE[key] = (time.monotonic(), payload, code)


def build_health(
    depth: str = "ready",
    include_config: bool = False,
    agent: Any = None,
    graph_data: Any = None,
    postgres_getter: Callable[[], Any] | None = None,
    cache_seconds: float | None = None,
) -> tuple[dict, int]:
    """构造健康检查响应。返回 `(payload, http_status_code)`。

    `depth`：
        live   不探外部依赖，只报进程/预热/配置
        ready  并发探本地基础设施 + LLM 本地可用性（默认）
        deep   在 ready 基础上额外做外部可用性探测（**会消耗 API 额度**）

    `agent` / `graph_data` / `postgres_getter` 由调用方注入：
    它们是 `api.server` 的模块级全局，从 `src.health` 反向 import 会形成
    循环导入。注入还有个好处——测试可以直接传替身，不必启动真实依赖。
    """
    started = time.perf_counter()
    depth = (depth or "ready").strip().lower()
    if depth not in ("live", "ready", "deep"):
        depth = "ready"

    # ★ 短 TTL 缓存（局限 2，见本函数上方注释块）
    #
    # `cache_seconds=None` → 读环境变量 `HEALTH_PROBE_CACHE_SECONDS`（未设=0=关）
    # 显式传 0  → 强制关闭（测试用）
    # `/api/health` 端点传入 5 秒
    if cache_seconds is None:
        try:
            cache_seconds = float(os.getenv(HEALTH_PROBE_CACHE_ENV) or 0)
        except (TypeError, ValueError):
            cache_seconds = 0.0

    # `live` 不探外部依赖（只读内存状态），缓存它没有收益且会掩盖
    # "刚预热完成"这类状态变化 —— 故只缓存 ready / deep。
    cache_key = (
        depth,
        id(agent),
        id(graph_data),
        id(postgres_getter),
        bool(include_config),
    )
    if depth != "live":
        cached = _cache_get(cache_key, cache_seconds)
        if cached is not None:
            payload, code = cached
            # 标出来，避免运维以为这是刚测的
            payload = dict(payload)
            payload["cached"] = True
            payload["cache_ttl_seconds"] = cache_seconds
            return payload, code

    components: list[dict] = []
    llm_deep: dict[str, Any] | None = None

    if depth in ("ready", "deep"):
        # ★ 并发探测：串行最坏 3 个依赖 × 3 秒 = 9 秒，
        #   并发后接近最慢的那个。健康检查要快才有用。
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="health") as pool:
            f_qdrant = pool.submit(probe_qdrant)
            f_neo4j = pool.submit(probe_neo4j, graph_data)
            f_pg = pool.submit(probe_postgres, postgres_getter)
            f_llm = pool.submit(probe_llm, agent)
            r_qdrant = f_qdrant.result()
            r_neo4j = f_neo4j.result()
            r_pg = f_pg.result()
            r_llm = f_llm.result()

        count = r_qdrant.get("count", -1)
        # ★ 明确标"约"（局限 3）：这个数是集合元数据的**近似值**，
        #   不是 `exact=True` 的精确点数。给监控看到 `points≈565`
        #   才不会把它当精确值去做点数告警。
        detail = f"points≈{count}" if count >= 0 else ""
        components.append(_component("qdrant", "data", True, r_qdrant, detail))
        components.append(_component("llm", "llm", True, r_llm,
                                     _llm_detail(r_llm)))
        components.append(_component("neo4j", "data", False, r_neo4j))
        components.append(_component("postgres", "data", False, r_pg))
        # Langfuse 是**可选**上报组件：未配置时不算"坏"，而是"未启用"。
        #
        # ★ 为什么必须区分"未启用"与"故障"
        # -----------------------------
        # 若把它当成必检组件，没配 key 的用户会看到 health 一直是 degraded
        # ——而那是**预期状态**（不配就是不用），不是故障。
        # 故这里 `ok=True` 表示"不拖累整体状态"，真实情况写在 detail 里。
        try:
            from src.langfuse_tracing import status as _lf_status

            _lf = _lf_status()
            _detail = (
                f"host={_lf.get('host')}"
                if _lf.get("enabled")
                else f"未启用（{_lf.get('reason') or '未配置'}）"
            )
            # ★ 第 4 个参数是 **probe 结果 dict**（形如 `{"ok": bool, ...}`），
            # 不是裸 bool——`_component` 内部会 `result.get("ok")`。
            # 我第一版传了 `True`，抛 AttributeError；而当时外面是
            # `except: pass`，于是组件静默消失、查不出原因（实测踩到）。
            components.append(
                _component("langfuse", "observability", False,
                           {"ok": True, "detail": _detail}, _detail)
            )
        except Exception as exc:  # noqa: BLE001 - 可选组件，取不到就降级报告
            # ★ 不能静默 pass：健康检查本身是诊断工具，
            # 把"取不到该组件状态"藏起来恰恰违背它的用途。
            # 故照样加组件，detail 写明失败原因。
            components.append(
                _component("langfuse", "observability", False,
                           {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                           f"状态读取失败：{type(exc).__name__}")
            )

        if depth == "deep":
            llm_deep = probe_llm_deep(agent)

    warm = warmup_state()
    status = overall_status(components, warm)

    payload: dict[str, Any] = {
        "status": status,
        "depth": depth,
        "uptime_s": int(time.time() - _STARTED_AT),
        "check_ms": 0.0,  # 填充在最后
        "warmup": warm,
        "components": components,
    }
    if llm_deep is not None:
        payload["llm_deep"] = llm_deep

    # 未通过的核心/可选组件单独列出：监控最常直接读这个字段做告警
    failing = [c["name"] for c in components if not c["ok"]]
    payload["failing"] = failing

    if include_config:
        payload["config"] = config_snapshot()

    payload["check_ms"] = round((time.perf_counter() - started) * 1000, 1)
    code = status_code(status)
    # 写入缓存（局限 2）。失败结果也缓存 —— 上游宕机时探针更需要轻量。
    if depth != "live" and cache_seconds and cache_seconds > 0:
        _cache_put(cache_key, payload, code)
    return payload, code


def _llm_detail(r_llm: dict) -> str:
    info = r_llm.get("llm") or {}
    if not info:
        return ""
    chain = "→".join(info.get("chain") or []) or "?"
    used = info.get("last_used") or "未使用"
    return f"chain={chain} last_used={used}"


def probe_llm_deep(agent: Any = None) -> dict:
    """**真的**调一次 LLM，验证鉴权与网络（会消耗额度）。

    ★ 为什么不放进默认深度
    --------------------
    Docker HEALTHCHECK 默认每 30 秒一次 → 一天 2880 次真实 API 调用。
    健康检查不该烧钱，故只在 `depth=deep` 时显式触发。

    用 `chat_raw` 而不是 `chat_stream`：非流式，一次往返即可，
    且不需要消费流。失败信息原样带上（401 / 403 / 超时都能一眼看出）。
    """
    def run() -> dict:
        if agent is None:
            raise RuntimeError("Agent 未初始化")
        client = agent._get_llm_client("")
        if not client.available():
            raise RuntimeError("候选链中没有可用的 provider")
        # 极短输入，把成本压到最低
        text = client.chat_raw([{"role": "user", "content": "ping"}])
        return {
            "ok": True,
            "provider": getattr(client, "last_used_provider", "") or "",
            "reply_chars": len(text or ""),
        }

    res = probe(run, timeout=15.0)
    return {
        "ok": bool(res.get("ok")),
        "ms": res.get("ms", -1),
        **({"error": res["error"]} if not res.get("ok") and res.get("error") else {}),
        **({"detail": res["value"]} if res.get("ok") else {}),
    }
