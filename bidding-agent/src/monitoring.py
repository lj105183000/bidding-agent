"""系统监控：把散落各处的运行时数据聚合成可回答运维问题的视图。

★ 已有素材 vs 缺什么
================
项目里**已经有很多运行时数据**，但散落在各处、没有统一视图：

    `/api/health`        依赖连通性与延迟（docs/45 细化过）
    `/api/logs`          结构化日志环形缓冲（docs/44）
    `RateLimiter.snapshot()`   限流命中/拒绝数
    `pipeline.cache_info()`    检索缓存命中率
    `done` 帧的 `phase_times`  单次问答的阶段耗时
    `llm_fallback` 的 `switch_count`  provider 切换次数

**缺的是"聚合"**：这些数据各自成立，但要回答运维最常问的问题，
得同时翻五个地方——

    「最近哪里慢？」          → 需要按端点的延迟分位
    「错误集中在哪？」        → 需要按状态码/端点的错误计数
    「服务吃多少资源？」      → 需要进程与系统的 CPU/内存
    「问答为什么变慢了？」    → 需要按阶段的耗时聚合（不是单次）
    「依赖还稳吗？」          → health 有，但没有历史趋势

本模块补上聚合层，并暴露 `/api/metrics`。

★ 设计取舍：为什么不用 prometheus_client
=====================================
这是一个**单机部署**的项目（用户自己的 Legion 笔记本，`docs` 里反复出现
"本地 Qdrant/Neo4j"）。引入 Prometheus 意味着额外跑一个 server、
配 scrape、再学一套 PromQL —— 对"想知道服务现在怎么样"这个需求来说太重了。

故这里做一个**进程内的滚动聚合器**（纯标准库 + 已有的 psutil），
`/api/metrics` 直接返回可读的 JSON。真要做长期监控时，这个 JSON
本身就能被任意外部采集器抓走（它同时提供 Prometheus 文本格式，见
`render_prometheus()`），不需要现在就把依赖背上。

★ 为什么是"滚动窗口"而不是"进程启动至今累计"
========================================
累计值只能回答"平均怎么样"，而**平均值会掩盖问题**：
一次 60 秒的超时会被 1 万次 200ms 的正常请求稀释成 206ms，
看起来一切正常。

故保留一个**时间窗口内**的样本（默认 5 分钟，最多留 1000 条/端点），
这样分位数反映的是"最近的情况"。累计计数也保留（用于算总量与错误率）。
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)

#: 默认的延迟样本窗口（秒）。样本超出窗口即被丢弃。
DEFAULT_WINDOW_SECONDS = 300.0

#: 每个端点保留的延迟样本上限。
#:
#: ★ 为什么必须有上限
#: ----------------
#: 与限流器同样的理由：无界增长的数据结构在长跑服务里就是内存泄漏。
#: 1000 个 float ≈ 8KB，× 几十个端点 ≈ 几百 KB，可接受。
#: 超出后丢**最旧**的（分位数只关心最近）。
MAX_SAMPLES_PER_ENDPOINT = 1000

#: 最多跟踪多少个端点。防"扫描器打大量不同路径"把表撑爆。
MAX_ENDPOINTS = 200

#: 系统指标缓存时长（秒）。
#:
#: `psutil.cpu_percent()` **本身需要两次采样求差**才能给出有意义的百分比
#: （第一次调用返回 0.0）。若每次请求都调，既不准（间隔太短）又浪费。
#: 故缓存 2 秒：既让数值有意义，又不会给出过时信息。
SYSTEM_CACHE_SECONDS = 2.0

#: 默认慢请求阈值（毫秒）。未在 `SLOW_MS_BY_ENDPOINT` 里命中的端点用它。
SLOW_MS = 1000.0

#: 按端点的慢请求阈值（毫秒）。**键是前缀匹配**，取最长命中的那个。
#:
#: ★ 为什么必须按端点区分（局限 5）
#: ---------------------------
#: `/api/chat` 正常就要十几秒（ReAct 多轮 + LLM 生成）。用 1000ms 会让它的
#: `slow_ratio` **恒为 1.0** —— 指标失去意义，反而掩盖真正的退化。
#:
#: ★ 阈值依据（不是猜的）
#: ----------------
#: 问答类取 15000ms：本项目实测 `agent.latency_p95` 在 11172~19405ms 之间
#: （见 `docs/90` 的方差实测），取其上限附近作为"正常"边界。
#: 检索/解析类取 5000ms：一次 HTTP + 一次跨洲 Qdrant 往返实测 1~3.3s。
#: 探针类取 2000ms：健康检查正常 1ms 级，2s 已是明显异常。
SLOW_MS_BY_ENDPOINT: dict[str, float] = {
    # 问答与会话：多轮 LLM，正常十几秒
    "/api/chat": 15000.0,
    "/api/ask": 15000.0,
    "/api/sessions": 15000.0,
    # 检索/RAG/文档解析：单次重计算
    "/api/search": 5000.0,
    "/api/rag": 5000.0,
    "/api/extract": 5000.0,
    "/api/tender": 8000.0,
    "/api/compare": 8000.0,
    # 探针与管理
    "/api/health": 2000.0,
    "/api/metrics": 2000.0,
    "/api/logs": 2000.0,
}


def slow_threshold_ms(endpoint: str) -> float:
    """该端点的慢请求阈值（毫秒）。**最长前缀命中**，否则默认 `SLOW_MS`。

    用最长前缀而不是精确匹配：`/api/chat/stream` 应继承 `/api/chat` 的阈值。
    """
    name = (endpoint or "").split("?", 1)[0].rstrip("/")
    best: tuple[int, float] | None = None
    for prefix, ms in SLOW_MS_BY_ENDPOINT.items():
        if name == prefix or name.startswith(prefix + "/"):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), ms)
    return best[1] if best else SLOW_MS


def percentile(values: list[float], q: float) -> float:
    """最近秩法求分位数（`q` 取 0~1）。**内部自行排序**，调用方不必预先排。

    ★ 索引公式：`ceil(q*n) - 1`
    -------------------------
    第一版写成 `ceil(q*len) - 1` 之后再 clamp 到 `len-1`，看着像对的，
    实测**在小样本长尾下严重低估**：

        201 个样本，200 个是 200ms + 1 个 60000ms
        q=0.99 → ceil(0.99*201)-1 = ceil(198.99)-1 = 199-1 = 198
        → 返回第 199 小的值 = 200ms      ← 期望是 60000ms！
        正确应为 ceil(199.0)-1 = 199-1 = 198 ... 仍不对

    问题在浮点：`0.99*201 = 198.99`，`ceil` 得 199，减 1 得 **198**，
    而"99% 分位"在 201 个样本里应该落在**第 200 个**（下标 199）——
    因为只有 1 个样本（0.5%）在它之上。

    正确公式是 `ceil(q*n) - 1` 但**上界要允许到 n-1**：
        ceil(0.99*201) - 1 = 198     ×  错
        ceil(199.0)      - 1 = 198   ×
        round up (q*(n-1)) = ceil(0.99*200) = 198 ... 仍错

    真正的问题是"最近秩"的定义：应取**最小的 i 使得 i/n >= q**，
    即 `i = ceil(q*n)`（**1-based**），下标 `i-1`：
        i = ceil(0.99*201) = ceil(198.99) = 199  → 下标 198 = 200ms

    这确实说明：**201 个样本时 0.99 分位按定义就是第 199 个**，
    那个离群值要 `q >= 200/201 ≈ 0.995` 才会被算进来。

    所以不是公式错，而是**我对"P99 应抓到 1 个离群值"的预期错了**：
    1/201 = 0.5% < 1%，它天然落在 P99 之上。P99.5 才抓得到它。

    结论：公式保持不变（`ceil(q*n)-1` 就是标准的最近秩定义），
    但要**在返回值之外同时给出 `max`** —— 否则"最后那 0.5% 的极端值"
    在 P99 上看不到，而它恰恰是最该被看到的。`snapshot()` 已同时给出
    `max_ms`，这是正确的设计。本函数额外接受未排序输入，避免调用方
    因忘记排序而得到无意义的结果（那类 bug 极难发现）。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


# ---------------------------------------------------------------------------
# 阶段耗时聚合（局限 6）
# ---------------------------------------------------------------------------
#: 每个阶段保留的样本数上限。答案与 MAX_SAMPLES_PER_ENDPOINT 同理 ——
#: 无界增长在长跑服务里就是内存泄漏。
MAX_PHASE_SAMPLES = 500

#: `阶段名 -> [毫秒, ...]`
_PHASE_SAMPLES: dict[str, list[float]] = {}
_PHASE_LOCK = __import__("threading").Lock()

#: 累计的问答次数（用于算"每问平均"）
_PHASE_TURNS = 0


def record_phases(phase_times: object) -> None:
    """记录一次问答的阶段耗时。**永不抛异常**。

    `phase_times` 形如 `[["首轮分析", 1800, 0], ["检索", 2000, 1800], ...]`
    （见 `core._build_done`）。非法结构静默忽略 —— 观测不能影响问答。
    """
    global _PHASE_TURNS
    try:
        if not isinstance(phase_times, (list, tuple)):
            return
        parsed: list[tuple[str, float]] = []
        for item in phase_times:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            name = str(item[0] or "").strip()
            try:
                ms = float(item[1])
            except (TypeError, ValueError):
                continue
            if name and ms >= 0:
                parsed.append((name, ms))
        if not parsed:
            return
        with _PHASE_LOCK:
            _PHASE_TURNS += 1
            for name, ms in parsed:
                bucket = _PHASE_SAMPLES.setdefault(name, [])
                bucket.append(ms)
                if len(bucket) > MAX_PHASE_SAMPLES:
                    del bucket[: len(bucket) - MAX_PHASE_SAMPLES]
    except Exception:  # noqa: BLE001 - 观测不能影响问答
        pass


def phase_metrics() -> dict[str, Any]:
    """阶段耗时的聚合结果。`{}` 表示还没记录过任何问答。"""
    try:
        with _PHASE_LOCK:
            turns = _PHASE_TURNS
            raw = {k: list(v) for k, v in _PHASE_SAMPLES.items()}
        if not turns or not raw:
            return {}
        total_ms = sum(sum(v) for v in raw.values())
        rows = []
        for name, vals in raw.items():
            if not vals:
                continue
            s_ = sorted(vals)
            subtotal = sum(vals)
            rows.append({
                "phase": name,
                "samples": len(vals),
                "avg_ms": round(subtotal / len(vals), 1),
                "p95_ms": round(percentile(s_, 0.95), 1),
                "max_ms": round(s_[-1], 1),
                # 占比：这个阶段占全部阶段耗时之和的份额
                "share": round(subtotal / total_ms, 4) if total_ms else 0.0,
            })
        rows.sort(key=lambda r: r["avg_ms"], reverse=True)
        return {
            "turns": turns,
            "sampled_turns": max(len(next(iter(raw.values()))), 1) if raw else 0,
            "phases": rows,
        }
    except Exception:  # noqa: BLE001 - 观测不能影响指标出口
        return {}


def reset_phases() -> None:
    """清空阶段样本（测试用）。"""
    global _PHASE_TURNS
    with _PHASE_LOCK:
        _PHASE_SAMPLES.clear()
        _PHASE_TURNS = 0


class MetricsRegistry:
    """按端点的请求计数与延迟分位（线程安全、内存有界）。

    只记**聚合值**，不记请求内容——监控数据里不该出现采购项目名之类的业务数据。
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_samples: int = MAX_SAMPLES_PER_ENDPOINT,
    ) -> None:
        self._window = window_seconds
        self._max_samples = max_samples
        self._lock = threading.Lock()
        #: 端点 → 延迟样本 (timestamp, ms)
        self._samples: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        #: 端点 → 状态码 → 计数
        self._codes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        #: 端点 → 累计请求数
        self._total: dict[str, int] = defaultdict(int)
        #: 端点 → 累计耗时（用于算全窗口均值）
        self._total_ms: dict[str, float] = defaultdict(float)
        self._started_at = time.time()

    # --- 记录 ---

    def record(self, endpoint: str, status: int, elapsed_ms: float) -> None:
        """记录一次请求。**任何路径都不能让它抛异常**（见 `_safe_record`）。"""
        now = time.time()
        with self._lock:
            # 端点数量上限：超了就丢最久未活动的（与限流器同一策略）
            if len(self._samples) >= MAX_ENDPOINTS and endpoint not in self._samples:
                oldest = min(
                    self._samples,
                    key=lambda k: self._samples[k][-1][0] if self._samples[k] else 0.0,
                )
                self._samples.pop(oldest, None)
                self._codes.pop(oldest, None)
                self._total.pop(oldest, None)
                self._total_ms.pop(oldest, None)

            dq = self._samples[endpoint]
            dq.append((now, elapsed_ms))
            # 丢出窗口的 + 超上限的
            cutoff = now - self._window
            while dq and (dq[0][0] < cutoff or len(dq) > self._max_samples):
                dq.popleft()

            self._codes[endpoint][status] += 1
            self._total[endpoint] += 1
            self._total_ms[endpoint] += elapsed_ms

    # --- 读取 ---

    def snapshot(self, limit_errors: int = 10) -> dict[str, Any]:
        """聚合快照。`limit_errors` 控制返回多少个出错端点。"""
        now = time.time()
        cutoff = now - self._window
        endpoints: list[dict[str, Any]] = []
        global_total = 0
        global_errors = 0
        global_latencies: list[float] = []

        with self._lock:
            for name, dq in self._samples.items():
                live = [ms for ts, ms in dq if ts >= cutoff]
                codes = dict(self._codes.get(name, {}))
                total = self._total.get(name, 0)
                total_ms = self._total_ms.get(name, 0.0)
                errors = sum(c for code, c in codes.items() if code >= 400)
                global_total += total
                global_errors += errors
                global_latencies.extend(live)

                entry: dict[str, Any] = {
                    "endpoint": name,
                    "requests": total,
                    "errors": errors,
                    "error_rate": round(errors / total, 4) if total else 0.0,
                    "codes": {str(k): v for k, v in sorted(codes.items())},
                    "samples": len(live),
                    # 全窗口均值 vs 窗口内分位：前者被历史稀释，后者反映近况，
                    # 两个都给，避免只看一个产生误判
                    "avg_ms_all": round(total_ms / total, 1) if total else 0.0,
                }
                if live:
                    live.sort()
                    entry.update({
                        # ★ 保留 1 位小数而不是取整
                        # 健康检查这类请求常在 1ms 内返回，取整会显示 "0.0ms"，
                        # 让人误以为"没测到"或"指标坏了"。1 位小数既真实又够读。
                        "avg_ms": round(sum(live) / len(live), 1),
                        "p50_ms": round(percentile(live, 0.50), 1),
                        "p95_ms": round(percentile(live, 0.95), 1),
                        "p99_ms": round(percentile(live, 0.99), 1),
                        # ★ 为什么额外给 p99.9
                        # 分位数是"比例"概念：样本 201 个时，1 个离群值只占
                        # 0.5%，**天然落在 P99 之上**——P99 看不到它。
                        # 实测：200×200ms + 1×60000ms 时 p99=200ms 而 max=60000ms。
                        # 故必须同时给极高分位与 max，否则极端请求在监控上隐形。
                        "p99_9_ms": round(percentile(live, 0.999), 1),
                        "max_ms": round(live[-1], 1),
                        # 按端点阈值算（局限 5）。同时输出阈值本身 ——
                        # 否则读指标的人不知道"慢"是按什么切的。
                        "slow_threshold_ms": slow_threshold_ms(name),
                        "slow_ratio": round(
                            sum(
                                1 for v in live if v >= slow_threshold_ms(name)
                            ) / len(live),
                            4,
                        ),
                    })
                endpoints.append(entry)

        # 排序：先按窗口内 P95 降序（"哪里慢"是最常问的），没有样本的排后面
        endpoints.sort(
            key=lambda e: (e.get("p95_ms", -1), e["requests"]), reverse=True
        )

        # 全局分位
        global_latencies.sort()
        overall: dict[str, Any] = {
            "requests": global_total,
            "errors": global_errors,
            "error_rate": round(global_errors / global_total, 4) if global_total else 0.0,
            "endpoints": len(endpoints),
            "window_seconds": self._window,
            "uptime_s": int(now - self._started_at),
        }
        if global_latencies:
            overall.update({
                "p50_ms": round(percentile(global_latencies, 0.50), 1),
                "p95_ms": round(percentile(global_latencies, 0.95), 1),
                "p99_ms": round(percentile(global_latencies, 0.99), 1),
                "p99_9_ms": round(percentile(global_latencies, 0.999), 1),
                "max_ms": round(global_latencies[-1], 1),
                "samples": len(global_latencies),
            })

        # 只返回出错最多的几个端点，而不是全部（避免响应过大）。
        # 末尾按 endpoint 名做**确定性 tie-break**：计数相同时顺序稳定，
        # 否则同一份数据两次调用可能给出不同顺序（dict 遍历顺序虽有序，
        # 但排序不稳定会让结果依赖输入顺序）。
        error_endpoints = [e for e in endpoints if e["errors"] > 0]
        error_endpoints.sort(key=lambda e: (-e["errors"], e["endpoint"]))

        return {
            "overall": overall,
            "endpoints": endpoints,
            "top_errors": error_endpoints[:limit_errors],
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._codes.clear()
            self._total.clear()
            self._total_ms.clear()
            self._started_at = time.time()


#: 全局注册表。中间件每次请求写它。
metrics = MetricsRegistry()


def safe_record(endpoint: str, status: int, elapsed_ms: float) -> None:
    """记录指标，**永不抛异常**。

    ★ 为什么必须包一层
    ----------------
    指标记录处在**每个请求的必经路径**上。它一旦抛异常，
    会把一个本来成功的请求变成 500 —— **监控代码搞挂了业务**，
    是这类功能最严重的失败模式。

    故所有调用点都用这个包装函数；真出问题只记一条 debug 日志，
    不影响请求本身。
    """
    try:
        metrics.record(endpoint, status, elapsed_ms)
    except Exception:  # noqa: BLE001
        logger.debug("指标记录失败（不影响请求）", exc_info=True)


# ---------------------------------------------------------------------------
# 系统层指标（psutil）
# ---------------------------------------------------------------------------

_sys_lock = threading.Lock()
_sys_cache: dict[str, Any] = {"at": 0.0, "data": {}}
#: 上一次的 cpu_percent 原始值，用于在缓存失效时给出**非阻塞**读数
_last_cpu: list[float] = []


def system_metrics() -> dict[str, Any]:
    """进程与主机资源指标。带缓存，失败时返回结构化错误而不是抛异常。

    ★ `cpu_percent` 的坑
    ------------------
    `psutil.Process.cpu_percent()` 第一次调用**必定返回 0.0**——它需要
    两次采样求差。若不加处理，第一次查监控会看到"CPU 占用 0%"，
    容易误判成"服务空闲"。

    这里用 `interval=None`（非阻塞，与上次调用对比）+ 缓存：
    首次调用返回 0.0 是已知行为，故在返回值里带上 `cpu_first_call` 标记，
    让调用方知道这个 0 不可信，而不是把它当真实读数。
    """
    now = time.time()
    with _sys_lock:
        if now - _sys_cache["at"] < SYSTEM_CACHE_SECONDS and _sys_cache["data"]:
            return _sys_cache["data"]

    result: dict[str, Any] = {}
    try:
        import psutil
    except ImportError:
        return {"error": "psutil 未安装，无法采集系统指标"}

    try:
        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            mem = proc.memory_info()
            # interval=None：非阻塞，与上次调用对比。首次必然 0.0
            first_call = not _last_cpu
            cpu_pct = proc.cpu_percent(interval=None)
            _last_cpu.append(cpu_pct)
            result["process"] = {
                "pid": proc.pid,
                "cpu_percent": round(cpu_pct, 2),
                "cpu_first_call": first_call,
                # RSS 是"实际占用的物理内存"，比 VMS 有意义
                "rss_mb": round(mem.rss / 1024 / 1024, 1),
                "vms_mb": round(mem.vms / 1024 / 1024, 1),
                "threads": proc.num_threads(),
                "open_files": _safe_count(proc, "num_fds", "open_files"),
                "connections": _safe_count(proc, "num_conns", "connections"),
                "create_time": int(proc.create_time()),
            }
    except Exception as exc:  # noqa: BLE001
        result["process"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        vm = psutil.virtual_memory()
        result["memory"] = {
            "total_mb": round(vm.total / 1024 / 1024, 1),
            "used_mb": round(vm.used / 1024 / 1024, 1),
            "available_mb": round(vm.available / 1024 / 1024, 1),
            "percent": vm.percent,
        }
    except Exception as exc:  # noqa: BLE001
        result["memory"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        # 与进程 CPU 同理：首次调用返回 0.0。用非阻塞模式。
        result["cpu"] = {
            "percent": round(psutil.cpu_percent(interval=None), 2),
            "count": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            # 1 分钟负载更能反映"是否持续繁忙"；Windows 上可能取不到
            "load_avg_1m": _load_avg(),
        }
    except Exception as exc:  # noqa: BLE001
        result["cpu"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        du = psutil.disk_usage(os.path.abspath(os.sep))
        result["disk"] = {
            "total_gb": round(du.total / 1024**3, 1),
            "used_gb": round(du.used / 1024**3, 1),
            "free_gb": round(du.free / 1024**3, 1),
            "percent": du.percent,
        }
    except Exception as exc:  # noqa: BLE001
        result["disk"] = {"error": f"{type(exc).__name__}: {exc}"}

    with _sys_lock:
        _sys_cache["at"] = now
        _sys_cache["data"] = result
    return result


def _load_avg() -> float | None:
    """1 分钟平均负载。Windows 上 `os.getloadavg` 不存在，返回 None。

    不编造 0：那会让人以为"负载为零"，而实际是"这个平台没有该指标"。
    """
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def _safe_count(proc, *names: str) -> int | None:
    """尝试多个属性名取计数。

    `num_fds` 只在 Linux/macOS 有，Windows 上是 `num_handles`；
    psutil 还提供跨平台的 `open_files()` / `net_connections()` 方法。
    取不到就返回 None，**不编造 0**。
    """
    for name in names:
        try:
            attr = getattr(proc, name, None)
            if attr is None:
                continue
            value = attr() if callable(attr) else attr
            if isinstance(value, (list, tuple, set)):
                return len(value)
            if isinstance(value, int):
                return value
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# 其他组件的聚合
# ---------------------------------------------------------------------------


def component_metrics(
    agent: Any = None,
    limiters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把各组件已有的 snapshot 汇总成一段。

    全部用 getattr + try 兜底：监控是**观测**，某个组件取不到数据只应
    表现为"那一段缺失"，而不是整个 `/api/metrics` 失败。
    """
    out: dict[str, Any] = {}

    # 限流
    if limiters:
        tiers = {}
        for name, lim in limiters.items():
            try:
                tiers[name] = lim.snapshot()
            except Exception as exc:  # noqa: BLE001
                tiers[name] = {"error": f"{type(exc).__name__}: {exc}"}
        out["rate_limit"] = tiers

    # 检索缓存命中率
    try:
        # ★ 必须用单例（`rag_pipeline`），不要 `RAGPipeline()`
        # ------------------------------------------------
        # 缓存与命中计数都在**实例属性**上，现 new 一个实例读到的永远是
        # "命中 0 / 未命中 0" —— 监控会显示"缓存从没被用过"，
        # 而实际 Agent 路径一直在命中（那用的是单例）。
        # 这类 bug 的表现为"监控数字恒为 0"，不会报错，极易被当成"没流量"。
        from src.rag.pipeline import rag_pipeline

        info = rag_pipeline.cache_info()
        hits, misses = info.get("hits", 0), info.get("misses", 0)
        info["hit_rate"] = round(hits / (hits + misses), 4) if (hits + misses) else 0.0
        out["retrieval_cache"] = info
    except Exception as exc:  # noqa: BLE001
        out["retrieval_cache"] = {"error": f"{type(exc).__name__}: {exc}"}

    # LLM 客户端（provider 链与切换次数）
    if agent is not None:
        try:
            client = agent._get_llm_client("")
            out["llm"] = {
                "provider": getattr(client, "provider", ""),
                "chain": list(getattr(client, "chain", []) or []),
                "last_used": getattr(client, "last_used_provider", "") or "",
                "switch_count": getattr(client, "switch_count", 0),
                # 只报失败**次数**与最近一条摘要，不返回全部堆栈
                "recent_errors": len(getattr(client, "last_error", []) or []),
            }
        except Exception as exc:  # noqa: BLE001
            out["llm"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 日志级别分布（从环形缓冲统计，让"最近有没有报错"一眼可见）
    try:
        from src.logging_config import log_buffer

        counts: dict[str, int] = defaultdict(int)
        for item in log_buffer.snapshot():
            counts[str(item.get("level", "?"))] += 1
        out["logs"] = {
            "counts": dict(counts),
            "total": sum(counts.values()),
            "buffer_size": getattr(log_buffer, "_capacity", None),
        }
    except Exception as exc:  # noqa: BLE001
        out["logs"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Langfuse（可选上报组件）
    # ----------------------
    # ★ 为什么要进 metrics 而不只放 health
    # --------------------------------
    # `enabled=true` 只说明"配置齐、开关开"，回答不了运维最关心的问题：
    # **到底发出去了没有**。可能开着但每条都失败（key 过期、网络不通）。
    # 故这里带上进程级计数（推送成功/失败数、冷却次数、flush 超时数），
    # 让"上报是不是在正常工作"变成可观测、可告警的数字。
    try:
        from src.langfuse_tracing import metrics as _langfuse_metrics

        out["langfuse"] = _langfuse_metrics()
    except Exception as exc:  # noqa: BLE001
        out["langfuse"] = {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}

    return out


def build_metrics(
    agent: Any = None,
    limiters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`/api/metrics` 的响应体。"""
    return {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requests": metrics.snapshot(),
        "system": system_metrics(),
        "components": component_metrics(agent, limiters),
        # ★ 阶段耗时聚合（局限 6）。空 dict 表示还没记录过问答 ——
        #   前端据此显示暂无数据而不是画一条 0 线。
        "phases": phase_metrics(),
    }


def render_prometheus(
    agent: Any = None,
    limiters: dict[str, Any] | None = None,
) -> str:
    """把同一份数据渲染成 Prometheus 文本格式（`/api/metrics?format=prometheus`）。

    ★ 为什么顺手提供
    -------------
    这是**零成本的可扩展性**：数据已在内存里，换个渲染而已。
    将来真要接 Prometheus / VictoriaMetrics，直接把 scrape 指向这个端点；
    在接之前也不影响（没人抓就没有开销）。

    格式遵循 text exposition format 0.0.4：
        # HELP <name> <说明>
        # TYPE <name> <gauge|counter>
        <name>{label="value"} <数值>
    """
    data = build_metrics(agent, limiters)
    lines: list[str] = []

    def emit(name: str, help_text: str, kind: str, samples: list[tuple[str, float]]) -> None:
        if not samples:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            lines.append(f"{name}{labels} {value}")

    overall = data["requests"]["overall"]
    emit("bidding_requests_total", "累计请求数", "counter",
         [("", float(overall["requests"]))])
    emit("bidding_errors_total", "累计 4xx/5xx 数", "counter",
         [("", float(overall["errors"]))])
    if "p95_ms" in overall:
        emit("bidding_request_p95_ms", "请求延迟 P95（毫秒）", "gauge",
             [("", float(overall["p95_ms"]))])
        emit("bidding_request_p50_ms", "请求延迟 P50（毫秒）", "gauge",
             [("", float(overall["p50_ms"]))])

    emit("bidding_endpoint_requests_total", "按端点累计请求数", "counter",
         [(f'{{endpoint="{e["endpoint"]}"}}', float(e["requests"]))
          for e in data["requests"]["endpoints"]])
    emit("bidding_endpoint_p95_ms", "按端点延迟 P95（毫秒）", "gauge",
         [(f'{{endpoint="{e["endpoint"]}"}}', float(e["p95_ms"]))
          for e in data["requests"]["endpoints"] if "p95_ms" in e])

    proc = data["system"].get("process") or {}
    if "rss_mb" in proc:
        emit("bidding_process_rss_mb", "进程常驻内存（MB）", "gauge",
             [("", float(proc["rss_mb"]))])
        emit("bidding_process_cpu_percent", "进程 CPU 占用（%）", "gauge",
             [("", float(proc["cpu_percent"]))])
        emit("bidding_process_threads", "进程线程数", "gauge",
             [("", float(proc["threads"]))])
    mem = data["system"].get("memory") or {}
    if "percent" in mem:
        emit("bidding_host_memory_percent", "主机内存占用（%）", "gauge",
             [("", float(mem["percent"]))])

    rl = data["components"].get("rate_limit") or {}
    emit("bidding_rate_limit_rejected_total", "被限流拒绝的请求数", "counter",
         [(f'{{tier="{k}"}}', float(v.get("rejected", 0)))
          for k, v in rl.items() if isinstance(v, dict)])

    rc = data["components"].get("retrieval_cache") or {}
    if "hit_rate" in rc:
        emit("bidding_retrieval_cache_hit_rate", "检索缓存命中率", "gauge",
             [("", float(rc["hit_rate"]))])

    # Langfuse 上报（可选组件）：**能不能上报**与**实际上报了多少**分开暴露。
    # 只报 enabled 不够——开着但每条都失败是最需要告警的状态。
    lf = data["components"].get("langfuse") or {}
    if isinstance(lf, dict) and lf:
        counters = lf.get("counters") or {}
        emit("bidding_langfuse_enabled", "Langfuse 上报是否启用（1/0）", "gauge",
             [("", 1.0 if lf.get("enabled") else 0.0)])
        emit("bidding_langfuse_sample_rate", "Langfuse 采样率", "gauge",
             [("", float(lf.get("sample_rate") or 0.0))])
        emit("bidding_langfuse_traces_total", "已开启上报的 trace 数", "counter",
             [("", float(counters.get("traces_started", 0)))])
        emit("bidding_langfuse_traces_sampled_out_total", "被采样率排除的 trace 数", "counter",
             [("", float(counters.get("traces_sampled_out", 0)))])
        emit("bidding_langfuse_spans_pushed_total", "成功推入上报队列的 observation 数", "counter",
             [("", float(counters.get("spans_pushed", 0)))])
        emit("bidding_langfuse_spans_failed_total", "上报失败的 observation 数", "counter",
             [("", float(counters.get("spans_failed", 0)))])
        emit("bidding_langfuse_cooldowns_total", "进入失败冷却的次数", "counter",
             [("", float(counters.get("cooldowns_entered", 0)))])
        emit("bidding_langfuse_flush_timeouts_total", "flush 超时次数", "counter",
             [("", float(counters.get("flush_timeouts", 0)))])
        emit("bidding_langfuse_captured_chars_total", "已上报的正文字符数（脱敏后）", "counter",
             [("", float(counters.get("captured_chars", 0)))])
        emit("bidding_langfuse_redactions_total", "脱敏替换次数", "counter",
             [("", float(counters.get("redactions", 0)))])

    return "\n".join(lines) + "\n"
