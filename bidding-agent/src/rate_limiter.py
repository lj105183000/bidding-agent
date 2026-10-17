"""请求限流：分层配额 + 可信代理下的真实客户端识别。

★ 原实现的两个严重问题（实测确认）
============================

**① 反向代理下所有人共享一个限流桶**

    `_client_ip` 只看 `request.client.host`，**完全不读 `X-Forwarded-For`**。

    任何反代/网关部署（nginx、Caddy、Traefik、云 LB、Docker 端口映射……）下，
    `request.client.host` 都是**代理的 IP**（如 `172.18.0.1`）。
    于是所有用户共用一个桶：30 次/分钟一到，**全站用户一起被拒**。
    这不是"限流偏严"，而是"限流完全失效且误伤所有人"。

**② 63 个端点只有 2 个前缀被限流**

    只有 `/api/chat*` 与 `/api/graph*`。而下面这些**重度消耗算力或额度**的
    端点完全没有保护：

        /api/extract          文档抽取（可触发 LLM 与本地模型）
        /api/vision           视觉识别（外部 API，按次计费）
        /api/tender/parse     PDF/Word 解析（CPU + IO 密集）
        /api/web/search       联网搜索（Tavily，按次计费）
        /api/price-compare    比价（多地抓取）

    一个客户端连点就能把额度和 CPU 打满。

★ 为什么 `X-Forwarded-For` 不能无条件信任
=====================================
`X-Forwarded-For` 是**客户端可伪造**的头。若直接采信，攻击者每次请求
换一个假 IP 就能绕过限流——**比不读它还危险**（看起来有防护，实际没有）。

正确做法：**只有直连方在可信代理列表里时，才采信转发头**。

    request.client.host ∈ TRUSTED_PROXIES  → 取 XFF 最右侧的**非可信** IP
    否则                                    → 用 request.client.host

取"最右侧非可信"而不是"最左侧"：XFF 是**追加**的，最左侧是客户端自己写的
（可伪造），最右侧是最后一个代理看到的真实来源。

★ 分层而不是统一配额
==================
统一 30 次/分钟对"读列表"太严（正常翻页就超了），对"调 LLM"又太松
（30 次 LLM 调用可能就是几块钱）。故按**代价**分层：

    heavy   抽取/解析/视觉/报告 —— 最贵，配额最低
    llm     问答/比对/合规     —— 走 LLM，很贵
    context 只读 + 图谱/统计   —— 便宜，给足
    none    健康/日志/静态     —— 不限（编排器与排障不能被限流挡住）

**限流健康检查是自找麻烦**：编排器每 10~30 秒探一次，
若被限流会判定"服务不可用"并重启——而服务其实好好的。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

#: 可信代理的直连 IP/CIDR（逗号分隔）。
#:
#: 留空 = **不信任任何转发头**（安全默认值）。只有当本服务**确实**部署在
#: 反代之后，才应把代理地址填进来。填错的后果很严重：
#:   · 填少了 → 所有用户仍共享一个桶（问题①依旧）
#:   · 填多了/填成 `0.0.0.0/0` → 任何人都能伪造 XFF 绕过限流
TRUSTED_PROXIES_ENV = "TRUSTED_PROXIES"

#: 单个分层最多跟踪多少个客户端。
#:
#: ★ 为什么必须设上限
#: ----------------
#: 原实现 `_requests: dict[str, deque]` **永不清理**。
#: 攻击者（或仅仅是一个扫描器）用大量不同源 IP 打过来，字典会无限增长——
#: 限流器自己成了内存耗尽攻击的入口。
#:
#: 10000 个 IP × 每个几十字节 ≈ 几 MB，可接受。
MAX_TRACKED_CLIENTS = 10_000

#: "丢弃最久未活动条目"警告的节流间隔（秒）。
#:
#: 持续被大量源 IP 冲击时，这条警告会变成日志噪音，
#: 而它想表达的只是"有人在用大量源 IP 打"——一条就够。
_WARN_INTERVAL_SECONDS = 300.0

#: IPv4 私有段 + 回环。默认把这些当作"可能的代理"来源，但仍需显式配置才采信。
_PRIVATE_PREFIXES = (
    "127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "::1", "fc", "fd",
)


def is_private_ip(ip: str) -> bool:
    """粗判是否内网/回环地址。

    刻意用前缀匹配而不是 `ipaddress` 模块：这里只需要一个**保守**的判断
    （判不准时返回 True 更安全，会走"需显式配置"路径），
    且避免为非 IPv4 的奇怪取值抛异常。
    """
    text = (ip or "").strip().lower()
    if not text:
        return True
    return text.startswith(_PRIVATE_PREFIXES)


def parse_trusted_proxies(raw: str | None = None) -> tuple[str, ...]:
    """解析可信代理配置。未配置时返回空元组（= 不信任任何转发头）。"""
    text = raw if raw is not None else os.getenv(TRUSTED_PROXIES_ENV, "")
    items = [p.strip() for p in str(text or "").split(",")]
    return tuple(p for p in items if p)


def _match_proxy(ip: str, trusted: tuple[str, ...]) -> bool:
    """`ip` 是否属于可信代理列表。

    支持精确匹配与前缀匹配（`172.18.` 匹配 `172.18.0.5`）——
    反代在容器网络里 IP 会变，写死精确值容易被下一次重建打破。
    """
    if not trusted:
        return False
    for entry in trusted:
        if entry == "*":
            # 显式允许"信任所有"：仅当本服务**只**能被代理访问
            # （如集群内 Service、前置 nginx 完全可控）时才该这么配。
            #
            # ⚠️ 这会让客户端可以伪造 XFF 来绕过限流——单列这个取值是
            # 为了让该取舍**显式**（写在配置里，而不是让运维去堆网段、
            # 结果不小心写了个覆盖全网的段）。
            return True
        if ip == entry or ip.startswith(entry.rstrip("/")):
            return True
    return False


def client_ip(
    direct_ip: str,
    forwarded_for: str = "",
    real_ip: str = "",
    trusted: tuple[str, ...] | None = None,
) -> tuple[str, bool]:
    """解析真实客户端 IP。返回 `(ip, 是否采信了转发头)`。

    ★ 取"最右侧非可信"而不是"最左侧"
    ------------------------------
    `X-Forwarded-For: client, proxy1, proxy2` —— 这个头是**逐跳追加**的。
    最左侧是客户端自己写的（**可随意伪造**），最右侧是最后一个代理看到的。

    正确算法：从右往左，跳过所有可信代理，第一个不可信的就是真实客户端。
    这样即使客户端伪造了一串假 IP 在最左边，也不会被采信。

    ★ 第二个返回值（`used_forwarded`）的意义
    ------------------------------------
    它表示"**最终用的 IP 是不是从转发头来的**"——直接回答
    "限流为什么把所有人当一个客户端"这类问题。

    注意语义是"IP 来源"而不是"有没有看到转发头"：
    若 XFF 全是可信代理、最终回退到直连 IP，则返回 False
    （因为用的是直连的那一个），尽管确实读了 XFF。
    """
    trusted = parse_trusted_proxies() if trusted is None else trusted
    direct = (direct_ip or "").strip()

    if not _match_proxy(direct, trusted):
        # 直连方不可信 → 转发头一律不采信（否则可被伪造绕过）
        return (direct or "unknown"), False

    chain = [p.strip() for p in (forwarded_for or "").split(",") if p.strip()]

    # ★ `"*"`（信任所有代理）要单独处理
    # --------------------------------
    # 泛化算法是"从右往左找第一个非可信 IP"。但 `"*"` 让**每一跳都可信**，
    # 于是循环找不到任何东西、最终回退到直连 IP —— 即 `"*"` 配了等于没配
    # （实测：`client_ip("203.0.113.9", "1.2.3.4", "", ("*",))` 返回直连的
    # `203.0.113.9`，而预期是 XFF 里的 `1.2.3.4`）。
    #
    # 语义上 `"*"` 的配置者是在说"我这套部署的前置代理全部可控，
    # 整条链都可以信"。那就取**最左侧**（客户端来源）那个：
    # 这是该配置下唯一有意义的取值方式，也把安全取舍摊开在配置层
    # （谁配 `"*"` 谁承担"客户端可伪造 XFF"的风险）。
    if "*" in trusted and chain:
        return chain[0], True

    # 常规：从右往左找第一个非可信 IP
    for hop in reversed(chain):
        if not _match_proxy(hop, trusted):
            return hop, True

    # XFF 全是可信代理或为空 → 退回 X-Real-IP（也是转发头）
    real = (real_ip or "").strip()
    if real and not _match_proxy(real, trusted):
        return real, True
    # 都不可用 → 用直连 IP。此时**不是**转发头来源
    return (direct or "unknown"), False


class RateLimiter:
    """基于客户端标识的滑动窗口限流器（线程安全、内存有界）。

    方法名 `allow` / `is_allowed` / `reset` 与既有测试契约保持一致。
    """

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: int = 60,
        *,
        max_tracked: int = MAX_TRACKED_CLIENTS,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._max_tracked = max_tracked
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        #: 上一次清理时间戳（惰性清理，避免每次请求都遍历全表）
        self._last_prune = time.monotonic()
        #: 上一次"超上限"警告时间戳（警告自身也要节流，否则刷屏）
        self._last_cap_warning = 0.0
        #: 观测计数
        self.allowed_count = 0
        self.rejected_count = 0

    def allow(self, client_ip: str) -> bool:
        """判断客户端当前请求是否允许通过（方法名与测试契约一致）。

        ★ 被拒绝时**不写入时间戳**
        ------------------------
        否则持续超限的客户端会把时间戳一直往后推，形成
        "被限流后窗口永远不释放"的效果（类似惩罚性封禁），
        而这与滑动窗口的语义不符：窗口只统计**已放行的**请求。
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            self._prune_locked(now, window_start)
            timestamps = self._requests[client_ip]

            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                self.rejected_count += 1
                return False

            timestamps.append(now)
            self.allowed_count += 1
            return True

    # 兼容别名：部分早期调用方使用 is_allowed
    def is_allowed(self, client_ip: str) -> bool:
        return self.allow(client_ip)

    def remaining(self, client_ip: str) -> int:
        """当前窗口内还剩多少次配额（供响应头使用）。"""
        now = time.monotonic()
        window_start = now - self.window_seconds
        with self._lock:
            timestamps = self._requests.get(client_ip)
            if not timestamps:
                return self.max_requests
            live = sum(1 for ts in timestamps if ts > window_start)
            return max(0, self.max_requests - live)

    def retry_after(self, client_ip: str) -> int:
        """建议多久后重试（秒）。至少 1，避免 `Retry-After: 0` 促使客户端立即重试。"""
        now = time.monotonic()
        window_start = now - self.window_seconds
        with self._lock:
            timestamps = self._requests.get(client_ip)
            if not timestamps:
                return 0
            live = [ts for ts in timestamps if ts > window_start]
            if len(live) < self.max_requests:
                return 0
            # 最早那次放行的请求滑出窗口时，就腾出一个名额
            oldest = min(live)
        return max(1, int(oldest + self.window_seconds - now) + 1)

    def snapshot(self) -> dict[str, Any]:
        """观测快照（供健康检查/排障展示）。"""
        with self._lock:
            tracked = len(self._requests)
        total = self.allowed_count + self.rejected_count
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "tracked_clients": tracked,
            "max_tracked": self._max_tracked,
            "allowed": self.allowed_count,
            "rejected": self.rejected_count,
            "reject_rate": round(self.rejected_count / total, 4) if total else 0.0,
        }

    def _prune_locked(self, now: float, window_start: float) -> None:
        """惰性清理：腾出过期条目，并在超上限时丢弃最久未活动的客户端。

        ★ 为什么是"惰性"而不是定时任务
        ---------------------------
        定时任务要在应用生命周期里管一个线程（启动/停止/异常）。
        而限流器本身就是高频调用的，顺手做清理更简单，也没有生命周期问题。

        ★ 为什么按"最久未活动"丢
        ----------------------
        正常用户会持续请求，扫描器换 IP 打一枪就换。丢掉长期不活动的条目
        对真实用户影响最小。

        ★ 两个节流（实测踩过刷屏）
        ----------------------
        ① `_last_prune` 只在**真正做了全表遍历**时才更新；
           第一版在提前返回前就赋了值，导致"上次清理的 30 秒内不再清理"
           的条件失效——超上限后**每次请求都触发一次丢弃 + 一条警告**，
           实测刷了 17 条同样的警告。
        ② 丢弃警告加独立的节流间隔（`_WARN_INTERVAL_SECONDS`）：
           在持续被大量 IP 冲击时，这个警告本身会变成日志噪音，
           而它想表达的只是"有人在用大量源 IP 打"——一条就够。
        """
        over_cap = len(self._requests) > self._max_tracked
        if not over_cap and now - self._last_prune < 30.0:
            return
        self._last_prune = now

        # 先清掉窗口已完全过期的条目
        stale = [
            ip for ip, ts in self._requests.items()
            if not ts or ts[-1] <= window_start
        ]
        for ip in stale:
            self._requests.pop(ip, None)

        # 仍超上限 → 丢最久未活动的一半（一次腾出较多空间，避免每次插入都淘汰）
        if len(self._requests) > self._max_tracked:
            ordered = sorted(self._requests.items(), key=lambda kv: kv[1][-1] if kv[1] else 0.0)
            drop = len(ordered) - self._max_tracked // 2
            for ip, _ in ordered[:drop]:
                self._requests.pop(ip, None)
            # 警告节流：这是"被大量源 IP 冲击"的信号，一条足够
            if now - self._last_cap_warning >= _WARN_INTERVAL_SECONDS:
                self._last_cap_warning = now
                logger.warning(
                    "限流器跟踪的客户端超过上限（%d），已丢弃 %d 个最久未活动的条目。"
                    "若持续出现，说明有大量不同源 IP 在打（扫描器/攻击），"
                    "或 TRUSTED_PROXIES 未配置导致所有请求共用一个 IP",
                    self._max_tracked, drop,
                )

    def reset(self, client_ip: str | None = None) -> None:
        """清空指定客户端（或全部）的限流计数，便于测试。"""
        with self._lock:
            if client_ip is None:
                self._requests.clear()
                self.allowed_count = 0
                self.rejected_count = 0
            else:
                self._requests.pop(client_ip, None)


# ---------------------------------------------------------------------------
# 分层配额
# ---------------------------------------------------------------------------

#: 端点 → 配额层。**按代价分层**，不是按路径前缀随便分组。
#:
#: 每项 `(前缀, 层名)`；匹配时取**最长匹配**（不是第一个匹配），
#: 这样 `/api/chat/stream` 能覆盖 `/api/chat`，而不受列表顺序影响。
TIER_RULES: tuple[tuple[str, str], ...] = (
    # --- none：绝不能限流 ---
    # 编排器按秒级间隔探活；限流会让它判定服务不可用并重启，而服务其实好好的。
    # `/api/logs` 是排障通道，出问题时更需要它可用。
    ("/api/health", "none"),
    ("/api/logs", "none"),

    # --- heavy：最贵（LLM 抽取 / 视觉 API / 文档解析 / 报告生成）---
    ("/api/extract", "heavy"),
    ("/api/vision", "heavy"),
    ("/api/tender/parse", "heavy"),
    ("/api/tender/extract", "heavy"),
    ("/api/report", "heavy"),
    ("/api/compliance/check/upload", "heavy"),
    ("/api/qualification/check/upload", "heavy"),

    # --- llm：走大模型（问答 / 比对 / 合规检查 / 联网搜索）---
    ("/api/chat", "llm"),
    ("/api/ask", "llm"),
    ("/api/web/search", "llm"),
    ("/api/tender/compare", "llm"),
    ("/api/price-compare", "llm"),
    ("/api/compliance", "llm"),
    ("/api/qualification", "llm"),
    ("/api/material/match", "llm"),

    # --- context：只读，便宜（图谱 / 统计 / 会话 / 列表）---
    ("/api/graph", "context"),
    ("/api/analysis", "context"),
    ("/api/competitor", "context"),
    ("/api/conversations", "context"),
    ("/api/tender", "context"),
    ("/api/material", "context"),
    ("/api/feedback", "context"),
    ("/api/web/coverage", "context"),
    ("/api/web/status", "context"),
    ("/api/debug", "context"),
)

#: 各层的默认配额 `(每窗口请求数, 窗口秒数)`。
#:
#: 取值依据：
#:   · llm 层保持与原实现一致的 30/60s（既有行为不变）
#:   · heavy 层收到 10/60s——单次就可能花几秒 CPU 或一次 API 计费
#:   · context 层放宽到 120/60s——正常前端翻页、切页签会打不少请求，
#:     卡太严会误伤（前端 Dashboard 一次加载就并发好几个）
#:   · none 层不设限
DEFAULT_TIERS: dict[str, tuple[int, int]] = {
    "heavy": (10, 60),
    "llm": (30, 60),
    "context": (120, 60),
    "none": (0, 0),  # 0 = 不限
}


def tier_of(path: str) -> str:
    """按**最长前缀**匹配配额层。未匹配到的走 `context`（宽松，不误伤）。"""
    best_tier = "context"
    best_len = -1
    for prefix, tier in TIER_RULES:
        if path.startswith(prefix) and len(prefix) > best_len:
            best_tier = tier
            best_len = len(prefix)
    return best_tier


def build_limiters(
    overrides: dict[str, tuple[int, int]] | None = None,
) -> dict[str, RateLimiter]:
    """按分层构造限流器。

    `overrides` 可覆盖默认配额（来自环境变量）。`none` 层**不构造限流器**——
    让"不限流"在代码里是"没有这个限流器"，而不是"配额为 0 的特殊分支"，
    少一处可能写错的地方。
    """
    tiers = dict(DEFAULT_TIERS)
    if overrides:
        tiers.update(overrides)
    limiters: dict[str, RateLimiter] = {}
    for name, (max_req, window) in tiers.items():
        if name == "none" or max_req <= 0 or window <= 0:
            continue
        limiters[name] = RateLimiter(max_requests=max_req, window_seconds=window)
    return limiters


def rate_limit_headers(limiter: RateLimiter, remaining: int, retry_after: int = 0) -> dict[str, str]:
    """标准限流响应头。

    ★ 为什么必须回传（原实现没有）
    ---------------------------
    客户端不知道配额，只能盲目重试；而"盲目重试"恰恰让限流更糟。
    回传 `X-RateLimit-Remaining` / `Retry-After` 后，前端可以做
    指数退避与友好提示，而不是让用户看着一个没有解释的失败。

    用 `X-RateLimit-*`（事实标准）而不是自定义头，便于通用客户端识别。
    """
    headers = {
        "X-RateLimit-Limit": str(limiter.max_requests),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Window": str(limiter.window_seconds),
    }
    if retry_after > 0:
        headers["Retry-After"] = str(retry_after)
    return headers


def check_request(
    path: str,
    direct_ip: str,
    forwarded_for: str = "",
    real_ip: str = "",
    limiters: dict[str, RateLimiter] | None = None,
    trusted: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """限流判定的**纯函数**版本（供中间件调用，也便于单测）。

    ``tier`` / ``limited`` / ``allowed`` / ``key`` / ``remaining`` / ``retry_after``
    / ``headers`` / ``used_forwarded`` / ``elapsed_ms``

    `elapsed_ms` 是**判定本身的耗时**（微秒级）。它存在的意义是让 429 响应
    也有一个真实的耗时值可记入指标——用 0 也能跑，但那样监控里会看到
    "429 的延迟恒为 0"，与真实情况不符（且掩盖了"限流器本身变慢"这种可能）。

    ★ 为什么抽成纯函数
    --------------
    中间件写在 `api/server.py` 里、且是 FastAPI 装饰器形态，
    要测它就得起整个应用——而 `api/server.py` 的 import 会拉起
    Agent、RAG、数据库客户端等一堆重依赖（实测：跑一个 `/api/ask`
    会触发真实模型下载并卡住）。把判定逻辑抽出来之后，测试可以对
    **任意路径**验证分层与配额行为，不必启动应用、不碰任何外部依赖。
    """
    import time as _time

    _t0 = _time.perf_counter()
    tier = tier_of(path)
    pool = limiters if limiters is not None else {}
    limiter = pool.get(tier)

    if limiter is None:
        # `none` 层或未配置的层：直接放行
        return {
            "tier": tier, "limited": False, "allowed": True,
            "key": "", "remaining": -1, "retry_after": 0,
            "headers": {}, "used_forwarded": False, "elapsed_ms": 0.0,
        }

    key, used_forwarded = client_ip(direct_ip, forwarded_for, real_ip, trusted)
    allowed = limiter.allow(key)
    elapsed_ms = round((_time.perf_counter() - _t0) * 1000, 3)
    if allowed:
        return {
            "tier": tier, "limited": True, "allowed": True, "key": key,
            "remaining": limiter.remaining(key), "retry_after": 0,
            "headers": rate_limit_headers(limiter, limiter.remaining(key)),
            "used_forwarded": used_forwarded, "elapsed_ms": elapsed_ms,
        }
    retry_after = limiter.retry_after(key)
    return {
        "tier": tier, "limited": True, "allowed": False, "key": key,
        "remaining": 0, "retry_after": retry_after,
        "headers": rate_limit_headers(limiter, 0, retry_after),
        "used_forwarded": used_forwarded, "elapsed_ms": elapsed_ms,
    }
