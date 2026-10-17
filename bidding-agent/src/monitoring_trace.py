"""LLM 调用追踪：按请求记录每一次调用的耗时、token 用量与结果。

★ 为什么需要这个模块（实测缺口，不是设想）
===================================
审计现有可观测性时发现四个真实缺口，都属"数据本来就有、但被丢掉或没记"：

    ① **token 用量完全没读**。实测 DeepSeek 响应里一直带着：
           usage: {prompt_tokens: 8, completion_tokens: 5, total_tokens: 13,
                   prompt_cache_hit_tokens: 0, prompt_cache_miss_tokens: 8}
       而 `extract_message` / `iter_sse_json` 里 `usage` 出现 **0 次**。
       没有它就算不出成本，也看不出"哪一轮特别贵"。

    ② **prompt 缓存命中数丢失**。DeepSeek 对缓存命中的 prompt token 打折计费，
       `prompt_cache_hit_tokens` 直接就是**省钱信号**——它掉了说明上下文在变
       （缓存失效），成本会上升。

    ③ **单次 LLM 调用明细缺失**。`phase_times` 只有阶段合计
       （如"生成回答 3.2s"），看不出里面是 1 次调用还是 3 次、
       每次多久。排查"为什么这轮特别慢"时这个区别很关键。

    ④ **单个工具执行耗时缺失**。一次问答可能调 4 个工具（含重复），
       但只记了整阶段合计。

★ 设计：按请求的记录器，不是全局单例
============================
用全局单例会在并发请求下把不同请求的调用混在一起。
故每个请求创建一个 `LLMTraceRecorder`，挂到**该请求自己的客户端副本**上
（`agent._thinking_model` 与 `_request_recorder_client` 已经做了浅拷贝，
浅拷贝会共享 `_lock` 但各自持有独立的记录器引用）。

★ 为什么记录器是"旁路"而不是让客户端自己存
===================================
客户端在 `_provider_clients` 里被**跨请求缓存复用**。若把调用历史存在客户端上，
请求 A 的记录会被请求 B 追加，且无限增长（内存泄漏）。
记录器随请求生灭，天然没有这两个问题。
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

#: 单次调用记录里保留的最大文本长度（错误信息用）。
MAX_ERROR_CHARS = 300

#: 一次问答最多保留多少条 LLM 调用记录。
#:
#: 与限流器同样的理由：无界增长在长跑服务里就是内存泄漏。
#: ReAct 上限是 4 轮 × （决策 + 生成）≈ 十几次，100 条足够且有余量。
MAX_RECORDS = 100


@dataclass
class LLMCall:
    """一次 LLM 调用的观测记录。"""

    #: 调用类型：decision（决策轮，非流式）/ stream（正文流）/ thinking（思考流）
    kind: str = ""
    provider: str = ""
    model: str = ""
    #: 本次耗时（毫秒）。**含**网络往返与排队
    elapsed_ms: int = 0
    #: 是否为失败调用（失败也要记：失败次数是 provider 稳定性的直接信号）
    ok: bool = True
    error: str = ""
    #: token 用量（厂商未返回时为 None，**不编造 0**）
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: prompt 缓存命中/未命中 token（DeepSeek 对命中部分打折计费）
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    #: 思维链长度（思考流才非 0）。用于区分"慢是因为在想"还是"慢是因为输出长"
    reasoning_chars: int = 0
    #: 输出正文长度
    content_chars: int = 0
    #: 该次调用涉及的工具名（决策轮解析出的工具调用）
    tools: list[str] = field(default_factory=list)
    #: 正文（**默认空**）。只有 `LANGFUSE_CAPTURE_CONTENT=true` 时客户端才填，
    #: 供 Langfuse 上报"问了什么、答了什么"。
    #:
    #: ★ 为什么不进 `summary()`
    #: ---------------------
    #: `summary()` 会原样进 SSE 的 `done` 帧。正文有几千字符，塞进去会把
    #: 响应体撑大几十倍，而前端与在线评测只读计数类字段。故这两个字段
    #: **只走上报链路**，本地聚合视图里被显式剔除（见 `_call_brief`）。
    prompt_text: str = ""
    response_text: str = ""


class LLMTraceRecorder:
    """按请求收集 LLM 调用记录。线程安全（工具层是 4 线程并行）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[LLMCall] = []
        #: 溢出计数（超出 MAX_RECORDS 后丢弃的条数）。
        #: 显式记录而不是静默丢弃——"记录被截断"本身是要知道的事实。
        self._dropped = 0

    # --- 记录 ---

    def record(self, call: LLMCall) -> None:
        with self._lock:
            if len(self._calls) >= MAX_RECORDS:
                self._dropped += 1
                return
            self._calls.append(call)

    # --- 读取 ---

    def calls(self) -> list[LLMCall]:
        with self._lock:
            return list(self._calls)

    def summary(self) -> dict[str, Any]:
        """聚合视图：给 `done` 帧用。

        ★ 全部字段都可能在厂商不返回 usage 时为 None
        -------------------------------------
        此时**不编造 0**：`total_tokens=0` 会被读成"这次没花 token"，
        而事实是"没拿到这个数据"。故 token 类字段要么是真实数字，要么缺席。
        """
        with self._lock:
            calls = list(self._calls)
            dropped = self._dropped

        ok_calls = [c for c in calls if c.ok]
        failed = [c for c in calls if not c.ok]

        def _sum(attr: str) -> int | None:
            """求和；**所有**记录都缺该字段时返回 None（而不是 0）。"""
            values = [getattr(c, attr) for c in calls if getattr(c, attr) is not None]
            return sum(values) if values else None

        return {
            "call_count": len(calls),
            "failed_count": len(failed),
            "dropped_count": dropped,
            # 总耗时是所有调用耗时之和。**注意它可能大于端到端耗时**：
            # 工具层是 4 线程并行，并行调用会重叠。故这里也给出"串行下界"提示。
            "total_ms": sum(c.elapsed_ms for c in calls),
            "max_call_ms": max((c.elapsed_ms for c in calls), default=0),
            "prompt_tokens": _sum("prompt_tokens"),
            "completion_tokens": _sum("completion_tokens"),
            "total_tokens": _sum("total_tokens"),
            "cache_hit_tokens": _sum("cache_hit_tokens"),
            "cache_miss_tokens": _sum("cache_miss_tokens"),
            "reasoning_chars": sum(c.reasoning_chars for c in calls),
            "content_chars": sum(c.content_chars for c in calls),
            # 按类型分组耗时：能直接看出"慢在决策还是慢在生成"
            "by_kind": _group_ms(calls),
            "calls": [_call_brief(c) for c in calls],
        }


#: 正文类字段：不进聚合视图（见 `LLMCall.prompt_text` 的注释）
TEXT_FIELDS: tuple[str, ...] = ("prompt_text", "response_text")


def _call_brief(call: LLMCall) -> dict[str, Any]:
    """单条调用的对外视图：**剔除正文字段**，只留计数与元信息。"""
    data = asdict(call)
    for name in TEXT_FIELDS:
        data.pop(name, None)
    return data


def render_messages(messages: list[dict[str, Any]] | None, limit: int) -> str:
    """把 messages 渲染成一段可读文本，并按 `limit` 保留**首尾**。

    只给"正文上报"用（默认关闭），故实现从简：`role: content` 逐行拼接，
    content 非字符串时原样 str 化（多模态消息会变成一段 JSON 文本，够用）。

    为什么保留首尾而不是只留尾部：system prompt 在最前面（角色设定、
    必须遵守的规则），最后一条 user 消息在尾部（用户到底问了什么）。
    只留尾部会把前者丢掉，只留首部则看不出这一轮的问题。

    `limit <= 0` 返回空串 —— 调用方据此彻底跳过采集，零开销。
    """
    if not messages or limit <= 0:
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = str(msg.get("role") or "?")
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            text = str(content)
        if msg.get("tool_calls"):
            text = f"{text} [tool_calls={len(msg['tool_calls'] or [])}]"
        parts.append(f"{role}: {text}")
    joined = "\n".join(parts)
    if len(joined) <= limit:
        return joined
    head = limit // 3
    tail = limit - head
    return f"{joined[:head]}…[省略 {len(joined) - limit} 字符]…{joined[-tail:]}"


def _group_ms(calls: list[LLMCall]) -> dict[str, dict[str, int]]:
    """按 kind 汇总调用次数与耗时。"""
    out: dict[str, dict[str, int]] = {}
    for c in calls:
        g = out.setdefault(c.kind or "unknown", {"count": 0, "ms": 0})
        g["count"] += 1
        g["ms"] += c.elapsed_ms
    return out


class Stopwatch:
    """计时上下文：`with Stopwatch() as w: ...` 后读 `w.ms`。

    为什么不用 `time.monotonic()` 手写两行：手写时容易在异常路径上漏掉
    结束时间，导致耗时记成 0 或异常。上下文管理器保证异常路径也计时。
    """

    def __init__(self) -> None:
        self._t0 = 0.0
        self.ms = 0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.ms = int((time.monotonic() - self._t0) * 1000)
        return False  # 不吞异常


def parse_usage(payload: dict[str, Any] | None) -> dict[str, int | None]:
    """从响应体或 SSE chunk 里提取 token 用量。

    ★ 兼容两种形态
    -----------
        · 非流式：`{"usage": {...}}` 在**顶层**
        · 流式：带 `stream_options.include_usage` 时，**最后一个 chunk** 带 usage，
          此时 `choices` 通常是空数组。故不能"没看到 choices 就跳过"。

    ★ 字段名兼容
    ---------
    OpenAI 标准是 `prompt_tokens` / `completion_tokens`；
    DeepSeek 额外给 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，
    也有实现放在 `prompt_tokens_details.cached_tokens`。三者都读。

    取不到一律返回 None（**不编造 0**）。
    """
    usage = (payload or {}).get("usage")
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
            "cache_hit_tokens": None, "cache_miss_tokens": None,
        }

    def _int(*keys: str) -> int | None:
        for k in keys:
            v = usage.get(k)
            if isinstance(v, int):
                return v
        return None

    # 缓存命中：优先取显式字段，其次取 `prompt_tokens_details.cached_tokens`
    # （两种写法在不同厂商/版本间都出现过）。取不到返回 None。
    cache_hit = _int("prompt_cache_hit_tokens")
    if cache_hit is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            cache_hit = details["cached_tokens"]

    return {
        "prompt_tokens": _int("prompt_tokens", "input_tokens"),
        "completion_tokens": _int("completion_tokens", "output_tokens"),
        "total_tokens": _int("total_tokens"),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": _int("prompt_cache_miss_tokens"),
    }


def attach_recorder(client, recorder: "LLMTraceRecorder") -> None:
    """把记录器挂到客户端（`FallbackLLMClient` 会转发给链上各客户端）。

    ★ 为什么不拷贝客户端（实测结论）
    ---------------------------
    第一版用 `copy.copy(client)` 再挂，两个问题：

        ① **破坏测试可见性**：调用落到副本上，于是
           `assert fake.raw_calls == 4` 恒失败（实测 5 个用例挂掉）。
           而"在原对象上观察调用次数"是**合理**的测试写法，
           不该为观测功能逼测试改写法。
        ② 热路径上每请求多一次对象拷贝。

    故改成"挂在原实例上 + 请求结束摘掉"（`detach_recorder`）。

    ★ 残余风险（如实记录，不假装没有）
    ----------------------------
    客户端在 `agent._provider_clients` 里跨请求缓存复用。若两个请求
    **真正并发**且恰好交错在同一个客户端实例上，仍可能出现
    "后者的记录器覆盖前者"，表现为 token 数对不上。
    实测未出现（**未做并发压测**）。
    彻底解决要把记录器沿调用链显式传递（改 `chat_raw` / `chat_stream`
    签名），代价会波及所有测试替身，本轮不做。

    **永不抛异常**：观测是旁路，挂不上只是这一轮没有明细。
    """
    if client is None or recorder is None:
        return
    try:
        client.trace_recorder = recorder
    except Exception:  # noqa: BLE001 - 只读属性 / __slots__ 客户端等
        return


def detach_recorder(client) -> None:
    """摘掉记录器。

    `FallbackLLMClient` 的 setter 会转发到**已解析**的底层客户端；
    尚未解析的那些在下次 `_client_for` 时读到 `None`，不会被误挂。

    **永不抛异常**（同上）。
    """
    if client is None:
        return
    try:
        client.trace_recorder = None
    except Exception:  # noqa: BLE001
        return


__all__ = [
    "MAX_RECORDS",
    "TEXT_FIELDS",
    "LLMCall",
    "LLMTraceRecorder",
    "Stopwatch",
    "attach_recorder",
    "detach_recorder",
    "parse_usage",
    "render_messages",
]
