"""在线评测样本的**聚合视图**（供 `/api/eval/online` 与诊断用）。

★ 为什么聚合放在后端而不是让前端算
==========================
与 `phase_total_ms` 同一考虑：两端口径不一致时，用户会看到互相矛盾的数字。
后端算一次，前端只渲染。

★ 核心口径：**只统计"可评"的请求**
============================
首轮追问（`clarified`）没有答案、极短问题（寒暄）不代表检索场景。
把它们算进"无来源率"会得出"20% 的请求检索失败"这种错误结论——
实际是"20% 的请求在按设计追问"。

★ 每个指标都标了"它不能说明什么"
============================
在线评测**没有参考答案**，故所有数字都是代理信号。
不写清楚的话，会被当成"质量得分"使用——那是最危险的误读。
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from src.eval_online.signals import is_evaluable


def _vals(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _rate(rows: list[dict], key: str) -> float | None:
    """布尔字段的比例。**没有任何样本时返回 None**（不是 0）。"""
    vals = [r.get(key) for r in rows if isinstance(r.get(key), bool)]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _positive_rate(rows: list[dict], key: str) -> float | None:
    """**计数型**字段的"大于 0 的比例"。

    ★ 与 `_rate` 的区别
    ----------------
    `_rate` 要求字段是 bool（它是/否）。
    而 `agent_repeated_calls` / `provider_switches` 这类是**计数**：
    值为 0/1/2…，关心的是"发生过吗"而不是"值是多少"。
    故单独一个函数，避免把计数当布尔用（那会让 2 和 1 无法区分）。

    没有任何样本时返回 None（不是 0）—— 与 `_rate` 同一原则。
    """
    vals = [
        r.get(key) for r in rows
        if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
    ]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0) / len(vals)


def _multi_round_rate(rows: list[dict]) -> float | None:
    """需要 >1 轮的比例 —— "打转"的量化指标。

    ★ 为什么单列而不复用 `_positive_rate(agent_rounds - 1)`
    -----------------------------------------------
    `agent_rounds` 的正常值是 1（一轮搞定），不是 0。
    用 `_positive_rate("agent_rounds")` 会把**所有**正常请求都算成"发生过"，
    得到恒为 1.0 的无意义指标。
    """
    vals = [
        r.get("agent_rounds") for r in rows
        if isinstance(r.get("agent_rounds"), (int, float))
        and not isinstance(r.get("agent_rounds"), bool)
    ]
    if not vals:
        return None
    return sum(1 for v in vals if v > 1) / len(vals)


def _no_tool_rate(rows: list[dict]) -> float | None:
    """完全没有调工具的比例。

    这是"答案没有外部依据"的信号（模型只凭参数记忆作答）。
    注意 `agent_tool_calls` 的正常值是 ≥1，不是 0 —— 故同样不能复用 `_positive_rate`。
    """
    vals = [
        r.get("agent_tool_calls") for r in rows
        if isinstance(r.get("agent_tool_calls"), (int, float))
        and not isinstance(r.get("agent_tool_calls"), bool)
    ]
    if not vals:
        return None
    return sum(1 for v in vals if v == 0) / len(vals)


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(_percentile(values, 95), 4),
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
        "n": len(values),
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """最近秩百分位（1-based，不需要 numpy，口径简单可解释）。

    ★ 第一版**算错了**（自查时手算比对发现）
    --------------------------------
    原实现：
        k = int(round((pct / 100) * len(vals) + 0.5)) - 1

    那个 `round(... + 0.5)` 会先四舍五入到整数、再被 `-1` 抵消，净效果是
    **向上取整**，于是取到了比应有位置更高的值：

        n=20,  p95 → round(19.5)=20 → 索引 19 = 最大值（应为第 19 个）
        n=100, p95 → round(95.5)=96 → 索引 95 = 第 96 个（应为第 95 个）

    即 API 返回的 `p95` 是**错的**，且偏差随 n 的小数部分变化，没有规律。

    正确式子是最近秩法：`k = ceil(pct/100 * n)`（1-based）→ 索引 `k-1`。
    小样本时自然退化为最大值（n=2 的 p95 就是较大者），这是预期行为。
    """
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    # 用 ceil 而不是 round：round 在 .5 处向偶数取整，会让 p95 时而偏高一格
    k = max(1, min(n, math.ceil((pct / 100.0) * n)))
    return sorted_vals[k - 1]


def summarize_rows(rows: list[dict]) -> dict[str, Any]:
    """把样本聚合成一份视图。"""
    total = len(rows)
    usable = [r for r in rows if is_evaluable(r)]
    excluded = total - len(usable)

    # 各步耗时：取所有样本里每个步骤的均值（null 不参与）
    step_values: dict[str, list[float]] = {}
    for r in usable:
        steps = r.get("rag_steps")
        if not isinstance(steps, dict):
            continue
        for k, v in steps.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                step_values.setdefault(str(k), []).append(float(v))

    # 用户反馈
    fb = [r.get("feedback") for r in usable]
    fb = [v for v in fb if isinstance(v, int) and not isinstance(v, bool)]
    fb_up = sum(1 for v in fb if v > 0)
    fb_down = sum(1 for v in fb if v < 0)

    return {
        "sample_count": total,
        "evaluable_count": len(usable),
        "excluded_count": excluded,
        "excluded_note": (
            "已排除：首轮追问（无答案）与极短问题（寒暄/误触）。"
            "把它们算进指标会得出错误的结论。"
        ),

        # ---- 检索 ----
        "relevance_top1": _stat(_vals(usable, "relevance_top1")),
        "relevance_mean": _stat(_vals(usable, "relevance_mean")),
        "source_count": _stat(_vals(usable, "source_count")),
        #: 完全没有来源的比例 —— "答案没有依据可查"
        "no_source_rate": _rate(usable, "no_source"),
        #: 本地无匹配而联网的比例 —— 知识库覆盖不足的信号
        "web_fallback_rate": _rate(usable, "web_fallback_used"),
        #: 检索缓存命中率
        "rag_cache_hit_rate": _rate(usable, "rag_cache_hit"),
        "rag_ms": _stat(_vals(usable, "rag_ms")),
        "rag_steps_mean": {
            k: round(statistics.mean(v), 1) for k, v in sorted(step_values.items())
        },
        "rerank_failed_total": sum(
            int(r.get("rerank_failed") or 0) for r in usable
        ),

        # ---- Agent 编排行为 ----
        #
        # ★ 这一组回答的是"Agent 干活干得顺不顺"，不是"答得对不对"
        # ------------------------------------------------
        # 在线没有参考答案，但**编排行为**本身就是质量信号：
        # 轮数多说明在打转，工具失败说明依据变弱，提供商切换说明主模型降级。
        # 这些都能从 done 帧直接推导，零额外开销。
        "agent_tool_calls": _stat(_vals(usable, "agent_tool_calls")),
        "agent_rounds": _stat(_vals(usable, "agent_rounds")),
        #: 需要 >1 轮的比例 —— "打转"的量化指标
        "agent_multi_round_rate": _multi_round_rate(usable),
        "agent_distinct_tools": _stat(_vals(usable, "agent_distinct_tools")),
        #: 完全没有调工具的比例（纯参数记忆作答，没有外部依据）
        "agent_no_tool_rate": _no_tool_rate(usable),
        #: ★ 重复调用同一工具的比例 —— 与项目里 `_spin_hint` 互补
        #:
        #: `_spin_hint` 的阈值是 3（"第 3 次还调同一工具基本确定在打转"），
        #: 且**只数工具名次数、不看参数**。这里把"重复过"的比例也报出来，
        #: 让 2 次的轻度重复也能被看见（它未触发提示，但已在浪费轮次与 token）。
        "agent_repeat_rate": _positive_rate(usable, "agent_repeated_calls"),
        "agent_repeated_calls_total": sum(
            int(r.get("agent_repeated_calls") or 0) for r in usable
        ),
        #: 有工具执行失败的比例（超时/异常）
        "agent_tool_failure_rate": _positive_rate(usable, "agent_tool_failures"),
        "agent_tool_failures_total": sum(
            int(r.get("agent_tool_failures") or 0) for r in usable
        ),
        #: 调了工具但 0 条来源的比例
        "agent_empty_result_rate": _positive_rate(usable, "agent_empty_result_calls"),
        #: 触发了上下文压缩的比例（早期信息已被丢弃）
        "compaction_rate": _rate(usable, "compacted"),
        #: 提供了执行计划的比例
        "has_plan_rate": _rate(usable, "has_plan"),
        #: 记忆命中（有可复用历史问答）的比例
        "memory_hit_rate": _positive_rate(usable, "memory_examples"),

        # ---- 提供商健康 ----
        #
        # ★ 这是"服务是否在降级"的最直接信号
        # ------------------------------
        # 实测见过 `provider_switch_count=1` 且 `provider_errors` 里是
        # deepseek 503 —— 用户拿到的答案来自备用模型，风格与质量可能不同，
        # 而**界面上看不出来**。故必须被统计。
        "provider_switch_rate": _positive_rate(usable, "provider_switches"),
        "provider_switches_total": sum(
            int(r.get("provider_switches") or 0) for r in usable
        ),
        "provider_error_rate": _positive_rate(usable, "provider_errors"),

        # ---- 可信度 ----
        "verify_ok_rate": _rate(usable, "verify_ok"),
        "verify_critical_ok_rate": _rate(usable, "verify_critical_ok"),
        "unsupported_laws_total": sum(
            int(r.get("unsupported_laws") or 0) for r in usable
        ),
        "unsupported_articles_total": sum(
            int(r.get("unsupported_articles") or 0) for r in usable
        ),
        "unsupported_numbers_total": sum(
            int(r.get("unsupported_numbers") or 0) for r in usable
        ),

        # ---- 延迟与成本 ----
        "duration_ms": _stat(_vals(usable, "duration_ms")),
        "llm_ms": _stat(_vals(usable, "llm_ms")),
        "tokens": _stat(_vals(usable, "tokens")),
        "llm_failed_calls_total": sum(
            int(r.get("llm_failed_calls") or 0) for r in usable
        ),
        #: prompt 缓存命中占比（省钱信号）
        "cache_hit_ratio": _cache_ratio(usable),

        # ---- 用户反馈 ----
        "feedback_up": fb_up,
        "feedback_down": fb_down,
        "feedback_count": len(fb),
        "feedback_up_rate": (fb_up / len(fb)) if fb else None,

        # ---- 分组（用于发现"某一类问题一直在失败"）----
        "by_turns": _group_by_turns(usable),

        # ⚠️ 口径提醒：随响应一起下发，避免被当成"质量得分"
        "caveats": [
            "在线评测**没有参考答案**，故以上都是代理信号，不等于答案正确性。",
            "verify_ok_rate 只反映『引用的法条/数字是否出现在检索结果里』，不判断答案对不对。",
            "feedback_up_rate 分母是『评价过的人数』，不是全部用户（多数人不评价）。",
            "relevance_* 依赖检索侧给的评分；未配置评分的来源不计入（故 n 可能小于样本数）。",
        ],
    }


def _cache_ratio(rows: list[dict]) -> dict[str, Any] | None:
    hit = 0.0
    miss = 0.0
    for r in rows:
        h, m = r.get("cache_hit_tokens"), r.get("cache_miss_tokens")
        if isinstance(h, (int, float)) and not isinstance(h, bool):
            hit += float(h)
        if isinstance(m, (int, float)) and not isinstance(m, bool):
            miss += float(m)
    if hit + miss <= 0:
        return None
    return {
        "hit_tokens": int(hit),
        "miss_tokens": int(miss),
        "hit_ratio": round(hit / (hit + miss), 4),
    }


def _group_by_turns(rows: list[dict]) -> dict[str, Any]:
    """按"首轮 / 多轮"分组。

    ★ 为什么要分组：多轮会把上一轮问题拼进检索式（指代消解），
    检索行为与首轮不同。混在一起算均值会掩盖"某一类在退化"。
    """
    groups: dict[str, list[dict]] = {"first_turn": [], "multi_turn": []}
    for r in rows:
        groups["multi_turn" if r.get("has_history") else "first_turn"].append(r)
    out: dict[str, Any] = {}
    for name, g in groups.items():
        if not g:
            continue
        out[name] = {
            "n": len(g),
            "no_source_rate": _rate(g, "no_source"),
            "relevance_top1_mean": (
                round(statistics.mean(_vals(g, "relevance_top1")), 4)
                if _vals(g, "relevance_top1") else None
            ),
            "duration_ms_mean": (
                round(statistics.mean(_vals(g, "duration_ms")), 1)
                if _vals(g, "duration_ms") else None
            ),
            "verify_ok_rate": _rate(g, "verify_ok"),
        }
    return out
