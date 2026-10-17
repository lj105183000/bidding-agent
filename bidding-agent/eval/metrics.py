"""Agent 过程指标：工具调用行为与答案可信度。

★ 为什么在"准确率 + 延迟"之外还需要这些
------------------------------------
现有 `agent_eval.py` 只测两件事：**最后一个工具选对没有**、**花了多久**。
但实测暴露了几类它完全看不见的问题：

    1. **打转**：复合问题调 5~6 次工具、`query_database` 被重复调 4 次、
       2/5 撞轮次上限。工具选对了（tool_name 命中），所以准确率 100%，
       但用户等了 30 秒。
    2. **编造引用**：模型在答案里补检索结果里没有的条款号
       （实测抓到过 `〔2023〕95号`、"第三十六条"）。工具选对、延迟正常，
       但**内容是错的**——这在招投标场景最致命。
    3. **编造工具不可用**：回答写「联网检索工具本轮不可用」而它成功调了 4 次。

这三类都是"指标全绿但产品有问题"。故本模块从 `chat()` 的返回里提取
**过程指标**，与准确率、延迟并列。

★ 口径依赖
--------
依赖 `done` 帧的 `verification` / `compaction` / `memory_examples` 字段
（见 docs/27、29、31）。缺字段时对应指标记 `None` 而不是 0——
**"没测"与"测了得 0 分"是两件事**，混在一起会让指标骗人。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: 轮次上限。**单一来源**：`src.agent.constants.MAX_TOOL_ROUNDS`。
#:
#: ★ 为什么曾是一个本地常量，以及那个写法为什么是错的
#: ------------------------------------------
#: 原来这里写 `TOOL_ROUNDS_CAP = 4` 并在注释里说"与 MAX_TOOL_ROUNDS 对齐"——
#: 但**对齐只是注释，没有约束**：真正的缺陷不在值，在**用法的语义**：
#:
#:     m.hit_rounds_cap = m.tool_calls >= TOOL_ROUNDS_CAP      # ← 拿调用次数比轮次上限
#:
#: 指标名是 `hit_rounds_cap`（轮次），判的却是"工具调用**次数** ≥ 4"。
#: 而一轮里可以并行调多个工具（实测：模型在同一轮发两个不同 query 的
#: `search_knowledge_base`，做两路检索），于是**1 轮被记成 2 次调用**。
#:
#: 后果是**假性未达标**：`agent.hit_rounds_cap_rate` 在历史趋势里
#: 从 0 一路涨到 0.2353 并让门禁判 fail，而真实轮次上限从未被触达
#: （实测 17 题 `max(rounds)` 全部 ≤ 2）。
#:
#: 故改用 `MAX_TOOL_ROUNDS` 本身，并让判定基于**轮次**而非调用次数。
from src.agent.constants import MAX_TOOL_ROUNDS as TOOL_ROUNDS_CAP


@dataclass
class ProcessMetrics:
    """单题的过程指标。缺失的用 None，不要用 0 冒充。"""

    #: 工具调用总次数（含重复调用同一个工具）
    tool_calls: int = 0
    #: ★ Agent 实际用掉的**轮次**（`tool_timings[].round` 的最大值）。
    #:
    #: ★ 为什么必须与 `tool_calls` 分开
    #: ---------------------------
    #: 一轮可以并行调多个工具（实测：模型在同一轮发两个**不同 query** 的
    #: `search_knowledge_base`，做两路检索）。故 `tool_calls >= 4` 并**不**意味着
    #: "用完了 4 轮预算" —— 它可能只是一轮里调了 4 次。
    #: 拿调用次数跟轮次上限比会得出**假性打转**（见 `TOOL_ROUNDS_CAP` 的注释）。
    rounds: int = 0
    #: 去重后的工具种类数
    distinct_tools: int = 0
    #: 重复调用：同一工具被调多次的次数（`tool_calls - distinct_tools`）
    #:
    #: ⚠️ 这个口径**只看工具名，不看参数**，故"同一轮两路不同 query 的检索"
    #: 也会被计入。它不是"无用的重复" —— 实测过重复调用的参数**都不同**
    #: （模型在换问法补充检索），故**不应据此去重**。要看"是否真打转"请用 `rounds`。
    redundant_calls: int = 0
    #: 是否撞到轮次上限（打转信号）。**按 `rounds` 判定**，不按 `tool_calls`。
    hit_rounds_cap: bool = False
    #: 答案核对是否通过；None = 没跑核对（如无工具调用）
    verification_ok: bool | None = None
    #: 是否**没有严重错误**（数字/条款号编造）。None = 没跑核对。
    #:
    #: ★ 与 `verification_ok` 分开统计的理由见 `verification.CheckResult.critical_ok`：
    #: 法规名未在结果中是低风险，"编了个 10%"是高风险，混在一起看不出真实状况。
    verification_critical_ok: bool | None = None
    #: 未经工具结果支持的法规名
    unsupported_laws: list[str] = field(default_factory=list)
    #: 未经工具结果支持的条款号 / 文号
    unsupported_articles: list[str] = field(default_factory=list)
    #: 未经工具结果支持的数字（百分比 / 金额）
    #:
    #: ★ 曾经漏了这个字段：报告里"答案核对通过率 0.6667 未达标"，
    #: 但备注只写"未支持法规 0 题、未支持条款 0 题"——**恰恰漏了真正的原因**
    #: （实测原因是 `数字未经检索结果支持：10%`）。
    #: 报告说"未达标"却不说为什么，是最难查的一类问题。
    unsupported_numbers: list[str] = field(default_factory=list)
    #: 声称某工具不可用但实际成功执行
    false_unavailable_claims: list[str] = field(default_factory=list)
    #: 是否压缩了历史；None = 没跑
    compacted: bool | None = None
    #: 召回的历史示例条数
    memory_examples: int = 0
    #: 答案长度（字符）
    answer_chars: int = 0
    #: 引用来源条数（本地 + 联网）
    source_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_process_metrics(result: dict) -> ProcessMetrics:
    """从 `chat()` 的返回里提取过程指标。**永不抛异常**。

    缺字段一律记 None / 0，并把"字段缺失"与"值为 0"区分开：
    例如 `verification_ok=None`（没跑核对）与 `False`（跑了且没过）含义完全不同。
    """
    m = ProcessMetrics()
    if not isinstance(result, dict):
        return m

    # --- 工具调用 ---
    names = result.get("tool_names")
    if isinstance(names, list) and names:
        m.tool_calls = len(names)
        m.distinct_tools = len(set(names))
        m.redundant_calls = max(0, m.tool_calls - m.distinct_tools)
    elif result.get("tool_called"):
        # 老路径只给 tool_name 单字符串，退化为"至少调了 1 次"
        m.tool_calls = 1
        m.distinct_tools = 1

    # --- 轮次（★ 与调用次数分开算）---
    #
    # `tool_timings[].round` 是 Agent 侧记录的真实轮次（一轮内并行调多个工具时，
    # 它们的 `round` 相同）。取最大值即"用掉了几轮"。
    #
    # 为什么不用 `tool_calls` 近似：实测模型会在**同一轮**里发两个不同 query 的
    # 检索（做两路召回），此时 `tool_calls=2` 而 `rounds=1`。
    # 用调用次数判"撞轮次上限"会得出假性打转（见 `TOOL_ROUNDS_CAP` 的注释）。
    timings = result.get("tool_timings")
    if isinstance(timings, list):
        rs = [
            t.get("round")
            for t in timings
            if isinstance(t, dict) and isinstance(t.get("round"), int)
        ]
        if rs:
            m.rounds = max(rs)
    # ★ 按**轮次**判定，不按调用次数
    m.hit_rounds_cap = m.rounds >= TOOL_ROUNDS_CAP

    # --- 答案核对 ---
    v = result.get("verification")
    if isinstance(v, dict):
        m.verification_ok = bool(v.get("ok"))
        # 老结果可能没有 critical_ok（核对器升级前跑的）：此时按 ok 兜底，
        # 而不是留 None——留 None 会被当成"没测"，让历史对比突然少一项。
        ck = v.get("critical_ok")
        m.verification_critical_ok = bool(ck) if ck is not None else m.verification_ok
        for key, attr in (
            ("unsupported_laws", "unsupported_laws"),
            ("unsupported_articles", "unsupported_articles"),
            ("unsupported_numbers", "unsupported_numbers"),
            ("false_unavailable_claims", "false_unavailable_claims"),
        ):
            val = v.get(key)
            if isinstance(val, list):
                setattr(m, attr, [str(x) for x in val])

    # --- 历史压缩 ---
    c = result.get("compaction")
    if isinstance(c, dict):
        m.compacted = bool(c.get("compacted"))

    # --- 会话记忆 ---
    ex = result.get("memory_examples")
    if isinstance(ex, list):
        m.memory_examples = len(ex)

    # --- 规模 ---
    answer = result.get("answer")
    m.answer_chars = len(str(answer)) if answer else 0
    for key in ("sources", "web_sources"):
        val = result.get(key)
        if isinstance(val, list):
            m.source_count += len(val)

    return m


def aggregate(metrics: list[ProcessMetrics]) -> dict:
    """把多题的过程指标聚合成一行汇总。

    求均值时**跳过 None**——把"没测"当 0 会把均值往好看的方向拉
    （例如只有一半题跑了核对，全算 0 会让"核对通过率"虚低或虚高）。
    """
    if not metrics:
        return {
            "cases": 0, "tool_calls_mean": 0.0, "redundant_calls_total": 0,
            "hit_rounds_cap": 0, "rounds_mean": 0.0, "rounds_max": 0,
            "verification_ok_rate": None,
            "verification_critical_ok_rate": None,
            "unsupported_law_cases": 0, "false_unavailable_cases": 0,
        }

    total_calls = sum(m.tool_calls for m in metrics)
    all_rounds = [m.rounds for m in metrics]
    verified = [m.verification_ok for m in metrics if m.verification_ok is not None]
    critical = [m.verification_critical_ok for m in metrics
                if m.verification_critical_ok is not None]
    return {
        "cases": len(metrics),
        "tool_calls_mean": round(total_calls / len(metrics), 2),
        # ★ 轮次（与调用次数分开报）。`rounds_max` 用于回答"离上限还有多远"——
        # 只看均值会掩盖"个别题撞满"的长尾。
        "rounds_mean": round(sum(all_rounds) / len(all_rounds), 2),
        "rounds_max": max(all_rounds) if all_rounds else 0,
        # 重复调用的**总量**比均值更有意义：它直接对应"白花的 LLM 轮次"
        "redundant_calls_total": sum(m.redundant_calls for m in metrics),
        "hit_rounds_cap": sum(1 for m in metrics if m.hit_rounds_cap),
        # None 表示一题都没跑核对——调用方必须据此显示"未测"而不是 0%
        "verification_ok_rate": (
            round(sum(1 for v in verified if v) / len(verified), 4) if verified else None
        ),
        # 严重错误通过率（数字/条款号编造）。**这才是真正的风险指标**——
        # 法规名未在结果中属低风险，混算会掩盖数字编造。
        "verification_critical_ok_rate": (
            round(sum(1 for v in critical if v) / len(critical), 4) if critical else None
        ),
        "unsupported_law_cases": sum(1 for m in metrics if m.unsupported_laws),
        "unsupported_article_cases": sum(1 for m in metrics if m.unsupported_articles),
        "unsupported_number_cases": sum(1 for m in metrics if m.unsupported_numbers),
        "false_unavailable_cases": sum(1 for m in metrics if m.false_unavailable_claims),
        "memory_hit_cases": sum(1 for m in metrics if m.memory_examples),
        "compacted_cases": sum(1 for m in metrics if m.compacted),
        "answer_chars_mean": round(
            sum(m.answer_chars for m in metrics) / len(metrics), 1
        ),
        "source_count_mean": round(
            sum(m.source_count for m in metrics) / len(metrics), 2
        ),
    }


__all__ = [
    "TOOL_ROUNDS_CAP",
    "ProcessMetrics",
    "aggregate",
    "extract_process_metrics",
]
