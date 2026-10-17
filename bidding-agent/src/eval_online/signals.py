"""RAG 在线评测：从**真实请求**里采集质量信号。

★ 与离线评测（`eval/`）的区别，以及为什么两者都需要
===========================================
    离线 `eval/ragas_eval.py`   固定测试集 + LLM 裁判
                                能算 Faithfulness / AnswerRelevancy / ContextRecall
                                **有参考答案**，故能测"答得对不对"
                                但：测试集是有限的，覆盖不到真实长尾

    在线 本模块                  真实用户请求 + 已有信号
                                **没有参考答案**，故测不了"答得对不对"
                                但：覆盖真实分布，能发现"线上某一类问题一直在失败"

★ 在线评测的根本约束：**不能拖慢用户请求**
====================================
这是在线评测最容易做错的地方——每个请求都跑一次 LLM 裁判，成本与延迟都不可接受。
故本模块分两层，且**只复用已经算出来的信号**：

    `evaluate()`          纯函数、零额外开销 —— 每请求都跑
                          输入是 `done` 帧里**本来就有**的字段
                          （verification / rag_timing / sources / llm_trace）
    `maybe_sample_judge()` 可选、按比例采样、**后台线程异步** —— 默认关闭

★ 在线评测**测不了**什么（必须说清，否则会被误读）
========================================
没有参考答案，故这些**只能反映代理信号，不等于质量**：

    "检索相关度均值 0.83"      ≠ "答案正确"
    "核对通过率 0.95"          ≠ "没有幻觉"（核对只查引用是否有依据）
    "点踩率 0.05"              ≠ "95% 满意"（大部分人不会点）

故本模块的定位是**发现异常与趋势**，不是判定绝对质量。
`metric.py` 里的每个指标都写了它**不能**说明什么。
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def _num(value: Any) -> float | None:
    """取数值；取不到返回 None（**不补 0**）。

    与项目其他模块同一原则：`0` 会被读成"确实是 0"，
    而事实可能是"没拿到这个数据"。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _top_relevance(sources: list) -> float | None:
    """来源里的最高相关度。

    ★ 为什么取**最高**而不是均值
    ----------------------
    "检索到了相关内容吗"是 top-1 就能回答的问题。均值会被 tail 拉平：
    5 条里 1 条相关（0.9）与全都中等相关（0.5）均值可能接近，
    但前者的答案质量好得多。故同时给 top1 与 mean，让使用者自己判断。
    """
    vals = [
        _num((s or {}).get("relevance"))
        for s in (sources or [])
        if isinstance(s, dict)
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _mean_relevance(sources: list) -> float | None:
    vals = [
        _num((s or {}).get("relevance"))
        for s in (sources or [])
        if isinstance(s, dict)
    ]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def evaluate(meta: dict, done: dict, *, question: str = "", has_history: bool = False,
             feedback: int | None = None) -> dict:
    """从一次真实请求里提取评测指标。**纯函数、永不抛异常**。

    参数都是**已经算好的**东西，故本函数零额外开销（无 IO、无模型调用）。
    `feedback` 是用户反馈（1 赞 / -1 踩 / None 未评），离线补录时传入。

    返回一行记录（可直接 append 进 jsonl）。
    """
    try:
        sources = done.get("sources") or []
        web_sources = done.get("web_sources") or []
        verification = done.get("verification") or {}
        rag = done.get("rag_timing") or {}
        llm = done.get("llm_trace") or {}

        # 各步耗时（检索内部构成）——用于"哪一步在退化"
        rag_steps = {
            str(s.get("step")): _num(s.get("sum"))
            for s in (rag.get("steps") or [])
            if isinstance(s, dict) and s.get("step")
        }

        duration = _num(done.get("elapsed_ms"))
        llm_ms = _num(llm.get("total_ms"))
        rag_ms = _num(rag.get("total_ms"))

        # ---- Agent 编排行为 ----
        #
        # ★ 为什么测"行为"而不是"答得对不对"
        # ------------------------------
        # 在线没有参考答案，故无法判断答案正确性。
        # 但**编排行为**本身就是质量信号，且完全可从 done 帧推导：
        #   · 轮数多        → 模型在打转（项目文档记过 6 次同一工具的实测）
        #   · 工具失败      → 数据源不可用，答案依据变弱
        #   · 提供商切换    → 主模型降级，风格与质量可能变化
        #   · 上下文压缩    → 对话太长，早期信息已丢失
        tool_timings = [
            t for t in (done.get("tool_timings") or []) if isinstance(t, dict)
        ]
        tool_names = [str(x) for x in (done.get("tool_names") or []) if x]
        #: 轮次取各次调用的 `round` 最大值（不是调用次数：同轮可并行多工具）
        rounds = [int(_num(t.get("round")) or 0) for t in tool_timings]
        #: 同名工具的重复调用次数（>1 说明模型对同一数据源重复试探）
        repeats = {}
        for n in tool_names:
            repeats[n] = repeats.get(n, 0) + 1
        providers = done.get("provider_chain") or []
        compaction = done.get("compaction") or {}

        return {
            # ---- 检索质量 ----
            #: 本地来源条数（0 通常是坏信号：说明知识库没命中）
            "source_count": len(sources),
            "web_source_count": len(web_sources),
            #: 最高/平均相关度（None = 没有可用评分，**不是 0**）
            "relevance_top1": _top_relevance(sources),
            "relevance_mean": _mean_relevance(sources),
            #: 检索缓存是否命中：命中与未命中差两个数量级，分开统计才有意义
            "rag_cache_hit": bool(rag.get("all_cached")),
            "rag_ms": rag_ms,
            "rag_steps": rag_steps,
            #: 精排失败次数（增益项失效，质量会略降）
            "rerank_failed": int(_num(rag.get("rerank_failed")) or 0),

            # ---- Agent 编排行为 ----
            #: 工具调用次数（0 = 没调工具，纯参数记忆作答）
            "agent_tool_calls": len(tool_timings),
            #: 最大轮次。轮数多 = 在打转，是最直观的编排质量信号
            "agent_rounds": max(rounds) if rounds else 0,
            #: 执行失败的工具调用次数（超时/异常，答案依据变弱）
            "agent_tool_failures": sum(
                1 for t in tool_timings if t.get("ok") is False
            ),
            #: 工具返回 0 条来源的次数（调了但没拿到东西）
            "agent_empty_result_calls": sum(
                1
                for t in tool_timings
                if t.get("ok") is True and int(_num(t.get("sources")) or 0) == 0
            ),
            #: 同一工具被调用 >1 次的次数之和（重复试探）
            "agent_repeated_calls": sum(v - 1 for v in repeats.values() if v > 1),
            #: 用到的**不同**工具数（多工具协作 vs 反复同一个）
            "agent_distinct_tools": len(repeats),
            #: 是否走首轮追问（此时没有答案，是**按设计**的行为而非失败）
            "agent_clarified": bool(done.get("clarified")),
            #: 提供商切换次数：>0 说明主模型降级过
            "provider_switches": int(_num(done.get("provider_switch_count")) or 0),
            #: 提供商错误条数（含 5xx/超时等）
            "provider_errors": len(done.get("provider_errors") or []),
            #: 用的提供商数量（>1 说明确实切过）
            "provider_count": len(providers) if isinstance(providers, list) else 0,
            #: 是否触发了上下文压缩（触发了说明早期信息已被丢弃）
            "compacted": bool(compaction),
            #: 是否有可复用的历史问答（记忆命中）
            "memory_examples": len(done.get("memory_examples") or []),
            #: 是否给出了执行计划（计划模式是否生效）
            "has_plan": bool(done.get("plan")),

            # ---- 答案可信度（已有核对的结果，非新计算）----
            #: 引用核对是否通过。**注意**：它只查"引用的法条/数字是否在检索结果里"，
            #: 不判断答案本身对不对。
            "verify_ok": bool(verification.get("ok")),
            "verify_critical_ok": bool(verification.get("critical_ok")),
            #: 未被检索结果支持的法条数（>0 说明可能编了法条）
            "unsupported_laws": len(verification.get("unsupported_laws") or []),
            "unsupported_articles": len(verification.get("unsupported_articles") or []),
            "unsupported_numbers": len(verification.get("unsupported_numbers") or []),

            # ---- 失败模式 ----
            #: 本地无匹配而联网（说明知识库覆盖不足）
            "web_fallback_used": bool(
                (done.get("web_fallback") or {}).get("enough") is False
            ),
            #: 完全没有来源：答案没有依据可查
            "no_source": len(sources) == 0 and len(web_sources) == 0,
            #: 是否实际调了工具
            "tool_called": bool(done.get("tool_called")),

            # ---- 延迟与成本 ----
            "duration_ms": duration,
            "llm_ms": llm_ms,
            "llm_calls": int(_num(llm.get("call_count")) or 0),
            "llm_failed_calls": int(_num(llm.get("failed_count")) or 0),
            "tokens": _num(llm.get("total_tokens")),
            #: prompt 缓存命中率（省钱信号）
            "cache_hit_tokens": _num(llm.get("cache_hit_tokens")),
            "cache_miss_tokens": _num(llm.get("cache_miss_tokens")),

            # ---- 上下文（用于分组分析）----
            #: 问题长度：极短问题（寒暄/误触）与长问题的指标不可比
            "question_chars": len(str(question or "")),
            #: 是否多轮（多轮与首轮的检索行为不同）
            "has_history": bool(has_history),
            #: 是否走首轮追问（此时没有答案，指标应排除）
            "clarified": bool(done.get("clarified")),
            #: 用户反馈（离线补录用；None = 未评）
            "feedback": feedback,
        }
    except Exception:  # noqa: BLE001 - 评测是旁路，绝不能影响请求
        logger.debug("在线评测采集失败（不影响请求）", exc_info=True)
        return {}


#: 一次回答"没有可评内容"的判据 —— 这类记录应被排除在指标之外。
#:
#: ★ 为什么必须排除（否则指标会被系统性拉低）
#: ------------------------------------
#: 首轮追问（`clarified`）**没有答案**，故没有来源、没有核对；
#: 若把它算进"无来源率"，会得出"20% 的请求检索失败"这种错误结论——
#: 实际是"20% 的请求在按设计追问"。
def is_evaluable(row: dict) -> bool:
    """该条记录是否可以计入质量指标。"""
    if not row:
        return False
    if row.get("clarified"):
        return False
    # 极短问题（寒暄/误触）不代表真实检索场景
    if int(_num(row.get("question_chars")) or 0) < 4:
        return False
    return True
