"""把一轮里的多次 RAG 检索耗时汇总成 done 帧用的结构。

★ 为什么要汇总而不是直接下发数组
==========================
一轮问答可能检索多次（ReAct 多轮、补充检索、降级路径），
前端的耗时面板需要的是"RAG 一共花了多久、每一步占多少"，
而不是让前端自己遍历数组做聚合（两端口径容易不一致——
项目里 `phase_total_ms` 就是为这个理由由后端算好的）。

★ 口径说明（必须显式，否则会被误读）
============================
    total_ms   **各次检索耗时之和**，不是墙钟。缓存命中记 0，
               故它与"墙钟上的检索时段"不等（并发检索会重叠）。
    max_ms     单次最长，用于识别"某一次特别慢"
    steps      按步骤聚合：sum 是总耗时、max 是单次最长
    cache_hits 命中次数 —— 命中与未命中差两个数量级，不标注会被当成抖动
"""

from __future__ import annotations

from typing import Any


def _int(value: Any, default: int = 0) -> int:
    """尽量转成 int；转不了返回默认值。

    ★ 为什么必须容错（实测踩到）
    ----------------------
    第一版直接写 `int(item.get("total_ms") or 0)`，而 `total_ms` 若是
    `"abc"` 这种脏值就抛 `ValueError` —— 直接违反本模块的契约
    （观测代码永不抛异常）。`done` 帧组装处一旦抛异常，
    **整个回答会被吞掉**，而这个字段本该只是附带信息。

    数据来源是内部产出的 dict，理论上形状可控；但"理论上"不足以作为
    一个会影响用户拿到回答的代码路径的依据。
    """
    if isinstance(value, bool):  # bool 是 int 子类，但语义上不是耗时
        return default
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return default
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, AttributeError):
            return default
    return default


def summarize(rag_timings: list[dict]) -> dict[str, Any] | None:
    """汇总多次检索的耗时。没有数据返回 None（前端据此不显示）。

    输入是 `RagTiming.marks()` 产出的列表（见 `src/rag/timing.py`）。
    **永不抛异常**：任何形状异常都按"该项无效"跳过。
    """
    if not rag_timings:
        return None

    total = 0
    max_ms = 0
    cache_hits = 0
    reranked = 0
    rerank_failed = 0
    result_count = 0
    categories: list[str] = []
    steps: dict[str, dict[str, Any]] = {}

    for item in rag_timings:
        if not isinstance(item, dict):
            continue
        ms = _int(item.get("total_ms"))
        total += ms
        max_ms = max(max_ms, ms)
        if item.get("cache_hit"):
            cache_hits += 1
        if item.get("reranked"):
            reranked += 1
        if item.get("rerank_failed"):
            rerank_failed += 1
        # 只累计**未命中**那次的结果数：命中时 result_count 是缓存大小，
        # 与"这次检索召回了几条"不是一回事，混在一起会让数字看起来随机
        if not item.get("cache_hit"):
            result_count = max(result_count, _int(item.get("result_count")))
        cat = str(item.get("question_category") or "").strip()
        if cat and cat not in categories:
            categories.append(cat)
        for s in item.get("steps") or []:
            if not isinstance(s, dict):
                continue
            key = str(s.get("step") or "")
            if not key:
                continue
            slot = steps.setdefault(
                key, {"step": key, "label": s.get("label") or key, "sum": 0, "max": 0}
            )
            v = _int(s.get("ms"))
            slot["sum"] += v
            slot["max"] = max(slot["max"], v)

    # 按总耗时降序：前端直接照顺序渲染即可，一眼看到大头
    ordered = sorted(steps.values(), key=lambda x: -x["sum"])
    step_sum = sum(s["sum"] for s in ordered)

    return {
        "count": len([x for x in rag_timings if isinstance(x, dict)]),
        "total_ms": total,
        "max_ms": max_ms,
        "steps": ordered,
        # 步骤合计与 total 的差额 = 未打点的开销。**显式给出**而不是让前端减。
        #
        # ★ 可能为**负**，且负值是有意义的信号（自查发现前端会把它藏起来）
        # ------------------------------------------------------
        # 为什么会有负值：	otal_ms 与各步骤的计时窗口**不完全嵌套** ——
        # 步骤计时起点在 search() 内部，而 	otal 的起点更早/更晚，
        # 加上毫秒级取整，步骤合计可能比 total 大几毫秒。
        #
        # 负值意味着步骤合计超过总耗时，通常说明**计时窗口有重叠**
        # （例如同一段耗时被两个步骤各记了一次）—— 这是**该被看到**的异常，
        # 而不是该被前端用 > 0 条件静默隐藏的东西（第一版就是那样）。
        # 故这里原样给出（可负），由前端显示并允许为负。
        "unaccounted_ms": total - step_sum,
        "cache_hits": cache_hits,
        "all_cached": cache_hits == len(rag_timings) and bool(rag_timings),
        "reranked": reranked,
        "rerank_failed": rerank_failed,
        "result_count": result_count,
        "categories": categories,
    }
