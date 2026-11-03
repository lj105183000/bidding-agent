"""多 specialist 结果的确定性合并。

并发完成顺序不可依赖：所有列表按 `agent_index` 排序后再拼接。
主 meta 只在主线程写一次，specialist 各自用 sub-meta。
"""

from __future__ import annotations

from src.agent.constants import MAX_CHARTS, MAX_LOCAL_SOURCES, MAX_WEB_SOURCES


def _dedup_sources(items: list[dict], cap: int) -> list[dict]:
    seen = set()
    merged: list[dict] = []
    for source in items or []:
        if not isinstance(source, dict):
            continue
        key = (source.get("question"), source.get("answer"), source.get("url"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    if len(merged) <= cap:
        return merged

    def score_of(item: dict) -> float:
        value = item.get("score")
        return float(value) if isinstance(value, (int, float)) else -1.0

    merged.sort(key=score_of, reverse=True)
    return merged[:cap]


def merge_results(results: list[dict]) -> dict:
    """按 index + 决策顺序合并；返回可直接写进主 meta 的字段。"""
    ordered = sorted(
        [r for r in (results or []) if isinstance(r, dict)],
        key=lambda r: int(r.get("index") or 0),
    )

    sources: list[dict] = []
    web_sources: list[dict] = []
    tool_names: list[str] = []
    tool_timings: list[dict] = []
    charts: list[dict] = []
    rag_timings: list[dict] = []
    succeeded: list[str] = []
    output_text: list[str] = []
    agent_trace: list[dict] = []
    agent_timings: list[dict] = []
    tool_deduped = 0
    rounds = 0
    partial = False

    for result in ordered:
        index = int(result.get("index") or 0)
        agent = str(result.get("agent") or "")
        sources.extend(result.get("sources") or [])
        web_sources.extend(result.get("web_sources") or [])
        tool_names.extend(result.get("tool_names") or [])
        for timing in result.get("tool_timings") or []:
            if isinstance(timing, dict):
                item = dict(timing)
                item.setdefault("agent", agent)
                item.setdefault("agent_index", index)
                tool_timings.append(item)
        charts.extend(result.get("charts") or [])
        rag_timings.extend(result.get("rag_timings") or [])
        for name in result.get("succeeded_tools") or []:
            if name not in succeeded:
                succeeded.append(name)
        output_text.extend(result.get("tool_output_text") or [])
        tool_deduped += int(result.get("tool_deduped") or 0)
        rounds = max(rounds, int(result.get("rounds") or 0))
        if result.get("status") not in ("ok",):
            partial = True
        agent_trace.append({
            "agent": agent,
            "index": index,
            "status": str(result.get("status") or ""),
            "elapsed_ms": int(result.get("elapsed_ms") or 0),
            "rounds": int(result.get("rounds") or 0),
            "tool_name": str(result.get("tool_name") or ""),
            "tool_calls": len(result.get("tool_names") or []),
            "sources": len(result.get("sources") or []),
            "web_sources": len(result.get("web_sources") or []),
            "provider": str(result.get("provider") or ""),
            "provider_switches": int(result.get("provider_switches") or 0),
            "error": str(result.get("error") or "")[:200],
        })
        agent_timings.append({
            "agent": agent,
            "index": index,
            "phase_times": list(result.get("phase_times") or []),
        })

    tool_name = ""
    for result in ordered:
        if result.get("tool_name"):
            tool_name = str(result["tool_name"])
            break

    return {
        "sources": _dedup_sources(sources, MAX_LOCAL_SOURCES),
        "web_sources": _dedup_sources(web_sources, MAX_WEB_SOURCES),
        "tool_called": bool(tool_names),
        "tool_name": tool_name,
        "tool_names": tool_names,
        "tool_timings": tool_timings,
        "tool_deduped": tool_deduped,
        "charts": charts[:MAX_CHARTS],
        "rag_timings": rag_timings,
        "succeeded_tools": succeeded,
        "tool_output_text": output_text,
        "rounds": rounds,
        "partial": partial,
        "agent_trace": agent_trace,
        "agent_timings": agent_timings,
    }
