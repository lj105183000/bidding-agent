"""多 Agent 灰度指标（src/monitoring_multi_agent.py）的纯单测。"""

from __future__ import annotations

import pytest

from src.monitoring_multi_agent import (
    MAX_SAMPLES_PER_GROUP,
    multi_agent_snapshot,
    record_done,
    reset_multi_agent_metrics,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_multi_agent_metrics()
    yield
    reset_multi_agent_metrics()


def _done(
    *,
    mode: str = "auto",
    decision: str = "activated",
    activated: bool = True,
    fallback: str = "none",
    partial: bool = False,
    elapsed: float | None = 1000,
    calls: float | None = 3,
    tokens: float | None = 1000,
    verification: dict | None = None,
    sources: list | None = None,
    configured: str | None = None,
    reason: str | None = None,
    router: str = "rules",
    deadline: bool = False,
    budget: bool = False,
) -> dict:
    return {
        "multi_agent": {
            "mode": mode,
            "configured_mode": configured or mode,
            "mode_reason": reason or ("active" if mode != "off" else "off"),
            "activated": activated,
            "decision": decision,
            "route": ["data", "retrieval"] if activated else [],
            "router": router,
            "fallback": fallback,
            "partial": partial,
            "deadline_exceeded": deadline,
            "budget_exhausted": budget,
        },
        "elapsed_ms": elapsed,
        "llm_trace": (
            {"call_count": calls, "total_tokens": tokens}
            if calls is not None or tokens is not None
            else None
        ),
        "verification": verification,
        "sources": sources if sources is not None else [{"id": 1}],
        "web_sources": [],
    }


def test_off_done_recorded_as_off():
    record_done(_done(mode="off", decision="off", activated=False))
    snap = multi_agent_snapshot()
    assert snap["total_requests"] == 1
    assert snap["summary"]["off"] == 1
    assert snap["by_mode"]["off"]["declined"] == 0
    assert snap["by_mode"]["off"]["activated_rate"] == 0.0


def test_auto_activated_and_declined_split():
    record_done(_done())
    record_done(_done(decision="declined_single_expert", activated=False, elapsed=500))
    snap = multi_agent_snapshot()
    assert snap["summary"]["auto_activated"] == 1
    assert snap["summary"]["auto_declined"] == 1
    assert snap["by_mode"]["auto"]["activated_rate"] == 0.5
    assert snap["by_mode"]["auto"]["decision_counts"]["activated"] == 1
    assert snap["by_mode"]["auto"]["decision_counts"]["declined_single_expert"] == 1


def test_missing_multi_agent_is_unknown_not_off():
    snap = multi_agent_snapshot()
    assert snap["total_requests"] == 0
    record_done({"elapsed_ms": 100})
    snap = multi_agent_snapshot()
    assert snap["summary"]["unknown"] == 1
    assert snap["summary"]["off"] == 0
    assert snap["by_mode"]["unknown"]["n"] == 1


def test_fallback_counts_only_activated():
    record_done(_done(fallback="legacy"))
    record_done(_done(decision="declined_single_expert", activated=False))
    snap = multi_agent_snapshot()
    assert snap["summary"]["fallback_legacy"] == 1
    assert snap["summary"]["fallback_none"] == 0
    assert snap["by_mode"]["auto"]["fallback_counts"] == {"legacy": 1}


def test_partial_rate():
    record_done(_done(partial=True))
    record_done(_done(partial=False))
    snap = multi_agent_snapshot()
    assert snap["summary"]["partial"] == 1
    assert snap["by_mode"]["auto"]["partial_rate"] == 0.5


def test_latency_llm_and_token_stats():
    record_done(_done(elapsed=1000, calls=3, tokens=1000))
    record_done(_done(elapsed=3000, calls=5, tokens=2000))
    snap = multi_agent_snapshot()
    mode = snap["by_mode"]["auto"]
    assert mode["latency_ms"]["mean"] == 2000.0
    assert mode["latency_ms"]["samples"] == 2
    assert mode["llm_calls"]["mean"] == 4.0
    assert mode["tokens"]["mean"] == 1500.0
    assert mode["tokens"]["total"] == 3000.0


def test_missing_tokens_not_zero():
    record_done(_done(tokens=None))
    snap = multi_agent_snapshot()
    assert snap["by_mode"]["auto"]["tokens"] is None


def test_verification_rates_use_checked_denominator():
    record_done(_done(verification={"ok": True, "critical_ok": True}))
    record_done(_done(verification={"ok": False, "critical_ok": True}))
    record_done(_done(verification=None))  # 没跑核对，不进分母
    snap = multi_agent_snapshot()
    ver = snap["by_mode"]["auto"]["verification"]
    assert ver["checked"] == 2
    assert ver["ok_rate"] == 0.5
    assert ver["critical_checked"] == 2
    assert ver["critical_ok_rate"] == 1.0


def test_no_source_excludes_clarified():
    record_done(_done(sources=[]), clarified=True)
    record_done(_done(sources=[{"id": 1}]))
    record_done(_done(sources=[]))
    snap = multi_agent_snapshot()
    sources = snap["by_mode"]["auto"]["sources"]
    assert sources["answered"] == 2
    assert sources["no_source"] == 1
    assert sources["no_source_rate"] == 0.5


def test_configured_mode_and_reason_counts():
    record_done(_done(mode="off", decision="off", activated=False, configured="auto", reason="langgraph_unavailable"))
    snap = multi_agent_snapshot()
    assert snap["configured_mode_counts"]["auto"] == 1
    assert snap["mode_reason_counts"]["langgraph_unavailable"] == 1
    assert snap["configured_reason_counts"] == [
        {"mode": "auto", "reason": "langgraph_unavailable", "requests": 1}
    ]


def test_router_deadline_budget_counts():
    record_done(_done(router="rules"))
    record_done(_done(router="llm", deadline=True))
    record_done(_done(router="llm_fallback", budget=True))
    snap = multi_agent_snapshot()
    summary = snap["summary"]
    assert summary["router_counts"] == {
        "rules": 1,
        "llm": 1,
        "llm_fallback": 1,
    }
    assert summary["routing_llm_rate"] == 0.6667
    assert summary["deadline_count"] == 1
    assert summary["deadline_rate"] == 0.3333
    assert summary["budget_count"] == 1
    assert summary["budget_rate"] == 0.3333

    auto = snap["by_mode"]["auto"]
    assert auto["router_counts"]["llm"] == 1
    assert auto["routing_llm_rate"] == 0.6667
    assert auto["deadline_rate"] == 0.3333
    assert auto["budget_rate"] == 0.3333


def test_record_never_raises_on_garbage():
    record_done({})  # 缺 multi_agent
    record_done({"multi_agent": "not-a-dict"})
    record_done(None)  # type: ignore[arg-type]
    snap = multi_agent_snapshot()
    assert snap["total_requests"] == 2  # None 不记录，两个 dict 记录为 unknown


def test_reset():
    record_done(_done())
    reset_multi_agent_metrics()
    snap = multi_agent_snapshot()
    assert snap["total_requests"] == 0
    assert snap["by_mode"] == {}


def test_sample_cap():
    for i in range(MAX_SAMPLES_PER_GROUP + 5):
        record_done(_done(elapsed=float(i), calls=1, tokens=10))
    snap = multi_agent_snapshot()
    latency = snap["by_mode"]["auto"]["latency_ms"]
    assert latency["samples"] == MAX_SAMPLES_PER_GROUP
    # 保留最近样本：最后一条 elapsed 应接近 cap+4
    assert latency["max"] == float(MAX_SAMPLES_PER_GROUP + 4)
