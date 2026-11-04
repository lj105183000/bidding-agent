"""多 Agent 评测线：注册、数据、纯函数与假 Agent 端到端。"""

from __future__ import annotations

import os
from pathlib import Path

import eval.run_all as RA
from eval import dataset_registry as DR
from eval import report as R
from eval.common import load_cases

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def test_agent_multi_registered_in_all_suites():
    assert "agent_multi" in RA.ALL_SUITES
    assert RA.parse_suites("agent_multi") == ["agent_multi"]


def test_agent_multi_metrics_registered():
    names = {name for name, spec in R.METRIC_SPECS.items() if spec.suite == "agent_multi"}
    required = {
        "agent_multi.routing_accuracy",
        "agent_multi.specialist_tool_accuracy",
        "agent_multi.synthesis_citation_rate",
        "agent_multi.latency_p95",
        "agent_multi.partial_rate",
        "agent_multi.fallback_rate",
        "agent_multi.llm_calls_mean",
        "agent_multi.routing_llm_rate",
        "agent_multi.hit_specialist_cap_rate",
        "agent_multi.verification_critical_ok_rate",
        "agent_multi.verification_ok_rate",
        "agent_multi.false_unavailable_rate",
        "agent_multi.answer_chars_mean",
        "agent_multi.tokens_per_case",
        "agent_multi.cost_usd_per_case",
    }
    assert required <= names


def test_agent_multi_thresholds_calibrated():
    """首轮标定后的门槛必须固化，避免后续改动把红线悄悄调松或调没。"""
    expected = {
        "agent_multi.routing_accuracy": 0.90,
        "agent_multi.specialist_tool_accuracy": 0.90,
        "agent_multi.synthesis_citation_rate": 0.90,
        "agent_multi.latency_p95": 60000.0,
        "agent_multi.partial_rate": 0.10,
        "agent_multi.fallback_rate": 0.10,
        "agent_multi.llm_calls_mean": 5.0,
        "agent_multi.routing_llm_rate": 0.15,
        "agent_multi.hit_specialist_cap_rate": 0.20,
        "agent_multi.verification_critical_ok_rate": 0.95,
        "agent_multi.verification_ok_rate": 0.90,
        "agent_multi.false_unavailable_rate": 0.0,
    }
    for name, threshold in expected.items():
        assert R.METRIC_SPECS[name].threshold == threshold, name

    # 首轮标定只有 3 次实测；报告会据此提示"门槛仍是临时的"。
    calibrated = (
        "agent_multi.routing_accuracy",
        "agent_multi.specialist_tool_accuracy",
        "agent_multi.synthesis_citation_rate",
        "agent_multi.partial_rate",
        "agent_multi.fallback_rate",
        "agent_multi.llm_calls_mean",
        "agent_multi.routing_llm_rate",
        "agent_multi.hit_specialist_cap_rate",
    )
    for name in calibrated:
        assert R.METRIC_SPECS[name].calibrated_on_runs == 3, name


def test_agent_multi_dataset_registered_and_cases_valid():
    info = DR.DATASETS["agent_multi"]
    assert info.path == EVAL_DIR / "test_cases_multi.json"
    cases = load_cases(info.path, required=("question", "expected_agents"))
    assert len(cases) >= 18
    assert any(case.get("requires_web") for case in cases)
    assert any(case.get("expect_fallback") for case in cases)
    for case in cases:
        assert isinstance(case["expected_agents"], list)
        expected_tools = case.get("expected_tools") or {}
        assert set(expected_tools) <= set(case["expected_agents"])


def test_routing_ok():
    case = {"expected_agents": ["data", "retrieval"]}
    assert RA._multi_routing_ok(case, ["retrieval", "data"], False)
    assert not RA._multi_routing_ok(case, ["retrieval"], False)
    # 额外专家默认允许（只看覆盖 + forbidden）；显式禁止才判失败。
    assert RA._multi_routing_ok(case, ["data", "retrieval", "compliance"], False)
    forbidden = {"expected_agents": ["data", "retrieval"], "forbidden_agents": ["compliance"]}
    assert not RA._multi_routing_ok(
        forbidden, ["data", "retrieval", "compliance"], False
    )

    fallback_case = {"expect_fallback": True, "expected_agents": []}
    assert RA._multi_routing_ok(fallback_case, [], True)
    assert not RA._multi_routing_ok(fallback_case, ["retrieval"], False)


def test_specialist_tools_ok():
    case = {
        "expected_tools": {
            "data": ["query_database"],
            "retrieval": ["search_knowledge_base"],
        }
    }
    assert not RA._multi_specialist_tools_ok(
        case,
        {"data": ["query_database"], "retrieval": ["search_knowledge_graph"]},
    )
    assert RA._multi_specialist_tools_ok(
        case,
        {"data": ["query_database"], "retrieval": ["search_knowledge_base"]},
    )
    assert not RA._multi_specialist_tools_ok(
        case,
        {"data": [], "retrieval": ["search_knowledge_base"]},
    )


def test_verification_policy_can_ignore_derived_numbers():
    """数据题的均值/占比是算出来的，规则核对无法验证；条款号仍必须核对。"""
    result = {
        "verification": {
            "ok": False,
            "critical_ok": False,
            "unsupported_laws": [],
            "unsupported_articles": [],
            "unsupported_numbers": ["249.9 万元"],
            "false_unavailable_claims": [],
            "summary": "数字未经检索结果支持：249.9 万元",
        }
    }
    # 未声明时保持原样（不能影响旧用例）
    assert RA._apply_verification_policy({}, result) is result

    patched = RA._apply_verification_policy({"verify_numbers": False}, result)
    assert patched is not result  # 不原地修改，避免污染其它指标
    assert patched["verification"]["unsupported_numbers"] == []
    assert patched["verification"]["ok"] is True
    assert patched["verification"]["critical_ok"] is True

    # 豁免数字后，条款号编造仍然是严重失败
    result2 = {
        "verification": {
            "ok": False,
            "critical_ok": False,
            "unsupported_laws": [],
            "unsupported_articles": ["第三十二条"],
            "unsupported_numbers": ["10%"],
            "false_unavailable_claims": [],
            "summary": "x",
        }
    }
    patched2 = RA._apply_verification_policy({"verify_numbers": False}, result2)
    assert patched2["verification"]["critical_ok"] is False
    assert patched2["verification"]["ok"] is False


def test_suite_env_forces_modes(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_AGENT", "auto")
    with RA._suite_env("agent"):
        assert os.environ["AGENT_MULTI_AGENT"] == "off"
    assert os.environ["AGENT_MULTI_AGENT"] == "auto"

    with RA._suite_env("agent_multi"):
        assert os.environ["AGENT_MULTI_AGENT"] == "on"
        assert os.environ["AGENT_MULTI_ROUTER"] == "off"
    assert os.environ["AGENT_MULTI_AGENT"] == "auto"


class _FakeAgent:
    def chat(self, question, **kwargs):
        return {
            "answer": "综合回答",
            "elapsed_ms": 1200,
            "orchestration": {
                "route": ["retrieval"],
                "fallback": "none",
                "router": "rules",
                "partial": False,
            },
            "agent_trace": [
                {
                    "agent": "retrieval",
                    "status": "ok",
                    "rounds": 1,
                    "tool_name": "search_knowledge_base",
                }
            ],
            "tool_timings": [
                {
                    "agent": "retrieval",
                    "name": "search_knowledge_base",
                    "ok": True,
                    "round": 1,
                }
            ],
            "tool_names": ["search_knowledge_base"],
            "tool_called": True,
            "verification": {"ok": True, "critical_ok": True},
            "sources": [{"question": "q", "answer": "a"}],
            "llm_trace": {"call_count": 3},
        }


def test_run_agent_multi_suite_metrics(monkeypatch):
    monkeypatch.setattr("src.agent.BiddingAgent", lambda: _FakeAgent())
    suite = RA.run_agent_multi_suite(limit=1, web_mode="off")

    assert suite.ran is True
    values = {result.name: result.value for result in suite.results}
    assert values["agent_multi.routing_accuracy"] == 1.0
    assert values["agent_multi.specialist_tool_accuracy"] == 1.0
    assert values["agent_multi.synthesis_citation_rate"] == 1.0
    assert values["agent_multi.partial_rate"] == 0.0
    assert values["agent_multi.fallback_rate"] == 0.0
    assert values["agent_multi.llm_calls_mean"] == 3.0
    assert values["agent_multi.latency_p95"] == 1200.0
    assert values["agent_multi.hit_specialist_cap_rate"] == 0.0
    assert values["agent_multi.verification_critical_ok_rate"] == 1.0
    assert values["agent_multi.verification_ok_rate"] == 1.0
    assert values["agent_multi.false_unavailable_rate"] == 0.0
