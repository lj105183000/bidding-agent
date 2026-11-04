"""离线评测看板的 agent_multi 标签/比例白名单回归。"""

from eval.report import METRIC_SPECS
from src.eval_offline import METRIC_LABELS, RATIO_METRICS, SUITE_LABELS


def test_agent_multi_suite_label():
    assert SUITE_LABELS["agent_multi"] == "多 Agent 编排"


def test_agent_multi_metric_labels_present():
    names = {name for name in METRIC_SPECS if name.startswith("agent_multi.")}
    assert names, "未注册 agent_multi 指标"
    assert names <= set(METRIC_LABELS)


def test_agent_multi_ratio_whitelist():
    ratios = {
        "agent_multi.routing_accuracy",
        "agent_multi.specialist_tool_accuracy",
        "agent_multi.synthesis_citation_rate",
        "agent_multi.partial_rate",
        "agent_multi.fallback_rate",
        "agent_multi.routing_llm_rate",
        "agent_multi.hit_specialist_cap_rate",
        "agent_multi.verification_critical_ok_rate",
        "agent_multi.verification_ok_rate",
        "agent_multi.false_unavailable_rate",
    }
    assert ratios <= RATIO_METRICS

    # 计数/耗时/成本不是比例，不能被画成百分比
    non_ratios = {
        "agent_multi.latency_p95",
        "agent_multi.llm_calls_mean",
        "agent_multi.answer_chars_mean",
        "agent_multi.tokens_per_case",
        "agent_multi.cost_usd_per_case",
    }
    assert not (non_ratios & RATIO_METRICS)
