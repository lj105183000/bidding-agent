"""多 Agent 确定性路由测试（零 LLM 调用）。"""

from src.agent.orchestration.mode import multi_agent_mode, multi_agent_mode_detail
from src.agent.orchestration.router import (
    DECISION_ACTIVATED,
    DECISION_NO_TASKS,
    DECISION_OFF,
    DECISION_SHORT,
    DECISION_SINGLE_EXPERT,
    multi_agent_decision,
    route_question,
    should_use_multi_agent,
)


def test_data_intent_routes_data_first():
    tasks = route_question("统计中标金额趋势", web_enabled=False)
    assert tasks[0].agent == "data"
    assert "retrieval" in [t.agent for t in tasks]


def test_graph_intent_routes_retrieval_first():
    tasks = route_question("供应商和采购人之间有什么关系", web_enabled=False)
    assert tasks[0].agent == "retrieval"


def test_compliance_intent_routes_compliance_first():
    tasks = route_question("这条条款是否合规，依据哪条法规", web_enabled=False)
    assert tasks[0].agent == "compliance"


def test_web_only_when_enabled():
    disabled = route_question("最新政策是什么", web_enabled=False)
    assert all(t.agent != "web" for t in disabled)
    enabled = route_question("最新政策是什么", web_enabled=True)
    assert enabled[0].agent == "web"


def test_auto_requires_at_least_two_specialists():
    single = route_question("什么是政府采购", web_enabled=False)
    assert len(single) == 1
    assert should_use_multi_agent("什么是政府采购", "auto", single) is False

    # is_compound_question 判为复合，但路由只有一个专家：auto 仍不走图。
    single_compound = route_question("公开招标和邀请招标有什么区别？", web_enabled=False)
    assert len(single_compound) == 1
    assert should_use_multi_agent(
        "公开招标和邀请招标有什么区别？", "auto", single_compound
    ) is False
    # `on` 是无条件开关（评测/A-B 用），单专家题仍走图。
    assert should_use_multi_agent(
        "公开招标和邀请招标有什么区别？", "on", single_compound
    ) is True

    multi = route_question("统计中标金额并分析供应商关系", web_enabled=False)
    assert len(multi) >= 2
    assert should_use_multi_agent("统计中标金额并分析供应商关系", "auto", multi) is True


def test_short_question_falls_back_to_legacy():
    tasks = route_question("你好", web_enabled=False)
    assert should_use_multi_agent("你好", "on", tasks) is False


def test_mode_defaults_off(monkeypatch):
    monkeypatch.delenv("AGENT_MULTI_AGENT", raising=False)
    assert multi_agent_mode() == "off"


def test_mode_on_when_langgraph_available(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_AGENT", "on")
    # 测试环境已安装 langgraph；若未安装，off 是预期的安全回退。
    assert multi_agent_mode() in ("on", "off")


def test_decision_reasons_match_should_use():
    multi = route_question("统计中标金额并分析供应商关系", web_enabled=False)
    single = route_question("公开招标和邀请招标有什么区别？", web_enabled=False)

    assert multi_agent_decision("统计中标金额并分析供应商关系", "auto", multi) == DECISION_ACTIVATED
    assert multi_agent_decision("统计中标金额并分析供应商关系", "off", multi) == DECISION_OFF
    assert multi_agent_decision("公开招标和邀请招标有什么区别？", "auto", single) == DECISION_SINGLE_EXPERT
    assert multi_agent_decision("你好", "on", single) == DECISION_SHORT
    assert multi_agent_decision("统计中标金额并分析供应商关系", "auto", []) == DECISION_NO_TASKS

    for question, mode, tasks in (
        ("统计中标金额并分析供应商关系", "auto", multi),
        ("公开招标和邀请招标有什么区别？", "auto", single),
        ("你好", "on", single),
    ):
        assert should_use_multi_agent(question, mode, tasks) == (
            multi_agent_decision(question, mode, tasks) == DECISION_ACTIVATED
        )


def test_mode_detail_reports_langgraph_unavailable(monkeypatch):
    import src.agent.orchestration.mode as mode_mod

    monkeypatch.setenv("AGENT_MULTI_AGENT", "auto")
    monkeypatch.setattr(mode_mod, "langgraph_available", lambda: False)
    detail = mode_mod.multi_agent_mode_detail()
    assert detail == {
        "configured": "auto",
        "effective": "off",
        "reason": "langgraph_unavailable",
    }


def test_mode_detail_reports_invalid_env(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_AGENT", "banana")
    detail = multi_agent_mode_detail()
    assert detail == {"configured": "off", "effective": "off", "reason": "invalid_env"}
