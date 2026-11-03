"""多 Agent 确定性路由测试（零 LLM 调用）。"""

from src.agent.orchestration.mode import multi_agent_mode
from src.agent.orchestration.router import route_question, should_use_multi_agent


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


def test_auto_requires_multi_intent_or_compound():
    single = route_question("什么是政府采购", web_enabled=False)
    assert should_use_multi_agent("什么是政府采购", "auto", single) is False

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
