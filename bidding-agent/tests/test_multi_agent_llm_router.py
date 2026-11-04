"""Phase 1b LLM 路由：成功、校验、超时回落与 auto 触发条件。"""

from __future__ import annotations

import json
import time

from src.agent.orchestration.llm_router import route_with_llm
from src.agent.orchestration.runtime import prepare_multi_agent


class _RouterClient:
    """最小 fake：只实现 chat_raw/clone，记录调用次数。"""

    def __init__(self, response="", *, delay=0.0):
        self.response = response
        self.delay = delay
        self.raw_calls = 0
        self.trace_recorder = None
        self.provider = "fake"
        self.last_used_provider = "fake"
        self.switch_count = 0

    def clone(self):
        return self

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _json(agents, reason="理由"):
    return json.dumps({"agents": agents, "reason": reason}, ensure_ascii=False)


def test_llm_router_success_builds_tasks():
    client = _RouterClient(_json(["data", "retrieval"]))
    result = route_with_llm("统计中标金额并分析供应商关系", web_enabled=False, client=client)
    assert [task.agent for task in result.tasks] == ["data", "retrieval"]
    assert result.reason == "理由"
    assert result.attempted is True
    assert result.error == ""
    assert client.raw_calls == 1


def test_llm_router_invalid_json_returns_error():
    client = _RouterClient("不是 JSON")
    result = route_with_llm("统计中标金额", web_enabled=False, client=client)
    assert result.tasks == ()
    assert result.error == "invalid_json"


def test_llm_router_fenced_json_is_parsed():
    client = _RouterClient("```json\n" + _json(["compliance"]) + "\n```")
    result = route_with_llm("这条条款合规吗", web_enabled=False, client=client)
    assert [task.agent for task in result.tasks] == ["compliance"]


def test_llm_router_web_disabled_filters_web():
    client = _RouterClient(_json(["web", "data"]))
    result = route_with_llm("最新政策和金额统计", web_enabled=False, client=client)
    assert [task.agent for task in result.tasks] == ["data"]

    only_web = _RouterClient(_json(["web"]))
    result2 = route_with_llm("最新政策", web_enabled=False, client=only_web)
    assert result2.tasks == ()
    assert result2.error == "no_allowed_agents"


def test_llm_router_caps_and_dedupes():
    client = _RouterClient(_json(["data", "data", "retrieval", "compliance", "web"]))
    result = route_with_llm("综合问题", web_enabled=True, client=client)
    assert [task.agent for task in result.tasks] == [
        "data",
        "retrieval",
        "compliance",
    ]
    assert [task.index for task in result.tasks] == [0, 1, 2]


def test_llm_router_timeout_returns_error():
    client = _RouterClient(_json(["data"]), delay=0.3)
    result = route_with_llm(
        "统计中标金额", web_enabled=False, client=client, timeout_s=0.05
    )
    assert result.tasks == ()
    assert result.error == "timeout"


def test_llm_router_client_unavailable():
    result = route_with_llm("统计中标金额", web_enabled=False, client=None)
    assert result.tasks == ()
    assert result.error == "client_unavailable"
    assert result.attempted is False


def test_auto_router_skips_clear_multi_expert_rules(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "auto")
    client = _RouterClient(_json(["data", "retrieval"]))
    prepared = prepare_multi_agent(
        "统计中标金额并分析供应商关系",
        web_enabled=False,
        mode="auto",
        client=client,
    )
    assert prepared is not None
    assert prepared.router == "rules"
    assert client.raw_calls == 0


def test_auto_router_tries_llm_for_compound_single_rule(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "auto")
    client = _RouterClient(_json(["data", "retrieval"]))
    prepared = prepare_multi_agent(
        "公开招标和邀请招标有什么区别？",
        web_enabled=False,
        mode="auto",
        client=client,
    )
    assert prepared is not None
    assert prepared.router == "llm"
    assert [task.agent for task in prepared.tasks] == ["data", "retrieval"]
    assert client.raw_calls == 1


def test_auto_llm_single_agent_declines(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "auto")
    client = _RouterClient(_json(["data"]))
    decision: dict = {}
    prepared = prepare_multi_agent(
        "公开招标和邀请招标有什么区别？",
        web_enabled=False,
        mode="auto",
        client=client,
        decision=decision,
    )
    assert prepared is None
    assert decision["router"] == "llm"
    assert decision["decision"] == "declined_single_expert"
    assert decision["route"] == ["data"]


def test_auto_router_does_not_call_for_simple_single(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "auto")
    client = _RouterClient(_json(["data", "retrieval"]))
    prepared = prepare_multi_agent(
        "什么是政府采购",
        web_enabled=False,
        mode="auto",
        client=client,
    )
    assert prepared is None
    assert client.raw_calls == 0


def test_llm_router_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "llm")
    client = _RouterClient("not json")
    prepared = prepare_multi_agent(
        "统计中标金额并分析供应商关系",
        web_enabled=False,
        mode="on",
        client=client,
    )
    assert prepared is not None
    assert prepared.router == "llm_fallback"
    assert prepared.router_reason == "invalid_json"
    assert "data" in [task.agent for task in prepared.tasks]


def test_off_mode_never_calls_router(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "llm")
    client = _RouterClient(_json(["data"]))
    prepared = prepare_multi_agent(
        "统计中标金额并分析供应商关系",
        web_enabled=False,
        mode="off",
        client=client,
    )
    assert prepared is None
    assert client.raw_calls == 0


def test_router_mode_off_never_calls_llm(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "off")
    client = _RouterClient(_json(["data"]))
    prepared = prepare_multi_agent(
        "统计中标金额并分析供应商关系",
        web_enabled=False,
        mode="on",
        client=client,
    )
    assert prepared is not None
    assert prepared.router == "rules"
    assert client.raw_calls == 0
