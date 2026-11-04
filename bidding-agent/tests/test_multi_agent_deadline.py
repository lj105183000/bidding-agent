"""多 Agent 总 deadline / idle timeout 的行为测试（假图，不连外部服务）。"""

from __future__ import annotations

import time

import pytest

from src.agent.orchestration import runtime
from src.agent.orchestration.router import AgentTask
from src.agent.orchestration.runtime import (
    LegacyFallback,
    PreparedRun,
    run_multi_agent_stream,
)


class _FakeGraph:
    def __init__(self, events, delay=1.0):
        self.events = list(events)
        self.delay = delay

    def stream(self, state, stream_mode="custom"):
        for event in self.events:
            yield event
        time.sleep(self.delay)


class _FakeAgent:
    pass


def _prepared(deadline_in=0.2):
    task = AgentTask(
        agent="retrieval",
        index=0,
        task="问题",
        tools=("search_knowledge_base",),
    )
    return PreparedRun(
        mode="on",
        tasks=(task,),
        deadline=time.monotonic() + deadline_in,
    )


def _run(agent, prepared, meta):
    return run_multi_agent_stream(
        agent,
        prepared,
        question="问题",
        history=None,
        session_id="",
        provider="",
        web_enabled=False,
        deep_thinking_enabled=False,
        meta=meta,
    )


def test_deadline_before_first_token_falls_back_to_legacy():
    agent = _FakeAgent()
    agent._multi_agent_graph = _FakeGraph([], delay=1.0)
    meta: dict = {}
    with pytest.raises(LegacyFallback):
        list(_run(agent, _prepared(0.2), meta))
    assert meta["_multi_agent_needs_legacy"] is True
    assert meta["_multi_agent_deadline_exceeded"] is True
    assert meta["orchestration"]["deadline_exceeded"] is True
    assert meta["orchestration"]["partial"] is True
    assert meta["orchestration"]["fallback"] == "legacy"


def test_deadline_after_first_token_returns_partial():
    agent = _FakeAgent()
    agent._multi_agent_graph = _FakeGraph(
        [{"type": "token", "content": "部分"}], delay=1.0
    )
    meta: dict = {}
    gen = _run(agent, _prepared(0.2), meta)
    first = next(gen)
    assert first == ("token", "部分")
    # 模拟 core 已经从 token 累积了正文；截止后不应再回退旧 ReAct
    meta["answer_text"] = "部分"
    assert list(gen) == []

    assert meta.get("_multi_agent_needs_legacy") is not True
    assert meta["_multi_agent_deadline_exceeded"] is True
    assert meta["orchestration"]["deadline_exceeded"] is True
    assert meta["orchestration"]["partial"] is True
    assert meta["orchestration"]["fallback"] == "none"


def test_idle_timeout_marks_deadline(monkeypatch):
    monkeypatch.setattr(runtime, "MULTI_AGENT_IDLE_TIMEOUT_S", 0.1)
    agent = _FakeAgent()
    agent._multi_agent_graph = _FakeGraph([], delay=1.0)
    meta: dict = {}
    with pytest.raises(LegacyFallback):
        list(_run(agent, _prepared(5.0), meta))
    assert meta["_multi_agent_timeout_reason"] == "idle"
    assert meta["orchestration"]["deadline_exceeded"] is True
