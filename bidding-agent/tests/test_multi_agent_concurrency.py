"""多 Agent 工具并发槽：共享 semaphore 限制跨 specialist 的工具并发。"""

from __future__ import annotations

import threading
import time

from src.agent import BiddingAgent
from src.agent import react_loop as loop_mod


class _DummyClient:
    def available(self):
        return True

    def chat_raw(self, messages, tools=None, **kwargs):
        return "结论"

    def chat_stream(self, messages, **kwargs):
        yield "回答"

    def thinking_enabled(self):
        return False

    def supports_reasoning(self):
        return False


def test_tool_semaphore_bounds_concurrent_executors(monkeypatch):
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_tool(args, question=""):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            return f"结果:{args.get('id')}", []
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", {"fake_tool": fake_tool})
    agent = BiddingAgent(llm_client=_DummyClient())
    semaphore = threading.BoundedSemaphore(2)

    calls = [
        {"name": "fake_tool", "arguments": {"id": index}}
        for index in range(4)
    ]
    results = agent._execute_tool_batch(
        calls, allowed=None, question="问题", tool_semaphore=semaphore
    )

    assert len(results) == 4
    assert all(ok for _text, _sources, ok, _obs, _dup in results)
    assert state["max_active"] <= 2


def test_tool_semaphore_default_none_preserves_behavior(monkeypatch):
    def fake_tool(args, question=""):
        return "ok", []

    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", {"fake_tool": fake_tool})
    agent = BiddingAgent(llm_client=_DummyClient())
    results = agent._execute_tool_batch(
        [{"name": "fake_tool", "arguments": {}}],
        allowed=None,
        question="问题",
    )
    assert results[0][0] == "ok"
    assert results[0][2] is True
