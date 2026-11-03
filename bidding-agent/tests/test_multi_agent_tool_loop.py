"""`_run_tool_loop` 拆分契约：只跑工具循环，不调最终生成。"""

import time

from src.agent import BiddingAgent
from src.agent import react_loop as loop_mod


class _ScriptedClient:
    def __init__(self, raw_responses):
        self.raw_responses = list(raw_responses)
        self.last_reasoning = ""
        self.raw_calls = 0
        self.provider = "fake"
        self.last_used_provider = "fake"
        self.switch_count = 0
        self.chain = []

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        return self.raw_responses.pop(0)

    def thinking_enabled(self):
        return False

    def supports_reasoning(self):
        return False


def _meta():
    now = time.monotonic()
    return {
        "sources": [],
        "web_sources": [],
        "tool_called": False,
        "tool_name": "",
        "phase_times": [],
        "t_start": now,
        "question": "统计中标金额",
        "retrieval_hint": "统计中标金额",
        "web_search_enabled": False,
        "clarify_allowed": False,
        "planning_enabled": False,
    }


def test_run_tool_loop_returns_result_without_final_generation(monkeypatch):
    call = '{"name": "query_database", "arguments": {"query_type": "amount"}}'
    client = _ScriptedClient([call, "数据库结论"])
    agent = BiddingAgent(llm_client=client)
    monkeypatch.setattr(
        loop_mod,
        "TOOL_EXECUTORS",
        {"query_database": lambda args, question="": ("金额数据", [{"question": "q", "answer": "a"}])},
    )

    meta = _meta()
    gen = agent._run_tool_loop(
        [{"role": "system", "content": "sys"}],
        client,
        meta,
        allowed_tools=("query_database",),
        max_rounds=2,
        agent_id="data",
    )
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break

    assert result is not None
    assert result.tool_called is True
    assert result.tool_name == "query_database"
    assert result.tool_names == ["query_database"]
    assert result.rounds == 2
    assert result.hit_cap is False
    assert result.tool_timings[0]["agent"] == "data"
    assert result.tool_timings[0]["round"] == 1
    assert result.messages[-1]["role"] == "user"  # _clean_for_final 的约束消息


def test_allowlist_blocks_tool_outside_specialist(monkeypatch):
    call = '{"name": "search_knowledge_base", "arguments": {"query": "x"}}'
    client = _ScriptedClient([call, "结论"])
    agent = BiddingAgent(llm_client=client)
    monkeypatch.setattr(
        loop_mod,
        "TOOL_EXECUTORS",
        {"search_knowledge_base": lambda args: ("不应执行", [])},
    )

    meta = _meta()
    gen = agent._run_tool_loop(
        [{"role": "system", "content": "sys"}],
        client,
        meta,
        allowed_tools=("query_database",),
        max_rounds=1,
        agent_id="data",
    )
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break

    assert result.tool_called is False
    assert result.tool_name == ""
