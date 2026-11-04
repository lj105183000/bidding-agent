"""多 Agent 请求级 LLM 预算：原子计数 + 专家/综合共享上限。"""

from __future__ import annotations

import json
import threading

from src.agent import BiddingAgent
from src.agent import core as core_mod
from src.agent import react_loop as loop_mod
from src.agent.llm_budget import CallBudget
from src.agent.orchestration import runtime


class _RoleClient:
    def __init__(self, role, raw_responses=(), stream_chunks=()):
        self.role = role
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.last_reasoning = ""
        self.raw_calls = 0
        self.stream_calls = 0
        self.provider = role
        self.last_used_provider = role
        self.switch_count = 0
        self.chain = [role]
        self.last_error = []

    def clone(self):
        return self

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        if not self.raw_responses:
            return "结论"
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages, **kwargs):
        self.stream_calls += 1
        yield from self.stream_chunks

    def thinking_enabled(self):
        return False

    def supports_reasoning(self):
        return False

    def available(self):
        return True


def _call(name, arguments):
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def _set_env(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_AGENT", "on")
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "off")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    monkeypatch.setenv("AGENT_COMPACT", "0")
    monkeypatch.setenv("AGENT_PLANNING", "off")


def _install_tools(monkeypatch):
    executors = {
        "query_database": lambda args, question="": (
            "金额：100 万元",
            [{"question": "金额", "answer": "100 万元", "score": 0.9}],
        ),
        "search_knowledge_base": lambda args, question="": (
            "关系：A 与 B 合作",
            [{"question": "关系", "answer": "A-B", "score": 0.8}],
        ),
    }
    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", executors)
    monkeypatch.setattr(core_mod, "TOOL_EXECUTORS", executors)


def _build_agent(monkeypatch):
    _install_tools(monkeypatch)
    data = _RoleClient(
        "data",
        [_call("query_database", {"query_type": "amount"}), "数据库结论"],
    )
    retrieval = _RoleClient(
        "retrieval",
        [_call("search_knowledge_base", {"query": "供应商关系"}), "知识库结论"],
    )
    synth = _RoleClient("synth", stream_chunks=["综合", "回答"])
    base = _RoleClient("base", ["基础回答"], ["基础回答"])

    agent = BiddingAgent(llm_client=base)
    agent._specialist_client_factory = lambda role, provider: {
        "data": data,
        "retrieval": retrieval,
    }.get(role, _RoleClient(role))
    agent._synthesis_client_factory = lambda provider: synth
    return agent, data, retrieval, synth, base


def test_call_budget_is_atomic():
    budget = CallBudget(8)
    acquired: list[bool] = []
    lock = threading.Lock()

    def worker():
        ok = budget.try_acquire()
        with lock:
            acquired.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sum(1 for ok in acquired if ok) == 8
    snap = budget.snapshot()
    assert snap["llm_calls_used"] == 8
    assert snap["llm_calls_denied"] == 12
    assert snap["budget_exhausted"] is True


def test_budget_is_shared_across_specialists_and_synthesis(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(runtime, "MULTI_AGENT_MAX_LLM_CALLS", 3)
    agent, data, retrieval, synth, _base = _build_agent(monkeypatch)

    events = list(agent._chat_events(
        "统计中标金额并分析供应商关系", web_search_enabled=False
    ))
    done = [payload for kind, payload in events if kind == "done"][0]

    # data 2 次 + retrieval 1 次；retrieval 第 2 次与 synthesis 被预算拒绝
    assert data.raw_calls + retrieval.raw_calls + synth.stream_calls == 3
    assert synth.stream_calls == 0
    assert done["orchestration"]["llm_calls_used"] == 3
    assert done["orchestration"]["llm_call_limit"] == 3
    assert done["orchestration"]["budget_exhausted"] is True
    assert done["multi_agent"]["budget_exhausted"] is True
    assert done["orchestration"]["fallback"] in ("fragment", "legacy")
    assert any(kind == "token" for kind, _ in events)
