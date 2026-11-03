"""多 Agent LangGraph 端到端测试（假 LLM，不连外部服务）。"""

import json

from src.agent import BiddingAgent
from src.agent import core as core_mod
from src.agent import react_loop as loop_mod


class RoleClient:
    """按角色隔离的 ScriptedLLM：每个 specialist / synthesizer 各一条队列。"""

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


def _set_env(monkeypatch):
    monkeypatch.setenv("AGENT_MULTI_AGENT", "on")
    monkeypatch.setenv("AGENT_MULTI_ROUTER", "off")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    monkeypatch.setenv("AGENT_COMPACT", "0")
    monkeypatch.setenv("AGENT_PLANNING", "off")


def _build_agent(monkeypatch):
    _install_tools(monkeypatch)
    data = RoleClient(
        "data",
        [_call("query_database", {"query_type": "amount"}), "数据库结论"],
    )
    retrieval = RoleClient(
        "retrieval",
        [_call("search_knowledge_base", {"query": "供应商关系"}), "知识库结论"],
    )
    synth = RoleClient("synth", stream_chunks=["综合", "回答"])
    base = RoleClient("base", ["基础回答"], ["基础回答"])

    agent = BiddingAgent(llm_client=base)
    agent._specialist_client_factory = lambda role, provider: {
        "data": data,
        "retrieval": retrieval,
    }.get(role, RoleClient(role))
    agent._synthesis_client_factory = lambda provider: synth
    return agent, data, retrieval, synth


def _done(events):
    done = [payload for kind, payload in events if kind == "done"]
    assert len(done) == 1, f"期望恰好一个 done，实际 {len(done)}"
    return done[0]


def test_multi_agent_graph_end_to_end(monkeypatch):
    _set_env(monkeypatch)
    agent, data, retrieval, synth = _build_agent(monkeypatch)

    events = list(agent._chat_events(
        "统计中标金额并分析供应商关系", web_search_enabled=False
    ))
    done = _done(events)
    answer = "".join(
        str(payload) for kind, payload in events if kind == "token"
    )

    assert answer == "综合回答"
    assert done["tool_called"] is True
    # primary = data（index 0）→ 最后一个成功工具
    assert done["tool_name"] == "query_database"
    assert done["tool_names"] == ["query_database", "search_knowledge_base"]
    assert len(done["sources"]) >= 2
    assert done["orchestration"]["enabled"] is True
    assert done["orchestration"]["route"] == ["data", "retrieval"]
    assert done["orchestration"]["fallback"] == "none"
    assert [item["agent"] for item in done["agent_trace"]] == ["data", "retrieval"]
    assert all(item["status"] == "ok" for item in done["agent_trace"])
    phases = [item[0] for item in done["phase_times"]]
    assert phases[:4] == ["问题路由", "并行检索", "合并结果", "生成回答"]
    # done 载荷必须 JSON 安全（_meta 是内部键，SSE 序列化时过滤）
    assert "_meta" in done
    assert "type" not in done
    json.dumps({k: v for k, v in done.items() if k != "_meta"}, ensure_ascii=False)
    assert data.raw_calls == 2
    assert retrieval.raw_calls == 2
    assert synth.stream_calls == 1


def test_auto_single_intent_uses_legacy_path(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("AGENT_MULTI_AGENT", "auto")
    agent, data, retrieval, synth = _build_agent(monkeypatch)

    events = list(agent._chat_events("什么是政府采购", web_search_enabled=False))
    done = _done(events)

    assert done["agent_trace"] == []
    assert done["orchestration"] is None
    assert data.raw_calls == 0
    assert retrieval.raw_calls == 0
    assert synth.stream_calls == 0


def test_flag_off_uses_legacy_path(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("AGENT_MULTI_AGENT", "off")
    agent, data, retrieval, synth = _build_agent(monkeypatch)

    events = list(agent._chat_events(
        "统计中标金额并分析供应商关系", web_search_enabled=False
    ))
    done = _done(events)

    assert done["agent_trace"] == []
    assert done["orchestration"] is None
    assert data.raw_calls == 0
    assert retrieval.raw_calls == 0
    assert synth.stream_calls == 0
