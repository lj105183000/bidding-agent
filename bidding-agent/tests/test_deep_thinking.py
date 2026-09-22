"""深度思考测试：思考模型切换 / max_tokens 预算 / 事件流 / 提示词注入回退。"""

from src.agent import BiddingAgent
from src.agent.constants import MAX_TOKENS_DEEP
from src.agent.prompts import DEEP_THINKING_INJECT


class ScriptedLLM:
    def __init__(self, raw_responses=(), stream_chunks=()):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.stream_messages = []
        self.raw_calls = 0

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages):
        self.stream_messages.append(messages)
        yield from self.stream_chunks


class ThinkingLLM(ScriptedLLM):
    """支持深度思考的客户端：chat_stream_thinking 产出 (kind, 文本)，kind ∈ thinking/content。"""

    def __init__(self):
        super().__init__(raw_responses=["直接回答"])
        self.thinking_kwargs = []

    def chat_stream_thinking(self, messages, max_tokens=None):
        self.thinking_kwargs.append({"max_tokens": max_tokens})
        self.stream_messages.append(messages)
        yield "thinking", "推理过程片段"
        yield "content", "最终答案"


def _events_by_type(events, event_type):
    return [e for e in events if e[0] == event_type]


def test_deep_thinking_emits_thinking_then_tokens():
    client = ThinkingLLM()
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    thinking = _events_by_type(events, "thinking")
    tokens = _events_by_type(events, "token")
    assert [e[1] for e in thinking] == ["推理过程片段"]
    assert "".join(e[1] for e in tokens) == "最终答案"
    # 思考增量先于正文增量
    assert events.index(thinking[0]) < events.index(tokens[0])
    assert any(e[0] == "done" for e in events)

    # 思考输出预算：reasoning 计入 max_tokens
    assert client.thinking_kwargs == [{"max_tokens": MAX_TOKENS_DEEP}]


def test_deep_thinking_status_sequence():
    client = ThinkingLLM()
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))
    statuses = [e[1] for e in _events_by_type(events, "status")]

    # §6.3 序列：正在初始化... → 正在深度思考... → ...
    assert statuses[0] == "正在初始化..."
    assert statuses[1] == "正在深度思考..."


def test_deep_thinking_falls_back_to_prompt_injection():
    # 客户端不支持 chat_stream_thinking：回退为提示词注入，且不产出 thinking 事件
    client = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["回", "答"])
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    assert _events_by_type(events, "thinking") == []
    assert any(e[0] == "done" for e in events)
    system = client.stream_messages[0][0]["content"]
    assert DEEP_THINKING_INJECT in system


def test_thinking_model_switch(monkeypatch):
    switched = []

    class SwitchableLLM(ScriptedLLM):
        def switch_thinking_model(self, model_name):
            switched.append(model_name)

    monkeypatch.setenv("DEEPSEEK_THINKING_MODEL", "deepseek-r2")
    agent = BiddingAgent(llm_client=SwitchableLLM(raw_responses=["答"], stream_chunks=["答"]))

    client = agent._thinking_model(agent._get_llm_client(), "deepseek")
    assert switched == ["deepseek-r2"]
    assert client is agent._get_llm_client()


def test_thinking_model_not_configured_returns_client_unchanged(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_THINKING_MODEL", raising=False)
    client = ScriptedLLM(raw_responses=["答"], stream_chunks=["答"])
    agent = BiddingAgent(llm_client=client)

    assert agent._thinking_model(client, "deepseek") is client