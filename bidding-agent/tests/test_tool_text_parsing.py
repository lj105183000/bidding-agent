"""工具文本防御测试：检测 / 归一化 / 解析 / 流式泄漏 reset + 兜底执行。"""

import src.agent.core as core_mod
import src.agent.react_loop as loop_mod
from src.agent import BiddingAgent
from src.agent.tool_defense import (
    _looks_like_tool_text,
    _normalize_tool_text,
    _parse_tool_text,
)


class ScriptedLLM:
    def __init__(self, raw_responses=(), stream_chunks=()):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.raw_calls = 0
        self.stream_calls = 0
        self.retry_messages = []

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages):
        self.stream_calls += 1
        yield from self.stream_chunks


def _install_executors(monkeypatch, mapping):
    registry = dict(mapping)
    monkeypatch.setattr(core_mod, "TOOL_EXECUTORS", registry)
    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", registry)


def _events_by_type(events, event_type):
    return [e for e in events if e[0] == event_type]


# --- 检测 ---

def test_detect_plain_text_is_negative():
    assert not _looks_like_tool_text("")
    assert not _looks_like_tool_text("中标金额为 120 万元。")
    # 提到 tool_calls 一词但无结构特征：不应误判
    assert not _looks_like_tool_text("响应里的 tool_calls 字段为空。")


def test_detect_tool_like_text_is_positive():
    assert _looks_like_tool_text('{"name": "search_web", "arguments": {}}')
    assert _looks_like_tool_text('```json\n{"name": "x", "arguments": {}}\n```')
    assert _looks_like_tool_text('<tool_call>{"name": "x"}</tool_call>')
    assert _looks_like_tool_text('{"tool_calls": []}')


# --- 归一化 ---

def test_normalize_strips_fences_and_tags():
    fenced = '```json\n{"name": "x", "arguments": {}}\n```'
    assert _normalize_tool_text(fenced) == '{"name": "x", "arguments": {}}'
    tagged = '前缀<tool_call>{"name": "x"}</tool_call>后缀'
    assert _normalize_tool_text(tagged) == '{"name": "x"}'


# --- 解析 ---

def test_parse_single_call():
    calls = _parse_tool_text('{"name": "query_database", "arguments": {"q": "金额"}}')
    assert calls == [{"name": "query_database", "arguments": {"q": "金额"}}]


def test_parse_tool_calls_array_and_string_arguments():
    text = (
        '{"tool_calls": ['
        '{"name": "a", "arguments": "{\\"q\\": 1}"},'
        '{"name": "b", "parameters": {"q": 2}}'
        ']}'
    )
    calls = _parse_tool_text(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[0]["arguments"] == {"q": 1}
    assert calls[1]["arguments"] == {"q": 2}


def test_parse_malformed_returns_empty():
    assert _parse_tool_text("") == []
    assert _parse_tool_text("普通回答文本") == []
    assert _parse_tool_text('{"name": broken') == []
    assert _parse_tool_text('{"arguments": {}}') == []  # 缺 name


def test_parse_deduplicates_same_call():
    text = '{"name": "a", "arguments": {"q": 1}}\n{"name": "a", "arguments": {"q": 1}}'
    assert len(_parse_tool_text(text)) == 1


# --- 流式泄漏：reset + 兜底执行 ---

def test_stream_leak_triggers_reset_and_fallback(monkeypatch):
    leak = '{"name": "search_knowledge_graph", "arguments": {"query": "钢材"}}'
    fake = ScriptedLLM(
        # ① 首轮决策调用工具 ② 结果足够直接回答（却在流式正文里再次泄漏） ③ 兜底重生成的回答
        raw_responses=[leak, "信息足够", "修正后的最终回答"],
        stream_chunks=[f'根据查询结果：{leak}'],
    )
    exec_args = []

    def graph_executor(arguments):
        exec_args.append(arguments)
        return "图谱结果", []

    _install_executors(monkeypatch, {"search_knowledge_graph": graph_executor})
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("钢材供应关系"))

    resets = _events_by_type(events, "reset")
    assert len(resets) == 1 and resets[0][1] is None

    # 决策轮 + 兜底各执行一次
    assert exec_args == [{"query": "钢材"}, {"query": "钢材"}]
    # 兜底重生成走 chat_raw
    assert fake.raw_calls == 3

    tokens = "".join(e[1] for e in _events_by_type(events, "token"))
    assert "修正后的最终回答" in tokens
    # 泄漏文本不再以 token 形式出现在 reset 之后
    assert "根据查询结果" not in tokens


def test_chat_resets_answer_on_leak(monkeypatch):
    leak = '{"name": "search_knowledge_graph", "arguments": {"query": "钢材"}}'
    fake = ScriptedLLM(
        raw_responses=[leak, "信息足够", "修正后的最终回答"],
        stream_chunks=[f'根据查询结果：{leak}'],
    )
    _install_executors(monkeypatch, {
        "search_knowledge_graph": lambda args: ("图谱结果", []),
    })
    agent = BiddingAgent(llm_client=fake)

    result = agent.chat("钢材供应关系")

    # 与前端 reset 语义一致：已流正文清空，最终答案为重生成的文本
    assert result["answer"] == "修正后的最终回答"
    assert result["tool_called"] is True
    assert result["tool_name"] == "search_knowledge_graph"


def test_decision_round_malformed_tool_text_retries_once(monkeypatch):
    # 决策轮输出畸形工具 JSON：约束重试一次后恢复正常回答
    fake = ScriptedLLM(
        raw_responses=['{"name": broken', "这次直接回答"],
        stream_chunks=["答"],
    )
    _install_executors(monkeypatch, {})
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("问点什么"))

    assert fake.raw_calls == 2
    assert fake.stream_calls == 1
    assert any(e[0] == "done" for e in events)
