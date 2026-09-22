"""SSE 事件协议测试：帧构造/解析往返、chat 收集、chat_stream 帧序列、done 帧契约。"""

import json

from src.agent import BiddingAgent
from src.agent.utils import (
    _sse,
    _truncate_history,
    SSE_DONE_MARK,
    SSE_PADDING,
)


class ScriptedLLM:
    def __init__(self, raw_responses=(), stream_chunks=()):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.raw_calls = 0
        self.stream_calls = 0

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages):
        self.stream_calls += 1
        yield from self.stream_chunks


def _parse_frames(frames):
    """把 SSE 帧序列还原为事件 dict 列表（跳过 padding 与 [DONE]）。"""
    events = []
    for frame in frames:
        for line in frame.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    continue
                events.append(json.loads(payload))
    return events


# --- 帧构造 ---

def test_sse_roundtrip_keeps_chinese():
    frame = _sse({"type": "token", "content": "中文内容·测试"})
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    parsed = json.loads(frame[len("data: "):])
    assert parsed == {"type": "token", "content": "中文内容·测试"}


def test_padding_frame_is_comment_about_2kb():
    assert SSE_PADDING.startswith(":")           # SSE 注释行
    assert 2000 <= len(SSE_PADDING) <= 2200      # 约 2KB，撑开代理缓冲


# --- 历史截断 ---

def test_truncate_history_keeps_last_rounds_aligned():
    history = [{"role": "assistant", "content": "旧开场"}]
    for i in range(6):
        history.append({"role": "user", "content": f"问{i}"})
        history.append({"role": "assistant", "content": f"答{i}"})
    trimmed = _truncate_history(history)

    assert len(trimmed) == 10                              # 5 轮 = 10 条
    assert trimmed[0]["role"] == "user"                    # 对齐到 user 开头
    assert trimmed[0]["content"] == "问1"                  # 保留最近 5 轮
    assert trimmed[-1]["content"] == "答5"
    # 非 user/assistant 角色、空内容被过滤
    assert _truncate_history([{"role": "system", "content": "x"}, None, {}]) == []


# --- chat 收集 ---

def test_chat_collects_done_contract():
    fake = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["你", "好"])
    agent = BiddingAgent(llm_client=fake)

    result = agent.chat("你好")

    assert result["answer"] == "你好"
    assert result["tool_called"] is False
    assert result["tool_name"] == ""                       # 单字符串契约
    assert result["sources"] == [] and result["web_sources"] == []
    assert result["elapsed_ms"] >= 0
    assert [p[0] for p in result["phase_times"]] == ["首轮分析", "生成回答"]


# --- chat_stream 帧序列 ---

def test_chat_stream_frame_sequence_and_done_frame():
    fake = ScriptedLLM(raw_responses=["好的"], stream_chunks=["回", "答"])
    agent = BiddingAgent(llm_client=fake)

    frames = list(agent.chat_stream("测试问题"))

    assert frames[0] == SSE_PADDING                        # 首帧 padding
    assert frames[-1] == SSE_DONE_MARK                     # 末帧 [DONE]

    events = _parse_frames(frames)
    assert events[0]["type"] == "status" and events[0]["content"] == "正在初始化..."
    assert events[-1]["type"] == "done"

    done = events[-1]
    for key in ("sources", "web_sources", "tool_called", "tool_name", "elapsed_ms", "phase_times"):
        assert key in done
    assert isinstance(done["tool_name"], str)              # tool_name 单字符串语义


def test_chat_stream_error_frame_when_no_backend():
    # LLM 工厂与 RAG 均不可用：产出 error 帧，不产出 done
    agent = BiddingAgent(llm_client=None, rag_pipeline=None)

    frames = list(agent.chat_stream("任何问题"))
    events = _parse_frames(frames)
    types = [e["type"] for e in events]

    assert "error" in types
    assert "done" not in types
    error = next(e for e in events if e["type"] == "error")
    assert error["content"]                                # 用户可见文案非空