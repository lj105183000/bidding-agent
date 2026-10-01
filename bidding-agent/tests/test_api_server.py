"""API 层 → BiddingAgent 接线测试（接口联调 spec §5）。

用模块级 monkeypatch 把 `api.server.bidding_agent` 替换为文件内局部 fake agent，
校验：
  - /api/chat/stream 透传 agent 产出的 SSE 帧（type 键、首帧 padding、末帧 [DONE]）；
  - /api/chat 返回 agent.chat 的 dict；
  - 请求参数正确传递给 agent（provider / 开关）；
  - Agent 未就绪（None）时流式返回 error 帧、非流式返回 503；
  - 无 PostgreSQL 时会话 / 反馈端点返回 503。

同步风格，不引入 asyncio 插件。
"""

import json

from fastapi.testclient import TestClient

from src.agent.utils import _sse, SSE_DONE_MARK, SSE_PADDING
from api import server


class FakeAgent:
    """局部 fake：只记录调用参数，按预设回放 SSE 帧 / chat dict。"""

    def __init__(self, stream_frames=None, chat_result=None):
        self.stream_frames = list(stream_frames or [])
        self.chat_result = chat_result or {
            "answer": "直接回答",
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",
            "elapsed_ms": 5,
            "phase_times": [["首轮分析", 3]],
        }
        self.stream_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    def chat_stream(self, question, history=None, web_search_enabled=False,
                    provider="", deep_thinking_enabled=False):
        self.stream_calls.append({
            "question": question,
            "history": history,
            "web_search_enabled": web_search_enabled,
            "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
        })
        yield from self.stream_frames

    def chat(self, question, history=None, web_search_enabled=False,
             provider="", deep_thinking_enabled=False):
        self.chat_calls.append({
            "question": question,
            "history": history,
            "web_search_enabled": web_search_enabled,
            "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
        })
        return self.chat_result


def _build_stream_frames():
    return [
        SSE_PADDING,
        _sse({"type": "status", "content": "正在初始化..."}),
        _sse({"type": "token", "content": "你"}),
        _sse({"type": "token", "content": "好"}),
        _sse({
            "type": "done",
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",
            "elapsed_ms": 12,
            "phase_times": [["首轮分析", 5]],
        }),
        SSE_DONE_MARK,
    ]


def _parse_sse(text: str):
    """把 SSE 响应文本还原为事件 dict 列表（跳过 padding 与 [DONE]）。"""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            if payload.startswith(":") or not payload.strip():
                continue
            events.append(json.loads(payload))
    return events


def test_chat_stream_proxies_agent_frames(monkeypatch):
    fake = FakeAgent(stream_frames=_build_stream_frames())
    monkeypatch.setattr(server, "bidding_agent", fake)

    client = TestClient(server.app)
    resp = client.post(
        "/api/chat/stream",
        json={"question": "你好"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("x-accel-buffering") == "no"

    text = resp.text
    # 首帧是 2KB padding 注释行
    assert text.startswith(SSE_PADDING.splitlines()[0])
    events = _parse_sse(text)
    assert events[0]["type"] == "status" and events[0]["content"] == "正在初始化..."
    assert [e["type"] for e in events].count("token") == 2
    done = events[-1]
    assert done["type"] == "done"
    for key in ("sources", "web_sources", "tool_called", "tool_name", "elapsed_ms", "phase_times"):
        assert key in done
    assert isinstance(done["tool_name"], str)
    # 末帧 [DONE]
    assert text.rstrip().endswith("[DONE]")


def test_chat_stream_passes_params_to_agent(monkeypatch):
    fake = FakeAgent(stream_frames=_build_stream_frames())
    monkeypatch.setattr(server, "bidding_agent", fake)

    client = TestClient(server.app)
    client.post(
        "/api/chat/stream",
        json={
            "question": "招标人是谁",
            "history": [{"role": "user", "content": "旧问题"}],
            "web_search_enabled": True,
            "provider": "deepseek",
            "deep_thinking_enabled": True,
        },
    )

    assert fake.stream_calls == [{
        "question": "招标人是谁",
        "history": [{"role": "user", "content": "旧问题"}],
        "web_search_enabled": True,
        "provider": "deepseek",
        "deep_thinking_enabled": True,
    }]


def test_chat_non_stream_returns_agent_dict(monkeypatch):
    fake = FakeAgent(stream_frames=[])
    monkeypatch.setattr(server, "bidding_agent", fake)

    client = TestClient(server.app)
    resp = client.post("/api/chat", json={"question": "你好"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "直接回答"
    assert body["tool_name"] == ""
    assert fake.chat_calls[0]["provider"] == ""


def test_chat_stream_error_frame_when_agent_unavailable(monkeypatch):
    monkeypatch.setattr(server, "bidding_agent", None)

    client = TestClient(server.app)
    resp = client.post("/api/chat/stream", json={"question": "你好"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "error" in types
    assert "done" not in types
    assert any(e.get("content") for e in events if e["type"] == "error")


def test_chat_returns_503_when_agent_unavailable(monkeypatch):
    monkeypatch.setattr(server, "bidding_agent", None)

    client = TestClient(server.app)
    resp = client.post("/api/chat", json={"question": "你好"})

    assert resp.status_code == 503


def test_empty_question_is_400(monkeypatch):
    monkeypatch.setattr(server, "bidding_agent", FakeAgent())

    client = TestClient(server.app)
    assert client.post("/api/chat/stream", json={"question": "  "}).status_code == 400
    assert client.post("/api/chat", json={"question": ""}).status_code == 400


def test_conversation_endpoints_503_without_pg(monkeypatch):
    """无 PostgreSQL 配置（conftest 未设置 POSTGRES_*）时会话/反馈端点 503。"""
    monkeypatch.setattr(server, "_get_postgres_client", lambda: None)

    client = TestClient(server.app)
    assert client.post(
        "/api/conversations",
        json={"session_id": "s1", "messages": []},
    ).status_code == 503
    assert client.get("/api/conversations").status_code == 503
    assert client.get("/api/conversations/s1").status_code == 503
    assert client.delete("/api/conversations/s1").status_code == 503
    assert client.post(
        "/api/feedback",
        json={"session_id": "s1", "message_id": 1, "rating": "up"},
    ).status_code == 503
