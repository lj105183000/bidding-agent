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
                    provider="", deep_thinking_enabled=False,
                    auto_web_fallback=False, session_id=""):
        self.stream_calls.append({
            "question": question,
            "history": history,
            "web_search_enabled": web_search_enabled,
            "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
            "auto_web_fallback": auto_web_fallback,
            # 契约新增：历史压缩的滚动水位线按 session_id 分桶，
            # API 层必须把它透传给 Agent（本测试断言这一点）
            "session_id": session_id,
        })
        yield from self.stream_frames

    def chat(self, question, history=None, web_search_enabled=False,
             provider="", deep_thinking_enabled=False, auto_web_fallback=False,
             session_id=""):
        self.chat_calls.append({
            "question": question,
            "history": history,
            "web_search_enabled": web_search_enabled,
            "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
            "session_id": session_id,
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
        # 未传 web_search_mode 时缺省是 "auto"，此时 web_search_enabled 取自旧的布尔字段
        "web_search_enabled": True,
        "provider": "deepseek",
        "deep_thinking_enabled": True,
        "auto_web_fallback": True,
        # 未传 session_id 时透传空串（历史压缩按"无会话"共享状态，功能仍可用）
        "session_id": "",
    }]


def test_chat_stream_passes_session_id_to_agent(monkeypatch):
    """★ 契约：`session_id` 必须透传到 Agent。

    它用于历史压缩的滚动水位线分桶——**不透传的话压缩状态无法按会话隔离**，
    表现是"不同会话共享同一份早期摘要"，凭空多出不属于当前会话的上下文。
    这类缺陷不会报错，只会让模型莫名提到别的话题。
    """
    fake = FakeAgent(stream_frames=[])
    monkeypatch.setattr(server, "bidding_agent", fake)

    client = TestClient(server.app)
    client.post("/api/chat/stream", json={"question": "你好", "session_id": "sess-abc"})

    assert fake.stream_calls[0]["session_id"] == "sess-abc"


def test_chat_non_stream_passes_session_id_to_agent(monkeypatch):
    """★ 回归哨兵：**非流式**端点也必须透传。

    实测踩坑：加参数时只改了流式端点，`/api/chat` 漏了。
    后果**不报错、不改行为**——只是每次请求都当作新会话、
    每轮重新压缩一遍，白花一次 LLM 调用（约 2~5 秒）。
    只测流式端点会让这个漏洞完全隐身。
    """
    fake = FakeAgent(stream_frames=[])
    monkeypatch.setattr(server, "bidding_agent", fake)

    client = TestClient(server.app)
    client.post("/api/chat", json={"question": "你好", "session_id": "sess-xyz"})

    assert fake.chat_calls[0]["session_id"] == "sess-xyz"


def test_debug_compaction_endpoint_shape(monkeypatch):
    """诊断端点：压缩状态是进程内的，从网络响应看不到，必须能直接读状态表。

    它同时是"多 worker 部署会让压缩失效"这一局限的验证手段——
    若部署了多 worker，不同请求的 `pid` / `store_id` 会不同。
    """
    from src.agent import compaction as C

    C.clear_all()
    C._set_state("sess-diag", "摘要内容", 12)

    client = TestClient(server.app)
    body = client.get("/api/debug/compaction").json()

    assert body["session_count"] == 1
    assert body["sessions"]["sess-diag"]["covered"] == 12
    assert body["sessions"]["sess-diag"]["summary_len"] == 4
    assert isinstance(body["pid"], int)
    C.clear_all()


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
