"""DeepSeek / 智谱客户端单测（原生 HTTP 实现）。

与改造前的区别
-------------
原测试 mock 的是 `openai.AsyncOpenAI`（SDK 对象），因此**测不到协议细节**——
SDK 自己怎么拼 URL、怎么分帧 SSE、怎么取增量，测试一概不知。
改造后 mock 的是 httpx 客户端，被测的是我们**自己写的**请求构造与 SSE 解析，
覆盖的是真正会出错的那部分。

本文件不依赖真实网络与 API Key：所有请求都由 fake httpx 客户端应答。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clients import base_client as B  # noqa: E402
from src.clients.deepseek_client import DeepSeekClient  # noqa: E402
from src.clients.zai_client import ZHIPU_BASE_URL, ZaiClient  # noqa: E402


# ===========================================================================
# fake httpx：应答可控，并记录收到的请求
# ===========================================================================


class _FakeResponse:
    def __init__(self, payload: Any = None, status: int = 200, lines: list[str] | None = None):
        self.status_code = status
        self._payload = payload
        self._lines = lines or []
        self.request_body: dict | None = None

    # --- 非流式用 ---
    def json(self):
        return self._payload

    def read(self):
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    # --- 流式用 ---
    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHTTPClient:
    """记录请求 + 回放预设响应。"""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        self._response.request_body = kwargs.get("json")
        return self._response

    def stream(self, method, url, **kwargs):
        self.calls.append({"url": url, "method": method, **kwargs})
        self._response.request_body = kwargs.get("json")
        return self._response


@pytest.fixture
def http(monkeypatch):
    """把 `get_http_client` 换成 fake；返回 (set_response, fake_client)。"""

    def _install(response: _FakeResponse):
        fake = _FakeHTTPClient(response)
        monkeypatch.setattr("src.http_client.get_http_client", lambda: fake)
        return fake

    return _install


def _sse(*payloads: dict, done: bool = True) -> list[str]:
    lines = [f"data: {json.dumps(p, ensure_ascii=False)}" for p in payloads]
    if done:
        lines.append("data: [DONE]")
    return lines


def _chunk(content: str | None = None, reasoning: str | None = None) -> dict:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta}]}


# ===========================================================================
# SSE 解析（最易出错的部分，单独测）
# ===========================================================================


class TestSSEParsing:
    def test_skips_done_marker(self):
        """`data: [DONE]` 不是消息，直接 json.loads 会抛。"""
        resp = _FakeResponse(lines=_sse(_chunk(content="a"), _chunk(content="b")))
        assert [c["choices"][0]["delta"]["content"] for c in B.iter_sse_json(resp)] == ["a", "b"]

    def test_skips_blank_and_comment_lines(self):
        """心跳注释行（`:` 开头）与空行都要跳过，某些网关会周期性发。"""
        resp = _FakeResponse(lines=["", ": ping", "data: " + json.dumps(_chunk(content="x")), ""])
        assert len(list(B.iter_sse_json(resp))) == 1

    def test_tolerates_missing_space_after_data(self):
        """有些网关不按规范加空格（`data:{...}`）。"""
        resp = _FakeResponse(lines=["data:" + json.dumps(_chunk(content="y"))])
        assert len(list(B.iter_sse_json(resp))) == 1

    def test_bad_line_does_not_break_stream(self):
        """★ 单行坏了不能让整流崩掉。

        服务端偶发发不完整帧。跳过它用户只是少看到几个字，
        中断则是整轮回答失败——两者的代价差一个数量级。
        """
        resp = _FakeResponse(
            lines=["data: {坏掉的 JSON", *[f"data: {json.dumps(_chunk(content=c))}" for c in "abc"]]
        )
        assert [c["choices"][0]["delta"]["content"] for c in B.iter_sse_json(resp)] == list("abc")

    def test_ignores_non_dict_json(self):
        """合法 JSON 但不是对象（如数组）也要跳过，否则下游 .get 会炸。"""
        resp = _FakeResponse(lines=["data: [1,2,3]", "data: " + json.dumps(_chunk(content="ok"))])
        assert len(list(B.iter_sse_json(resp))) == 1


class TestExtractDelta:
    def test_content_only(self):
        assert B.extract_delta(_chunk(content="hi")) == ("", "hi")

    def test_reasoning_content_field(self):
        """DeepSeek reasoner 用 `reasoning_content`。"""
        assert B.extract_delta(_chunk(reasoning="想")) == ("想", "")

    def test_reasoning_field_alias(self):
        """★ 部分智谱模型用 `reasoning`（不带 _content 后缀）。

        实测：智谱返回了 460 个 thinking 块——正是靠这个别名取到的。
        """
        chunk = {"choices": [{"delta": {"reasoning": "考虑中"}}]}
        assert B.extract_delta(chunk) == ("考虑中", "")

    def test_empty_choices(self):
        assert B.extract_delta({"choices": []}) == ("", "")

    def test_malformed_shapes(self):
        for bad in ({}, {"choices": None}, {"choices": [{}]}, {"choices": [None]}):
            assert B.extract_delta(bad) == ("", "")


class TestFormatToolCalls:
    def test_formats_to_project_contract(self):
        """必须与 `tool_defense._parse_tool_text` 的期望一致。"""
        out = B.format_tool_calls([{
            "function": {"name": "search_knowledge_base", "arguments": '{"query": "保证金"}'}
        }])
        payload = json.loads(out)
        assert payload["tool_calls"][0]["name"] == "search_knowledge_base"
        assert payload["tool_calls"][0]["arguments"] == {"query": "保证金"}

    def test_arguments_already_dict(self):
        out = B.format_tool_calls([{"function": {"name": "t", "arguments": {"a": 1}}}])
        assert json.loads(out)["tool_calls"][0]["arguments"] == {"a": 1}

    def test_malformed_arguments_becomes_empty_dict(self):
        out = B.format_tool_calls([{"function": {"name": "t", "arguments": "不是JSON"}}])
        assert json.loads(out)["tool_calls"][0]["arguments"] == {}

    def test_skips_entries_without_name(self):
        """没有 name 的条目无法调用，跳过而不是产出一个空工具名。"""
        assert B.format_tool_calls([{"function": {"arguments": "{}"}}]) == ""

    def test_empty_input(self):
        assert B.format_tool_calls([]) == ""
        assert B.format_tool_calls(None) == ""


# ===========================================================================
# DeepSeek 客户端
# ===========================================================================


class TestDeepSeekClient:
    def test_endpoint_and_headers(self, http):
        """★ 端点拼接不能出现 `/v1/v1/`。

        `DEEPSEEK_BASE_URL` 默认已含 `/v1`，若这里再补一次就 404。
        """
        fake = http(_FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
        client = DeepSeekClient(api_key="sk-x", model="deepseek-chat")
        client.chat_raw([{"role": "user", "content": "hi"}])

        call = fake.calls[-1]
        assert call["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert "/v1/v1/" not in call["url"]
        assert call["headers"]["Authorization"] == "Bearer sk-x"

    def test_custom_base_url_no_double_slash(self, http):
        fake = http(_FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
        client = DeepSeekClient(api_key="k", model="m", base_url="https://x.com/v1/")
        client.chat_raw([{"role": "user", "content": "hi"}])
        assert fake.calls[-1]["url"] == "https://x.com/v1/chat/completions"

    def test_stream_yields_content_only(self, http):
        """★ 正文流必须**丢弃**思维链，否则用户会在回答里看到大段思考过程。"""
        http(_FakeResponse(lines=_sse(_chunk(reasoning="想"), _chunk(content="你"), _chunk(content="好"))))
        client = DeepSeekClient(api_key="k", model="m")
        assert list(client.chat_stream([{"role": "user", "content": "hi"}])) == ["你", "好"]

    def test_thinking_stream_separates(self, http):
        http(_FakeResponse(lines=_sse(_chunk(reasoning="想"), _chunk(content="你"))))
        client = DeepSeekClient(api_key="k", model="m")
        assert list(client.chat_stream_thinking([{"role": "user", "content": "hi"}])) == [
            {"type": "thinking", "content": "想"},
            {"type": "content", "content": "你"},
        ]

    def test_chat_raw_returns_text(self, http):
        http(_FakeResponse(payload={"choices": [{"message": {"content": "完整回答"}}]}))
        client = DeepSeekClient(api_key="k", model="m")
        assert client.chat_raw([{"role": "user", "content": "hi"}]) == "完整回答"

    def test_chat_raw_prefers_tool_calls(self, http):
        """有工具调用时返回约定的 JSON 文本，而不是 message.content。"""
        http(_FakeResponse(payload={"choices": [{"message": {
            "content": "我要调工具",
            "tool_calls": [{"function": {"name": "t", "arguments": '{"a":1}'}}],
        }}]}))
        client = DeepSeekClient(api_key="k", model="m")
        out = client.chat_raw([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        assert json.loads(out)["tool_calls"][0]["name"] == "t"

    def test_http_error_raises_with_body(self, http):
        """★ 错误必须带上响应体——厂商的原因只在 body 里。

        只看状态码会得到一句无用的 400，排查时得再复现一次才能看到原因。
        """
        http(_FakeResponse(payload={"error": {"message": "model not found"}}, status=400))
        client = DeepSeekClient(api_key="k", model="m")
        with pytest.raises(RuntimeError) as exc:
            client.chat_raw([{"role": "user", "content": "hi"}])
        assert "400" in str(exc.value)
        assert "model not found" in str(exc.value)

    def test_unavailable_without_key(self):
        client = DeepSeekClient(api_key="", model="m")
        assert client.available() is False
        with pytest.raises(RuntimeError):
            client.chat_raw([{"role": "user", "content": "hi"}])

    def test_unavailable_without_model(self):
        assert DeepSeekClient(api_key="k", model="").available() is False


# ===========================================================================
# 智谱客户端
# ===========================================================================


class TestZaiClient:
    def test_endpoint(self, http):
        fake = http(_FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
        client = ZaiClient(api_key="k", model="glm-4.7-flashx")
        client.chat_raw([{"role": "user", "content": "hi"}])
        assert fake.calls[-1]["url"] == f"{ZHIPU_BASE_URL}/chat/completions"

    def test_provider_is_zhipu_not_zai(self, http):
        """★ provider 必须报 `zhipu`。

        配置里是 `LLM_PROVIDER=zhipu`，改成 "zai" 会让
        `get_llm_client("zhipu")` 的匹配与前端展示失配。
        """
        assert ZaiClient(api_key="k", model="m").provider == "zhipu"

    def test_omits_parallel_tool_calls(self, http):
        """★ 智谱兼容层不认 OpenAI 的 `parallel_tool_calls`，传了会报参数错误。"""
        fake = http(_FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
        client = ZaiClient(api_key="k", model="m")
        client.chat_raw([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        body = fake.calls[-1]["json"]
        assert "tools" in body
        assert "parallel_tool_calls" not in body

    def test_deepseek_does_send_parallel_tool_calls(self, http):
        """对照：DeepSeek 接受该字段（决策轮一次只调一个工具，降工具风暴风险）。"""
        fake = http(_FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}))
        client = DeepSeekClient(api_key="k", model="m")
        client.chat_raw([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        assert fake.calls[-1]["json"]["parallel_tool_calls"] is False

    def test_reasoning_alias_in_thinking_stream(self, http):
        """智谱用 `reasoning`（无 _content 后缀），要能取到思维链。"""
        http(_FakeResponse(lines=_sse({"choices": [{"delta": {"reasoning": "思考"}}]}, _chunk(content="答"))))
        client = ZaiClient(api_key="k", model="m")
        assert list(client.chat_stream_thinking([{"role": "user", "content": "hi"}])) == [
            {"type": "thinking", "content": "思考"},
            {"type": "content", "content": "答"},
        ]


# ===========================================================================
# 不可用客户端：四个方法都要实现
# ===========================================================================


class TestUnavailableClient:
    """★ 四个方法缺一不可——漏一个就会让"优雅降级"路径崩成 500。

    历史事故：本类最初只有 available() 与 chat_stream()，而 ReAct 决策轮
    会直接调 chat_raw()，于是抛 AttributeError，
    "没配密钥"从功能降级变成了服务不可用。
    """

    def test_available_false(self):
        assert B.UnavailableLLMClient("没配").available() is False

    def test_chat_stream_empty(self):
        assert list(B.UnavailableLLMClient("没配").chat_stream([])) == [""]

    def test_chat_raw_empty_string(self):
        """返回空串是**有意的**：决策轮拿到空串即视为"不调用工具"。"""
        assert B.UnavailableLLMClient("没配").chat_raw([]) == ""

    def test_chat_stream_thinking_empty(self):
        assert list(B.UnavailableLLMClient("没配").chat_stream_thinking([])) == []
