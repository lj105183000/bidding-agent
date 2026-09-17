"""DeepSeek 客户端单元测试。

通过 monkeypatch 替换 src.clients.deepseek.AsyncOpenAI 为 fake，
不依赖真实网络与 API Key。
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.clients.deepseek import DeepSeekClient


class _FakeDelta:
    """模拟 openai 流式 chunk 的 delta。"""

    def __init__(self, content: str | None = None, reasoning: str | None = None):
        self.content = content
        # DeepSeek 用 reasoning_content 字段承载思考过程
        self.reasoning_content = reasoning


class _FakeChunk:
    def __init__(self, content: str | None = None, reasoning: str | None = None):
        self.choices = [SimpleNamespace(delta=_FakeDelta(content, reasoning))]


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """模拟 chat.completions.create。"""

    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        stream = kwargs.get("stream", False)

        if not stream:
            return _FakeResponse("你好，我是 DeepSeek")

        # 流式：先输出一段思考，再输出正文
        async def gen():
            yield _FakeChunk(reasoning="正在思考")
            yield _FakeChunk(content="你")
            yield _FakeChunk(content="好")

        return gen()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    """模拟 openai.AsyncOpenAI。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _FakeChat()


@pytest.fixture
def patched_client(monkeypatch):
    """注入 fake AsyncOpenAI，返回已构造好的 DeepSeekClient。"""
    fake = _FakeAsyncOpenAI()
    monkeypatch.setattr("src.clients.deepseek.AsyncOpenAI", lambda **kw: fake)
    client = DeepSeekClient(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
    )
    return client, fake


def test_model_name_and_provider(patched_client):
    client, _ = patched_client
    assert client.model_name == "deepseek-v4-flash"
    assert client.provider == "deepseek"


def test_chat_returns_full_content(patched_client):
    client, fake = patched_client
    answer = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))

    assert answer == "你好，我是 DeepSeek"
    # 应透传 model 与 messages，且非流式
    call = fake.chat.completions.calls[-1]
    assert call["model"] == "deepseek-v4-flash"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call.get("stream") is not True


def test_chat_stream_yields_content_only(patched_client):
    client, _ = patched_client
    chunks = list(asyncio.run(_collect(client.chat_stream([{"role": "user", "content": "hi"}]))))

    # chat_stream 只产出 content 增量
    assert chunks == ["你", "好"]


def test_chat_stream_thinking_separates_thinking_and_content(patched_client):
    client, _ = patched_client
    events = list(
        asyncio.run(
            _collect(client.chat_stream_thinking([{"role": "user", "content": "hi"}]))
        )
    )

    # 思考与正文应被正确分流
    assert events == [
        {"type": "thinking", "content": "正在思考"},
        {"type": "content", "content": "你"},
        {"type": "content", "content": "好"},
    ]


async def _collect(async_iter):
    """将异步迭代器收集为列表。"""
    return [item async for item in async_iter]
