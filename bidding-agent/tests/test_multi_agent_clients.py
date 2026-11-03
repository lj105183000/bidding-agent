"""并行 specialist 的客户端隔离：thread-local reasoning + clone。"""

import threading

from src.clients.base_client import BaseLLMClient
from src.clients.fallback import FallbackLLMClient


class _Client(BaseLLMClient):
    def available(self):
        return True

    def chat_stream(self, messages, **kwargs):
        yield ""

    def chat_raw(self, messages, tools=None, **kwargs):
        return ""


def test_last_reasoning_is_thread_local():
    client = _Client()
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def worker(name: str):
        client.last_reasoning = name
        barrier.wait()
        results[name] = client.last_reasoning

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results == {"a": "a", "b": "b"}


def test_clone_resets_reasoning_and_recorder():
    client = _Client()
    client.last_reasoning = "old"
    client.trace_recorder = object()

    clone = client.clone()

    assert clone is not client
    assert clone.last_reasoning == ""
    assert clone.trace_recorder is None
    assert clone.thinking_enabled() is False


def test_fallback_clone_isolates_underlying_clients():
    base = _Client()
    fallback = FallbackLLMClient(["deepseek"], lambda name: base)

    clone = fallback.clone()

    assert clone is not fallback
    assert clone.chain == ["deepseek"]
    assert clone._resolver("deepseek") is not base
