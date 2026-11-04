"""Qdrant 读重试：瞬时超时重试、永久错误不重试、写路径不受影响。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag.vector_store import VectorStore


class _ReadTimeout(Exception):
    pass


class _FakeClient:
    def __init__(self, failures: int = 0, exc: Exception | None = None):
        self.failures = failures
        self.exc = exc or _ReadTimeout("read timeout")
        self.count_calls = 0
        self.query_calls = 0
        self.exists_calls = 0

    def count(self, **kwargs):
        self.count_calls += 1
        if self.count_calls <= self.failures:
            raise self.exc
        return SimpleNamespace(count=7)

    def collection_exists(self, name):
        self.exists_calls += 1
        if self.exists_calls <= self.failures:
            raise self.exc
        return True

    def query_points(self, **kwargs):
        self.query_calls += 1
        if self.query_calls <= self.failures:
            raise self.exc
        return SimpleNamespace(points=[])


def _store(client) -> VectorStore:
    return VectorStore(
        url="http://example.test",
        collection="c",
        read_retries=2,
        retry_backoff_s=0,
        client=client,
    )


def test_count_retries_transient_timeout():
    client = _FakeClient(failures=2)
    assert _store(client).count() == 7
    assert client.count_calls == 3


def test_count_raises_after_retries_exhausted():
    client = _FakeClient(failures=3)
    with pytest.raises(_ReadTimeout):
        _store(client).count()
    assert client.count_calls == 3


def test_non_transient_error_is_not_retried():
    client = _FakeClient(failures=1, exc=ValueError("bad request"))
    with pytest.raises(ValueError):
        _store(client).count()
    assert client.count_calls == 1


def test_collection_exists_retries():
    client = _FakeClient(failures=1)
    assert _store(client).collection_exists() is True
    assert client.exists_calls == 2


def test_query_points_retries():
    client = _FakeClient(failures=1)
    store = _store(client)
    assert store.dense_search([0.0, 0.0], limit=3) == []
    assert client.query_calls == 2
