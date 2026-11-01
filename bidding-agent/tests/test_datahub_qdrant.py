"""Qdrant 只读适配器测试。

只验证“数点数 / 判断集合存在”与失败映射；不启动 Qdrant。
"""

import pytest

from src.datahub import qdrant as Q
from src.datahub.contract import DataHubUnavailableError


class _FakeClient:
    def __init__(self, exists=True):
        self.exists = exists

    def collection_exists(self, collection):
        return self.exists


class _FakeStore:
    def __init__(self, points=42, exists=True, error=None):
        self.points = points
        self.client = _FakeClient(exists)
        self.error = error

    def count(self):
        if self.error is not None:
            raise self.error
        return self.points


def test_count_uses_cached_store(monkeypatch):
    monkeypatch.setattr(Q, "_store", lambda collection: _FakeStore(points=42))
    assert Q.count("bid_qa") == 42


def test_count_maps_failure(monkeypatch):
    monkeypatch.setattr(
        Q, "_store", lambda collection: _FakeStore(error=RuntimeError("down"))
    )
    with pytest.raises(DataHubUnavailableError) as exc:
        Q.count("bid_qa")
    assert "Qdrant 查询失败" in str(exc.value)


def test_collection_exists(monkeypatch):
    monkeypatch.setattr(Q, "_store", lambda collection: _FakeStore(exists=True))
    assert Q.collection_exists("bid_qa") is True
    monkeypatch.setattr(Q, "_store", lambda collection: _FakeStore(exists=False))
    assert Q.collection_exists("bid_qa") is False


def test_collection_exists_maps_failure(monkeypatch):
    class _BrokenStore:
        client = None

        def __init__(self):
            self.client = _BrokenClient()

    class _BrokenClient:
        def collection_exists(self, collection):
            raise RuntimeError("down")

    monkeypatch.setattr(Q, "_store", lambda collection: _BrokenStore())
    with pytest.raises(DataHubUnavailableError):
        Q.collection_exists("bid_qa")


def test_known_collections_contains_regulations():
    collections = Q.known_collections()
    assert collections["regulations"] == "bid_regulations"
    assert "qa" in collections
