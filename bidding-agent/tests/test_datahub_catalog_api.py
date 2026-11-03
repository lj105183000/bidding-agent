"""/api/data/* 目录 / 血缘端点接线测试。

用文件内 `_FakeDataHub` 替换 `api.server.datahub_service`：
不连数据库、不依赖注册表内容，只验证 API 层的参数透传、
错误码与 Envelope 形状（真注册表由 test_datahub_registry.py 覆盖）。
"""

import inspect

from fastapi.testclient import TestClient

from api import server


class _FakeDataHub:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def catalog(self, kind=None, layer=None):
        self.calls.append(("catalog", kind, layer))
        if kind == "not-a-kind":
            raise ValueError("未知资产类型：not-a-kind")
        return {
            "success": True,
            "asset": "catalog",
            "results": [{"name": "pg.bidding_records", "kind": "postgres_table"}],
            "count": 1,
            "elapsed_ms": 1,
            "cached": False,
            "cache_ttl_seconds": 60,
        }

    def asset(self, name):
        self.calls.append(("asset", name))
        if name not in ("pg.bidding_records", "bidding_records"):
            return None
        return {
            "success": True,
            "asset": "pg.bidding_records",
            "data": {"name": "pg.bidding_records", "fields": []},
            "elapsed_ms": 1,
        }

    def lineage(self, asset=None, direction="both", depth=2):
        self.calls.append(("lineage", asset, direction, depth))
        if asset == "not.exists":
            return None
        if direction not in ("both", "upstream", "downstream"):
            raise ValueError("direction 只能是 both / upstream / downstream")
        return {
            "success": True,
            "asset": asset or "lineage",
            "data": {"nodes": [], "edges": [], "count": 0},
            "count": 0,
            "elapsed_ms": 1,
        }


def _client(monkeypatch, fake) -> TestClient:
    monkeypatch.setattr(server, "datahub_service", fake, raising=False)
    return TestClient(server.app)


def test_catalog_returns_envelope(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 1
    assert body["results"][0]["name"] == "pg.bidding_records"
    assert fake.calls == [("catalog", None, None)]


def test_catalog_passes_filters(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/catalog", params={"kind": "postgres_table", "layer": "serving"})
    assert response.status_code == 200
    assert fake.calls == [("catalog", "postgres_table", "serving")]


def test_catalog_invalid_kind_is_400(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/catalog", params={"kind": "not-a-kind"})
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "未知资产类型" in body["error"]


def test_asset_detail_and_short_name(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/catalog/bidding_records")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["name"] == "pg.bidding_records"
    assert fake.calls == [("asset", "bidding_records")]


def test_asset_unknown_is_404(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/catalog/not.exists")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert "未知数据资产" in body["error"]


def test_lineage_passes_direction_and_depth(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get(
        "/api/data/lineage",
        params={"asset": "pg.bidding_records", "direction": "upstream", "depth": 3},
    )
    assert response.status_code == 200
    assert fake.calls == [("lineage", "pg.bidding_records", "upstream", 3)]


def test_lineage_invalid_direction_is_400(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/lineage", params={"direction": "sideways"})
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "direction" in body["error"]


def test_lineage_unknown_asset_is_404(monkeypatch):
    fake = _FakeDataHub()
    client = _client(monkeypatch, fake)
    response = client.get("/api/data/lineage", params={"asset": "not.exists"})
    assert response.status_code == 404


def test_datahub_unavailable_is_503(monkeypatch):
    client = _client(monkeypatch, None)
    for path in ("/api/data/catalog", "/api/data/catalog/bidding_records", "/api/data/lineage"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["success"] is False


def test_datahub_endpoints_are_sync_def():
    """项目约定：全部端点用同步 def，FastAPI 自动放线程池。"""
    assert not inspect.iscoroutinefunction(server.data_catalog)
    assert not inspect.iscoroutinefunction(server.data_catalog_asset)
    assert not inspect.iscoroutinefunction(server.data_lineage)
