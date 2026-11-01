"""/api/data/quality/latest 端点测试。"""

import inspect

from fastapi.testclient import TestClient

from api import server


class _FakeDataHub:
    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error

    def quality_latest(self):
        if self.error is not None:
            raise self.error
        return self.report


def _client(monkeypatch, fake) -> TestClient:
    monkeypatch.setattr(server, "datahub_service", fake, raising=False)
    return TestClient(server.app)


def test_quality_latest_returns_report(monkeypatch):
    report = {
        "success": True,
        "asset": "quality",
        "data": {"overall": "ok", "checks": [{"name": "x", "status": "pass"}]},
        "count": 1,
        "elapsed_ms": 2,
    }
    client = _client(monkeypatch, _FakeDataHub(report=report))
    response = client.get("/api/data/quality/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["overall"] == "ok"
    assert body["count"] == 1


def test_quality_latest_without_report_is_503(monkeypatch):
    client = _client(monkeypatch, _FakeDataHub(report=None))
    response = client.get("/api/data/quality/latest")
    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert "质量报告尚未生成" in body["error"]


def test_quality_latest_service_missing_is_503(monkeypatch):
    client = _client(monkeypatch, None)
    response = client.get("/api/data/quality/latest")
    assert response.status_code == 503
    assert response.json()["success"] is False


def test_quality_latest_unexpected_error_is_500(monkeypatch):
    client = _client(monkeypatch, _FakeDataHub(error=RuntimeError("boom")))
    response = client.get("/api/data/quality/latest")
    assert response.status_code == 500
    assert response.json()["success"] is False


def test_quality_latest_endpoint_is_sync_def():
    assert not inspect.iscoroutinefunction(server.data_quality_latest)
