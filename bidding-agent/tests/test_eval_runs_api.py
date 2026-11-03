"""/api/eval/runs、/runs/{id}、/compare 的接线测试。

用文件内 fake service 替换 `api.server.eval_runs_service`：
只验证参数透传、错误码与响应形状，不连数据库、不依赖真实历史。
"""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api import server


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def list_runs(self, limit=20, offset=0, suite=None, verdict=None, source=None):
        self.calls.append(("list", limit, offset, suite, verdict, source))
        return {
            "runs": [{"run_id": "r1"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
            "storage": {
                "backend": "db",
                "degraded": False,
                "reason": "",
                "db_available": True,
                "file_available": True,
            },
            "capabilities": {"per_case": False, "compare": True},
        }

    def get_run(self, run_id):
        self.calls.append(("get", run_id))
        if run_id == "missing":
            return None
        return {
            "run": {"run_id": run_id, "metrics": []},
            "storage": {
                "backend": "db",
                "degraded": False,
                "reason": "",
                "db_available": True,
                "file_available": True,
            },
        }

    def compare_runs(self, base, head, *, tolerance=0.02):
        self.calls.append(("compare", base, head, tolerance))
        if "missing" in (base, head):
            return None
        return {
            "base": {"run_id": base},
            "head": {"run_id": head},
            "metrics": [],
            "storage": {"base": {"backend": "db"}, "head": {"backend": "db"}},
        }

    def list_jobs(self, limit=20, offset=0, status=None):
        self.calls.append(("jobs", limit, offset, status))
        return {
            "jobs": [{"job_id": "j1"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
            "storage": {"backend": "db", "degraded": False},
            "capabilities": {"per_case": False, "compare": True},
        }

    def get_job(self, job_id):
        self.calls.append(("job", job_id))
        if job_id == "missing":
            return None
        return {
            "job": {"job_id": job_id, "status": "succeeded"},
            "storage": {"backend": "db", "degraded": False},
            "capabilities": {"per_case": False, "compare": True},
        }

    def db_storage(self):
        return {
            "backend": "db",
            "degraded": False,
            "reason": "",
            "db_available": True,
            "file_available": True,
        }


def _client(monkeypatch, fake=None) -> TestClient:
    monkeypatch.setattr(server, "eval_runs_service", fake or _FakeService())
    return TestClient(server.app)


def test_runs_forwards_filters_and_pagination(monkeypatch):
    fake = _FakeService()
    client = _client(monkeypatch, fake)

    response = client.get(
        "/api/eval/runs?limit=5&offset=2&suite=agent&verdict=pass&source=cli"
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert fake.calls == [("list", 5, 2, "agent", "pass", "cli")]


def test_runs_rejects_unknown_filters(monkeypatch):
    fake = _FakeService()
    client = _client(monkeypatch, fake)

    assert client.get("/api/eval/runs?suite=bogus").status_code == 400
    assert client.get("/api/eval/runs?verdict=maybe").status_code == 400
    assert client.get("/api/eval/runs?source=ftp").status_code == 400
    assert fake.calls == []  # 参数校验在调用 service 之前


def test_runs_rejects_invalid_pagination(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/api/eval/runs?limit=0").status_code == 422
    assert client.get("/api/eval/runs?limit=201").status_code == 422
    assert client.get("/api/eval/runs?offset=-1").status_code == 422


def test_run_detail_returns_payload(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/api/eval/runs/r1")
    assert response.status_code == 200
    assert response.json()["run"]["run_id"] == "r1"


def test_run_detail_missing_is_404(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/api/eval/runs/missing")
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


def test_compare_returns_payload(monkeypatch):
    fake = _FakeService()
    client = _client(monkeypatch, fake)

    response = client.get("/api/eval/compare?base=b&head=h&tolerance=0.05")

    assert response.status_code == 200
    assert response.json()["base"]["run_id"] == "b"
    assert fake.calls == [("compare", "b", "h", 0.05)]


def test_compare_requires_both_ids(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/api/eval/compare?base=b").status_code == 400
    assert client.get("/api/eval/compare").status_code == 400


def test_compare_missing_run_is_404(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/api/eval/compare?base=missing&head=h")
    assert response.status_code == 404


def test_service_exception_is_500_without_leaking_details(monkeypatch):
    class _Broken(_FakeService):
        def list_runs(self, *args, **kwargs):
            raise RuntimeError("postgres://secret-host")

    client = _client(monkeypatch, _Broken())
    response = client.get("/api/eval/runs")

    assert response.status_code == 500
    assert "secret-host" not in response.text


def test_existing_offline_endpoint_shape_is_unchanged(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/api/eval/offline")
    assert response.status_code == 200
    body = response.json()
    assert "capabilities" in body
    assert "series" in body
    assert body["capabilities"]["per_case"] is False


def test_jobs_list_forwards_filters(monkeypatch):
    fake = _FakeService()
    client = _client(monkeypatch, fake)
    response = client.get("/api/eval/jobs?limit=5&offset=1&status=running")
    assert response.status_code == 200
    assert fake.calls == [("jobs", 5, 1, "running")]


def test_jobs_list_rejects_unknown_status(monkeypatch):
    fake = _FakeService()
    client = _client(monkeypatch, fake)
    assert client.get("/api/eval/jobs?status=bogus").status_code == 400
    assert fake.calls == []


def test_job_detail_returns_payload_and_404(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/api/eval/jobs/j1").status_code == 200
    assert client.get("/api/eval/jobs/missing").status_code == 404


def test_job_detail_db_error_is_503(monkeypatch):
    class _Broken(_FakeService):
        def get_job(self, job_id):
            raise server.EvalStoreError("down")

    client = _client(monkeypatch, _Broken())
    assert client.get("/api/eval/jobs/j1").status_code == 503


def _trigger_client(monkeypatch, fake, key="secret"):
    monkeypatch.setenv("EVAL_TRIGGER_API_KEY", key)
    return _client(monkeypatch, fake), {"X-API-Key": key}


def test_trigger_requires_key(monkeypatch):
    monkeypatch.setenv("EVAL_TRIGGER_API_KEY", "secret")
    monkeypatch.delenv("EVAL_TRIGGER_ENABLED", raising=False)
    client = _client(monkeypatch)
    # TestClient 的 host 是 testclient（非本机），无 key 必须 403
    assert client.post("/api/eval/trigger", json={}).status_code == 403
    assert (
        client.post(
            "/api/eval/trigger", json={}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 403
    )


def test_trigger_disabled_is_403(monkeypatch):
    client, headers = _trigger_client(monkeypatch, _FakeService())
    monkeypatch.delenv("EVAL_TRIGGER_ENABLED", raising=False)
    response = client.post("/api/eval/trigger", json={}, headers=headers)
    assert response.status_code == 403
    assert "EVAL_TRIGGER_ENABLED" in response.json()["detail"]


def test_trigger_success_returns_202(monkeypatch):
    fake = _FakeService()
    client, headers = _trigger_client(monkeypatch, fake)
    monkeypatch.setenv("EVAL_TRIGGER_ENABLED", "1")
    monkeypatch.setattr(
        server.eval_runs_jobs,
        "trigger_job",
        lambda suites, limit=0, note="": {
            "job_id": "j1",
            "status": "queued",
            "suites": suites,
            "limit": limit,
            "note": note,
        },
    )

    response = client.post(
        "/api/eval/trigger",
        json={"suites": ["retrieval_offline"], "limit": 5, "note": "冒烟"},
        headers=headers,
    )

    assert response.status_code == 202
    assert response.json()["job"]["job_id"] == "j1"
    assert response.json()["storage"]["backend"] == "db"


def test_trigger_conflict_and_unavailable(monkeypatch):
    client, headers = _trigger_client(monkeypatch, _FakeService())
    monkeypatch.setenv("EVAL_TRIGGER_ENABLED", "1")

    def _conflict(*args, **kwargs):
        raise server.eval_runs_jobs.EvalTriggerConflict("busy")

    monkeypatch.setattr(server.eval_runs_jobs, "trigger_job", _conflict)
    assert (
        client.post("/api/eval/trigger", json={}, headers=headers).status_code
        == 409
    )

    def _unavailable(*args, **kwargs):
        raise server.eval_runs_jobs.EvalTriggerUnavailable("down")

    monkeypatch.setattr(server.eval_runs_jobs, "trigger_job", _unavailable)
    assert (
        client.post("/api/eval/trigger", json={}, headers=headers).status_code
        == 503
    )


def test_eval_endpoints_are_sync_def():
    assert not inspect.iscoroutinefunction(server.get_eval_runs)
    assert not inspect.iscoroutinefunction(server.get_eval_run)
    assert not inspect.iscoroutinefunction(server.get_eval_compare)
    assert not inspect.iscoroutinefunction(server.trigger_eval)
    assert not inspect.iscoroutinefunction(server.get_eval_jobs)
    assert not inspect.iscoroutinefunction(server.get_eval_job)
