"""手动触发评测的任务编排测试（不启动真实子进程）。

覆盖：开关、suite 白名单、单任务/冷却冲突、DB 不可用、
子进程退出码/超时/未落库，以及 argv 与 env 的构造。
"""

from __future__ import annotations

import subprocess
import time

import pytest

from src.eval_runs import jobs as J
from src.eval_runs.db import EvalStoreError


class _FakeThread:
    def __init__(self, target=None, args=(), daemon=None, name=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True


class _FakeProc:
    def __init__(self, code=0, timeout=False):
        self.code = code
        self.timeout = timeout
        self.killed = False

    def wait(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return self.code

    def kill(self):
        self.killed = True


@pytest.fixture(autouse=True)
def _clean_state():
    J.reset_state()
    yield
    J.reset_state()


def _enable(monkeypatch):
    monkeypatch.setenv("EVAL_TRIGGER_ENABLED", "1")


def _patch_store(monkeypatch, *, active=None, create_error=None, has_run=True):
    calls = {"reconcile": [], "create": [], "update": [], "has_run": []}

    def _reconcile(seconds):
        calls["reconcile"].append(seconds)
        return 0

    def _create(job):
        if create_error is not None:
            raise create_error
        calls["create"].append(job)

    def _update(job_id, **fields):
        calls["update"].append((job_id, fields))
        return True

    def _has(run_id):
        calls["has_run"].append(run_id)
        return has_run

    monkeypatch.setattr(J.store, "reconcile_stale_jobs", _reconcile)
    monkeypatch.setattr(J.store, "active_job", lambda: active)
    monkeypatch.setattr(J.store, "create_job", _create)
    monkeypatch.setattr(J.store, "update_job", _update)
    monkeypatch.setattr(J.store, "has_run", _has)
    return calls


def test_trigger_disabled_is_403(monkeypatch):
    monkeypatch.delenv("EVAL_TRIGGER_ENABLED", raising=False)
    with pytest.raises(J.EvalTriggerDisabled) as exc:
        J.trigger_job(["retrieval_offline"])
    assert exc.value.status_code == 403


def test_unknown_suite_rejected(monkeypatch):
    _enable(monkeypatch)
    with pytest.raises(J.EvalTriggerError, match="只允许触发"):
        J.trigger_job(["rag"])


def test_empty_and_bad_limit_rejected(monkeypatch):
    _enable(monkeypatch)
    with pytest.raises(J.EvalTriggerError, match="至少选择"):
        J.trigger_job([])
    with pytest.raises(J.EvalTriggerError, match="limit"):
        J.trigger_job(["retrieval_offline"], limit=201)


def test_db_unavailable_is_503(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(
        J.store,
        "reconcile_stale_jobs",
        lambda seconds: (_ for _ in ()).throw(EvalStoreError("down")),
    )
    with pytest.raises(J.EvalTriggerUnavailable) as exc:
        J.trigger_job(["retrieval_offline"])
    assert exc.value.status_code == 503


def test_active_job_conflict(monkeypatch):
    _enable(monkeypatch)
    _patch_store(monkeypatch, active={"job_id": "running-1"})
    with pytest.raises(J.EvalTriggerConflict, match="running-1"):
        J.trigger_job(["retrieval_offline"])


def test_cooldown_conflict(monkeypatch):
    _enable(monkeypatch)
    _patch_store(monkeypatch)
    monkeypatch.setattr(J, "_last_requested_at", time.monotonic())
    monkeypatch.setenv("EVAL_TRIGGER_MIN_INTERVAL_SECONDS", "60")
    with pytest.raises(J.EvalTriggerConflict, match="过于频繁"):
        J.trigger_job(["retrieval_offline"])


def test_success_creates_job_and_starts_thread(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch)
    threads = []

    class _Thread(_FakeThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

    monkeypatch.setattr(J.threading, "Thread", _Thread)
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("EVAL_TRIGGER_TIMEOUT_SECONDS", "600")

    job = J.trigger_job(["retrieval_offline"], limit=5, note="冒烟")

    assert job["status"] == "queued"
    assert job["run_id"] == job["job_id"]
    assert job["limit"] == 5
    assert job["note"] == "冒烟"
    assert calls["reconcile"] == [600 + J.STALE_GRACE_SECONDS]
    assert calls["create"][0]["job_id"] == job["job_id"]
    assert threads[0].started is True
    assert (tmp_path / "logs/eval_jobs").exists()


def test_run_job_success_updates_status(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch, has_run=True)
    captured = {}

    class _Popen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            captured["shell"] = kwargs["shell"]
            captured["cwd"] = kwargs["cwd"]

        def wait(self, timeout=None):
            return 0

        def kill(self):
            captured["killed"] = True

    monkeypatch.setattr(J.subprocess, "Popen", _Popen)
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)
    job = {
        "job_id": "job-1",
        "suites": ["retrieval_offline"],
        "limit": 3,
        "log_path": "logs/eval_jobs/job-1.log",
    }

    J._run_job(job)

    assert "eval/run_all.py" in captured["command"]
    assert "--save-db" in captured["command"]
    assert "--limit" in captured["command"]
    assert captured["env"]["EVAL_RUN_ID"] == "job-1"
    assert captured["env"]["EVAL_RUN_SOURCE"] == "api"
    assert captured["shell"] is False
    assert calls["update"][0][1]["status"] == "running"
    assert calls["update"][-1][1]["status"] == "succeeded"
    assert calls["update"][-1][1]["exit_code"] == 0
    assert calls["has_run"] == ["job-1"]


def test_run_job_timeout_marks_failed(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch)
    proc = _FakeProc(timeout=True)
    monkeypatch.setattr(J.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)

    J._run_job(
        {"job_id": "job-2", "suites": ["retrieval_offline"], "log_path": "logs/x.log"}
    )

    assert proc.killed is True
    assert calls["update"][-1][1]["status"] == "failed"
    assert "超时" in calls["update"][-1][1]["error"]


def test_run_job_bad_exit_code_marks_failed(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch)
    monkeypatch.setattr(J.subprocess, "Popen", lambda *a, **k: _FakeProc(code=3))
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)

    J._run_job(
        {"job_id": "job-3", "suites": ["retrieval_offline"], "log_path": "logs/x.log"}
    )

    assert calls["update"][-1][1]["status"] == "failed"
    assert "exit_code=3" in calls["update"][-1][1]["error"]


def test_run_job_without_db_record_marks_failed(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch, has_run=False)
    monkeypatch.setattr(J.subprocess, "Popen", lambda *a, **k: _FakeProc(code=0))
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)

    J._run_job(
        {"job_id": "job-4", "suites": ["retrieval_offline"], "log_path": "logs/x.log"}
    )

    assert calls["update"][-1][1]["status"] == "failed"
    assert "未写入数据库" in calls["update"][-1][1]["error"]


def test_run_job_popen_exception_marks_failed(monkeypatch, tmp_path):
    _enable(monkeypatch)
    calls = _patch_store(monkeypatch)

    def _boom(*args, **kwargs):
        raise OSError("no python")

    monkeypatch.setattr(J.subprocess, "Popen", _boom)
    monkeypatch.setattr(J, "PROJECT_ROOT", tmp_path)

    J._run_job(
        {"job_id": "job-5", "suites": ["retrieval_offline"], "log_path": "logs/x.log"}
    )

    assert calls["update"][-1][1]["status"] == "failed"
    assert calls["update"][-1][1]["error"]


def test_trigger_status_reflects_env(monkeypatch):
    monkeypatch.delenv("EVAL_TRIGGER_ENABLED", raising=False)
    assert J.trigger_status()["enabled"] is False
    monkeypatch.setenv("EVAL_TRIGGER_ENABLED", "1")
    assert J.trigger_status()["enabled"] is True
