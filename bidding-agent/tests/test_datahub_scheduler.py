"""数据中台调度器测试。

重点：
    - 任务白名单固定，绝不注册 ingest / pg / graph 等破坏性任务；
    - cron 写错只跳过该任务，不影响其他任务；
    - --list 不执行任务；
    - --run-once 的失败返回非零，但不抛异常。
"""

import pytest

from src.datahub import scheduler as S


def _fake_job(job_id="datahub.quality_scan", result=None, error=None):
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if error is not None:
            raise error
        return dict(result or {"ok": True})

    return (
        S.JobDef(job_id, "data_scheduler_quality_cron", "fake", run),
        calls,
    )


def test_job_registry_has_only_whitelisted_jobs():
    assert {job.job_id for job in S.JOB_DEFS} == {
        "datahub.regulations_incremental",
        "datahub.quality_scan",
        "datahub.catalog_snapshot",
    }


def test_no_destructive_jobs():
    for job in S.JOB_DEFS:
        assert "ingest" not in job.job_id
        assert "rebuild" not in job.job_id
        assert "clear" not in job.job_id


def test_build_scheduler_registers_whitelisted_jobs():
    scheduler = S.build_scheduler(start=False)
    assert {job.id for job in scheduler.get_jobs()} == {
        job.job_id for job in S.JOB_DEFS
    }


def test_invalid_cron_skips_only_that_job(monkeypatch):
    monkeypatch.setattr(
        S.settings, "data_scheduler_quality_cron", "not a cron", raising=False
    )
    scheduler = S.build_scheduler(start=False)
    ids = {job.id for job in scheduler.get_jobs()}
    assert "datahub.quality_scan" not in ids
    assert "datahub.regulations_incremental" in ids
    assert "datahub.catalog_snapshot" in ids


def test_invalid_timezone_falls_back(monkeypatch):
    monkeypatch.setattr(
        S.settings, "data_scheduler_timezone", "Not/AZone", raising=False
    )
    assert str(S._timezone()) == "Asia/Shanghai"


def test_disabled_scheduler_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(
        S.settings, "data_scheduler_enabled", False, raising=False
    )
    assert S.run_scheduler([]) == 1
    assert "未启用" in capsys.readouterr().out


def test_list_does_not_run_jobs(monkeypatch, capsys):
    fake, calls = _fake_job(error=RuntimeError("should not run"))
    monkeypatch.setattr(S, "JOB_DEFS", (fake,))
    assert S.run_scheduler(["--list"]) == 0
    assert calls["n"] == 0
    assert "datahub.quality_scan" in capsys.readouterr().out


def test_run_once_quality(monkeypatch, capsys):
    fake, calls = _fake_job(result={"overall": "ok"})
    monkeypatch.setattr(S, "JOB_DEFS", (fake,))
    assert S.run_scheduler(["--run-once", "quality"]) == 0
    assert calls["n"] == 1
    assert '"ok": true' in capsys.readouterr().out


def test_run_once_all_runs_every_job(monkeypatch):
    first, first_calls = _fake_job("datahub.quality_scan")
    second, second_calls = _fake_job(
        "datahub.catalog_snapshot", result={"snapshot_id": 1}
    )
    monkeypatch.setattr(S, "JOB_DEFS", (first, second))
    assert S.run_scheduler(["--run-once", "all"]) == 0
    assert first_calls["n"] == 1
    assert second_calls["n"] == 1


def test_run_once_unknown_returns_nonzero(capsys):
    assert S.run_scheduler(["--run-once", "nope"]) == 1
    assert "未知任务" in capsys.readouterr().out


def test_job_failure_returns_nonzero_without_raising(monkeypatch, capsys):
    fake, _calls = _fake_job(error=RuntimeError("boom"))
    monkeypatch.setattr(S, "JOB_DEFS", (fake,))
    assert S.run_scheduler(["--run-once", "quality"]) == 1
    output = capsys.readouterr().out
    assert '"ok": false' in output
    assert "RuntimeError" in output
