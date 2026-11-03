"""评测读取门面的降级测试：DB 优先、趋势文件回退、来源标注。

★ 为什么专门测降级
------------
数据库不可用时接口仍要 200，但**必须如实告诉前端**数字来自旧文件。
把“DB 挂了”静默当成“没有数据”，看板会显示空白而没人知道原因。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.eval_runs import service as SVC
from src.eval_runs.db import EvalStoreError


def _record(run_id="r1", *, value=0.9, verdict="pass", **overrides):
    record = {
        "run_id": run_id,
        "created_at": datetime(2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc),
        "version": "v1",
        "schema_version": 1,
        "source": "cli",
        "job_id": None,
        "suites": ["agent"],
        "suites_source": "observed",
        "verdict": verdict,
        "case_counts": {"agent": 5},
        "suite_verdicts": {"agent": verdict},
        "metrics": {
            "agent.tool_accuracy": {
                "value": value,
                "status": verdict,
                "threshold": 0.9,
                "note": "",
            }
        },
        "suite_notes": {},
        "datasets": {},
        "config": {},
    }
    record.update(overrides)
    return record


def test_list_runs_prefers_db(monkeypatch):
    monkeypatch.setattr(
        SVC.store, "list_records", lambda *a, **k: ([_record()], 1)
    )
    payload = SVC.list_runs()
    assert payload["total"] == 1
    assert payload["storage"]["backend"] == "db"
    assert payload["storage"]["degraded"] is False
    assert payload["runs"][0]["metric_count"] == 1


def test_list_runs_falls_back_when_db_empty(monkeypatch):
    monkeypatch.setattr(SVC.store, "list_records", lambda *a, **k: ([], 0))
    monkeypatch.setattr(SVC, "_trend_records", lambda: [_record("trend-1")])
    monkeypatch.setattr(SVC, "_file_available", lambda: True)

    payload = SVC.list_runs()

    assert payload["total"] == 1
    assert payload["runs"][0]["run_id"] == "trend-1"
    assert payload["storage"]["backend"] == "trend_file"
    assert payload["storage"]["reason"] == "db_empty"
    assert payload["storage"]["db_available"] is True


def test_list_runs_falls_back_when_db_unavailable(monkeypatch):
    def _boom(*args, **kwargs):
        raise EvalStoreError("down")

    monkeypatch.setattr(SVC.store, "list_records", _boom)
    monkeypatch.setattr(SVC, "_trend_records", lambda: [_record("trend-1")])
    monkeypatch.setattr(SVC, "_file_available", lambda: True)

    payload = SVC.list_runs()

    assert payload["storage"]["backend"] == "trend_file"
    assert payload["storage"]["reason"] == "db_unavailable"
    assert payload["storage"]["db_available"] is False
    assert payload["runs"][0]["run_id"] == "trend-1"


def test_list_runs_both_unavailable_returns_none_backend(monkeypatch):
    monkeypatch.setattr(
        SVC.store, "list_records", lambda *a, **k: (_ for _ in ()).throw(EvalStoreError("down"))
    )
    monkeypatch.setattr(SVC, "_trend_records", lambda: [])
    monkeypatch.setattr(SVC, "_file_available", lambda: False)

    payload = SVC.list_runs()

    assert payload["runs"] == []
    assert payload["storage"]["backend"] == "none"
    assert payload["storage"]["db_available"] is False


def test_list_runs_filters_and_paginates_fallback(monkeypatch):
    monkeypatch.setattr(SVC.store, "list_records", lambda *a, **k: (_ for _ in ()).throw(EvalStoreError("down")))
    monkeypatch.setattr(
        SVC,
        "_trend_records",
        lambda: [
            _record("r1", source="cli"),
            _record("r2", source="api"),
            _record("r3", source="cli"),
        ],
    )
    monkeypatch.setattr(SVC, "_file_available", lambda: True)

    payload = SVC.list_runs(limit=1, offset=1, source="cli")

    # 两条 cli 记录按 run_id 倒序为 r3、r1；offset=1 取到 r1。
    assert payload["total"] == 2
    assert payload["runs"][0]["run_id"] == "r1"
    assert payload["limit"] == 1
    assert payload["offset"] == 1


def test_get_run_db_hit_returns_detail(monkeypatch):
    monkeypatch.setattr(SVC.store, "get_record", lambda run_id: _record(run_id))
    payload = SVC.get_run("r1")
    assert payload["storage"]["backend"] == "db"
    assert payload["run"]["run_id"] == "r1"
    assert payload["run"]["metrics"][0]["metric"] == "agent.tool_accuracy"
    assert payload["run"]["metrics"][0]["is_ratio"] is True
    assert payload["run"]["created_at"] == "2026-10-31T15:21:32+00:00"


def test_get_run_falls_back_on_db_miss(monkeypatch):
    monkeypatch.setattr(SVC.store, "get_record", lambda run_id: None)
    monkeypatch.setattr(SVC, "_trend_records", lambda: [_record("trend-1")])
    payload = SVC.get_run("trend-1")
    assert payload["storage"]["backend"] == "trend_file"
    assert payload["storage"]["reason"] == "db_miss"


def test_get_run_returns_none_when_missing_everywhere(monkeypatch):
    monkeypatch.setattr(SVC.store, "get_record", lambda run_id: None)
    monkeypatch.setattr(SVC, "_trend_records", lambda: [])
    assert SVC.get_run("missing") is None


def test_compare_can_mix_db_and_trend_sources(monkeypatch):
    trend_record = _record("trend-1", value=0.8)

    def _get_record(run_id):
        return _record(run_id, value=0.9) if run_id == "db-1" else None

    monkeypatch.setattr(SVC.store, "get_record", _get_record)
    monkeypatch.setattr(SVC, "_trend_records", lambda: [trend_record])

    payload = SVC.compare_runs("db-1", "trend-1")

    assert payload["storage"]["base"]["backend"] == "db"
    assert payload["storage"]["head"]["backend"] == "trend_file"
    row = payload["metrics"][0]
    assert row["delta"] == pytest.approx(-0.1)


def test_compare_returns_none_if_either_missing(monkeypatch):
    monkeypatch.setattr(SVC.store, "get_record", lambda run_id: None)
    monkeypatch.setattr(SVC, "_trend_records", lambda: [_record("only-1")])
    assert SVC.compare_runs("only-1", "missing") is None
