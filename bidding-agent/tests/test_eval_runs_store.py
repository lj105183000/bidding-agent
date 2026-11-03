"""`eval_runs` / `eval_jobs` 存储层测试（不连 PostgreSQL）。

用文件内 fake 游标捕获 SQL：
    - 建表幂等；
    - 写入用 Jsonb、重复 run_id 不覆盖、`--overwrite` 只碰 backfill；
    - 列表筛选与分页参数正确；
    - 数据库失败后熔断，第二次请求不再等待连接超时；
    - job 更新只允许白名单字段。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from src.eval_runs import db
from src.eval_runs import store as S


class _FakeCursor:
    def __init__(self, *, rows=None, row=None):
        self._rows = list(rows or [])
        self._row = row
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows = self._rows
        self._rows = []
        return rows


class _Ctx:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


def _install(monkeypatch, cursor):
    monkeypatch.setattr(S, "PGTransaction", lambda *a, **k: _Ctx(cursor))
    monkeypatch.setattr(S, "PGCursor", lambda *a, **k: _Ctx(cursor))
    monkeypatch.setattr(S.db, "ensure_schema", lambda: None)


def _record(run_id: str = "r1") -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime(2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc),
        "version": "v1",
        "schema_version": 1,
        "source": "cli",
        "job_id": None,
        "suites": ["agent"],
        "suites_source": "observed",
        "verdict": "pass",
        "case_counts": {"agent": 5},
        "suite_verdicts": {"agent": "pass"},
        "metrics": {"agent.tool_accuracy": {"value": 1.0, "status": "pass"}},
        "suite_notes": {},
        "datasets": {},
        "config": {"LLM_PROVIDER": "deepseek"},
    }


@pytest.fixture(autouse=True)
def _reset_breaker():
    db.reset_breaker()
    yield
    db.reset_breaker()


def test_ensure_schema_is_idempotent(monkeypatch):
    monkeypatch.setattr(db, "_schema_ready", False)
    cursor = _FakeCursor()
    monkeypatch.setattr(db, "PGTransaction", lambda *a, **k: _Ctx(cursor))

    db.ensure_schema()
    assert len(cursor.executed) == len(db._DDL)
    assert "CREATE TABLE IF NOT EXISTS eval_runs" in cursor.executed[0][0]
    assert "CREATE TABLE IF NOT EXISTS eval_jobs" in cursor.executed[3][0]

    db.ensure_schema()
    assert len(cursor.executed) == len(db._DDL)  # 第二次不再执行 DDL


def test_save_records_inserts_jsonb_and_skips_duplicates(monkeypatch):
    cursor = _FakeCursor(rows=[("r1",)])
    _install(monkeypatch, cursor)

    counts = S.save_records([_record("r1")])

    assert counts == {"inserted": 1, "skipped": 0, "overwritten": 0, "protected": 0}
    sql, params = cursor.executed[0]
    assert "ON CONFLICT (run_id) DO NOTHING" in sql
    assert params[0] == "r1"
    assert params[4] == "cli"
    assert isinstance(params[9], Jsonb)   # case_counts
    assert isinstance(params[11], Jsonb)  # metrics
    assert isinstance(params[14], Jsonb)  # config


def test_save_records_overwrite_only_backfill(monkeypatch):
    cursor = _FakeCursor(rows=[None, ("r1",)])
    _install(monkeypatch, cursor)

    counts = S.save_records([_record("r1")], overwrite_backfill=True)

    assert counts["overwritten"] == 1
    update_sql, update_params = cursor.executed[1]
    assert "source = 'backfill'" in update_sql
    assert update_params[-1] == "r1"
    # UPDATE 不更新 source（params[4] 被跳过），占位符数量与参数一致
    assert update_sql.count("%s") == len(update_params)


def test_save_records_protects_realtime_rows(monkeypatch):
    cursor = _FakeCursor(rows=[None, None])
    _install(monkeypatch, cursor)

    counts = S.save_records([_record("r1")], overwrite_backfill=True)
    assert counts["protected"] == 1


def test_save_records_skips_when_overwrite_disabled(monkeypatch):
    cursor = _FakeCursor(rows=[None])
    _install(monkeypatch, cursor)

    counts = S.save_records([_record("r1")])
    assert counts["skipped"] == 1
    assert len(cursor.executed) == 1


def test_list_records_builds_filters_and_pagination(monkeypatch):
    cursor = _FakeCursor(
        row={"total": 1},
        rows=[
            {
                "run_id": "r1",
                "created_at": datetime(2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc),
                "version": "v1",
                "suites": ["agent"],
                "verdict": "pass",
            }
        ],
    )
    _install(monkeypatch, cursor)

    rows, total = S.list_records(5, 10, suite="agent", verdict="pass")

    assert total == 1
    assert rows[0]["created_at"] == "2026-10-31T15:21:32+00:00"
    count_sql, count_params = cursor.executed[0]
    assert "%s = ANY(suites)" in count_sql
    assert count_params == ("agent", "pass")
    select_sql, select_params = cursor.executed[1]
    assert "LIMIT %s OFFSET %s" in select_sql
    assert select_params == ("agent", "pass", 5, 10)


def test_get_record_returns_none_when_missing(monkeypatch):
    cursor = _FakeCursor(row=None)
    _install(monkeypatch, cursor)
    assert S.get_record("missing") is None


def test_db_failure_trips_breaker_and_skips_second_call(monkeypatch):
    _install(monkeypatch, _FakeCursor())
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("database down")

    monkeypatch.setattr(S, "PGTransaction", _boom)

    with pytest.raises(db.EvalStoreError):
        S.save_records([_record("r1")])
    assert calls["n"] == 1
    assert not db.is_available()

    # 熔断窗口内不再尝试连接（否则每个请求都要等 10 秒超时）
    with pytest.raises(db.EvalStoreError, match="暂不可用"):
        S.save_records([_record("r2")])
    assert calls["n"] == 1


def test_update_job_whitelists_fields(monkeypatch):
    cursor = _FakeCursor(row=("j1",))
    _install(monkeypatch, cursor)

    assert S.update_job("j1", status="running", evil="drop table") is True

    sql, params = cursor.executed[0]
    assert "status = %s" in sql
    assert "evil" not in sql
    assert params == ("running", "j1")


def test_create_job_and_reconcile(monkeypatch):
    cursor = _FakeCursor(rows=[("j1",), ("j2",)])
    _install(monkeypatch, cursor)

    S.create_job(
        {
            "job_id": "j1",
            "status": "queued",
            "suites": ["retrieval_offline"],
            "note": "冒烟",
            "run_id": "j1",
            "log_path": "logs/eval_jobs/j1.log",
        }
    )
    insert_sql, insert_params = cursor.executed[0]
    assert "INSERT INTO eval_jobs" in insert_sql
    assert insert_params[0] == "j1"
    assert insert_params[2] == ["retrieval_offline"]

    assert S.reconcile_stale_jobs(600) == 2
    reconcile_sql, reconcile_params = cursor.executed[1]
    assert "make_interval(secs => %s)" in reconcile_sql
    assert reconcile_params == (600,)
