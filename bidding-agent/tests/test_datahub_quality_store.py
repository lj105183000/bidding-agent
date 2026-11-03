"""质量报告持久化测试。

用 fake 事务/游标捕获 SQL，不连 PostgreSQL：
    - 建表 + 索引是幂等的；
    - 写报告用 Jsonb，并按 retention 清理；
    - 表不存在 / 无数据时 latest 返回 None；
    - latest 把 report JSONB 扁平化，前端不必再拆一层。
"""

from datetime import datetime
from types import SimpleNamespace

from psycopg.types.json import Jsonb

from src.datahub import quality_store as QS


class _FakeCursor:
    def __init__(self, row=None):
        self.row = row
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row

    def fetchall(self):
        return []


class _FakeTransaction:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


class _FakeCursorContext:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


def _install_transaction(monkeypatch, cursor):
    monkeypatch.setattr(QS, "PGTransaction", lambda *a, **k: _FakeTransaction(cursor))


def test_ensure_table_creates_table_and_index(monkeypatch):
    cursor = _FakeCursor()
    _install_transaction(monkeypatch, cursor)
    QS.ensure_table()
    assert len(cursor.executed) == 2
    assert "CREATE TABLE IF NOT EXISTS data_quality_reports" in cursor.executed[0][0]
    assert "CREATE INDEX IF NOT EXISTS" in cursor.executed[1][0]


def test_save_report_uses_jsonb_and_prunes(monkeypatch):
    cursor = _FakeCursor(row=(7,))
    monkeypatch.setattr(QS, "ensure_table", lambda: None)
    monkeypatch.setattr(QS, "settings", SimpleNamespace(data_quality_retention_days=30))
    _install_transaction(monkeypatch, cursor)

    report = {
        "overall": "ok",
        "duration_ms": 12,
        "summary": {"pass": 1},
        "generated_at": "2026-11-01T00:00:00+00:00",
    }
    assert QS.save_report(report, source="scheduler") == 7

    insert_sql, insert_params = cursor.executed[0]
    assert "INSERT INTO data_quality_reports" in insert_sql
    assert isinstance(insert_params[3], Jsonb)
    assert isinstance(insert_params[4], Jsonb)
    assert insert_params[5] == "scheduler"

    prune_sql, prune_params = cursor.executed[1]
    assert "DELETE FROM data_quality_reports" in prune_sql
    assert "make_interval" in prune_sql
    assert prune_params == (30,)


def test_save_report_skips_prune_when_retention_zero(monkeypatch):
    cursor = _FakeCursor(row=(8,))
    monkeypatch.setattr(QS, "ensure_table", lambda: None)
    monkeypatch.setattr(QS, "settings", SimpleNamespace(data_quality_retention_days=0))
    _install_transaction(monkeypatch, cursor)

    QS.save_report({"overall": "ok", "duration_ms": 1, "summary": {}})
    assert len(cursor.executed) == 1  # 只有 INSERT，没有 DELETE


def test_latest_report_returns_none_on_table_missing(monkeypatch):
    class _BrokenCursor(_FakeCursor):
        def execute(self, sql, params=()):
            raise RuntimeError("relation does not exist")

    monkeypatch.setattr(
        QS, "PGCursor", lambda *a, **k: _FakeCursorContext(_BrokenCursor())
    )
    assert QS.latest_report() is None


def test_latest_report_returns_none_when_empty(monkeypatch):
    cursor = _FakeCursor(row=None)
    monkeypatch.setattr(QS, "PGCursor", lambda *a, **k: _FakeCursorContext(cursor))
    assert QS.latest_report() is None


def test_latest_report_flattens_report(monkeypatch):
    cursor = _FakeCursor(
        row={
            "id": 7,
            "generated_at": datetime(2026, 1, 2, 3, 4, 5),
            "overall": "ok",
            "duration_ms": 5,
            "summary": {"pass": 1},
            "report": {"overall": "ok", "checks": [{"name": "x"}]},
            "source": "scheduler",
        }
    )
    monkeypatch.setattr(QS, "PGCursor", lambda *a, **k: _FakeCursorContext(cursor))
    result = QS.latest_report()
    assert result["overall"] == "ok"
    assert result["checks"] == [{"name": "x"}]
    assert result["report_id"] == 7
    assert result["source"] == "scheduler"
    assert result["generated_at"] == "2026-01-02T03:04:05"
