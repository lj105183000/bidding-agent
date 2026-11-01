"""资产目录快照与漂移检测测试。

不连真库：通过注入 `pg_tables/qdrant_exists/neo4j_labels` 三个函数，
验证“缺表/缺集合/缺标签/后端错误”都被如实记录，且快照能持久化。
"""

from psycopg.types.json import Jsonb

from src.datahub import catalog_store as CS
from src.datahub.lineage import EDGES
from src.datahub.registry import ASSETS
from src.database.neo4j_client import ALLOWED_LABELS
from src.rag.regulation_store import REGULATION_COLLECTION


class _FakeCursor:
    def __init__(self, row=(9,)):
        self.row = row
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row


class _FakeTransaction:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


def _install_transaction(monkeypatch, cursor):
    monkeypatch.setattr(CS, "PGTransaction", lambda *a, **k: _FakeTransaction(cursor))


def test_snapshot_records_asset_count_and_drift():
    result = CS.save_snapshot(
        persist=False,
        pg_tables=lambda: [],
        qdrant_exists=lambda collection: True,
        neo4j_labels=lambda: list(ALLOWED_LABELS),
    )
    assert result["asset_count"] == len(ASSETS)
    assert len(result["snapshot"]["assets"]) == len(ASSETS)
    assert len(result["snapshot"]["edges"]) == len(EDGES)
    assert result["drift_summary"]["missing_pg_tables"]
    assert result["drift_summary"]["missing_collections"] == []
    assert result["drift_summary"]["missing_neo4j_labels"] == []
    assert result["drift_summary"]["errors"] == []


def test_snapshot_detects_missing_collection():
    def fake_exists(collection):
        return collection != REGULATION_COLLECTION

    result = CS.save_snapshot(
        persist=False,
        pg_tables=lambda: [],
        qdrant_exists=fake_exists,
        neo4j_labels=lambda: list(ALLOWED_LABELS),
    )
    assert "qdrant.bid_regulations" in result["drift_summary"]["missing_collections"]


def test_snapshot_records_backend_errors_without_raising():
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    result = CS.save_snapshot(
        persist=False,
        pg_tables=boom,
        qdrant_exists=boom,
        neo4j_labels=boom,
    )
    errors = result["drift_summary"]["errors"]
    assert len(errors) == 3
    assert any("PostgreSQL" in item for item in errors)
    assert any("Qdrant" in item for item in errors)
    assert any("Neo4j" in item for item in errors)


def test_snapshot_persists_jsonb(monkeypatch):
    cursor = _FakeCursor(row=(9,))
    monkeypatch.setattr(CS, "ensure_table", lambda: None)
    _install_transaction(monkeypatch, cursor)

    result = CS.save_snapshot(
        source="scheduler",
        persist=True,
        pg_tables=lambda: [],
        qdrant_exists=lambda collection: True,
        neo4j_labels=lambda: list(ALLOWED_LABELS),
    )
    assert result["snapshot_id"] == 9
    insert_sql, params = cursor.executed[0]
    assert "INSERT INTO data_catalog_snapshots" in insert_sql
    assert isinstance(params[2], Jsonb)
    assert isinstance(params[3], Jsonb)
    assert params[4] == "scheduler"


def test_ensure_table_creates_table_and_index(monkeypatch):
    cursor = _FakeCursor()
    _install_transaction(monkeypatch, cursor)
    CS.ensure_table()
    assert len(cursor.executed) == 2
    assert "CREATE TABLE IF NOT EXISTS data_catalog_snapshots" in cursor.executed[0][0]
    assert "CREATE INDEX IF NOT EXISTS" in cursor.executed[1][0]
