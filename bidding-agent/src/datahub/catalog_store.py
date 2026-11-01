"""资产目录快照与漂移检测。

目录本身是静态声明（`registry.py` / `lineage.py`），快照任务把“声明的”
与“实库的”对照一次，落进 `data_catalog_snapshots`：

    - 缺 PG 表 / 缺 Qdrant 集合 → drift 里如实记录（任务只告警，不阻断）；
    - 缺 Neo4j 标签 → 同样记录；
    - 某个后端连不上 → 记在 `errors`，其余后端继续检查。

为什么不在 API 进程里做：快照要连三套存储，可能有秒级延迟；放进独立调度
进程，目录端点才能保持“不连库、快速返回”。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from psycopg.types.json import Jsonb

from src.config import settings
from src.datahub import neo4j as neo4j_adapter
from src.datahub import qdrant as qdrant_adapter
from src.datahub.contract import DataHubQueryError
from src.datahub.lineage import EDGES
from src.datahub.pg import PGTransaction, query_dicts
from src.datahub.registry import ASSETS, asset_to_dict
from src.database.neo4j_client import ALLOWED_LABELS
from src.rag.regulation_store import REGULATION_COLLECTION

logger = logging.getLogger(__name__)

TABLE = "data_catalog_snapshots"

_DDL = (
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id           BIGSERIAL PRIMARY KEY,
        generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        asset_count  INTEGER NOT NULL DEFAULT 0,
        snapshot     JSONB NOT NULL,
        drift_summary JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        source       TEXT NOT NULL DEFAULT 'manual'
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_generated_at "
    f"ON {TABLE} (generated_at DESC)",
)


def ensure_table() -> None:
    with PGTransaction(DataHubQueryError) as cur:
        for statement in _DDL:
            cur.execute(statement)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pg_tables() -> list[str]:
    rows = query_dicts(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public'"
    )
    return [str(row.get("table_name")) for row in rows]


def _default_qdrant_exists(collection: str) -> bool:
    return qdrant_adapter.collection_exists(collection)


def _default_neo4j_labels() -> list[str]:
    rows = neo4j_adapter.run(
        "CALL db.labels() YIELD label RETURN label",
        raise_on_error=True,
        error_cls=DataHubQueryError,
    )
    return [str(row.get("label")) for row in rows]


def save_snapshot(
    source: str = "manual",
    persist: bool = True,
    *,
    pg_tables: Callable[[], list[str]] | None = None,
    qdrant_exists: Callable[[str], bool] | None = None,
    neo4j_labels: Callable[[], list[str]] | None = None,
) -> dict[str, Any]:
    """生成一份目录快照并（默认）持久化。

    后端探测失败不会抛异常：失败信息写进 `drift_summary.errors`，
    让“目录快照”本身始终能产出——这比因为 Neo4j 抖动就整份快照丢失有用。
    """
    snapshot = {
        "assets": [asset_to_dict(asset, include_fields=False) for asset in ASSETS],
        "edges": [asdict(edge) for edge in EDGES],
    }
    drift: dict[str, Any] = {
        "missing_pg_tables": [],
        "missing_collections": [],
        "missing_neo4j_labels": [],
        "errors": [],
    }

    try:
        actual_tables = set((pg_tables or _default_pg_tables)())
    except Exception as exc:  # noqa: BLE001 - 记录后继续检查其他后端
        drift["errors"].append(f"PostgreSQL: {type(exc).__name__}")
    else:
        expected_tables = {
            asset.name.split(".", 1)[1]
            for asset in ASSETS
            if asset.kind == "postgres_table" and asset.status == "active"
        }
        drift["missing_pg_tables"] = sorted(expected_tables - actual_tables)

    expected_collections = {
        "qdrant.bid_qa": settings.qdrant_collection,
        "qdrant.bid_regulations": REGULATION_COLLECTION,
    }
    for asset_name, collection in expected_collections.items():
        try:
            exists = (qdrant_exists or _default_qdrant_exists)(collection)
        except Exception as exc:  # noqa: BLE001
            drift["errors"].append(f"Qdrant: {type(exc).__name__}")
            break
        if not exists:
            drift["missing_collections"].append(asset_name)

    try:
        actual_labels = set((neo4j_labels or _default_neo4j_labels)())
    except Exception as exc:  # noqa: BLE001
        drift["errors"].append(f"Neo4j: {type(exc).__name__}")
    else:
        drift["missing_neo4j_labels"] = sorted(ALLOWED_LABELS - actual_labels)

    result: dict[str, Any] = {
        "generated_at": _now_iso(),
        "asset_count": len(ASSETS),
        "drift_summary": drift,
        "snapshot": snapshot,
    }

    if persist:
        ensure_table()
        with PGTransaction(DataHubQueryError) as cur:
            cur.execute(
                f"""
                INSERT INTO {TABLE}
                    (generated_at, asset_count, snapshot, drift_summary, source)
                VALUES
                    (COALESCE(%s::timestamptz, now()), %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    result["generated_at"],
                    result["asset_count"],
                    Jsonb(snapshot),
                    Jsonb(drift),
                    source,
                ),
            )
            row = cur.fetchone()
            result["snapshot_id"] = int(row[0]) if row else 0

    logger.info(
        "目录快照完成（source=%s, assets=%d, missing_pg=%d, missing_collections=%d, errors=%d）",
        source,
        result["asset_count"],
        len(drift["missing_pg_tables"]),
        len(drift["missing_collections"]),
        len(drift["errors"]),
    )
    return result
