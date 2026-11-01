"""数据质量报告的 PostgreSQL 持久化。

为什么用 PG 表而不是 JSON 文件
-----------------------------
1. scheduler 与 API 是**两个进程**：PG 天然共享；JSON 文件需要原子写/文件锁，
   跨进程并发写很容易留下半截文件。
2. 可直接按时间查历史、按 retention SQL 清理；JSON 只能全量读。
3. 与 `session_traces` / `tender_documents` 的“幂等建表 + 服务自建表”模式一致，
   不需要使用者先手工执行 DDL。

表不存在 / 无数据时 `latest_report()` 返回 None，由 API 转 503；
写入失败向上抛 `DataHubQueryError`，由任务层记录并继续（不拖垮其他规则）。
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg.types.json import Jsonb

from src.config import settings
from src.datahub.contract import DataHubQueryError
from src.datahub.pg import PGCursor, PGTransaction
from src.database.postgresql_client import _jsonable

logger = logging.getLogger(__name__)

TABLE = "data_quality_reports"

_DDL = (
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id           BIGSERIAL PRIMARY KEY,
        generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        overall      TEXT NOT NULL,
        duration_ms  INTEGER NOT NULL DEFAULT 0,
        summary      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        report       JSONB NOT NULL,
        source       TEXT NOT NULL DEFAULT 'manual'
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_generated_at "
    f"ON {TABLE} (generated_at DESC)",
)


def ensure_table() -> None:
    """幂等建表 + 索引；失败抛 DataHubQueryError。"""
    with PGTransaction(DataHubQueryError) as cur:
        for statement in _DDL:
            cur.execute(statement)


def save_report(report: dict[str, Any], source: str = "manual") -> int:
    """写入一份质量报告并清理过期行，返回新报告 id。"""
    ensure_table()
    overall = str(report.get("overall") or "unknown")
    duration_ms = int(report.get("duration_ms") or 0)
    summary = report.get("summary") or {}
    generated_at = report.get("generated_at")

    with PGTransaction(DataHubQueryError) as cur:
        cur.execute(
            f"""
            INSERT INTO {TABLE}
                (generated_at, overall, duration_ms, summary, report, source)
            VALUES
                (COALESCE(%s::timestamptz, now()), %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (generated_at, overall, duration_ms, Jsonb(summary), Jsonb(report), source),
        )
        row = cur.fetchone()
        report_id = int(row[0]) if row else 0

        # 0 / 负数表示不清理；否则会把刚写入的这份也删掉。
        retention_days = max(0, int(settings.data_quality_retention_days))
        if retention_days > 0:
            cur.execute(
                f"DELETE FROM {TABLE} "
                "WHERE generated_at < now() - make_interval(days => %s)",
                (retention_days,),
            )
    logger.info("质量报告已写入（id=%s, overall=%s）", report_id, overall)
    return report_id


def latest_report() -> dict[str, Any] | None:
    """最近一份报告；表不存在 / 无数据返回 None（不抛）。"""
    try:
        with PGCursor(DataHubQueryError, dict_rows=True) as cur:
            cur.execute(
                f"""
                SELECT id, generated_at, overall, duration_ms, summary, report, source
                FROM {TABLE}
                ORDER BY generated_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - 表不存在/库连不上都按“暂无报告”处理
        logger.debug("读取质量报告失败：%s", exc)
        return None
    if not row:
        return None
    # 扁平返回：报告本体（overall / summary / checks）+ 持久化元数据。
    # 不把整行原样返回，避免前端被迫区分 `data.report.checks` 与 `data.checks`。
    stored = row.get("report")
    report = dict(stored) if isinstance(stored, dict) else {"raw": _jsonable(stored)}
    report.setdefault("generated_at", _jsonable(row.get("generated_at")))
    report["report_id"] = row.get("id")
    report["source"] = row.get("source")
    return report
