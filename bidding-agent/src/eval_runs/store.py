"""`eval_runs` / `eval_jobs` 的读写入口。

所有函数都经 `_guarded`：
    - 熔断窗口内直接抛 `EvalStoreError`，调用方（API）立刻走文件回退；
    - 任何底层异常统一映射成 `EvalStoreError`，完整异常只进日志。
不在这里做“捕获后返回空列表”的降级——那会让“数据库挂了”和“没有数据”
在调用方看起来一样，正是质量问题最难排查的一类。
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from psycopg.types.json import Jsonb

from src.database.postgresql_client import _jsonable
from src.datahub.pg import PGCursor, PGTransaction
from src.eval_runs import db
from src.eval_runs.db import EvalStoreError

logger = logging.getLogger(__name__)

_RUN_COLUMNS = (
    "run_id, created_at, version, schema_version, source, job_id, suites, "
    "suites_source, verdict, case_counts, suite_verdicts, metrics, "
    "suite_notes, datasets, config"
)

_INSERT_RUN = f"""
INSERT INTO {db.TABLE_RUNS}
    (run_id, created_at, version, schema_version, source, job_id, suites,
     suites_source, verdict, case_counts, suite_verdicts, metrics,
     suite_notes, datasets, config)
VALUES
    (%s, COALESCE(%s::timestamptz, now()), %s, %s, %s, %s, %s,
     %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id) DO NOTHING
RETURNING run_id
"""

_UPDATE_RUN = f"""
UPDATE {db.TABLE_RUNS} SET
    created_at = COALESCE(%s::timestamptz, created_at),
    version = %s,
    schema_version = %s,
    job_id = %s,
    suites = %s,
    suites_source = %s,
    verdict = %s,
    case_counts = %s,
    suite_verdicts = %s,
    metrics = %s,
    suite_notes = %s,
    datasets = %s,
    config = %s
WHERE run_id = %s AND source = 'backfill'
RETURNING run_id
"""

_JOB_COLUMNS = (
    "job_id, status, suites, note, requested_at, started_at, finished_at, "
    "run_id, exit_code, error, log_path"
)

#: `update_job` 允许改的列。白名单而不是拼接调用方字段：
#: job 状态由本进程写，但字段名一旦来自外部就是注入口子。
_JOB_UPDATABLE = frozenset(
    {"status", "started_at", "finished_at", "run_id", "exit_code", "error", "log_path"}
)


def _guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not db.is_available():
            raise EvalStoreError("评测数据库暂不可用")
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 统一映射并熔断
            db.note_failure(exc)
            if isinstance(exc, EvalStoreError):
                raise
            raise EvalStoreError(f"评测数据库操作失败：{type(exc).__name__}") from exc
        db.note_success()
        return result

    return wrapper


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in row.items()}


def _run_params(record: dict[str, Any]) -> tuple:
    return (
        record.get("run_id"),
        record.get("created_at"),
        str(record.get("version") or "unknown"),
        int(record.get("schema_version") or 1),
        str(record.get("source") or "cli"),
        record.get("job_id"),
        [str(suite) for suite in (record.get("suites") or [])],
        str(record.get("suites_source") or "observed"),
        str(record.get("verdict") or "pass"),
        Jsonb(dict(record.get("case_counts") or {})),
        Jsonb(dict(record.get("suite_verdicts") or {})),
        Jsonb(dict(record.get("metrics") or {})),
        Jsonb(dict(record.get("suite_notes") or {})),
        Jsonb(dict(record.get("datasets") or {})),
        Jsonb(dict(record.get("config") or {})),
    )


def _update_params(record: dict[str, Any]) -> tuple:
    params = _run_params(record)
    # INSERT 的参数顺序是 run_id 在最前、source 在中间；UPDATE 不更新 source，
    # 故跳过 params[4]，并把 run_id 挪到最后给 WHERE。
    return params[1:4] + params[5:] + (params[0],)


@_guarded
def save_records(
    records: list[dict[str, Any]], *, overwrite_backfill: bool = False
) -> dict[str, int]:
    """批量写入。返回各结果的计数，便于回填脚本如实汇报。

    `overwrite_backfill=True` 只更新 `source='backfill'` 的既有行；
    API/CLI 实时记录永远不会被回填覆盖——否则一次误操作就能改掉
    看板上的历史数字。
    """
    counts = {"inserted": 0, "skipped": 0, "overwritten": 0, "protected": 0}
    valid = [record for record in records if record.get("run_id")]
    if not valid:
        return counts

    db.ensure_schema()
    with PGTransaction(EvalStoreError) as cur:
        for record in valid:
            cur.execute(_INSERT_RUN, _run_params(record))
            if cur.fetchone():
                counts["inserted"] += 1
                continue
            if not overwrite_backfill:
                counts["skipped"] += 1
                continue
            cur.execute(_UPDATE_RUN, _update_params(record))
            if cur.fetchone():
                counts["overwritten"] += 1
            else:
                counts["protected"] += 1
    logger.info(
        "评测记录写入完成：新增 %s、跳过 %s、覆盖 %s、受保护 %s",
        counts["inserted"],
        counts["skipped"],
        counts["overwritten"],
        counts["protected"],
    )
    return counts


def save_record(
    record: dict[str, Any], *, overwrite_backfill: bool = False
) -> str:
    """单条写入，返回 `inserted / skipped / overwritten / protected`。"""
    counts = save_records([record], overwrite_backfill=overwrite_backfill)
    for key in ("inserted", "overwritten", "protected", "skipped"):
        if counts.get(key):
            return key
    return "skipped"


def _where(
    *,
    suite: str | None = None,
    verdict: str | None = None,
    source: str | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if suite:
        clauses.append("%s = ANY(suites)")
        params.append(suite)
    if verdict:
        clauses.append("verdict = %s")
        params.append(verdict)
    if source:
        clauses.append("source = %s")
        params.append(source)
    return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), params


@_guarded
def list_records(
    limit: int = 20,
    offset: int = 0,
    *,
    suite: str | None = None,
    verdict: str | None = None,
    source: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """分页列出运行记录，附带满足筛选条件的总数。"""
    db.ensure_schema()
    where, params = _where(suite=suite, verdict=verdict, source=source)
    with PGCursor(EvalStoreError, dict_rows=True) as cur:
        cur.execute(
            f"SELECT COUNT(*) AS total FROM {db.TABLE_RUNS}{where}",
            tuple(params),
        )
        row = cur.fetchone() or {"total": 0}
        total = int(row.get("total") or 0)
        cur.execute(
            f"SELECT {_RUN_COLUMNS} FROM {db.TABLE_RUNS}{where} "
            "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
            tuple(params + [limit, offset]),
        )
        rows = cur.fetchall() or []
    return [_jsonable_row(row) for row in rows], total


@_guarded
def get_record(run_id: str) -> dict[str, Any] | None:
    db.ensure_schema()
    with PGCursor(EvalStoreError, dict_rows=True) as cur:
        cur.execute(
            f"SELECT {_RUN_COLUMNS} FROM {db.TABLE_RUNS} WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    return _jsonable_row(row) if row else None


@_guarded
def has_run(run_id: str) -> bool:
    db.ensure_schema()
    with PGCursor(EvalStoreError) as cur:
        cur.execute(
            f"SELECT 1 FROM {db.TABLE_RUNS} WHERE run_id = %s LIMIT 1",
            (run_id,),
        )
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# eval_jobs
# ---------------------------------------------------------------------------


@_guarded
def create_job(job: dict[str, Any]) -> None:
    db.ensure_schema()
    with PGTransaction(EvalStoreError) as cur:
        cur.execute(
            f"""
            INSERT INTO {db.TABLE_JOBS}
                (job_id, status, suites, note, requested_at, run_id, log_path)
            VALUES (%s, %s, %s, %s, COALESCE(%s::timestamptz, now()), %s, %s)
            """,
            (
                job.get("job_id"),
                str(job.get("status") or "queued"),
                [str(suite) for suite in (job.get("suites") or [])],
                str(job.get("note") or ""),
                job.get("requested_at"),
                job.get("run_id"),
                str(job.get("log_path") or ""),
            ),
        )


@_guarded
def update_job(job_id: str, **fields: Any) -> bool:
    """按白名单更新任务字段；返回是否命中一行。"""
    updates = {key: value for key, value in fields.items() if key in _JOB_UPDATABLE}
    if not updates:
        return False
    assignments = ", ".join(f"{key} = %s" for key in updates)
    params = list(updates.values()) + [job_id]
    db.ensure_schema()
    with PGTransaction(EvalStoreError) as cur:
        cur.execute(
            f"UPDATE {db.TABLE_JOBS} SET {assignments} WHERE job_id = %s "
            "RETURNING job_id",
            tuple(params),
        )
        return cur.fetchone() is not None


@_guarded
def get_job(job_id: str) -> dict[str, Any] | None:
    db.ensure_schema()
    with PGCursor(EvalStoreError, dict_rows=True) as cur:
        cur.execute(
            f"SELECT {_JOB_COLUMNS} FROM {db.TABLE_JOBS} WHERE job_id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    return _jsonable_row(row) if row else None


@_guarded
def active_job() -> dict[str, Any] | None:
    """当前 queued/running 的任务；用于单任务锁的二次确认。"""
    db.ensure_schema()
    with PGCursor(EvalStoreError, dict_rows=True) as cur:
        cur.execute(
            f"SELECT {_JOB_COLUMNS} FROM {db.TABLE_JOBS} "
            "WHERE status IN ('queued','running') "
            "ORDER BY requested_at DESC LIMIT 1"
        )
        row = cur.fetchone()
    return _jsonable_row(row) if row else None


@_guarded
def list_jobs(
    limit: int = 20, offset: int = 0, *, status: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    where = ""
    params: list[Any] = []
    if status:
        where = " WHERE status = %s"
        params.append(status)
    db.ensure_schema()
    with PGCursor(EvalStoreError, dict_rows=True) as cur:
        cur.execute(
            f"SELECT COUNT(*) AS total FROM {db.TABLE_JOBS}{where}",
            tuple(params),
        )
        row = cur.fetchone() or {"total": 0}
        total = int(row.get("total") or 0)
        cur.execute(
            f"SELECT {_JOB_COLUMNS} FROM {db.TABLE_JOBS}{where} "
            "ORDER BY requested_at DESC LIMIT %s OFFSET %s",
            tuple(params + [limit, offset]),
        )
        rows = cur.fetchall() or []
    return [_jsonable_row(row) for row in rows], total


@_guarded
def reconcile_stale_jobs(max_age_seconds: int) -> int:
    """把超龄的 queued/running 任务标成 lost，返回条数。

    ★ 为什么按时间而不是 pid
    --------------------
    `python main.py dev` 的 reload、进程崩溃都会让后台线程消失；
    API 重启后无法可靠判断子进程是否还活着（pid 可能被复用）。
    按时间判断是“诚实的降级”：不确定就标 lost，并在错误信息里
    让用户去看日志，而不是伪造 succeeded。
    """
    db.ensure_schema()
    with PGTransaction(EvalStoreError) as cur:
        cur.execute(
            f"""
            UPDATE {db.TABLE_JOBS}
            SET status = 'lost',
                finished_at = now(),
                error = COALESCE(NULLIF(error, ''), '任务超龄，未能确认最终状态；请查看日志')
            WHERE status IN ('queued','running')
              AND requested_at < now() - make_interval(secs => %s)
            RETURNING job_id
            """,
            (max(0, int(max_age_seconds)),),
        )
        rows = cur.fetchall() or []
    return len(rows)


__all__ = [
    "active_job",
    "create_job",
    "get_job",
    "get_record",
    "has_run",
    "list_jobs",
    "list_records",
    "reconcile_stale_jobs",
    "save_record",
    "save_records",
    "update_job",
]
