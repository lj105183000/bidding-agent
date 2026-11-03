"""评测中台的 PostgreSQL 基础设施：建表、异常、熔断。

★ 为什么建表放在服务里而不是 `to_postgresql.py`
------------------------------------------
`to_postgresql.py` 管的是**业务表重建**（`bidding_records` 会 DROP）。
评测结果与业务数据无关，且必须能在已有库上增量创建；放进重建流程会让
“导入业务数据”顺带碰评测历史，风险不成比例。这里用幂等
`CREATE TABLE IF NOT EXISTS`，第一次读/写时自动建。

★ 为什么要熔断
----------
数据库挂掉时，`PGCursor` 的连接超时是 10 秒。看板每 15 秒轮询一次，
没有熔断就会每个请求都等 10 秒，页面看起来像卡死。熔断后 30 秒内直接
走文件回退，失败日志也只记一次，避免刷屏。
"""

from __future__ import annotations

import logging
import threading
import time

from src.datahub.contract import DataHubError
from src.datahub.pg import PGTransaction

logger = logging.getLogger(__name__)


class EvalStoreError(DataHubError):
    """评测数据库不可用或读写失败。

    继承 `DataHubError`，让上层可以按“数据访问失败”统一降级，
    不必记住又多了一种异常类型。
    """


TABLE_RUNS = "eval_runs"
TABLE_JOBS = "eval_jobs"

#: 幂等 DDL。顺序有意义：先建表再建索引。
#:
#: `eval_jobs` 的部分唯一索引是单任务锁的**第二道保险**：
#: 进程内锁只能约束一个 API 进程，而生产要求单进程，理论上足够；
#: 但重启/误开多进程时，数据库这层会直接拒绝第二个 active 任务。
_DDL = (
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_RUNS} (
        id              BIGSERIAL PRIMARY KEY,
        run_id          TEXT NOT NULL UNIQUE,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        version         TEXT NOT NULL DEFAULT 'unknown',
        schema_version  INTEGER NOT NULL DEFAULT 1,
        source          TEXT NOT NULL DEFAULT 'cli',
        job_id          TEXT,
        suites          TEXT[] NOT NULL DEFAULT '{{}}',
        suites_source   TEXT NOT NULL DEFAULT 'observed',
        verdict         TEXT NOT NULL DEFAULT 'pass',
        case_counts     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        suite_verdicts  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        metrics         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        suite_notes     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        datasets        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        config          JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        stored_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_RUNS}_created_at "
    f"ON {TABLE_RUNS} (created_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_RUNS}_suites "
    f"ON {TABLE_RUNS} USING GIN (suites)",
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_JOBS} (
        job_id        TEXT PRIMARY KEY,
        status        TEXT NOT NULL,
        suites        TEXT[] NOT NULL,
        note          TEXT NOT NULL DEFAULT '',
        requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        started_at    TIMESTAMPTZ,
        finished_at   TIMESTAMPTZ,
        run_id        TEXT,
        exit_code     INTEGER,
        error         TEXT,
        log_path      TEXT NOT NULL DEFAULT ''
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_JOBS}_requested_at "
    f"ON {TABLE_JOBS} (requested_at DESC)",
    f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{TABLE_JOBS}_active "
    f"ON {TABLE_JOBS} ((status IN ('queued','running'))) "
    f"WHERE status IN ('queued','running')",
)

#: 熔断窗口（秒）。与看板 15 秒轮询配合：一次失败最多影响两轮。
BREAKER_SECONDS = 30.0

_schema_lock = threading.Lock()
_schema_ready = False

_breaker_lock = threading.Lock()
_unavailable_until = 0.0
_last_failure_logged_at = 0.0


def ensure_schema() -> None:
    """幂等建表；失败抛 `EvalStoreError`。

    双检锁：并发的第一批请求只让一个去执行 DDL，其余等锁后直接命中
    `_schema_ready`。与项目里模型懒加载的写法保持一致。
    """
    global _schema_ready
    if _schema_ready:
        return
    if not is_available():
        raise EvalStoreError("评测数据库暂不可用")
    with _schema_lock:
        if _schema_ready:
            return
        with PGTransaction(EvalStoreError) as cur:
            for statement in _DDL:
                cur.execute(statement)
        _schema_ready = True


def is_available() -> bool:
    """熔断窗口内返回 False；调用方据此直接走文件回退。"""
    with _breaker_lock:
        return time.monotonic() >= _unavailable_until


def note_failure(exc: BaseException) -> None:
    """记录一次数据库失败并打开熔断窗口。

    只在窗口首次打开时 warning，其余衰减为 debug：
    数据库宕机时每个请求都打完整堆栈，日志会被淹没。
    """
    global _unavailable_until, _last_failure_logged_at
    now = time.monotonic()
    with _breaker_lock:
        _unavailable_until = now + BREAKER_SECONDS
        should_log = now - _last_failure_logged_at >= BREAKER_SECONDS
        if should_log:
            _last_failure_logged_at = now
    if should_log:
        logger.warning(
            "评测数据库不可用，%.0f 秒内直接回退文件：%s",
            BREAKER_SECONDS,
            exc,
        )
    else:
        logger.debug("评测数据库仍不可用：%s", exc)


def note_success() -> None:
    """一次成功访问后关闭熔断。"""
    global _unavailable_until
    with _breaker_lock:
        _unavailable_until = 0.0


def reset_breaker() -> None:
    """清除熔断（测试与运维排障用）。"""
    note_success()
    global _last_failure_logged_at
    with _breaker_lock:
        _last_failure_logged_at = 0.0


__all__ = [
    "BREAKER_SECONDS",
    "EvalStoreError",
    "TABLE_JOBS",
    "TABLE_RUNS",
    "ensure_schema",
    "is_available",
    "note_failure",
    "note_success",
    "reset_breaker",
]
