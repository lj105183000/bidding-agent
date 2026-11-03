"""评测中台的读取门面：DB 优先、趋势文件回退、字段补全。

★ 为什么要有“DB 空/不可用 → 回退 trend.jsonl”
----------------------------------------
新部署的库还没有回填时，看板不应该是一片空白：趋势文件里已有 34 次
历史运行，正是回填脚本的输入。回退让“刚上线”也有数据可看，同时用
`storage.backend` 如实告诉前端数字来自哪里。

★ 为什么回退只读、不写
------------------
读取路径每次请求都可能触发；如果顺手把趋势记录写进 DB，回填的去重、
来源标记（backfill）和 `--overwrite` 边界都会被请求时序搅乱。
写入只发生在 `--save-db` 与 `backfill_db.py` 两个显式入口。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from eval.report import get_spec
from src.eval_runs import jobs, store
from src.eval_runs.compare import compare_records
from src.eval_runs.db import EvalStoreError
from src.eval_runs.records import normalize_trend_record

logger = logging.getLogger(__name__)


def _file_available() -> bool:
    try:
        from src.eval_offline import TREND

        return TREND.exists()
    except Exception:  # noqa: BLE001 - 文件探测失败不该影响接口
        return False


def _storage(
    backend: str, *, reason: str = "", db_available: bool = False
) -> dict[str, Any]:
    return {
        "backend": backend,
        "degraded": backend != "db",
        "reason": reason,
        "db_available": db_available,
        "file_available": _file_available(),
    }


def _created_at(record: dict[str, Any]) -> Any:
    value = record.get("created_at")
    return value.isoformat() if isinstance(value, datetime) else value


def _trend_records() -> list[dict[str, Any]]:
    """读取并归一化趋势文件；任何失败都返回空列表（降级用）。"""
    try:
        from src.eval_offline import read_trend_records

        return [
            normalize_trend_record(raw)
            for raw in read_trend_records()
            if isinstance(raw, dict) and raw.get("run_id")
        ]
    except Exception as exc:  # noqa: BLE001 - 回退失败只能空列表
        logger.warning("读取趋势文件失败：%s", exc, exc_info=True)
        return []


def _filter_records(
    records: list[dict[str, Any]],
    *,
    suite: str | None,
    verdict: str | None,
    source: str | None,
) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if suite and suite not in (record.get("suites") or []):
            continue
        if verdict and record.get("verdict") != verdict:
            continue
        if source and record.get("source") != source:
            continue
        out.append(record)
    out.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )
    return out


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "created_at": _created_at(record),
        "version": record.get("version"),
        "schema_version": record.get("schema_version"),
        "source": record.get("source"),
        "job_id": record.get("job_id"),
        "suites": record.get("suites") or [],
        "suites_source": record.get("suites_source"),
        "verdict": record.get("verdict"),
        "case_counts": record.get("case_counts") or {},
        "suite_verdicts": record.get("suite_verdicts") or {},
        "metric_count": len(record.get("metrics") or {}),
    }


def _detail(record: dict[str, Any]) -> dict[str, Any]:
    metrics = []
    for name, value in sorted((record.get("metrics") or {}).items()):
        if not isinstance(value, dict):
            continue
        spec = get_spec(name)
        unit = spec.unit if spec is not None else ""
        metrics.append(
            {
                "metric": name,
                "label": spec.label if spec is not None else name,
                "suite": (
                    spec.suite if spec is not None else name.split(".", 1)[0]
                ),
                "value": value.get("value"),
                "status": str(value.get("status") or "not_run"),
                "threshold": value.get(
                    "threshold", spec.threshold if spec is not None else None
                ),
                "unit": unit,
                "direction": spec.direction.value if spec is not None else "",
                "is_ratio": unit == "ratio",
                "note": str(value.get("note") or ""),
            }
        )
    return {
        **_summary(record),
        "config": record.get("config") or {},
        "suite_notes": record.get("suite_notes") or {},
        "datasets": record.get("datasets") or {},
        "metrics": metrics,
    }


def db_storage() -> dict[str, Any]:
    """任务/触发等只存 DB 的响应，统一用这个来源标记。"""
    return _storage("db", db_available=True)


def capabilities() -> dict[str, Any]:
    """能力声明：前端据此隐藏做不到的功能，而不是画一张空表。"""
    return {
        "per_case": False,
        "per_case_reason": (
            "EvalReport 只保存聚合后的 MetricResult；逐用例结果在 run_all.py "
            "内部用完即弃。"
        ),
        "compare": True,
        "trigger": jobs.trigger_status(),
    }


def _list_payload(
    records: list[dict[str, Any]],
    total: int,
    limit: int,
    offset: int,
    storage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runs": [_summary(record) for record in records],
        "total": total,
        "limit": limit,
        "offset": offset,
        "storage": storage,
        "capabilities": capabilities(),
    }


def list_runs(
    limit: int = 20,
    offset: int = 0,
    *,
    suite: str | None = None,
    verdict: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """运行列表。DB 优先；DB 完全为空时用趋势文件回退。"""
    try:
        rows, total = store.list_records(
            limit, offset, suite=suite, verdict=verdict, source=source
        )
    except EvalStoreError as exc:
        logger.warning("评测运行列表读取失败，回退趋势文件：%s", exc)
        records = _filter_records(
            _trend_records(), suite=suite, verdict=verdict, source=source
        )
        storage = _storage("trend_file", reason="db_unavailable", db_available=False)
        if not records and not _file_available():
            storage = _storage("none", reason="db_unavailable", db_available=False)
        return _list_payload(
            records[offset : offset + limit], len(records), limit, offset, storage
        )

    # 筛选后为 0 不代表库是空的。只有“整库为空”才回退，
    # 否则用户在 DB 里筛一个不存在的条件时，会被旧趋势文件误导。
    if total == 0:
        try:
            _, any_total = store.list_records(1, 0)
        except EvalStoreError:
            any_total = 0
        if any_total == 0:
            records = _filter_records(
                _trend_records(), suite=suite, verdict=verdict, source=source
            )
            if records:
                return _list_payload(
                    records[offset : offset + limit],
                    len(records),
                    limit,
                    offset,
                    _storage("trend_file", reason="db_empty", db_available=True),
                )

    return _list_payload(rows, total, limit, offset, _storage("db", db_available=True))


def _load_record(run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """返回原始记录（指标仍是 dict）与来源信息。"""
    db_available = True
    try:
        record = store.get_record(run_id)
    except EvalStoreError as exc:
        logger.warning("评测运行详情读取失败，回退趋势文件：%s", exc)
        record = None
        db_available = False

    if record is not None:
        return record, _storage("db", db_available=True)

    for fallback in _trend_records():
        if fallback.get("run_id") == run_id:
            reason = "db_miss" if db_available else "db_unavailable"
            return fallback, _storage(
                "trend_file", reason=reason, db_available=db_available
            )
    return None, _storage(
        "trend_file" if db_available else "none",
        reason="db_miss" if db_available else "db_unavailable",
        db_available=db_available,
    )


def get_run(run_id: str) -> dict[str, Any] | None:
    record, storage = _load_record(run_id)
    if record is None:
        return None
    return {"run": _detail(record), "storage": storage}


def compare_runs(
    base: str, head: str, *, tolerance: float = 0.02
) -> dict[str, Any] | None:
    base_record, base_storage = _load_record(base)
    head_record, head_storage = _load_record(head)
    if base_record is None or head_record is None:
        return None
    result = compare_records(base_record, head_record, tolerance=tolerance)
    result["storage"] = {"base": base_storage, "head": head_storage}
    return result


#: 任务状态枚举。API 用它校验筛选参数。
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "lost"})


def list_jobs(
    limit: int = 20, offset: int = 0, *, status: str | None = None
) -> dict[str, Any]:
    """评测任务列表。任务状态只存 DB，故 DB 不可用时返回空列表 + degraded。"""
    try:
        rows, total = store.list_jobs(limit, offset, status=status)
    except EvalStoreError as exc:
        logger.warning("评测任务列表读取失败：%s", exc)
        return {
            "jobs": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "storage": _storage("none", reason="db_unavailable", db_available=False),
            "capabilities": capabilities(),
        }
    return {
        "jobs": rows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "storage": _storage("db", db_available=True),
        "capabilities": capabilities(),
    }


def get_job(job_id: str) -> dict[str, Any] | None:
    """任务详情；DB 不可用抛 `EvalStoreError`（由 API 转 503）。"""
    job = store.get_job(job_id)
    if job is None:
        return None
    payload: dict[str, Any] = {
        "job": job,
        "storage": _storage("db", db_available=True),
        "capabilities": capabilities(),
    }
    run_id = job.get("run_id")
    if run_id:
        record, _ = _load_record(str(run_id))
        if record is not None:
            payload["run"] = _summary(record)
    return payload


__all__ = [
    "JOB_STATUSES",
    "capabilities",
    "db_storage",
    "compare_runs",
    "get_job",
    "get_run",
    "list_jobs",
    "list_runs",
]
