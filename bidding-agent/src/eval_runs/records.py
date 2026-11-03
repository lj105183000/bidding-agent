"""把评测报告 / 趋势记录归一成 `eval_runs` 的一行。

★ 为什么复用 `eval.trend.report_to_record`
--------------------------------------
趋势记录是门禁的既有契约，字段含义（`suites` 只记真正跑过的线、
`case_counts` 保证样本量可比）都经过实测修正。这里在它之上只补
数据库需要的字段（`created_at`、逐指标 `threshold/note`、`suite_verdicts`），
不另造一套映射，避免两套事实源慢慢漂移。

★ 为什么这里要自己把项目根加进 `sys.path`
------------------------------------
`run_all.py` 启动时已经加过一次；但 API 进程是 `uvicorn api.server:app`，
`src` 包能被导入不代表 `eval`（无 `__init__.py` 的命名空间包）也在路径上。
显式补一次，比在部署文档里要求设置 PYTHONPATH 可靠。
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.report import get_spec  # noqa: E402
from eval.trend import report_to_record  # noqa: E402

#: run_id 的默认格式：`20261031T152132Z`。手动触发会在后面加 `-<uuid8>`，
#: 故只匹配前缀；解析失败时由数据库 `now()` 兜底。
_RUN_ID_TS = re.compile(r"^(\d{8}T\d{6}Z)")

#: 允许的来源。未知值归一到 cli，避免前端按 source 筛选时出现拼写漂移。
_SOURCES = frozenset({"cli", "api", "backfill", "ci", "trend_file"})


def parse_created_at(run_id: str) -> datetime | None:
    """从 run_id 前缀解析 UTC 时间；解析不了返回 None。"""
    match = _RUN_ID_TS.match(run_id or "")
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _normalize_source(source: str | None) -> str:
    value = (source or "").strip().lower()
    return value if value in _SOURCES else "cli"


def _metrics_from_report(report: Any) -> dict[str, dict[str, Any]]:
    """逐指标保留 threshold/note，供详情页离线阅读时不必再查注册表。"""
    metrics: dict[str, dict[str, Any]] = {}
    for result in report.all_results:
        spec = get_spec(result.name)
        threshold = result.threshold
        if threshold is None and spec is not None:
            threshold = spec.threshold
        metrics[result.name] = {
            "value": result.value,
            "status": result.status.value,
            "threshold": threshold,
            "note": result.note,
        }
    return metrics


def _verdict_from_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    """与 `EvalReport.verdict` 一致：fail 才算失败，not_run 不算。"""
    return "fail" if any(m.get("status") == "fail" for m in metrics.values()) else "pass"


def _derive_suite_verdicts(
    metrics: dict[str, dict[str, Any]], suites: list[str]
) -> dict[str, str]:
    """旧趋势记录没有分线结论，从指标状态反推。

    “所有指标都是 not_run”与“没有任何指标”统一记为 not_run：
    没测不等于通过，否则看板会把空结果画成绿色。
    """
    verdicts: dict[str, str] = {}
    for suite in suites:
        statuses = [
            str(metric.get("status") or "not_run")
            for name, metric in metrics.items()
            if name.startswith(f"{suite}.")
        ]
        if not statuses or all(status == "not_run" for status in statuses):
            verdicts[suite] = "not_run"
        elif any(status == "fail" for status in statuses):
            verdicts[suite] = "fail"
        else:
            verdicts[suite] = "pass"
    return verdicts


def record_from_report(
    report: Any, *, source: str = "cli", job_id: str | None = None
) -> dict[str, Any]:
    """`EvalReport` → `eval_runs` 行（普通 dict，便于 fake store 注入）。"""
    record = report_to_record(report)
    metrics = _metrics_from_report(report)
    return {
        **record,
        "created_at": parse_created_at(record.get("run_id", "")),
        "source": _normalize_source(source),
        "job_id": job_id,
        "metrics": metrics,
        "suite_verdicts": {
            suite.suite: suite.verdict for suite in report.suites if suite.ran
        },
        "datasets": dict(report.datasets or {}),
    }


def normalize_trend_record(
    record: dict[str, Any], *, source: str = "trend_file"
) -> dict[str, Any]:
    """把 `trend.jsonl` 的一行补成 `eval_runs` 行。

    趋势文件只存 `{value, status}`，没有 threshold/note（旧版本也没有
    `datasets`）。这里用**当前** `METRIC_SPECS` 补 threshold——文档里
    要注明这是“当前注册表值”，不是当时的历史门槛。
    """
    raw_metrics = record.get("metrics")
    metrics: dict[str, dict[str, Any]] = {}
    if isinstance(raw_metrics, dict):
        for name, value in raw_metrics.items():
            if not isinstance(value, dict):
                continue
            spec = get_spec(str(name))
            metrics[str(name)] = {
                "value": value.get("value"),
                "status": str(value.get("status") or "not_run"),
                "threshold": value.get(
                    "threshold", spec.threshold if spec is not None else None
                ),
                "note": str(value.get("note") or ""),
            }

    suites = record.get("suites")
    if not isinstance(suites, list) or not suites:
        suites = sorted(
            {name.split(".", 1)[0] for name in metrics if "." in name}
        )
    suites = [str(suite) for suite in suites if str(suite).strip()]

    suite_verdicts = record.get("suite_verdicts")
    if not isinstance(suite_verdicts, dict) or not suite_verdicts:
        suite_verdicts = _derive_suite_verdicts(metrics, suites)

    run_id = str(record.get("run_id") or "")
    return {
        "schema_version": int(record.get("schema_version") or 1),
        "run_id": run_id,
        "created_at": parse_created_at(run_id),
        "version": str(record.get("version") or "unknown"),
        "source": _normalize_source(source),
        "job_id": record.get("job_id"),
        "suites": suites,
        "suites_source": str(record.get("suites_source") or "observed"),
        "verdict": str(record.get("verdict") or _verdict_from_metrics(metrics)),
        "case_counts": dict(record.get("case_counts") or {}),
        "suite_verdicts": {str(k): str(v) for k, v in suite_verdicts.items()},
        "metrics": metrics,
        "suite_notes": dict(record.get("suite_notes") or {}),
        "datasets": dict(record.get("datasets") or {}),
        "config": dict(record.get("config") or {}),
    }


__all__ = [
    "normalize_trend_record",
    "parse_created_at",
    "record_from_report",
]
