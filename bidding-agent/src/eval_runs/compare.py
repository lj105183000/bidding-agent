"""两次评测运行的对比（纯函数，不碰文件/数据库）。

★ 为什么要把判定规则单独成纯函数
--------------------------
“退化”是最容易被写错的结论：方向、容差、样本量、指标是否参与门禁，
任何一项弄反都会静默地给出错误结论。抽成纯函数后可以用构造数据把
`pass->fail`、`not_run`、`track_regression=False`、schema 不一致等
边界逐条钉死（见 `tests/test_eval_compare.py`）。

判定与 `eval/trend.py` 的门禁保持一致：
    - `delta = head - base`；
    - 容差是**相对基线**的（base 为 0 时退化为绝对值）；
    - `track_regression=False` 的指标只展示变化，不判退化；
    - schema_version 不一致直接 `comparable=False`；
    - 套件/样本量/config/数据集差异只进 `caveats`，不阻止对比，
      但前端必须醒目标识，避免把不可比的分差读成退化。
"""

from __future__ import annotations

from typing import Any

from eval.report import Direction, get_spec


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metric(record: dict[str, Any], name: str) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    value = metrics.get(name)
    return value if isinstance(value, dict) else {}


def _change(
    delta: float | None,
    base_value: float | None,
    direction: Direction | None,
    *,
    tolerance: float,
    track_regression: bool,
) -> str:
    if delta is None or base_value is None:
        return "not_comparable"
    # 未注册指标没有方向，不能判进步/退化，只能如实标 unknown。
    if direction is None:
        return "unknown"
    if not track_regression:
        return "observed"
    scale = abs(base_value) if base_value != 0 else 1.0
    if abs(delta) <= tolerance * scale:
        return "unchanged"
    if direction is Direction.HIGHER_IS_BETTER:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _diff(
    base: dict[str, Any], head: dict[str, Any]
) -> list[dict[str, Any]]:
    """两段 JSON 配置的浅层 diff；值不同才列出。"""
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base) | set(head)):
        left = base.get(key)
        right = head.get(key)
        if left != right:
            rows.append({"key": key, "base": left, "head": right})
    return rows


def _caveats(
    base: dict[str, Any],
    head: dict[str, Any],
    *,
    new_metrics: list[str],
    missing_metrics: list[str],
) -> list[str]:
    caveats: list[str] = []
    if base.get("schema_version") != head.get("schema_version"):
        caveats.append(
            f"schema_version 不一致（{base.get('schema_version')} → "
            f"{head.get('schema_version')}），指标口径可能不可比"
        )

    base_suites = set(base.get("suites") or [])
    head_suites = set(head.get("suites") or [])
    if base_suites != head_suites:
        caveats.append(
            "套件集合不同："
            f"{sorted(base_suites)} → {sorted(head_suites)}；"
            "缺测的线不能当成“没退化”"
        )

    base_counts = base.get("case_counts") or {}
    head_counts = head.get("case_counts") or {}
    for suite in sorted(set(base_counts) | set(head_counts)):
        if base_counts.get(suite) != head_counts.get(suite):
            caveats.append(
                f"样本量不一致：{suite} {base_counts.get(suite)} → "
                f"{head_counts.get(suite)} 题；分数差异不作为退化判据"
            )

    config_rows = _diff(base.get("config") or {}, head.get("config") or {})
    if config_rows:
        caveats.append(f"运行配置不同（{len(config_rows)} 项），对比仅供参考")

    dataset_rows = _diff(base.get("datasets") or {}, head.get("datasets") or {})
    if dataset_rows:
        caveats.append(f"数据集登记不同（{len(dataset_rows)} 项），对比仅供参考")

    if new_metrics:
        caveats.append(f"head 新增 {len(new_metrics)} 个指标，基线中无对应值")
    if missing_metrics:
        caveats.append(f"head 缺失 {len(missing_metrics)} 个基线指标")
    return caveats


def compare_records(
    base: dict[str, Any],
    head: dict[str, Any],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """对比两条归一化后的运行记录（`service` 已补齐 threshold）。"""
    base_metrics = base.get("metrics") or {}
    head_metrics = head.get("metrics") or {}
    all_names = sorted(set(base_metrics) | set(head_metrics))
    new_metrics = sorted(set(head_metrics) - set(base_metrics))
    missing_metrics = sorted(set(base_metrics) - set(head_metrics))

    rows: list[dict[str, Any]] = []
    for name in all_names:
        left = _metric(base, name)
        right = _metric(head, name)
        spec = get_spec(name)

        base_value = left.get("value")
        head_value = right.get("value")
        if not _is_number(base_value):
            base_value = None
        if not _is_number(head_value):
            head_value = None

        delta = (
            float(head_value) - float(base_value)
            if base_value is not None and head_value is not None
            else None
        )
        pct_delta = (
            delta / abs(float(base_value))
            if delta is not None and float(base_value) != 0
            else None
        )
        base_status = str(left.get("status") or "not_run")
        head_status = str(right.get("status") or "not_run")
        direction = spec.direction if spec is not None else None
        track_regression = bool(spec.track_regression) if spec is not None else False

        rows.append(
            {
                "metric": name,
                "label": spec.label if spec is not None else name,
                "suite": (
                    spec.suite if spec is not None else name.split(".", 1)[0]
                ),
                "unit": spec.unit if spec is not None else "",
                "direction": direction.value if direction is not None else "",
                "is_ratio": bool(spec is not None and spec.unit == "ratio"),
                "base": {
                    "value": base_value,
                    "status": base_status,
                    "threshold": left.get("threshold"),
                },
                "head": {
                    "value": head_value,
                    "status": head_status,
                    "threshold": right.get("threshold"),
                },
                "delta": delta,
                "pct_delta": pct_delta,
                "status_changed": base_status != head_status,
                "status_change": f"{base_status}->{head_status}",
                "change": _change(
                    delta,
                    base_value,
                    direction,
                    tolerance=tolerance,
                    track_regression=track_regression,
                ),
                "track_regression": track_regression,
                "note": str(right.get("note") or left.get("note") or ""),
            }
        )

    base_verdicts = base.get("suite_verdicts") or {}
    head_verdicts = head.get("suite_verdicts") or {}
    changed = {
        suite: {"base": base_verdicts.get(suite), "head": head_verdicts.get(suite)}
        for suite in sorted(set(base_verdicts) | set(head_verdicts))
        if base_verdicts.get(suite) != head_verdicts.get(suite)
    }

    return {
        "base": {
            "run_id": base.get("run_id"),
            "version": base.get("version"),
            "verdict": base.get("verdict"),
            "created_at": base.get("created_at"),
            "suites": base.get("suites") or [],
            "case_counts": base.get("case_counts") or {},
        },
        "head": {
            "run_id": head.get("run_id"),
            "version": head.get("version"),
            "verdict": head.get("verdict"),
            "created_at": head.get("created_at"),
            "suites": head.get("suites") or [],
            "case_counts": head.get("case_counts") or {},
        },
        "comparable": base.get("schema_version") == head.get("schema_version"),
        "caveats": _caveats(
            base,
            head,
            new_metrics=new_metrics,
            missing_metrics=missing_metrics,
        ),
        "metrics": rows,
        "suite_verdicts": {
            "base": base_verdicts,
            "head": head_verdicts,
            "changed": changed,
        },
        "config_diff": _diff(base.get("config") or {}, head.get("config") or {}),
        "dataset_diff": _diff(base.get("datasets") or {}, head.get("datasets") or {}),
        "new_metrics": new_metrics,
        "missing_metrics": missing_metrics,
        "tolerance": tolerance,
    }


__all__ = ["compare_records"]
