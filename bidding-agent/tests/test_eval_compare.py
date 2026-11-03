"""两次运行对比的判定逻辑测试（纯函数，不碰 DB/文件）。

这些断言钉的是“什么算退化/进步/不可比”。判定写错不会报错，
只会让看板给出错误结论，故每个边界都用构造数据覆盖。
"""

from __future__ import annotations

import pytest

from src.eval_runs.compare import compare_records


def _metric(value, status="pass", threshold=None, note=""):
    return {"value": value, "status": status, "threshold": threshold, "note": note}


def _record(run_id="b", metrics=None, **overrides):
    record = {
        "run_id": run_id,
        "version": "v1",
        "schema_version": 1,
        "verdict": "pass",
        "suites": ["agent"],
        "case_counts": {"agent": 5},
        "suite_verdicts": {"agent": "pass"},
        "metrics": metrics or {},
        "config": {},
        "datasets": {},
    }
    record.update(overrides)
    return record


def test_higher_is_better_regression():
    base = _record(metrics={"agent.tool_accuracy": _metric(0.95)})
    head = _record(
        "h",
        metrics={"agent.tool_accuracy": _metric(0.80, status="fail")},
        verdict="fail",
        suite_verdicts={"agent": "fail"},
    )

    result = compare_records(base, head)
    row = result["metrics"][0]

    assert row["metric"] == "agent.tool_accuracy"
    assert row["delta"] == pytest.approx(-0.15)
    assert row["pct_delta"] == pytest.approx(-0.15 / 0.95)
    assert row["change"] == "regressed"
    assert row["status_changed"] is True
    assert row["status_change"] == "pass->fail"
    assert row["is_ratio"] is True
    assert result["suite_verdicts"]["changed"] == {
        "agent": {"base": "pass", "head": "fail"}
    }


def test_lower_is_better_improvement():
    base = _record(metrics={"agent.tool_calls_mean": _metric(3.0)})
    head = _record("h", metrics={"agent.tool_calls_mean": _metric(2.5)})

    row = compare_records(base, head)["metrics"][0]
    assert row["change"] == "improved"
    assert row["delta"] == -0.5


def test_tolerance_marks_small_change_unchanged():
    base = _record(metrics={"agent.tool_accuracy": _metric(0.95)})
    head = _record("h", metrics={"agent.tool_accuracy": _metric(0.951)})

    row = compare_records(base, head, tolerance=0.02)["metrics"][0]
    assert row["change"] == "unchanged"


def test_track_regression_false_is_observed_only():
    base = _record(metrics={"agent.latency_p95": _metric(12000.0)})
    head = _record("h", metrics={"agent.latency_p95": _metric(10000.0)})

    row = compare_records(base, head)["metrics"][0]
    assert row["change"] == "observed"
    assert row["track_regression"] is False
    assert row["delta"] == -2000.0


def test_missing_value_is_not_comparable():
    base = _record(metrics={"agent.tool_accuracy": _metric(0.95)})
    head = _record("h", metrics={"agent.tool_accuracy": _metric(None, status="not_run")})

    row = compare_records(base, head)["metrics"][0]
    assert row["change"] == "not_comparable"
    assert row["delta"] is None
    assert row["status_change"] == "pass->not_run"


def test_unknown_metric_is_unknown_but_delta_visible():
    base = _record(metrics={"agent.brand_new": _metric(1.0)})
    head = _record("h", metrics={"agent.brand_new": _metric(2.0)})

    row = compare_records(base, head)["metrics"][0]
    assert row["change"] == "unknown"
    assert row["label"] == "agent.brand_new"
    assert row["delta"] == 1.0
    assert row["is_ratio"] is False


def test_schema_mismatch_is_not_comparable():
    base = _record(metrics={"agent.tool_accuracy": _metric(0.9)})
    head = _record(
        "h", metrics={"agent.tool_accuracy": _metric(0.9)}, schema_version=2
    )

    result = compare_records(base, head)
    assert result["comparable"] is False
    assert any("schema_version 不一致" in item for item in result["caveats"])


def test_case_count_and_config_and_dataset_diffs_become_caveats():
    base = _record(
        metrics={"agent.tool_accuracy": _metric(0.9)},
        case_counts={"agent": 37},
        config={"LLM_PROVIDER": "deepseek"},
        datasets={"agent": {"sha": "a"}},
    )
    head = _record(
        "h",
        metrics={"agent.tool_accuracy": _metric(0.9)},
        case_counts={"agent": 35},
        config={"LLM_PROVIDER": "zhipu"},
        datasets={"agent": {"sha": "b"}},
    )

    result = compare_records(base, head)
    text = "\n".join(result["caveats"])
    assert "样本量不一致：agent 37 → 35" in text
    assert "运行配置不同" in text
    assert "数据集登记不同" in text
    assert result["config_diff"] == [
        {"key": "LLM_PROVIDER", "base": "deepseek", "head": "zhipu"}
    ]
    assert result["dataset_diff"] == [
        {"key": "agent", "base": {"sha": "a"}, "head": {"sha": "b"}}
    ]


def test_new_and_missing_metrics_are_listed():
    base = _record(metrics={"agent.tool_accuracy": _metric(0.9)})
    head = _record("h", metrics={"retrieval.precision": _metric(0.8)})

    result = compare_records(base, head)
    assert result["new_metrics"] == ["retrieval.precision"]
    assert result["missing_metrics"] == ["agent.tool_accuracy"]
    assert any("新增" in item for item in result["caveats"])
    assert any("缺失" in item for item in result["caveats"])
