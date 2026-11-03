"""评测报告 → `eval_runs` 记录的归一化测试。

这一层最容易出现的静默错误是“字段丢了但没人发现”：
详情页仍能打开，只是 threshold 变成 null、分线结论为空。
故用构造报告把每个字段钉死，不连数据库。
"""

from __future__ import annotations

from datetime import datetime, timezone

from eval import report as R
from src.eval_runs import records as REC


def _report() -> R.EvalReport:
    suite = R.SuiteReport(suite="agent", ran=True, case_count=5)
    suite.results = [
        R.MetricResult(
            name="agent.tool_accuracy",
            value=0.9,
            status=R.MetricStatus.PASS,
            threshold=0.9,
            note="ok",
        ),
        R.MetricResult(
            name="agent.latency_p95",
            value=1200.0,
            status=R.MetricStatus.FAIL,
            threshold=1000.0,
            note="slow",
        ),
    ]
    return R.EvalReport(
        schema_version=R.SCHEMA_VERSION,
        run_id="20261031T152132Z-a1b2c3d4",
        version="abc1234",
        suites=[suite],
        datasets={"agent": {"cases": 5}},
    )


def test_parse_created_at_accepts_timestamp_prefix():
    assert REC.parse_created_at("20261031T152132Z") == datetime(
        2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc
    )
    # 手动触发的 job_id 带后缀，仍应解析出前缀时间。
    assert REC.parse_created_at("20261031T152132Z-a1b2c3d4") == datetime(
        2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc
    )


def test_parse_created_at_returns_none_for_unknown_shape():
    assert REC.parse_created_at("") is None
    assert REC.parse_created_at("run-1") is None
    assert REC.parse_created_at("20261399T999999Z") is None


def test_record_from_report_preserves_metric_metadata():
    record = REC.record_from_report(_report(), source="api", job_id="job-1")

    assert record["run_id"] == "20261031T152132Z-a1b2c3d4"
    assert record["created_at"] == datetime(
        2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc
    )
    assert record["source"] == "api"
    assert record["job_id"] == "job-1"
    assert record["suites"] == ["agent"]
    assert record["case_counts"] == {"agent": 5}
    assert record["verdict"] == "fail"
    assert record["suite_verdicts"] == {"agent": "fail"}
    assert record["datasets"] == {"agent": {"cases": 5}}

    metric = record["metrics"]["agent.tool_accuracy"]
    assert metric == {
        "value": 0.9,
        "status": "pass",
        "threshold": 0.9,
        "note": "ok",
    }


def test_record_from_report_normalizes_unknown_source():
    record = REC.record_from_report(_report(), source="something-else")
    assert record["source"] == "cli"


def test_normalize_trend_record_fills_threshold_and_derives_verdicts():
    raw = {
        "schema_version": 1,
        "run_id": "20261031T152132Z",
        "version": "v1",
        "suites": ["agent"],
        "case_counts": {"agent": 5},
        "metrics": {
            "agent.tool_accuracy": {"value": 0.91, "status": "pass"},
            "agent.latency_p95": {"value": 1200.0, "status": "fail"},
        },
        "config": {"LLM_PROVIDER": "deepseek"},
    }

    record = REC.normalize_trend_record(raw)

    assert record["source"] == "trend_file"
    assert record["created_at"] == datetime(
        2026, 10, 31, 15, 21, 32, tzinfo=timezone.utc
    )
    # 旧趋势记录没有 threshold，用当前注册表补齐。
    assert record["metrics"]["agent.tool_accuracy"]["threshold"] == 0.9
    assert record["metrics"]["agent.latency_p95"]["threshold"] == 60000
    assert record["suite_verdicts"] == {"agent": "fail"}
    assert record["verdict"] == "fail"
    assert record["config"] == {"LLM_PROVIDER": "deepseek"}


def test_normalize_trend_record_all_not_run_is_not_pass():
    raw = {
        "run_id": "20261031T152132Z",
        "suites": ["rag"],
        "metrics": {"rag.faithfulness": {"value": None, "status": "not_run"}},
    }
    record = REC.normalize_trend_record(raw)
    assert record["suite_verdicts"] == {"rag": "not_run"}
    # not_run 不算失败（与 EvalReport.verdict 的既有语义一致）。
    assert record["verdict"] == "pass"


def test_normalize_trend_record_infers_suites_when_missing():
    raw = {
        "run_id": "x",
        "metrics": {
            "retrieval.precision": {"value": 0.9, "status": "pass"},
            "rag.faithfulness": {"value": 0.8, "status": "pass"},
        },
    }
    record = REC.normalize_trend_record(raw)
    assert record["suites"] == ["rag", "retrieval"]
    assert record["suite_verdicts"] == {"rag": "pass", "retrieval": "pass"}
    assert record["created_at"] is None


def test_normalize_trend_record_keeps_unknown_metrics():
    raw = {
        "run_id": "x",
        "metrics": {"agent.new_metric": {"value": 1, "status": "pass"}},
    }
    record = REC.normalize_trend_record(raw)
    assert record["metrics"]["agent.new_metric"]["threshold"] is None
