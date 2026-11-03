"""`run_all.py --save-db` 的持久化与降级测试。

★ 关键断言
--------
数据库写入失败**必须**只告警、不改退出码。这是门禁可信度的前提：
基础设施故障不能被误报成“质量退化”。故用 fake store 直接制造失败，
并验证 `main()` 的调用门控（只有 `--save-db` 才写库）。
"""

from __future__ import annotations

import sys

import eval.run_all as RA
from eval import report as R
from eval import trend as T


def _report() -> R.EvalReport:
    suite = R.SuiteReport(suite="retrieval_offline", ran=True, case_count=1)
    suite.results = [
        R.MetricResult(
            name="retrieval_offline.top1_relevant_rate",
            value=1.0,
            status=R.MetricStatus.PASS,
            threshold=0.7,
        )
    ]
    return R.EvalReport(
        schema_version=R.SCHEMA_VERSION,
        run_id="20261031T152132Z",
        version="v1",
        suites=[suite],
    )


def test_persist_report_to_db_success_uses_injected_store():
    written = []

    def _store(record):
        written.append(record)
        return "inserted"

    assert RA.persist_report_to_db(
        _report(), source="api", job_id="job-1", store=_store
    )
    assert written[0]["source"] == "api"
    assert written[0]["job_id"] == "job-1"
    assert written[0]["metrics"]["retrieval_offline.top1_relevant_rate"]["value"] == 1.0


def test_persist_report_to_db_failure_is_non_fatal(capsys):
    def _store(record):
        raise RuntimeError("database down")

    assert RA.persist_report_to_db(_report(), store=_store) is False
    output = capsys.readouterr().out
    assert "评测数据库写入失败" in output
    assert "不影响评测结果与门禁" in output


def test_env_run_id_accepts_safe_value(monkeypatch):
    monkeypatch.setenv("EVAL_RUN_ID", "20261103T120000Z-a1b2c3d4")
    assert RA._env_run_id() == "20261103T120000Z-a1b2c3d4"


def test_env_run_id_rejects_unsafe_value(monkeypatch, capsys):
    monkeypatch.setenv("EVAL_RUN_ID", "../evil path")
    assert RA._env_run_id() == ""
    assert "忽略非法的 EVAL_RUN_ID" in capsys.readouterr().out


def test_env_run_source_normalizes_unknown(monkeypatch):
    monkeypatch.setenv("EVAL_RUN_SOURCE", "API")
    assert RA._env_run_source() == "api"
    monkeypatch.setenv("EVAL_RUN_SOURCE", "something")
    assert RA._env_run_source() == "cli"


def _patch_main(monkeypatch, report, persist_calls):
    monkeypatch.setattr(RA, "setup_stdio", lambda: None)
    monkeypatch.setattr(RA, "build_report", lambda *a, **k: report)
    monkeypatch.setattr(RA, "render_text", lambda rep: "")
    monkeypatch.setattr(
        RA,
        "persist_report_to_db",
        lambda *a, **k: persist_calls.append((a, k)) or True,
    )
    monkeypatch.setattr(T, "decide_exit_code", lambda *a, **k: (0, {}))
    monkeypatch.setattr(T, "render_exit_decision", lambda detail: "")


def test_main_save_db_writes_without_trend(monkeypatch, capsys):
    calls: list = []
    _patch_main(monkeypatch, _report(), calls)
    monkeypatch.setattr(
        sys, "argv", ["run_all", "--suites", "retrieval_offline", "--save-db"]
    )

    assert RA.main() == 0
    assert len(calls) == 1
    # 只传 --save-db，不应打印“已存入趋势文件”
    assert "已存入趋势文件" not in capsys.readouterr().out


def test_main_without_save_db_does_not_write(monkeypatch):
    calls: list = []
    _patch_main(monkeypatch, _report(), calls)
    monkeypatch.setattr(sys, "argv", ["run_all", "--suites", "retrieval_offline"])

    assert RA.main() == 0
    assert calls == []
