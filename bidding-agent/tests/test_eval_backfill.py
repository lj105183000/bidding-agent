"""`eval/backfill_db.py` 测试：dry-run、去重、幂等参数与失败退出码。"""

from __future__ import annotations

import json
import sys

import eval.backfill_db as B
from src.eval_runs import store as S


def _write_trend(path, records):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _raw(run_id: str, value: float = 0.9) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "version": "v1",
        "suites": ["agent"],
        "case_counts": {"agent": 5},
        "metrics": {"agent.tool_accuracy": {"value": value, "status": "pass"}},
        "config": {},
    }


def test_build_records_dedupes_and_normalizes(tmp_path):
    path = tmp_path / "trend.jsonl"
    _write_trend(path, [_raw("20261031T152132Z", 0.8), _raw("20261031T152132Z", 0.95)])

    records = B.build_records(path)

    assert len(records) == 1
    assert records[0]["run_id"] == "20261031T152132Z"
    assert records[0]["source"] == "backfill"
    # 重复 run_id 保留最后一条
    assert records[0]["metrics"]["agent.tool_accuracy"]["value"] == 0.95
    assert records[0]["created_at"] is not None


def test_main_dry_run_does_not_touch_store(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trend.jsonl"
    _write_trend(path, [_raw("20261031T152132Z")])

    called = {"n": 0}
    monkeypatch.setattr(
        S, "save_records", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    )
    monkeypatch.setattr(sys, "argv", ["backfill_db", "--trend", str(path), "--dry-run"])

    assert B.main() == 0
    assert called["n"] == 0
    assert "dry-run" in capsys.readouterr().out


def test_main_writes_and_reports_counts(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trend.jsonl"
    _write_trend(path, [_raw("20261031T152132Z")])
    captured = {}

    def _save(records, *, overwrite_backfill=False):
        captured["records"] = records
        captured["overwrite"] = overwrite_backfill
        return {"inserted": 1, "skipped": 0, "overwritten": 0, "protected": 0}

    monkeypatch.setattr(S, "save_records", _save)
    monkeypatch.setattr(
        sys, "argv", ["backfill_db", "--trend", str(path), "--overwrite"]
    )

    assert B.main() == 0
    assert captured["overwrite"] is True
    assert len(captured["records"]) == 1
    assert "新增 1" in capsys.readouterr().out


def test_main_returns_one_on_store_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trend.jsonl"
    _write_trend(path, [_raw("20261031T152132Z")])

    def _boom(*args, **kwargs):
        raise RuntimeError("database down")

    monkeypatch.setattr(S, "save_records", _boom)
    monkeypatch.setattr(sys, "argv", ["backfill_db", "--trend", str(path)])

    assert B.main() == 1
    assert "回填失败" in capsys.readouterr().out


def test_main_returns_one_when_trend_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["backfill_db", "--trend", str(tmp_path / "missing.jsonl")]
    )
    assert B.main() == 1
    assert "趋势文件不存在" in capsys.readouterr().out
