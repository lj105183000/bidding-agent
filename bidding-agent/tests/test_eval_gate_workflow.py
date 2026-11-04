"""Eval Gate workflow 的基线/缓存契约回归（只读 YAML 文本，不跑 Actions）。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "eval-gate.yml"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_restore_keys_are_suite_scoped_only():
    """禁止跨线 fallback：旧 `eval-trend-main-` 会把别的线快照覆盖到本线。"""
    lines = [line.strip() for line in _text().splitlines()]
    assert "eval-trend-${{ github.ref_name }}-${{ matrix.suite }}-" in lines
    assert "eval-trend-${{ github.ref_name }}-" not in lines


def test_gate_validates_baseline_presence():
    text = _text()
    assert "校验门禁基线" in text
    assert "no_baseline_reason" in text
    assert "save != 'true'" in text


def test_agent_multi_is_part_of_gate():
    text = _text()
    assert "matrix.suite == 'agent_multi'" in text
    assert "AGENT_MULTI_AGENT" in text
