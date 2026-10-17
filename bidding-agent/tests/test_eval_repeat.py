"""`--repeat N` 与中位聚合的回归测试（`docs/93` 第 3 条的修法）。

★ 要锁住的行为
==========
1. 中位数**抗离群**（那正是选它而不选均值的理由）
2. `NOT_RUN` **不参与**中位 —— 它是"没测到"，不是"测出 0"
3. 状态要**按中位重新判定**，不能沿用某一次的
4. `repeat>1` 必须记进 `config` —— 否则"3 次中位"会被当成基线去比"1 次实测"
5. 用例数取**最小值**（各次不同时最保守）
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.repeat import merge_suite_runs  # noqa: E402
from eval.report import (  # noqa: E402
    MetricStatus,
    SuiteReport,
    get_spec,
    make_result,
)


def _suite(values: list[float | None], *, name: str = "agent.tool_accuracy",
           case_count: int = 10, ran: bool = True) -> SuiteReport:
    sr = SuiteReport(suite="agent", ran=ran, case_count=case_count)
    for v in values:
        sr.results.append(make_result(name, v))
    return sr


class TestMedianNotMean:
    def test_median_resists_outlier(self):
        """★ 核心：中位抗离群 —— 选它而不选均值的全部理由。

        0.90 / 0.92 / 0.10（一次跑飞）：
            均值 = 0.64  ← 被离群值拉到门槛下
            中位 = 0.90  ← 不受影响
        若这里用均值，一次跑飞就会让门禁误报退化。
        """
        runs = [
            _suite([0.90]),
            _suite([0.92]),
            _suite([0.10]),        # 一次跑飞
        ]
        merged = merge_suite_runs(runs)
        v = merged.results[0].value
        assert v == pytest.approx(0.90), f"中位应为 0.90，实为 {v}"
        mean = statistics.mean([0.90, 0.92, 0.10])
        assert abs(mean - v) > 0.2, "均值与中位应显著不同（证明测试有效）"

    def test_note_keeps_all_values_visible(self):
        """note 必须列出各次实值 —— 否则看不出波动有多大。"""
        merged = merge_suite_runs([_suite([0.90]), _suite([0.10]), _suite([0.92])])
        note = merged.results[0].note
        assert "3 次中位" in note
        for v in ("0.1", "0.9", "0.92"):
            assert v in note, f"note 缺实值 {v}：{note}"

    def test_even_count_takes_middle_two_average(self):
        """偶数次时 statistics.median 取中间两个的均值 —— 明确钉住口径。"""
        merged = merge_suite_runs([_suite([0.80]), _suite([0.90])])
        assert merged.results[0].value == pytest.approx(0.85)


class TestNotRunExcluded:
    def test_not_run_does_not_drag_median(self):
        """★ NOT_RUN 是"没测到"，不能当 0 参与中位。

        若把 NOT_RUN 当 0：中位会变成 (0.90, 0.92, 0) → 0.90，
        看起来没差别；但只有 2 次有效 + 1 次 NOT_RUN 时：
            当 0 参与 → 中位 0.45（被拉低一半）
            不参与   → 中位 0.91
        下面用 2 个有效 + 1 个 NOT_RUN 暴露这个差异。
        """
        runs = [
            _suite([0.90]),
            _suite([0.92]),
            _suite([None]),        # NOT_RUN
        ]
        merged = merge_suite_runs(runs)
        v = merged.results[0].value
        assert v == pytest.approx(0.91), (
            f"中位应为两个有效值的中位 0.91，实为 {v} —— "
            "说明 NOT_RUN 被当成 0 参与了聚合"
        )
        assert "1 次未测未计入" in merged.results[0].note

    def test_all_not_run_stays_not_run(self):
        """全 NOT_RUN → 仍是 NOT_RUN，不冒充 0。"""
        merged = merge_suite_runs([_suite([None]), _suite([None])])
        r = merged.results[0]
        assert r.value is None
        assert r.status is MetricStatus.NOT_RUN
        assert "均为未测" in r.note


class TestStatusRecomputed:
    def test_status_follows_median_not_any_single_run(self):
        """★ 状态必须按**中位**重新判定。

        `agent.tool_accuracy` 门槛 0.90（HIGHER_IS_BETTER）。
        三次 0.70 / 0.95 / 0.96 → 中位 0.95 → **PASS**。
        但首次是 FAIL、末次是 PASS —— 若沿用"第一次"的状态就会错判。
        """
        runs = [
            _suite([0.70]),        # FAIL
            _suite([0.95]),        # PASS
            _suite([0.96]),        # PASS
        ]
        merged = merge_suite_runs(runs)
        spec = get_spec("agent.tool_accuracy")
        assert merged.results[0].value == pytest.approx(0.95)
        assert merged.results[0].status is spec.judge(0.95)
        assert merged.results[0].status is MetricStatus.PASS

    def test_status_can_flip_to_fail_if_median_fails(self):
        """反向：中位低于门槛时必须是 FAIL（即使有一次达标）。"""
        runs = [
            _suite([0.99]),        # PASS
            _suite([0.80]),        # FAIL
            _suite([0.81]),        # FAIL
        ]
        merged = merge_suite_runs(runs)
        assert merged.results[0].value == pytest.approx(0.81)
        assert merged.results[0].status is MetricStatus.FAIL


class TestSuiteMetadata:
    def test_case_count_takes_minimum(self):
        """各次用例数不同时取**最小** —— 最保守。"""
        merged = merge_suite_runs([
            _suite([0.9], case_count=18),
            _suite([0.9], case_count=3),
            _suite([0.9], case_count=18),
        ])
        assert merged.case_count == 3

    def test_ran_true_if_any_run_ran(self):
        a = _suite([0.9], ran=True)
        b = _suite([0.9], ran=False)
        assert merge_suite_runs([a, b]).ran is True

    def test_note_marks_repeat_count(self):
        merged = merge_suite_runs([_suite([0.9]), _suite([0.9])])
        assert "2 次取中位" in merged.note

    def test_empty_runs_does_not_crash(self):
        merged = merge_suite_runs([])
        assert merged.ran is False
        assert merged.results == []


class TestConfigRecordsRepeat:
    def test_repeat_goes_into_config(self):
        """★ `repeat>1` 必须进 config。

        趋势文件按 `config` 判可比。若"3 次中位"与"1 次实测"config 相同，
        门禁会把它们当可比 —— 而它们的分数量纲不同（中位更平滑）。
        """
        from eval.run_all import build_report

        rep = build_report([], limit=0, web_mode="off", repeat=3)
        assert rep.config.get("REPEAT") == 3

    def test_single_run_has_no_repeat_key(self):
        """`repeat=1` 时不加该键 —— 保持旧趋势文件的 config 形状不变。"""
        from eval.run_all import build_report

        rep = build_report([], limit=0, web_mode="off", repeat=1)
        assert "REPEAT" not in rep.config
