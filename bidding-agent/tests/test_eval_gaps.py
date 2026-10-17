"""评测体系「补充」的回归测试：基线缺口、分线退出码、阈值标定、分母陷阱、数据集重叠。

对应七条局限（逐条一组）：

    ① rag 线没有基线时，门禁要说清**是哪条线缺**（而不是笼统"没有历史"）
    ② 回填推断缺一条线的**具体情形**要能识别（跑了但用例数为 0）
    ③/⑦ `--fail-on` / `--allow-fail`：分线红（拆 CI job 的前提）
    ④ 门槛只跑过一遍 → 标定工具要能说"样本不足，还差几次"
    ⑤ `first_relevant_rank_mean` 的分母陷阱 → 单列"前 5 条全不切题比例"
    ⑥ 两套（实为三份）评测数据的重叠与不可比性要进报告

★ 这一组测试的共同点：被测的都是**判定逻辑**，出错时不会报错，
只会让评估结论悄悄变错（例如"已忽略的线"被静默吞掉 → CI 永远绿）。
故每一条都用构造数据把边界钉死，不依赖真实检索/LLM。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import dataset_registry as DS  # noqa: E402
from eval import report as R  # noqa: E402
from eval import trend as T  # noqa: E402
from eval.calibrate import (  # noqa: E402
    MIN_RUNS_FOR_THRESHOLD,
    _verdict,
    calibration,
    collect_values,
    describe,
    suggest,
)
from eval.run_all import parse_suite_list, ranking_metrics  # noqa: E402


def _suite(name: str, metrics: list[tuple[str, float | None]], case_count: int = 5):
    sr = R.SuiteReport(suite=name, ran=True, case_count=case_count)
    sr.results = [R.make_result(n, v) for n, v in metrics]
    return sr


def _report(suites: list[R.SuiteReport], run_id: str = "cur") -> R.EvalReport:
    return R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id=run_id, version="v", suites=suites)


# ===========================================================================
# ① 分线基线：说清"缺哪条线的基线"，并给出建立命令
# ===========================================================================


class TestPerSuiteBaseline:
    def test_latest_baseline_per_suite_ignores_other_suites(self):
        """按线找基线：**不能**取"最近一次运行"，要取最近一次**含该线**的运行。"""
        history = [
            {"schema_version": R.SCHEMA_VERSION, "run_id": "a1", "suites": ["agent"],
             "metrics": {"agent.tool_accuracy": {"value": 1.0}}},
            {"schema_version": R.SCHEMA_VERSION, "run_id": "r1", "suites": ["retrieval"],
             "metrics": {"retrieval.precision": {"value": 0.9}}},
            {"schema_version": R.SCHEMA_VERSION, "run_id": "a2", "suites": ["agent"],
             "metrics": {"agent.tool_accuracy": {"value": 0.8}}},
        ]
        assert T.latest_baseline_for(history, "agent", R.SCHEMA_VERSION) == "a2"
        assert T.latest_baseline_for(history, "retrieval", R.SCHEMA_VERSION) == "r1"
        assert T.latest_baseline_for(history, "rag", R.SCHEMA_VERSION) == ""

    def test_suite_baselines_reported_for_ran_suites(self):
        history = [
            {"schema_version": R.SCHEMA_VERSION, "run_id": "a1", "suites": ["agent"],
             "metrics": {"agent.tool_accuracy": {"value": 1.0}}},
        ]
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.9)]),
                       _suite("rag", [("rag.faithfulness", 0.8)])])
        cmp = T.compare_with_previous(rep, history)
        assert cmp.suite_baselines == {"agent": "a1", "rag": ""}

    def test_no_baseline_render_names_missing_suite_and_command(self):
        """★ 只写"没有基线"没用 —— 要指出**缺哪条线**并给出建立命令。

        这正是 `--suites rag --gate` 首次运行时的场景：不是"趋势文件坏了"，
        而是"rag 这条线从来没存档过"。
        """
        rep = _report([_suite("rag", [("rag.faithfulness", 0.8)])])
        cmp = T.compare_with_previous(rep, [])
        out = cmp.render()
        assert "分线基线" in out
        assert "rag" in out and "无基线" in out
        assert "--suites rag --save" in out, "必须给出建立基线的命令"

    def test_no_baseline_render_still_avoids_noise(self):
        """既有纪律不能破：没有基线时不罗列 missing（那是噪音）。"""
        rep = _report([_suite("rag", [("rag.faithfulness", 0.8)])])
        out = T.compare_with_previous(rep, []).render()
        assert "本次未测" not in out
        assert "不是退化" in out

    def test_all_suites_from_history(self):
        history = [
            {"suites": ["agent", "retrieval"], "case_counts": {"agent": 3}},
            {"suites": ["rag"]},
        ]
        assert T.all_suites(history) == ["agent", "rag", "retrieval"]


# ===========================================================================
# ② 回填推断：把"少一条线"的具体情形识别出来
# ===========================================================================


class TestInferredSuitesGap:
    def test_metric_prefixes_reveal_ran_suite(self):
        """`metrics` 的键前缀能反推跑过哪些线（即使 0 用例也留痕）。"""
        from eval.migrate_suites import metric_suite_prefixes

        rec = {"metrics": {"rag.faithfulness": {"value": None},
                           "agent.tool_accuracy": {"value": 0.9}}}
        assert metric_suite_prefixes(rec) == {"agent", "rag"}

    def test_gap_is_suite_with_metrics_but_no_case_count(self):
        """★ 这就是"跑了但用例数为 0"的形态：有指标键、却没有 case_count。"""
        from eval.migrate_suites import infer_gap

        rec = {
            "case_counts": {"agent": 3},
            "metrics": {"agent.tool_accuracy": {"value": 0.9},
                        "rag.faithfulness": {"value": None}},
        }
        assert infer_gap(rec) == ["rag"]

    def test_infer_suites_unions_both_sources(self):
        """并集：`case_counts` 的键 ∪ `metrics` 的键前缀。"""
        from eval.migrate_suites import infer_suites

        rec = {
            "case_counts": {"agent": 3},
            "metrics": {"agent.tool_accuracy": {"value": 0.9},
                        "rag.faithfulness": {"value": None}},
        }
        assert infer_suites(rec) == ["agent", "rag"]

    def test_infer_suites_keeps_old_behavior_without_metrics(self):
        """没有 `metrics` 时与第一版一致（既有测试的契约）。"""
        from eval.migrate_suites import infer_suites

        assert infer_suites({"case_counts": {"agent": 5}}) == ["agent"]
        assert infer_suites({"case_counts": {}}) is None
        assert infer_suites({}) is None
        assert infer_suites({"case_counts": None}) is None

    def test_migration_records_source_and_gap(self, tmp_path):
        """回填要写 `suites_source: inferred` 与 `suites_inferred_gap`。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        p.write_text(
            json.dumps({
                "schema_version": T.SCHEMA_VERSION, "run_id": "old",
                "metrics": {"agent.tool_accuracy": {"value": 0.9},
                            "rag.faithfulness": {"value": None}},
                "case_counts": {"agent": 3},
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stat = migrate(p, apply=True)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["suites"] == ["agent", "rag"], "漏掉的 rag 应被补回"
        assert rec["suites_source"] == "inferred"
        assert rec["suites_inferred"] is True
        assert rec["suites_inferred_gap"] == ["rag"]
        assert stat["gap_records"] == 1
        assert stat["gap_suites"] == ["rag"]

    def test_migration_omits_gap_key_when_consistent(self, tmp_path):
        """没有漏项时不写 gap 键（避免全是噪音）。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        p.write_text(
            json.dumps({
                "schema_version": T.SCHEMA_VERSION, "run_id": "old",
                "metrics": {"agent.tool_accuracy": {"value": 0.9}},
                "case_counts": {"agent": 3},
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        migrate(p, apply=True)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert "suites_inferred_gap" not in rec
        assert rec["suites_source"] == "inferred"

    def test_new_records_are_marked_observed(self):
        """新记录必须写 `observed`，与回填值一眼可分。"""
        rec = T.report_to_record(_report([_suite("agent", [("agent.tool_accuracy", 0.9)])]))
        assert rec["suites_source"] == "observed"
        assert "suites_inferred" not in rec

    def test_gate_uses_inferred_baseline_but_warns(self):
        """推断基线**照用**（否则回填白做），但报告要提示它是推断的。"""
        history = [{
            "schema_version": R.SCHEMA_VERSION, "run_id": "old",
            "suites": ["agent"], "suites_source": "inferred",
            "suites_inferred_gap": ["rag"],
            "case_counts": {"agent": 5},
            "metrics": {"agent.tool_accuracy": {"value": 0.95, "status": "pass"}},
        }]
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.90)])], )
        cmp = T.compare_with_previous(rep, history)
        assert cmp.baseline_run_id == "old"
        assert cmp.baseline_suites_source == "inferred"
        out = cmp.render()
        assert "回填推断" in out
        assert "rag" in out, "要指出疑似漏项"

    def test_legacy_record_without_source_is_treated_as_inferred(self):
        """老记录只有 `suites_inferred: true`（无 source）时同样按推断处理。"""
        history = [{
            "schema_version": R.SCHEMA_VERSION, "run_id": "old",
            "suites": ["agent"], "suites_inferred": True,
            "case_counts": {"agent": 5},
            "metrics": {"agent.tool_accuracy": {"value": 0.95, "status": "pass"}},
        }]
        cmp = T.compare_with_previous(
            _report([_suite("agent", [("agent.tool_accuracy", 0.90)])]), history
        )
        assert cmp.baseline_suites_source == "inferred"


# ===========================================================================
# ③ / ⑦ 分线退出码：--fail-on / --allow-fail
# ===========================================================================


class TestDecideExitCode:
    def _hist(self, value: float = 0.95, case_count: int = 5) -> list[dict]:
        return [T.report_to_record(
            R.EvalReport(
                schema_version=R.SCHEMA_VERSION, run_id="base", version="v1",
                suites=[_suite("agent", [("agent.tool_accuracy", value)], case_count)],
            )
        )]

    def test_pass_gives_zero(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.95)])])
        code, detail = T.decide_exit_code(rep, None)
        assert code == 0
        assert detail["selected_suites"] == ["agent"]

    def test_threshold_failure_gives_one(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        code, detail = T.decide_exit_code(rep, None)
        assert code == 1
        assert detail["metric_failed_suites"] == ["agent"]

    def test_regression_gives_two(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        assert cmp.regressed
        code, detail = T.decide_exit_code(rep, cmp)
        assert code == 2
        assert detail["regressed_suites"] == ["agent"]

    def test_regression_beats_threshold_failure(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50),
                                        ("agent.latency_p95", 99999.0)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        code, _ = T.decide_exit_code(rep, cmp)
        assert code == 2

    def test_fail_on_scopes_the_verdict(self):
        """★ `--fail-on retrieval`：agent 退化**不**让退出码非零。

        这正是"哪条线挂了就单独红"的实现方式：
        CI 每条线一个 job，各自 `--fail-on <自己>`。
        """
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        code, detail = T.decide_exit_code(rep, cmp, fail_on=["retrieval"])
        assert code == 0
        assert detail["ignored_suites"] == ["agent"], "被忽略的线必须点名，不能静默吞掉"
        assert detail["in_scope_regressed_suites"] == []

    def test_fail_on_keeps_own_suite_red(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        code, _ = T.decide_exit_code(rep, cmp, fail_on=["agent"])
        assert code == 2

    def test_allow_fail_excludes_suite(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        code, detail = T.decide_exit_code(rep, cmp, allow_fail=["agent"])
        assert code == 0
        assert detail["ignored_suites"] == ["agent"]

    def test_allow_fail_still_fails_other_suites(self):
        """`--allow-fail agent` 只放过 agent，其它线照常红。"""
        rep = _report([
            _suite("agent", [("agent.tool_accuracy", 0.50)]),
            _suite("retrieval", [("retrieval.precision", 0.10)]),
        ])
        code, detail = T.decide_exit_code(rep, None, allow_fail=["agent"])
        assert code == 1
        assert detail["selected_suites"] == ["retrieval"]
        assert detail["in_scope_failed_suites"] == ["retrieval"]

    def test_both_flags_rejected(self):
        code_or_error = None
        try:
            T.decide_exit_code(_report([_suite("agent", [("agent.tool_accuracy", 0.9)])]),
                               None, fail_on=["agent"], allow_fail=["rag"])
        except ValueError as exc:
            code_or_error = str(exc)
        assert code_or_error and "不能同时使用" in code_or_error

    def test_unknown_suite_rejected(self):
        """★ 拼错线名必须报错：静默失效会让门禁**永远绿**。"""
        with pytest.raises(ValueError) as ei:
            T.decide_exit_code(_report([_suite("agent", [("agent.tool_accuracy", 0.9)])]),
                               None, fail_on=["retrival"])
        assert "未知的评测线" in str(ei.value)

    def test_case_count_mismatch_blocks_regression(self):
        """样本量不一致时不算退化 → 退出码不该是 2（既有纪律）。"""
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.10)], case_count=3)])
        cmp = T.compare_with_previous(rep, self._hist(0.95, case_count=17))
        assert cmp.case_count_mismatch
        code, detail = T.decide_exit_code(rep, cmp)
        assert code == 1, "退化不成立，但未达门槛（0.10 < 0.90）仍应为 1"
        assert detail["regressed_suites"] == []

    def test_detail_has_ci_fields(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.95)])])
        _, detail = T.decide_exit_code(rep, None)
        for key in ("exit_code", "ran_suites", "selected_suites", "ignored_suites",
                    "regressed_suites", "metric_failed_suites", "fail_on", "allow_fail"):
            assert key in detail

    def test_render_exit_decision_mentions_code_and_ignored(self):
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.50)])])
        cmp = T.compare_with_previous(rep, self._hist(0.95))
        _, detail = T.decide_exit_code(rep, cmp, fail_on=["retrieval"])
        text = T.render_exit_decision(detail)
        assert "退出码 0" in text
        assert "已忽略" in text and "agent" in text

    def test_suite_of_registered_and_unknown(self):
        assert T.suite_of("agent.tool_accuracy") == "agent"
        assert T.suite_of("retrieval.no_relevant_in_topk_rate") == "retrieval"
        assert T.suite_of("togo.something") == "togo", "未注册指标退回前缀"


class TestParseSuiteList:
    def test_empty_means_not_specified(self):
        assert parse_suite_list("", "--fail-on") == []
        assert parse_suite_list("   ", "--fail-on") == []

    def test_valid_list(self):
        assert parse_suite_list("agent,retrieval", "--fail-on") == ["agent", "retrieval"]
        assert parse_suite_list(" agent , rag ", "--allow-fail") == ["agent", "rag"]

    def test_unknown_raises(self):
        with pytest.raises(SystemExit) as ei:
            parse_suite_list("bogus", "--fail-on")
        assert "未知的评测线" in str(ei.value)


# ===========================================================================
# ④ 门槛标定
# ===========================================================================


def _hist_record(run_id: str, metrics: dict, *, config: dict | None = None,
                 case_counts: dict | None = None, suites: list[str] | None = None) -> dict:
    return {
        "schema_version": R.SCHEMA_VERSION,
        "run_id": run_id,
        "config": config if config is not None else {"model": "m"},
        "suites": suites or ["retrieval"],
        "case_counts": case_counts if case_counts is not None else {"retrieval": 18},
        "metrics": {k: {"value": v, "status": "pass"} for k, v in metrics.items()},
    }


class TestCalibration:
    def test_collect_skips_other_configs(self):
        """★ 换了模型/参数的观测属于另一个分布，不能混进同一条标定带。"""
        history = [
            _hist_record("a", {"retrieval.precision": 0.9}, config={"model": "m"}),
            _hist_record("b", {"retrieval.precision": 0.5}, config={"model": "other"}),
            _hist_record("c", {"retrieval.precision": 0.92}, config={"model": "m"}),
        ]
        assert collect_values(history, "retrieval.precision") == [0.9, 0.92]

    def test_collect_skips_other_case_counts(self):
        """★ `--limit 3` 与 18 题的分数不可比（实测踩过的坑）。"""
        history = [
            _hist_record("a", {"retrieval.precision": 1.0}, case_counts={"retrieval": 18}),
            _hist_record("b", {"retrieval.precision": 0.3}, case_counts={"retrieval": 3}),
            _hist_record("c", {"retrieval.precision": 0.99}, case_counts={"retrieval": 18}),
        ]
        assert collect_values(history, "retrieval.precision") == [1.0, 0.99]

    def test_collect_skips_runs_without_the_suite(self):
        history = [
            _hist_record("a", {"retrieval.precision": 0.9}, suites=["agent"]),
            _hist_record("b", {"retrieval.precision": 0.95}, suites=["retrieval"]),
        ]
        assert collect_values(history, "retrieval.precision") == [0.95]

    def test_describe_marks_unknown_stdev(self):
        """样本 < 2 时标准差无意义 → None（不是 0）。"""
        assert describe([])["n"] == 0
        assert describe([0.9])["stdev"] is None
        assert describe([0.9, 1.0])["stdev"] is not None

    def test_suggest_refuses_with_too_few_runs(self):
        """★ 只有一两次观测时**不给建议值** —— 给出比不给更危险。"""
        values = [0.9444]
        value, why = suggest(values, R.Direction.HIGHER_IS_BETTER, 0.80)
        assert value is None
        assert f"{MIN_RUNS_FOR_THRESHOLD}" in why
        assert "样本不足" in why

    def test_suggest_higher_is_better_uses_control_limit(self):
        values = [1.0, 0.96, 0.98, 1.0, 0.94]
        value, why = suggest(values, R.Direction.HIGHER_IS_BETTER, 0.80)
        assert value is not None
        assert value <= min(values), "门槛不得高于观测最低值（否则比历史最差还松）"
        assert "μ−2σ" in why

    def test_suggest_lower_is_better(self):
        values = [1.0, 1.2, 1.1, 1.3, 0.9]
        value, why = suggest(values, R.Direction.LOWER_IS_BETTER, 1.6)
        assert value is not None
        assert value >= max(values), "门槛不得低于观测最高值"
        assert "μ+2σ" in why

    def test_suggest_no_threshold_metric(self):
        value, why = suggest([1.0] * 6, R.Direction.HIGHER_IS_BETTER, None)
        assert value is None and "观测指标" in why

    @pytest.mark.parametrize(
        "n,threshold,direction,expect",
        [
            (0, 0.8, R.Direction.HIGHER_IS_BETTER, "no_data"),
            (1, 0.8, R.Direction.HIGHER_IS_BETTER, "insufficient_samples"),
            (6, None, R.Direction.HIGHER_IS_BETTER, "no_threshold"),
        ],
    )
    def test_verdict_short_circuits(self, n, threshold, direction, expect):
        stats = describe([0.9] * n)
        assert _verdict(stats, threshold, direction) == expect

    def test_verdict_too_loose_and_tight(self):
        stats = describe([0.96, 0.97, 0.98, 0.99, 1.0, 0.95])
        assert _verdict(stats, 0.80, R.Direction.HIGHER_IS_BETTER) == "too_loose", (
            "门槛低于观测最低值 → 比历史最差还松"
        )
        assert _verdict(stats, 0.97, R.Direction.HIGHER_IS_BETTER) == "ok", (
            "落在观测区间内才算合理"
        )
        # 比例类指标不可能 > 1.0，故用另一组"最大值还不到 1"的观测来测"偏紧"
        lower = describe([0.90, 0.92, 0.94, 0.91, 0.93])
        assert _verdict(lower, 0.99, R.Direction.HIGHER_IS_BETTER) == "too_tight", (
            "门槛高于历史最好 → 正常波动就会报警"
        )
        assert _verdict(lower, 0.92, R.Direction.HIGHER_IS_BETTER) == "ok"

    def test_calibration_covers_retrieval_ordering_metrics(self):
        """★ ④ 的核心：那两个"只跑过一遍就定门槛"的指标必须被点名样本不足。"""
        history = [_hist_record("only-one", {
            "retrieval.top1_relevant_rate": 0.9444,
            "retrieval.first_relevant_rank_mean": 1.06,
        })]
        data = calibration(history)
        by_name = {i["name"]: i for i in data["items"]}
        for name in ("retrieval.top1_relevant_rate", "retrieval.first_relevant_rank_mean"):
            item = by_name[name]
            assert item["observed"]["n"] == 1
            assert item["suggested"] is None
            assert item["verdict"] == "insufficient_samples"

    def test_calibration_json_serializable(self):
        data = calibration([_hist_record("a", {"retrieval.precision": 0.9})])
        json.dumps(data, ensure_ascii=False)  # 不该抛


# ===========================================================================
# ⑤ 分母陷阱：前 5 条全不切题必须单独成指标
# ===========================================================================


class TestRankingMetrics:
    def test_no_hit_topics_do_not_move_the_mean(self):
        """★ 分母陷阱的实证：一条都没切题的题**不拉高**位次均值。

        若只看 `first_relevant_rank_mean`，下面这组数据的均值是 2.0（看起来还行），
        而真相是 3 个主题里有 1 个前 5 条完全找不到切题文档。
        """
        result = ranking_metrics([
            ([{"question": "切题A"}, {"question": "无关"}], ("切题A",)),
            ([{"question": "无关"}, {"question": "x"}, {"question": "切题B"}], ("切题B",)),
            ([{"question": "无关"}, {"question": "y"}], ("切题C",)),
        ])
        assert result["ranked_total"] == 3
        assert result["first_ranks"] == [1, 3]
        assert sum(result["first_ranks"]) / len(result["first_ranks"]) == 2.0
        assert result["no_hit"] == 1
        assert result["no_hit"] / result["ranked_total"] == pytest.approx(1 / 3)

    def test_all_topics_hit_gives_zero_rate(self):
        result = ranking_metrics([
            ([{"question": "a"}], ("a",)),
            ([{"question": "b"}], ("b",)),
        ])
        assert result["no_hit"] == 0
        assert result["top1_ok"] == 2

    def test_all_topics_miss_gives_no_mean(self):
        """全部不切题时 `first_ranks` 为空 → 位次均值必须是"未测"而不是 0。"""
        result = ranking_metrics([
            ([{"question": "x"}], ("a",)),
            ([{"question": "y"}], ("b",)),
        ])
        assert result["first_ranks"] == []
        assert result["no_hit"] == 2
        assert result["top1_ok"] == 0

    def test_empty_result_sets_are_skipped(self):
        """检索返回空的主题不算"有结果的主题"（否则比率会被稀释）。"""
        result = ranking_metrics([([], ("a",)), ([{"question": "a"}], ("a",))])
        assert result["ranked_total"] == 1

    def test_empty_input_no_zero_division(self):
        result = ranking_metrics([])
        assert result == {"ranked_total": 0, "top1_ok": 0, "first_ranks": [], "no_hit": 0}

    def test_answer_field_also_matches(self):
        result = ranking_metrics([([{"question": "", "answer": "切题"}], ("切题",))])
        assert result["top1_ok"] == 1

    def test_metric_spec_registered(self):
        """新指标必须登记（否则 `make_result` 记成未注册 → 永远不判定）。"""
        spec = R.get_spec("retrieval.no_relevant_in_topk_rate")
        assert spec is not None
        assert spec.direction is R.Direction.LOWER_IS_BETTER
        assert spec.threshold == 0.20
        assert spec.rationale, "门槛必须有留痕"
        # 与首位切题率互补：都在同一份 note 的分母上做文章
        assert R.get_spec("retrieval.first_relevant_rank_mean") is not None

    def test_new_metric_judgement(self):
        assert R.make_result("retrieval.no_relevant_in_topk_rate", 0.10).status is R.MetricStatus.PASS
        assert R.make_result("retrieval.no_relevant_in_topk_rate", 0.40).status is R.MetricStatus.FAIL


# ===========================================================================
# ⑥ 数据集：三份数据、重叠 12 条、不可跨集比
# ===========================================================================


class TestDatasets:
    def test_registry_covers_all_suites(self):
        """每条评测线都要登记，否则报告里会出现"未知数据来源"。"""
        from eval.run_all import ALL_SUITES

        assert set(DS.DATASETS) == set(ALL_SUITES)

    def test_counts_and_overlap_are_as_measured(self):
        """实测值：Agent 18 条、RAG 25 条、交集 12 条。"""
        summary = DS.summary()
        entry = summary["datasets"]
        assert entry["agent"]["cases"] == 18
        assert entry["rag"]["cases"] == 25
        assert entry["retrieval"]["cases"] == 18
        pair = next(p for p in summary["overlaps"] if p["pair"] == ["agent", "rag"])
        assert pair["shared"] == 12
        assert pair["only_a"] == 6
        assert pair["only_b"] == 13

    def test_shared_questions_have_identical_references(self):
        """交集里同题必须同参考答案 —— 否则横向比分数是拿两把尺子量同一根棍子。"""
        pair = next(p for p in DS.summary()["overlaps"] if p["pair"] == ["agent", "rag"])
        assert pair["reference_mismatch"] == 0, (
            f"发现同题不同参考答案：{pair['mismatch_examples']}"
        )

    def test_caveat_says_scores_are_not_comparable(self):
        summary = DS.summary()
        assert summary["caveats"], "必须给出不可比的提醒"
        assert any("不可直接横向比较" in c for c in summary["caveats"])

    def test_render_shows_sources_and_caveat(self):
        text = DS.render(DS.summary())
        assert "test_cases.json" in text
        assert "test_questions.json" in text
        assert "retrieval_precision.py" in text, "内联数据也要标出来源"
        assert "重叠" in text

    def test_fingerprint_stable_then_changes(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
        first = DS.fingerprint(p)
        assert first["sha256_12"] and first["bytes"] > 0
        assert DS.fingerprint(p) == first, "内容没变时指纹必须稳定"
        p.write_text(json.dumps([{"question": "q"}, {"question": "q2"}]), encoding="utf-8")
        assert DS.fingerprint(p)["sha256_12"] != first["sha256_12"]

    def test_fingerprint_missing_file_is_empty_not_error(self, tmp_path):
        assert DS.fingerprint(tmp_path / "nope.json") == {"sha256_12": "", "bytes": 0}

    def test_overlap_detects_reference_mismatch(self, tmp_path, monkeypatch):
        """同题不同参考答案必须能被检出（这是"两套集"最危险的情形）。"""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps([{"question": "q1", "reference": "甲"}] * 1), encoding="utf-8")
        b.write_text(json.dumps([{"question": "q1", "reference": "乙"}]), encoding="utf-8")
        monkeypatch.setitem(DS.DATASETS, "agent", DS.DatasetInfo(
            suite="agent", label="A", source_kind="file", path=a, description="t"))
        monkeypatch.setitem(DS.DATASETS, "rag", DS.DatasetInfo(
            suite="rag", label="B", source_kind="file", path=b, description="t"))
        pair = DS.overlap("agent", "rag")
        assert pair["shared"] == 1
        assert pair["reference_mismatch"] == 1

    def test_report_carries_datasets_and_renders_them(self):
        """结论旁边必须跟着"这是在什么数据上测的"。"""
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.95)])])
        rep.datasets = DS.summary()
        assert rep.to_dict()["datasets"]["datasets"]["agent"]["cases"] == 18
        text = R.render_text(rep)
        assert "评测数据集" in text
        assert "不可直接横向比较" in text

    def test_report_without_datasets_still_renders(self):
        """取不到数据登记时报告照常渲染（不能因此打挂评估）。"""
        text = R.render_text(_report([_suite("agent", [("agent.tool_accuracy", 0.95)])]))
        assert "总体" in text
        assert "评测数据集" not in text


# ===========================================================================
# 门槛留痕：只在少数几次实测上标定的门槛要就地提示（④ 的可见化部分）
# ===========================================================================


class TestProvisionalThresholdWarning:
    def test_render_warns_for_provisional_threshold(self):
        rep = _report([_suite("retrieval", [("retrieval.top1_relevant_rate", 0.95)])])
        text = R.render_text(rep)
        assert "仅在 1 次实测上标定过" in text
        assert "eval.calibrate" in text

    def test_no_warning_for_well_calibrated_metric(self):
        """`calibrated_on_runs=0`（未记录）不提示 —— 避免每行都挂警告。"""
        rep = _report([_suite("agent", [("agent.tool_accuracy", 0.95)])])
        assert "次实测上标定过" not in R.render_text(rep)

    def test_specs_carry_calibration_field(self):
        assert R.get_spec("retrieval.top1_relevant_rate").calibrated_on_runs == 1
        assert R.get_spec("retrieval.first_relevant_rank_mean").calibrated_on_runs == 1
        assert R.get_spec("retrieval.no_relevant_in_topk_rate").calibrated_on_runs == 1


# ===========================================================================
# 模块撞名：eval/ 会被插进 sys.path，其中的模块名不得与第三方包重名
# ===========================================================================


class TestEvalModuleNamesDoNotShadowLibraries:
    """★ 实测踩到的坑：`eval/datasets.py` 把 HuggingFace 的 `datasets` 顶掉了。

    机制：`run_all.run_retrieval_suite()` 会
    `sys.path.insert(0, str(EVAL_DIR))`（为了 `import retrieval_precision`），
    于是 `eval/` 进了模块搜索路径，**优先级还高于 site-packages**。
    接着 `sentence_transformers` 里的 `from datasets import Dataset`
    就解析到我们的 `eval/datasets.py` → `ImportError`。

    现象离原因极远：报告上只有一句"18 个主题检索异常"、四条指标全"未测"，
    而真实原因是一次 import 撞名。故把约束钉成测试。
    """

    def test_no_eval_module_shadows_installed_package(self):
        import importlib.metadata as md

        installed = set(md.packages_distributions())
        eval_dir = ROOT / "eval"
        offenders = sorted(
            p.name
            for p in eval_dir.glob("*.py")
            if p.stem != "__init__" and p.stem in installed
        )
        assert not offenders, (
            f"eval/ 下这些模块名与已安装第三方包重名，会被 eval 目录遮蔽：{offenders}"
            "（`sys.path` 里 eval/ 优先于 site-packages）——请改名"
        )

    def test_datasets_library_still_importable_with_eval_on_path(self):
        """把 `eval/` 插到 sys.path 最前面后，`import datasets` 仍须是第三方库。"""
        import importlib

        eval_dir = str(ROOT / "eval")
        added = eval_dir not in sys.path
        if added:
            sys.path.insert(0, eval_dir)
        try:
            spec = importlib.util.find_spec("datasets")
            origin = getattr(spec, "origin", "") or ""
            assert "eval" not in Path(origin).parts or "dataset_registry" in origin, (
                f"`datasets` 被 eval/ 下的文件遮蔽了：{origin}"
            )
        finally:
            if added:
                sys.path.remove(eval_dir)


# ===========================================================================
# 基线质量：全是"未测"的记录不该当基线（否则门禁静默失效）
# ===========================================================================


class TestBaselineSkipsAllNotRun:
    """★ 实测踩到：一次"检索全部异常"的运行被存进趋势后**成了基线**。

    那条记录里 `retrieval.*` 全是 `value: null`（NOT_RUN）。于是下一次
    `--suites retrieval --gate` 选中它当基线 → 每个指标 `prev_value is None`
    → 全被记成"新增指标" → **门禁静默失效**（永远绿），而报告上看不出异常。

    故基线选择要跳过"一条都没有实测值"的记录，并如实报告跳过了谁。
    """

    @staticmethod
    def _record(run_id: str, values: dict) -> dict:
        return {
            "schema_version": R.SCHEMA_VERSION,
            "run_id": run_id,
            "version": "v",
            "suites": ["retrieval"],
            "case_counts": {"retrieval": 18},
            "metrics": {k: {"value": v, "status": "pass" if v is not None else "not_run"}
                        for k, v in values.items()},
        }

    def test_all_not_run_record_is_skipped(self):
        history = [
            self._record("good", {"retrieval.precision": 0.92}),
            self._record("broken", {"retrieval.precision": None}),
        ]
        cur = _report([_suite("retrieval", [("retrieval.precision", 0.90)], case_count=18)])
        cmp = T.compare_with_previous(cur, history)
        assert cmp.baseline_run_id == "good", "应跳过全未测的那次"
        assert cmp.baseline_skipped_all_not_run == ["broken"]
        assert "broken" in cmp.render(), "跳过谁要写出来，不能静默"

    def test_partially_measured_record_is_usable(self):
        """只有部分指标有值（如只测了 precision）仍可作为基线。"""
        history = [self._record("partial", {"retrieval.precision": 0.92,
                                            "retrieval.top1_relevant_rate": None})]
        cur = _report([_suite("retrieval", [("retrieval.precision", 0.90)], case_count=18)])
        cmp = T.compare_with_previous(cur, history)
        assert cmp.baseline_run_id == "partial"
        assert cmp.baseline_skipped_all_not_run == []

    def test_falls_back_to_all_not_run_when_nothing_else(self):
        """全都是"未测"记录时只能用它，但要如实说明（不是静默）。"""
        history = [self._record("broken", {"retrieval.precision": None})]
        cur = _report([_suite("retrieval", [("retrieval.precision", 0.90)], case_count=18)])
        cmp = T.compare_with_previous(cur, history)
        assert cmp.baseline_run_id == "broken"
        assert "全为未测" in cmp.render() or "没有实测值" in cmp.render()


# ===========================================================================
# rag 线的 case_count 口径：数据集规模，不是"有效判分例数"
# ===========================================================================


class TestRagCaseCountSemantics:
    """★ 实测踩到：rag 基线的 `case_counts` 记成了**有效判分例数**。

    首次跑出的 rag 记录写着 `{"rag": 21}`，而这套评测集是 **25 题**
    （4 题判分失败，按既有口径计入分母）。危害是**悄悄关掉 rag 线的门禁**：

        case_count 是可比性判据（`compare_with_previous` 用它判断两次是否同一批样本）
        裁判抖动一次（21 → 23）→ "⚠️ 样本量不一致 → 不据分数判退化"
        → 退化永远报不出来

    故 `case_count` 必须用数据集规模；有效/失败数写在 note 里（信息不丢）。
    """

    @staticmethod
    def _patch_ragas(monkeypatch, *, total: int = 25, valid: int = 21):
        """把 ragas_eval 的重依赖全部换成替身，只验证 run_all 侧的口径。"""
        import eval  # noqa: F401 - 确保包已导入

        sys.path.insert(0, str(ROOT / "eval"))
        import ragas_eval as RE

        cases = [{"question": f"q{i}", "reference": f"r{i}"} for i in range(total)]

        class _Judge:
            model_name = "fake-judge"

        monkeypatch.setattr(RE, "CASES_PATH", ROOT / "eval" / "test_questions.json")
        monkeypatch.setattr(RE, "_PRESETS", {"fake-cfg": {"name": "fake-cfg"}})
        monkeypatch.setattr(RE, "_PRODUCTION_CONFIG", "fake-cfg")
        monkeypatch.setattr(RE, "load_cases", lambda path, required=None: list(cases))
        monkeypatch.setattr(RE, "build_evaluator_llm", lambda: _Judge())
        monkeypatch.setattr(RE, "_BGEEmbeddings", lambda: object())
        monkeypatch.setattr(
            RE, "_run_config",
            lambda cfg, cs, judge, emb, **kw: {
                "summary": {
                    "faithfulness": {"mean": 0.81, "median": 0.9, "min": 0.0, "zeros": 1},
                    "relevancy": {"mean": 0.86, "median": 0.89, "min": 0.57, "zeros": 0},
                    "context_recall": {"mean": 1.0, "median": 1.0, "min": 1.0, "zeros": 0},
                },
                "valid": valid, "total": total, "errors": total - valid,
            },
        )
        monkeypatch.setattr(
            RE, "judge_provider_matches_generator", lambda: (False, "deepseek", "fake-judge")
        )

    def test_case_count_is_dataset_size_not_valid_count(self, monkeypatch):
        from eval.run_all import run_rag_suite

        self._patch_ragas(monkeypatch, total=25, valid=21)
        suite = run_rag_suite()
        assert suite.ran
        assert suite.case_count == 25, "case_count 必须是数据集规模，不能是有效判分例数"
        assert "有效 21/25" in suite.note, "有效/失败数要写进 note"

    def test_metrics_extracted_and_judged(self, monkeypatch):
        from eval.run_all import run_rag_suite

        self._patch_ragas(monkeypatch)
        suite = run_rag_suite()
        by_name = {r.name: r for r in suite.results}
        assert by_name["rag.faithfulness"].value == 0.81
        assert by_name["rag.answer_relevancy"].value == 0.86
        assert by_name["rag.context_recall"].value == 1.0
        assert by_name["rag.faithfulness"].status is R.MetricStatus.PASS

    def test_stable_case_count_keeps_gate_usable(self, monkeypatch):
        """★ 两次运行的有效例数不同（21 vs 23），**不该**触发样本量不一致。"""
        from eval.run_all import run_rag_suite

        self._patch_ragas(monkeypatch, total=25, valid=21)
        first = run_rag_suite()
        self._patch_ragas(monkeypatch, total=25, valid=23)
        second = run_rag_suite()

        base = R.EvalReport(
            schema_version=R.SCHEMA_VERSION, run_id="base",
            suites=[R.SuiteReport(suite="rag", ran=True, case_count=first.case_count,
                                  results=first.results)],
        )
        cur = R.EvalReport(
            schema_version=R.SCHEMA_VERSION, run_id="cur",
            suites=[R.SuiteReport(suite="rag", ran=True, case_count=second.case_count,
                                  results=second.results)],
        )
        cmp = T.compare_with_previous(cur, [T.report_to_record(base)])
        assert not cmp.case_count_mismatch, (
            "有效例数变化不该被当成样本变化 —— 否则裁判抖一次就关掉 rag 门禁"
        )


# ===========================================================================
# 难例集：真实问法下的召回/排序（评测集已贴天花板，测不出排序改动）
# ===========================================================================


class TestHardTopicSet:
    """★ 为什么要有难例集：评测集首位切题率长期 0.9444、位次 1.06，
    **只剩 1 题的余量** —— 排序类改动在它上面好方案坏方案都测不出差别。
    难例集（真实问法）实测 0.6923 / 1.67，才看得见收益与代价。
    """

    @staticmethod
    def _mod():
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "eval"))
        import hard_topics

        return hard_topics

    def test_set_shape_and_size(self):
        HT = self._mod()
        assert len(HT.HARD_TOPICS) >= 10, "样本太小（1 题 = 7~10 个百分点）没有分辨力"
        for label, question, keys in HT.HARD_TOPICS:
            assert label and question and keys, "题目三元组不能有空项"
            assert all(k.strip() for k in keys), f"{question} 的判据关键词不能为空"
            # 判据关键词必须**够具体**：通用词会让任何文档都算切题
            for generic in ("不得", "应当", "可以", "规定", "投标文件", "招标文件"):
                assert generic not in keys, (
                    f"{question} 的判据含通用词 {generic!r} —— 任何法规都命中，指标会虚高"
                )

    def test_corpus_gaps_excluded_from_pairs(self):
        """★ 库内无据的题必须从指标里剔除：它测的是"知识库覆盖"而不是检索能力。

        实测：14 题里 2 题"前 5 条全不切题"，全量扫语料才发现成因不同 ——
        一题库里真有（真·召回失手），一题库里一篇都没有（该补语料）。
        把后者算成召回退化会指向错误的修法（调检索参数）。
        """
        HT = self._mod()
        pairs = HT.as_pairs()
        assert len(pairs) == len(HT.HARD_TOPICS) - len(HT.KNOWN_CORPUS_GAPS)
        questions = [q for q, _keys in pairs]
        for gap in HT.KNOWN_CORPUS_GAPS:
            assert gap not in questions, "知识库缺口题不该进检索指标"
        assert HT.corpus_gap_note(), "有缺口时必须在 note 里报出来（不能静默剔除）"

    def test_no_gap_gives_empty_note(self, monkeypatch):
        HT = self._mod()
        monkeypatch.setattr(HT, "KNOWN_CORPUS_GAPS", frozenset())
        assert HT.corpus_gap_note() == ""

    def test_metrics_registered_with_rationale(self):
        for name in ("retrieval.hard_top1_relevant_rate",
                     "retrieval.hard_first_relevant_rank_mean",
                     "retrieval.hard_no_relevant_in_topk_rate"):
            spec = R.get_spec(name)
            assert spec is not None, f"{name} 未登记 → 永远不判定"
            assert spec.suite == "retrieval"
            assert spec.rationale, f"{name} 缺少门槛依据"
            assert spec.calibrated_on_runs == 1, "小样本下的门槛要标成临时值"

    def test_judgements_match_measured_values(self):
        """按实测值判：达标；比实测差一题的档位仍达标（留了噪声余量）。"""
        assert R.make_result("retrieval.hard_top1_relevant_rate", 0.6923).status is R.MetricStatus.PASS
        assert R.make_result("retrieval.hard_top1_relevant_rate", 0.5385).status is R.MetricStatus.FAIL
        assert R.make_result("retrieval.hard_first_relevant_rank_mean", 1.67).status is R.MetricStatus.PASS
        assert R.make_result("retrieval.hard_no_relevant_in_topk_rate", 0.0769).status is R.MetricStatus.PASS
        assert R.make_result("retrieval.hard_no_relevant_in_topk_rate", 0.2308).status is R.MetricStatus.FAIL
        # 30 题版本实测值：0.6667 / 1.40 / 0.1667（门槛据此重标定）
        assert R.make_result("retrieval.hard_top1_relevant_rate", 0.6667).status is R.MetricStatus.PASS
        assert R.make_result("retrieval.hard_no_relevant_in_topk_rate", 0.1667).status is R.MetricStatus.PASS


class TestCorpusAudit:
    """语料真值审计：独立于被测系统的"库里到底有没有据"。

    ★ 为什么必须独立：**不能用 `search()` 去确认"库里有据"** ——
    那正是被测对象，检索没捞上来就会被误判成"库里没有"（循环论证）。
    审计直接 scroll 存储，绕开检索。
    """

    @staticmethod
    def _mod():
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "eval"))
        import corpus_audit

        return corpus_audit

    @staticmethod
    def _hard():
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "eval"))
        import hard_topics

        return hard_topics

    def test_audit_marks_evidence_and_gaps(self):
        CA = self._mod()
        docs = [
            CA.Doc(doc_id=1, question="投标保证金什么时候退还？", answer="招标人应当在5日内退还。"),
            CA.Doc(doc_id=2, question="评标委员会怎么组成？", answer="由招标人代表和技术经济专家组成。"),
        ]
        result = CA.audit(
            [("钱多久退回来？", ("退还", "5日内")), ("外星人怎么投标？", ("外星人",))],
            docs=docs,
        )
        assert result["corpus_size"] == 2
        assert result["items"][0]["has_evidence"] is True
        assert result["items"][0]["hits"] == 1
        assert result["items"][1]["has_evidence"] is False
        assert result["corpus_gaps"] == ["外星人怎么投标？"]
        assert "库内无据" in CA.render(result)

    def test_audit_is_offline_with_injected_docs(self):
        """注入 docs 时**不连任何服务**（审计本身要能在离线跑）。"""
        CA = self._mod()
        result = CA.audit([("x", ("a",))], docs=[CA.Doc(doc_id=1, question="q", answer="a")])
        assert result["items"][0]["hits"] == 1

    def test_known_gap_stays_in_the_set(self):
        """已标注的知识库缺口题必须**留在** HARD_TOPICS 里（否则 note 报不出来）。"""
        HT = self._hard()
        questions = [q for _l, q, _k in HT.HARD_TOPICS]
        for gap in HT.KNOWN_CORPUS_GAPS:
            assert gap in questions, "缺口题被删了 —— 应保留并标注，而不是悄悄丢掉"
        assert R.make_result("retrieval.hard_no_relevant_in_topk_rate", 0.2308).status is R.MetricStatus.FAIL
