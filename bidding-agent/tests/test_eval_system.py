"""评估体系单测：指标注册表、报告模型、趋势门禁。

★ 为什么这三块必须有测试
--------------------
它们本身**不产出答案**，只做判定——所以它们出错时不会有人发现，
只会让评估结论悄悄变错：

    · 方向搞反 → 门禁把"延迟降低"判成退化，把"准确率下降"判成进步
    · NOT_RUN 当 FAIL → "跳过 RAGAS"被当成退化，门禁天天误报直到被无视
    · 样本量不校验 → `--limit 3` 与 18 题基线比，报一堆假退化

这三类错误的共同点是**结果看起来正常**，只有专门测判定逻辑才能发现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import report as R  # noqa: E402
from eval import trend as T  # noqa: E402
from eval.metrics import ProcessMetrics, aggregate, extract_process_metrics  # noqa: E402
from eval.run_all import _tool_matched  # noqa: E402


# ===========================================================================
# 工具选择判定口径
# ===========================================================================


class TestToolMatched:
    """★ 口径必须覆盖两种情形，否则准确率指标会误判。

    实测踩过两类误判：
        ① `#15 2026年最新法规` 期望 search_web，但 off 口径不暴露联网工具
           → 该题必然失败，是**评测设置缺陷**（已用 requires_web 剔除）
        ② `#9` 是多工具题：Agent 调了 query_database 两次，最后又补了一次
           知识库检索，`tool_name` 记成 search_knowledge_base → 判成错。
           题目考的是"有没有去业务库查"，**这是口径问题不是能力问题**。
    """

    def test_exact_match(self):
        assert _tool_matched(
            {"expected_tool": "search_knowledge_base"},
            {"tool_name": "search_knowledge_base"},
        )

    def test_mismatch(self):
        assert not _tool_matched(
            {"expected_tool": "search_knowledge_base"},
            {"tool_name": "query_database"},
        )

    def test_no_tool_expected_and_none_called(self):
        """闲聊/超纲题期望不调工具，此时 tool_name 为空串。"""
        assert _tool_matched({"expected_tool": ""}, {"tool_name": ""})

    def test_expected_tools_any_matches_sequence(self):
        """★ 多工具题：序列里出现任一期望工具即算对。"""
        assert _tool_matched(
            {"expected_tool": "query_database",
             "expected_tools_any": ["query_database", "search_knowledge_base"]},
            {"tool_name": "search_knowledge_base",
             "tool_names": ["search_knowledge_graph", "query_database",
                            "search_knowledge_base"]},
        )

    def test_expected_tools_any_still_rejects_unrelated(self):
        """★ 放宽不能放到什么都算对。"""
        assert not _tool_matched(
            {"expected_tool": "query_database",
             "expected_tools_any": ["query_database"]},
            {"tool_name": "search_web", "tool_names": ["search_web"]},
        )

    def test_expected_tools_any_via_tool_name(self):
        assert _tool_matched(
            {"expected_tool": "a", "expected_tools_any": ["b"]},
            {"tool_name": "b", "tool_names": []},
        )

    def test_missing_tool_names_falls_back_gracefully(self):
        """老结果没有 tool_names 字段时不该崩。"""
        assert not _tool_matched(
            {"expected_tool": "a", "expected_tools_any": ["b"]},
            {"tool_name": "c"},
        )


class TestEvalCaseFields:
    """用例文件的契约检查——这些字段是**数据**，写错不会报错，只会静默误判。"""

    @staticmethod
    def _cases():
        return json.loads(
            (ROOT / "eval" / "test_cases.json").read_text(encoding="utf-8")
        )

    def test_all_cases_have_required_fields(self):
        for i, c in enumerate(self._cases(), 1):
            assert "question" in c and c["question"], f"#{i} 缺 question"
            assert c.get("expected_tool") is not None, f"#{i} 缺 expected_tool"

    def test_web_dependent_cases_are_marked(self):
        """★ 期望联网工具的用例必须标 `requires_web`。

        不标的话，`--web-mode off` 跑评测时这些题**必然失败**，
        把 Agent 的准确率从 0.94 拉到 0.89，指向错误的优化方向。
        """
        web_tools = {"search_web", "search_web_cn", "search_exa"}
        for i, c in enumerate(self._cases(), 1):
            if c.get("expected_tool") in web_tools:
                assert c.get("requires_web") is True, (
                    f"#{i} 期望联网工具但未标 requires_web：{c['question'][:40]}"
                )

    def test_expected_tools_any_contains_expected_tool(self):
        """放宽列表里必须含原期望工具，否则口径自相矛盾。"""
        for i, c in enumerate(self._cases(), 1):
            any_of = c.get("expected_tools_any")
            if isinstance(any_of, list) and any_of:
                assert c["expected_tool"] in any_of, (
                    f"#{i} expected_tools_any 不含 expected_tool"
                )


# ===========================================================================
# 指标注册表
# ===========================================================================


class TestMetricSpec:
    def test_higher_is_better_judgement(self):
        spec = R.MetricSpec("x", "X", R.Direction.HIGHER_IS_BETTER, threshold=0.8)
        assert spec.judge(0.9) is R.MetricStatus.PASS
        assert spec.judge(0.8) is R.MetricStatus.PASS  # 边界含等号
        assert spec.judge(0.7) is R.MetricStatus.FAIL

    def test_lower_is_better_judgement(self):
        spec = R.MetricSpec("x", "X", R.Direction.LOWER_IS_BETTER, threshold=100.0)
        assert spec.judge(99.0) is R.MetricStatus.PASS
        assert spec.judge(100.0) is R.MetricStatus.PASS
        assert spec.judge(101.0) is R.MetricStatus.FAIL

    def test_none_value_is_not_run_not_fail(self):
        """★ `None` 必须是 NOT_RUN。

        把"没测"当 FAIL，会让"跳过 RAGAS"变成门禁失败；
        当 PASS 则会让真正的退化被掩盖。必须是第三种状态。
        """
        spec = R.MetricSpec("x", "X", R.Direction.HIGHER_IS_BETTER, threshold=0.8)
        assert spec.judge(None) is R.MetricStatus.NOT_RUN

    def test_observation_only_metric_never_fails(self):
        """threshold=None 的观测指标：只要有值就算 PASS，不该有判定标准。"""
        spec = R.MetricSpec("x", "X", R.Direction.HIGHER_IS_BETTER, threshold=None)
        assert spec.judge(0.0) is R.MetricStatus.PASS
        assert spec.judge(99999.0) is R.MetricStatus.PASS
        assert spec.judge(None) is R.MetricStatus.NOT_RUN


class TestRegistry:
    def test_all_specs_have_rationale(self):
        """★ 每个门槛都要写明"为什么是这个数"。

        门槛会随项目演进需要重标定；没有留痕就只能靠猜当初的意图。
        """
        for name, spec in R.METRIC_SPECS.items():
            assert spec.rationale, f"{name} 缺少 rationale"
            assert spec.label, f"{name} 缺少中文标签"
            assert spec.suite, f"{name} 缺少所属线"

    def test_spec_name_matches_key(self):
        for key, spec in R.METRIC_SPECS.items():
            assert spec.name == key, f"注册表键 {key} 与 spec.name {spec.name} 不一致"

    def test_make_result_for_unknown_metric_is_not_run(self):
        """未知指标名不能静默通过——那会让拼错的指标名看起来"达标"。"""
        r = R.make_result("根本不存在的指标", 1.0)
        assert r.status is R.MetricStatus.NOT_RUN
        assert "未注册" in r.note

    def test_make_result_copies_threshold(self):
        r = R.make_result("agent.tool_accuracy", 0.95)
        assert r.threshold == 0.90
        assert r.status is R.MetricStatus.PASS

    @pytest.mark.parametrize(
        "name,value,expect",
        [
            ("agent.tool_accuracy", 0.95, R.MetricStatus.PASS),
            ("agent.tool_accuracy", 0.89, R.MetricStatus.FAIL),
            ("agent.latency_p95", 50000, R.MetricStatus.PASS),
            ("agent.latency_p95", 70000, R.MetricStatus.FAIL),
            # 门槛 0：编造工具不可用一次都不该有
            ("agent.false_unavailable_rate", 0.0, R.MetricStatus.PASS),
            ("agent.false_unavailable_rate", 0.001, R.MetricStatus.FAIL),
        ],
    )
    def test_real_metric_judgements(self, name, value, expect):
        assert R.make_result(name, value).status is expect


# ===========================================================================
# 报告模型
# ===========================================================================


def _mk_report(metrics: list[tuple[str, float | None]]) -> R.EvalReport:
    suite = R.SuiteReport(suite="agent", ran=True, case_count=10)
    suite.results = [R.make_result(n, v) for n, v in metrics]
    return R.EvalReport(
        schema_version=R.SCHEMA_VERSION, run_id="r1", version="abc",
        suites=[suite],
    )


class TestEvalReport:
    def test_verdict_pass_when_no_failures(self):
        rep = _mk_report([("agent.tool_accuracy", 0.95)])
        assert rep.verdict == "pass"

    def test_verdict_fail_on_any_failure(self):
        rep = _mk_report([("agent.tool_accuracy", 0.95), ("agent.latency_p95", 99999)])
        assert rep.verdict == "fail"
        assert len(rep.failed) == 1

    def test_not_run_does_not_fail_verdict(self):
        """★ "没测"不是"退化"——否则跳过 RAGAS 就会让门禁失败。"""
        rep = _mk_report([("agent.tool_accuracy", 0.95), ("rag.faithfulness", None)])
        assert rep.verdict == "pass"
        assert len(rep.not_run) == 1

    def test_to_dict_shape(self):
        d = _mk_report([("agent.tool_accuracy", 0.95)]).to_dict()
        for key in ("schema_version", "run_id", "version", "config",
                    "verdict", "failed_count", "not_run_count", "suites"):
            assert key in d

    def test_render_includes_reason_for_failure(self):
        """★ 报告必须说明"为什么未达标"，不能只给结论。

        实测踩过：报告写"答案核对通过率未达标"，备注却只列了法规与条款，
        漏了真正的原因（数字未经支持）——**说未达标却不说为什么**最难查。
        """
        suite = R.SuiteReport(suite="agent", ran=True, case_count=1)
        suite.results = [
            R.MetricResult(name="agent.tool_accuracy", value=0.5,
                           status=R.MetricStatus.FAIL, threshold=0.9,
                           note="具体原因在这里")
        ]
        text = R.render_text(R.EvalReport(
            schema_version=1, run_id="r", suites=[suite]))
        assert "具体原因在这里" in text

    def test_render_shows_not_run_suite(self):
        suite = R.SuiteReport(suite="rag", ran=False, note="没装 RAGAS")
        text = R.render_text(R.EvalReport(schema_version=1, run_id="r", suites=[suite]))
        assert "未执行" in text
        assert "没装 RAGAS" in text

    def test_render_formats_ratio_with_decimals(self):
        """★ 比例类不能用 `:g`——1.0 会显示成 `1`，看起来像整数计数。"""
        suite = R.SuiteReport(suite="agent", ran=True)
        suite.results = [R.MetricResult(
            name="agent.tool_accuracy", value=1.0, status=R.MetricStatus.PASS)]
        text = R.render_text(R.EvalReport(schema_version=1, run_id="r", suites=[suite]))
        assert "1.0000" in text


# ===========================================================================
# 过程指标提取
# ===========================================================================


class TestProcessMetrics:
    def test_extracts_tool_sequence(self):
        m = extract_process_metrics({"tool_names": ["a", "b", "a"]})
        assert m.tool_calls == 3
        assert m.distinct_tools == 2
        assert m.redundant_calls == 1

    def test_hit_rounds_cap_uses_rounds_not_tool_calls(self):
        """★ 撞轮次上限按 **tool_timings[].round** 判定，**不**按 tool_calls。

        这条测试的前身断言的是"4 次调用 → 撞上限"，而那正是被修掉的缺陷：
        指标名是 hit_rounds_cap（轮次），却拿**调用次数**跟**轮次上限**比，
        而一轮可以并行调多个工具。

        实测形态：模型在同一轮里发两个**不同 query** 的 search_knowledge_base
        （两路召回），于是 tool_calls=2 而 rounds=1。
        用调用次数判定会把它读成"打转"，而 agent.hit_rounds_cap_rate
        在历史趋势里正是因此从 0 涨到 0.2353 并让门禁 FAIL ——
        而真实的 max(rounds) 实测全部 <= 2，从未触达 4 轮上限。
        """
        one_round = extract_process_metrics({
            "tool_names": ["a", "b", "c", "d"],
            "tool_timings": [{"round": 1, "name": n} for n in ("a", "b", "c", "d")],
        })
        assert one_round.tool_calls == 4
        assert one_round.rounds == 1
        assert one_round.hit_rounds_cap is False, "一轮内多路调用不是打转"

        two = extract_process_metrics({
            "tool_names": ["a", "b", "a", "b"],
            "tool_timings": [
                {"round": 1, "name": "a"}, {"round": 1, "name": "b"},
                {"round": 2, "name": "a"}, {"round": 2, "name": "b"},
            ],
        })
        assert two.rounds == 2 and two.hit_rounds_cap is False

        capped = extract_process_metrics({
            "tool_names": ["a"] * 4,
            "tool_timings": [{"round": i, "name": "a"} for i in (1, 2, 3, 4)],
        })
        assert capped.rounds == 4
        assert capped.hit_rounds_cap is True, "真打转必须仍能被检出"

    def test_hit_rounds_cap_without_timings_does_not_false_alarm(self):
        """没有 tool_timings 时（老结果）**不误报**。

        没有轮次数据就判不了是否撞上限。此时 rounds=0 → hit_rounds_cap=False：
        **宁可漏报，也不假报** —— 假报会让门禁 FAIL 并指向错误的优化方向。
        """
        m = extract_process_metrics({"tool_names": ["a", "b", "c", "d", "e"]})
        assert m.tool_calls == 5
        assert m.rounds == 0
        assert m.hit_rounds_cap is False

    def test_rounds_takes_max_not_count(self):
        m = extract_process_metrics({
            "tool_names": ["a", "b", "c"],
            "tool_timings": [
                {"round": 1, "name": "a"},
                {"round": 3, "name": "b"},
                {"round": 2, "name": "c"},
            ],
        })
        assert m.rounds == 3, "应取最大值而非出现次数"

    def test_rounds_ignores_malformed_timings(self):
        m = extract_process_metrics({
            "tool_names": ["a"],
            "tool_timings": [None, "x", {"round": "abc"}, {"round": 2, "name": "a"}],
        })
        assert m.rounds == 2, "只认 int 类型的 round"

    def test_aggregate_reports_rounds_separately(self):
        ms = [
            extract_process_metrics({
                "tool_names": ["a", "b"],
                "tool_timings": [{"round": 1, "name": "a"}, {"round": 1, "name": "b"}],
            }),
            extract_process_metrics({
                "tool_names": ["a"] * 4,
                "tool_timings": [{"round": i, "name": "a"} for i in (1, 2, 3, 4)],
            }),
        ]
        agg = aggregate(ms)
        assert agg["tool_calls_mean"] == 3.0
        assert agg["rounds_mean"] == 2.5
        assert agg["rounds_max"] == 4
        assert agg["hit_rounds_cap"] == 1

    def test_falls_back_to_tool_called(self):
        """老路径只给 tool_name 单字符串时退化为"至少调了 1 次"，不崩。"""
        m = extract_process_metrics({"tool_called": True, "tool_name": "x"})
        assert m.tool_calls == 1

    def test_extracts_verification_fields(self):
        m = extract_process_metrics({"verification": {
            "ok": False,
            "unsupported_laws": ["《X法》"],
            "unsupported_articles": ["第一条"],
            "unsupported_numbers": ["2%"],
            "false_unavailable_claims": ["联网"],
        }})
        assert m.verification_ok is False
        assert m.unsupported_laws == ["《X法》"]
        assert m.unsupported_numbers == ["2%"]
        assert m.false_unavailable_claims == ["联网"]

    def test_verification_none_when_absent(self):
        """★ 没跑核对时是 None，不是 False——两者含义完全不同。"""
        assert extract_process_metrics({}).verification_ok is None
        assert extract_process_metrics({"verification": {"ok": False}}).verification_ok is False

    def test_never_raises_on_garbage(self):
        for bad in ({}, {"tool_names": "不是列表"}, {"verification": "不是字典"},
                    {"sources": None}, {"answer": None}, None, []):
            extract_process_metrics(bad)  # 不该抛

    def test_counts_sources(self):
        m = extract_process_metrics({"sources": [1, 2], "web_sources": [3]})
        assert m.source_count == 3


class TestAggregate:
    def test_skips_none_when_averaging(self):
        """★ 求均值必须跳过 None。

        把"没测"当 0 会双向污染：既可能虚低（没跑核对的题算不通过），
        也可能虚高（没调工具的题算 0 次调用，把均值往好里拉）。
        """
        metrics = [
            ProcessMetrics(tool_calls=2, verification_ok=True),
            ProcessMetrics(tool_calls=4, verification_ok=None),
        ]
        agg = aggregate(metrics)
        assert agg["verification_ok_rate"] == 1.0  # 只基于那 1 条有值的
        assert agg["tool_calls_mean"] == 3.0       # 但调用次数两条都算

    def test_verification_rate_none_when_all_missing(self):
        agg = aggregate([ProcessMetrics(), ProcessMetrics()])
        assert agg["verification_ok_rate"] is None

    def test_empty_list(self):
        agg = aggregate([])
        assert agg["cases"] == 0
        assert agg["verification_ok_rate"] is None

    def test_counts_unsupported_number_cases(self):
        """★ 回归哨兵：数字也必须被统计。

        实测漏过：报告说"未达标"，备注只列法规与条款，漏了真正的原因（数字）。
        """
        metrics = [ProcessMetrics(unsupported_numbers=["10%"])]
        assert aggregate(metrics)["unsupported_number_cases"] == 1


# ===========================================================================
# 趋势与门禁
# ===========================================================================


class TestTrendStorage:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "trend.jsonl"
        rep = _mk_report([("agent.tool_accuracy", 0.95)])
        T.append_report(rep, path)
        T.append_report(rep, path)
        assert len(T.load_history(path)) == 2

    def test_broken_line_is_skipped(self, tmp_path):
        """★ 一行写坏不该毁掉整个历史文件——这是选 jsonl 的直接理由。"""
        path = tmp_path / "trend.jsonl"
        T.append_report(_mk_report([("agent.tool_accuracy", 0.9)]), path)
        with path.open("a", encoding="utf-8") as f:
            f.write("{这行不是合法 JSON\n")
        T.append_report(_mk_report([("agent.tool_accuracy", 0.9)]), path)
        assert len(T.load_history(path)) == 2

    def test_missing_file_returns_empty(self, tmp_path):
        assert T.load_history(tmp_path / "不存在.jsonl") == []

    def test_case_counts_recorded(self):
        """★ 必须记录样本量，否则无法判断两次运行是否可比。"""
        rec = T.report_to_record(_mk_report([("agent.tool_accuracy", 0.9)]))
        assert rec["case_counts"] == {"agent": 10}

    def test_truncates_beyond_max(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "MAX_HISTORY", 3)
        path = tmp_path / "trend.jsonl"
        for _ in range(5):
            T.append_report(_mk_report([("agent.tool_accuracy", 0.9)]), path)
        assert len(T.load_history(path)) == 3


class TestGate:
    def _hist(self, value: float, case_count: int = 10) -> list[dict]:
        suite = R.SuiteReport(suite="agent", ran=True, case_count=case_count)
        suite.results = [R.make_result("agent.tool_accuracy", value)]
        rep = R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="base",
                           version="v1", suites=[suite])
        return [T.report_to_record(rep)]

    # ------------------------------------------------------------------
    # ★ 套件隔离：这是「分别评测 RAG 与 Agent」的**前提**
    #
    # 原实现里 `compare_with_previous` 只按 schema 版本过滤基线、取最新那次，
    # 于是"只跑一条线"时必然产出假信号（见各测试的 docstring）。
    # ------------------------------------------------------------------

    def _hist_multi(self) -> list[dict]:
        """造三段历史：无 suites 的旧记录 / 只有 agent / agent+retrieval。"""

        def rec(rid: str, suites: list[str] | None, metrics: dict) -> dict:
            d: dict = {
                "schema_version": R.SCHEMA_VERSION,
                "run_id": rid,
                "version": "v1",
                "metrics": {k: {"value": v, "status": "pass"} for k, v in metrics.items()},
            }
            if suites is not None:
                d["suites"] = suites
            return d

        return [
            # 本字段引入前写的：**没法判断跑了哪些线** → 不作基线
            rec("old-no-suites", None, {"agent.tool_accuracy": 0.5}),
            rec("agent-only", ["agent"], {"agent.tool_accuracy": 0.90}),
            rec(
                "agent+retrieval",
                ["agent", "retrieval"],
                {"agent.tool_accuracy": 0.90, "retrieval.precision": 0.80},
            ),
        ]

    def _rep(self, suites: dict[str, list[tuple[str, float]]]) -> R.EvalReport:
        out = []
        for name, metrics in suites.items():
            sr = R.SuiteReport(suite=name, ran=True, case_count=5)
            sr.results = [R.make_result(n, v) for n, v in metrics]
            out.append(sr)
        return R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="cur", suites=out)

    def test_baseline_matches_suite_set_not_just_schema(self):
        """★ 基线必须**套件集合一致**，不能只看 schema 版本。

        原实现取 `candidates[-1]`，于是在「只跑一条线」时会拿混合运行的
        基线去比 —— 实测历史里前 6 次只有 agent、之后变成 agent+retrieval。
        """
        cur = self._rep(
            {
                "agent": [("agent.tool_accuracy", 0.90)],
                "retrieval": [("retrieval.precision", 0.80)],
            }
        )
        cmp = T.compare_with_previous(cur, self._hist_multi())
        assert cmp.baseline_run_id == "agent+retrieval", (
            "应挑中套件集合一致的那次，而不是最近的那次"
        )
        assert not cmp.new_metrics, "套件一致时不该出现「新增指标」"
        assert not cmp.missing

    def test_agent_only_run_picks_agent_only_baseline(self):
        """只跑 agent 时，基线应是同样只有 agent 的那次（不是混合那次）。

        否则 retrieval 指标会因为「本次没测」被塞进 `missing`，
        读起来像「指标丢了」，实际只是**没跑那条线**。
        """
        cur = self._rep({"agent": [("agent.tool_accuracy", 0.95)]})
        cmp = T.compare_with_previous(cur, self._hist_multi())
        assert cmp.baseline_run_id == "agent-only"
        assert cmp.missing == [], f"不该有 missing，实有 {cmp.missing}"
        # ★ 0.90 → 0.95 是**提升**，进 `improvements` 而非 `regressions`。
        #   门禁纪律②强调方向必须对 —— 这条顺带验了"提升不被误判成退化"。
        assert not cmp.regressions, f"提升不该算退化，实有 {cmp.regressions}"
        assert [r.name for r in cmp.improvements] == ["agent.tool_accuracy"]

    def test_no_matching_suite_set_says_so_instead_of_false_alarm(self):
        """历史里没有同套件组合时：**明确说「没法比」**，不产出假信号。

        原实现在这种情况下会把本次**全部**指标记成 `new_metrics`
        （因为基线里一条都没有）—— 那是「没法比」，不是「发明了新指标」。
        """
        cur = self._rep({"retrieval": [("retrieval.precision", 0.80)]})
        cmp = T.compare_with_previous(cur, self._hist_multi())
        assert cmp.baseline_run_id == ""
        assert cmp.no_baseline_reason, "必须给出原因，不能只是沉默"
        assert "retrieval" in cmp.no_baseline_reason
        # 关键：不产出退化，也不谎称「新增指标」
        assert not cmp.regressed
        assert not cmp.new_metrics

    def test_records_without_suites_field_are_not_used_as_baseline(self):
        """没有 `suites` 字段的旧记录**不参与基线选择**。

        宁可「没有基线可比」（明确告知），也不要拿一个**不知道包含什么**
        的基线去比 —— 后者会产出看似有理的假结论。
        """
        only_old = [
            {
                "schema_version": R.SCHEMA_VERSION,
                "run_id": "old",
                "version": "v",
                "metrics": {"agent.tool_accuracy": {"value": 0.5, "status": "pass"}},
            }
        ]
        cur = self._rep({"agent": [("agent.tool_accuracy", 0.95)]})
        cmp = T.compare_with_previous(cur, only_old)
        assert cmp.baseline_run_id == ""
        assert "suites" in cmp.no_baseline_reason

    def test_render_explains_no_baseline_before_listing_noise(self):
        """`render()` 必须**先讲「能不能比」**。

        没有基线时 `missing` 会被塞满。若不先说明原因，读者会以为
        「指标退化了 / 指标没了」，而实际只是两次运行跑的线不同。
        """
        cur = self._rep({"retrieval": [("retrieval.precision", 0.80)]})
        cmp = T.compare_with_previous(cur, self._hist_multi())
        out = cmp.render()
        assert "没有可比基线" in out
        assert "不是退化" in out
        assert "本次未测" not in out, "没有基线时不该再罗列 missing 噪音"

    def test_record_includes_only_ran_suites(self):
        """`report_to_record` 只记**真正跑过**的线。

        没跑的线若也被记进去，「跳过」就与「跑了且没问题」混为一谈，
        下一轮基线选择会挑错。
        """
        sr_run = R.SuiteReport(suite="agent", ran=True, case_count=5)
        sr_run.results = [R.make_result("agent.tool_accuracy", 0.9)]
        sr_skip = R.SuiteReport(suite="rag", ran=False, case_count=0)
        rep = R.EvalReport(
            schema_version=R.SCHEMA_VERSION, run_id="x", suites=[sr_run, sr_skip]
        )
        rec = T.report_to_record(rep)
        assert rec["suites"] == ["agent"], "未跑的线不该记进 suites"
        assert "rag" not in rec["case_counts"]

    def test_detects_regression_higher_is_better(self):
        """准确率下降 = 退化。"""
        cur = _mk_report([("agent.tool_accuracy", 0.70)])
        cmp = T.compare_with_previous(cur, self._hist(0.95))
        assert cmp.regressed
        assert cmp.regressions[0].name == "agent.tool_accuracy"

    def test_detects_regression_lower_is_better(self):
        """★ 延迟**升高**才是退化。方向搞反会把变慢判成进步。"""
        suite = R.SuiteReport(suite="agent", ran=True, case_count=10)
        suite.results = [R.make_result("agent.latency_p95", 90000.0)]
        cur = R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="now", suites=[suite])

        base_suite = R.SuiteReport(suite="agent", ran=True, case_count=10)
        base_suite.results = [R.make_result("agent.latency_p95", 30000.0)]
        base = R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="base",
                            version="v1", suites=[base_suite])
        cmp = T.compare_with_previous(cur, [T.report_to_record(base)])
        assert cmp.regressed, "延迟从 30s 涨到 90s 必须判退化"

    def test_latency_decrease_is_improvement(self):
        suite = R.SuiteReport(suite="agent", ran=True, case_count=10)
        suite.results = [R.make_result("agent.latency_p95", 10000.0)]
        cur = R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="now", suites=[suite])
        base_suite = R.SuiteReport(suite="agent", ran=True, case_count=10)
        base_suite.results = [R.make_result("agent.latency_p95", 30000.0)]
        base = R.EvalReport(schema_version=R.SCHEMA_VERSION, run_id="base",
                            version="v1", suites=[base_suite])
        cmp = T.compare_with_previous(cur, [T.report_to_record(base)])
        assert cmp.improvements
        assert not cmp.regressed

    def test_tolerance_suppresses_noise(self):
        """★ 容差必须留：LLM 有随机性，±2% 波动不该报退化。

        不留容差的门禁会因噪音频繁报警，久而久之被无视——那比没有门禁更糟。
        """
        cur = _mk_report([("agent.tool_accuracy", 0.94)])
        cmp = T.compare_with_previous(cur, self._hist(0.95), tolerance=0.05)
        assert not cmp.regressed

    def test_not_run_does_not_count_as_regression(self):
        """★ 本次没测但历史有值 → 记 missing，**不判退化**。"""
        cur = _mk_report([("agent.tool_accuracy", None)])
        cmp = T.compare_with_previous(cur, self._hist(0.95))
        assert not cmp.regressed
        assert cmp.missing

    def test_new_metric_not_compared(self):
        """历史没有的指标不该被当成退化。"""
        cur = _mk_report([("rag.faithfulness", 0.10)])
        cmp = T.compare_with_previous(cur, self._hist(0.95))
        assert not cmp.regressed
        assert "rag.faithfulness" in cmp.new_metrics

    def test_case_count_mismatch_blocks_verdict(self):
        """★ 样本量不一致时**一律不判退化**。

        实测踩过：`--limit 3` 与 18 题基线比，报出 3 项"退化"，
        实际只是样本从 18 变成 3。
        """
        cur = _mk_report([("agent.tool_accuracy", 0.10)])  # case_count=10
        cmp = T.compare_with_previous(cur, self._hist(0.95, case_count=18))
        assert cmp.regressions, "差异仍应列出供参考"
        assert cmp.case_count_mismatch
        assert not cmp.regressed, "但不得据此判退化"

    def test_schema_version_mismatch_returns_empty(self):
        """字段语义变过之后不混着比——旧记录数值含义可能不同。"""
        cur = _mk_report([("agent.tool_accuracy", 0.1)])
        old = self._hist(0.95)
        old[0]["schema_version"] = 999
        cmp = T.compare_with_previous(cur, old)
        assert not cmp.regressed
        assert cmp.missing  # 全部记为"无基线"

    def test_empty_history(self):
        cmp = T.compare_with_previous(_mk_report([("agent.tool_accuracy", 0.9)]), [])
        assert not cmp.regressed
        assert cmp.missing

    def test_render_mentions_mismatch(self):
        cur = _mk_report([("agent.tool_accuracy", 0.10)])
        cmp = T.compare_with_previous(cur, self._hist(0.95, case_count=18))
        text = cmp.render()
        assert "样本量不一致" in text
        assert "不作为退化判据" in text


# ===========================================================================
# token / 成本指标
# ===========================================================================


def _trace(**overrides) -> dict:
    trace = {
        "call_count": 2,
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "cache_hit_tokens": 400,
        "dropped_count": 0,
        "calls": [
            {"model": "m", "prompt_tokens": 600, "completion_tokens": 100, "cache_hit_tokens": 300},
            {"model": "m", "prompt_tokens": 400, "completion_tokens": 100, "cache_hit_tokens": 100},
        ],
    }
    trace.update(overrides)
    return trace


class TestTokenCostMetrics:
    def test_extracts_tokens_and_per_call_cost(self):
        m = extract_process_metrics(
            {"llm_trace": _trace()},
            cost_fn=lambda model, usage: {"total": 0.001},
        )
        assert m.llm_calls == 2
        assert m.total_tokens == 1200
        assert m.cache_hit_tokens == 400
        assert m.cost_usd == 0.002
        assert m.priced_calls == 2 and m.unpriced_calls == 0

    def test_missing_trace_is_none_not_zero(self):
        m = extract_process_metrics({})
        assert m.total_tokens is None
        assert m.cost_usd is None
        assert m.llm_calls is None

    def test_unpriced_call_makes_cost_none_but_keeps_tokens(self):
        """未知模型：成本算不出来，但 token 仍然有效。"""
        m = extract_process_metrics(
            {"llm_trace": _trace()},
            cost_fn=lambda model, usage: None,
        )
        assert m.total_tokens == 1200
        assert m.cost_usd is None
        assert m.unpriced_calls == 2

    def test_dropped_calls_make_cost_incomplete(self):
        m = extract_process_metrics(
            {"llm_trace": _trace(dropped_count=1)},
            cost_fn=lambda model, usage: {"total": 0.001},
        )
        assert m.cost_usd is None, "记录被截断时不能报一个偏低的成本"
        assert m.priced_calls == 2

    def test_aggregate_skips_missing_usage(self):
        with_usage = extract_process_metrics(
            {"llm_trace": _trace()},
            cost_fn=lambda model, usage: {"total": 0.001},
        )
        without = extract_process_metrics({})
        agg = aggregate([with_usage, without])
        assert agg["usage_cases"] == 1
        assert agg["cost_cases"] == 1
        assert agg["tokens_per_case"] == 1200
        assert agg["cost_usd_per_case"] == 0.002
        assert agg["prompt_cache_hit_rate"] == 0.4
        assert agg["unpriced_calls"] == 0

    def test_aggregate_all_missing_is_none_not_zero(self):
        agg = aggregate([extract_process_metrics({}), extract_process_metrics({})])
        assert agg["tokens_per_case"] is None
        assert agg["cost_usd_per_case"] is None
        assert agg["prompt_cache_hit_rate"] is None
