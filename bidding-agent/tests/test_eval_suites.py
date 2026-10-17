"""分线评测的回归测试（docs/77 与 docs/78 的修复）。

覆盖四组：
    ① `SuiteReport.verdict` / `failed` / `not_run`（此前**不存在**，我以为有）
    ② 检索 `score()` 的 recall@k 曲线与 `not_in_pool` 区分
    ③ `--suites` 空值校验（**零条线却报通过**是最危险的一类缺陷）
    ④ `migrate_suites` 的推断与标记（回填不能冒充实测值）
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import report as R  # noqa: E402
from eval import trend as T  # noqa: E402


# ===========================================================================
# ① SuiteReport 的分线结论
# ===========================================================================


class TestSuiteVerdict:
    """★ 这组属性此前**不存在** —— 我最初误以为 `EvalReport.verdict` 是它的。

    实测报 `AttributeError: 'SuiteReport' object has no attribute 'verdict'`
    才发现。它是"分线评测"的基础：没有它，门禁只能说"评测没过"，
    说不出**是哪条线没过** —— 而 RAG 与 Agent 的修法完全不同。
    """

    def test_pass(self):
        sr = R.SuiteReport(
            suite="agent", ran=True,
            results=[R.make_result("agent.tool_accuracy", 0.95)],
        )
        assert sr.verdict == "pass"
        assert sr.failed == []

    def test_fail(self):
        sr = R.SuiteReport(
            suite="agent", ran=True,
            results=[R.make_result("agent.latency_p95", 99999.0)],
        )
        assert sr.verdict == "fail"
        assert len(sr.failed) == 1

    def test_not_ran_is_not_run_not_pass(self):
        """没跑的线是 `not_run`，**不能**说成 `pass`。

        把它归入 pass 会让"跳过 RAGAS"看起来像"RAG 没问题"。
        """
        sr = R.SuiteReport(suite="rag", ran=False)
        assert sr.verdict == "not_run"

    def test_ran_but_all_not_run(self):
        """跑了但一条指标都没测出来（RAGAS 依赖缺失）→ 也是 `not_run`。"""
        sr = R.SuiteReport(
            suite="rag", ran=True,
            results=[R.make_result("rag.faithfulness", None)],
        )
        assert sr.verdict == "not_run"

    def test_not_run_does_not_count_as_failed(self):
        """`not_run` 不是失败（与 EvalReport 同一口径：没测不等于退化）。"""
        sr = R.SuiteReport(
            suite="rag", ran=True,
            results=[
                R.make_result("rag.faithfulness", None),
                R.make_result("rag.answer_relevancy", 0.9),
            ],
        )
        assert sr.failed == []
        assert sr.verdict == "pass", "有一项测了且达标，就不该是 not_run"

    def test_verdict_in_to_dict(self):
        """`to_dict` 必须带 verdict —— 前端 `/eval` 与趋势文件都靠它。"""
        sr = R.SuiteReport(
            suite="agent", ran=True,
            results=[R.make_result("agent.tool_accuracy", 0.95)],
        )
        assert sr.to_dict()["verdict"] == "pass"


# ===========================================================================
# ② 检索 score()：recall@k 与 not_in_pool
# ===========================================================================


class _FakeRetriever:
    """最小替身：给定每题一个候选池，池内顺序可控。"""

    def __init__(self, pools: dict[str, list[str]]) -> None:
        self._pools = pools
        self.entries = [{"question": q} for q in pools]
        self._cache: dict = {}

    def candidates(self, style: str, query: str) -> list[dict]:
        return [{"question": c} for c in self._pools[query]]


class TestRetrievalScore:
    def test_not_in_pool_counted_separately(self):
        """★ 连候选池都没进的题必须**单独计数**，不能与"排序靠后"混为一谈。

        两者的修法相反：前者要查索引/切分/嵌入，后者调 k 或权重就行。
        原实现把两种情况都折进 `continue`，看 `topk` 分不出来。
        """
        from eval.retrieval_eval import score

        # 题 a 命中第 1；题 b 在池里但排第 3；题 c **不在池里**
        rt = _FakeRetriever({
            "a": ["a", "x", "y"],
            "b": ["x", "y", "b"],
            "c": ["x", "y", "z"],          # 没有 c
        })
        import eval.retrieval_eval as RE

        orig = RE.build_queries
        RE.build_queries = lambda entries, style: [e["question"] for e in entries]
        try:
            r = score(rt, "none", "keyword", top_k=5, limit=0)
        finally:
            RE.build_queries = orig

        assert r["n"] == 3
        assert r["not_in_pool"] == 1, "c 不在池里，应计 1"
        assert r["in_pool_rate"] == pytest.approx(2 / 3)
        # top1 = 只有 a；topk = a 与 b（k=5）
        assert r["top1"] == pytest.approx(1 / 3)
        assert r["topk"] == pytest.approx(2 / 3)
        # ★ 分母是全部题（3），不是池内题（2）—— 否则会算出"完美"的假象
        assert r["recall_at"][1] == pytest.approx(1 / 3)

    def test_recall_curve_monotonic_with_k(self):
        """recall@k 必须随 k 单调不减 —— 违反说明聚合写错了。"""
        from eval.retrieval_eval import score
        import eval.retrieval_eval as RE

        rt = _FakeRetriever({
            "a": ["a", "x", "y", "z"],
            "b": ["x", "b", "y", "z"],
            "c": ["x", "y", "c", "z"],
            "d": ["x", "y", "z", "w"],      # 不在池里
        })
        orig = RE.build_queries
        RE.build_queries = lambda entries, style: [e["question"] for e in entries]
        try:
            r = score(rt, "none", "keyword", top_k=5, limit=0)
        finally:
            RE.build_queries = orig

        curve = [r["recall_at"][k] for k in sorted(r["recall_at"])]
        assert curve == sorted(curve), f"recall@k 应单调不减，实为 {curve}"
        assert r["recall_at"][1] == pytest.approx(1 / 4)
        assert r["recall_at"][3] == pytest.approx(3 / 4), "k=3 时 a/b/c 都进了"
        assert r["recall_at"][5] == pytest.approx(3 / 4), "d 不在池里，k 再大也进不来"

    def test_empty_entries_no_zero_division(self):
        """空题集不能除零（会得到 NaN 并污染聚合）。"""
        from eval.retrieval_eval import score
        import eval.retrieval_eval as RE

        rt = _FakeRetriever({})
        orig = RE.build_queries
        RE.build_queries = lambda entries, style: []
        try:
            r = score(rt, "none", "keyword", top_k=5, limit=0)
        finally:
            RE.build_queries = orig
        assert r["n"] == 0
        assert r["mrr"] == 0.0
        assert r["mean_rank"] is None
        assert r["recall_at"] == {}

    def test_pool_size_reported_so_k_can_be_judged(self):
        """返回值必须给 `candidate_k` / `pool_size`。

        因为 recall@k 在 k >= 候选池深度时**必然饱和**，
        调用方需要这两个值才能判断"这个 k 还有没有信息量"。
        """
        from eval.retrieval_eval import score, CANDIDATE_K
        import eval.retrieval_eval as RE

        rt = _FakeRetriever({"a": ["a", "x"]})
        orig = RE.build_queries
        RE.build_queries = lambda entries, style: ["a"]
        try:
            r = score(rt, "none", "keyword", top_k=5, limit=0)
        finally:
            RE.build_queries = orig
        assert r["candidate_k"] == CANDIDATE_K
        assert r["pool_size"] == 2, "实际池深只有 2，虽然上限是 20"
        assert max(r["recall_at"]) <= max(CANDIDATE_K, 1)


# ===========================================================================
# ③ --suites 空值校验
# ===========================================================================


class TestSuitesGuard:
    """★ `--suites ""` 原先**零条线却报通过并退出 0**。

    实测：`"".split(",")` → `[""]` → 被 `if s.strip()` 滤成 `[]`，
    逃过 unknown 校验 → `build_report([])` 没有任何 suite →
    打印「总体：✅ 通过（0 项未测）」、退出码 0。

    这是**最危险的一类缺陷**：什么都没测的流水线报成功。
    CI 里若写 `--suites "$SUITES"` 而变量为空，会得到"全绿但零覆盖"。

    ★ 用 `parse_suites` 纯函数测，**不启动子进程**
    ----------------------------------------
    最初我写的是 `subprocess.run([... "--limit", "0"])` —— 而 `--limit 0`
    的含义是"**不限**"，于是它**真跑完了整个 agent 套件**直到超时。
    校验是纯字符串逻辑，抽成函数后毫秒级完成。（这个坑本身值得记：
    用 `--limit 0` 想"跑得快"是反的。）
    """

    @pytest.mark.parametrize("value", ["", "  ", ",", ",,", " , "])
    def test_empty_suites_raises(self, value):
        from eval.run_all import parse_suites

        with pytest.raises(SystemExit) as ei:
            parse_suites(value)
        assert "不能为空" in str(ei.value)

    def test_unknown_suite_reported_as_unknown(self):
        """空与未知是**不同成因**，提示词要分开（否则误导排查方向）。"""
        from eval.run_all import parse_suites

        with pytest.raises(SystemExit) as ei:
            parse_suites("bogus")
        msg = str(ei.value)
        assert "未知的评测线" in msg
        assert "不能为空" not in msg

    def test_trailing_comma_is_tolerated(self):
        """`agent,` 里的空段应被忽略，**不是**报错（常见的手滑写法）。"""
        from eval.run_all import parse_suites

        assert parse_suites("agent,") == ["agent"]
        assert parse_suites("agent, retrieval") == ["agent", "retrieval"]
        assert parse_suites(" agent ,, retrieval ") == ["agent", "retrieval"]

    def test_all_valid_suites_accepted(self):
        from eval.run_all import parse_suites, ALL_SUITES

        assert parse_suites(",".join(ALL_SUITES)) == list(ALL_SUITES)


# ===========================================================================
# ④ migrate_suites 的推断
# ===========================================================================


class TestMigrateSuites:
    def test_infer_from_case_counts_keys(self):
        from eval.migrate_suites import infer_suites

        assert infer_suites({"case_counts": {"agent": 5}}) == ["agent"]
        assert infer_suites({"case_counts": {"agent": 5, "retrieval": 3}}) == [
            "agent", "retrieval",
        ]

    def test_uninferable_returns_none_not_guess(self):
        """推不出就返回 None —— **不猜**。

        宁可"没有基线可比"（明确告知），也不要拿一个不知道包含什么的
        基线去比，那会产出看似有理的假结论。
        """
        from eval.migrate_suites import infer_suites

        assert infer_suites({}) is None
        assert infer_suites({"case_counts": {}}) is None
        assert infer_suites({"case_counts": "not-a-dict"}) is None
        assert infer_suites({"case_counts": None}) is None

    def test_migration_marks_inferred_not_observed(self, tmp_path):
        """回填的记录必须带 `suites_inferred: true` —— **不冒充实测值**。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        p.write_text(
            json.dumps({
                "schema_version": T.SCHEMA_VERSION, "run_id": "old",
                "metrics": {"agent.tool_accuracy": {"value": 0.9, "status": "pass"}},
                "case_counts": {"agent": 5},
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stat = migrate(p, apply=True)
        assert stat["filled"] == 1
        assert stat["applied"] is True

        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["suites"] == ["agent"]
        assert rec["suites_inferred"] is True

    def test_migration_skips_existing_and_leaves_them_unmarked(self, tmp_path):
        """已有 `suites` 的记录**不动**，也不误加推断标记。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        p.write_text(
            json.dumps({
                "schema_version": T.SCHEMA_VERSION, "run_id": "new",
                "metrics": {}, "suites": ["retrieval"], "case_counts": {"retrieval": 3},
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stat = migrate(p, apply=True)
        assert stat["already"] == 1
        assert stat["filled"] == 0
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert "suites_inferred" not in rec, "实测记录的记录不该被标记成推断值"

    def test_preview_does_not_write(self, tmp_path):
        """预览模式**不得写盘** —— 否则"先看看"会变成"已经改了"。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        original = json.dumps({
            "schema_version": T.SCHEMA_VERSION, "run_id": "old",
            "metrics": {}, "case_counts": {"agent": 5},
        }, ensure_ascii=False) + "\n"
        p.write_text(original, encoding="utf-8")

        stat = migrate(p, apply=False)
        assert stat["applied"] is False
        assert stat["filled"] == 1
        assert p.read_text(encoding="utf-8") == original, "预览模式改了文件"

    def test_migration_backs_up_before_writing(self, tmp_path):
        """写盘前必须备份 —— 趋势文件是长期资产，写坏就没了。"""
        from eval.migrate_suites import migrate

        p = tmp_path / "trend.jsonl"
        p.write_text(
            json.dumps({
                "schema_version": T.SCHEMA_VERSION, "run_id": "old",
                "metrics": {}, "case_counts": {"agent": 5},
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stat = migrate(p, apply=True)
        assert stat["backup"]
        assert pathlib.Path(stat["backup"]).exists()
