"""离线检索线（retrieval_offline）的纯逻辑与注册测试。

所有用例都注入 fake retriever / 临时 fixture，**不加载 BGE、不连 Qdrant**。
真实模型的跑通验证放在 eval 命令里（见 docs/开发文档.md §10）。
"""

from __future__ import annotations

import json

import pytest

from eval import dataset_registry as DS
from eval import report as R
from eval import trend as T
from eval.retrieval_offline_eval import (
    OfflineQuery,
    load_fixture,
    qrels_metrics,
    run_offline_suite,
)
from eval.run_all import ALL_SUITES, parse_suites


class _FakeRetriever:
    """按 query 返回预置文档；文档形状与服务端 hybrid_search 一致。"""

    def __init__(self, by_query: dict[str, list[dict]]) -> None:
        self.by_query = by_query
        self.calls: list[tuple[str, str]] = []

    def candidates(self, style: str, query: str) -> list[dict]:
        self.calls.append((style, query))
        value = self.by_query[query]
        if isinstance(value, Exception):
            raise value
        return value


def _doc(name: str) -> dict:
    return {"question": f"问题{name}", "answer": f"答案{name}"}


def _temp_fixture(tmp_path, *, queries=None, corpus=None):
    data = {
        "schema_version": 1,
        "name": "unit_fixture",
        "corpus": corpus or [
            {"id": "d1", "question": "问题一", "answer": "答案一"},
            {"id": "d2", "question": "问题二", "answer": "答案二"},
            {"id": "d3", "question": "问题三", "answer": "答案三"},
        ],
        "queries": queries or [
            {"id": "q1", "query": "查一", "relevant_ids": ["d1", "d2"]},
            {"id": "q2", "query": "查二", "relevant_ids": ["d3"]},
        ],
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ===========================================================================
# fixture 加载与校验
# ===========================================================================


class TestFixture:
    def test_real_fixture_has_40_queries(self):
        fixture = load_fixture()
        assert fixture.schema_version == 1
        assert len(fixture.queries) == 40
        assert len(fixture.corpus) == 565

    def test_every_relevant_id_exists(self):
        fixture = load_fixture()
        doc_ids = {d["id"] for d in fixture.corpus}
        for query in fixture.queries:
            assert query.relevant_ids, query.id
            assert set(query.relevant_ids) <= doc_ids, query.id

    def test_missing_file_raises_readable_error(self, tmp_path):
        with pytest.raises(ValueError, match="文件不存在"):
            load_fixture(tmp_path / "nope.json")

    def test_dangling_relevant_id_rejected(self, tmp_path):
        path = _temp_fixture(tmp_path, queries=[
            {"id": "q1", "query": "查一", "relevant_ids": ["d404"]},
        ])
        with pytest.raises(ValueError, match="不存在的文档"):
            load_fixture(path)

    def test_duplicate_question_rejected(self, tmp_path):
        path = _temp_fixture(tmp_path, corpus=[
            {"id": "d1", "question": "同题", "answer": "一"},
            {"id": "d2", "question": "同题", "answer": "二"},
        ])
        with pytest.raises(ValueError, match="question 重复"):
            load_fixture(path)


# ===========================================================================
# qrels 指标（纯函数）
# ===========================================================================


class TestQrelsMetrics:
    def test_perfect_ranking(self):
        queries = [
            OfflineQuery("q1", "查一", ("d1", "d2")),
            OfflineQuery("q2", "查二", ("d3",)),
        ]
        ranked = [["d1", "d2", "x"], ["d3", "y", "z"]]
        m = qrels_metrics(ranked, queries, top_k=5)
        assert m["cases"] == 2
        assert m["top1_relevant_rate"] == 1.0
        assert m["recall_at_k"] == 1.0
        # q1: 2/5, q2: 1/5 → 0.3
        assert m["precision_at_k"] == 0.3
        assert m["mrr"] == 1.0
        assert m["no_relevant_in_topk_rate"] == 0.0

    def test_empty_results_count_as_miss(self):
        queries = [OfflineQuery("q1", "查一", ("d1",)), OfflineQuery("q2", "查二", ("d2",))]
        m = qrels_metrics([[], []], queries, top_k=5)
        assert m["top1_relevant_rate"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["precision_at_k"] == 0.0
        assert m["mrr"] == 0.0
        assert m["no_relevant_in_topk_rate"] == 1.0

    def test_rank_affects_mrr(self):
        queries = [OfflineQuery("q1", "查一", ("d3",))]
        m = qrels_metrics([["x", "y", "d3"]], queries, top_k=5)
        assert m["top1_relevant_rate"] == 0.0
        assert m["mrr"] == pytest.approx(1 / 3, abs=1e-4)
        assert m["no_relevant_in_topk_rate"] == 0.0

    def test_shorter_ranked_list_is_not_truncated_away(self):
        """ranked_ids 比 queries 短时，缺失题按未命中计入分母。"""
        queries = [OfflineQuery("q1", "查一", ("d1",)), OfflineQuery("q2", "查二", ("d2",))]
        m = qrels_metrics([["d1"]], queries, top_k=5)
        assert m["cases"] == 2
        assert m["top1_relevant_rate"] == 0.5
        assert m["no_relevant_in_topk_rate"] == 0.5

    def test_empty_queries_returns_none(self):
        m = qrels_metrics([], [], top_k=5)
        assert m["cases"] == 0
        assert m["top1_relevant_rate"] is None


# ===========================================================================
# suite runner（fake retriever）
# ===========================================================================


class TestRunOfflineSuite:
    def test_runs_with_injected_retriever(self, tmp_path):
        path = _temp_fixture(tmp_path)
        fake = _FakeRetriever({
            "查一": [_doc("一"), _doc("二")],
            "查二": [_doc("三")],
        })
        suite = run_offline_suite(fixture_path=path, retriever_factory=lambda entries: fake)
        assert suite.ran is True
        assert suite.case_count == 2
        assert suite.note == ""
        names = [r.name for r in suite.results]
        assert names == [
            "retrieval_offline.top1_relevant_rate",
            "retrieval_offline.recall_at_5",
            "retrieval_offline.precision_at_5",
            "retrieval_offline.mrr",
            "retrieval_offline.no_relevant_in_top5_rate",
        ]
        values = {r.name: r.value for r in suite.results}
        assert values["retrieval_offline.top1_relevant_rate"] == 1.0
        assert values["retrieval_offline.recall_at_5"] == 1.0
        assert values["retrieval_offline.mrr"] == 1.0
        assert all(r.value is not None for r in suite.results), "五项都必须有实测值"
        assert fake.calls == [("exact", "查一"), ("exact", "查二")]

    def test_limit_changes_case_count(self, tmp_path):
        path = _temp_fixture(tmp_path)
        fake = _FakeRetriever({"查一": [_doc("一")]})
        suite = run_offline_suite(
            limit=1, fixture_path=path, retriever_factory=lambda entries: fake
        )
        assert suite.case_count == 1
        assert len(fake.calls) == 1

    def test_bad_fixture_records_not_run(self, tmp_path):
        suite = run_offline_suite(fixture_path=tmp_path / "missing.json")
        assert suite.ran is False
        assert "加载失败" in suite.note

    def test_retriever_init_failure_records_not_run(self, tmp_path):
        path = _temp_fixture(tmp_path)

        def boom(_entries):
            raise RuntimeError("模型未就绪")

        suite = run_offline_suite(fixture_path=path, retriever_factory=boom)
        assert suite.ran is False
        assert "初始化失败" in suite.note

    def test_single_query_error_counts_in_denominator(self, tmp_path):
        path = _temp_fixture(tmp_path)
        fake = _FakeRetriever({
            "查一": [_doc("一")],
            "查二": RuntimeError("qdrant 抖动"),
        })
        suite = run_offline_suite(fixture_path=path, retriever_factory=lambda entries: fake)
        assert suite.ran is True
        assert suite.case_count == 2, "异常题必须计入分母"
        assert "1/2 题检索异常" in suite.note
        values = {r.name: r.value for r in suite.results}
        # 第一题完美、第二题未命中 → 首位切题率 0.5
        assert values["retrieval_offline.top1_relevant_rate"] == 0.5
        assert values["retrieval_offline.no_relevant_in_top5_rate"] == 0.5


# ===========================================================================
# 框架注册 / 门禁
# ===========================================================================


class TestRegistration:
    def test_suite_registered_everywhere(self):
        assert "retrieval_offline" in ALL_SUITES
        assert "retrieval_offline" in DS.DATASETS
        assert parse_suites("retrieval_offline") == ["retrieval_offline"]
        for name in (
            "retrieval_offline.top1_relevant_rate",
            "retrieval_offline.recall_at_5",
            "retrieval_offline.precision_at_5",
            "retrieval_offline.mrr",
            "retrieval_offline.no_relevant_in_top5_rate",
        ):
            spec = R.get_spec(name)
            assert spec is not None and spec.suite == "retrieval_offline"

    def test_dataset_registry_reads_offline_questions(self):
        assert len(DS.questions_of("retrieval_offline")) == 40
        assert DS.references_of("retrieval_offline") == {}
        assert DS.fingerprint(DS.DATASETS["retrieval_offline"].path)["sha256_12"]

    def test_fail_on_accepts_offline_suite(self):
        rep = R.EvalReport(schema_version=1, run_id="t", version="v")
        suite = R.SuiteReport(suite="agent", ran=True, case_count=1)
        suite.results.append(R.make_result("agent.tool_accuracy", 0.1))
        rep.suites.append(suite)
        code, detail = T.decide_exit_code(rep, None, fail_on=["retrieval_offline"])
        assert code == 0
        assert detail["ignored_suites"] == ["agent"]

    def test_unknown_suite_still_rejected(self):
        rep = R.EvalReport(schema_version=1, run_id="t", version="v")
        with pytest.raises(ValueError, match="未知的评测线"):
            T.decide_exit_code(rep, None, fail_on=["nope"])


class TestEmbedderInjection:
    def test_retriever_uses_injected_embedder(self):
        """离线线/测试必须能绕开全局 BGE 单例，否则 CI 每次都要下载模型。"""
        from eval.retrieval_eval import Retriever

        class FakeEmbedder:
            dim = 8

            def __init__(self):
                self.doc_calls = 0
                self.query_calls = 0

            def _vec(self, text):
                vec = [0.0] * self.dim
                for i, ch in enumerate(text):
                    vec[i % self.dim] += (ord(ch) % 97) / 97.0
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                return [x / norm for x in vec]

            def encode_documents(self, texts):
                self.doc_calls += len(texts)
                return [self._vec(t) for t in texts]

            def encode_query(self, text):
                self.query_calls += 1
                return self._vec(text)

        entries = [
            {"question": "投标保证金什么时候退还", "answer": "合同签订后5日内退还"},
            {"question": "履约保证金比例", "answer": "不得超过10%"},
        ]
        fake = FakeEmbedder()
        retriever = Retriever(entries, embedder_impl=fake)
        docs = retriever.candidates("exact", "保证金退还")
        assert fake.doc_calls == len(entries)
        assert fake.query_calls == 1
        assert docs, "注入 embedder 后内存索引应能返回候选"


class TestUsageMetrics:
    def test_cost_is_excluded_from_relative_regression(self):
        """成本升高 10 倍也不该触发相对门禁（track_regression=False）。"""
        def report(value):
            rep = R.EvalReport(schema_version=1, run_id="t", version="v")
            suite = R.SuiteReport(suite="agent", ran=True, case_count=1)
            suite.results.append(R.make_result("agent.cost_usd_per_case", value))
            rep.suites.append(suite)
            return rep

        baseline = T.report_to_record(report(0.001))
        cmp = T.compare_with_previous(report(0.01), [baseline], tolerance=0.02)
        assert cmp.regressions == []
        assert cmp.regressed is False

    def test_usd_value_keeps_six_decimals(self):
        spec = R.get_spec("agent.cost_usd_per_case")
        assert R._fmt_value(0.0034, spec) == "0.003400"
