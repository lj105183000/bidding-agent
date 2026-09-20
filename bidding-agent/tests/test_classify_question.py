"""问题分类 + RRF k 映射测试。

覆盖 `src/rag/classify_question.py` 的三类分类规则、参数映射表、
应用层 RRF 融合函数，以及 pipeline 集成后的端到端行为。

全部用纯数据 mock，不加载任何模型、不连外部服务（见 docs/开发文档.md §9.3）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.rag.classify_question import (
    QuestionCategory,
    RRF_PARAMS,
    classify,
    fuse_rrf,
    get_params,
)


# =========================================================================
# 一、分类规则
# =========================================================================


class TestClassify:
    """分类器：按问句特征分桶，纯函数、确定可复现。"""

    @pytest.mark.parametrize(
        "question",
        [
            "单一来源采购",           # 6 字，无标点
            "招标",                  # 2 字
            "公开招标限额标准",        # 8 字
            "质疑投诉",              # 4 字
        ],
    )
    def test_short_no_punctuation_is_keyword(self, question):
        assert classify(question) is QuestionCategory.KEYWORD

    @pytest.mark.parametrize(
        "question",
        [
            "什么是政府采购的单一来源采购方式？",
            "公开招标和邀请招标有什么区别？",
            "政府采购法对质疑投诉是怎么规定的。",
            "招标投标法里关于投标保证金的要求是什么？",
        ],
    )
    def test_long_with_punctuation_is_sentence(self, question):
        assert classify(question) is QuestionCategory.SENTENCE

    @pytest.mark.parametrize(
        "question",
        [
            "政府采购法第二十六条",            # "第N条" + "法"
            "招标投标法第三十四条的规定",
            "政府采购法第三十一条",
            "第二十八条",                    # 纯"第N条"
            "政府采购法",                    # "法"结尾
            "招标投标法",                   # "法"结尾
            "政府采购条例",                  # "条例"结尾
            "政府采购货物服务招标投标管理办法",     # "办法"结尾
        ],
    )
    def test_regulation_pattern_is_citation(self, question):
        assert classify(question) is QuestionCategory.CITATION

    def test_citation_takes_priority_over_keyword(self):
        """法条引用即使短也归 citation——法条编号是强信号。"""
        # "政府采购法" 5 字无标点，单看长度/标点该归 KEYWORD
        assert classify("政府采购法") is QuestionCategory.CITATION

    def test_citation_takes_priority_over_sentence(self):
        """法条引用即使带标点也归 citation。"""
        assert classify("政府采购法第三十一条？") is QuestionCategory.CITATION

    def test_empty_question_defaults_to_sentence(self):
        """空查询走默认参数，不阻断流程。"""
        assert classify("") is QuestionCategory.SENTENCE
        assert classify(None) is QuestionCategory.SENTENCE  # type: ignore[arg-type]
        assert classify("   ") is QuestionCategory.SENTENCE

    def test_classification_is_deterministic(self):
        """同一输入多次分类结果一致——CI 与测试可复现的基础。"""
        question = "单一来源采购的条件"
        first = classify(question)
        for _ in range(10):
            assert classify(question) is first


# =========================================================================
# 二、RRF 参数映射
# =========================================================================


class TestRRFParams:
    """分类 → 检索参数映射。"""

    def test_all_categories_have_params(self):
        """每个类别都必须有对应参数，缺一个会让检索走到 KeyError 而非降级。"""
        for cat in QuestionCategory:
            assert cat in RRF_PARAMS, f"{cat} 缺少 RRF_PARAMS 条目"

    @pytest.mark.parametrize(
        "cat, expected_keys",
        [
            (QuestionCategory.KEYWORD, {"rrf_k", "prefetch_limit", "top_k"}),
            (QuestionCategory.SENTENCE, {"rrf_k", "prefetch_limit", "top_k"}),
            (QuestionCategory.CITATION, {"rrf_k", "prefetch_limit", "top_k"}),
        ],
    )
    def test_params_have_required_keys(self, cat, expected_keys):
        assert set(RRF_PARAMS[cat].keys()) == expected_keys

    def test_keyword_has_smallest_rrf_k(self):
        """关键词查询 k 最小：放大 BM25 头部排名优势。"""
        k_kw = RRF_PARAMS[QuestionCategory.KEYWORD]["rrf_k"]
        k_sent = RRF_PARAMS[QuestionCategory.SENTENCE]["rrf_k"]
        k_cit = RRF_PARAMS[QuestionCategory.CITATION]["rrf_k"]
        assert k_kw < k_sent, "关键词 k 应小于自然语言"
        assert k_cit < k_sent, "法条 k 应小于自然语言（法条也需 BM25 但不完全倚重）"

    def test_citation_has_largest_prefetch(self):
        """法条引用需更大召回池：编号关键词 + 语义描述都要召回。"""
        p_cit = RRF_PARAMS[QuestionCategory.CITATION]["prefetch_limit"]
        p_kw = RRF_PARAMS[QuestionCategory.KEYWORD]["prefetch_limit"]
        p_sent = RRF_PARAMS[QuestionCategory.SENTENCE]["prefetch_limit"]
        assert p_cit >= p_kw and p_cit >= p_sent

    def test_get_params_returns_correct_mapping(self):
        """get_params 返回的参数与分类结果一致。"""
        assert get_params("单一来源") == RRF_PARAMS[QuestionCategory.KEYWORD]
        assert get_params("什么是单一来源采购？") == RRF_PARAMS[QuestionCategory.SENTENCE]
        assert get_params("政府采购法第三十一条") == RRF_PARAMS[QuestionCategory.CITATION]


# =========================================================================
# 三、应用层 RRF 融合
# =========================================================================


class TestFuseRRF:
    """fuse_rrf：两路检索结果按 1/(k+rank) 合并。"""

    DENSE = [
        {"question": "Q1", "answer": "A1", "score": 0.9},
        {"question": "Q2", "answer": "A2", "score": 0.7},
        {"question": "Q3", "answer": "A3", "score": 0.5},
    ]
    SPARSE = [
        {"question": "Q2", "answer": "A2", "score": 0.8},
        {"question": "Q4", "answer": "A4", "score": 0.6},
        {"question": "Q1", "answer": "A1", "score": 0.4},
    ]

    def test_fuse_returns_merged_results(self):
        result = fuse_rrf(self.DENSE, self.SPARSE, k=60, limit=5)
        assert len(result) == 4, "两路并集去重后应有 4 条（Q1-Q4）"

    def test_item_in_both_lists_scores_higher(self):
        """两路都命中的条目分数更高：Q1 和 Q2 各得两次 1/(k+rank) 贡献。"""
        result = fuse_rrf(self.DENSE, self.SPARSE, k=60, limit=4)
        questions = [r["question"] for r in result]
        # Q1 和 Q2 在两路都出现，应排前面
        assert questions[0] in ("Q1", "Q2")
        assert questions[1] in ("Q1", "Q2")

    def test_limit_truncates_output(self):
        result = fuse_rrf(self.DENSE, self.SPARSE, k=60, limit=2)
        assert len(result) == 2

    def test_empty_inputs_return_empty(self):
        assert fuse_rrf([], [], k=60, limit=5) == []

    def test_one_empty_list_returns_other_as_is(self):
        result = fuse_rrf(self.DENSE, [], k=60, limit=5)
        assert [r["question"] for r in result] == ["Q1", "Q2", "Q3"]

    def test_smaller_k_amplifies_head_advantage(self):
        """k 越小，头部排名优势越大：Q1（dense 排第1 + sparse 排第3）的领先
        在小 k 下应更明显。"""
        k_small = fuse_rrf(self.DENSE, self.SPARSE, k=1, limit=4)
        k_large = fuse_rrf(self.DENSE, self.SPARSE, k=100, limit=4)
        # 小 k 下 Q1 和 Q2（两路都命中）比 Q3/Q4 优势更大
        score_q1_small = next(r["score"] for r in k_small if r["question"] == "Q1")
        score_q3_small = next(r["score"] for r in k_small if r["question"] == "Q3")
        score_q1_large = next(r["score"] for r in k_large if r["question"] == "Q1")
        score_q3_large = next(r["score"] for r in k_large if r["question"] == "Q3")
        gap_small = score_q1_small - score_q3_small
        gap_large = score_q1_large - score_q3_large
        assert gap_small > gap_large, "小 k 应放大头部优势"

    def test_score_field_is_overwritten(self):
        """fuse 后的 score 是 RRF 分值，覆盖原始相似度分数。"""
        result = fuse_rrf(self.DENSE, self.SPARSE, k=60, limit=4)
        for item in result:
            assert item["score"] <= 1.0, "RRF 分值应 ≤ 1（两条路的贡献和上限）"

    def test_answer_payload_is_preserved(self):
        """融合后 answer 字段不丢——dense 路的 payload 应被保留。"""
        result = fuse_rrf(self.DENSE, self.SPARSE, k=60, limit=4)
        answers = {r["question"]: r["answer"] for r in result}
        assert answers["Q1"] == "A1"
        assert answers["Q4"] == "A4"  # 只在 sparse 路出现


# =========================================================================
# 四、Pipeline 集成
# =========================================================================


class TestPipelineIntegration:
    """pipeline.search() 走分类 → 选参数 → 检索的端到端行为。"""

    def _make_pipeline(self, *, rrf_k_expected=None):
        """构造带 mock store/embedder 的 pipeline。

        rrf_k_expected 用于断言走了哪条检索路径（60=服务端 RRF，非60=应用层 RRF）。
        """
        from qdrant_client import models

        from src.rag.pipeline import RAGPipeline

        store = MagicMock()
        hits = [{"question": "Q", "answer": "A", "score": 0.9}]
        store.hybrid_search.return_value = hits
        store.dense_search.return_value = hits
        store.sparse_search.return_value = hits

        embedder = MagicMock()
        embedder.encode_query.return_value = [0.1, 0.2]

        bm25 = MagicMock()
        bm25.encode_query.return_value = models.SparseVector(indices=[1], values=[1.0])

        pipeline = RAGPipeline(store=store, embedder=embedder, bm25=bm25, llm=None)
        return pipeline, store

    def test_keyword_question_uses_app_level_rrf(self):
        """关键词查询 k≠60 → 走 dense_search + sparse_search + fuse_rrf。"""
        pipeline, store = self._make_pipeline()
        pipeline.search("单一来源")  # KEYWORD → rrf_k=20
        assert store.dense_search.called, "k≠60 应走 dense_search"
        assert store.sparse_search.called, "k≠60 应走 sparse_search"
        assert not store.hybrid_search.called, "k≠60 不应走 hybrid_search"

    def test_sentence_question_uses_server_rrf(self):
        """自然语言 k=60 → 走 Qdrant 内置 hybrid_search。"""
        pipeline, store = self._make_pipeline()
        pipeline.search("什么是单一来源采购方式？")  # SENTENCE → rrf_k=60
        assert store.hybrid_search.called, "k=60 应走 hybrid_search"
        assert not store.dense_search.called, "k=60 不应走 dense_search"

    def test_citation_question_uses_app_level_rrf(self):
        """法条引用 k≠60 → 走应用层 RRF。"""
        pipeline, store = self._make_pipeline()
        pipeline.search("政府采购法第三十一条")  # CITATION → rrf_k=40
        assert store.dense_search.called
        assert store.sparse_search.called
        assert not store.hybrid_search.called

    def test_prefetch_limit_passed_to_hybrid_search(self):
        """SENTENCE 的 prefetch_limit（40）应传给 hybrid_search。"""
        pipeline, store = self._make_pipeline()
        pipeline.search("什么是单一来源采购方式？")
        _, kwargs = store.hybrid_search.call_args
        assert kwargs.get("prefetch_limit") == RRF_PARAMS[QuestionCategory.SENTENCE]["prefetch_limit"]

    def test_prefetch_limit_passed_to_dense_sparse_search(self):
        """KEYWORD 的 prefetch_limit 应传给 dense_search / sparse_search。"""
        pipeline, store = self._make_pipeline()
        pipeline.search("单一来源")
        _, d_kwargs = store.dense_search.call_args
        _, s_kwargs = store.sparse_search.call_args
        expected = RRF_PARAMS[QuestionCategory.KEYWORD]["prefetch_limit"]
        assert d_kwargs.get("limit") == expected
        assert s_kwargs.get("limit") == expected
