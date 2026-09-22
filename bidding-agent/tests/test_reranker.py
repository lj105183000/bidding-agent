"""CrossEncoder 精排：排序与归一化、懒加载语义、以及"精排失败不拖垮检索"。

用文件内 fake 替换 CrossEncoder（见 docs/开发文档.md §9.3），不加载真实模型。
"""

from __future__ import annotations

import sys
import threading
import types

import pytest
from qdrant_client import models

from src.config import settings
from src.rag.constants import RERANK_CANDIDATE_K
from src.rag.embedder import Reranker, _sigmoid
from src.rag.pipeline import RAGPipeline

CANDIDATES = [
    {"question": "问题A", "answer": "答案A", "score": 0.9},
    {"question": "问题B", "answer": "答案B", "score": 0.8},
    {"question": "问题C", "answer": "答案C", "score": 0.7},
]


class _ScriptedCrossEncoder:
    """按调用顺序返回预设的**原始 logit**。"""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen_pairs: list[list[tuple[str, str]]] = []
        self.seen_activation = "unset"

    def predict(self, pairs, activation_fn="default"):
        self.seen_pairs.append(list(pairs))
        self.seen_activation = activation_fn
        return self.scores


def _reranker_with(scores: list[float]) -> tuple[Reranker, _ScriptedCrossEncoder]:
    encoder = _ScriptedCrossEncoder(scores)
    reranker = Reranker()
    reranker._model = encoder  # 绕过懒加载，直接注入
    return reranker, encoder


# ---- 排序 ----


def test_rerank_reorders_by_model_score():
    # 故意让模型分数与原始 RRF 顺序相反
    reranker, _ = _reranker_with([0.1, 0.5, 0.9])
    result = reranker.rerank("查询", CANDIDATES, top_k=3)
    assert [item["question"] for item in result] == ["问题C", "问题B", "问题A"]


def test_rerank_truncates_to_top_k():
    reranker, _ = _reranker_with([0.1, 0.5, 0.9])
    assert len(reranker.rerank("查询", CANDIDATES, top_k=2)) == 2


def test_rerank_keeps_original_fields():
    reranker, _ = _reranker_with([0.3, 0.2, 0.1])
    item = reranker.rerank("查询", CANDIDATES, top_k=1)[0]
    assert item["question"] == "问题A"
    assert item["answer"] == "答案A"


def test_rerank_empty_candidates_short_circuits():
    reranker, encoder = _reranker_with([])
    assert reranker.rerank("查询", [], top_k=5) == []
    assert encoder.seen_pairs == [], "没有候选时不应调用模型"


# ---- 分数语义 ----


def test_score_is_normalised_but_raw_logit_is_kept():
    """`score` 对外是 0-1 的相关度（前端展示），原始 logit 留在 rerank_score
    供排查——两者混用会让阈值判断失去意义。"""
    reranker, _ = _reranker_with([-2.0, 0.0, 3.0])
    result = reranker.rerank("查询", CANDIDATES, top_k=3)
    for item in result:
        assert 0.0 <= item["score"] <= 1.0
        assert item["score"] == pytest.approx(_sigmoid(item["rerank_score"]))
    # 单调性：原始分越高，归一化分越高
    assert result[0]["rerank_score"] > result[-1]["rerank_score"]
    assert result[0]["score"] > result[-1]["score"]


def test_requests_raw_logits_to_avoid_double_normalisation():
    """回归测试：`CrossEncoder.predict()` 对单标签模型**默认已套 Sigmoid**。

    若拿它的输出再做一次 sigmoid，logit 为 -5.9 的不相关候选会被抬到 0.50，
    来源卡片的分数就完全失去区分度。必须显式请求未过激活的原始 logit。
    """
    reranker, encoder = _reranker_with([5.38, -5.88, -6.20])
    result = reranker.rerank("查询", CANDIDATES, top_k=3)

    assert encoder.seen_activation != "default", "必须显式传入 activation_fn"
    worst = result[-1]
    assert worst["rerank_score"] < 0, "原始 logit 对不相关候选应为负数"
    assert worst["score"] < 0.01, "双重 sigmoid 会把它抬到 0.5 附近"


def test_sigmoid_is_monotonic_and_bounded():
    assert _sigmoid(0) == pytest.approx(0.5)
    assert _sigmoid(-6) < _sigmoid(0) < _sigmoid(6)
    assert 0.0 < _sigmoid(-6) < 0.01
    assert 0.99 < _sigmoid(6) < 1.0


def test_sigmoid_saturates_for_large_logits():
    """CrossEncoder 的 logit 常有 ±8 以上的量级，此时 sigmoid 会饱和：
    正侧在 float64 下直接舍入成 1.0（1+1.9e-22 就是 1.0），负侧则趋于 0。
    前端把 score 当作 0-1 相关度展示是可以的，但**不能**拿它做精细阈值判断。"""
    assert _sigmoid(50) == 1.0
    assert 0 < _sigmoid(-50) < 1e-20


def test_rerank_pairs_query_with_candidate_text():
    reranker, encoder = _reranker_with([0.1, 0.2, 0.3])
    reranker.rerank("用户的问题", CANDIDATES, top_k=3)
    pairs = encoder.seen_pairs[0]
    assert len(pairs) == len(CANDIDATES)
    assert all(pair[0] == "用户的问题" for pair in pairs), "每个候选都要与同一个查询配对"


def test_rerank_pairs_against_kb_question_not_answer():
    """配对字段是"问题"而非"答案"：实测用答案配对会把 top-1 命中率从 98.7%
    打到 78.7%（见 eval/retrieval_eval.py）。这是回归防线。"""
    reranker, encoder = _reranker_with([0.1, 0.2, 0.3])
    reranker.rerank("用户的问题", CANDIDATES, top_k=3)
    paired_texts = [pair[1] for pair in encoder.seen_pairs[0]]
    assert paired_texts == ["问题A", "问题B", "问题C"]


def test_rerank_text_key_is_overridable():
    reranker, encoder = _reranker_with([0.1, 0.2, 0.3])
    reranker.rerank("用户的问题", CANDIDATES, top_k=3, text_key="answer")
    assert [pair[1] for pair in encoder.seen_pairs[0]] == ["答案A", "答案B", "答案C"]


def test_rerank_falls_back_to_question_when_text_key_missing():
    """候选缺字段时不能让 pair 变成 (query, None)。"""
    reranker, encoder = _reranker_with([0.1])
    reranker.rerank("查询", [{"question": "只有问题", "score": 0.5}], top_k=1, text_key="answer")
    assert encoder.seen_pairs[0] == [("查询", "只有问题")]


# ---- 懒加载 ----


def _stub_sentence_transformers(monkeypatch, calls: list[str]) -> None:
    """用桩模块替换 sentence_transformers，避免真的去加载模型权重。"""

    class _StubCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append(name)

        def predict(self, pairs):
            return [0.0] * len(pairs)

    stub = types.ModuleType("sentence_transformers")
    stub.CrossEncoder = _StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


def test_model_is_loaded_lazily_on_first_use(monkeypatch):
    calls: list[str] = []
    _stub_sentence_transformers(monkeypatch, calls)
    reranker = Reranker()
    assert calls == [], "构造时不应加载模型"
    reranker.model  # noqa: B018
    assert calls == ["BAAI/bge-reranker-base"]


def test_concurrent_first_use_loads_model_once(monkeypatch):
    """双重检查锁：并发首用只加载一份，否则多个线程会各自吃一份显存。"""
    calls: list[str] = []
    _stub_sentence_transformers(monkeypatch, calls)
    reranker = Reranker()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        reranker.model  # noqa: B018

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == ["BAAI/bge-reranker-base"], "并发首用必须只加载一次"


# ---- 与检索流水线的协作 ----


class _FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeBM25:
    def encode_query(self, text: str) -> models.SparseVector:
        return models.SparseVector(indices=[1], values=[1.0])


class _RecordingStore:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.limit: int | None = None

    def hybrid_search(self, dense, sparse, limit=5, prefetch_limit=30):
        self.limit = limit
        return self.hits[:limit]

    def dense_search(self, dense_vector, *, limit=30):
        self.limit = limit
        return self.hits[:limit]

    def sparse_search(self, sparse_vector, *, limit=30):
        return self.hits[:limit]


class _FakeReranker:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.called_with: tuple | None = None

    def rerank(self, query, candidates, top_k):
        self.called_with = (query, len(candidates), top_k)
        if self.error:
            raise self.error
        # 反转顺序，便于断言"结果确实来自精排"
        return [dict(c) for c in reversed(candidates)][:top_k]


def _pipeline(store, reranker) -> RAGPipeline:
    return RAGPipeline(store=store, embedder=_FakeEmbedder(), bm25=_FakeBM25(), reranker=reranker)


@pytest.fixture
def rerank_on(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    yield
    monkeypatch.setattr(settings, "rerank_enabled", False)


def test_rerank_disabled_uses_top_k_directly(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    store = _RecordingStore(CANDIDATES)
    fake = _FakeReranker()
    result = _pipeline(store, fake).search("查询什么？", top_k=2)

    assert store.limit == 2, "不精排时无需扩大候选池"
    assert fake.called_with is None, "关闭精排不应调用模型"
    assert len(result) == 2


def test_rerank_enabled_widens_candidate_pool_and_reorders(rerank_on):
    store = _RecordingStore(CANDIDATES)
    fake = _FakeReranker()
    result = _pipeline(store, fake).search("查询什么？", top_k=2)

    assert store.limit == RERANK_CANDIDATE_K, "精排前必须先扩大候选池"
    assert fake.called_with == ("查询什么？", len(CANDIDATES), 2)
    assert [item["question"] for item in result] == ["问题C", "问题B"]


def test_rerank_failure_falls_back_to_hybrid_order(rerank_on):
    """精排是增益项：模型挂了也不能让用户拿不到答案。"""
    store = _RecordingStore(CANDIDATES)
    fake = _FakeReranker(error=RuntimeError("CUDA out of memory"))
    result = _pipeline(store, fake).search("查询什么？", top_k=2)

    assert [item["question"] for item in result] == ["问题A", "问题B"], "应退回混合检索顺序"
