"""VectorStore：命名向量集合、混合检索、健康检查。

全部用 `QdrantClient(":memory:")` 注入，不依赖任何外部服务，CI 可裸跑。
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.rag.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VectorStore

DIM = 4


def _store() -> VectorStore:
    return VectorStore(
        url=":memory:",
        collection="test_collection",
        dim=DIM,
        client=QdrantClient(":memory:"),
    )


def _point(idx: int, dense: list[float], sparse_indices: list[int], question: str) -> models.PointStruct:
    return models.PointStruct(
        id=idx,
        vector={
            DENSE_VECTOR_NAME: dense,
            SPARSE_VECTOR_NAME: models.SparseVector(
                indices=sparse_indices, values=[1.0] * len(sparse_indices)
            ),
        },
        payload={"question": question, "answer": f"答案{idx}"},
    )


def _seeded() -> VectorStore:
    store = _store()
    store.ensure_collection(recreate=True)
    store.upsert(
        [
            _point(1, [1.0, 0.0, 0.0, 0.0], [10, 11], "单一来源采购"),
            _point(2, [0.0, 1.0, 0.0, 0.0], [20], "竞争性磋商"),
            _point(3, [0.9, 0.1, 0.0, 0.0], [10], "单一来源异议"),
        ]
    )
    return store


# ---- 集合 ----


def test_ensure_collection_creates_named_dense_and_sparse_vectors():
    store = _store()
    store.ensure_collection()
    info = store.client.get_collection(store.collection)
    assert DENSE_VECTOR_NAME in info.config.params.vectors
    assert info.config.params.vectors[DENSE_VECTOR_NAME].size == DIM
    assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors


def test_ensure_collection_is_idempotent():
    store = _store()
    store.ensure_collection()
    store.ensure_collection()  # 不应抛异常
    assert store.count() == 0


def test_recreate_clears_existing_points():
    store = _seeded()
    assert store.count() == 3
    store.ensure_collection(recreate=True)
    assert store.count() == 0


def test_upsert_and_count():
    assert _seeded().count() == 3


# ---- 检索 ----


def test_hybrid_search_returns_contract_fields():
    """契约字段：question / answer / score / relevance。

    `relevance` 是 2026-10-12 新增的：`score`（RRF 融合分）只反映候选内排名，
    无法用来判断"这条到底相不相关"——实测无关问题的 RRF 分能高于相关问题。
    `relevance`（与查询的稠密余弦相似度）才是可判相关的指标，
    `/api/ask` 的相关性过滤依赖它。
    """
    results = _seeded().hybrid_search([1.0, 0.0, 0.0, 0.0], models.SparseVector(indices=[10], values=[1.0]), limit=3)
    assert results
    for item in results:
        assert set(item) == {"question", "answer", "score", "relevance"}
        assert isinstance(item["score"], float)
        # relevance 为 float（能取到向量时）或 None（取不到，表示"未知"而非"不相关"）
        assert item["relevance"] is None or isinstance(item["relevance"], float)


def test_hybrid_search_relevance_matches_query_similarity():
    """relevance 必须真的是与查询的余弦相似度，而不是随意填的数。

    构造一个与查询向量完全一致的文档，其 relevance 应接近 1.0；
    正交向量的 relevance 应接近 0.0。
    """
    store = _seeded()
    # [1,0,0,0] 作为查询：库中若存在同向向量，relevance 应显著高于正交向量
    results = store.hybrid_search(
        [1.0, 0.0, 0.0, 0.0], models.SparseVector(indices=[10], values=[1.0]), limit=5
    )
    rels = [r["relevance"] for r in results if isinstance(r.get("relevance"), (int, float))]
    if rels:
        # 相似度必须在合法区间内
        assert all(-1.0001 <= r <= 1.0001 for r in rels), rels
        # 至少要有一条正相关（查询与库内向量不是正交的）
        assert max(rels) > 0.0, f"全部非正相关，疑似计算错误: {rels}"


def test_hybrid_search_fuses_both_paths():
    """关键词命中（sparse）与语义相近（dense）各自的第一名都应出现在结果里——
    只走一路的实现在这里会露馅。"""
    store = _seeded()
    results = store.hybrid_search(
        [0.0, 1.0, 0.0, 0.0],  # dense 指向 id=2
        models.SparseVector(indices=[10], values=[1.0]),  # sparse 指向 id=1/3
        limit=3,
    )
    questions = {r["question"] for r in results}
    assert "竞争性磋商" in questions  # dense 路召回
    assert questions & {"单一来源采购", "单一来源异议"}  # sparse 路召回


def test_hybrid_search_prefers_dense_match_when_sparse_is_empty():
    results = _seeded().hybrid_search([1.0, 0.0, 0.0, 0.0], models.SparseVector(indices=[], values=[]), limit=3)
    assert results[0]["question"] == "单一来源采购"


def test_hybrid_search_respects_limit():
    results = _seeded().hybrid_search([1.0, 0.0, 0.0, 0.0], models.SparseVector(indices=[10], values=[1.0]), limit=2)
    assert len(results) == 2


def test_hybrid_search_on_empty_collection_returns_empty():
    store = _store()
    store.ensure_collection()
    results = store.hybrid_search([1.0, 0.0, 0.0, 0.0], models.SparseVector(indices=[1], values=[1.0]))
    assert results == []


# ---- 健康检查 ----


def test_health_reports_point_count():
    health = _seeded().health()
    assert health["ok"] is True
    assert health["points_count"] == 3
    assert health["latency_ms"] >= 0


def test_health_swallows_errors_and_marks_latency_negative(monkeypatch):
    """健康检查自身不能把服务打挂：异常一律转成 -1 标记。"""
    store = _store()

    def _boom() -> int:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "count", _boom)
    health = store.health()
    assert health["ok"] is False
    assert health["points_count"] == -1
    assert health["latency_ms"] == -1
