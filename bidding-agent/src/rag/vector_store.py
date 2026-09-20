"""Qdrant 向量库封装：命名向量（dense + sparse）+ 原生混合检索。

**为什么用命名向量而不是两个集合**：把 dense 与 sparse 存在同一个点上，
检索时用 `Prefetch` 两路并行召回、再交给 `FusionQuery(RRF)` 融合，一次
网络往返就完成混合检索；分成两个集合则要在应用层对齐 id、自己实现融合，
并且拿不到 Qdrant 服务端的执行优化。

**版本注意**：qdrant-client ≥1.9 已移除 `client.search()`，全部走
`query_points`（见 docs/技术栈.md §12 踩坑记录）。
"""

from __future__ import annotations

import threading
import time

from qdrant_client import QdrantClient, models

from src.logging_config import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
UPSERT_BATCH = 256

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


class VectorStore:
    """集合与检索操作。客户端可注入（`client=`），便于测试用内存实例。"""

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        collection: str,
        dim: int = 512,
        distance: str = "cosine",
        timeout: int = 30,
        client: QdrantClient | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.dim = dim
        self.distance = _DISTANCE_MAP.get(distance.lower(), models.Distance.COSINE)
        self.timeout = timeout
        self._client = client
        self._lock = threading.Lock()

    # ---- 客户端 ----

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def _build_client(self) -> QdrantClient:
        """构造客户端。

        `trust_env=False` 是刻意的：桌面代理软件关闭后，继承系统代理会把所有
        出站连接路由到失效端口（WinError 10061），见 docs/开发文档.md §11.3。
        需要代理时显式配置 HTTP_PROXY/HTTPS_PROXY。

        URL 为 `:memory:` 或本地路径时走嵌入式模式——离线开发与测试可用，
        生产配置的是 Cloud/自建服务的 http(s) 地址。
        """
        if self.url == ":memory:":
            logger.info("使用 Qdrant 嵌入式内存模式")
            return QdrantClient(":memory:")
        if self.url and not self.url.startswith(("http://", "https://")):
            logger.info("使用 Qdrant 嵌入式本地路径: %s", self.url)
            return QdrantClient(path=self.url)
        return QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            timeout=self.timeout,
            trust_env=False,
        )

    # ---- 集合管理 ----

    def ensure_collection(self, recreate: bool = False) -> None:
        """确保集合存在；`recreate=True` 先删后建（全量重建语义）。"""
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            logger.info("删除已存在的集合 %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            logger.info("创建集合 %s (dense=%d维, sparse=BM25)", self.collection, self.dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(size=self.dim, distance=self.distance)
                },
                sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
            )

    def upsert(self, points: list[models.PointStruct]) -> int:
        for start in range(0, len(points), UPSERT_BATCH):
            batch = points[start : start + UPSERT_BATCH]
            self.client.upsert(collection_name=self.collection, points=batch, wait=True)
        return len(points)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    # ---- 检索 ----

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        *,
        limit: int = 5,
        prefetch_limit: int = 30,
    ) -> list[dict]:
        """dense + sparse 双路召回 → RRF 融合。

        RRF（Reciprocal Rank Fusion）只看两路的**排名**而不看分数：稠密余弦
        相似度与 BM25 分值量纲完全不同，直接加权求和需要反复调参，RRF 用
        `1/(k+rank)` 求和天然规避了归一化问题，是本项目的默认融合策略。
        """
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using=DENSE_VECTOR_NAME, limit=prefetch_limit
                ),
                models.Prefetch(
                    query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=prefetch_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "question": (p.payload or {}).get("question", ""),
                "answer": (p.payload or {}).get("answer", ""),
                "score": float(p.score),
            }
            for p in response.points
        ]

    def dense_search(
        self, dense_vector: list[float], *, limit: int = 30
    ) -> list[dict]:
        """单路稠密向量检索。供应用层 RRF（`classify_question.fuse_rrf`）使用——
        当分类器要求非默认 k 值时，需分别取 dense 与 sparse 两路结果自行融合。
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "question": (p.payload or {}).get("question", ""),
                "answer": (p.payload or {}).get("answer", ""),
                "score": float(p.score),
            }
            for p in response.points
        ]

    def sparse_search(
        self, sparse_vector: models.SparseVector, *, limit: int = 30
    ) -> list[dict]:
        """单路稀疏向量（BM25）检索。与 `dense_search` 配对使用。"""
        response = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "question": (p.payload or {}).get("question", ""),
                "answer": (p.payload or {}).get("answer", ""),
                "score": float(p.score),
            }
            for p in response.points
        ]

    # ---- 健康检查 ----

    def health(self) -> dict:
        """返回 `{ok, points_count, latency_ms}`；异常不外抛，用 -1 标记。"""
        started = time.perf_counter()
        try:
            points = self.count()
            return {
                "ok": True,
                "points_count": points,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        except Exception as exc:
            logger.warning("Qdrant 健康检查失败: %s", exc)
            return {"ok": False, "points_count": -1, "latency_ms": -1}
