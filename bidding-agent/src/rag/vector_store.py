"""Qdrant 向量库封装：命名向量（dense + sparse）+ 原生混合检索。

**为什么用命名向量而不是两个集合**：把 dense 与 sparse 存在同一个点上，
检索时用 `Prefetch` 两路并行召回、再交给 `FusionQuery(RRF)` 融合，一次
网络往返就完成混合检索；分成两个集合则要在应用层对齐 id、自己实现融合，
并且拿不到 Qdrant 服务端的执行优化。

**版本注意**：qdrant-client ≥1.9 已移除 `client.search()`，全部走
`query_points`（见 docs/技术栈.md §12 踩坑记录）。
"""

from __future__ import annotations

import math
import threading
import time

from qdrant_client import QdrantClient, models

from src.logging_config import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
UPSERT_BATCH = 256

# 判定"检索结果是否真的与该问题相关"的余弦相似度阈值。
#
# 标定数据（2026-10-12，本项目 75 条知识库语料）：
#
#     口径：每个问题的**最佳结果**（top1）——决定"该不该回答"看的是最佳匹配
#     相关问题 18 个：最低 0.5806  中位 0.6257  最高 0.7306
#     无关问题 12 个：最高 0.4256  中位 0.3555  最低 0.2964
#     >>> 间隔 0.1550，取中点 0.5 留有余量（距两侧各约 0.08）
#
# ⚠️ 不要用 `score`（RRF 融合分）代替它做这个判断：RRF 只反映候选内排名，
# 实测无关问题的 RRF 分（最高 0.500）能超过相关问题（最低 0.081），完全不可用。
#
# ⚠️ 语料规模变化后需重新标定：语料从 75 条扩到数千条时，
# 无关问题的 top1 相似度通常会上移（候选变多，总能撞到一条"看起来像"的），
# 阈值可能要上调。
MIN_RELEVANCE = 0.5

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


def _cosine(a, b) -> float:
    """两个等长向量的余弦相似度。

    为什么不用 numpy：这里每次只算 5 条 × 512 维，纯 Python 点积的耗时可忽略
    （微秒级），而引入 numpy 依赖只为这点计算不划算。项目其他位置也没用 numpy。

    零向量（范数为 0）返回 0.0 而不是抛错或 NaN——向量库里若混入零向量，
    检索不该因此崩掉；返回 0 表示"无相似性"，语义合理。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


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
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrrf_k: int | None = None,
    ) -> list[dict]:
        """dense + sparse 双路召回 → **加权** RRF 融合（服务端）。

        ★ 权重与 k（两者独立）
        -------------------
            dense_weight / sparse_weight   调"信哪一路"（整体影响力）
            rrrf_k                          调"多看重头部"（排名衰减速度）

        Qdrant 1.19 的 `Rrf` 模型支持 `weights` 与 `k`：
            "Weights for each prefetch source. Higher weight gives more influence
             on the final ranking... The number of weights should match the number
             of prefetches."

        故服务端能直接做加权融合，无需在应用层重算。
        **权重顺序必须与 prefetch 顺序一致**——这里是 `[dense, sparse]`。

        ★ `rrrf_k` 传 None 时用 Qdrant 默认
        ---------------------------------
        `Rrf(k=None)` 表示用服务端默认（60）。保留该参数是为了让调用方
        在 k=60 时也能走服务端路径（省一次网络往返）。

        ★ 返回值同时给出 `score` 与 `relevance`，两者含义**不同**，不要混用：

            score      加权 RRF 融合分。只反映"在两路候选中的排名"，取值范围小，
                       **任何查询都会有第一名**，因此**不能用它判断相关性**。
            relevance  与查询的稠密余弦相似度（-1~1）。实测相关问 0.67~0.71、
                       无关问 0.29~0.34，**可以**用来判断"这条到底相不相关"。

        需要判断"检索结果是否真的相关"时必须用 `relevance`（例如
        `/api/ask` 要避免把无关问答当作答案返回）。

        实现上向量来自 `with_vectors`——与 payload 在同一次查询里取回，
        不产生额外 IO；计算是本地点积，5 条记录的代价可忽略。
        之所以不用 `score` 近似，是因为 RRF 分与相似度之间没有单调关系。
        """
        # 权重与 k 只在非默认时显式构造 `Rrf`，默认路径保持原样
        # （少传参数 = 少一处与服务端默认值耦合）。
        use_explicit = (dense_weight != 1.0 or sparse_weight != 1.0 or rrrf_k is not None)
        if use_explicit:
            fusion = models.RrfQuery(
                rrf=models.Rrf(
                    weights=[dense_weight, sparse_weight],
                    k=rrrf_k,
                )
            )
        else:
            fusion = models.FusionQuery(fusion=models.Fusion.RRF)

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
            query=fusion,
            limit=limit,
            with_payload=True,
        )
        points = list(response.points)

        # ★ 向量要单独取：`FusionQuery`（聚合查询）**不会**返回向量，
        # 即使传了 `with_vectors` 也是 None —— 实测踩过这个坑。
        # 故对已返回的 top_k 个点做一次定点 retrieve。
        # 代价很小：只取 limit（默认 5）条，且本地 Qdrant 下是毫秒级；
        # 集合是 512 维，单条 2KB，5 条共 10KB。
        vectors = self._fetch_vectors([p.id for p in points])
        return [self._to_record(p, dense_vector, vectors.get(p.id)) for p in points]

    def _fetch_vectors(self, ids: list) -> dict:
        """按 id 取回稠密向量。失败时返回空 dict（relevance 会是 None 而非 0）。"""
        if not ids:
            return {}
        try:
            recs = self.client.retrieve(
                collection_name=self.collection,
                ids=ids,
                with_vectors=[DENSE_VECTOR_NAME],
                with_payload=False,
            )
        except Exception as exc:
            logger.warning("取回向量失败，relevance 将不可用: %s", exc)
            return {}

        out: dict = {}
        for r in recs:
            vec = r.vector
            dense = vec.get(DENSE_VECTOR_NAME) if isinstance(vec, dict) else vec
            if dense is not None:
                out[r.id] = dense
        return out

    def attach_relevance(self, records: list[dict], query_vector: list[float]) -> list[dict]:
        """给检索结果补算 `relevance`（与查询的稠密余弦相似度），就地返回。

        **为什么必须做成公开方法**：`RAGPipeline._hybrid_retrieve` 有两条路径——
        `rrf_k == 60` 走服务端 `hybrid_search`，否则走
        `dense_search` + `sparse_search` + 应用层 `fuse_rrf`。
        实测分类器给出的 `rrf_k` 恒为 20，即**实际总是走应用层那条**，
        所以只在 `hybrid_search` 里算 relevance 等于写了死代码。

        该方法需要结果里带 `id` 字段才能取向量；没有 id 的记录会保持
        `relevance=None`（未知，而非"不相关"）。

        ★ 曾尝试在这里省掉一次往返，**测不出收益，已回滚**
        ------------------------------------------------
        思路：让 `dense_search` 带 `with_vectors` 把向量一起返回，
        就不必再 `_fetch_vectors` 往返一次。

        实测**无法证明有效**（同一进程交替 30 轮，配对差值符号检验 p=0.29）：

            A（现状：搜 + _fetch）   中位 1738ms
            B（优化：搜时带向量）     中位 1520ms
            配对差值中位 -230ms，但 p25/p75 = -883 / +407ms
            B 更快的轮次 17/30（约等于抛硬币）

        问题在于**这台机器与 Qdrant 的噪声大于方案差异**——
        同一操作实测在 208ms 与 2383ms 之间跳动（11 倍）。
        按"不确定就不改"的原则回滚：多一个内部字段要穿过 `fuse_rrf`
        再被剥离，复杂度的代价换不到可验证的收益。

        **若日后要重试**：先解决测量稳定性（受控的本地 Qdrant +
        多次采样取 p25 而非中位），否则同样的结论会重现。
        """
        ids = [r.get("id") for r in records if r.get("id") is not None]
        if not ids:
            return records
        vectors = self._fetch_vectors(ids)
        for r in records:
            rid = r.get("id")
            dense = vectors.get(rid) if rid is not None else None
            if dense is None or not query_vector:
                r.setdefault("relevance", None)
                continue
            try:
                r["relevance"] = float(_cosine(query_vector, dense))
            except (TypeError, ValueError, ZeroDivisionError):
                r.setdefault("relevance", None)
        return records

    @staticmethod
    def _to_record(point, query_vector: list[float], dense=None) -> dict:
        """把 Qdrant 返回的 point 归一成内部记录，并补算 relevance。

        向量可能取不到（聚合查询不返回、或集合结构特殊），此时 `relevance`
        为 None 而不是 0——**未知与"完全无关"是两回事**，
        用 0 会让调用方误以为已经算过了。
        """
        payload = point.payload or {}
        relevance = None
        if dense is not None and query_vector:
            try:
                relevance = float(_cosine(query_vector, dense))
            except (TypeError, ValueError, ZeroDivisionError):
                relevance = None
        return {
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "score": float(point.score),
            "relevance": relevance,
        }

    def dense_search(
        self, dense_vector: list[float], *, limit: int = 30
    ) -> list[dict]:
        """单路稠密向量检索。供应用层 RRF（`classify_question.fuse_rrf`）使用——
        当分类器要求非默认 k 值时，需分别取 dense 与 sparse 两路结果自行融合。

        `id` 必须保留：`attach_relevance` 靠它取回向量计算余弦相似度。
        `fuse_rrf` 用 `dict(payload[key])` 复制记录，故 id 能穿过应用层融合。
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
                "id": p.id,
                "question": (p.payload or {}).get("question", ""),
                "answer": (p.payload or {}).get("answer", ""),
                "score": float(p.score),
            }
            for p in response.points
        ]

    def sparse_search(
        self, sparse_vector: models.SparseVector, *, limit: int = 30
    ) -> list[dict]:
        """单路稀疏向量（BM25）检索。与 `dense_search` 配对使用。

        同样保留 `id`（理由见 `dense_search`）——不过要注意：BM25 路命中的记录
        若 dense 路没命中，取回向量后算出的相似度可能偏低，
        这不代表记录不相关，只是"向量层面不像"。故 relevance 只应与
        同一路径的结果比较，不要跨路径下结论。
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": p.id,
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
