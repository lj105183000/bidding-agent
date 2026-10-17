"""法规库：把 `src.constants.regulations` 的条文写入 Qdrant **独立集合**并检索。

## 为什么单独一个集合

`bid_qa`（默认集合，`settings.qdrant_collection`）里是 75 条**经验问答对**，
payload 是 `{question, answer}`；法规库的 payload 是**逐条原文 + 条号定位**。
两者检索语义完全不同：问答对回答"通常怎么做"，法条回答"依据是什么、怎么规定的"。
混在一个集合里，`/api/ask` 的知识库检索会开始返回法条原文当成"经验答案"，
而合规检查又会被问答对污染。故本模块使用**独立集合** `bid_regulations`，
**不读写 `bid_qa`**。

## 三个设计决定

1. **point id 用 `law|article` 的 UUID5，不用自增序号**。
   序号式 id（`ingest.py` 里用的 `idx`）在语料增删后会整体错位：插入一条就
   让后面所有点换了 id，重复入库会写出重复点。UUID5 由内容决定，
   同一部法规的同一个条号永远映射到同一个 id，**重复入库天然幂等**。

2. **稀疏向量用"无词表 BM25"（`_term_frequencies`），不用 `BM25Encoder`**。
   `ingest.py` 的语料导入是全量重建语义，因为它把 BM25 的 idf 落盘到
   `data/vocab.json`——词表索引一变，历史稀疏向量的含义就全错了。
   法规库**刻意不引入这个耦合**：稀疏向量的下标由词条哈希决定、权重只用词频，
   **不依赖任何语料级统计量**，因此增一条法条不必重建全库，
   也不产生任何需要与 Qdrant 保持同步的文件。代价是没有 idf 加权，
   但 RRF 融合只看排名（见下条），对权重尺度不敏感。

3. **检索是 dense + sparse 双路 → RRF 融合，而 `score` 仍返回余弦相似度**。
   纯稠密检索在"问法措辞与法条不一致"的问题上会漏——实测（真实模型 + 180 条
   真实库）：问"要求投标人在本地有分支机构"，最贴切的《招标投标领域公平竞争
   审查规则》第六条（"不得要求经营主体在本地区设立分支机构……"）被挤出前 3；
   加上关键词路后升到第 1（对比数据见 `search_regulations`）。
   ★ 但 `score` **不用** RRF 融合分：RRF 只反映候选内排名，任何查询都会有第一名，
   无法判断"这条到底相不相关"（这一坑见 `rag/vector_store.py` 的长注释）。
   故排完序后补算余弦，供调用方设阈值。

4. **不做查询改写、不做精排**。本模块只负责"给定问题，取回最相关的条文"，
   是否要精排、要不要让 LLM 判读由上层（合规检查服务）决定。
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter
from typing import Any, Iterable, Sequence

from qdrant_client import models

from src.constants.regulations import RegulationArticle, all_articles
from src.logging_config import get_logger

logger = get_logger(__name__)

# 独立集合名。**不要**改成 settings.qdrant_collection——那是问答集合 bid_qa。
REGULATION_COLLECTION = "bid_regulations"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
UPSERT_BATCH = 128
# 双路召回各自的候选池：法规库才 180 条，取 30 条已覆盖很宽，
# 又给 RRF 留足重排空间（与 rag/constants.py 的 PREFETCH_LIMIT 同量级）
PREFETCH_LIMIT = 30
# UUID5 命名空间：固定值，绝不能改——改了所有 point id 都会变，
# 旧点不会被覆盖而是与新点共存，表现为"条数翻倍"。
POINT_NAMESPACE = uuid.UUID("6f3c1a52-9b7e-4d21-8c4f-2e5d7a90b1c3")

# 稀疏向量下标空间：Qdrant 要求 index 为非负整数
_SPARSE_INDEX_SPACE = 2**31 - 1

# 检索时取回的字段
_PAYLOAD_FIELDS = ("law", "law_short", "article", "text", "topics", "level", "effective")


class RegulationError(Exception):
    """法规库操作的统一异常。

    调用方（合规检查服务 / API 端点）只需捕获这一个类型；集合不存在、
    Qdrant 不可达、向量化失败等都归一到这里，避免把 qdrant_client 的内部
    异常类型泄漏到业务层。
    """


# ---------------------------------------------------------------------
# point id
# ---------------------------------------------------------------------


def point_id(article: RegulationArticle | str, article_no: str | None = None) -> str:
    """由 `law|article` 生成稳定 UUID 字符串。

    支持两种调用方式：
        point_id(article)                 # 传 RegulationArticle
        point_id("中华人民共和国招标投标法", "第四十三条")   # 传法规全称 + 条号

    返回标准 UUID 字符串（Qdrant 接受 str 形式的 UUID 作为 point id）。
    """
    if isinstance(article, RegulationArticle):
        key = article.key
    elif article_no is not None:
        key = f"{article}|{article_no}"
    else:
        raise RegulationError("point_id 需要 RegulationArticle，或 (law, article) 两个字符串")
    return str(uuid.uuid5(POINT_NAMESPACE, key))


def document_text(item: RegulationArticle) -> str:
    """用于**稠密向量**的文本：`法规简称 条号 topics：原文`。

    为什么要拼法规名与条号：条文正文里往往不出现法规名（例如"投标保证金不得
    超过招标项目估算价的2%"），而用户的提问可能只说"保证金上限是多少"。
    把法规名与条号拼进去能让"招标投标法实施条例 第二十六条"这类定位式追问
    也命中，同时不显著稀释语义（法规名只有十来个字）。

    为什么还要拼 `topics`：主题标签是我们对"这条能答什么问题"的人工归纳
    （例如保证金的条目标了"比例上限""退还"），而条文原文用词古板
    （"不得超过招标项目估算价的2%"）。拼进去相当于给每条法条挂上一组
    可被检索到的同义说法，实测能提升"口语化问法"的召回（见模块 docstring 第 3 条）。
    标签不会写进 payload 之外的地方，前端展示的仍是干净的原文。
    """
    topics = " ".join(item.topics)
    prefix = f"{item.law_short} {item.article} {topics}" if topics else f"{item.law_short} {item.article}"
    return f"{prefix}：{item.text}"


# ---------------------------------------------------------------------
# 稀疏向量：无词表 BM25（只依赖词频，不依赖语料统计量）
# ---------------------------------------------------------------------


def _term_frequencies(text: str) -> list[str]:
    """分词，复用 `rag.embedder` 的词表清洗规则。

    与 `ingest.py` 的 `BM25Encoder` 用同一套 `_tokenize`：过滤单字、纯数字、
    纯符号。复用它而不是自己写一遍，是为了让两个集合的关键词路行为一致——
    同一句查询在两处的分词结果不同会很难排查。

    取不到 `_tokenize`（例如它被重命名）时退回 jieba 精确模式，
    **不让分词实现细节成为硬依赖**。
    """
    try:
        from src.rag.embedder import _tokenize

        return _tokenize(text)
    except Exception:  # pragma: no cover - 仅在 embedder 内部实现变更时触发
        import jieba

        return [t.strip() for t in jieba.lcut(text or "") if len(t.strip()) >= 2]


def _sparse_index(term: str) -> int:
    """词条 → 稀疏向量下标。

    用 `blake2b` 而不是 Python 内置 `hash()`：后者对 str 加了**进程级随机盐**
    （PYTHONHASHSEED），换个进程跑下标就全变了，入库的向量与查询的向量
    会对不上——而且是静默地"检索结果莫名其妙变差"，极难排查。
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SPARSE_INDEX_SPACE


def _tf_weights(text: str) -> models.SparseVector:
    """把文本转成"词频饱和"稀疏向量：`weight = tf / (tf + 1)`。

    - 下标由词条哈希决定，**同一词条在任何进程、任何时候都是同一个下标**；
    - 权重只与本文本内的词频有关，**不需要词表、不需要 idf、不需要平均文档长度**。

    为什么不像 `BM25Encoder` 那样算完整 BM25：那需要把全语料的 idf 与 avgdl
    落盘（`data/vocab.json`），法规语料一增补就必须整库重建，否则历史向量
    与新词表错位。法规库要的是"随时能加一条法条"，故放弃 idf——
    RRF 融合只看排名不看分值，缺 idf 只影响单路内部排序的细节，
    对最终融合结果影响有限（实测 5 个合规问题的 top-3 全部命中预期条文）。

    `tf / (tf + 1)` 而不是原始词频：把权重压在 [0, 1) 区间，避免长条文里
    高频词产生过大数值；同时它是单调递增的，不改变单篇内词条的相对次序。
    """
    counts = Counter(_term_frequencies(text))
    if not counts:
        return models.SparseVector(indices=[], values=[])
    pairs = sorted((_sparse_index(term), float(tf)) for term, tf in counts.items())
    indices = [idx for idx, _ in pairs]
    values = [tf / (tf + 1.0) for _, tf in pairs]
    return models.SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------
# 存储实例
# ---------------------------------------------------------------------


def _build_store(store: Any | None = None):
    """构造（或复用外部注入的）法规库。

    `store` 是"鸭子类型"参数：只要能提供
    `ensure_collection / upsert / count / client / collection` 即可，
    因此测试可注入 `VectorStore(url=":memory:", ...)`，一行不碰真实 Qdrant。
    """
    if store is not None:
        return store

    from src.config import settings
    from src.rag.vector_store import VectorStore

    return VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=REGULATION_COLLECTION,
        dim=settings.qdrant_vector_size,
        distance=settings.qdrant_distance,
        timeout=settings.qdrant_timeout,
    )


def _resolve_embedder(embedder: Any | None = None):
    if embedder is not None:
        return embedder
    from src.rag.embedder import embedder as default_embedder

    return default_embedder


def _ensure_topics_index(store: Any) -> None:
    """为 `topics` 建 keyword payload 索引。**幂等**。

    ★ 为什么必须建（实测重大缺陷）
    ---------------------------
    `_topic_filter` 用 `FieldCondition(key="topics", match=MatchValue(...))` 过滤，
    而 **Qdrant 对数组字段做过滤必须先建 payload 索引**，否则直接 400：

        Bad request: Index required but not found for "topics" ...

    而 `search_regulations` 会把 400 包成 `RegulationError` 抛出去，
    `compliance._fetch_law_basis` 又把它当成"语料不可用"静默降级——
    **于是「按主题取法规依据」这条路从来没工作过**：

        · 带 topics 的检索必然 400 → 候选里只剩不带过滤的少数几条
        · 那几条再被分数门槛（0.60）筛掉 → 法规依据恒为空
        · 而 `POST /api/compliance/check` 一直是这样返回的

    这类"两层静默降级叠加"的缺陷最难发现：每一层单独看都合理。
    Qdrant 的报错本身已给出解法（"Create an index for this key"），
    但它出现在被 catch 的日志里，此前没被读到。
    """
    client = getattr(store, "client", None) or getattr(store, "_client", None)
    if client is None:
        return
    try:
        client.create_payload_index(
            collection_name=REGULATION_COLLECTION,
            field_name="topics",
            field_schema="keyword",
            wait=True,
        )
        logger.info("已为 %s.topics 建 keyword 索引", REGULATION_COLLECTION)
    except Exception as exc:
        # 索引已存在时不同 Qdrant 版本行为不一（有的报错有的幂等），
        # 故只记日志不抛——**真正的问题会在检索时报出来**。
        logger.debug("创建 topics 索引未成功（可能已存在）: %s", exc)


def ensure_regulation_collection(store: Any | None = None, recreate: bool = False) -> Any:
    """确保法规集合存在（幂等），返回底层 store。

    `recreate=True` 先删后建——只在语料结构变化（例如向量维度调整）时用，
    日常增补语料直接 `upsert_regulations()` 即可，point id 稳定所以不会重复。
    """
    target = _build_store(store)
    try:
        # 法规库需要 dense + sparse 两个命名向量（双路检索，见模块 docstring）。
        # `VectorStore.ensure_collection` 恰好建的就是这套结构，
        # 因此这里直接复用它——**不修改 vector_store.py**（那是问答链路的代码）。
        target.ensure_collection(recreate=recreate)
    except Exception as exc:  # 网络/权限/磁盘等
        raise RegulationError(f"创建法规集合失败: {exc}") from exc
    # 建 topics 索引。**recreate=True 会连索引一起删掉**，
    # 故必须在建集合之后、检索之前补建。
    _ensure_topics_index(target)
    return target


# ---------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------


def _as_article_list(articles: Iterable[RegulationArticle] | None) -> list[RegulationArticle]:
    items = list(articles) if articles is not None else list(all_articles())
    if not items:
        raise RegulationError("没有可入库的条文（articles 为空）")

    seen: set[str] = set()
    deduped: list[RegulationArticle] = []
    for item in items:
        if not isinstance(item, RegulationArticle):
            raise RegulationError(f"articles 里混入了非 RegulationArticle 对象: {type(item)!r}")
        if not item.law or not item.article or not item.text:
            raise RegulationError(f"条文缺少必填字段（law/article/text）: {item.key}")
        if item.key in seen:
            # 同 law+article 出现两次：这是语料 bug，静默去重会掩盖问题
            raise RegulationError(f"条文重复: {item.key}")
        seen.add(item.key)
        deduped.append(item)
    return deduped


def upsert_regulations(
    articles: Iterable[RegulationArticle] | None = None,
    *,
    recreate: bool = False,
    store: Any | None = None,
    embedder: Any | None = None,
    batch_size: int = UPSERT_BATCH,
    progress: bool = True,
    sparse: bool = True,
) -> int:
    """把法规条文写入 `bid_regulations`，返回写入的条数。

    - **幂等**：point id 由 `law|article` 决定，重复调用只会覆盖同一条，
      不会产生重复点（这是本模块对 `ingest.py` 序号式 id 的关键改进）。
    - `recreate=True` 先删集合再建，用于语料结构变更后的全量重建。
    - `store` / `embedder` 可注入，便于测试与离线验证。
    - `progress=True` 时按批打印进度（入库脚本依赖它展示"入库进度"）。
    - `sparse=False` 只写稠密向量：用于目标集合是 dense-only 的极端情况
      （Qdrant 集合建成后不能新增向量字段，写多余的字段会被拒）。

    `articles=None` 表示写入模块内置的**全部**语料。
    """
    items = _as_article_list(articles)

    docs = [document_text(item) for item in items]

    target = ensure_regulation_collection(store, recreate=recreate)
    enc = _resolve_embedder(embedder)

    logger.info("法规库向量化 %d 条条文...", len(docs))
    try:
        vectors = enc.encode_documents(docs)
    except Exception as exc:
        raise RegulationError(f"条文向量化失败: {exc}") from exc
    if len(vectors) != len(items):
        raise RegulationError(
            f"向量条数与条文条数不一致: {len(vectors)} vs {len(items)}"
        )

    points = []
    for idx, item in enumerate(items):
        # 稀疏向量与稠密向量用**同一段文本**（含 topics），保证两路看到的内容一致
        vectors_map: dict[str, Any] = {DENSE_VECTOR_NAME: vectors[idx]}
        if sparse:
            vectors_map[SPARSE_VECTOR_NAME] = _tf_weights(docs[idx])
        points.append(
            models.PointStruct(
                id=point_id(item),
                vector=vectors_map,
                payload=item.to_payload(),
            )
        )

    written = 0
    step = max(1, int(batch_size))
    for start in range(0, len(points), step):
        chunk = points[start : start + step]
        try:
            target.upsert(chunk)
        except Exception as exc:
            raise RegulationError(
                f"写入法规集合失败（第 {start + 1}~{start + len(chunk)} 条）: {exc}"
            ) from exc
        written += len(chunk)
        if progress:
            print(f"  入库进度 {written}/{len(points)}", flush=True)

    logger.info("法规库写入完成: %d 条", written)
    return written


# ---------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------


def _topic_filter(topics: Sequence[str] | None) -> models.Filter | None:
    """把主题标签转成 Qdrant 过滤条件（任一命中即保留）。

    Qdrant 的 `MatchValue` 作用在数组字段上时是"数组包含该值"的语义，
    所以 `topics` 存成数组是必要的（见 `RegulationArticle.to_payload`）。
    """
    cleaned = [t.strip() for t in (topics or []) if t and t.strip()]
    if not cleaned:
        return None
    return models.Filter(
        should=[models.FieldCondition(key="topics", match=models.MatchValue(value=t)) for t in cleaned]
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。与 `rag/vector_store.py` 的 `_cosine` 同语义。

    这里不 import 它的私有函数：那是问答链路的实现细节，法规链路
    自己留一份 8 行的实现，比跨模块依赖私有名更稳（改动互不影响）。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search_regulations(
    query: str,
    top_k: int = 5,
    topics: Sequence[str] | None = None,
    *,
    store: Any | None = None,
    embedder: Any | None = None,
    hybrid: bool = True,
) -> list[dict]:
    """检索最相关的法规条文。

    返回列表，每项含：
        law / law_short / article / text / topics / level / effective
        score     与查询的**余弦相似度**（-1~1，法规场景通常在 0.4~0.8）
        id        稳定 point id（便于去重、回查、反馈闭环）

    `topics` 非空时做标签过滤（任一命中），用于把检索收窄到某一类合规问题
    （例如只看 `("保证金",)`）。**过滤是先于排序的硬条件**，因此过滤后的
    结果可能少于 `top_k`。

    ## 检索策略：`hybrid=True` 为默认

    双路召回 → Qdrant 服务端 RRF 融合 → 取 top_k → **补算余弦当 `score`**：

        dense 路   bge 向量，管"意思相近"（问"本地要设分公司吗"能召回讲
                   "分支机构"的条文）
        sparse 路  词频饱和的稀疏向量（`_tf_weights`），管"字面命中"
                   （"保证金""2%""15日"这类硬词）

    为什么必须两条路：纯稠密在这份语料上会漏掉最贴切的条文。实测
    （真实模型 + 180 条真实库，`hybrid=False` 即纯稠密）：

        问："要求投标人在本地有分支机构"
        纯稠密 top-3：87号令第十七条 / 招标投标法第十八条 / 招标投标法第三十七条
        混合   top-1：公平竞争审查规则第六条  ← 唯一直接禁止"要求在本地区设立
                      分支机构"的条文，纯稠密下不在前 3

    ★ 为什么 `score` 不是 RRF 融合分：RRF 只反映候选内排名，**任何查询都会有
    第一名**，无法判断"这条到底相不相关"（实测无关问题的融合分能高于相关问题，
    见 `rag/vector_store.py` 的长注释）。故融合排完序后，用
    `with_vectors=True` 顺带取回的稠密向量现场算余弦，作为可设阈值的分数。
    取向量不产生额外 IO——它们与 payload 在同一次查询里回来。

    ⚠️ **已知不足（上线前请看）**：融合排序偶尔会把"定义性"条文排在"给出具体
    数字"的条文之前。实测例子——问"投标保证金比例"，top-1 是《招标投标法实施
    条例》第二十六条（"不得超过招标项目估算价的2%"，正确），但第 2 名是同条例
    第五十八条（**履约**保证金 10%，另一类保证金），而《政府采购法实施条例》
    第三十三条（采购项目预算金额的 2%）不在前 3；问"招标文件澄清修改提前多少天"，
    top-1 是《招标投标法》第二十三条（答"十五日"，正确），但最细的《实施条例》
    第二十一条（区分 15 日 / 3 日）落到第 3。根因是这几条**都确实相关**，
    要分辨"哪一类项目、哪一类保证金"得读懂语义细节，超出向量相似度的能力。
    对策：
      - 取 `top_k=5~8` 而不是 3，让依据更全（合规结论本就不该只押一条）；
      - 拿到候选后自行精排，或交给 LLM 判读（本模块刻意不做，见模块 docstring 第 4 条）。
    **不要把 top-1 当成唯一依据。**

    异常统一为 `RegulationError`。
    """
    text = (query or "").strip()
    if not text:
        raise RegulationError("检索问题不能为空")
    if top_k <= 0:
        raise RegulationError(f"top_k 必须为正整数，当前 {top_k}")

    target = _build_store(store)
    enc = _resolve_embedder(embedder)

    try:
        vector = enc.encode_query(text)
    except Exception as exc:
        raise RegulationError(f"查询向量化失败: {exc}") from exc

    query_filter = _topic_filter(topics)
    use_hybrid = bool(hybrid) and hasattr(target.client, "query_points")

    try:
        if use_hybrid:
            response = target.client.query_points(
                collection_name=target.collection,
                prefetch=[
                    models.Prefetch(
                        query=vector, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=_tf_weights(text), using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=int(top_k),
                with_payload=True,
                with_vectors=[DENSE_VECTOR_NAME],
            )
        else:
            response = target.client.query_points(
                collection_name=target.collection,
                query=vector,
                using=DENSE_VECTOR_NAME,
                limit=int(top_k),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=[DENSE_VECTOR_NAME],
            )
    except RegulationError:
        raise
    except Exception as exc:
        raise RegulationError(f"法规检索失败: {exc}") from exc

    results: list[dict] = []
    for point in response.points:
        payload = point.payload or {}
        record = {field: payload.get(field) for field in _PAYLOAD_FIELDS}
        record["topics"] = list(payload.get("topics") or [])
        record["level"] = payload.get("level") or ""
        record["effective"] = payload.get("effective") or ""
        record["id"] = str(point.id)

        dense = _extract_dense(getattr(point, "vector", None))
        if dense is not None:
            # 融合查询取回的是稠密向量，现场算余弦；这样 score 的含义与
            # 纯稠密模式完全一致，调用方切换 hybrid 不必改阈值逻辑。
            record["score"] = float(_cosine(vector, dense))
        else:
            # 取不到向量时退回 Qdrant 给的分数（融合分或余弦），并标明来源，
            # 避免调用方误以为它是可比较的余弦值
            record["score"] = float(point.score)
            record["score_source"] = "qdrant"
        record.setdefault("score_source", "cosine")
        results.append(record)
    return results


def _extract_dense(vector: Any) -> list[float] | None:
    """从 Qdrant 返回的 `point.vector` 里取出稠密向量。

    命名向量模式下它是 `{"dense": [...]}`；退出命名模式时可能是裸列表。
    两种都兼容，取不到返回 None（调用方据此降级，而不是抛错）。
    """
    if vector is None:
        return None
    if isinstance(vector, dict):
        value = vector.get(DENSE_VECTOR_NAME)
        return list(value) if value is not None else None
    if isinstance(vector, (list, tuple)):
        return list(vector)
    return None


# ---------------------------------------------------------------------
# 统计与校验
# ---------------------------------------------------------------------


def regulation_stats(store: Any | None = None, *, raise_on_error: bool = False) -> dict:
    """库里现有多少条、按法规分布。

    返回：
        {
          "collection": "bid_regulations",
          "points": int,                 # 集合内实际点数（连不上时为 -1）
          "laws": int,                   # 实际出现的法规数
          "by_law": {法规全称: 条数},
          "corpus_articles": int,        # 模块内置语料条数（用于比对是否漏入库）
          "corpus_laws": int,
          "in_sync": bool,               # points == corpus_articles
        }

    默认**不抛异常**：统计接口常用于健康检查，连不上库时用 -1 标记并记日志，
    比让整个检查端点 500 更有用。需要严格语义时传 `raise_on_error=True`。
    """
    target = _build_store(store)
    stats: dict = {
        "collection": getattr(target, "collection", REGULATION_COLLECTION),
        "points": -1,
        "laws": 0,
        "by_law": {},
        "corpus_articles": len(all_articles()),
        "corpus_laws": len({item.law for item in all_articles()}),
        "in_sync": False,
    }

    try:
        stats["points"] = int(target.count())
    except Exception as exc:
        logger.warning("法规库计数失败: %s", exc)
        if raise_on_error:
            raise RegulationError(f"法规库计数失败: {exc}") from exc
        return stats

    try:
        by_law = _count_by_law(target)
    except Exception as exc:  # 计数成功但分组失败：不影响 points 的可用性
        logger.warning("法规库按法规分组失败: %s", exc)
        if raise_on_error:
            raise RegulationError(f"法规库分组统计失败: {exc}") from exc
        by_law = {}

    stats["by_law"] = by_law
    stats["laws"] = len(by_law)
    stats["in_sync"] = stats["points"] == stats["corpus_articles"]
    return stats


def _count_by_law(target: Any, page: int = 512) -> dict[str, int]:
    """滚动翻页统计每个法规的条数。

    为什么不用 `count(exact=True)` 逐法规过滤：那要对 9 部法规发 9 次请求，
    而且法规清单一旦增补还得改代码；翻页一次扫完、按 payload 聚合更省事，
    180 条的体量下开销可忽略。
    """
    by_law: dict[str, int] = {}
    offset = None
    while True:
        records, offset = target.client.scroll(
            collection_name=target.collection,
            limit=page,
            offset=offset,
            with_payload=["law"],
            with_vectors=False,
        )
        for record in records:
            law = (record.payload or {}).get("law") or "（缺失 law 字段）"
            by_law[law] = by_law.get(law, 0) + 1
        if offset is None:
            break
    return dict(sorted(by_law.items(), key=lambda kv: (-kv[1], kv[0])))


def verify_regulations(store: Any | None = None) -> dict:
    """入库后校验：库里条数、与语料是否一致、抽样条文。

    返回 `regulation_stats()` 的结果，另加：
        sample  随机抽取的 3 条（law / article / text 前 60 字 / topics），
                供人工确认"写进去的确实是法条而不是别的集合的残留"。
    """
    stats = regulation_stats(store, raise_on_error=True)
    target = _build_store(store)

    sample: list[dict] = []
    try:
        records, _ = target.client.scroll(
            collection_name=target.collection,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            sample.append(
                {
                    "law": payload.get("law", ""),
                    "article": payload.get("article", ""),
                    "text": str(payload.get("text", ""))[:60],
                    "topics": list(payload.get("topics") or []),
                }
            )
    except Exception as exc:
        logger.warning("法规库抽样失败: %s", exc)
        raise RegulationError(f"法规库抽样失败: {exc}") from exc

    stats["sample"] = sample
    return stats
