"""会话记忆与反馈回流。

★ 现实前提：不能依赖样本量（实测只有 1 条反馈）
--------------------------------------------
设计前先查了库：`feedback` 表 1 条、`messages` 12 条、`conversations` 4 条。
这个量级下任何"从历史学习"的方案都无法验证，也谈不上收益。

故本模块**不依赖统计规律**，只做两件确定的事：

    1. **反馈回流**：把**被采纳过**（`rating='up'`）的历史问答作为参考示例
       注入生成上下文。用一条就有一条的效果，不需要样本量。
    2. **缺口聚合**：把被点踩（`rating='down'`）的问题聚成清单，
       交给人工补语料——这是把反馈变成**可执行动作**，而不是停在看板数字上。

为什么不直接用全部历史问答：历史里混着**答错的、被点踩的**回答，
把它们当示例注入等于教模型重复错误。只用 `rating='up'` 是有意的筛选。

★ 与知识库的关系
---------------
知识库（75 条）是**权威语料**；会话记忆是**经验**。二者定位不同：
    知识库：整理过、可信、但覆盖有限（实测 5/25 评测题库内无对应内容）
    会话记忆：未经整理、但有具体问题的完整回答（实测平均 480 字，
              而库内答案平均 185 字——历史上限更长、更有解释性）

故记忆注入时明确标注来源与"仅供参考"，不与知识库混淆。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.logging_config import get_logger

logger = get_logger(__name__)

#: 记忆示例的条数上限。
#:
#: 取 2 而非更多：示例占 prompt 预算，且历史回答平均 480 字——
#: 3 条就到 1400 字，会挤压真正的知识库参考。2 条足够起示范作用。
MAX_MEMORY_EXAMPLES = 2

#: 相似度阈值。低于它说明这道历史问题与当前问题关系不大，注入只会干扰。
#:
#: 与 `vector_store.MIN_RELEVANCE` 同一量纲（都是稠密余弦），但**取值不同**：
#: 知识库那道阈值是"能否回答"的判据（0.5）；这里的阈值是"是否值得当示例"，
#: 要求更高——半相关的示例会误导模型套用错误结构。实测同一问题与历史答案的
#: 相似度普遍高于问题对问题（答案更长、词更多），故 0.55 偏保守。
MIN_MEMORY_SIMILARITY = 0.55


@dataclass
class MemoryExample:
    """一条可复用的历史问答。"""

    question: str
    answer: str
    rating: str
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "rating": self.rating,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class KnowledgeGap:
    """一条待补的知识缺口（来自被点踩的问答）。"""

    question: str
    answer: str
    comment: str = ""

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "comment": self.comment}


@dataclass
class _Cache:
    """进程内缓存：编码历史问答是 CPU 密集操作，不能每次问答都重做。"""

    lock: threading.Lock = field(default_factory=threading.Lock)
    examples: list[MemoryExample] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    failed: bool = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class SessionMemory:
    """会话记忆：从被采纳的历史问答里取相似示例。

    依赖注入 `loader`（返回已采纳问答）与 `embedder`，便于测试替换；
    默认走 `PostgresClient` 与 `src.rag.embedder.embedder`。
    """

    def __init__(self, loader=None, embedder=None) -> None:
        self._loader = loader
        self._embedder = embedder
        self._cache = _Cache()

    # ---- 依赖懒加载 ----

    def _get_loader(self):
        if self._loader is not None:
            return self._loader
        return _load_accepted_from_db

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from src.rag.embedder import embedder

        return embedder

    # ---- 缓存 ----

    def _ensure_cache(self) -> None:
        """加载并编码历史问答（只做一次，失败后不再重试）。

        失败后**不再重试**是有意的：若数据库暂时不可用，每次问答都重试会
        把延迟叠加到每个请求上。失败即降级为"无记忆"，日志留痕。
        """
        cache = self._cache
        if cache.examples or cache.failed:
            return
        with cache.lock:
            if cache.examples or cache.failed:
                return
            try:
                rows = list(self._get_loader()() or [])
            except Exception:
                logger.warning("会话记忆加载失败，本次及后续降级为无记忆", exc_info=True)
                cache.failed = True
                return
            examples: list[MemoryExample] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                q = str(row.get("question") or "").strip()
                a = str(row.get("answer") or "").strip()
                if q and a:
                    examples.append(MemoryExample(
                        question=q, answer=a, rating=str(row.get("rating") or "")
                    ))
            if not examples:
                cache.failed = True
                logger.info("会话记忆为空（没有已采纳的历史问答），后续按无记忆处理")
                return
            try:
                vectors = self._get_embedder().encode_documents(
                    [e.question for e in examples]
                )
            except Exception:
                logger.warning("会话记忆编码失败，降级为无记忆", exc_info=True)
                cache.failed = True
                return
            cache.examples = examples
            cache.vectors = [list(v) for v in vectors]
            logger.info("会话记忆已加载 %d 条已采纳问答", len(examples))

    def recall(self, question: str, top_k: int = MAX_MEMORY_EXAMPLES) -> list[MemoryExample]:
        """取与当前问题最相似的历史示例。**永不抛异常**。

        只返回相似度 >= `MIN_MEMORY_SIMILARITY` 的；一条都不满足时返回空列表——
        宁可不给示例，也不给一个半相关的把模型带偏。
        """
        q = (question or "").strip()
        if not q:
            return []
        self._ensure_cache()
        if not self._cache.examples:
            return []
        try:
            qv = list(self._get_embedder().encode_query(q))
        except Exception:
            logger.warning("会话记忆查询编码失败", exc_info=True)
            return []
        if not qv:
            return []

        scored: list[MemoryExample] = []
        for example, vec in zip(self._cache.examples, self._cache.vectors):
            sim = _cosine(qv, vec)
            if sim >= MIN_MEMORY_SIMILARITY:
                scored.append(MemoryExample(
                    question=example.question,
                    answer=example.answer,
                    rating=example.rating,
                    similarity=sim,
                ))
        scored.sort(key=lambda e: e.similarity, reverse=True)
        return scored[: max(0, top_k)]

    def invalidate(self) -> None:
        """清空缓存。用户新增反馈后应调用，否则新采纳的问答不会进入记忆。"""
        with self._cache.lock:
            self._cache.examples = []
            self._cache.vectors = []
            self._cache.failed = False


def build_memory_note(examples: list[MemoryExample]) -> str:
    """把历史示例渲染成注入 system 的文本。

    措辞要点：
        · 标明**来源是历史对话**，不是知识库——两者可信度不同，不能混
        · 标明**仅供参考**，并以"若与知识库参考冲突，以知识库为准"
          给出优先级，避免历史里的过时内容压过权威语料
        · 明确它**不是**当前问题的答案，防止模型直接照抄
    """
    if not examples:
        return ""
    blocks = [
        f"【历史问答{i}（来源：本系统过往被采纳的回答，仅供参考）】\n"
        f"问：{e.question}\n答：{e.answer}"
        for i, e in enumerate(examples, 1)
    ]
    return (
        "以下是本系统过往**被用户采纳**的相似问答，可作为作答风格与详略程度的参考：\n\n"
        + "\n\n".join(blocks)
        + "\n\n注意：以上是**历史经验**而非当前问题的答案，不要直接照搬；"
        "若与「知识库参考」冲突，**以知识库为准**。"
    )


# ===========================================================================
# 默认数据源
# ===========================================================================


def _load_accepted_from_db(limit: int = 200) -> list[dict]:
    """读取被点赞的问答对（问题取自同会话中该回答之前的用户消息）。

    返回 `[{question, answer, rating}]`。

    SQL 说明：`feedback.message_id` 指向被评价的 assistant 消息；
    它对应的问题需要回溯同 `conversation_id` 里 id 更小的最近一条 user 消息。
    用相关子查询而不是 join，是因为"最近一条更小的 user 消息"没有等值连接键。
    """
    from src.database.postgresql_client import PostgresClient

    sql = """
    SELECT
        (
            SELECT u.content
            FROM messages u
            WHERE u.conversation_id = m.conversation_id
              AND u.role = 'user'
              AND u.id < m.id
            ORDER BY u.id DESC
            LIMIT 1
        ) AS question,
        m.content AS answer,
        f.rating  AS rating
    FROM feedback f
    JOIN messages m ON m.id = f.message_id
    WHERE f.rating = 'up'
    ORDER BY f.id DESC
    LIMIT %s
    """
    return PostgresClient()._query(sql, (limit,))


def load_knowledge_gaps(limit: int = 100) -> list[KnowledgeGap]:
    """读取被点踩的问答，作为"待补的知识缺口"清单。

    用途不是自动做什么，而是**给出可执行动作**：运营照这份清单补语料。
    这比看板上的"踩 3 次"有用得多——那个数字不告诉你该补什么。
    """
    from src.database.postgresql_client import PostgresClient

    sql = """
    SELECT
        (
            SELECT u.content
            FROM messages u
            WHERE u.conversation_id = m.conversation_id
              AND u.role = 'user'
              AND u.id < m.id
            ORDER BY u.id DESC
            LIMIT 1
        ) AS question,
        m.content AS answer,
        COALESCE(f.comment, '') AS comment
    FROM feedback f
    JOIN messages m ON m.id = f.message_id
    WHERE f.rating = 'down'
    ORDER BY f.id DESC
    LIMIT %s
    """
    rows = PostgresClient()._query(sql, (limit,))
    out: list[KnowledgeGap] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        out.append(KnowledgeGap(
            question=q,
            answer=str(row.get("answer") or ""),
            comment=str(row.get("comment") or ""),
        ))
    return out


#: 进程内单例。跨请求复用编码结果——历史问答的编码是 CPU 密集操作。
session_memory = SessionMemory()


__all__ = [
    "MAX_MEMORY_EXAMPLES",
    "MIN_MEMORY_SIMILARITY",
    "KnowledgeGap",
    "MemoryExample",
    "SessionMemory",
    "build_memory_note",
    "load_knowledge_gaps",
    "session_memory",
]
