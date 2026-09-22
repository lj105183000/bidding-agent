"""知识库导入：Excel 问/答 → Qdrant（dense + BM25 sparse）+ 词表落盘。

导入是**全量重建**语义（`force=True`）：删旧集合 → 重建 → 重新 fit 词表 →
写入全部向量。之所以不能增量追加，是因为 BM25 的 idf 与平均文档长度依赖
整个语料，词表索引一变，历史稀疏向量的含义就全错了（见 docs/开发文档.md §4.1）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qdrant_client import models

from src.config import PROJECT_ROOT
from src.logging_config import get_logger
from src.rag.constants import DATA_DIR
from src.rag.embedder import BM25Encoder, Embedder
from src.rag.vector_store import SPARSE_VECTOR_NAME, DENSE_VECTOR_NAME

logger = get_logger(__name__)

# 列名候选：中文表头优先，其次英文，最后退回"前两列"
QUESTION_ALIASES = ("问", "问题", "question", "q")
ANSWER_ALIASES = ("答", "答案", "answer", "a")


def find_source_excel(explicit: Path | None = None) -> Path:
    """定位源数据：显式路径 > `data/processed/` 下第一个 xlsx > `data/` 递归第一个。"""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"指定的数据文件不存在: {path}")
        return path

    for directory in (DATA_DIR / "processed", DATA_DIR):
        if not directory.exists():
            continue
        for candidate in sorted(directory.rglob("*.xlsx")) + sorted(directory.rglob("*.xls")):
            if not candidate.name.startswith("~$"):  # 跳过 Excel 打开时的临时文件
                return candidate
    raise FileNotFoundError(
        f"{DATA_DIR} 下未找到 .xlsx/.xls 数据文件（需要两列：问、答）"
    )


def _pick_columns(df: pd.DataFrame) -> tuple[str, str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}

    def match(aliases: tuple[str, ...]):
        for alias in aliases:
            if alias in lowered:
                return lowered[alias]
        return None

    q_col, a_col = match(QUESTION_ALIASES), match(ANSWER_ALIASES)
    if q_col is not None and a_col is not None:
        return q_col, a_col
    if df.shape[1] < 2:
        raise ValueError(f"数据表至少需要两列（问、答），当前列: {list(df.columns)}")
    # 表头无法识别时按位置取前两列，兼容无表头/自定义表头的数据
    return df.columns[0], df.columns[1]


def clean_qa(rows: list[dict]) -> list[dict]:
    """清洗：去空白 → 去空行 → 按问题去重（保留首次出现）。

    训练数据里重复问题往往对应不同的标答，保留首次出现是为了让结果**可复现**；
    若后续需要择优，应在此处补充明确规则，而不是让顺序决定结果。
    """
    cleaned: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        if question in seen:
            continue
        seen.add(question)
        cleaned.append({"question": question, "answer": answer})
    return cleaned


def load_qa(path: Path | None = None) -> list[dict]:
    source = find_source_excel(path)
    logger.info("读取数据文件: %s", source)
    df = pd.read_excel(source)
    q_col, a_col = _pick_columns(df)
    rows = [
        {"question": row[q_col], "answer": row[a_col]}
        for _, row in df.iterrows()
    ]
    return clean_qa(rows)


def ingest_data(
    force: bool = True,
    path: Path | None = None,
    store=None,
    embedder: Embedder | None = None,
) -> dict:
    """执行导入，返回统计摘要。`store` / `embedder` 可注入以便测试。"""
    from src.rag.embedder import embedder as default_embedder
    from src.rag.vector_store import VectorStore
    from src.config import settings

    records = load_qa(path)
    if not records:
        raise ValueError("清洗后没有可导入的问答数据")

    store = store or VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        dim=settings.qdrant_vector_size,
        distance=settings.qdrant_distance,
        timeout=settings.qdrant_timeout,
    )
    embedder = embedder or default_embedder

    questions = [r["question"] for r in records]

    # BM25 词表必须基于**本次导入的同一批语料**拟合，否则 idf 与向量数据错位
    bm25 = BM25Encoder().fit(questions)

    logger.info("embedding 文档 (%d 条)...", len(questions))
    dense_vectors = embedder.encode_documents(questions)
    sparse_vectors = bm25.encode_documents(questions)

    store.ensure_collection(recreate=force)
    points = [
        models.PointStruct(
            id=idx,
            vector={
                DENSE_VECTOR_NAME: dense_vectors[idx],
                SPARSE_VECTOR_NAME: sparse_vectors[idx],
            },
            payload={"question": records[idx]["question"], "answer": records[idx]["answer"]},
        )
        for idx in range(len(records))
    ]
    written = store.upsert(points)

    vocab_path = bm25.save()
    logger.info("词表已写入: %s", vocab_path)

    return {
        "source": str(find_source_excel(path)),
        "imported": written,
        "collection": store.collection,
        "vocab_file": str(vocab_path),
    }
