"""RAG 包：混合检索 + 生成。

`from src.rag import RAGPipeline, rag_pipeline` 等公共导入路径为对外契约
（测试与 eval 依赖），勿破坏（见 docs/开发文档.md §13.3）。
"""

from src.rag.classify_question import QuestionCategory, classify, fuse_rrf, get_params
from src.rag.embedder import BM25Encoder, Embedder, Reranker, embedder, reranker
from src.rag.ingest import clean_qa, ingest_data, load_qa
from src.rag.pipeline import RAGPipeline, cache_info, clear_search_cache, rag_pipeline
from src.rag.regulation_store import (
    DENSE_VECTOR_NAME,
    POINT_NAMESPACE,
    REGULATION_COLLECTION,
    SPARSE_VECTOR_NAME,
    RegulationArticle,
    RegulationError,
    all_articles,
    ensure_regulation_collection,
    point_id,
    regulation_stats,
    search_regulations,
    upsert_regulations,
    verify_regulations,
)
from src.rag.vector_store import VectorStore

__all__ = [
    "BM25Encoder",
    "DENSE_VECTOR_NAME",
    "Embedder",
    "POINT_NAMESPACE",
    "QuestionCategory",
    "RAGPipeline",
    "REGULATION_COLLECTION",
    "Reranker",
    "RegulationArticle",
    "RegulationError",
    "SPARSE_VECTOR_NAME",
    "VectorStore",
    "all_articles",
    "ensure_regulation_collection",
    "point_id",
    "regulation_stats",
    "search_regulations",
    "upsert_regulations",
    "verify_regulations",
    "cache_info",
    "classify",
    "clean_qa",
    "clear_search_cache",
    "embedder",
    "fuse_rrf",
    "get_params",
    "ingest_data",
    "load_qa",
    "rag_pipeline",
    "reranker",
]
