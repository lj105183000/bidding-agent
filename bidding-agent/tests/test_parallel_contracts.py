"""并行开发契约测试：保护跨 lane 依赖的公共面不被意外破坏。

这些断言对应 `bidding-agent/docs/并行开发协作.md` §4.1 的冻结清单。
只做 required-subset 断言（允许加法、不允许删除/改名），
这样新增导出不需要改测试，删除或改名会立刻变红。
"""

import inspect
import time

import src.agent as agent_pkg
import src.rag as rag_pkg
from src.agent import BiddingAgent
from src.agent.constants import BASE_TOOL_NAMES, WEB_TOOL_NAMES
from src.rag import RAGPipeline
from src.tools.base import ToolResult
from src.tools.rag_tools import TOOL_EXECUTORS

# --- 公共导出面 ---

REQUIRED_AGENT_EXPORTS = {
    "AgentError",
    "BiddingAgent",
    "LLMUnavailableError",
    "MAX_HISTORY_ROUNDS",
    "MAX_TOKENS_DEEP",
    "MAX_TOOL_ROUNDS",
    "STREAM_CHUNK_CHARS",
    "STREAM_CHUNK_DELAY",
    "STREAM_FLUSH_CHARS",
    "SYSTEM_PROMPT",
    "TOOL_TIMEOUT_SECONDS",
}

REQUIRED_RAG_EXPORTS = {
    "BM25Encoder",
    "Embedder",
    "RAGPipeline",
    "Reranker",
    "VectorStore",
    "RegulationArticle",
    "RegulationError",
    "all_articles",
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
    "search_regulations",
    "upsert_regulations",
    "verify_regulations",
}


def test_agent_public_exports():
    """`from src.agent import ...` 是测试与 eval 依赖的公共路径。"""
    assert REQUIRED_AGENT_EXPORTS <= set(agent_pkg.__all__)
    for name in REQUIRED_AGENT_EXPORTS:
        assert hasattr(agent_pkg, name), f"src.agent 缺少导出: {name}"


def test_rag_public_exports():
    """`from src.rag import ...` 是 tools / eval / 测试依赖的公共路径。"""
    assert REQUIRED_RAG_EXPORTS <= set(rag_pkg.__all__)
    for name in REQUIRED_RAG_EXPORTS:
        assert hasattr(rag_pkg, name), f"src.rag 缺少导出: {name}"


# --- 工具注册表与执行器 ---


def test_tool_registry_matches_name_constants():
    """TOOL_EXECUTORS 的键必须与 BASE_TOOL_NAMES ∪ WEB_TOOL_NAMES 一致。"""
    expected = set(BASE_TOOL_NAMES) | set(WEB_TOOL_NAMES)
    assert set(TOOL_EXECUTORS) == expected
    assert all(callable(executor) for executor in TOOL_EXECUTORS.values())


def test_tool_executor_bindings_are_identical():
    """core.py 与 react_loop.py 各有一份模块级绑定，必须指向同一批工具。"""
    import src.agent.core as core_mod
    import src.agent.react_loop as react_mod

    assert set(core_mod.TOOL_EXECUTORS) == set(TOOL_EXECUTORS)
    assert set(react_mod.TOOL_EXECUTORS) == set(TOOL_EXECUTORS)


def test_tool_result_contract_shape():
    """ToolResult.to_dict() 恒为 success + data/results + error 三键。"""
    ok = ToolResult(success=True, data={"text": "x"})
    assert ok.to_dict() == {"success": True, "data": {"text": "x"}, "error": None}

    web = ToolResult(success=True, data=[{"title": "x"}], payload_key="results")
    assert web.to_dict() == {
        "success": True,
        "results": [{"title": "x"}],
        "error": None,
    }

    failed = ToolResult.failure("boom")
    assert failed.to_dict() == {"success": False, "data": None, "error": "boom"}


# --- SSE done 帧 ---


def test_done_frame_red_line_contract():
    """done 帧的红线字段与 tool_name 单字符串语义。"""
    meta = {
        "sources": [],
        "web_sources": [],
        "tool_called": False,
        "tool_name": "search_knowledge_base",
        "phase_times": [("首轮分析", 1, 0), ("生成回答", 2, 1)],
    }
    done = BiddingAgent._build_done(meta, time.monotonic())

    assert {"sources", "web_sources", "tool_name", "elapsed_ms", "phase_times"} <= set(done)
    assert isinstance(done["tool_name"], str)
    assert done["tool_name"] == "search_knowledge_base"
    # 多 Agent 可观测字段是 additive；旧路径为 None/[]。
    assert "orchestration" in done
    assert "agent_trace" in done
    assert "agent_timings" in done
    # _meta 是内部传递键，chat_stream 序列化时会过滤；这里只确认它没丢。
    assert "_meta" in done


# --- RAG search 契约 ---


class _FakeEmbedder:
    def encode_query(self, question):
        return [0.1, 0.2, 0.3]


class _FakeBM25:
    def encode_query(self, question):
        return {"indices": [0], "values": [1.0]}


class _FakeStore:
    def dense_search(self, dense_vector, *, limit=30):
        return [
            {"id": "1", "question": "q1", "answer": "a1", "score": 0.9, "relevance": 0.9},
            {"id": "2", "question": "q2", "answer": "a2", "score": 0.8, "relevance": 0.8},
        ][:limit]

    def sparse_search(self, sparse_vector, *, limit=30):
        return [
            {"id": "2", "question": "q2", "answer": "a2", "score": 0.7, "relevance": 0.7},
        ][:limit]


class _FakeReranker:
    def rerank(self, question, candidates, top_k):
        return candidates[:top_k]


def test_rag_search_signature_and_result_contract():
    """RAGPipeline.search 的签名与结果键是 agent/tools/eval 共同消费的契约。"""
    sig = inspect.signature(RAGPipeline.search)
    assert list(sig.parameters) == ["self", "question", "top_k", "min_relevance"]
    assert sig.parameters["min_relevance"].kind is inspect.Parameter.KEYWORD_ONLY

    pipeline = RAGPipeline(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        reranker=_FakeReranker(),
        bm25=_FakeBM25(),
    )
    records = pipeline.search("测试问题", top_k=2, min_relevance=0.0)

    assert records, "search 应返回至少一条记录"
    for record in records:
        assert {"question", "answer", "score", "relevance"} <= set(record)


# --- 迁移 shim ---


def test_regulation_store_shim_forwards_public_names():
    """services.regulation_store 是 rag.regulation_store 的兼容别名。"""
    import src.rag.regulation_store as new_store
    import src.services.regulation_store as old_store

    for name in (
        "RegulationArticle",
        "RegulationError",
        "all_articles",
        "search_regulations",
        "upsert_regulations",
        "verify_regulations",
    ):
        assert getattr(old_store, name) is getattr(new_store, name), (
            f"兼容 shim 未正确转发: {name}"
        )
