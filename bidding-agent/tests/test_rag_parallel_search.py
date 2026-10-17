"""双路检索并发（`RAGPipeline._two_path_search`）的回归测试。

★ 这次优化改的是**并发**，风险不在"快不快"，而在三件事：

    ① **结果必须一模一样**（并发只该改延迟）：两路 Future 不能拿反、
       `fuse_rrf` 的输入顺序不能变 —— 否则排序会悄悄变，而指标只是"降了一点"，
       很难归因到"某次并发改造"
    ② **失败语义不变**：任一路抛异常，`search()` 仍要失败（此前行为），
       不能变成"只跑通一路就返回半份结果"
    ③ **开关有效**：`RAG_PARALLEL_SEARCH=0` 要真的退回串行（排查环境差异用）

配套的实测见 `logs/test_rag_parallel_ab.py`（配对 A/B + 符号检验）。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.rag.pipeline as P  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402


class _Recorder:
    """记录每次调用发生在哪个线程、以什么顺序。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []   # (路径, 线程名)

    def note(self, path: str) -> None:
        with self.lock:
            self.calls.append((path, threading.current_thread().name))


class _PathStore:
    """dense / sparse 返回**可区分**的结果，便于验证"没拿反"。"""

    def __init__(self, recorder: _Recorder | None = None,
                 dense_error: Exception | None = None,
                 sparse_error: Exception | None = None) -> None:
        self.recorder = recorder or _Recorder()
        self.dense_error = dense_error
        self.sparse_error = sparse_error

    def dense_search(self, dense_vector, *, limit=30):
        self.recorder.note("dense")
        if self.dense_error:
            raise self.dense_error
        return [{"id": 1, "question": "D1", "answer": "d1", "score": 0.9},
                {"id": 2, "question": "D2", "answer": "d2", "score": 0.8}]

    def sparse_search(self, sparse_vector, *, limit=30):
        self.recorder.note("sparse")
        if self.sparse_error:
            raise self.sparse_error
        return [{"id": 3, "question": "S1", "answer": "s1", "score": 0.7}]

    def health(self) -> dict:
        return {"ok": True, "points_count": 3, "latency_ms": 1}


def _pipeline(store) -> RAGPipeline:
    class _E:
        def encode_query(self, text): return [0.1, 0.2, 0.3, 0.4]

    class _B:
        def encode_query(self, text): return object()

    return RAGPipeline(store=store, embedder=_E(), bm25=_B())


def _seq(merged: list[dict]) -> list[str]:
    return [str(r.get("question")) for r in merged]


class TestResultsInvariant:
    def test_parallel_matches_serial(self, monkeypatch):
        """★ 并发与串行必须给出**完全相同**的融合结果（顺序也一致）。"""
        serial = _pipeline(_PathStore())
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "0")
        a = serial._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)

        par = _pipeline(_PathStore())
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        b = par._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)

        assert _seq(a) == _seq(b), "并发改变了融合顺序 —— 这是不可接受的"
        assert [r["score"] for r in a] == [r["score"] for r in b]

    def test_dense_results_are_not_swapped_with_sparse(self, monkeypatch):
        """两路 Future 不能拿反：dense 的结果必须还是 dense 那一份。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        store = _PathStore()
        dense, sparse = _pipeline(store)._two_path_search([1.0], object(), 30)
        assert [r["question"] for r in dense] == ["D1", "D2"]
        assert [r["question"] for r in sparse] == ["S1"]

    def test_weights_still_applied(self, monkeypatch):
        """并发不改变加权语义：只信 sparse 时 dense 独有项不该排前面。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        merged = _pipeline(_PathStore())._hybrid_retrieve(
            [1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5,
            dense_weight=0.0, sparse_weight=1.0,
        )
        assert _seq(merged)[0] == "S1"


class TestFailureSemantics:
    def test_dense_failure_propagates(self, monkeypatch):
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        pipe = _pipeline(_PathStore(dense_error=RuntimeError("dense 挂了")))
        with pytest.raises(RuntimeError, match="dense 挂了"):
            pipe._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)

    def test_sparse_failure_propagates(self, monkeypatch):
        """单路失败仍然整体失败 —— 不能返回"半份结果"当成功。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        pipe = _pipeline(_PathStore(sparse_error=RuntimeError("sparse 挂了")))
        with pytest.raises(RuntimeError, match="sparse 挂了"):
            pipe._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)

    def test_both_failures_prefers_dense_error(self, monkeypatch):
        """两路都失败时抛 dense 那个（与串行"先 dense 后 sparse"的观察一致）。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        pipe = _pipeline(_PathStore(dense_error=RuntimeError("dense 先"),
                                    sparse_error=RuntimeError("sparse 后")))
        with pytest.raises(RuntimeError, match="dense 先"):
            pipe._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)

    def test_search_failure_is_not_cached(self, monkeypatch):
        """失败不得进缓存（与既有纪律一致：缓存只存成功结果）。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        pipe = _pipeline(_PathStore(dense_error=RuntimeError("挂了")))
        with pytest.raises(RuntimeError):
            pipe.search("投标保证金上限", top_k=3)

    def test_serial_and_parallel_failure_agree(self, monkeypatch):
        """两条路径的失败表现必须一致（否则开关会改变可用性）。"""
        for mode in ("0", "1"):
            monkeypatch.setenv("RAG_PARALLEL_SEARCH", mode)
            pipe = _pipeline(_PathStore(sparse_error=ValueError("x")))
            with pytest.raises(ValueError):
                pipe._hybrid_retrieve([1.0], object(), rrf_k=60, prefetch_limit=30, top_k=5)


class TestSwitch:
    def test_parallel_uses_other_threads(self, monkeypatch):
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "1")
        rec = _Recorder()
        _pipeline(_PathStore(rec))._two_path_search([1.0], object(), 30)
        names = {name for _path, name in rec.calls}
        assert any("rag-search" in n for n in names), f"并发路径没起工作线程：{rec.calls}"

    def test_serial_runs_on_caller_thread(self, monkeypatch):
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "0")
        rec = _Recorder()
        _pipeline(_PathStore(rec))._two_path_search([1.0], object(), 30)
        assert all(n == threading.current_thread().name for _p, n in rec.calls)

    @pytest.mark.parametrize("value,expected", [
        ("", True), ("1", True), ("true", True), ("yes", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("OFF", False), ("disable", False),
    ])
    def test_env_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", value)
        assert P._parallel_search_enabled() is expected

    def test_invalid_value_keeps_default(self, monkeypatch):
        """非法值不抛异常（与 RAG_PROMPT_VARIANT 同口径），按默认（开）处理。"""
        monkeypatch.setenv("RAG_PARALLEL_SEARCH", "随便写的")
        assert P._parallel_search_enabled() is True
