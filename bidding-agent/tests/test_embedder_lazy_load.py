"""嵌入/精排模型懒加载并发安全测试（双检锁）。

验证 `Embedder` 与 `Reranker` 的 `.model` 属性：
1. 构造时不加载（首次访问才加载）；
2. 并发首用只加载一份（双检锁的正确性）；
3. 已加载后无锁快路径。

全部 mock，不加载真实模型（见 docs/开发文档.md §9.3）。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from src.rag.embedder import Embedder, Reranker


# =========================================================================
# 一、懒加载：构造不触发加载
# =========================================================================


class TestLazyLoad:
    def test_embedder_not_loaded_at_construction(self):
        """构造 Embedder 时不应触发模型加载。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            emb = Embedder()
            assert not emb.loaded, "构造后 model 必须为 None"
            mock_st.assert_not_called()

    def test_reranker_not_loaded_at_construction(self):
        """构造 Reranker 时不应触发模型加载。"""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            rr = Reranker()
            # Reranker 没有 loaded 属性，但 _model 应为 None
            assert rr._model is None
            mock_ce.assert_not_called()

    def test_embedder_loaded_on_first_access(self):
        """首次访问 .model 触发加载。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = [[0.1, 0.2]]
            emb = Embedder()
            _ = emb.model  # 触发加载
            assert emb.loaded
            mock_st.assert_called_once()

    def test_embedder_not_loaded_twice(self):
        """第二次访问不重复加载。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = [[0.1, 0.2]]
            emb = Embedder()
            _ = emb.model
            _ = emb.model  # 第二次访问
            mock_st.assert_called_once()


# =========================================================================
# 二、并发首用只加载一份（双检锁）
# =========================================================================


class TestConcurrentLoad:
    """多线程同时首用 .model，构造器应只被调用一次。"""

    def _make_slow_constructor(self, delay: float = 0.05):
        """返回一个有延迟的 mock 构造器，放大并发竞争窗口。

        延迟是刻意的：没有它，GIL 串行化下测试可能碰巧通过，
        但真正的双检锁需要在"构造进行中"有其他线程同时进入慢路径。
        """
        def _slow(*args, **kwargs):
            time.sleep(delay)
            instance = type("FakeModel", (), {"encode": lambda *a, **k: [[0.1]], "to": lambda *a, **k: None})()
            return instance
        return _slow

    def test_embedder_concurrent_first_access_loads_once(self):
        """Embedder：8 线程并发首用，SentenceTransformer 只构造一次。"""
        with patch("sentence_transformers.SentenceTransformer",
                   side_effect=self._make_slow_constructor()) as mock_st:
            emb = Embedder()
            barrier = threading.Barrier(8)

            def access():
                barrier.wait()  # 所有线程同时起跑，最大化竞争
                _ = emb.model

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(access) for _ in range(8)]
                for f in futures:
                    f.result()

            assert mock_st.call_count == 1, (
                f"双检锁失效：8 线程并发加载触发了 {mock_st.call_count} 次构造"
            )

    def test_reranker_concurrent_first_access_loads_once(self):
        """Reranker：8 线程并发首用，CrossEncoder 只构造一次。"""
        with patch("sentence_transformers.CrossEncoder",
                   side_effect=self._make_slow_constructor()) as mock_ce:
            rr = Reranker()
            barrier = threading.Barrier(8)

            def access():
                barrier.wait()
                _ = rr.model

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(access) for _ in range(8)]
                for f in futures:
                    f.result()

            assert mock_ce.call_count == 1, (
                f"双检锁失效：8 线程并发加载触发了 {mock_ce.call_count} 次构造"
            )

    def test_embedder_model_instance_is_shared(self):
        """并发加载后，所有线程拿到同一个模型实例。"""
        with patch("sentence_transformers.SentenceTransformer",
                   side_effect=self._make_slow_constructor()):
            emb = Embedder()
            ids = []

            def access():
                ids.append(id(emb.model))

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(access) for _ in range(8)]
                for f in futures:
                    f.result()

            assert len(set(ids)) == 1, "所有线程应拿到同一实例"


# =========================================================================
# 三、已加载后的快路径
# =========================================================================


class TestFastPath:
    def test_loaded_flag_true_after_load(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = [[0.1]]
            emb = Embedder()
            assert emb.loaded is False
            _ = emb.model
            assert emb.loaded is True

    def test_second_access_returns_same_instance(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = [[0.1]]
            emb = Embedder()
            first = emb.model
            second = emb.model
            assert first is second, "第二次访问应返回同一实例"


# =========================================================================
# 四、OOM 降级
# =========================================================================


class TestOOMFallback:
    def test_embedder_falls_back_to_cpu_on_oom(self):
        """GPU 显存不足时降级 CPU 重试，不让整个检索链路失败。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            instance = mock_st.return_value
            # 第一次抛 OOM，第二次正常返回
            instance.encode.side_effect = [
                RuntimeError("CUDA out of memory"),
                [[0.1, 0.2, 0.3]],
            ]
            instance.to.return_value = None
            emb = Embedder()
            _ = emb.model  # 先触发加载（加载时会按 torch.cuda.is_available 设 device）
            emb._device = "cuda"  # 加载后强制设为 cuda，模拟 GPU 场景
            result = emb.encode_query("test")
            assert result == [0.1, 0.2, 0.3], "OOM 后应降级 CPU 重试成功"
            assert emb._device == "cpu", "降级后设备应标记为 cpu"
            instance.to.assert_called_once_with("cpu")

    def test_embedder_non_oom_runtime_error_reraises(self):
        """非 OOM 的 RuntimeError 不降级，直接抛出。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.side_effect = RuntimeError("some other error")
            emb = Embedder()
            _ = emb.model  # 触发加载
            with pytest.raises(RuntimeError, match="some other error"):
                emb.encode_query("test")
