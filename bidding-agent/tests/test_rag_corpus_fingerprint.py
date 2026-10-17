"""缺陷 3 的回归测试：缓存的语料指纹必须能检测「只改内容」的变更。

★ 要锁住的行为
==========
`search_cached` 的缓存键含语料指纹。原来的指纹是 `(点数, (词表长度, n_docs))`
—— 而 QA 语料的 point id 是**顺序整数**且走 `upsert`，于是：

    · 改某条问答的答案（条数不变）
    · 两条问答换序（条数不变）

**两个分量都不变** → 缓存键不变 → **继续返回修改前的旧答案**。

修法是给指纹加第三个分量：词表文件（`data/vocab.json`）的 mtime。
`ingest` 每次都会重写它（`bm25.save()`），故 mtime 等价于
「语料被重新导入过」，成本是一次 `stat`。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import pipeline as P  # noqa: E402


class TestCorpusFingerprintShape:
    """指纹的结构契约 —— 第三分量是「内容变更信号」，不能悄悄去掉。"""

    def test_fingerprint_has_three_components(self, monkeypatch, tmp_path):
        """形状必须是 `(点数, (词表长度, n_docs), vocab_mtime)`。

        这条测试的意义：**第三分量容易被后人当成冗余删掉** ——
        而它正是唯一能检测「只改内容」的信号。
        """
        vocab = tmp_path / "vocab.json"
        vocab.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(P, "VOCAB_FILE", vocab)

        pipe = object.__new__(P.RAGPipeline)
        pipe._corpus_count = 42
        pipe._corpus_count_probed = True

        # 用一个只提供 bm25 的最小替身
        class _BM:
            vocab = {"a": 1, "b": 2}
            n_docs = 7

        class _Store:
            def count(self):
                return 42

        pipe._store = _Store()
        pipe._bm25 = _BM()

        fp = pipe._corpus_fingerprint()
        assert isinstance(fp, tuple) and len(fp) == 3, f"指纹应为三元组，实为 {fp!r}"
        assert fp[0] == 42
        assert fp[1] == (2, 7)
        assert fp[2] == vocab.stat().st_mtime_ns, "第三分量必须是词表 mtime"

    def test_vocab_mtime_change_alters_fingerprint(self, monkeypatch, tmp_path):
        """★ 核心断言：**只 touch 词表文件**（内容/点数都不变）也必须换指纹。

        这模拟「重新 ingest 了但条数没变」—— 正是原来检测不到的场景。
        """
        vocab = tmp_path / "vocab.json"
        vocab.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(P, "VOCAB_FILE", vocab)

        pipe = object.__new__(P.RAGPipeline)
        pipe._corpus_count = 42

        class _BM:
            vocab = {"a": 1}
            n_docs = 1

        class _Store:
            def count(self):
                return 42

        pipe._store = _Store()
        pipe._bm25 = _BM()

        before = pipe._corpus_fingerprint()
        # 只改 mtime，不改内容、不改点数
        st = vocab.stat()
        import os

        os.utime(vocab, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
        after = pipe._corpus_fingerprint()

        assert before != after, (
            "词表 mtime 变了但指纹没变 —— 「只改内容」的语料变更会被漏掉，"
            "缓存将继续返回旧答案"
        )
        # 前两个分量应保持不变（证明变化只来自第三分量）
        assert before[:2] == after[:2]

    def test_missing_vocab_file_degrades_to_none_not_crash(self, monkeypatch, tmp_path):
        """词表文件不存在时第三分量退化为 None —— 不能抛异常。

        没有 `data/vocab.json`（从未 ingest）时缓存仍须能工作。
        """
        missing = tmp_path / "nope.json"
        monkeypatch.setattr(P, "VOCAB_FILE", missing)

        pipe = object.__new__(P.RAGPipeline)
        pipe._corpus_count = 1

        class _BM:
            vocab = {"a": 1}
            n_docs = 1

        class _Store:
            def count(self):
                return 1

        pipe._store = _Store()
        pipe._bm25 = _BM()

        fp = pipe._corpus_fingerprint()
        assert fp is not None
        assert fp[2] is None, "取不到 mtime 时该分量为 None，而不是抛异常"
