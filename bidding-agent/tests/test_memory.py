"""会话记忆与反馈回流单测。

★ 现实前提：本项目的反馈数据极少（实测 1 条）
------------------------------------------
故测试**不依赖真实样本量**，全部用注入的 fake loader/embedder。
这也正是模块设计的要求：机制要能在 1 条数据上工作，而不是等数据攒够。

测试重点：
    1. **只取被采纳的**（rating='up'）——把答错的历史当示例等于教模型重复错误
    2. **相似度门槛**——半相关的示例会误导模型套用错误结构，宁可不给
    3. **永不抛异常**——记忆是增益项，它坏掉不能让问答坏掉
    4. **注入文本的口径**——必须标明"历史经验、仅供参考、与知识库冲突以知识库为准"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import memory as M  # noqa: E402


class _FakeEmbedder:
    """可控的假嵌入器：用关键词命中模拟相似度，避免依赖真实模型。"""

    def __init__(self, mapping: dict[str, float] | None = None):
        self.mapping = mapping or {}
        self.doc_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts):
        self.doc_calls += 1
        # 每条历史问答编码成一个"命中自己关键词"的 2 维向量
        return [[1.0, 0.0] if "香港" in t else [0.0, 1.0] for t in texts]

    def encode_query(self, text):
        self.query_calls += 1
        return [1.0, 0.0] if "香港" in text else [0.0, 1.0]


def _loader(rows):
    return lambda: rows


class TestSessionMemoryRecall:
    def test_recalls_similar_example(self):
        mem = M.SessionMemory(
            loader=_loader([{"question": "香港招投标法规", "answer": "答案A", "rating": "up"}]),
            embedder=_FakeEmbedder(),
        )
        got = mem.recall("香港法规有什么区别？")
        assert len(got) == 1
        assert got[0].answer == "答案A"
        assert got[0].similarity == pytest.approx(1.0)

    def test_returns_empty_below_threshold(self):
        """★ 相似度不够时**宁可不给**。半相关的示例会把模型带偏。"""
        mem = M.SessionMemory(
            loader=_loader([{"question": "香港招投标法规", "answer": "答案A", "rating": "up"}]),
            embedder=_FakeEmbedder(),
        )
        assert mem.recall("投标保证金退还时限是多久？") == []

    def test_empty_question_returns_empty(self):
        mem = M.SessionMemory(loader=_loader([]), embedder=_FakeEmbedder())
        assert mem.recall("") == []
        assert mem.recall("   ") == []

    def test_respects_top_k(self):
        rows = [
            {"question": f"香港问题{i}", "answer": f"答案{i}", "rating": "up"}
            for i in range(5)
        ]
        mem = M.SessionMemory(loader=_loader(rows), embedder=_FakeEmbedder())
        assert len(mem.recall("香港", top_k=2)) == 2
        assert len(mem.recall("香港", top_k=0)) == 0

    def test_sorted_by_similarity_desc(self):
        rows = [
            {"question": "香港甲", "answer": "A", "rating": "up"},
            {"question": "香港乙", "answer": "B", "rating": "up"},
        ]
        mem = M.SessionMemory(loader=_loader(rows), embedder=_FakeEmbedder())
        got = mem.recall("香港", top_k=2)
        assert [e.similarity for e in got] == sorted(
            [e.similarity for e in got], reverse=True
        )

    def test_skips_rows_missing_question_or_answer(self):
        rows = [
            {"question": "", "answer": "有答案", "rating": "up"},
            {"question": "有问题", "answer": "", "rating": "up"},
            {"question": "香港有效", "answer": "有效答案", "rating": "up"},
            "裸字符串",
            None,
        ]
        mem = M.SessionMemory(loader=_loader(rows), embedder=_FakeEmbedder())
        got = mem.recall("香港")
        assert len(got) == 1
        assert got[0].answer == "有效答案"


class TestSessionMemoryResilience:
    """★ 记忆是**增益项**：任何失败都必须降级为"无记忆"，不能影响问答。"""

    def test_loader_exception_degrades_silently(self):
        def boom():
            raise RuntimeError("数据库挂了")

        mem = M.SessionMemory(loader=boom, embedder=_FakeEmbedder())
        assert mem.recall("香港") == []

    def test_embed_documents_exception_degrades(self):
        class _Boom:
            def encode_documents(self, texts):
                raise RuntimeError("编码失败")

            def encode_query(self, text):
                return [1.0, 0.0]

        mem = M.SessionMemory(
            loader=_loader([{"question": "香港", "answer": "A", "rating": "up"}]),
            embedder=_Boom(),
        )
        assert mem.recall("香港") == []

    def test_encode_query_exception_degrades(self):
        class _Boom:
            def encode_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def encode_query(self, text):
                raise RuntimeError("查询编码失败")

        mem = M.SessionMemory(
            loader=_loader([{"question": "香港", "answer": "A", "rating": "up"}]),
            embedder=_Boom(),
        )
        assert mem.recall("香港") == []

    def test_empty_history_degrades(self):
        """没有已采纳问答时不该报错，也不该每次问答都重查数据库。"""
        calls = {"n": 0}

        def counting_loader():
            calls["n"] += 1
            return []

        mem = M.SessionMemory(loader=counting_loader, embedder=_FakeEmbedder())
        assert mem.recall("香港") == []
        assert mem.recall("香港") == []
        # 失败后不重试：数据库不可用时每请求重试会把延迟叠加到每次问答上
        assert calls["n"] == 1

    def test_loader_called_once_due_to_cache(self):
        calls = {"n": 0}

        def counting_loader():
            calls["n"] += 1
            return [{"question": "香港", "answer": "A", "rating": "up"}]

        mem = M.SessionMemory(loader=counting_loader, embedder=_FakeEmbedder())
        mem.recall("香港")
        mem.recall("香港")
        mem.recall("香港")
        assert calls["n"] == 1, "历史问答的加载与编码必须只做一次"

    def test_invalidate_forces_reload(self):
        """★ 新增反馈后必须能刷新。

        不做失效的后果是**静默的**：用户点了赞，但记忆里还是旧缓存，
        新内容要等重启后端才生效，且没有任何提示。
        """
        calls = {"n": 0}

        def counting_loader():
            calls["n"] += 1
            return [{"question": "香港", "answer": f"A{calls['n']}", "rating": "up"}]

        mem = M.SessionMemory(loader=counting_loader, embedder=_FakeEmbedder())
        first = mem.recall("香港")
        assert first[0].answer == "A1"
        mem.invalidate()
        second = mem.recall("香港")
        assert second[0].answer == "A2"
        assert calls["n"] == 2


class TestBuildMemoryNote:
    def test_marks_source_and_precedence(self):
        """★ 注入文本必须标明来源与优先级。

        会话记忆是**未经整理的经验**，知识库是**权威语料**。
        不标清楚会让历史里的过时内容压过权威语料。
        """
        note = M.build_memory_note([
            M.MemoryExample(question="问", answer="答", rating="up", similarity=0.9)
        ])
        assert "历史问答" in note
        assert "被用户采纳" in note
        assert "仅供参考" in note
        assert "以知识库为准" in note

    def test_says_not_the_answer_to_current_question(self):
        """防止模型把历史答案直接照抄成当前答案。"""
        note = M.build_memory_note([
            M.MemoryExample(question="问", answer="答", rating="up")
        ])
        assert "不要直接照搬" in note

    def test_empty_for_no_examples(self):
        assert M.build_memory_note([]) == ""

    def test_numbers_multiple_examples(self):
        note = M.build_memory_note([
            M.MemoryExample(question="问1", answer="答1", rating="up"),
            M.MemoryExample(question="问2", answer="答2", rating="up"),
        ])
        assert "历史问答1" in note
        assert "历史问答2" in note


class TestExampleSerialization:
    def test_to_dict_shape(self):
        d = M.MemoryExample(question="q", answer="a", rating="up", similarity=0.81234).to_dict()
        assert d["similarity"] == 0.8123
        assert set(d) == {"question", "answer", "rating", "similarity"}

    def test_knowledge_gap_to_dict(self):
        assert set(M.KnowledgeGap("q", "a", "c").to_dict()) == {"question", "answer", "comment"}
