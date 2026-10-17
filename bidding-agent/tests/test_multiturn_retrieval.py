"""多轮对话的检索式增强单测。

背景（实测，见 docs/25）：
    真实使用里用户的第二句几乎总是省略式的——

        用户：投标保证金的退还时限是多久？
        用户：那履约保证金呢？

    而 `search_cached` 原先直接用 `question` 检索、不做指代消解。实测：

        只用追问当检索式      平均最高相似度 0.627（5 题 1 题检索不到）
        拼接上一轮用户问题     平均最高相似度 0.868
        人工改写成完整问题     平均最高相似度 0.908（上界）

    拼接式拿到上界 96% 的收益，且**零额外 LLM 调用**。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline  # noqa: E402


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str) -> dict:
    return {"role": "assistant", "content": text}


class TestRetrievalQuery:
    def test_no_history_returns_question(self):
        assert RAGPipeline._retrieval_query("保证金？", None) == "保证金？"
        assert RAGPipeline._retrieval_query("保证金？", []) == "保证金？"

    def test_prepends_previous_user_question(self):
        """★ 核心行为：拼上一轮用户问题，让"那 X 呢"有指代对象。"""
        history = [_u("投标保证金的退还时限是多久？"), _a("五日。")]
        out = RAGPipeline._retrieval_query("那履约保证金呢？", history)
        assert out == "投标保证金的退还时限是多久？ 那履约保证金呢？"

    def test_only_uses_last_user_turn_not_earlier(self):
        """★ 只拼**最近一轮**。

        更早的轮次与当前问题的指代关系很弱，拼进来会引入无关词。
        这里构造三轮，验证取的是最后那轮 user。
        """
        history = [
            _u("完全不相关的第一轮问题"),
            _a("答一"),
            _u("投标保证金的退还时限是多久？"),
            _a("答二"),
        ]
        out = RAGPipeline._retrieval_query("那履约保证金呢？", history)
        assert "完全不相关的第一轮问题" not in out
        assert "投标保证金的退还时限是多久？" in out

    def test_ignores_assistant_answers(self):
        """★ 助手回答**不拼**——它通常很长（实测平均 480 字），
        拼进来会把检索式稀释成"什么都有点像"。"""
        history = [_u("保证金？"), _a("这是一段非常长的助手回答" * 20)]
        out = RAGPipeline._retrieval_query("那履约呢？", history)
        assert "非常长的助手回答" not in out

    def test_skips_trailing_assistant_to_find_user(self):
        """历史末尾是助手回答（正常时序）时，仍要向前找到最近的 user。"""
        history = [_u("上轮问题"), _a("上轮回答")]
        assert RAGPipeline._retrieval_query("追问", history).startswith("上轮问题")

    def test_does_not_duplicate_identical_question(self):
        """重复提问（用户重发同一句）不该拼成"X X"。"""
        history = [_u("保证金？")]
        assert RAGPipeline._retrieval_query("保证金？", history) == "保证金？"

    def test_strips_whitespace_before_comparing(self):
        history = [_u("  保证金？  ")]
        assert RAGPipeline._retrieval_query("保证金？", history) == "保证金？"

    def test_skips_blank_previous_content(self):
        """上一轮 user 内容为空白时跳过，不要拼出" 追问"这种检索式。"""
        assert RAGPipeline._retrieval_query("追问", [_u("   ")]) == "追问"

    def test_no_user_message_at_all(self):
        assert RAGPipeline._retrieval_query("追问", [_a("只有助手的回答")]) == "追问"

    def test_tolerates_malformed_history_entries(self):
        """历史里混入非 dict（前端传参异常）不能让检索崩掉。"""
        weird = ["裸字符串", None, 42, {"role": "user", "content": "有效问题"}]
        out = RAGPipeline._retrieval_query("追问", weird)  # type: ignore[arg-type]
        assert out == "有效问题 追问"

    def test_ignores_unknown_roles(self):
        history = [{"role": "system", "content": "系统消息"}, {"role": "user", "content": "用户问题"}]
        out = RAGPipeline._retrieval_query("追问", history)
        assert "系统消息" not in out
        assert out == "用户问题 追问"

    def test_missing_content_key(self):
        assert RAGPipeline._retrieval_query("追问", [{"role": "user"}]) == "追问"


class TestChatEventsUsesEnrichedQuery:
    def test_chat_events_passes_enriched_query_to_search(self, monkeypatch):
        """★ 端到端验证：`chat_events` 必须把**增强后的检索式**传给检索。

        只测 `_retrieval_query` 不够——若 `chat_events` 忘了用它，
        增强逻辑就是死代码，单测照样全绿。本测试钉住这处接线。
        """
        from src.rag import pipeline as mod

        captured: list[str] = []

        class _FakePipeline(RAGPipeline):
            def search_cached(self, question, top_k=5):
                captured.append(question)
                return []  # 空结果 → 走"知识库中没有找到相关内容"分支，不再调 LLM

        pipe = _FakePipeline()
        history = [{"role": "user", "content": "投标保证金的退还时限是多久？"}]
        list(pipe.chat_events("那履约保证金呢？", history=history, paced=False))

        assert captured, "检索未被调用"
        assert "投标保证金的退还时限是多久？" in captured[0]
        assert "那履约保证金呢？" in captured[0]

    def test_generation_still_uses_original_question(self, monkeypatch):
        """★ 生成阶段必须用**原始问题**，不能用增强后的检索式。

        否则模型会把上一轮的问题也当成待答内容，答出多余的东西。
        """
        captured_messages: list[list[dict]] = []

        class _FakeLLM:
            def available(self):
                return True

            def chat_stream(self, messages, **kwargs):
                captured_messages.append(messages)
                yield "回答"

        class _FakePipeline(RAGPipeline):
            def search_cached(self, question, top_k=5):
                return [{"question": "q", "answer": "a", "score": 0.5, "relevance": 0.9}]

        pipe = _FakePipeline(llm=_FakeLLM())
        history = [{"role": "user", "content": "上一轮的问题内容"}]
        list(pipe.chat_events("本轮问题内容", history=history, paced=False))

        assert captured_messages, "未调用 LLM"
        joined = "\n".join(m.get("content", "") for m in captured_messages[0])
        # 末条 user 消息里应是本轮问题，且**不含**被拼进来的上一轮问题
        last_user = [m for m in captured_messages[0] if m.get("role") == "user"][-1]
        assert "本轮问题内容" in last_user["content"]
        assert not last_user["content"].startswith("上一轮的问题内容")
