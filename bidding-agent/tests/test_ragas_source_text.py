"""ragas_eval 取正文的回归测试。

守住一个已修复的严重缺陷：`eval_one` 曾用 `s.get("content", "")` 取 RAG source
正文，而 RAG source 的正文在 `answer` 字段——于是每条都取到空串，
`retrieved_contexts` 变成 `["", "", ...]`。

这个 bug 的危害在于**静默给出错误结论**：
    Faithfulness / LLMContextRecall 恒为 0.000，
    而 AnswerRelevancy（不看 context）仍显示 0.847 ——
指标表看起来"跑通了"，却把有充分依据的回答判成"完全不忠实于上下文"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.ragas_eval import _source_text  # noqa: E402


def test_takes_answer_field():
    """★ 核心回归：RAG source 的正文在 `answer`，必须取到。"""
    assert _source_text({"question": "标题", "answer": "正文内容", "score": 0.7}) == "正文内容"


def test_full_rag_source_contract():
    """用生产里真实的字段集（vector_store._payload_to_record）验证。"""
    source = {
        "question": "投标保证金的退还时限是多久？",
        "answer": "招标人最迟应当在书面合同签订后5日内退还投标保证金。",
        "score": 0.7012,
        "relevance": 0.6258,
    }
    text = _source_text(source)
    assert text
    assert "5日内" in text


def test_falls_back_to_content_for_web_sources():
    """联网/API 那种 `content` 形态也要兼容（同一函数服务两类调用方）。"""
    assert _source_text({"title": "t", "content": "联网正文"}) == "联网正文"


def test_skips_blank_answer_and_uses_content():
    """`answer` 存在但为空白时应继续回落，而不是返回空串。"""
    assert _source_text({"answer": "   ", "content": "备选正文"}) == "备选正文"


def test_never_returns_empty_for_dict_without_known_keys():
    """未知字段时兜底序列化整条记录——

    喂空串给裁判等于告诉它"没有检索到任何依据"，会把指标打成 0，
    比返回一段丑 JSON 有害得多。
    """
    out = _source_text({"foo": "bar", "baz": 1})
    assert out.strip()
    assert "bar" in out
    json.loads(out)  # 兜底产物必须是合法 JSON


def test_handles_non_dict_source():
    assert _source_text("纯字符串来源") == "纯字符串来源"
    assert _source_text(12345) == "12345"


def test_prefers_answer_over_content():
    """两者都在时以 `answer` 为准——那是 RAG 契约里的正文字段。"""
    assert _source_text({"answer": "A", "content": "C"}) == "A"


@pytest.mark.parametrize("key", ["text", "page_content"])
def test_extra_aliases_supported(key):
    """langchain Document 风格字段也认（防止换数据源时再次静默取空）。"""
    assert _source_text({key: "别名正文"}) == "别名正文"
