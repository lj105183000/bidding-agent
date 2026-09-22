"""BM25 稀疏编码：分词过滤、权重公式、词表持久化。

权重公式部分直接调用 `_weights` 传入**受控的词序列**，绕开 jieba 分词的不确定性，
使断言能与 BM25 公式逐项对应。
"""

from __future__ import annotations

import math

import pytest
from qdrant_client import models

from src.rag.embedder import BM25Encoder, _tokenize


def _fitted() -> BM25Encoder:
    docs = [
        "单一来源采购方式公示",
        "单一来源异议处理流程",
        "竞争性磋商采购程序",
        "公开招标投标保证金",
    ]
    return BM25Encoder().fit(docs)


# ---- 分词 ----


def test_tokenize_drops_single_chars_punctuation_and_digits():
    tokens = _tokenize("单一来源采购方式，2024年怎么办？")
    assert tokens, "正常中文句子不应被过滤成空"
    assert all(len(t) >= 2 for t in tokens), "单字与标点必须被丢弃"
    assert "，" not in tokens and "？" not in tokens
    assert "2024" not in tokens, "纯数字对语义检索无贡献"


def test_tokenize_handles_empty_input():
    assert _tokenize("") == []


# ---- 拟合 ----


def test_fit_builds_sorted_vocab_and_positive_idf():
    encoder = _fitted()
    assert encoder.n_docs == 4
    assert encoder.avgdl > 0
    # 词表按字典序编号 → 同一语料每次 fit 的索引一致
    assert list(encoder.vocab) == sorted(encoder.vocab)
    # ln(1+...) 形式的 idf 恒为正，避免高频词产生负权重
    assert all(value > 0 for value in encoder.idf)


def test_rarer_term_gets_higher_idf():
    """`单一`（2 篇出现）的 idf 必须低于 `保证金`（仅 1 篇）。
    注意 jieba 会把"单一来源"切成"单一"+"来源"，词表里没有"单一来源"整词。"""
    encoder = _fitted()
    common = encoder.vocab["单一"]
    rare = encoder.vocab["保证金"]
    assert encoder.idf[rare] > encoder.idf[common]


# ---- 权重公式 ----


def test_weight_of_single_occurrence_equals_idf_at_average_length():
    """tf=1 且文档长度恰好等于平均长度时，长度归一化项为 1，权重退化为 idf。"""
    encoder = _fitted()
    term = "采购"
    idx = encoder.vocab[term]
    vector = encoder._weights([term], float(encoder.avgdl))
    assert vector.indices == [idx]
    assert vector.values[0] == pytest.approx(encoder.idf[idx])


def test_term_frequency_saturates():
    """词频翻倍不会让权重翻倍——这正是 BM25 相对于朴素词频的价值。"""
    encoder = _fitted()
    term = "采购"
    idx = encoder.vocab[term]
    once = encoder._weights([term], float(encoder.avgdl)).values[0]
    twice = encoder._weights([term, term], float(encoder.avgdl)).values[0]
    assert twice > once
    assert twice < 2 * once  # 饱和而非线性


def test_longer_document_gets_lower_weight():
    """同样的词频，文档越长权重越低（b 参数的归一化作用）。"""
    encoder = _fitted()
    term = "采购"
    short = encoder._weights([term], float(encoder.avgdl)).values[0]
    long = encoder._weights([term], float(encoder.avgdl) * 2).values[0]
    assert long < short


def test_query_weight_has_no_length_normalisation():
    """查询侧不做长度归一化：长查询自带更多信息量，不该被惩罚。"""
    encoder = _fitted()
    term = "采购"
    idx = encoder.vocab[term]
    vector = encoder.encode_query(term)
    assert vector.indices == [idx]
    expected = encoder.idf[idx] * (1 * (encoder.k1 + 1)) / (1 + encoder.k1)
    assert vector.values[0] == pytest.approx(expected)


def test_unknown_terms_are_dropped():
    encoder = _fitted()
    vector = encoder.encode_query("区块链元宇宙")
    assert vector.indices == [] and vector.values == []


def test_encode_documents_returns_one_vector_per_document():
    encoder = _fitted()
    vectors = encoder.encode_documents(["单一来源采购", "公开招标"])
    assert len(vectors) == 2
    assert all(isinstance(v, models.SparseVector) for v in vectors)
    assert all(len(v.indices) == len(v.values) for v in vectors)


# ---- 持久化 ----


def test_save_and_load_roundtrip(tmp_path):
    encoder = _fitted().fit(
        ["单一来源采购", "竞争性磋商程序", "公开招标保证金"]
    )
    path = encoder.save(tmp_path / "vocab.json")

    restored = BM25Encoder.load(path)
    assert restored.vocab == encoder.vocab
    assert restored.idf == pytest.approx(encoder.idf)
    assert restored.avgdl == pytest.approx(encoder.avgdl)
    assert restored.n_docs == encoder.n_docs
    # 加载后编码结果必须与保存前一致，否则历史向量会与查询向量错位
    assert restored.encode_query("采购").indices == encoder.encode_query("采购").indices


def test_load_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        BM25Encoder.load(tmp_path / "not-there.json")


def test_idf_formula_matches_reference():
    """对照 BM25 定义式直接验算，防止后续"优化"改坏公式。"""
    docs = ["采购 采购 采购", "采购 公示", "异议 公示 流程"]
    encoder = BM25Encoder().fit(docs)
    term = "采购"
    df = 2  # 出现在前两篇
    expected = math.log(1.0 + (3 - df + 0.5) / (df + 0.5))
    assert encoder.idf[encoder.vocab[term]] == pytest.approx(expected)
