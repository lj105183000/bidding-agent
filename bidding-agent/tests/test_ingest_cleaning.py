"""知识库导入：列名识别、清洗规则、数据源定位。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.rag.ingest import _pick_columns, clean_qa, find_source_excel


# ---- 列名识别 ----


def test_pick_columns_chinese_headers():
    df = pd.DataFrame({"问": ["q"], "答": ["a"]})
    assert _pick_columns(df) == ("问", "答")


def test_pick_columns_english_headers_case_insensitive():
    df = pd.DataFrame({"Question": ["q"], "Answer": ["a"]})
    assert _pick_columns(df) == ("Question", "Answer")


def test_pick_columns_falls_back_to_first_two_columns():
    """表头无法识别时按位置取前两列，兼容自定义表头的数据源。"""
    df = pd.DataFrame({"栏位一": ["q"], "栏位二": ["a"], "备注": ["x"]})
    assert _pick_columns(df) == ("栏位一", "栏位二")


def test_pick_columns_rejects_single_column_frame():
    with pytest.raises(ValueError, match="至少需要两列"):
        _pick_columns(pd.DataFrame({"问": ["q"]}))


# ---- 清洗 ----


def test_clean_qa_drops_blank_rows():
    rows = [
        {"question": "有效问题", "answer": "有效答案"},
        {"question": "", "answer": "有答无问"},
        {"question": "有问无答", "answer": "   "},
        {"question": None, "answer": None},
    ]
    assert clean_qa(rows) == [{"question": "有效问题", "answer": "有效答案"}]


def test_clean_qa_dedupes_by_question_keeping_first():
    rows = [
        {"question": "同一问题", "answer": "第一个答案"},
        {"question": "同一问题", "answer": "第二个答案"},
    ]
    result = clean_qa(rows)
    assert len(result) == 1
    assert result[0]["answer"] == "第一个答案"


def test_clean_qa_treats_whitespace_only_difference_as_duplicate():
    """真实数据里同一问题常因首尾空格被当成两条，清洗后必须收敛为一条。"""
    rows = [
        {"question": "单一来源异议处理", "answer": "A"},
        {"question": "  单一来源异议处理  ", "answer": "A"},
    ]
    assert len(clean_qa(rows)) == 1


def test_clean_qa_preserves_order():
    rows = [
        {"question": "第一", "answer": "1"},
        {"question": "第二", "answer": "2"},
        {"question": "第三", "answer": "3"},
    ]
    assert [r["question"] for r in clean_qa(rows)] == ["第一", "第二", "第三"]


# ---- 数据源定位 ----


def test_find_source_excel_explicit_path(tmp_path):
    target = tmp_path / "data.xlsx"
    target.write_bytes(b"stub")
    assert find_source_excel(target) == target


def test_find_source_excel_missing_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="不存在"):
        find_source_excel(tmp_path / "nope.xlsx")
