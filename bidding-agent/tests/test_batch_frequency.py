"""batch/统计数据_插交易频次.py 的频次统计逻辑（docs/开发文档.md §9.2 点名的 batch 脚本测试）。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符而失败。

管道最后一步，产出直接拿去比对/入库：并列项的排序必须确定，空标的物不能混进榜单，
否则同一份输入两次跑出两个文件，没人说得清哪个是对的。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "统计数据_插交易频次.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_freq", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


freq_mod = _load_module()


def _frame(values: list) -> pd.DataFrame:
    return pd.DataFrame({"项目名称": [f"项目{i}" for i in range(len(values))], "标的物": values})


def _pairs(table: pd.DataFrame) -> list[tuple[str, int]]:
    return list(zip(table[freq_mod.SUBJECT_COLUMN], table[freq_mod.COUNT_COLUMN]))


# ---- 统计 ----


def test_counts_occurrences():
    table, stats = freq_mod.count_subject_matter(_frame(["空调", "空调", "胶圈"]))
    assert _pairs(table) == [("空调", 2), ("胶圈", 1)]
    assert stats["unique"] == 2
    assert stats["counted"] == 3


def test_sorted_by_count_descending():
    table, _ = freq_mod.count_subject_matter(_frame(["甲", "乙", "乙", "丙", "丙", "丙"]))
    assert _pairs(table) == [("丙", 3), ("乙", 2), ("甲", 1)]


def test_ties_are_broken_deterministically():
    """并列项顺序不确定 = 同一份输入两次跑出两个文件，汇总表就没法比对。

    并列时按标的物的**码点**升序（丁 U+4E01 < 丙 U+4E19 < 乙 U+4E59 < 甲 U+7532），
    不是拼音序——这里要的只是"确定"，不是"符合中文语感"。
    """
    table, _ = freq_mod.count_subject_matter(_frame(["乙", "甲", "丁", "丙"]))
    assert _pairs(table) == [("丁", 1), ("丙", 1), ("乙", 1), ("甲", 1)]


def test_same_input_twice_gives_identical_output():
    values = ["乙", "甲", "丙", "甲", "乙", "丁"]
    first, _ = freq_mod.count_subject_matter(_frame(values))
    second, _ = freq_mod.count_subject_matter(_frame(list(reversed(values))))
    pd.testing.assert_frame_equal(first, second)


def test_whitespace_is_stripped_before_counting():
    """单元格里混进空格是 Excel 的日常，不该把同一个标的物拆成两个。"""
    table, _ = freq_mod.count_subject_matter(_frame([" 空调", "空调 ", "胶圈"]))
    assert _pairs(table) == [("空调", 2), ("胶圈", 1)]


def test_empty_values_are_not_counted():
    """未抽出标的物的行不是一笔可统计的交易，当成空标的物计数会污染榜首。"""
    table, stats = freq_mod.count_subject_matter(_frame(["空调", None, float("nan"), "  ", "空调"]))
    assert _pairs(table) == [("空调", 2)]
    assert stats["skipped_empty"] == 3
    assert stats["counted"] == 2


def test_non_string_values_are_counted_as_text():
    table, _ = freq_mod.count_subject_matter(_frame([2023, 2023, "空调"]))
    assert ("2023", 2) in _pairs(table)


def test_output_has_exactly_two_columns_in_order():
    table, _ = freq_mod.count_subject_matter(_frame(["空调", "胶圈"]))
    assert list(table.columns) == [freq_mod.SUBJECT_COLUMN, freq_mod.COUNT_COLUMN]


def test_count_column_is_integer():
    table, _ = freq_mod.count_subject_matter(_frame(["空调", "空调"]))
    assert table[freq_mod.COUNT_COLUMN].tolist() == [2]
    assert str(table[freq_mod.COUNT_COLUMN].dtype).startswith("int")


def test_does_not_mutate_input_frame():
    df = _frame(["空调", "胶圈"])
    before = df.copy()
    freq_mod.count_subject_matter(df)
    pd.testing.assert_frame_equal(df, before)


# ---- 边界 ----


def test_empty_frame_yields_empty_table():
    table, stats = freq_mod.count_subject_matter(pd.DataFrame())
    assert table.empty
    assert list(table.columns) == [freq_mod.SUBJECT_COLUMN, freq_mod.COUNT_COLUMN]
    assert stats["rows"] == 0


def test_all_empty_values_yield_empty_table():
    table, stats = freq_mod.count_subject_matter(_frame([None, "  "]))
    assert table.empty
    assert stats["skipped_empty"] == 2


def test_missing_column_raises_actionable_error():
    df = pd.DataFrame({"项目名称": ["x"]})
    with pytest.raises(ValueError, match="标的物"):
        freq_mod.count_subject_matter(df)


# ---- 输入探测 ----


def test_resolve_default_src_prefers_known_candidates(tmp_path, monkeypatch):
    known = tmp_path / "known_标的物.xlsx"
    known.write_bytes(b"x")
    monkeypatch.setattr(freq_mod, "DEFAULT_SRC_CANDIDATES", (known,))
    monkeypatch.setattr(freq_mod, "DATA_DIR", tmp_path)
    assert freq_mod.resolve_default_src() == known


def test_resolve_default_src_falls_back_to_newest_glob(tmp_path, monkeypatch):
    """第 4 步的产物名派生自输入表名，而输入表被人工改过几轮名字——认不下就取最新的。"""
    old = tmp_path / "old_标的物.xlsx"
    new = tmp_path / "new_标的物.xlsx"
    other = tmp_path / "无关.xlsx"
    for path in (old, new, other):
        path.write_bytes(b"x")
    os.utime(old, (1_600_000_000, 1_600_000_000))
    os.utime(new, (1_700_000_000, 1_700_000_000))
    monkeypatch.setattr(freq_mod, "DEFAULT_SRC_CANDIDATES", (tmp_path / "缺失.xlsx",))
    monkeypatch.setattr(freq_mod, "DATA_DIR", tmp_path)
    assert freq_mod.resolve_default_src() == new


def test_resolve_default_src_returns_none_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setattr(freq_mod, "DEFAULT_SRC_CANDIDATES", (tmp_path / "缺失.xlsx",))
    monkeypatch.setattr(freq_mod, "DATA_DIR", tmp_path)
    assert freq_mod.resolve_default_src() is None


def test_write_excel_reports_locked_file(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(pd.DataFrame, "to_excel", _raise)
    with pytest.raises(PermissionError, match="关闭"):
        freq_mod.write_excel(_frame(["空调"]), tmp_path / "out.xlsx")


# ---- 端到端 ----


def _write_src(tmp_path, values) -> Path:
    src = tmp_path / "训练数据_标的物.xlsx"
    _frame(values).to_excel(src, index=False)
    return src


def test_main_writes_two_column_table(tmp_path, capsys):
    src = _write_src(tmp_path, ["空调", "空调", "胶圈", None])
    dest = tmp_path / "out" / "标的物_交易频次.xlsx"

    assert freq_mod.main(["--src", str(src), "-o", str(dest)]) == 0
    written = pd.read_excel(dest)
    assert list(written.columns) == [freq_mod.SUBJECT_COLUMN, freq_mod.COUNT_COLUMN]
    assert _pairs(written) == [("空调", 2), ("胶圈", 1)]
    assert "唯一数 : 2" in capsys.readouterr().out


def test_main_dry_run_does_not_write(tmp_path, capsys):
    src = _write_src(tmp_path, ["空调"])
    dest = tmp_path / "out.xlsx"

    assert freq_mod.main(["--src", str(src), "-o", str(dest), "--dry-run"]) == 0
    assert not dest.exists()
    assert "未写出文件" in capsys.readouterr().out


def test_main_missing_source_returns_error(tmp_path, capsys):
    assert freq_mod.main(["--src", str(tmp_path / "nope.xlsx")]) == 1
    assert "不存在" in capsys.readouterr().out


def test_main_reports_missing_column_without_traceback(tmp_path, capsys):
    src = tmp_path / "in.xlsx"
    pd.DataFrame({"项目名称": ["x"]}).to_excel(src, index=False)

    assert freq_mod.main(["--src", str(src)]) == 1
    assert "标的物" in capsys.readouterr().out


def test_main_returns_zero_and_writes_nothing_when_all_empty(tmp_path, capsys):
    src = _write_src(tmp_path, [None, None])
    dest = tmp_path / "out.xlsx"

    assert freq_mod.main(["--src", str(src), "-o", str(dest)]) == 0
    assert not dest.exists()
    assert "没有任何" in capsys.readouterr().out
