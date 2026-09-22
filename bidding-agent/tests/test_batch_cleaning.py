"""batch/去重.py 的清洗逻辑（docs/开发文档.md §9.2 点名的测试文件）。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符
而失败。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "去重.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_dedupe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dedupe_mod = _load_module()


def _row(**overrides) -> dict:
    """一行基准数据；用 overrides 制造差异。"""
    base = {
        "项目编号": "A001",
        "中标金额": 100.0,
        "标题": "某项目",
        "发布时间": "2023-10-13",
        "采购人": "某单位",
        "代理机构": "某代理",
        "县城": "某县",
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---- 去重键 ----


def test_same_project_and_amount_is_deduplicated():
    df = _frame([_row(), _row(标题="某项目见证书")])
    result, stats = dedupe_mod.dedupe(df)
    assert len(result) == 1
    assert stats["removed"] == 1


def test_same_project_different_amount_is_kept():
    """同一项目不同金额 = 不同标段/不同次中标，是真实的不同交易，不能合并。"""
    df = _frame([_row(中标金额=100.0), _row(中标金额=200.0)])
    result, _ = dedupe_mod.dedupe(df)
    assert len(result) == 2


def test_same_amount_different_project_is_kept():
    df = _frame([_row(项目编号="A001"), _row(项目编号="A002")])
    result, _ = dedupe_mod.dedupe(df)
    assert len(result) == 2


# ---- 保留策略 ----


def test_keeps_row_with_fewer_nulls():
    """信息更完整的一条优先——后续提取标的物时可用字段更多。"""
    df = _frame(
        [
            _row(标题="缺字段的那条", 代理机构=None, 县城=None),
            _row(标题="完整的那条"),
        ]
    )
    result, _ = dedupe_mod.dedupe(df)
    assert len(result) == 1
    assert result.iloc[0]["标题"] == "完整的那条"


def test_ties_on_nulls_prefer_latest_publish_date():
    """空值数相同时取发布时间最新的一条——变更/更正公告后发且通常更准确。"""
    df = _frame(
        [
            _row(标题="旧的", 发布时间="2023-10-01"),
            _row(标题="新的", 发布时间="2023-10-20"),
        ]
    )
    result, _ = dedupe_mod.dedupe(df)
    assert len(result) == 1
    assert result.iloc[0]["标题"] == "新的"


def test_null_count_beats_recency():
    """空值少优先于时间新：信息完整度是第一位。"""
    df = _frame(
        [
            _row(标题="新的但缺字段", 发布时间="2023-10-20", 代理机构=None),
            _row(标题="旧的但完整", 发布时间="2023-10-01"),
        ]
    )
    result, _ = dedupe_mod.dedupe(df)
    assert result.iloc[0]["标题"] == "旧的但完整"


def test_falls_back_to_中标时间_when_发布时间_absent():
    df = _frame([_row(标题="旧"), _row(标题="新")]).drop(columns=["发布时间"])
    df["中标时间"] = ["2023-01-01", "2023-12-31"]
    result, _ = dedupe_mod.dedupe(df)
    assert result.iloc[0]["标题"] == "新"


def test_result_is_deterministic_when_all_keys_tie():
    """两个排序键都相同时，保留哪条不能是未定义行为——否则同一份输入两次跑出不同结果。"""
    rows = [_row(标题=f"第{i}条") for i in range(6)]
    first, _ = dedupe_mod.dedupe(_frame(rows))
    second, _ = dedupe_mod.dedupe(_frame(list(reversed(rows))))
    # 组内完全并列时保留输入中靠前的一条
    assert first.iloc[0]["标题"] == "第0条"
    assert second.iloc[0]["标题"] == "第5条"


# ---- 缺键的行 ----


def test_rows_without_project_id_are_all_kept():
    """缺「项目编号」无法判重：按金额硬合并会把不同项目错误地压成一条。"""
    df = _frame([_row(项目编号=None), _row(项目编号=None)])
    result, stats = dedupe_mod.dedupe(df)
    assert len(result) == 2
    assert stats["skipped_no_key"] == 2


def test_null_project_id_does_not_absorb_real_project_rows():
    df = _frame([_row(项目编号=None), _row(项目编号="A001"), _row(项目编号="A001")])
    result, stats = dedupe_mod.dedupe(df)
    assert len(result) == 2  # 无编号那条 + 去重后的 A001
    assert stats["skipped_no_key"] == 1


# ---- 边界 ----


def test_empty_frame_returns_empty_without_error():
    result, stats = dedupe_mod.dedupe(pd.DataFrame())
    assert result.empty
    assert stats["input"] == 0


def test_missing_key_columns_raise_actionable_error():
    df = pd.DataFrame({"标题": ["x"], "中标金额": [1.0]})
    with pytest.raises(ValueError, match="项目编号"):
        dedupe_mod.dedupe(df)


def test_original_columns_are_preserved():
    df = _frame([_row(), _row(标题="另一条")])
    result, _ = dedupe_mod.dedupe(df)
    # 排序用的辅助列不能泄漏到结果里
    assert list(result.columns) == list(df.columns)
    assert not any(str(c).startswith("_") for c in result.columns)


def test_output_keeps_input_order_of_surviving_rows():
    df = _frame(
        [
            _row(项目编号="A001"),
            _row(项目编号="A002"),
            _row(项目编号="A001", 标题="重复的那条"),
        ]
    )
    result, _ = dedupe_mod.dedupe(df)
    assert list(result["项目编号"]) == ["A001", "A002"]


def test_duplicate_columns_are_not_mutated():
    """去重不应修改调用方传入的 DataFrame（避免上游数据被悄悄改掉）。"""
    df = _frame([_row(), _row(标题="重复")])
    before = df.copy()
    dedupe_mod.dedupe(df)
    pd.testing.assert_frame_equal(df, before)


# ---- 路径与输出 ----


def test_resolve_output_lands_in_processed_with_suffix():
    """去重产出固定落 data/processed 并加 `_2` 后缀，与 data/raw 的原始数据分离。"""
    src = Path("/tmp/data/raw/训练数据.xlsx")
    expected = dedupe_mod.DEFAULT_PROCESSED_DIR / "训练数据_2.xlsx"
    assert dedupe_mod.resolve_output(src) == expected


def test_load_data_skips_empty_leading_sheet(tmp_path):
    """源文件常见空 Sheet1 + 数据在 Sheet2 的结构，写死 sheet 名换数据就踩空。"""
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Sheet1", index=False)
        _frame([_row()]).to_excel(writer, sheet_name="Sheet2", index=False)
    frame, sheet_name = dedupe_mod.load_data(path)
    assert sheet_name == "Sheet2"
    assert len(frame) == 1


def test_load_data_raises_when_no_sheet_has_rows(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="非空工作表"):
        dedupe_mod.load_data(path)


def test_write_excel_reports_locked_file(tmp_path, monkeypatch):
    """Excel 打开着目标文件时会抛 PermissionError，必须转成可操作的中文提示。"""
    def _raise(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(pd.DataFrame, "to_excel", _raise)
    with pytest.raises(PermissionError, match="关闭"):
        dedupe_mod.write_excel(_frame([_row()]), tmp_path / "out.xlsx", "Sheet1")


# ---- 端到端 ----
#
# 产出目录固定是 data/processed，与 --src 无关，所以端到端用例必须把
# DEFAULT_PROCESSED_DIR 指到 tmp_path：否则测试会写进仓库真实的数据目录，
# 而且不同用例的 in.xlsx 会撞同一个输出文件、互相覆盖。


def test_main_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dedupe_mod, "DEFAULT_PROCESSED_DIR", tmp_path)
    src = tmp_path / "in.xlsx"
    _frame([_row(), _row(标题="重复")]).to_excel(src, index=False)

    assert dedupe_mod.main(["--src", str(src), "--dry-run"]) == 0
    assert not dedupe_mod.resolve_output(src).exists()
    assert "未写出文件" in capsys.readouterr().out


def test_main_writes_deduped_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dedupe_mod, "DEFAULT_PROCESSED_DIR", tmp_path)
    src = tmp_path / "in.xlsx"
    _frame([_row(), _row(标题="重复的那条")]).to_excel(src, index=False)

    assert dedupe_mod.main(["--src", str(src)]) == 0
    written = pd.read_excel(dedupe_mod.resolve_output(src))
    assert len(written) == 1


def test_main_missing_source_returns_error(tmp_path):
    assert dedupe_mod.main(["--src", str(tmp_path / "nope.xlsx")]) == 1
