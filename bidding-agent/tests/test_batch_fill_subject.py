"""batch/提数据_插标的物.py 的回填逻辑（docs/开发文档.md §9.2 点名的 batch 脚本测试）。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符而失败。

这一步是整条管道唯一"写回业务表"的地方：填错行的代价是统计出错的标的物频次且无从
察觉，所以序号对齐、越界拦截、空答案留空三件事都要有用例守着。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "提数据_插标的物.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_fill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fill_mod = _load_module()


# ---- 造数据 ----


def _result(seq: int, content: str = "空调", status_code: int = 200) -> dict:
    """平台结果行的形状（见 BigModel 文档的结果文件示例）。"""
    return {
        "custom_id": f"request-{seq}",
        "id": "batch_1791490810192076800",
        "response": {
            "status_code": status_code,
            "body": {"choices": [{"finish_reason": "stop", "index": 0,
                                  "message": {"role": "assistant", "content": content}}]},
        },
    }


def _frame(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"项目名称": names, "中标金额": [float(i) for i in range(len(names))]})


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    return path


# ---- 清洗模型输出 ----


def test_clean_keeps_plain_value():
    assert fill_mod.clean_subject_matter("大气污染防控精准溯源服务") == "大气污染防控精准溯源服务"


def test_clean_strips_markdown_fence():
    """官方文档的结果示例里 content 就是被 ``` 包着的，不能原样写进表。"""
    assert fill_mod.clean_subject_matter("```\n胶圈\n```") == "胶圈"
    assert fill_mod.clean_subject_matter("```json\n胶圈\n```") == "胶圈"


def test_clean_strips_label_prefix():
    """提示词要求只输出标的物，但模型偶尔会把标签一起吐出来。"""
    assert fill_mod.clean_subject_matter("标的物：胶圈") == "胶圈"
    assert fill_mod.clean_subject_matter("标的物: 胶圈") == "胶圈"


def test_clean_strips_quotes_and_tail_punctuation():
    assert fill_mod.clean_subject_matter("“胶圈”") == "胶圈"
    assert fill_mod.clean_subject_matter('"胶圈"。') == "胶圈"


def test_clean_does_not_touch_inner_content():
    """只剪包装，不动内容——内部标点和括号是标的物的一部分。"""
    text = "检察、法警服装服饰"
    assert fill_mod.clean_subject_matter(text) == text


def test_clean_of_whitespace_only_is_empty():
    assert fill_mod.clean_subject_matter("   ") == ""


# ---- 从结果行取答案 ----


def test_extract_returns_value_and_ok_status():
    value, status = fill_mod.extract_subject_matter(_result(1, "空调"))
    assert (value, status) == ("空调", "ok")


def test_extract_reads_json_content():
    """官方示例的 content 是 JSON 字符串，模型改用 JSON 作答时要能取出来。"""
    value, status = fill_mod.extract_subject_matter(_result(1, '{"标的物": "空调"}'))
    assert (value, status) == ("空调", "ok")


def test_extract_marks_empty_answer():
    assert fill_mod.extract_subject_matter(_result(1, "   ")) == ("", "empty")


def test_extract_marks_platform_error_field():
    record = _result(1)
    record["error"] = {"code": "rate_limit", "message": "too many requests"}
    assert fill_mod.extract_subject_matter(record) == ("", "failed")


def test_extract_marks_non_200_status():
    assert fill_mod.extract_subject_matter(_result(1, status_code=500)) == ("", "failed")


def test_extract_marks_missing_choices():
    record = {"custom_id": "request-1", "response": {"status_code": 200, "body": {}}}
    assert fill_mod.extract_subject_matter(record) == ("", "failed")


def test_extract_flags_record_without_response():
    """请求文件的每行都没有 response——这正是拿错文件时的信号。"""
    request_line = {"custom_id": "request-1", "method": "POST", "url": "/v4/chat/completions"}
    assert fill_mod.extract_subject_matter(request_line) == ("", "no_response")


# ---- 回填 ----


def test_fills_by_row_number():
    df = _frame(["甲", "乙"])
    filled, stats = fill_mod.apply_subject_matter(df, [_result(2, "乙的标的"), _result(1, "甲的标的")])
    assert list(filled[fill_mod.SUBJECT_COLUMN]) == ["甲的标的", "乙的标的"]
    assert stats["filled"] == 2


def test_missing_rows_stay_empty():
    """第 2 步跳过空「项目名称」会造成序号缺口，对应行留空即可，不是错误。"""
    df = _frame(["甲", None, "丙"])
    filled, stats = fill_mod.apply_subject_matter(df, [_result(1, "甲的"), _result(3, "丙的")])
    assert list(filled[fill_mod.SUBJECT_COLUMN]) == ["甲的", None, "丙的"]
    assert stats["filled"] == 2


def test_failed_records_leave_row_empty():
    df = _frame(["甲", "乙"])
    filled, stats = fill_mod.apply_subject_matter(df, [_result(1, "甲的"), _result(2, status_code=500)])
    assert list(filled[fill_mod.SUBJECT_COLUMN]) == ["甲的", None]
    assert stats["failed"] == 1


def test_fills_existing_column_in_place():
    """列已存在就原地填，不能再追加一列同名列。"""
    df = _frame(["甲"])
    df[fill_mod.SUBJECT_COLUMN] = "旧值"
    filled, _ = fill_mod.apply_subject_matter(df, [_result(1, "新值")])
    assert list(filled.columns).count(fill_mod.SUBJECT_COLUMN) == 1
    assert filled[fill_mod.SUBJECT_COLUMN].iloc[0] == "新值"


def test_appends_column_at_the_end():
    """追加到最后一列而不是插在中间——不打乱下游按列序读表的假设。"""
    df = _frame(["甲"])
    filled, _ = fill_mod.apply_subject_matter(df, [_result(1, "空调")])
    assert list(filled.columns)[-1] == fill_mod.SUBJECT_COLUMN
    assert list(filled.columns)[:-1] == list(df.columns)


def test_out_of_range_sequence_is_fatal():
    """序号越界说明结果文件和这张表不是一对，硬填会把整表错位。"""
    df = _frame(["甲", "乙"])
    with pytest.raises(ValueError, match="越界"):
        fill_mod.apply_subject_matter(df, [_result(99, "越界")])


def test_zero_sequence_is_fatal():
    df = _frame(["甲"])
    with pytest.raises(ValueError, match="越界"):
        fill_mod.apply_subject_matter(df, [_result(0, "第 0 号不存在")])


def test_request_file_is_rejected_with_hint():
    """把 batch_requests.jsonl 当结果文件用是最容易犯的错，要能一句话说清。"""
    df = _frame(["甲"])
    requests = [{"custom_id": "request-1", "method": "POST", "url": "/v4/chat/completions"}]
    with pytest.raises(ValueError, match="请求文件"):
        fill_mod.apply_subject_matter(df, requests)


def test_does_not_mutate_input_frame():
    df = _frame(["甲"])
    before = df.copy()
    fill_mod.apply_subject_matter(df, [_result(1, "空调")])
    pd.testing.assert_frame_equal(df, before)


def test_empty_frame_or_no_records():
    filled, stats = fill_mod.apply_subject_matter(pd.DataFrame(), [_result(1)])
    assert filled.empty
    filled, stats = fill_mod.apply_subject_matter(_frame(["甲"]), [])
    assert stats["filled"] == 0
    assert fill_mod.SUBJECT_COLUMN not in filled.columns


def test_bad_custom_id_is_reported():
    df = _frame(["甲"])
    with pytest.raises(ValueError, match="无法解析"):
        fill_mod.apply_subject_matter(df, [{"custom_id": "oops", "response": {}}])


# ---- 读文件与路径 ----


def test_load_jsonl_skips_blank_lines_and_handles_bom(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps(_result(1), ensure_ascii=False).encode("utf-8") + b"\n\n"
    )
    assert len(fill_mod.load_jsonl(path)) == 1


def test_load_jsonl_reports_line_number_for_bad_json(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps(_result(1)) + "\n{坏行\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        fill_mod.load_jsonl(path)


def test_load_jsonl_reports_missing_custom_id(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps({"response": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="custom_id"):
        fill_mod.load_jsonl(path)


def test_resolve_output_appends_suffix():
    src = Path("/tmp/data/训练数据_2.xlsx")
    assert fill_mod.resolve_output(src) == Path("/tmp/data/训练数据_2_标的物.xlsx")


def test_default_excel_reads_the_same_file_as_step_two():
    """与第 2 步读同一份去重产出（data/processed 下的 `_2.xlsx`），不做多路径探测
    ——探测出"读错了批次"的隐患，比多写两行候选路径更值得避免。"""
    assert fill_mod.resolve_default_excel() == fill_mod.DEFAULT_EXCEL
    assert fill_mod.DEFAULT_EXCEL.parent.name == "processed"
    assert fill_mod.DEFAULT_EXCEL.stem.endswith("_2")


def test_write_excel_reports_locked_file(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(pd.DataFrame, "to_excel", _raise)
    with pytest.raises(PermissionError, match="关闭"):
        fill_mod.write_excel(_frame(["甲"]), tmp_path / "out.xlsx", "Sheet1")


# ---- 端到端 ----


def _setup(tmp_path, names, records):
    excel = tmp_path / "训练数据_去重版.xlsx"
    _frame(names).to_excel(excel, index=False)
    src = _write_jsonl(tmp_path / "batch_results_processed.jsonl", records)
    return excel, src


def test_main_writes_filled_excel(tmp_path, capsys):
    excel, src = _setup(tmp_path, ["甲", "乙"], [_result(1, "空调"), _result(2, "胶圈")])
    assert fill_mod.main(["--src", str(src), "--excel", str(excel)]) == 0

    written = pd.read_excel(fill_mod.resolve_output(excel))
    assert list(written[fill_mod.SUBJECT_COLUMN]) == ["空调", "胶圈"]
    assert "已回填   : 2" in capsys.readouterr().out


def test_main_dry_run_does_not_write(tmp_path, capsys):
    excel, src = _setup(tmp_path, ["甲"], [_result(1, "空调")])
    assert fill_mod.main(["--src", str(src), "--excel", str(excel), "--dry-run"]) == 0
    assert not fill_mod.resolve_output(excel).exists()
    assert "未写出文件" in capsys.readouterr().out


def test_main_writes_custom_output_path(tmp_path):
    excel, src = _setup(tmp_path, ["甲"], [_result(1, "空调")])
    dest = tmp_path / "out" / "结果.xlsx"
    assert fill_mod.main(["--src", str(src), "--excel", str(excel), "-o", str(dest)]) == 0
    assert pd.read_excel(dest)[fill_mod.SUBJECT_COLUMN].iloc[0] == "空调"


def test_main_missing_sources_return_error(tmp_path, capsys):
    excel, _ = _setup(tmp_path, ["甲"], [_result(1, "空调")])
    assert fill_mod.main(["--src", str(tmp_path / "nope.jsonl"), "--excel", str(excel)]) == 1
    assert "按custom_id排序" in capsys.readouterr().out

    src = _write_jsonl(tmp_path / "r.jsonl", [_result(1, "空调")])
    assert fill_mod.main(["--src", str(src), "--excel", str(tmp_path / "nope.xlsx")]) == 1
    assert "去重" in capsys.readouterr().out


def test_main_reports_request_file_mistake_without_traceback(tmp_path, capsys):
    excel = tmp_path / "训练数据_去重版.xlsx"
    _frame(["甲"]).to_excel(excel, index=False)
    src = _write_jsonl(
        tmp_path / "batch_requests.jsonl",
        [{"custom_id": "request-1", "method": "POST", "url": "/v4/chat/completions"}],
    )
    assert fill_mod.main(["--src", str(src), "--excel", str(excel)]) == 1
    assert "请求文件" in capsys.readouterr().out
