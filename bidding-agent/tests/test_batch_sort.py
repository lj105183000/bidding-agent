"""batch/按custom_id排序.py 的排序逻辑（docs/开发文档.md §9.2 点名的 batch 脚本测试）。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符而失败。

头号用例是 `test_sorts_numerically_not_lexicographically`：字符串排序把 request-10
插到 request-2 前面，整个文件不报错但结果全错位，是这一步唯一致命的坑。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "按custom_id排序.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_sort", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sort_mod = _load_module()


def _record(seq: int, **extra) -> dict:
    return {"custom_id": f"request-{seq}", "response": {"body": extra or {"n": seq}}}


def _write(path: Path, records: list, encoding: str = "utf-8") -> Path:
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    path.write_text(payload, encoding=encoding)
    return path


def _seqs(records: list[dict]) -> list[int]:
    return [sort_mod.parse_custom_id(r["custom_id"]) for r in records]


# ---- 排序本身 ----


def test_sorts_numerically_not_lexicographically():
    """`request-10` 的字典序在 `request-2` 前面——按字符串排会把结果整体错位。"""
    indexed = [(10, _record(10)), (2, _record(2)), (1, _record(1))]
    records, _ = sort_mod.sort_records(indexed)
    assert _seqs(records) == [1, 2, 10]


def test_output_is_strictly_increasing():
    indexed = [(seq, _record(seq)) for seq in [7, 3, 9, 1, 5]]
    records, _ = sort_mod.sort_records(indexed)
    assert _seqs(records) == [1, 3, 5, 7, 9]


def test_sort_does_not_mutate_the_input_list():
    indexed = [(2, _record(2)), (1, _record(1))]
    before = list(indexed)
    sort_mod.sort_records(indexed)
    assert indexed == before


def test_records_keep_their_payload():
    indexed = [(2, _record(2, 项目="乙")), (1, _record(1, 项目="甲"))]
    records, _ = sort_mod.sort_records(indexed)
    assert records[0]["response"]["body"]["项目"] == "甲"


def test_duplicate_ids_keep_file_order():
    """重复序号不致命，但必须是稳定排序——否则同一份输入两次跑出不同结果。"""
    first = {"custom_id": "request-1", "tag": "文件里靠前的"}
    second = {"custom_id": "request-1", "tag": "文件里靠后的"}
    records, stats = sort_mod.sort_records([(1, first), (1, second)])
    assert [r["tag"] for r in records] == ["文件里靠前的", "文件里靠后的"]
    assert stats["duplicates"] == 1


def test_stats_report_range_and_gaps():
    """缺口是正常的（第 2 步跳过空名称的行），但要报出来——也可能是平台漏返回。"""
    indexed = [(1, _record(1)), (4, _record(4))]
    _, stats = sort_mod.sort_records(indexed)
    assert (stats["first"], stats["last"]) == (1, 4)
    assert stats["gaps"] == 2


def test_empty_input_yields_empty_stats():
    records, stats = sort_mod.sort_records([])
    assert records == []
    assert stats["first"] is None


# ---- custom_id 解析 ----


def test_parse_custom_id_reads_trailing_number():
    assert sort_mod.parse_custom_id("request-12") == 12


def test_parse_custom_id_tolerates_other_prefixes():
    """只认最后一段数字：第 2 步的 `request-` 是约定，不该让脚本绑死在它上面。"""
    assert sort_mod.parse_custom_id("req-3") == 3
    assert sort_mod.parse_custom_id("12") == 12
    assert sort_mod.parse_custom_id(12) == 12


def test_parse_custom_id_rejects_non_numeric():
    with pytest.raises(ValueError, match="无法解析"):
        sort_mod.parse_custom_id("request-abc")


def test_parse_custom_id_rejects_empty_suffix():
    with pytest.raises(ValueError, match="无法解析"):
        sort_mod.parse_custom_id("request-")


# ---- 读文件 ----


def test_load_jsonl_indexes_records_by_sequence(tmp_path):
    path = _write(tmp_path / "in.jsonl", [_record(2), _record(1)])
    indexed, stats = sort_mod.load_jsonl(path)
    assert [seq for seq, _ in indexed] == [2, 1]  # 读进来时保持文件顺序
    assert stats["input"] == 2


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps(_record(1)) + "\n\n   \n" + json.dumps(_record(2)) + "\n",
        encoding="utf-8",
    )
    indexed, stats = sort_mod.load_jsonl(path)
    assert len(indexed) == 2
    assert stats["blank"] == 2


def test_load_jsonl_handles_bom(tmp_path):
    """带 BOM 时首行的 `{` 会解析失败——平台/Windows 工具导出常见的坑。"""
    path = tmp_path / "in.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_record(1)).encode("utf-8"))
    indexed, _ = sort_mod.load_jsonl(path)
    assert len(indexed) == 1


def test_load_jsonl_reports_line_number_for_bad_json(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps(_record(1)) + "\n{坏行\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        sort_mod.load_jsonl(path)


def test_load_jsonl_reports_line_number_for_missing_custom_id(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps({"response": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 行.*custom_id"):
        sort_mod.load_jsonl(path)


def test_load_jsonl_reports_line_number_for_bad_custom_id(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps({"custom_id": "oops"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 行"):
        sort_mod.load_jsonl(path)


def test_load_jsonl_rejects_non_object_line(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是 JSON 对象"):
        sort_mod.load_jsonl(path)


def test_load_jsonl_of_empty_file_is_empty(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("", encoding="utf-8")
    indexed, stats = sort_mod.load_jsonl(path)
    assert indexed == []
    assert stats["input"] == 0


# ---- 序列化与路径 ----


def test_render_jsonl_one_object_per_line_with_trailing_newline():
    payload = sort_mod.render_jsonl([_record(1), _record(2)])
    assert len(payload.splitlines()) == 2
    assert payload.endswith("\n")


def test_render_jsonl_keeps_chinese_readable():
    payload = sort_mod.render_jsonl([_record(1, 项目="庐江县供水工程")])
    assert "庐江县供水工程" in payload
    assert "\\u" not in payload


def test_resolve_output_maps_raw_to_processed():
    """契约里这一步的产物名是固定的，别加 `_sorted` 后缀另起炉灶。"""
    src = Path("/tmp/data/batch_results_raw.jsonl")
    assert sort_mod.resolve_output(src) == Path("/tmp/data/batch_results_processed.jsonl")


def test_resolve_output_appends_suffix_for_other_files():
    """排序其他 JSONL（如第 2 步的请求文件）时，产物落在源文件旁边。"""
    src = Path("/tmp/data/batch_requests.jsonl")
    assert sort_mod.resolve_output(src) == Path("/tmp/data/batch_requests_sorted.jsonl")


def test_write_text_reports_locked_file(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "write_text", _raise)
    with pytest.raises(PermissionError, match="无法写入"):
        sort_mod.write_text(tmp_path / "out.jsonl", "{}\n")


# ---- 端到端 ----


def test_main_sorts_and_writes(tmp_path, capsys):
    src = _write(tmp_path / "batch_results_raw.jsonl", [_record(10), _record(2), _record(1)])
    assert sort_mod.main(["--src", str(src)]) == 0

    dest = tmp_path / "batch_results_processed.jsonl"
    records = [json.loads(line) for line in dest.read_text("utf-8").splitlines()]
    assert _seqs(records) == [1, 2, 10]
    assert "序号   : 1 .. 10" in capsys.readouterr().out


def test_main_dry_run_does_not_write(tmp_path, capsys):
    src = _write(tmp_path / "batch_results_raw.jsonl", [_record(2), _record(1)])
    assert sort_mod.main(["--src", str(src), "--dry-run"]) == 0
    assert not (tmp_path / "batch_results_processed.jsonl").exists()
    assert "未写出文件" in capsys.readouterr().out


def test_main_writes_custom_output_path(tmp_path):
    src = _write(tmp_path / "in.jsonl", [_record(2), _record(1)])
    dest = tmp_path / "out" / "sorted.jsonl"
    assert sort_mod.main(["--src", str(src), "-o", str(dest)]) == 0
    assert _seqs([json.loads(l) for l in dest.read_text("utf-8").splitlines()]) == [1, 2]


def test_main_refuses_to_overwrite_source(tmp_path, capsys):
    """结果文件是平台下载的原始凭据，覆盖掉就没了。"""
    src = _write(tmp_path / "in.jsonl", [_record(1)])
    assert sort_mod.main(["--src", str(src), "-o", str(src)]) == 1
    assert "覆盖" in capsys.readouterr().out


def test_main_missing_source_returns_error(tmp_path, capsys):
    assert sort_mod.main(["--src", str(tmp_path / "nope.jsonl")]) == 1
    assert "--src" in capsys.readouterr().out


def test_main_empty_file_returns_zero_without_writing(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text("\n\n", encoding="utf-8")
    assert sort_mod.main(["--src", str(src)]) == 0
    assert not (tmp_path / "in_sorted.jsonl").exists()
    assert "没有可排序的记录" in capsys.readouterr().out


def test_main_reports_bad_line_without_traceback(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text("{不是 JSON\n", encoding="utf-8")
    assert sort_mod.main(["--src", str(src)]) == 1
    assert "第 1 行" in capsys.readouterr().out


def test_main_reports_duplicates_in_output(tmp_path, capsys):
    src = _write(tmp_path / "in.jsonl", [_record(1), _record(1)])
    assert sort_mod.main(["--src", str(src), "--dry-run"]) == 0
    assert "重复" in capsys.readouterr().out
