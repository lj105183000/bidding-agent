"""batch/提数据_编JSONL.py 的请求构造逻辑（docs/开发文档.md §9.2 点名的 batch 脚本测试）。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符
而失败。

重点守住的是 **custom_id 契约**：第 3 步按它排序、第 4 步按它回填「标的物」列，
序号一旦错位，整条管道会把 A 行的标的物填到 B 行，而且不报错。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "提数据_编JSONL.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_jsonl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


jsonl_mod = _load_module()

MODEL = "glm-4-test"


def _frame(names: list) -> pd.DataFrame:
    """两列：待抽取的「项目名称」+ 一列无关字段（不该被泄漏进请求）。"""
    return pd.DataFrame({"项目名称": names, "项目编号": [f"A{i}" for i in range(len(names))]})


def _requests(names: list, limit: int | None = None):
    return jsonl_mod.build_requests(_frame(names), MODEL, limit)


def _user_content(request: dict) -> str:
    return request["body"]["messages"][-1]["content"]


# ---- 抽取 ----


def test_one_request_per_project_name():
    requests, stats = _requests(["某个项目", "另一个项目"])
    assert len(requests) == 2
    assert stats["rows"] == 2
    assert stats["emitted"] == 2


def test_project_name_is_embedded_in_user_message():
    requests, _ = _requests(["庐江县2023年大气污染防控精准溯源服务项目"])
    assert _user_content(requests[0]) == "项目名称：庐江县2023年大气污染防控精准溯源服务项目"


def test_messages_are_system_then_user():
    """顺序不能反：系统提示词给出抽取规则和示例，用户消息才是待抽的那一行。"""
    requests, _ = _requests(["某个项目"])
    messages = requests[0]["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == jsonl_mod.SYSTEM_PROMPT


def test_request_shape_matches_bigmodel_batch_format():
    requests, _ = _requests(["某个项目"])
    request = requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "/v4/chat/completions"
    assert request["body"]["model"] == MODEL


def test_temperature_is_low_for_reproducibility():
    """抽取要的是可复现：同一行两次跑出不同标的物，第 5 步的频次统计就没法比。"""
    requests, _ = _requests(["某个项目"])
    assert requests[0]["body"]["temperature"] <= 0.2


# ---- custom_id ----


def test_custom_id_starts_at_one_and_increments():
    requests, _ = _requests(["甲", "乙", "丙"])
    assert [r["custom_id"] for r in requests] == ["request-1", "request-2", "request-3"]


def test_custom_id_is_zero_padding_free_numeric_suffix():
    """约定是 `request-<行号>`（无补零），第 3 步靠 split("-")[-1] 取数值排序。"""
    requests, _ = _requests(["甲"] * 12)
    assert requests[11]["custom_id"] == "request-12"
    assert jsonl_mod.make_custom_id(7) == "request-7"


def test_custom_id_keeps_row_number_when_blank_rows_are_skipped():
    """空名称的行不发请求，但序号照旧占位——否则后面所有结果整体错位一行。"""
    requests, _ = _requests(["甲", None, "丙"])
    assert [r["custom_id"] for r in requests] == ["request-1", "request-3"]


def test_custom_ids_are_unique():
    requests, _ = _requests([f"项目{i}" for i in range(50)])
    ids = [r["custom_id"] for r in requests]
    assert len(set(ids)) == len(ids)


# ---- 空值与缺列 ----


def test_blank_names_are_skipped_and_counted():
    requests, stats = _requests(["甲", None, float("nan"), "   ", "乙"])
    assert [r["custom_id"] for r in requests] == ["request-1", "request-5"]
    assert stats["skipped_empty"] == 3


def test_whitespace_is_stripped_from_names():
    requests, _ = _requests(["  带空格的名称  "])
    assert _user_content(requests[0]) == "项目名称：带空格的名称"


def test_all_blank_names_yield_no_requests():
    requests, stats = _requests([None, "  "])
    assert requests == []
    assert stats["emitted"] == 0


def test_missing_name_column_raises_actionable_error():
    df = pd.DataFrame({"标题": ["x"], "项目编号": ["A1"]})
    with pytest.raises(ValueError, match="项目名称"):
        jsonl_mod.build_requests(df, MODEL)


def test_empty_frame_returns_empty_without_error():
    requests, stats = jsonl_mod.build_requests(pd.DataFrame(), MODEL)
    assert requests == []
    assert stats["rows"] == 0


def test_other_columns_do_not_leak_into_request():
    requests, _ = _requests(["某个项目"])
    assert "A0" not in json.dumps(requests[0], ensure_ascii=False)


# ---- limit ----


def test_limit_caps_processed_rows():
    requests, stats = _requests(["甲", "乙", "丙"], limit=2)
    assert [r["custom_id"] for r in requests] == ["request-1", "request-2"]
    assert stats["emitted"] == 2


def test_limit_counts_rows_not_emitted_requests():
    """--limit 10 是"只看前 10 行"，不是"产出 10 条"——试跑时要能对上原始行号。"""
    requests, _ = _requests([None, None, "丙", "丁"], limit=3)
    assert [r["custom_id"] for r in requests] == ["request-3"]


# ---- 序列化 ----


def test_render_jsonl_one_object_per_line_with_trailing_newline():
    requests, _ = _requests(["甲", "乙"])
    payload = jsonl_mod.render_jsonl(requests)
    lines = payload.splitlines()
    assert len(lines) == 2
    assert payload.endswith("\n")
    assert json.loads(lines[0])["custom_id"] == "request-1"


def test_render_jsonl_keeps_chinese_readable():
    """转义成 \\uXXXX 的中文没法人工抽查，而 batch 文件动辄上千行。"""
    requests, _ = _requests(["中文名称"])
    payload = jsonl_mod.render_jsonl(requests)
    assert "中文名称" in payload
    assert "\\u" not in payload


def test_render_jsonl_of_empty_list_is_empty_string():
    assert jsonl_mod.render_jsonl([]) == ""


# ---- 单文件上限 ----

def test_limit_violation_reports_too_many_requests(monkeypatch):
    monkeypatch.setattr(jsonl_mod, "MAX_REQUESTS_PER_FILE", 2)
    requests, _ = _requests(["甲", "乙", "丙"])
    violation = jsonl_mod.limit_violation(len(requests), jsonl_mod.render_jsonl(requests))
    assert violation and "3" in violation


def test_limit_violation_reports_oversized_file(monkeypatch):
    monkeypatch.setattr(jsonl_mod, "MAX_FILE_MB", 0.0001)
    requests, _ = _requests(["甲" * 200])
    violation = jsonl_mod.limit_violation(len(requests), jsonl_mod.render_jsonl(requests))
    assert violation and "MB" in violation


def test_limit_violation_is_none_within_limits():
    requests, _ = _requests(["甲", "乙"])
    assert jsonl_mod.limit_violation(len(requests), jsonl_mod.render_jsonl(requests)) is None


# ---- 模型名与路径解析 ----


def test_default_model_is_a_batch_capable_model(monkeypatch):
    """**不读 .env 的 ZHIPU_MODEL**：那是 Chat 侧模型名（glm-4.7-flashx），批量接口不认，
    平台会在上传时以 1210「模型名称错误」把整批拒掉。默认值必须是批量接口支持的名字。"""
    monkeypatch.setenv("ZHIPU_MODEL", "glm-4.7-flashx")
    assert jsonl_mod.default_model() == jsonl_mod.BATCH_MODEL
    assert jsonl_mod.default_model() != "glm-4.7-flashx"


def test_resolve_default_src_points_at_processed_dedup_file():
    """输入固定是 data/processed 下的去重版（`_2` 后缀，见 去重.py），不去 raw 探测
    ——原始数据与加工数据分离，避免管道里"读错了批次"。"""
    assert jsonl_mod.resolve_default_src() == jsonl_mod.DEFAULT_SRC
    assert jsonl_mod.DEFAULT_SRC.parent.name == "processed"
    assert jsonl_mod.DEFAULT_SRC.stem.endswith("_2")


def test_load_data_skips_empty_leading_sheet(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Sheet1", index=False)
        _frame(["某个项目"]).to_excel(writer, sheet_name="Sheet2", index=False)
    frame, sheet_name = jsonl_mod.load_data(path)
    assert sheet_name == "Sheet2"
    assert len(frame) == 1


def test_write_text_reports_locked_file(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "write_text", _raise)
    with pytest.raises(PermissionError, match="无法写入"):
        jsonl_mod.write_text(tmp_path / "out.jsonl", "{}\n")


# ---- 端到端 ----


def _write_src(tmp_path, names) -> Path:
    src = tmp_path / "in.xlsx"
    _frame(names).to_excel(src, index=False)
    return src


def test_main_writes_parseable_jsonl(tmp_path, capsys):
    src = _write_src(tmp_path, ["甲", "乙"])
    dest = tmp_path / "out" / "requests.jsonl"

    assert jsonl_mod.main(["--src", str(src), "-o", str(dest), "--model", MODEL]) == 0
    records = [json.loads(line) for line in dest.read_text("utf-8").splitlines()]
    assert [r["custom_id"] for r in records] == ["request-1", "request-2"]
    assert "生成数 : 2" in capsys.readouterr().out


def test_main_dry_run_does_not_write(tmp_path, capsys):
    src = _write_src(tmp_path, ["甲"])
    dest = tmp_path / "out.jsonl"

    assert jsonl_mod.main(["--src", str(src), "-o", str(dest), "--dry-run"]) == 0
    assert not dest.exists()
    assert "未写出文件" in capsys.readouterr().out


def test_main_missing_source_returns_error(tmp_path, capsys):
    assert jsonl_mod.main(["--src", str(tmp_path / "nope.xlsx")]) == 1
    assert "去重版" in capsys.readouterr().out


def test_main_rejects_non_positive_limit(tmp_path, capsys):
    src = _write_src(tmp_path, ["甲"])
    assert jsonl_mod.main(["--src", str(src), "--limit", "0"]) == 1
    assert "--limit" in capsys.readouterr().out


def test_main_reports_missing_column_without_traceback(tmp_path, capsys):
    """缺「项目名称」列是"数据换了一批"的常见症状，要给能照做的提示而不是堆栈。"""
    src = tmp_path / "in.xlsx"
    pd.DataFrame({"标题": ["x"]}).to_excel(src, index=False)

    assert jsonl_mod.main(["--src", str(src)]) == 1
    assert "项目名称" in capsys.readouterr().out


def test_main_returns_zero_and_writes_nothing_when_all_names_blank(tmp_path, capsys):
    src = _write_src(tmp_path, [None, None])
    dest = tmp_path / "out.jsonl"

    assert jsonl_mod.main(["--src", str(src), "-o", str(dest)]) == 0
    assert not dest.exists()
    assert "没有可抽取" in capsys.readouterr().out
