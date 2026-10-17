"""标讯提取相关单测：schema 归一化、JSON 容错解析、可疑值清洗。

覆盖的是**最容易出错且最影响下游**的部分：
    - 金额单位换算（公告里"12.5万元"很常见，换算错就是 12.5 元进库）
    - 日期补全（TIMESTAMPTZ 需要完整时间，只给日期会与 xlsx 导入格式不一致）
    - 模型输出容错（加代码块、加说明文字是实战高频情况）
    - 可疑值清洗（模型填 0 或 1900 年这类占位值不能进库污染统计）

不覆盖网络调用（LLM / 视觉模型），那些靠端到端脚本验证。
"""

import pytest

from src.constants.bid_schema import (
    BID_FIELDS,
    column_map,
    completeness,
    extraction_prompt,
    json_keys,
    normalize_record,
    text_fields_for_display,
)
from src.services.extract import ExtractError, parse_json_object, _sanitize


# ---------------------------------------------------------------------------
# schema 定义
# ---------------------------------------------------------------------------


def test_field_count_is_17():
    """字段数必须与前端的 17 列表格一致；改动这里要同步前端。"""
    assert len(BID_FIELDS) == 17


def test_required_fields_present():
    """核心字段必须标记为必填（缺失会降低完整度权重）。"""
    required = {f.key for f in BID_FIELDS if f.required}
    assert required == {"title", "project_id", "project_name"}


def test_keys_unique():
    keys = [f.key for f in BID_FIELDS]
    assert len(keys) == len(set(keys))
    llm_names = [f.llm_name for f in BID_FIELDS]
    assert len(llm_names) == len(set(llm_names))


def test_json_keys_match_llm_names():
    assert json_keys() == tuple(f.llm_name for f in BID_FIELDS)


def test_extraction_prompt_mentions_every_field():
    """提示词必须覆盖全部字段——早期手写 prompt 漏过字段，故用生成式并测住。"""
    prompt = extraction_prompt()
    for f in BID_FIELDS:
        assert f.llm_name in prompt, f"提示词缺少字段 {f.llm_name}"


def test_extraction_prompt_forbids_fabrication():
    assert "不要编造" in extraction_prompt()


def test_column_map_covers_all_fields():
    mapping = column_map()
    for f in BID_FIELDS:
        assert mapping.get(f.label) == f.key


def test_column_map_aliases():
    """与 to_postgresql.py 兼容的同义列映射。"""
    mapping = column_map()
    assert mapping["供应商"] == "supplier"
    assert mapping["交易额"] == "amount"
    assert mapping["标的物"] == "subject_matter"


def test_text_fields_for_display_order_matches_fields():
    assert text_fields_for_display() == tuple((f.key, f.label) for f in BID_FIELDS)


# ---------------------------------------------------------------------------
# 金额归一化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("366500", 366500),
        ("366500.0", 366500),
        ("36.65万元", 366500),
        ("1,684,000元", 1684000),
        ("1.2亿", 120000000),
        ("0.5万", 5000),
        (1684000, 1684000),
        (366500.5, 366500.5),  # 小数保留
    ],
)
def test_money_normalization(raw, expected):
    assert normalize_record({"中标金额": raw})["amount"] == expected


@pytest.mark.parametrize("raw", ["", None, "面议", "—", "-", "元"])
def test_money_empty_inputs(raw):
    assert normalize_record({"中标金额": raw})["amount"] is None


def test_money_integer_returns_int_type():
    """整数金额要返回 int，与 xlsx 导入的 int64 保持一致。"""
    result = normalize_record({"中标金额": "366500.0"})["amount"]
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# 日期归一化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2023-10-13", "2023-10-13 00:00:00"),
        ("2023-10-13 18:17", "2023-10-13 18:17:00"),
        ("2023-10-13 18:17:16", "2023-10-13 18:17:16"),
        ("2023-10-13T18:17:16", "2023-10-13 18:17:16"),
        ("2023年10月13日", "2023-10-13 00:00:00"),
        ("2023/10/13", "2023-10-13 00:00:00"),
        ("2023年10月", "2023-10-01 00:00:00"),
    ],
)
def test_datetime_normalization(raw, expected):
    assert normalize_record({"发布时间": raw})["publish_time"] == expected


def test_datetime_unparseable_kept_as_is():
    """认不出的日期原样保留——保留原文比丢信息更好排查。"""
    assert normalize_record({"发布时间": "去年十月"})["publish_time"] == "去年十月"


def test_datetime_empty():
    assert normalize_record({"发布时间": None})["publish_time"] == ""
    assert normalize_record({"发布时间": ""})["publish_time"] == ""


# ---------------------------------------------------------------------------
# 枚举归一化与未知键处理
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("service", "服务"), ("货物", "货物"), ("works", "工程"), ("其他", "其他")],
)
def test_category_alias(raw, expected):
    assert normalize_record({"类别": raw})["category"] == expected


def test_source_alias():
    assert normalize_record({"来源": "gov"})["source"] == "政府采购"


def test_unknown_keys_dropped():
    record = normalize_record({"标题": "x", "模型乱加的字段": "y"})
    assert "模型乱加的字段" not in record
    assert "title" in record


def test_missing_keys_filled_empty():
    """缺失键要补空值，下游 SQL 不必到处判 None。"""
    record = normalize_record({})
    assert len(record) == len(BID_FIELDS)
    assert record["title"] == ""
    assert record["amount"] is None


def test_english_key_fallback():
    """模型偶尔用英文键返回，应能兼容。"""
    assert normalize_record({"title": "中文标题"})["title"] == "中文标题"


# ---------------------------------------------------------------------------
# 完整度
# ---------------------------------------------------------------------------


def test_completeness_zero_for_empty():
    assert completeness({}) == 0.0


def test_completeness_one_for_full():
    full = {f.key: "x" for f in BID_FIELDS}
    assert completeness(full) == 1.0


def test_completeness_weights_required_fields():
    """只填必填字段 vs 只填同数量的非必填字段，前者得分更高。"""
    required_only = {f.key: "x" for f in BID_FIELDS if f.required}
    optional = [f for f in BID_FIELDS if not f.required][: len(required_only)]
    optional_only = {f.key: "x" for f in optional}
    assert completeness(required_only) > completeness(optional_only)


# ---------------------------------------------------------------------------
# JSON 容错解析
# ---------------------------------------------------------------------------


def test_parse_plain_json():
    assert parse_json_object('{"标题": "x"}') == {"标题": "x"}


def test_parse_json_in_code_fence():
    assert parse_json_object('```json\n{"标题": "x"}\n```') == {"标题": "x"}


def test_parse_json_in_bare_fence():
    assert parse_json_object('```\n{"标题": "x"}\n```') == {"标题": "x"}


def test_parse_json_with_leading_prose():
    """模型常在前面加一句"好的，以下是结果："。"""
    raw = '好的，以下是提取结果：\n{"标题": "x", "金额": 1}\n希望有帮助。'
    assert parse_json_object(raw) == {"标题": "x", "金额": 1}


def test_parse_json_with_trailing_prose_only():
    assert parse_json_object('{"标题": "x"}\n以上。') == {"标题": "x"}


def test_parse_json_line_by_line_fallback():
    raw = '结果如下\n{"标题": "x"}\n'
    assert parse_json_object(raw) == {"标题": "x"}


@pytest.mark.parametrize("raw", ["", "完全不是 JSON", "[1,2,3]", "null"])
def test_parse_json_failure_raises(raw):
    with pytest.raises(ExtractError):
        parse_json_object(raw)


def test_parse_json_rejects_array():
    """顶层是数组也要拒绝——本接口契约是对象。"""
    with pytest.raises(ExtractError):
        parse_json_object('[{"标题": "x"}]')


# ---------------------------------------------------------------------------
# 可疑值清洗
# ---------------------------------------------------------------------------


def test_sanitize_clears_non_positive_amount():
    record, warnings = _sanitize({"amount": 0, "budget": -1})
    assert record["amount"] is None
    assert record["budget"] is None
    assert len(warnings) == 2


def test_sanitize_keeps_positive_amount():
    record, warnings = _sanitize({"amount": 100})
    assert record["amount"] == 100
    assert warnings == []


@pytest.mark.parametrize("bad", ["1900-01-01 00:00:00", "2999-01-01 00:00:00"])
def test_sanitize_clears_absurd_years(bad):
    record, warnings = _sanitize({"publish_time": bad})
    assert record["publish_time"] == ""
    assert warnings


def test_sanitize_keeps_normal_year():
    record, warnings = _sanitize({"publish_time": "2023-10-13 00:00:00"})
    assert record["publish_time"] == "2023-10-13 00:00:00"
    assert warnings == []


def test_sanitize_strips_announcement_suffix_from_project_name():
    """项目名称不该带"中标结果公告"后缀，模型偶尔漏执行该规则。"""
    record, warnings = _sanitize({"project_name": "某某项目中标结果公告"})
    assert record["project_name"] == "某某项目"
    assert warnings


def test_sanitize_keeps_clean_project_name():
    record, warnings = _sanitize({"project_name": "某某项目"})
    assert record["project_name"] == "某某项目"
    assert warnings == []


def test_sanitize_handles_none_dates():
    record, warnings = _sanitize({"publish_time": "", "bid_time": None})
    assert record["publish_time"] == ""
    assert warnings == []
