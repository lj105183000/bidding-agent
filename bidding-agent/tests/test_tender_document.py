"""文档解析与招标文件抽取单测。

只测**不依赖 LLM、不依赖真实文件**的部分：格式识别、章节切分、批次构造、合并逻辑。
真实 PDF/Word 的解析由 `logs/test_document_parser.py` 端到端覆盖。

重点守住几处**踩过的坑**：
    - 扫描件必须明确报错，不能返回空字符串（否则上层以为"文件没内容"）
    - 日期必须保留时分（招标截止时间丢小时 = 可能废标）
    - 批次合并时列表章节要并集去重（资质要求可能分散在多章）
"""

import io

import pytest

from src.constants import tender_schema as TS
from src.services import document_parser as DP
from src.services import tender_extractor as TE


# ---------------------------------------------------------------------------
# 格式识别
# ---------------------------------------------------------------------------


def test_detect_pdf_by_magic():
    assert DP.detect_kind("x.pdf", b"%PDF-1.7\n...") == "pdf"


def test_detect_docx_by_magic():
    assert DP.detect_kind("x.docx", b"PK\x03\x04" + b"\x00" * 100) == "docx"


def test_detect_zip_that_is_not_docx():
    """xlsx/pptx 也是 PK 开头，但没有 word/ 目录 → 必须区分出来。

    否则会对一个 Excel 报"Word 打开失败"，用户看不懂。
    """
    data = b"PK\x03\x04" + b"\x00" * 100
    assert DP.detect_kind("x.pdf", data) == "zip"


def test_magic_wins_over_extension():
    """内容才是可信来源：把 zip 改名成 .pdf 不应被当成 PDF。"""
    assert DP.detect_kind("fake.pdf", b"PK\x03\x04" + b"\x00" * 100) == "zip"


def test_detect_unknown():
    assert DP.detect_kind("x.txt", b"hello world") == "unknown"


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------


def test_empty_file_raises():
    with pytest.raises(DP.DocumentParseError):
        DP.parse_document("x.pdf", b"")


def test_unsupported_type_raises_with_readable_message():
    with pytest.raises(DP.DocumentParseError) as ei:
        DP.parse_document("x.txt", b"hello " * 100)
    assert "不支持的文件类型" in str(ei.value)


def test_zip_masquerading_as_pdf_raises():
    with pytest.raises(DP.DocumentParseError) as ei:
        DP.parse_document("fake.pdf", b"PK\x03\x04" + b"\x00" * 500)
    assert "压缩包" in str(ei.value)


def test_broken_pdf_raises_not_silent():
    """损坏的 PDF 必须报错——静默返回空会让上层以为文件没内容。"""
    with pytest.raises(DP.DocumentParseError):
        # 中文说明需显式编码：bytes 字面量只能含 ASCII 字符
        DP.parse_document("bad.pdf", b"%PDF-1.4\n" + "这不是真的 PDF".encode("utf-8"))


def test_parse_error_is_runtime_error():
    assert issubclass(DP.DocumentParseError, RuntimeError)


# ---------------------------------------------------------------------------
# 章节切分
# ---------------------------------------------------------------------------


def test_split_detects_chapter_headings():
    text = "第一章 总则\n内容一。\n第二章 资格要求\n内容二。"
    chunks = DP.split_into_chunks(text)
    heads = [c.heading for c in chunks]
    assert "第一章 总则" in heads
    assert "第二章 资格要求" in heads


def test_split_detects_various_heading_styles():
    text = (
        "第一章 总则\n甲。\n"
        "1.1 小节标题\n乙。\n"
        "一、中文序号标题\n丙。\n"
        "（一）括号序号标题\n丁。\n"
        "附件一：投标文件格式\n戊。\n"
    )
    chunks = DP.split_into_chunks(text)
    heads = " | ".join(c.heading for c in chunks)
    assert "第一章 总则" in heads
    assert "1.1 小节标题" in heads
    assert "一、中文序号标题" in heads
    assert "（一）括号序号标题" in heads
    assert "附件一：投标文件格式" in heads


def test_long_paragraph_is_not_treated_as_heading():
    """正文恰好以数字开头时不应被误判为标题。

    否则一个长段落会被切碎，章节归属全乱。
    """
    long_line = "1. " + "这是一段很长的正文内容" * 20  # > _HEADING_MAX_CHARS
    chunks = DP.split_into_chunks(long_line)
    assert chunks
    assert all(c.heading == "" for c in chunks)


def test_split_categorizes_relevant_sections():
    text = (
        "第二章 投标人资格要求\n具有独立法人资格。\n"
        "第六章 废标条款\n未密封的作废标处理。\n"
        "第三章 评标办法\n综合评分法。\n"
    )
    chunks = DP.split_into_chunks(text)
    cats = {c.heading: c.categories for c in chunks}
    assert "qualifications" in cats["第二章 投标人资格要求"]
    assert "disqualification" in cats["第六章 废标条款"]
    assert "scoring_items" in cats["第三章 评标办法"]


def test_split_removes_page_markers():
    """我们自己插的页码标记不应出现在块内容里。"""
    text = "<<<第 1 页>>>\n第一章 总则\n内容。"
    chunks = DP.split_into_chunks(text)
    assert all("<<<" not in c.text for c in chunks)


def test_split_hard_splits_oversized_chunk():
    """超长无标题内容要硬切，否则单块会超过模型上下文。"""
    huge = "内" * (DP.CHUNK_MAX_CHARS * 3)
    chunks = DP.split_into_chunks(huge)
    assert len(chunks) >= 3
    assert all(len(c.text) <= DP.CHUNK_MAX_CHARS for c in chunks)


def test_split_empty_input_returns_no_chunks():
    assert DP.split_into_chunks("") == []


# ---------------------------------------------------------------------------
# 相关块选择
# ---------------------------------------------------------------------------


def _fake_doc(headings_and_texts):
    chunks = [
        DP.Chunk(index=i, heading=h, text=t, categories=DP._categorize(h))
        for i, (h, t) in enumerate(headings_and_texts)
    ]
    return DP.ParsedDocument(
        filename="f.pdf", kind="pdf", pages=1,
        total_chars=sum(len(c.text) for c in chunks),
        truncated=False, text="\n".join(c.text for c in chunks), chunks=chunks,
    )


def test_relevant_chunks_prioritize_business_sections():
    doc = _fake_doc([
        ("第一章 总则", "无关内容" * 10),
        ("第二章 投标人资格要求", "资格内容" * 10),
        ("第三章 其他", "其他内容" * 10),
        ("第六章 废标条款", "废标内容" * 10),
    ])
    chosen = DP.select_relevant_chunks(doc, max_chars=40)
    heads = {c.heading for c in chosen}
    # 命中业务的章节应优先于「总则」「其他」
    assert "第二章 投标人资格要求" in heads or "第六章 废标条款" in heads


def test_relevant_chunks_returned_in_original_order():
    """返回必须按原始顺序——乱序内容会让模型串章节。"""
    doc = _fake_doc([
        ("第一章 总则", "a" * 20),
        ("第二章 投标人资格要求", "b" * 20),
        ("第三章 评标办法", "c" * 20),
    ])
    chosen = DP.select_relevant_chunks(doc, max_chars=1000)
    assert [c.index for c in chosen] == sorted(c.index for c in chosen)


def test_relevant_chunks_respect_max_chars():
    doc = _fake_doc([(f"第{i}章 评标办法", "x" * 100) for i in range(1, 11)])
    chosen = DP.select_relevant_chunks(doc, max_chars=250)
    assert sum(len(c.text) for c in chosen) <= 250


# ---------------------------------------------------------------------------
# 招标文件 schema：日期必须保留时分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026年11月20日 09:30", "2026-11-20 09:30:00"),
        ("2026年11月20日 9:30", "2026-11-20 09:30:00"),
        ("2026年11月20日9：30", "2026-11-20 09:30:00"),
        ("2026年11月20日 09:30:45", "2026-11-20 09:30:45"),
        ("2026-11-20 09:30", "2026-11-20 09:30:00"),
        ("2026/11/20 14:00", "2026-11-20 14:00:00"),
        ("2026-11-20T09:30:00", "2026-11-20 09:30:00"),
    ],
)
def test_datetime_keeps_time_of_day(raw, expected):
    """★ 核心回归：中文日期的时间部分不能丢。

    第一版只捕获年月日，`2026年11月20日 09:30` 被归一成 `00:00:00`。
    投标截止时间差几小时就意味着废标，所以这条必须有测试守住。
    """
    got = TS.normalize_tender({"投标截止时间": raw})["fields"]["bid_deadline"]
    assert got == expected
    assert "09:30" in got or "14:00" in got


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026年11月20日", "2026-11-20 00:00:00"),
        ("2026-11-20", "2026-11-20 00:00:00"),
    ],
)
def test_datetime_without_time_defaults_to_midnight(raw, expected):
    assert TS.normalize_tender({"投标截止时间": raw})["fields"]["bid_deadline"] == expected


def test_datetime_unparseable_kept():
    assert TS.normalize_tender({"投标截止时间": "另行通知"})["fields"]["bid_deadline"] == "另行通知"


# ---------------------------------------------------------------------------
# 招标文件 schema：金额与列表章节
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("125万元", 1250000), ("1,200,000元", 1200000), ("25000", 25000), (1250000, 1250000)],
)
def test_tender_money_normalization(raw, expected):
    assert TS.normalize_tender({"预算金额": raw})["fields"]["budget"] == expected


def test_empty_money_is_none_not_zero():
    """空金额必须是 None 而不是 0——0 会被当成"预算为零"这种荒谬结论。"""
    assert TS.normalize_tender({"预算金额": ""})["fields"]["budget"] is None


def test_sections_parse_list_of_dicts():
    raw = {
        "资格要求": [
            {"text": "具有独立法人资格", "category": "其他", "is_mandatory": "true"},
            {"text": "具有二级资质", "category": "资质等级", "is_mandatory": "否"},
        ]
    }
    got = TS.normalize_tender(raw)["sections"]["qualifications"]
    assert len(got) == 2
    assert got[0]["is_mandatory"] is True
    assert got[1]["is_mandatory"] is False


def test_sections_accept_plain_string_list():
    """模型偶尔只给字符串数组，要能兼容而不是丢弃。"""
    got = TS.normalize_tender({"废标条款": ["未密封的", "逾期送达的"]})["sections"]["disqualification"]
    assert len(got) == 2
    assert got[0]["text"] == "未密封的"


def test_sections_drop_empty_items():
    got = TS.normalize_tender({"废标条款": [{"text": ""}, {"text": "有效条目"}]})["sections"]["disqualification"]
    assert len(got) == 1


def test_unknown_keys_dropped():
    parsed = TS.normalize_tender({"某个不存在的字段": "x", "项目名称": "y"})
    assert "某个不存在的字段" not in parsed["fields"]
    assert parsed["fields"]["project_name"] == "y"


def test_completeness_reflects_sections():
    """列表章节有内容应提高完整度——它们对投标决策与字段同样重要。"""
    only_fields = {
        "fields": {f.key: "x" for f in TS.TENDER_FIELDS},
        "sections": {s.key: [] for s in TS.TENDER_SECTIONS},
    }
    with_sections = {
        "fields": {f.key: "x" for f in TS.TENDER_FIELDS},
        "sections": {s.key: [{"text": "y"}] for s in TS.TENDER_SECTIONS},
    }
    assert TS.completeness(with_sections) > TS.completeness(only_fields)
    assert TS.completeness(with_sections) == 1.0


def test_extraction_prompt_mentions_every_field_and_section():
    prompt = TS.extraction_prompt()
    for f in TS.TENDER_FIELDS:
        assert f.llm_name in prompt, f"提示词缺字段 {f.llm_name}"
    for s in TS.TENDER_SECTIONS:
        assert s.llm_name in prompt, f"提示词缺章节 {s.llm_name}"


def test_extraction_prompt_highlights_disqualification():
    """实测不加这句时废标条款检出率明显偏低，故必须保留强调。"""
    assert "废标条款务必仔细找" in TS.extraction_prompt()


# ---------------------------------------------------------------------------
# 抽取器：JSON 解析与批次
# ---------------------------------------------------------------------------


def test_parse_json_plain():
    assert TE.parse_json_object('{"项目名称": "x"}') == {"项目名称": "x"}


def test_parse_json_in_code_fence():
    assert TE.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_prose():
    assert TE.parse_json_object('好的：\n{"a": 1}\n以上。') == {"a": 1}


@pytest.mark.parametrize("raw", ["", "完全不是 JSON", "[1,2]"])
def test_parse_json_rejects_non_object(raw):
    with pytest.raises(TE.TenderExtractError):
        TE.parse_json_object(raw)


def test_batches_respect_batch_chars():
    doc = _fake_doc([(f"第{i}章 技术要求", "x" * 500) for i in range(1, 21)])
    batches = TE._build_batches(doc, batch_chars=1200)
    assert len(batches) >= 2
    assert all(len(b) <= 1200 + 200 for b in batches)  # 允许标题拼接的少量超出


def test_batches_capped_by_max_batches():
    doc = _fake_doc([(f"第{i}章 技术要求", "x" * 500) for i in range(1, 101)])
    batches = TE._build_batches(doc, batch_chars=600)
    assert len(batches) <= TE.MAX_BATCHES


# ---------------------------------------------------------------------------
# 抽取器：合并
# ---------------------------------------------------------------------------


def test_merge_scalars_first_non_empty_wins():
    a = {"fields": {"project_name": "项目A", "budget": 100}, "sections": {}}
    b = {"fields": {"project_name": "项目B", "budget": 200}, "sections": {}}
    merged, warnings = TE.merge_batches([a, b])
    assert merged["fields"]["project_name"] == "项目A"
    assert merged["fields"]["budget"] == 100
    # 冲突要提示，而不是静默丢弃
    assert any("项目名称" in w for w in warnings)


def test_merge_scalars_fills_gaps_from_later_batches():
    a = {"fields": {"project_name": "项目A"}, "sections": {}}
    b = {"fields": {"project_id": "X-1", "budget": 500}, "sections": {}}
    merged, _ = TE.merge_batches([a, b])
    assert merged["fields"]["project_id"] == "X-1"
    assert merged["fields"]["budget"] == 500


def test_merge_sections_union_and_dedupe():
    """★ 关键：资质要求可能分散在多章，任何一块都不能丢。"""
    a = {"fields": {}, "sections": {"qualifications": [{"text": "具有独立法人资格"}]}}
    b = {"fields": {}, "sections": {"qualifications": [{"text": "具有二级资质"}]}}
    merged, _ = TE.merge_batches([a, b])
    assert len(merged["sections"]["qualifications"]) == 2


def test_merge_sections_dedupe_ignores_numbering():
    """同一要求带不同序号不应重复计入。

    "1. 具有独立法人资格" 与 "（一）具有独立法人资格" 是同一条。
    """
    a = {"fields": {}, "sections": {"qualifications": [{"text": "1. 具有独立法人资格"}]}}
    b = {"fields": {}, "sections": {"qualifications": [{"text": "（一）具有独立法人资格"}]}}
    merged, _ = TE.merge_batches([a, b])
    assert len(merged["sections"]["qualifications"]) == 1


def test_merge_sections_respects_item_cap():
    a = {
        "fields": {},
        "sections": {"disqualification": [{"text": f"条款{i}"} for i in range(200)]},
    }
    merged, _ = TE.merge_batches([a])
    assert len(merged["sections"]["disqualification"]) <= TE.MAX_ITEMS_PER_SECTION


def test_merge_warnings_deduped():
    a = {"fields": {"project_name": "A"}, "sections": {}}
    b = {"fields": {"project_name": "B"}, "sections": {}}
    c = {"fields": {"project_name": "C"}, "sections": {}}
    _, warnings = TE.merge_batches([a, b, c])
    # 同一冲突被多批重复记录时应去重
    assert len(warnings) == len(set(warnings))


def test_extract_error_is_runtime_error():
    assert issubclass(TE.TenderExtractError, RuntimeError)


# ---------------------------------------------------------------------------
# schema 契约
# ---------------------------------------------------------------------------


def test_field_and_section_keys_unique():
    assert len(set(TS.field_keys())) == len(TS.field_keys())
    assert len(set(TS.section_keys())) == len(TS.section_keys())
    # 标量与章节的 key 不能撞（diff 时要区分）
    assert not (set(TS.field_keys()) & set(TS.section_keys()))


def test_display_helpers_match_definitions():
    assert TS.text_fields_for_display() == tuple((f.key, f.label) for f in TS.TENDER_FIELDS)
    assert TS.sections_for_display() == tuple((s.key, s.label) for s in TS.TENDER_SECTIONS)


def test_critical_fields_present():
    """投标决策必需字段不能少。"""
    keys = set(TS.field_keys())
    for k in ("project_name", "project_id", "bid_deadline", "budget", "evaluation_method"):
        assert k in keys, f"缺少关键字段 {k}"


def test_critical_sections_present():
    keys = set(TS.section_keys())
    for k in ("qualifications", "scoring_items", "disqualification"):
        assert k in keys, f"缺少关键章节 {k}"
