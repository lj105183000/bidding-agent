"""变更追踪单测：`diff_parsed` 与 `classify_severity`。

为什么这两个函数要密集地单测
--------------------------
变更判定的准确性没法靠端到端验证：LLM 抽取本身有随机性，同一个文件抽两次，
条目文字都可能不同，端到端只能证明"链路能跑通"，证明不了"该报的报了"。
而"该报没报"（漏掉一条 critical 变更）的后果是**废标**，所以判定规则必须有
稳定的回归基线——本文件全部用**手工构造的 parsed**，不碰 LLM、不碰文件、
不碰数据库。

重点守住三类"想当然会出错"的地方：
    1. **格式差异不算变更**：金额 "125万元" vs 1250000、日期 "2026-1-5" vs
       "2026-01-05"、条目 "1. 具有…" vs "（一）具有…"。归一化没做干净，
       告警就会被噪音淹没，用户就不看了。
    2. **空值等价**：None / "" / "暂无" 都是"没填"，不能互报变更。
    3. **分数变了也要报**：评分项文字没改、只有分值从 25 改成 20，纯文本集合
       比对会漏掉，而它属于实质性变更。
"""

import pytest

from src.services import change_tracker as CT


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def P(fields: dict | None = None, sections: dict | None = None) -> dict:
    """造一份 `parsed`（结构同 `extract_from_document()` 的返回值）。"""
    return {"fields": dict(fields or {}), "sections": dict(sections or {})}


def items(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


def changes_by_key(result: dict) -> dict:
    return {c["key"]: c for c in result["field_changes"]}


def notes_text(result: dict) -> str:
    return "\n".join(result["impact_notes"])


# ---------------------------------------------------------------------------
# 无变更
# ---------------------------------------------------------------------------


def test_identical_parsed_has_no_changes():
    """同一份内容比两次必须什么都没有——这是所有判定的基线。"""
    same = P(
        {"project_name": "政务云服务采购项目", "max_price": 1200000},
        {"qualifications": items("具有独立法人资格", "具有二级资质")},
    )
    result = CT.diff_parsed(same, same)
    assert result["summary"] == {"total": 0, "critical": 0, "major": 0, "minor": 0}
    assert result["field_changes"] == []
    assert all(
        not delta["added"] and not delta["removed"]
        for delta in result["section_changes"].values()
    )


def test_empty_inputs_have_no_changes():
    assert CT.diff_parsed(P(), P())["summary"]["total"] == 0


def test_no_change_result_still_explains_scope():
    """无变更时也要说清比对范围，否则用户会以为"连附件都比过了"。"""
    note = CT.diff_parsed(P(), P())["impact_notes"][0]
    assert "未检出变更" in note
    assert "不在比对范围内" in note


def test_unchanged_sections_keep_item_count():
    """没变的章节要如实给出条目数：前端用它显示"其余 7 条未变"。"""
    parsed = P(sections={"qualifications": items("A", "B", "C")})
    delta = CT.diff_parsed(parsed, parsed)["section_changes"]["qualifications"]
    assert delta == {"added": [], "removed": [], "unchanged_count": 3}


# ---------------------------------------------------------------------------
# 字段：新增 / 删除 / 修改
# ---------------------------------------------------------------------------


def test_field_added():
    result = CT.diff_parsed(P(), P({"purchaser": "合肥市数据资源局"}))
    change = changes_by_key(result)["purchaser"]
    assert change["change_type"] == "added"
    assert change["old"] == ""
    assert change["new"] == "合肥市数据资源局"
    assert result["summary"] == {"total": 1, "critical": 0, "major": 0, "minor": 1}


def test_field_removed():
    result = CT.diff_parsed(P({"agency": "安徽正阳工程咨询有限公司"}), P())
    change = changes_by_key(result)["agency"]
    assert change["change_type"] == "removed"
    assert change["old"] == "安徽正阳工程咨询有限公司"
    assert change["new"] == ""


def test_field_modified():
    result = CT.diff_parsed(
        P({"payment_terms": "验收合格后 30 日内付 95%"}),
        P({"payment_terms": "验收合格后 60 日内付 90%"}),
    )
    change = changes_by_key(result)["payment_terms"]
    assert change["change_type"] == "modified"
    assert change["key"] == "payment_terms"
    assert change["label"] == "付款方式"
    assert change["severity"] == "major"


def test_field_change_entry_shape_is_stable():
    """条目字段是前端契约，多一个少一个都会让前端崩，故用测试锁住。"""
    result = CT.diff_parsed(P(), P({"contact": "张工"}))
    assert set(result["field_changes"][0]) == {
        "key", "label", "old", "new", "change_type", "severity"
    }


def test_diff_result_top_level_shape():
    result = CT.diff_parsed(P(), P())
    assert set(result) == {"field_changes", "section_changes", "summary", "impact_notes"}
    assert set(result["summary"]) == {"total", "critical", "major", "minor"}
    assert set(result["section_changes"]) == set(CT._SECTION_LABEL)
    for delta in result["section_changes"].values():
        assert set(delta) == {"added", "removed", "unchanged_count"}


def test_project_id_absence_is_not_reported_when_both_missing():
    """两份文件都没抽到编号 → 不是变更，而是抽取完整度问题（在 meta.warnings 里提示）。"""
    result = CT.diff_parsed(P({"project_name": "X"}), P({"project_name": "X"}))
    assert result["summary"]["total"] == 0


# ---------------------------------------------------------------------------
# 空值等价
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_value,new_value", [(None, ""), ("", None), ("暂无", ""), ("无", "-")])
def test_empty_equivalents_are_not_changes(old_value, new_value):
    """None / "" / "暂无" / "-" 都是"没填"，互相比不能报变更。"""
    result = CT.diff_parsed(
        P({"bid_open_place": old_value}), P({"bid_open_place": new_value})
    )
    assert result["summary"]["total"] == 0


def test_empty_money_equivalents():
    """金额字段的缺失是 None（schema 约定），与 "" 等价。"""
    assert CT.diff_parsed(P({"budget": None}), P({"budget": ""}))["summary"]["total"] == 0


def test_zero_money_is_a_real_value_not_empty():
    """0 是真实值（如"不收取保证金"），不能被当成"没填"而漏报。"""
    result = CT.diff_parsed(P({"deposit": 0}), P({"deposit": ""}))
    change = changes_by_key(result)["deposit"]
    assert change["change_type"] == "removed"
    assert change["old"] == 0


def test_empty_to_value_is_added_not_modified():
    result = CT.diff_parsed(P({"deposit": None}), P({"deposit": 50000}))
    assert changes_by_key(result)["deposit"]["change_type"] == "added"


# ---------------------------------------------------------------------------
# 格式化差异不应报为变更
# ---------------------------------------------------------------------------


def test_money_format_difference_is_not_a_change():
    """"125万元" 与 1250000 是同一笔钱——入口没归一干净时 diff 也必须扛住。"""
    result = CT.diff_parsed(
        P({"max_price": "125万元"}), P({"max_price": 1250000})
    )
    assert result["summary"]["total"] == 0


def test_money_int_vs_float_is_not_a_change():
    assert CT.diff_parsed(
        P({"max_price": 1250000}), P({"max_price": 1250000.0})
    )["summary"]["total"] == 0


def test_money_thousands_separator_is_not_a_change():
    assert CT.diff_parsed(
        P({"budget": "1,200,000.00 元"}), P({"budget": 1200000})
    )["summary"]["total"] == 0


def test_datetime_writing_style_is_not_a_change():
    """中文写法与 ISO 写法、有无秒、全角冒号，都是同一个时间点。"""
    result = CT.diff_parsed(
        P({"bid_deadline": "2026年11月20日 09:30"}),
        P({"bid_deadline": "2026-11-20T09:30:00"}),
    )
    assert result["summary"]["total"] == 0


def test_datetime_without_zero_padding_is_not_a_change():
    """★ "2026-1-5" 与 "2026-01-05" 是同一时刻。

    `tender_schema._to_datetime()` 认不出未补零写法（会原样返回），
    若 diff 直接拿它比字符串，投标截止时间就会冒出一条假变更——
    这个方向最危险的是"假变更掩盖真变更"。
    """
    result = CT.diff_parsed(
        P({"bid_deadline": "2026-1-5"}), P({"bid_deadline": "2026-01-05 00:00:00"})
    )
    assert result["summary"]["total"] == 0


def test_text_whitespace_and_punctuation_are_not_changes():
    result = CT.diff_parsed(
        P({"contact": "张工 0551-63531234"}), P({"contact": "张工0551-63531234。"})
    )
    assert result["summary"]["total"] == 0


def test_unparseable_datetime_change_is_still_reported():
    """认不出的日期（"另行通知"）不能因为归一失败就漏报。"""
    result = CT.diff_parsed(
        P({"bid_deadline": "2026-11-20 09:30:00"}), P({"bid_deadline": "另行通知"})
    )
    change = changes_by_key(result)["bid_deadline"]
    assert change["change_type"] == "modified"
    assert change["new"] == "另行通知"


# ---------------------------------------------------------------------------
# 列表章节：条目集合比对
# ---------------------------------------------------------------------------


def test_section_item_added_and_removed():
    result = CT.diff_parsed(
        P(sections={"qualifications": items("具有独立法人资格", "具有 ISO9001 认证")}),
        P(sections={"qualifications": items("具有独立法人资格", "具有 CMMI3 认证")}),
    )
    delta = result["section_changes"]["qualifications"]
    assert [i["text"] for i in delta["added"]] == ["具有 CMMI3 认证"]
    assert [i["text"] for i in delta["removed"]] == ["具有 ISO9001 认证"]
    assert delta["unchanged_count"] == 1


def test_same_requirement_with_different_numbering_is_not_a_change():
    """★ "1. 具有…" 与 "（一）具有…" 是同一条要求，只换了编号。

    招标文件正文用 "1."，澄清公告常改成 "（一）"；按原文比对会把整章
    都报成"删除 + 新增"，用户直接失去对告警的信任。
    """
    result = CT.diff_parsed(
        P(sections={"qualifications": items("1. 具有独立承担民事责任的能力。")}),
        P(sections={"qualifications": items("（一）具有独立承担民事责任的能力")}),
    )
    delta = result["section_changes"]["qualifications"]
    assert delta["added"] == []
    assert delta["removed"] == []
    assert delta["unchanged_count"] == 1
    assert result["summary"]["total"] == 0


def test_clause_number_and_chapter_prefix_ignored():
    """澄清公告常写"第 3.2 条 具有…"，与原文的同一条要求应视为一致。"""
    result = CT.diff_parsed(
        P(sections={"disqualification": items("投标文件未密封的。")}),
        P(sections={"disqualification": items("第 3.2 条 投标文件未密封的")}),
    )
    assert result["summary"]["total"] == 0


def test_fullwidth_digits_and_parens_normalized():
    result = CT.diff_parsed(
        P(sections={"qualifications": items("（1）具有二级资质")}),
        P(sections={"qualifications": items("(1) 具有二级资质")}),
    )
    assert result["summary"]["total"] == 0


def test_star_marker_difference_is_reported():
    """★ 用来标"实质性要求"，星号本身的变化就是实质性变化，不能归一掉。"""
    result = CT.diff_parsed(
        P(sections={"disqualification": items("★未按要求提交保证金的")}),
        P(sections={"disqualification": items("未按要求提交保证金的")}),
    )
    delta = result["section_changes"]["disqualification"]
    assert len(delta["added"]) == 1 or len(delta["removed"]) == 1


def test_section_item_subfields_are_kept():
    """条目要带分值与类别：前端表格要显示"新增评分项（10 分）"。"""
    result = CT.diff_parsed(
        P(sections={"scoring_items": []}),
        P(sections={"scoring_items": [{"text": "绿色节能方案", "score": 5, "category": "技术"}]}),
    )
    added = result["section_changes"]["scoring_items"]["added"]
    assert added == [{"text": "绿色节能方案", "score": 5, "category": "技术"}]


def test_score_change_is_reported_as_remove_plus_add():
    """★ 评分项文字没改、只有分值从 25 改成 20 —— 纯文本集合比对会漏掉它。

    这是实质性变更（评分标准变了），故报成"旧条删除 + 新条新增"。
    """
    result = CT.diff_parsed(
        P(sections={"scoring_items": [{"text": "技术方案", "score": 25}]}),
        P(sections={"scoring_items": [{"text": "技术方案", "score": 20}]}),
    )
    delta = result["section_changes"]["scoring_items"]
    assert [i["score"] for i in delta["removed"]] == [25]
    assert [i["score"] for i in delta["added"]] == [20]
    assert delta["unchanged_count"] == 0
    assert "分值调整" in notes_text(result)
    assert result["summary"]["critical"] == 2  # 删除 + 新增各算一条


def test_score_missing_on_one_side_is_not_reported():
    """抽取噪音防线：一边抽到分值、另一边没抽到，不能报成评分标准变更。"""
    result = CT.diff_parsed(
        P(sections={"scoring_items": [{"text": "技术方案", "score": 25}]}),
        P(sections={"scoring_items": [{"text": "技术方案", "score": None}]}),
    )
    delta = result["section_changes"]["scoring_items"]
    assert delta["added"] == [] and delta["removed"] == []
    assert delta["unchanged_count"] == 1


def test_duplicate_items_counted_once():
    """同一句话出现两遍只算一条要求（抽取器已去重，这里再兜一次）。"""
    result = CT.diff_parsed(
        P(sections={"disqualification": items("逾期送达的", "逾期送达的")}),
        P(sections={"disqualification": items("逾期送达的")}),
    )
    assert result["summary"]["total"] == 0


def test_placeholder_section_item_is_ignored():
    """"无" 不是一条要求。"""
    result = CT.diff_parsed(
        P(sections={"disqualification": items("无")}),
        P(sections={"disqualification": []}),
    )
    assert result["summary"]["total"] == 0


# ---------------------------------------------------------------------------
# 补遗模式（新文件只列变更条目）
# ---------------------------------------------------------------------------


def test_patch_mode_does_not_report_unmentioned_as_removed():
    """补遗/澄清公告只列改了什么，没提的不等于被删除。"""
    old = P(
        {"max_price": 1200000, "deposit": 25000},
        {"disqualification": items("未密封的", "逾期送达的", "未缴保证金的")},
    )
    patch = P({"max_price": 1080000}, {"disqualification": items("未缴保证金的")})
    result = CT.diff_parsed(old, patch, new_is_patch=True)
    assert [c["key"] for c in result["field_changes"]] == ["max_price"]
    delta = result["section_changes"]["disqualification"]
    assert delta["removed"] == []
    assert delta["unchanged_count"] == 1
    assert "补遗" in notes_text(result)


def test_full_mode_reports_unmentioned_as_removed():
    """默认（全文模式）必须如实报删除——这是与补遗模式的对照。"""
    old = P(sections={"disqualification": items("未密封的", "逾期送达的")})
    new = P(sections={"disqualification": items("未密封的")})
    delta = CT.diff_parsed(old, new)["section_changes"]["disqualification"]
    assert [i["text"] for i in delta["removed"]] == ["逾期送达的"]


def test_patch_mode_still_reports_added():
    old = P(sections={"qualifications": items("A")})
    new = P(sections={"qualifications": items("A", "B")})
    delta = CT.diff_parsed(old, new, new_is_patch=True)["section_changes"]["qualifications"]
    assert [i["text"] for i in delta["added"]] == ["B"]


# ---------------------------------------------------------------------------
# 输入容错与异常收敛
# ---------------------------------------------------------------------------


def test_flat_fields_dict_is_tolerated():
    """容忍只给一层字段快照的调用方式（如从库里取历史值）。"""
    result = CT.diff_parsed({"max_price": 1200000}, {"max_price": 1000000})
    assert changes_by_key(result)["max_price"]["change_type"] == "modified"


def test_section_string_shorthand_is_tolerated():
    result = CT.diff_parsed(
        P(sections={"qualifications": "具有独立法人资格"}),
        P(sections={"qualifications": items("具有独立法人资格")}),
    )
    assert result["summary"]["total"] == 0


def test_raw_chinese_keys_are_tolerated():
    """直接喂模型原始输出的中文键也要能用（normalize_tender 支持）。"""
    result = CT.diff_parsed(
        {"fields": {"最高限价": "120万元"}}, {"fields": {"最高限价": "125万元"}}
    )
    assert changes_by_key(result)["max_price"]["new"] == 1250000


@pytest.mark.parametrize("bad", ["not a dict", 123, None, ["a"]])
def test_non_dict_input_raises_change_track_error(bad):
    with pytest.raises(CT.ChangeTrackError):
        CT.diff_parsed(bad, P())


def test_bad_fields_type_raises():
    with pytest.raises(CT.ChangeTrackError):
        CT.diff_parsed({"fields": ["a"]}, P())


def test_bad_section_type_raises():
    """章节结构非法必须报错，不能静默当成空——那会得出"本章无变更"的错误结论。"""
    with pytest.raises(CT.ChangeTrackError):
        CT.diff_parsed(P(sections={"qualifications": 123}), P())


def test_diff_is_pure_function_without_llm_or_file(monkeypatch):
    """纯函数证明：把解析器与抽取器都换成"一调用就炸"，diff 照常工作。"""
    from src.services import document_parser as DP
    from src.services import tender_extractor as TE

    def boom(*args, **kwargs):
        raise AssertionError("diff_parsed 不应触碰解析/抽取")

    monkeypatch.setattr(DP, "parse_document", boom)
    monkeypatch.setattr(TE, "extract_from_parsed", boom)
    monkeypatch.setattr(TE, "extract_from_document", boom)

    result = CT.diff_parsed(P({"deposit": 25000}), P({"deposit": 50000}))
    assert result["summary"]["critical"] == 1


# ---------------------------------------------------------------------------
# classify_severity：分级规则
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "bid_deadline", "open_time", "max_price", "budget", "deposit",
        "qualifications", "disqualification", "scoring_items", "price_weight",
    ],
)
def test_critical_fields(key):
    """钱 / 时间 / 资格 / 废标 / 评分变了 → 必须重做标书。"""
    assert CT.classify_severity(key) == "critical"


@pytest.mark.parametrize(
    "key",
    [
        "duration", "payment_terms", "bid_validity",
        "evaluation_method", "technical_requirements", "required_documents",
    ],
)
def test_major_fields(key):
    assert CT.classify_severity(key) == "major"


@pytest.mark.parametrize(
    "key", ["contact", "bid_open_place", "purchaser", "agency", "project_name", "project_id"]
)
def test_minor_fields(key):
    assert CT.classify_severity(key) == "minor"


def test_money_up_and_down_are_both_critical():
    """涨跌都改变报价上限 → 都 critical；差别在 impact_notes 的方向描述里。"""
    assert CT.classify_severity("max_price", 1200000, 1080000) == "critical"
    assert CT.classify_severity("max_price", 1080000, 1200000) == "critical"


def test_unparseable_key_falls_back_to_keyword_scan():
    """schema 之外的键（未来新增字段）：宁可报高不报低。"""
    assert CT.classify_severity("other_disqualification_rules") == "critical"
    assert CT.classify_severity("new_field", "无", "投标报价超过最高限价的，作废标处理") == "critical"
    assert CT.classify_severity("payment_schedule") == "major"
    assert CT.classify_severity("unknown_thing", "a", "b") == "minor"


def test_unknown_key_is_not_silently_minor_when_text_is_risky():
    assert CT.classify_severity("", None, "资格要求调整") == "critical"


def test_severity_is_deterministic_for_same_key():
    assert CT.classify_severity("contact", "a", "b") == CT.classify_severity("contact", "c", "d")


def test_severity_applied_to_field_changes():
    result = CT.diff_parsed(
        P({"max_price": 1200000, "contact": "张工"}),
        P({"max_price": 1080000, "contact": "李工"}),
    )
    by_key = changes_by_key(result)
    assert by_key["max_price"]["severity"] == "critical"
    assert by_key["contact"]["severity"] == "minor"


def test_section_changes_use_section_severity():
    result = CT.diff_parsed(
        P(sections={"qualifications": []}),
        P(sections={"qualifications": items("具有 CMMI3 认证")}),
    )
    assert result["summary"]["critical"] == 1


def test_field_changes_sorted_by_severity():
    """危险项排前面：投标员按顺序读，必须先看到要重做标书的那几条。"""
    result = CT.diff_parsed(
        P({"contact": "张工", "payment_terms": "30 日", "max_price": 1200000}),
        P({"contact": "李工", "payment_terms": "60 日", "max_price": 1000000}),
    )
    assert [c["severity"] for c in result["field_changes"]] == ["critical", "major", "minor"]


# ---------------------------------------------------------------------------
# summary 计数
# ---------------------------------------------------------------------------


def test_summary_counts_each_change_entry():
    result = CT.diff_parsed(
        P({"max_price": 1200000, "contact": "张工"}),
        P(
            {"max_price": 1080000, "contact": "李工", "deposit": 50000},
            {"qualifications": items("A", "B"), "scoring_items": []},
        ),
    )
    # max_price(critical) + deposit(critical) + 资格新增 2 条(critical) = 4
    # contact(minor) = 1；原文件的 qualifications 为空，故无删除
    assert result["summary"] == {"total": 5, "critical": 4, "major": 0, "minor": 1}


def test_summary_total_equals_severity_sum():
    result = CT.diff_parsed(
        P({"max_price": 1200000, "duration": "12 个月", "contact": "张工"}),
        P(
            {"max_price": 1000000, "duration": "18 个月", "contact": "李工"},
            {"disqualification": items("未密封的")},
        ),
    )
    summary = result["summary"]
    assert summary["total"] == summary["critical"] + summary["major"] + summary["minor"]
    assert summary["critical"] == 2  # max_price + 废标条款新增 1 条
    assert summary["major"] == 1
    assert summary["minor"] == 1


def test_section_removal_counts_toward_summary():
    result = CT.diff_parsed(
        P(sections={"disqualification": items("A", "B")}),
        P(sections={"disqualification": items("A")}),
    )
    assert result["summary"] == {"total": 1, "critical": 1, "major": 0, "minor": 0}


# ---------------------------------------------------------------------------
# impact_notes：方向与关注点
# ---------------------------------------------------------------------------


def test_max_price_decrease_note_shows_direction():
    """★ 最高限价下调是"压缩利润空间"，必须明确写出方向与金额差。"""
    result = CT.diff_parsed(
        P({"max_price": 1200000}), P({"max_price": 1080000})
    )
    note = notes_text(result)
    assert "最高限价下调" in note
    assert "120,000" in note  # 差额
    assert "1,200,000" in note and "1,080,000" in note
    assert "重新核算" in note


def test_max_price_increase_note_shows_direction():
    result = CT.diff_parsed(P({"max_price": 1200000}), P({"max_price": 1500000}))
    assert "最高限价上调 300,000 元" in notes_text(result)


def test_budget_decrease_also_directional():
    assert "预算金额下调" in notes_text(
        CT.diff_parsed(P({"budget": "500万元"}), P({"budget": 4000000}))
    )


def test_deposit_increase_note():
    note = notes_text(CT.diff_parsed(P({"deposit": 25000}), P({"deposit": 50000})))
    assert "投标保证金上调 25,000 元" in note
    assert "废标" in note


def test_deadline_advanced_note():
    """截止时间**提前**比延后危险，提示要能区分。"""
    note = notes_text(
        CT.diff_parsed(
            P({"bid_deadline": "2026-11-20 09:30:00"}),
            P({"bid_deadline": "2026-11-17 09:30:00"}),
        )
    )
    assert "投标截止时间提前 3 天" in note
    assert "排期" in note


def test_deadline_postponed_note():
    note = notes_text(
        CT.diff_parsed(
            P({"bid_deadline": "2026-11-20 09:30:00"}),
            P({"bid_deadline": "2026-11-21 09:30:00"}),
        )
    )
    assert "投标截止时间延后 1 天" in note


def test_time_difference_in_hours():
    note = notes_text(
        CT.diff_parsed(
            P({"open_time": "2026-11-20 09:30:00"}),
            P({"open_time": "2026-11-20 14:30:00"}),
        )
    )
    assert "开标时间延后 5 小时" in note


def test_minor_change_note_says_no_rework_needed():
    note = notes_text(CT.diff_parsed(P({"contact": "张工"}), P({"contact": "李工"})))
    assert "无需重做标书" in note


def test_open_place_note_mentions_risk():
    note = notes_text(
        CT.diff_parsed(P({"bid_open_place": "201 室"}), P({"bid_open_place": "305 室"}))
    )
    assert "开标地点变更" in note or "开标" in note
    assert "走错场地" in note


def test_required_documents_note_warns_about_rejection():
    note = notes_text(
        CT.diff_parsed(
            P(sections={"required_documents": []}),
            P(sections={"required_documents": items("信用中国查询截图")}),
        )
    )
    assert "需提交材料" in note
    assert "废标" in note


def test_identity_note_when_project_id_differs():
    """编号不同要先怀疑"这两份文件不是同一个项目"，否则结果不成立。"""
    note = notes_text(
        CT.diff_parsed(P({"project_id": "2026BFA0001"}), P({"project_id": "2026BFA0002"}))
    )
    assert "项目编号不同" in note
    assert "同一项目" in note


def test_no_identity_note_when_project_id_same():
    note = notes_text(
        CT.diff_parsed(P({"project_id": "X", "contact": "a"}), P({"project_id": "X", "contact": "b"}))
    )
    assert "项目编号不同" not in note


def test_long_values_are_truncated_in_notes():
    """整段付款方式不该把提示刷屏。"""
    long_text = "验收合格后按季度分期付款，" * 20
    note = notes_text(
        CT.diff_parsed(
            P({"payment_terms": long_text}),
            P({"payment_terms": long_text + "尾款质保期满支付。"}),
        )
    )
    assert "…" in note
    assert len(note) < 400


def test_scoring_items_note_mentions_strategy():
    note = notes_text(
        CT.diff_parsed(
            P(sections={"scoring_items": []}),
            P(sections={"scoring_items": [{"text": "绿色节能方案", "score": 5}]}),
        )
    )
    assert "评分标准" in note
    assert "报价策略" in note


# ---------------------------------------------------------------------------
# 编排：track_changes / stream_track
#
# 用假解析器 + 假抽取器替换掉真实实现，专测编排本身：事件序列、进度上报、
# 失败是否收敛成 ChangeTrackError、补遗模式的自动判断。
# 真实的 PDF 与真实 LLM 由 `logs/test_change_tracker.py` 端到端覆盖。
# ---------------------------------------------------------------------------

OLD_PARSED = P(
    {"project_name": "政务云服务采购项目", "max_price": 1200000, "deposit": 25000},
    {"qualifications": items("具有独立法人资格", "具有 ISO9001 认证")},
)
NEW_PARSED = P(
    {"project_name": "政务云服务采购项目", "max_price": 1080000, "deposit": 50000},
    {"qualifications": items("具有独立法人资格", "具有 CMMI3 认证")},
)
# 信息量明显少于原文件 → 应被判为"只列变更条目的补遗公告"
PATCH_PARSED = P({"max_price": 1080000}, {"qualifications": items("具有独立法人资格")})


@pytest.fixture
def fake_pipeline(monkeypatch):
    from src.services import document_parser as DP
    from src.services import tender_extractor as TE

    payloads = {
        "old.pdf": OLD_PARSED,
        "new.pdf": NEW_PARSED,
        "patch.docx": PATCH_PARSED,
    }

    def fake_parse(filename, data):
        if data == b"broken":
            raise DP.DocumentParseError("该 PDF 没有文字层（很可能是扫描件）")
        return DP.ParsedDocument(
            filename=filename,
            kind="pdf",
            pages=3,
            total_chars=1200,
            truncated=False,
            text="正文",
            chunks=[DP.Chunk(0, "第一章 招标公告", "正文")],
            warnings=[f"{filename} 的解析警告"],
        )

    def fake_extract(parsed, progress=None):
        if parsed.filename == "llm_down.pdf":
            raise TE.TenderExtractError("LLM 客户端不可用（provider=deepseek）")
        if progress:
            progress(0, 2)
            progress(2, 2)
        return {
            "parsed": payloads[parsed.filename],
            "meta": {"kind": parsed.kind, "warnings": [f"{parsed.filename} 的抽取警告"]},
        }

    monkeypatch.setattr(DP, "parse_document", fake_parse)
    monkeypatch.setattr(TE, "extract_from_parsed", fake_extract)


def test_track_changes_end_to_end(fake_pipeline):
    result = CT.track_changes("old.pdf", b"old", "new.pdf", b"new")
    assert set(result) == {"diff", "old", "new", "meta"}
    assert result["old"]["parsed"] == OLD_PARSED
    assert result["new"]["parsed"] == NEW_PARSED
    assert result["diff"]["summary"]["critical"] > 0
    assert result["meta"]["requires_rebid"] is True
    assert result["meta"]["old_kind"] == "pdf"
    assert result["meta"]["new_treated_as_patch"] is False
    assert result["meta"]["elapsed_ms"] >= 0


def test_track_changes_collects_warnings_from_both_documents(fake_pipeline):
    warnings = CT.track_changes("old.pdf", b"old", "new.pdf", b"new")["meta"]["warnings"]
    assert any(w.startswith("[原招标文件]") for w in warnings)
    assert any(w.startswith("[澄清文件]") for w in warnings)
    assert any("实质性" in w for w in warnings)


def test_track_changes_warns_when_same_file_uploaded_twice(fake_pipeline):
    """误传同一份文件是很常见的操作失误，要明确提示而不是静默给"无变更"。"""
    result = CT.track_changes("old.pdf", b"same-bytes", "old.pdf", b"same-bytes")
    assert result["diff"]["summary"]["total"] == 0
    assert any("完全相同" in w for w in result["meta"]["warnings"])


def test_stream_track_event_sequence(fake_pipeline):
    events = list(CT.stream_track("old.pdf", b"old", "new.pdf", b"new"))
    names = [name for name, _ in events]
    assert names[0] == "parse_old"
    assert names[-2:] == ["diff", "done"]
    # 抽取事件里既有进度（phase=progress）也有收尾（phase=done，带 result）
    phases = [d["phase"] for name, d in events if name == "extract_old"]
    assert phases[0] == "progress"
    assert phases[-1] == "done"
    assert "result" in [d for n, d in events if n == "extract_old"][-1]
    assert names.count("parse_new") == 1 and names.count("parse_old") == 1


def test_stream_track_progress_reports_batch_numbers(fake_pipeline):
    events = list(CT.stream_track("old.pdf", b"old", "new.pdf", b"new"))
    ticks = [d for n, d in events if n == "extract_old" and d["phase"] == "progress"]
    assert [t["done"] for t in ticks] == [0, 2]
    assert all(t["total"] == 2 for t in ticks)
    assert "正在抽取第 1/2 部分" in ticks[0]["message"]


def test_stream_track_parse_event_carries_parse_summary(fake_pipeline):
    """解析事件要带页数/字数/块数，前端据此显示"已解析 3 页、切成 1 块"。"""
    first = next(
        data
        for name, data in CT.stream_track("old.pdf", b"old", "new.pdf", b"new")
        if name == "parse_old"
    )
    assert first["stage"] == "原招标文件"
    assert first["pages"] == 3
    assert first["chunk_count"] == 1
    assert first["filename"] == "old.pdf"


def test_parse_failure_becomes_error_event(fake_pipeline):
    """扫描件（无文字层）必须变成 error 事件，而不是抛异常把 SSE 打断。"""
    events = list(CT.stream_track("broken.pdf", b"broken", "new.pdf", b"new"))
    assert events == [
        ("error", {"content": "原招标文件解析失败：该 PDF 没有文字层（很可能是扫描件）"})
    ]


def test_extract_failure_becomes_error_event(fake_pipeline):
    events = list(CT.stream_track("llm_down.pdf", b"old", "new.pdf", b"new"))
    assert len(events) == 2
    assert events[0][0] == "parse_old"
    assert events[1][0] == "error"
    assert "抽取失败" in events[1][1]["content"]


def test_track_changes_raises_change_track_error_on_failure(fake_pipeline):
    """非流式接口把 error 事件转成异常，且异常类型统一。"""
    with pytest.raises(CT.ChangeTrackError) as excinfo:
        CT.track_changes("broken.pdf", b"broken", "new.pdf", b"new")
    assert "解析失败" in str(excinfo.value)


def test_track_changes_progress_callback(fake_pipeline):
    seen: list[str] = []
    CT.track_changes(
        "old.pdf", b"old", "new.pdf", b"new", progress=lambda event, data: seen.append(event)
    )
    assert seen[0] == "parse_old"
    assert "extract_new" in seen
    assert "diff" in seen
    assert "done" not in seen  # done 只在返回值里，不重复回调


def test_patch_document_is_autodetected(fake_pipeline):
    """澄清文件信息量远小于原文件 → 自动按补遗公告处理，并说明理由。"""
    result = CT.track_changes("old.pdf", b"old", "patch.docx", b"patch")
    assert result["meta"]["new_treated_as_patch"] is True
    delta = result["diff"]["section_changes"]["qualifications"]
    assert delta["removed"] == []  # 未提及的不算删除
    assert any("补遗" in w for w in result["meta"]["warnings"])
    assert any("补遗" in n for n in result["diff"]["impact_notes"])


def test_patch_autodetect_can_be_overridden(fake_pipeline):
    result = CT.track_changes("old.pdf", b"old", "patch.docx", b"patch", new_is_patch=False)
    assert result["meta"]["new_treated_as_patch"] is False
    delta = result["diff"]["section_changes"]["qualifications"]
    assert [i["text"] for i in delta["removed"]] == ["具有 ISO9001 认证"]


def test_full_document_is_not_treated_as_patch(fake_pipeline):
    assert CT.track_changes("old.pdf", b"old", "new.pdf", b"new")["meta"][
        "new_treated_as_patch"
    ] is False
