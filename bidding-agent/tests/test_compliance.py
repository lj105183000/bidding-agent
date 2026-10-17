"""合规性检查单测。

只测**不依赖法规语料、不依赖 LLM** 的规则检出层。

本文件的重心是**误报与漏报**，不是"功能能不能跑"：
    - **误报**（把合规写法标成违规）→ 用户会去质疑一个本来没问题的条款，丢脸且浪费机会
    - **漏报**（把违规条款放过）→ 用户以为"检查过了没问题"，风险更大

故每个规则都配了**正例（应命中）与反例（不应命中）**。反例尤其重要：
"或同等档次""8-10 分"这类合规写法必须不被标记。
"""

from __future__ import annotations

import pytest

from src.services import compliance as C


def _check(*clauses: str, section: str = "qualifications") -> dict:
    """把若干条款塞进一个 parsed 结构后检查。"""
    return C.check_compliance(
        {"sections": {section: [{"text": c} for c in clauses]}},
        include_law_basis=False,
    )


def _titles(result: dict) -> list[str]:
    return [f["title"] for f in result["findings"]]


def _keys(result: dict) -> set[str]:
    return {f["rule"] for f in result["findings"]}


# ===========================================================================
# 规则库结构
# ===========================================================================


def test_rules_have_required_fields():
    for r in C.RULES:
        assert r.key and r.title and r.why and r.how_to_verify
        assert r.severity in ("high", "medium", "low"), f"{r.key} 严重度非法"
        assert callable(r.detect)


def test_rule_keys_unique():
    keys = [r.key for r in C.RULES]
    assert len(keys) == len(set(keys))


def test_catalog_shape():
    cat = C.rules_catalog()
    assert cat["count"] == len(C.RULES)
    assert len(cat["rules"]) == len(C.RULES)
    assert set(cat["severity_levels"]) == {"high", "medium", "low"}


def test_severity_order_covers_all_levels():
    assert set(C.SEVERITY_ORDER) == {"high", "medium", "low"}
    assert C.SEVERITY_ORDER["high"] < C.SEVERITY_ORDER["medium"] < C.SEVERITY_ORDER["low"]


def test_compliance_error_is_runtime_error():
    assert issubclass(C.ComplianceError, RuntimeError)


# ===========================================================================
# 本地化要求
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "投标人须在本市设有分支机构并提供本地办公场所证明。",
        "供应商必须在本地注册并缴纳社保。",
        "投标人应具有本地业绩不少于 2 个。",
    ],
)
def test_local_requirement_detected(clause):
    assert "local_requirement" in _keys(_check(clause))


@pytest.mark.parametrize(
    "clause",
    [
        "中标人须在合同签订后 30 日内在项目所在地设立服务机构。",  # 合规写法：承诺而非资格条件
        "投标人须具有独立承担民事责任的能力。",
    ],
)
def test_local_requirement_not_false_positive(clause):
    """★ "中标后承诺设立本地机构"是**合规写法**，不能与"把本地化当资格条件"混为一谈。"""
    assert "local_requirement" not in _keys(_check(clause))


# ===========================================================================
# 注册资本
# ===========================================================================


def test_registered_capital_high_detected():
    assert "registered_capital" in _keys(_check("注册资本不低于 5000 万元人民币。"))


@pytest.mark.parametrize(
    "clause",
    [
        "注册资本不低于 100 万元。",  # 门槛合理，报了会淹没真问题
        "注册资本人民币 1000 万元。",
    ],
)
def test_registered_capital_reasonable_not_flagged(clause):
    assert "registered_capital" not in _keys(_check(clause))


# ===========================================================================
# 指定品牌
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "服务器品牌须为华为或浪潮，不接受其他品牌。",
        "本项目指定品牌为西门子。",
    ],
)
def test_specific_brand_detected(clause):
    assert "specific_brand" in _keys(_check(clause))


@pytest.mark.parametrize(
    "clause",
    [
        "打印机参考品牌：惠普，或同等档次产品。",
        "空调参考品牌：格力，或相当于该档次。",
    ],
)
def test_specific_brand_with_equivalent_not_flagged(clause):
    """★ 写了「或同等」就是**合规写法**，必须不标记。

    这是实践中最标准的合规表述，误报会让用户以为自己写错了。
    """
    assert "specific_brand" not in _keys(_check(clause))


# ===========================================================================
# 特定奖项 / 协会会员
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "获得鲁班奖的加 3 分。",
        "投标人须加入本行业协会。",
        "获得省级以上奖项的加分。",
    ],
)
def test_specific_certificate_detected(clause):
    assert "specific_certificate" in _keys(_check(clause))


def test_normal_qualification_not_flagged():
    assert _keys(_check("投标人须具有电子与智能化工程专业承包二级及以上资质。")) == set()


# ===========================================================================
# 所有制歧视
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "投标人须为国有企业或国有控股企业。",
        "本项目不接受外资企业投标。",
        "仅限国有企业参与。",
    ],
)
def test_ownership_discrimination_detected(clause):
    assert "ownership_discrimination" in _keys(_check(clause))


# ===========================================================================
# 规模歧视
# ===========================================================================


def test_scale_discrimination_detected():
    assert "scale_discrimination" in _keys(_check("投标人上年度营业收入不低于 1 亿元。"))


def test_scale_discrimination_small_not_flagged():
    assert "scale_discrimination" not in _keys(_check("投标人上年度营业收入不低于 500 万元。"))


# ===========================================================================
# 业绩门槛
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "近三年内完成过 6 个以上同类项目业绩。",
        "近三年内完成过六个以上同类项目业绩。",   # ★ 中文数字
        "近三年内完成过十五个同类项目业绩。",     # ★ 中文数字（两位数）
        "近五年内完成过 8 项类似业绩。",
    ],
)
def test_excessive_experience_detected(clause):
    """★ 中文数字必须认。

    实测漏报：第一版正则只认 `\\d+`，"近三年内完成过 6 个以上"里的"三"
    导致整条被静默漏过。漏报比误报危险——用户会以为检查过了没问题。
    """
    assert "excessive_experience" in _keys(_check(clause))


@pytest.mark.parametrize(
    "clause",
    [
        "近三年内完成过 2 个同类项目业绩。",
        "近三年内完成过三 个同类项目业绩。",
    ],
)
def test_reasonable_experience_not_flagged(clause):
    assert "excessive_experience" not in _keys(_check(clause))


def test_excessive_experience_by_amount():
    assert "excessive_experience" in _keys(_check("单项合同金额不低于 2000 万元。"))
    assert "excessive_experience" not in _keys(_check("单项合同金额不低于 200 万元。"))


# ===========================================================================
# 保证金
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "投标保证金仅接受银行转账形式。",
        "本项目不接受银行保函。",
    ],
)
def test_deposit_only_cash_detected(clause):
    assert "deposit_only_cash" in _keys(_check(clause))


def test_deposit_amount_prompts_manual_check():
    """出现保证金金额时提示人工核对比例（2% 上限）。

    规则本身**不做比例计算**——那需要预算金额，而它可能不在同一条里。
    故这条规则的定位是"提示核对"，`how_to_verify` 里写明了算法。
    """
    result = _check("投标保证金为人民币 25,000 元。")
    assert "excessive_deposit" in _keys(result)
    f = next(x for x in result["findings"] if x["rule"] == "excessive_deposit")
    assert "2%" in f["why"] or "百分之二" in f["why"]
    assert "÷" in f["how_to_verify"]


# ===========================================================================
# 评审因素量化
# ===========================================================================


@pytest.mark.parametrize(
    "clause",
    [
        "技术方案根据情况酌情打分。",
        "根据投标文件整体印象评分。",
    ],
)
def test_unquantified_scoring_detected(clause):
    assert "unquantified_scoring" in _keys(_check(clause, section="scoring_items"))


@pytest.mark.parametrize(
    "clause",
    [
        "技术方案优得 8-10 分，良得 5-7 分，一般得 0-4 分。",
        "技术方案完整性（25 分）。",
    ],
)
def test_quantified_scoring_not_flagged(clause):
    """★ 给了明确分值区间就是**合规写法**，必须不标记。"""
    assert "unquantified_scoring" not in _keys(_check(clause, section="scoring_items"))


# ===========================================================================
# 付款条件
# ===========================================================================


def test_payment_terms_unfair_detected():
    assert "payment_terms_unfair" in _keys(_check("中标人须全额垫资施工。"))


def test_normal_payment_not_flagged():
    assert "payment_terms_unfair" not in _keys(_check("验收合格后 30 日内支付合同金额的 95%。"))


# ===========================================================================
# 汇总与排序
# ===========================================================================


def test_findings_sorted_by_severity():
    result = _check(
        "注册资本不低于 5000 万元。",                     # medium
        "投标人须为国有企业。",                            # high
        "评标办法由招标人另行确定。",                      # low
    )
    sevs = [f["severity"] for f in result["findings"]]
    assert sevs == sorted(sevs, key=lambda s: C.SEVERITY_ORDER[s])
    assert sevs[0] == "high"


def test_summary_counts_match_findings():
    result = _check(
        "投标人须为国有企业。",
        "注册资本不低于 5000 万元。",
    )
    s = result["summary"]
    assert s["total"] == len(result["findings"])
    assert s["counts"]["high"] + s["counts"]["medium"] + s["counts"]["low"] == s["total"]
    assert sum(s["by_rule"].values()) == s["total"]


def test_summary_reports_checked_clause_count():
    result = _check("条款一。", "条款二。", "投标人须为国有企业。")
    assert result["summary"]["checked_clauses"] == 3
    assert result["summary"]["flagged_clauses"] == 1


def test_clean_document_reports_no_findings():
    result = _check(
        "投标人须具有独立承担民事责任的能力。",
        "投标人须具有电子与智能化工程专业承包二级及以上资质。",
        "验收合格后 30 日内支付合同金额的 95%。",
    )
    assert result["findings"] == []
    assert result["summary"]["total"] == 0
    assert "未检出" in result["summary"]["conclusion"]


def test_high_severity_drives_conclusion():
    result = _check("投标人须为国有企业。")
    assert "高风险" in result["summary"]["conclusion"]
    assert "质疑" in result["summary"]["conclusion"]


# ===========================================================================
# 免责与措辞（合规检查的核心约束）
# ===========================================================================


def test_disclaimer_is_mandatory_and_substantive():
    """★ 免责说明必须存在且说清三件事。

    合规判断是要担责任的。把"疑似"包装成"违规认定"会让用户拿去做错误决策，
    比不做检查更糟。故免责说明不是摆设，必须包含：
        1. 只覆盖规则库（未命中不代表合规）
        2. 结论是"疑似"而非定性
        3. 不构成法律意见
    """
    d = C.check_compliance({"sections": {"qualifications": [{"text": "x"}]}})["disclaimer"]
    assert "规则匹配" in d
    assert "未命中的条款不代表合规" in d
    assert "疑似" in d
    assert "不构成法律意见" in d


def test_findings_use_suspected_wording_not_conviction():
    """每条命中都必须给出"为什么可疑"与"怎么核实"，而不是直接定性。"""
    result = _check("投标人须为国有企业。")
    f = result["findings"][0]
    assert f["why"]
    assert f["how_to_verify"]
    assert f["evidence"], "必须给出证据片段（原文中命中的部分）"
    assert f["evidence"] in f["clause"], "证据必须来自原文，不能是凭空生成的描述"


def test_finding_carries_clause_and_location():
    result = C.check_compliance(
        {"sections": {"qualifications": [{"text": "投标人须为国有企业。"}]}},
        include_law_basis=False,
    )
    f = result["findings"][0]
    assert f["section"] == "qualifications"
    assert f["item_index"] == 0
    assert "国有企业" in f["clause"]


# ===========================================================================
# 输入边界
# ===========================================================================


def test_empty_parsed_raises_not_silent():
    """空输入必须报错。

    静默返回"未检出问题"会让用户以为**检查过了且没问题**——
    这是最危险的失败模式（把"没检查"伪装成"检查通过"）。
    """
    with pytest.raises(C.ComplianceError):
        C.check_compliance({"sections": {}})
    with pytest.raises(C.ComplianceError):
        C.check_compliance({"fields": {}, "sections": {}})


def test_non_dict_raises():
    with pytest.raises(C.ComplianceError):
        C.check_compliance("not a dict")  # type: ignore[arg-type]


def test_scalar_fields_are_checked_too():
    """标量字段也要检查——"付款方式"这类要求就在标量字段里。"""
    result = C.check_compliance(
        {"fields": {"payment_terms": "中标人须全额垫资施工。"}, "sections": {}},
        include_law_basis=False,
    )
    assert "payment_terms_unfair" in _keys(result)


def test_law_basis_absent_does_not_break_detection():
    """法规语料不可用时，规则检出仍必须完整工作。

    这是有意的降级设计：语料是增强项，不是前提。
    """
    result = C.check_compliance(
        {"sections": {"qualifications": [{"text": "投标人须为国有企业。"}]}},
        include_law_basis=False,
    )
    assert result["findings"]
    assert result["findings"][0]["law_basis"] == []


# ===========================================================================
# 中文数字辅助
# ===========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [("三", 3), ("六", 6), ("十", 10), ("十五", 15), ("三十", 30), ("两", 2), ("99", 99)],
)
def test_cn_num_parsing(text, expected):
    assert C._cn_num(text) == expected


@pytest.mark.parametrize("text", ["", "很多", "abc"])
def test_cn_num_returns_none_on_unparseable(text):
    assert C._cn_num(text) is None


@pytest.mark.parametrize(
    "raw,unit,expected",
    [("5000", "万", 50_000_000), ("1.2", "亿", 120_000_000), ("1000", "", 1000)],
)
def test_to_amount(raw, unit, expected):
    assert C._to_amount(raw, unit) == expected


def test_to_amount_invalid_returns_none():
    assert C._to_amount("abc", "万") is None


def test_public_api_exports():
    for name in ("RULES", "ComplianceError", "check_compliance", "rules_catalog"):
        assert name in C.__all__
