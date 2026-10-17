"""资格审查辅助（⑤）单测。

重点守住三件事，它们都是**会让用户做错决策**的地方：

    1. **「需人工确认」不能省**：信息不足时必须 uncertain，不能判成 unmet。
       把不确定判成"不满足"会让用户白白放弃一个能投的标。
    2. **等级要真的比大小**：一级 > 二级 > 三级，且"二级"与"乙级"属于两套
       命名体系、不可比（硬比会给出没有依据的结论）。
    3. **overall 必须保守**：有 uncertain 就不能报 qualified；gaps 只装硬性缺口，
       加分项没满足不是缺口。

`match_qualification` 是纯函数（不碰数据库 / LLM），本文件里绝大多数用例
只依赖它。真实 PostgreSQL 的档案 CRUD 用例在文件末尾：**连不上库时自动跳过**
（CI 里 conftest 只填占位 env，必然跳过），连得上则真跑并在结束时清理测试数据。
"""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path

import pytest

from src.services import qualification as Q

# 固定参考日期：所有"近三年"之类的窗口都相对它算，否则结果随运行日期漂移
REF = datetime.date(2026, 11, 20)

# 招标文件资格要求样本（取自 logs/test_document_parser.py 造的 tender_sample.pdf）
SAMPLE_REQUIREMENTS = [
    "具有独立承担民事责任的能力，提供营业执照副本。",
    "具有电子与智能化工程专业承包二级及以上资质。",
    "具有有效的 ISO9001 质量管理体系认证证书。",
    "近三年内完成过单项合同金额不低于 500 万元的政务云项目，提供合同及验收报告。",
    "项目负责人须具备信息系统项目管理师（高级）资格。",
    "财务状况良好，提供近一年经审计的财务报告。",
    "未被列入失信被执行人名单、重大税收违法案件当事人名单。",
]


def _parsed(texts=SAMPLE_REQUIREMENTS, mandatory=True):
    """构造归一化后的抽取结果（与 normalize_tender 的输出同构）。"""
    return {
        "fields": {"project_name": "政务云服务采购项目"},
        "sections": {
            "qualifications": [
                {"text": t, "category": "", "is_mandatory": mandatory} for t in texts
            ]
        },
    }


def _req(text: str, mandatory: bool = True, category: str = "") -> dict:
    """把一条条文过一遍真实解析链路，得到可比对的要求字典。"""
    parsed = {
        "sections": {
            "qualifications": [{"text": text, "category": category, "is_mandatory": mandatory}]
        }
    }
    return Q.parse_qualification_requirements(parsed, REF)[0]


def _match(text: str, profile: dict, mandatory: bool = True) -> dict:
    return Q.match_qualification(_req(text, mandatory), profile, REF)


def _status(text: str, profile: dict, mandatory: bool = True) -> str:
    return _match(text, profile, mandatory)["status"]


# 完整且合规的企业档案：样本 7 条要求应全部满足
FULL_PROFILE = {
    "company_name": "安徽智云信息技术有限公司",
    "business_license": "91340100MA2TXXXXXX",
    "qualifications": [
        {
            "name": "电子与智能化工程专业承包",
            "level": "二级",
            "cert_no": "D2340012345",
            "valid_until": "2027-06-30",
        }
    ],
    "certifications": ["ISO9001", "ISO27001"],
    "performance": [{"project": "合肥市政务云平台建设项目", "amount": 5_800_000, "year": 2024}],
    "financial": {"revenue": 120_000_000, "audited": True, "healthy": True},
    "personnel": [{"name": "张工", "cert": "信息系统项目管理师", "level": "高级"}],
    "credit": {"dishonest": False, "tax_violation": False},
}


# ---------------------------------------------------------------------------
# 等级解析与比较（"一级 > 二级"要真的实现，不能只做字符串相等）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,level,system,rank",
    [
        ("电子与智能化工程专业承包一级", "一级", "numeric", 80),
        ("电子与智能化工程专业承包二级及以上", "二级", "numeric", 60),
        ("建筑工程施工总承包三级", "三级", "numeric", 40),
        ("施工总承包特级", "特级", "numeric", 100),
        ("工程设计综合资质", "综合", "numeric", 100),
        ("工程设计甲级资质", "甲级", "grade", 80),
        ("工程监理乙级", "乙级", "grade", 60),
        ("信息系统项目管理师（高级）", "高级", "seniority", 80),
        ("（中级）职称", "中级", "seniority", 60),
    ],
)
def test_parse_level_recognizes_all_systems(text, level, system, rank):
    assert Q.parse_level(text) == (level, system, rank)


def test_parse_level_prefers_longest_token():
    """★ 长词优先：'副高级' 不能被 '高级' 截胡。

    若先匹配到"高级"（秩 80），"副高级"（秩 75）就会错误地满足"正高级"要求。
    """
    assert Q.parse_level("具有副高级职称") == ("副高级", "seniority", 75)
    assert Q.parse_level("具有正高级职称") == ("正高级", "seniority", 100)


def test_parse_level_returns_none_without_level():
    assert Q.parse_level("具有电子与智能化工程专业承包资质") == (None, None, None)


@pytest.mark.parametrize(
    "required,owned,expected",
    [
        ("一级", "特级", True),    # 等级更高 → 满足
        ("二级", "一级", True),
        ("二级", "二级", True),    # 同级 → 满足
        ("二级", "三级", False),   # 等级不足
        ("一级", "二级", False),
        ("特级", "一级", False),
        ("甲级", "乙级", False),
        ("甲级", "甲级", True),
        ("高级", "中级", False),
    ],
)
def test_compare_level(required, owned, expected):
    assert Q.compare_level(required, owned) is expected


def test_compare_level_cross_system_is_undecidable():
    """★ '二级' 与 '乙级' 属于两套命名体系，谁高谁低没有依据 → 必须返回 None。

    返回 False 会让企业白白放弃一个可能能投的标。
    """
    assert Q.compare_level("二级", "乙级") is None
    assert Q.compare_level("甲级", "一级") is None


def test_compare_level_without_requirement_returns_none():
    """要求里没写等级 = 只要求"有这个东西"，比较不适用。"""
    assert Q.compare_level(None, "二级") is None
    assert Q.compare_level("", "二级") is None


def test_compare_level_missing_owned_level_is_unknown():
    assert Q.compare_level("二级", "") is None
    assert Q.compare_level("二级", None) is None


def test_compare_level_falls_back_to_owned_text():
    """等级嵌在证书名里（"一级建造师"）时，档案的 level 字段可以是空的。"""
    assert Q.compare_level("一级", "", owned_text="一级建造师") is True
    assert Q.compare_level("二级", "", owned_text="一级建造师") is True
    assert Q.compare_level("特级", "", owned_text="一级建造师") is False


def test_compare_level_embedded_level_across_systems_is_unknown():
    """要求"高级"（职称序列）、证书名带"一级"（资质序列）→ 不可比，交人工。"""
    assert Q.compare_level("高级", "", owned_text="一级建造师") is None


def test_compare_level_exact_op():
    """'仅限二级' → 一级也不满足。"""
    assert Q.compare_level("二级", "一级", op="==") is False
    assert Q.compare_level("二级", "二级", op="==") is True


# ---------------------------------------------------------------------------
# 企业资质（名称 + 等级）
# ---------------------------------------------------------------------------

REQ_QUAL = "具有电子与智能化工程专业承包二级及以上资质。"


def test_qualification_met_same_level():
    profile = {"qualifications": [{"name": "电子与智能化工程专业承包", "level": "二级"}]}
    r = _match(REQ_QUAL, profile)
    assert r["status"] == Q.STATUS_MET
    assert "二级" in r["reason"]


def test_qualification_met_higher_level():
    """一级企业满足"二级及以上"——这是招投标行业惯例，也是本模块必须实现比较的原因。"""
    profile = {"qualifications": [{"name": "电子与智能化工程专业承包", "level": "一级"}]}
    assert _status(REQ_QUAL, profile) == Q.STATUS_MET


def test_qualification_unmet_lower_level():
    profile = {"qualifications": [{"name": "电子与智能化工程专业承包", "level": "三级"}]}
    r = _match(REQ_QUAL, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "等级不足" in r["reason"]


def test_qualification_uncertain_when_level_missing():
    """★ 档案里写了资质但没写等级 → 需人工确认证书原件，不能判 unmet。"""
    profile = {"qualifications": [{"name": "电子与智能化工程专业承包"}]}
    r = _match(REQ_QUAL, profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "等级" in r["reason"]


def test_qualification_unmet_name_absent():
    profile = {"qualifications": [{"name": "建筑工程施工总承包", "level": "一级"}]}
    r = _match(REQ_QUAL, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "没有" in r["reason"]
    # 提示里要给出整改动作，否则用户只会看到"不满足"却不知道下一步做什么
    assert "补录" in r["evidence"]


def test_qualification_uncertain_on_empty_profile():
    """空档案不能判 unmet——"没填"与"没有"是两件事。"""
    r = _match(REQ_QUAL, {})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "未登记" in r["reason"]


def test_qualification_uncertain_on_weak_name_match():
    """要求写得太笼统（"施工总承包"）时，命中可能只是共享了通用词 → 需人工确认。"""
    assert Q.name_match("施工总承包", "建筑工程施工总承包") == "weak"
    profile = {"qualifications": [{"name": "建筑工程施工总承包", "level": "一级"}]}
    assert _status("具有施工总承包一级资质。", profile) == Q.STATUS_UNCERTAIN


def test_qualification_met_without_level_requirement():
    """条文只要求"有该资质"，档案没写等级也应判满足。"""
    profile = {"qualifications": [{"name": "电子与智能化工程专业承包"}]}
    assert _status("具有电子与智能化工程专业承包资质。", profile) == Q.STATUS_MET


def test_qualification_met_ignores_suffix_noise():
    """档案名称带后缀（"…资质"/"（二级）"）不影响匹配。"""
    assert Q.name_match("电子与智能化工程专业承包", "电子与智能化工程专业承包资质（二级）") != "none"


def test_qualification_unmet_when_certificate_expired():
    """过期的资质证书在评标时按无效处理——这条必须报出来。"""
    profile = {
        "qualifications": [
            {"name": "电子与智能化工程专业承包", "level": "二级", "valid_until": "2025-01-31"}
        ]
    }
    r = _match(REQ_QUAL, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "过期" in r["reason"]


def test_qualification_met_when_certificate_still_valid():
    profile = {
        "qualifications": [
            {"name": "电子与智能化工程专业承包", "level": "二级", "valid_until": "2027-06-30"}
        ]
    }
    assert _status(REQ_QUAL, profile) == Q.STATUS_MET


# ---------------------------------------------------------------------------
# 体系认证
# ---------------------------------------------------------------------------

REQ_ISO = "具有有效的 ISO9001 质量管理体系认证证书。"


def test_certification_met():
    assert _status(REQ_ISO, {"certifications": ["ISO9001"]}) == Q.STATUS_MET


def test_certification_met_via_chinese_alias():
    """招标文件写 ISO 号、档案写中文体系名（或反过来），都算同一个证书。"""
    assert _status(REQ_ISO, {"certifications": ["质量管理体系认证"]}) == Q.STATUS_MET
    assert _status("具有质量管理体系认证证书。", {"certifications": ["ISO9001"]}) == Q.STATUS_MET


def test_certification_met_fullwidth_text():
    assert Q.extract_certifications("ＩＳＯ９００１认证") == ["ISO9001"]


def test_certification_unmet_when_absent():
    r = _match(REQ_ISO, {"certifications": ["ISO14001"]})
    assert r["status"] == Q.STATUS_UNMET
    assert "ISO9001" in r["reason"]


def test_certification_uncertain_on_empty_profile():
    """★ 档案里一条认证都没填 → 需人工确认，不能判"没有 ISO9001"。"""
    assert _status(REQ_ISO, {}) == Q.STATUS_UNCERTAIN
    assert _status(REQ_ISO, {"certifications": []}) == Q.STATUS_UNCERTAIN


def test_certification_unmet_when_expired():
    profile = {"certifications": [{"name": "ISO9001", "valid_until": "2024-12-31"}]}
    r = _match(REQ_ISO, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "过期" in r["reason"]


def test_certification_uncertain_when_name_unrecognizable():
    """条文里认不出具体认证名 → 无法比对，只能人工确认。"""
    assert _status("具有有效的认证证书。", {"certifications": ["ISO9001"]}) == Q.STATUS_UNCERTAIN


def test_certification_denbao_levels_are_different():
    """等保二级 ≠ 等保三级（不同测评结论，不能互相顶替）。"""
    assert Q.extract_certifications("须通过等保三级测评") == ["等保三级"]
    assert _status("须具备等保三级证书。", {"certifications": ["等保二级"]}) == Q.STATUS_UNMET
    assert _status("须具备等保三级证书。", {"certifications": ["等保三级"]}) == Q.STATUS_MET


# ---------------------------------------------------------------------------
# 业绩（金额 / 年限 / 项目类型）
# ---------------------------------------------------------------------------

REQ_PERF = "近三年内完成过单项合同金额不低于 500 万元的政务云项目，提供合同及验收报告。"


def test_performance_amount_parsed_to_yuan():
    """'不低于 500 万元' 必须解析成 5000000，而不是 500。"""
    c = _req(REQ_PERF)["constraints"]
    assert c["amount"] == 5_000_000
    assert c["years"] == 3
    assert c["since_year"] == 2023  # 2026 - 3
    assert c["keyword"] == "政务云"


def test_performance_met():
    profile = {"performance": [{"project": "某市政务云平台项目", "amount": 5_800_000, "year": 2024}]}
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_MET
    assert "5,800,000" in r["evidence"]


def test_performance_met_when_amount_exactly_at_threshold():
    profile = {"performance": [{"project": "政务云项目", "amount": 5_000_000, "year": 2024}]}
    assert _status(REQ_PERF, profile) == Q.STATUS_MET


def test_performance_unmet_when_amount_too_low():
    profile = {"performance": [{"project": "政务云项目", "amount": 4_000_000, "year": 2024}]}
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "业绩金额不达标" in r["reason"]


def test_performance_unmet_when_out_of_window():
    """2020 年的业绩不在"近三年"（2023 起）内。"""
    profile = {"performance": [{"project": "政务云项目", "amount": 8_000_000, "year": 2020}]}
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "时间窗口" in r["reason"]


def test_performance_window_depends_on_reference_date():
    """★ 同一个档案、同一条要求，换参考日期结论不同——证明窗口是相对"当前"算的。

    项目数据本身是 2023 年前后的，但"近三年"永远相对**投标这天**：
    用数据集年份去推算窗口会得出完全错误的结论。
    """
    profile = {"performance": [{"project": "政务云项目", "amount": 8_000_000, "year": 2024}]}
    assert Q.match_qualification(_req(REQ_PERF), profile, datetime.date(2026, 11, 20))["status"] == Q.STATUS_MET
    assert Q.match_qualification(_req(REQ_PERF), profile, datetime.date(2030, 1, 1))["status"] == Q.STATUS_UNMET


def test_performance_uncertain_when_year_missing():
    """★ 业绩没写年份 → 无法判断是否在窗口内 → 需人工确认（不能判不满足）。"""
    profile = {"performance": [{"project": "政务云项目", "amount": 8_000_000}]}
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "年份" in r["reason"]


def test_performance_uncertain_on_empty_profile():
    """★ 空业绩档案 → uncertain，不是 unmet。"""
    r = _match(REQ_PERF, {})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "业绩档案为空" in r["reason"]


def test_performance_uncertain_when_keyword_not_found():
    """项目类型关键词是猜出来的：一条都没命中时不能断言"没有同类业绩"。"""
    profile = {"performance": [{"project": "某办公楼装修工程", "amount": 9_000_000, "year": 2024}]}
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "政务云" in r["reason"]


def test_performance_uncertain_when_only_cumulative_meets():
    """★ 单项都不够、累计够 → uncertain（"单项合同"与"累计业绩"是两种口径）。"""
    profile = {
        "performance": [
            {"project": "政务云项目一", "amount": 3_000_000, "year": 2024},
            {"project": "政务云项目二", "amount": 3_000_000, "year": 2025},
        ]
    }
    r = _match(REQ_PERF, profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "累计" in r["reason"]


def test_performance_uncertain_when_no_threshold_parseable():
    """业绩条文里解析不出金额/数量 → 人工确认，而不是猜一个门槛。"""
    assert _status("近三年具有类似项目业绩。", {"performance": []}) == Q.STATUS_UNCERTAIN


def test_performance_count_threshold_met():
    profile = {
        "performance": [
            {"project": "政务云项目一", "amount": 1_000_000, "year": 2024},
            {"project": "政务云项目二", "amount": 1_000_000, "year": 2025},
        ]
    }
    assert _status("近三年至少完成 2 个政务云项目。", profile) == Q.STATUS_MET


def test_performance_amount_threshold_ignores_non_amount_numbers():
    """★ 'ISO9001' 这类标准号里的数字不能被当成金额（实测踩过的坑）。"""
    amount, _ = Q.parse_amount("具有有效的 ISO9001 质量管理体系认证证书")
    assert amount is None
    assert Q.parse_amount("不低于 500 万元")[0] == 5_000_000
    assert Q.parse_amount("不少于500万")[0] == 5_000_000
    assert Q.parse_amount("不低于 1.2 亿元")[0] == 120_000_000


# ---------------------------------------------------------------------------
# 财务
# ---------------------------------------------------------------------------

REQ_FIN = "财务状况良好，提供近一年经审计的财务报告。"


def test_financial_met_when_audited():
    assert _status(REQ_FIN, {"financial": {"audited": True}}) == Q.STATUS_MET


def test_financial_unmet_when_not_audited():
    r = _match(REQ_FIN, {"financial": {"audited": False}})
    assert r["status"] == Q.STATUS_UNMET
    assert "审计" in r["reason"]


def test_financial_uncertain_when_audited_unknown():
    """★ 档案没写是否审计 → 需人工确认（投标要交审计报告，判错了代价很大）。"""
    r = _match(REQ_FIN, {"financial": {"revenue": 1}})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "audited" in r["evidence"]


def test_financial_uncertain_on_empty_profile():
    assert _status(REQ_FIN, {}) == Q.STATUS_UNCERTAIN


def test_financial_revenue_met():
    profile = {"financial": {"revenue": 120_000_000, "audited": True}}
    assert _status("近三年营业收入不低于 1000 万元。", profile) == Q.STATUS_MET


def test_financial_revenue_unmet():
    profile = {"financial": {"revenue": 8_000_000, "audited": True}}
    r = _match("近三年营业收入不低于 1000 万元。", profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "营业收入不足" in r["reason"]


def test_financial_revenue_uncertain_when_missing():
    profile = {"financial": {"audited": True}}
    r = _match("近三年营业收入不低于 1000 万元。", profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "revenue" in r["evidence"]


def test_financial_registered_capital():
    """注册资本是另一个字段，不能拿营业收入顶替。"""
    assert _status("注册资本不低于 1000 万元。", {"registered_capital": 20_000_000}) == Q.STATUS_MET
    assert _status("注册资本不低于 1000 万元。", {"financial": {"revenue": 90_000_000}}) == Q.STATUS_UNCERTAIN
    assert _status("注册资本不低于 1000 万元。", {"registered_capital": 5_000_000}) == Q.STATUS_UNMET


def test_financial_healthy_alone():
    """只写"财务状况良好"、没有任何可判指标时：档案填了 healthy 才给结论。"""
    text = "投标人财务状况良好，无亏损。"
    assert _status(text, {"financial": {"healthy": True}}) == Q.STATUS_MET
    assert _status(text, {"financial": {"healthy": False}}) == Q.STATUS_UNMET
    assert _status(text, {"financial": {}}) == Q.STATUS_UNCERTAIN


# ---------------------------------------------------------------------------
# 人员
# ---------------------------------------------------------------------------

REQ_PERSON = "项目负责人须具备信息系统项目管理师（高级）资格。"


def test_personnel_met():
    profile = {"personnel": [{"name": "张工", "cert": "信息系统项目管理师", "level": "高级"}]}
    r = _match(REQ_PERSON, profile)
    assert r["status"] == Q.STATUS_MET
    assert "张工" in r["evidence"]


def test_personnel_unmet_when_cert_absent():
    profile = {"personnel": [{"name": "李工", "cert": "二级建造师", "level": "中级"}]}
    r = _match(REQ_PERSON, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "信息系统项目管理师" in r["reason"]


def test_personnel_unmet_when_level_lower():
    profile = {"personnel": [{"name": "王工", "cert": "信息系统项目管理师", "level": "中级"}]}
    r = _match(REQ_PERSON, profile)
    assert r["status"] == Q.STATUS_UNMET
    assert "等级不达标" in r["reason"]


def test_personnel_uncertain_when_level_missing():
    """★ 有人有证但没写等级 → 需人工核对证书，不能判不满足。"""
    profile = {"personnel": [{"name": "赵工", "cert": "信息系统项目管理师"}]}
    r = _match(REQ_PERSON, profile)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "等级" in r["reason"]


def test_personnel_uncertain_on_empty_profile():
    r = _match(REQ_PERSON, {})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "人员档案为空" in r["reason"]


def test_personnel_met_when_level_embedded_in_cert_name():
    """'一级建造师' 本身就是等级信息，档案 level 字段为空也应判满足。"""
    profile = {"personnel": [{"name": "孙工", "cert": "一级建造师"}]}
    assert _status("项目负责人须具备一级建造师资格。", profile) == Q.STATUS_MET


def test_personnel_met_among_multiple_people():
    """只要有一个人满足即可（招标文件要求的是"拟派项目负责人"）。"""
    profile = {
        "personnel": [
            {"name": "李工", "cert": "信息系统项目管理师", "level": "中级"},
            {"name": "张工", "cert": "信息系统项目管理师", "level": "高级"},
        ]
    }
    assert _status(REQ_PERSON, profile) == Q.STATUS_MET


def test_personnel_count_threshold():
    text = "项目团队须配备不少于 2 名高级工程师。"
    two = {
        "personnel": [
            {"name": "A", "cert": "高级工程师", "level": "高级"},
            {"name": "B", "cert": "高级工程师", "level": "高级"},
        ]
    }
    one = {"personnel": [{"name": "A", "cert": "高级工程师", "level": "高级"}]}
    assert _status(text, two) == Q.STATUS_MET
    assert _status(text, one) == Q.STATUS_UNMET


# ---------------------------------------------------------------------------
# 信用 / 主体资格
# ---------------------------------------------------------------------------

REQ_CREDIT = "未被列入失信被执行人名单、重大税收违法案件当事人名单。"


def test_credit_met():
    assert _status(REQ_CREDIT, {"credit": {"dishonest": False, "tax_violation": False}}) == Q.STATUS_MET


def test_credit_unmet_on_dishonest():
    r = _match(REQ_CREDIT, {"credit": {"dishonest": True, "tax_violation": False}})
    assert r["status"] == Q.STATUS_UNMET
    assert "dishonest" in r["reason"]


def test_credit_unmet_on_tax_violation():
    r = _match(REQ_CREDIT, {"credit": {"dishonest": False, "tax_violation": True}})
    assert r["status"] == Q.STATUS_UNMET


def test_credit_uncertain_when_key_missing():
    """★ 只填了失信、没填税收违法 → 该项需人工确认，不能当作"没有"。"""
    r = _match(REQ_CREDIT, {"credit": {"dishonest": False}})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "tax_violation" in r["reason"]


def test_credit_uncertain_on_empty_profile():
    assert _status(REQ_CREDIT, {}) == Q.STATUS_UNCERTAIN


def test_credit_parsed_items():
    c = _req(REQ_CREDIT)["constraints"]
    assert set(c["items"]) == {"dishonest", "tax_violation"}
    assert c["expected"] is False  # "未被列入" = 期望值为 False


REQ_BASIC = "具有独立承担民事责任的能力，提供营业执照副本。"


def test_basic_met_with_license():
    assert _status(REQ_BASIC, {"business_license": "91340100MA2TXXXXXX"}) == Q.STATUS_MET


def test_basic_uncertain_with_only_company_name():
    r = _match(REQ_BASIC, {"company_name": "某某有限公司"})
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "营业执照" in r["evidence"]


def test_basic_uncertain_on_empty_profile():
    assert _status(REQ_BASIC, {}) == Q.STATUS_UNCERTAIN


# ---------------------------------------------------------------------------
# 规则覆盖不到的条文 → 一律 uncertain
# ---------------------------------------------------------------------------


def test_other_kind_is_always_uncertain():
    item = _req("投标人须在合肥市设有常驻服务机构。")
    assert item["kind"] == Q.KIND_OTHER
    r = Q.match_qualification(item, FULL_PROFILE, REF)
    assert r["status"] == Q.STATUS_UNCERTAIN
    assert "人工核对" in r["evidence"]


def test_unknown_category_falls_back_to_other():
    assert _req("详见招标文件附件说明。")["kind"] == Q.KIND_OTHER


def test_category_used_as_fallback():
    """关键词分不出来时用抽取给的 category 兜底。"""
    item = _req("提供相关证明材料。", category="业绩要求")
    assert item["kind"] == Q.KIND_PERFORMANCE


# ---------------------------------------------------------------------------
# match_qualification 的契约（纯函数、输入容忍）
# ---------------------------------------------------------------------------


def test_match_qualification_accepts_bare_raw_text():
    """只给 raw_text 也能判（内部会现解析）——便于单条试算与调试。"""
    r = Q.match_qualification({"raw_text": REQ_ISO}, {"certifications": ["ISO9001"]}, REF)
    assert r["status"] == Q.STATUS_MET


def test_match_qualification_returns_three_keys():
    r = _match(REQ_ISO, {"certifications": ["ISO9001"]})
    assert set(r) == {"status", "reason", "evidence"}
    assert r["status"] in (Q.STATUS_MET, Q.STATUS_UNMET, Q.STATUS_UNCERTAIN)
    assert isinstance(r["reason"], str) and r["reason"]


def test_match_qualification_rejects_non_dict():
    with pytest.raises(Q.QualificationError):
        Q.match_qualification("具有 ISO9001", {}, REF)  # type: ignore[arg-type]


def test_match_qualification_rejects_empty_text():
    with pytest.raises(Q.QualificationError):
        Q.match_qualification({"raw_text": "   "}, {}, REF)


def test_match_qualification_accepts_none_profile():
    """profile=None 与 {} 等价，且不能抛异常。"""
    assert Q.match_qualification(_req(REQ_ISO), None, REF)["status"] == Q.STATUS_UNCERTAIN


def test_match_qualification_is_pure_no_database(monkeypatch):
    """★ 纯函数契约：有人把 `_dsn` 换成会爆炸的实现，判定依然要能跑完。"""

    def _boom():
        raise AssertionError("match_qualification 不应触碰数据库")

    monkeypatch.setattr(Q, "_dsn", _boom)
    assert Q.match_qualification(_req(REQ_ISO), {"certifications": ["ISO9001"]}, REF)["status"] == Q.STATUS_MET


def test_match_qualification_never_raises_on_weird_profile():
    """档案字段类型写错（列表里塞字符串/数字）也不应把整份报告带崩。"""
    weird = {"qualifications": ["电子与智能化工程专业承包"], "personnel": [123], "financial": "n/a"}
    r = Q.match_qualification(_req(REQ_QUAL), weird, REF)
    assert r["status"] in (Q.STATUS_MET, Q.STATUS_UNMET, Q.STATUS_UNCERTAIN)


# ---------------------------------------------------------------------------
# parse_qualification_requirements
# ---------------------------------------------------------------------------


def test_parse_accepts_full_extract_result():
    """`extract_from_document` 的完整输出（{"parsed": ..., "meta": ...}）也要能吃。"""
    full = {"parsed": _parsed(), "meta": {"filename": "x.pdf"}}
    assert len(Q.parse_qualification_requirements(full, REF)) == len(SAMPLE_REQUIREMENTS)


def test_parse_accepts_plain_string_list():
    items = Q.parse_qualification_requirements(["具有 ISO9001 认证。", "  ", "具有营业执照。"], REF)
    assert [i["raw_text"] for i in items] == ["具有 ISO9001 认证。", "具有营业执照。"]
    assert all(i["is_mandatory"] for i in items)  # 缺省按硬性处理


def test_parse_keeps_raw_text_and_mandatory_flag():
    items = Q.parse_qualification_requirements(_parsed(), REF)
    assert items[0]["raw_text"] == SAMPLE_REQUIREMENTS[0]
    assert all(i["is_mandatory"] is True for i in items)


def test_parse_string_false_is_optional():
    """`is_mandatory` 由 JSON 来时可能是字符串，'否'/'false' 要判成加分项。"""
    parsed = {
        "sections": {
            "qualifications": [
                {"text": "具有 ISO27001 认证。", "is_mandatory": "否"},
                {"text": "具有 CMMI5 认证。", "is_mandatory": "false"},
                {"text": "具有 ITSS 认证。", "is_mandatory": "是"},
            ]
        }
    }
    flags = [i["is_mandatory"] for i in Q.parse_qualification_requirements(parsed, REF)]
    assert flags == [False, False, True]


def test_parse_classifies_sample_requirements():
    """样本 7 条的分类必须一次到位（分类错了规则就全错）。"""
    kinds = [i["kind"] for i in Q.parse_qualification_requirements(_parsed(), REF)]
    assert kinds == [
        Q.KIND_BASIC,
        Q.KIND_QUALIFICATION,
        Q.KIND_CERTIFICATION,
        Q.KIND_PERFORMANCE,
        Q.KIND_PERSONNEL,
        Q.KIND_FINANCIAL,
        Q.KIND_CREDIT,
    ]


def test_parse_strips_numbering_from_qualification_name():
    item = _req("2. 具有电子与智能化工程专业承包二级及以上资质。")
    assert item["constraints"]["name"] == "电子与智能化工程专业承包二级及以上资质。"
    assert item["constraints"]["core_name"] == "电子与智能化工程专业承包"


def test_parse_handles_empty_input():
    assert Q.parse_qualification_requirements({}, REF) == []
    assert Q.parse_qualification_requirements(None, REF) == []
    assert Q.parse_qualification_requirements({"sections": {}}, REF) == []


def test_parse_normalizes_reference_date_inputs():
    """reference_date 收 date / datetime / 字符串。"""
    assert Q.parse_years("近三年", Q._as_date("2026-11-20")) == (3, 2023)
    assert Q._as_date(datetime.datetime(2026, 11, 20, 9, 30)) == REF
    assert Q._as_date("2026-11-20") == REF


def test_parse_rejects_bad_reference_date():
    with pytest.raises(Q.QualificationError):
        Q._as_date("去年")


# ---------------------------------------------------------------------------
# check_qualifications：overall 判定必须保守
# ---------------------------------------------------------------------------


def test_overall_qualified_when_all_met():
    res = Q.check_qualifications(_parsed(), FULL_PROFILE, REF)
    assert res["overall"] == Q.OVERALL_QUALIFIED
    assert res["summary"]["met"] == 7
    assert res["summary"]["unmet"] == 0
    assert res["gaps"] == []


def test_overall_unqualified_on_mandatory_unmet():
    profile = dict(FULL_PROFILE, certifications=["ISO14001"])
    res = Q.check_qualifications(_parsed(), profile, REF)
    assert res["overall"] == Q.OVERALL_UNQUALIFIED
    assert len(res["gaps"]) == 1
    assert "ISO9001" in res["gaps"][0]["reason"]


def test_overall_need_review_when_uncertain():
    """★ 有 uncertain 就不许报 qualified——这是本模块存在的理由。"""
    profile = {k: v for k, v in FULL_PROFILE.items() if k != "certifications"}
    res = Q.check_qualifications(_parsed(), profile, REF)
    assert res["overall"] == Q.OVERALL_NEED_REVIEW
    assert res["summary"]["uncertain"] >= 1


def test_overall_unqualified_beats_uncertain():
    """既有硬性 unmet 又有 uncertain 时，结论取更严重的 unqualified。"""
    profile = dict(FULL_PROFILE, certifications=["ISO14001"], personnel=[])
    res = Q.check_qualifications(_parsed(), profile, REF)
    assert res["overall"] == Q.OVERALL_UNQUALIFIED


def test_overall_need_review_when_optional_uncertain():
    """加分项判不了同样不能报 qualified（不确定就是不不确定）。"""
    parsed = {
        "sections": {
            "qualifications": [
                {"text": "具有 ISO9001 认证。", "is_mandatory": True},
                {"text": "在合肥市设有常驻服务机构。", "is_mandatory": False},
            ]
        }
    }
    res = Q.check_qualifications(parsed, {"certifications": ["ISO9001"]}, REF)
    assert res["overall"] == Q.OVERALL_NEED_REVIEW
    assert res["gaps"] == []  # 硬性要求全满足，所以没有缺口


def test_optional_unmet_is_not_a_gap():
    """★ 加分项没满足不进 gaps、不影响 overall——否则用户会以为标投不了。"""
    parsed = {
        "sections": {
            "qualifications": [
                {"text": "具有 ISO9001 认证。", "is_mandatory": True},
                {"text": "具有 CMMI5 认证。", "is_mandatory": False},
            ]
        }
    }
    res = Q.check_qualifications(parsed, {"certifications": ["ISO9001"]}, REF)
    assert res["overall"] == Q.OVERALL_QUALIFIED
    assert res["gaps"] == []
    assert res["summary"]["optional_unmet"] == 1
    # 但它仍要出现在 requirements 里，供用户参考
    optional = [r for r in res["requirements"] if not r["is_mandatory"]]
    assert optional and optional[0]["status"] == Q.STATUS_UNMET


def test_empty_profile_yields_uncertain_not_unmet():
    """★ 空档案必须得到大量 uncertain 而不是全 unmet。

    这是"需人工确认"存在的意义：用户什么都没填时，系统没有资格说"不满足"。
    """
    res = Q.check_qualifications(_parsed(), {}, REF)
    assert res["overall"] == Q.OVERALL_NEED_REVIEW
    assert res["summary"]["unmet"] == 0
    assert res["summary"]["uncertain"] >= 5
    assert res["gaps"] == []
    # 每条 uncertain 都必须说清"去档案的哪个字段补什么"，否则用户无从下手
    assert all("档案" in r["evidence"] for r in res["requirements"])


def test_gaps_only_contain_mandatory_unmet():
    profile = {
        "company_name": "某某公司",
        "qualifications": [{"name": "建筑工程施工总承包", "level": "三级"}],
        "certifications": ["ISO14001"],
        "performance": [{"project": "装修工程", "amount": 1_000_000, "year": 2022}],
        "financial": {"revenue": 1_000_000, "audited": False},
        "personnel": [{"name": "李工", "cert": "二级建造师", "level": "中级"}],
        "credit": {"dishonest": True, "tax_violation": False},
    }
    res = Q.check_qualifications(_parsed(), profile, REF)
    assert res["overall"] == Q.OVERALL_UNQUALIFIED
    gap_texts = {g["raw_text"] for g in res["gaps"]}
    assert SAMPLE_REQUIREMENTS[1] in gap_texts   # 资质不符
    assert SAMPLE_REQUIREMENTS[2] in gap_texts   # 缺 ISO9001
    assert SAMPLE_REQUIREMENTS[5] in gap_texts   # 无审计报告
    assert SAMPLE_REQUIREMENTS[6] in gap_texts   # 失信
    # 业绩那条是 uncertain（项目类型关键词未命中）→ 不是缺口
    assert SAMPLE_REQUIREMENTS[3] not in gap_texts
    assert all(g["kind"] and g["reason"] and g["suggestion"] for g in res["gaps"])


def test_summary_counts_are_consistent():
    res = Q.check_qualifications(_parsed(), FULL_PROFILE, REF)
    s = res["summary"]
    assert s["total"] == len(res["requirements"]) == 7
    assert s["met"] + s["unmet"] + s["uncertain"] == s["total"]
    assert s["mandatory_total"] == 7
    assert s["mandatory_unmet"] == 0
    assert s["reference_date"] == "2026-11-20"
    assert sum(b["total"] for b in s["by_kind"].values()) == s["total"]
    assert s["conclusion"] and s["caveats"]


def test_summary_conclusion_mentions_counts():
    res = Q.check_qualifications(_parsed(), FULL_PROFILE, REF)
    assert "7" in res["summary"]["conclusion"]
    profile = dict(FULL_PROFILE, credit={"dishonest": True, "tax_violation": False})
    res2 = Q.check_qualifications(_parsed(), profile, REF)
    assert "1 项硬性" in res2["summary"]["conclusion"]


def test_requirements_keep_status_and_label():
    res = Q.check_qualifications(_parsed(), FULL_PROFILE, REF)
    for r in res["requirements"]:
        assert r["status"] in Q.STATUS_LABELS
        assert r["status_label"] == Q.STATUS_LABELS[r["status"]]
        assert r["raw_text"] and r["evidence"] != "" or r["status"] == Q.STATUS_MET


def test_caveats_must_state_the_reference_date():
    """结论必须自带"近三年按哪天算"的说明，否则用户无法复核窗口口径。"""
    res = Q.check_qualifications(_parsed(), FULL_PROFILE, REF)
    caveats = " ".join(res["summary"]["caveats"])
    assert "2026-11-20" in caveats
    assert "需人工确认" in caveats
    assert "废标条款" in caveats  # 说明结论的覆盖范围只到资格要求章


def test_check_handles_absent_qualifications_section():
    res = Q.check_qualifications({"fields": {}, "sections": {}}, FULL_PROFILE, REF)
    assert res["overall"] == Q.OVERALL_QUALIFIED  # 没有要求 = 没有缺口
    assert res["requirements"] == []


# ---------------------------------------------------------------------------
# 企业资质档案存取（真 PostgreSQL；连不上自动跳过）
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _real_dsn() -> dict:
    """从项目 .env 读真实连接参数。

    刻意不走 `os.environ`：conftest 里填的是占位值（localhost/test/test），
    用它去连真库必然失败。这里直接读 .env 文件，**不改动全局环境**，
    因此不会影响其他测试模块。
    """
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover
        dotenv_values = None  # type: ignore[assignment]

    values: dict = {}
    env_file = PROJECT_ROOT / ".env"
    if dotenv_values and env_file.exists():
        values = dotenv_values(env_file)

    def pick(key: str, default: str = "") -> str:
        raw = values.get(key) or os.getenv(key) or default
        return str(raw).strip()

    return {
        "host": pick("POSTGRES_HOST"),
        "port": pick("POSTGRES_PORT", "5432"),
        "user": pick("POSTGRES_USER", "postgres"),
        # 密码末尾可能带 \r（Windows 上编辑 .env 常见），必须清掉
        "password": pick("POSTGRES_PASSWORD").rstrip("\r\n"),
        "dbname": pick("POSTGRES_DB", "chatbot"),
    }


@pytest.fixture
def live_db(monkeypatch):
    """连真库的夹具：不可用则 skip，可用则清掉本次测试写入的档案。"""
    import psycopg

    dsn = _real_dsn()
    if not dsn["host"]:
        pytest.skip("未配置 POSTGRES_HOST，跳过数据库集成用例")
    try:
        with psycopg.connect(**dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # 库没起 / 密码不对 / CI 无库
        pytest.skip(f"PostgreSQL 不可用（{type(exc).__name__}），跳过数据库集成用例")

    monkeypatch.setattr(Q, "_dsn", lambda: dsn)
    name = f"单测临时档案-{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        try:
            Q.delete_profile(name)
        except Exception:  # pragma: no cover - 清理失败不应让测试变红
            pass


def test_profile_roundtrip(live_db):
    """保存 → 读回：档案内容逐字段一致（_meta 是读取时附加的元信息）。"""
    payload = {
        "company_name": "单测科技有限公司",
        "qualifications": [{"name": "电子与智能化工程专业承包", "level": "二级"}],
        "certifications": ["ISO9001"],
        "performance": [{"project": "某政务云项目", "amount": 5_800_000, "year": 2024}],
        "financial": {"revenue": 120_000_000, "audited": True},
        "personnel": [{"name": "张工", "cert": "信息系统项目管理师", "level": "高级"}],
        "credit": {"dishonest": False, "tax_violation": False},
    }
    saved = Q.save_profile(live_db, payload)
    assert saved["name"] == live_db
    assert saved["created"] is True
    assert saved["updated_at"]

    loaded = Q.load_profile(live_db)
    meta = loaded.pop("_meta")
    assert meta["name"] == live_db
    assert loaded == payload


def test_save_profile_upserts(live_db):
    """同名重复保存 = 更新，不新增行，且 updated_at 会变。"""
    first = Q.save_profile(live_db, {"company_name": "第一版"})
    second = Q.save_profile(live_db, {"company_name": "第二版", "certifications": ["ISO27001"]})
    assert first["created"] is True
    assert second["created"] is False
    assert Q.load_profile(live_db)["company_name"] == "第二版"
    names = [p["name"] for p in Q.list_profiles()]
    assert names.count(live_db) == 1


def test_list_profiles_summary(live_db):
    Q.save_profile(
        live_db,
        {
            "company_name": "单测科技有限公司",
            "qualifications": [{"name": "A"}, {"name": "B"}],
            "certifications": ["ISO9001", "ISO27001"],
            "performance": [{"project": "P"}],
            "personnel": [],
        },
    )
    row = next(p for p in Q.list_profiles() if p["name"] == live_db)
    assert row["company_name"] == "单测科技有限公司"
    assert row["qualification_count"] == 2
    assert row["certification_count"] == 2
    assert row["performance_count"] == 1
    assert row["personnel_count"] == 0
    assert row["created_at"] and row["updated_at"]


def test_list_profiles_tolerates_wrong_field_types(live_db):
    """档案里把 qualifications 写成字符串时，列表接口不能崩（jsonb_array_length 会报错）。"""
    Q.save_profile(live_db, {"qualifications": "二级", "certifications": {"a": 1}})
    row = next(p for p in Q.list_profiles() if p["name"] == live_db)
    assert row["qualification_count"] == 0
    assert row["certification_count"] == 0


def test_delete_profile(live_db):
    Q.save_profile(live_db, {"company_name": "待删除"})
    assert Q.delete_profile(live_db) is True
    assert Q.delete_profile(live_db) is False  # 幂等：再删一次不算错
    with pytest.raises(Q.QualificationError) as ei:
        Q.load_profile(live_db)
    assert "不存在" in str(ei.value)


def test_delete_unknown_profile_returns_false(live_db):
    assert Q.delete_profile(f"绝不存在的档案-{uuid.uuid4().hex}") is False


def test_load_unknown_profile_raises_with_hint(live_db):
    with pytest.raises(Q.QualificationError) as ei:
        Q.load_profile(f"绝不存在的档案-{uuid.uuid4().hex}")
    assert "不存在" in str(ei.value)


def test_save_profile_rejects_empty_name(live_db):
    with pytest.raises(Q.QualificationError):
        Q.save_profile("   ", {})


def test_check_qualifications_with_stored_profile(live_db):
    """端到端一致性：从库里读回的档案能直接喂给 check_qualifications。"""
    Q.save_profile(live_db, FULL_PROFILE)
    res = Q.check_qualifications(_parsed(), Q.load_profile(live_db), REF)
    assert res["overall"] == Q.OVERALL_QUALIFIED
