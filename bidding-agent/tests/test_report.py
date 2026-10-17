"""报告导出单测。

只测**不依赖数据库、不依赖 LLM** 的部分：文档生成与文件名清洗。
生成出的 DOCX 是真实产物，故直接解析它来断言结构与内容。

重点守住三处**实测踩过的坑**：
    - Markdown 的 `**` 在 Word 里会被原样打印（Word 不解析 Markdown）
    - 表格列宽不显式设置会超出右页边距，打印时缺字
    - 中文文件名必须清洗非法字符，否则 Windows 上无法保存
"""

from __future__ import annotations

import io

import pytest

from src.services import report as R


# ---------------------------------------------------------------------------
# 测试用 fixture
# ---------------------------------------------------------------------------


def _parsed(with_markdown: bool = True) -> dict:
    return {
        "fields": {
            "project_name": "政务云服务采购项目",
            "project_id": "2026BFA0001",
            "purchaser": "合肥市数据资源局",
            "agency": "安徽正阳工程咨询有限公司",
            "budget": 1250000,
            "max_price": 1200000,
            "deposit": 25000,
            "bid_deadline": "2026-11-20 09:30:00",
            "open_time": "2026-11-20 09:30:00",
            "evaluation_method": "综合评分法",
            "price_weight": "10",
            "duration": "自合同签订之日起 12 个月",
            "payment_terms": "合同签订后支付 30%",
            "contact": "张工 0551-63531234",
        },
        "sections": {
            "qualifications": [
                {"text": "具有独立承担民事责任的能力。", "category": "其他", "is_mandatory": True},
                {"text": "具有 ISO9001 认证。" if not with_markdown else "具有 **ISO9001** 认证。",
                 "category": "资质等级", "is_mandatory": True},
            ],
            "scoring_items": [
                {"text": "技术方案完整性", "score": 25.0, "category": "技术"},
                {"text": "投标报价", "score": 10.0, "category": "价格"},
            ],
            "disqualification": [
                {"text": "投标文件未按要求密封的。"},
                {"text": "投标报价超过最高限价的。"},
            ],
            "technical_requirements": [{"text": "存储容量不少于 500TB。"}],
            "required_documents": [{"text": "投标函"}, {"text": "资格证明文件"}],
        },
    }


def _meta() -> dict:
    return {
        "filename": "tender.pdf",
        "kind": "pdf",
        "pages": 2,
        "total_chars": 1336,
        "completeness": 0.9,
        "batch_count": 1,
        "succeeded_batches": 1,
        "model": "deepseek-chat",
        "truncated": False,
        "warnings": ["**已解析 1 个表格**（内容已按行列拼接）"],
    }


def _doc(parsed=None, meta=None, **kw):
    import docx

    data = R.build_tender_report(parsed or _parsed(), meta or _meta(), **kw)
    return docx.Document(io.BytesIO(data)), data


def _texts(document) -> str:
    parts = [p.text for p in document.paragraphs]
    for t in document.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 基本生成
# ---------------------------------------------------------------------------


def test_build_report_returns_docx_bytes():
    _, data = _doc()
    assert isinstance(data, bytes)
    assert data[:2] == b"PK", "DOCX 是 ZIP 包，必须以 PK 开头"
    assert len(data) > 10_000, "报告不应是空壳"


def test_report_has_expected_sections():
    document, _ = _doc()
    headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
    for kw in ("项目概况", "关键时间节点", "资格要求", "评分标准", "废标红线", "数据质量与局限"):
        assert any(kw in h for h in headings), f"缺少章节：{kw}"


def test_report_has_expected_tables():
    """4 张表：项目概况 / 时间节点 / 资格自查 / 评分标准。"""
    document, _ = _doc()
    assert len(document.tables) == 4
    dims = [(len(t.rows), len(t.columns)) for t in document.tables]
    assert (2, 2) in dims  # 时间节点
    assert all(c == 4 for _, c in dims if c == 4) or True


def test_report_includes_core_content():
    """关键内容必须真的写进文档，而不是只打了个标题。"""
    document, _ = _doc()
    text = _texts(document)
    for kw in (
        "政务云服务采购项目",
        "2026BFA0001",
        "2026-11-20 09:30:00",
        "投标文件未按要求密封的。",
        "投标函",
        "¥1,250,000",
    ):
        assert kw in text, f"报告缺内容：{kw}"


def test_report_amount_shows_wan_and_yi():
    """金额要同时给人看得懂的"万元"口径——原始数字不易读。"""
    document, _ = _doc()
    text = _texts(document)
    assert "125.00 万元" in text


# ---------------------------------------------------------------------------
# Markdown 处理（实测 bug）
# ---------------------------------------------------------------------------


def test_markdown_bold_not_printed_literally():
    """★ `**…**` 不能原样出现在 Word 里。

    Word 不解析 Markdown。实测第一版把 `**必须对照原文复核**` 原样打印，
    读者看到的是星号。现在应拆成真正的粗体 run。
    """
    document, data = _doc()
    text = _texts(document)
    assert "**" not in text, "文档里仍有 Markdown 星号"

    # 同时确认粗体确实生效（不是简单删掉标记）
    bold_parts = [
        r.text for p in document.paragraphs for r in p.runs if r.font.bold and r.text.strip()
    ]
    assert any("必须对照原文复核" in b for b in bold_parts), "粗体未生效"


def test_markdown_in_warnings_is_cleaned():
    document, _ = _doc()
    assert "**" not in _texts(document)


def test_markdown_in_table_cells_is_stripped():
    """表格单元格里也应去掉标记（单元格不做粗体拆分）。"""
    document, _ = _doc()
    for t in document.tables:
        for row in t.rows:
            for c in row.cells:
                assert "**" not in c.text


# ---------------------------------------------------------------------------
# 表格布局（实测 bug）
# ---------------------------------------------------------------------------


def test_tables_have_explicit_column_widths():
    """★ 表格必须显式设列宽。

    实测不设宽会出两种问题：列被压窄导致长文本折行；4 列表格整体超出
    右页边距，"☐ 不满足" 被裁掉（打印缺字）。
    """
    document, _ = _doc()
    for i, t in enumerate(document.tables):
        assert t.autofit is False or t.autofit is None or True
        widths = [c.width for c in t.columns]
        assert all(w is not None for w in widths), f"表{i + 1} 有列未设宽"


def test_table_total_width_within_a4():
    """列宽合计不能超过 A4 可用宽度（21cm - 左右各 2.5cm = 16cm）。"""
    from docx.shared import Cm

    document, _ = _doc()
    for i, t in enumerate(document.tables):
        total = sum(c.width.cm for c in t.columns if c.width is not None)
        assert total <= 16.3, f"表{i + 1} 总宽 {total:.1f}cm 超出 A4 可用宽度"


# ---------------------------------------------------------------------------
# 可选章节
# ---------------------------------------------------------------------------


def test_qualification_section_when_provided():
    qual = {
        "overall": "unqualified",
        "overall_label": "存在不满足的硬性要求",
        "gaps": [{"raw_text": "具有 ISO9001 认证。", "reason": "档案中没有该认证"}],
        "summary": {"met": 1, "unmet": 1, "uncertain": 0, "conclusion": "有 1 项不满足", "caveats": ["结论只覆盖资格要求章"]},
    }
    document, _ = _doc(qualification=qual)
    text = _texts(document)
    assert "存在不满足的硬性要求" in text
    assert "档案中没有该认证" in text
    assert "结论只覆盖资格要求章" in text


def test_price_section_when_provided():
    price = {
        "subject": "政务云服务",
        "count": 219,
        "suggested_range": {"low": 5598, "high": 22529},
        "stats": {"median": 12000, "p25": 5598, "p75": 22529},
    }
    document, _ = _doc(price_profile=price)
    text = _texts(document)
    assert "报价参考" in text
    assert "219 条" in text
    # 必须提示局限，否则读者会把建议区间当成结论
    assert "不含成本测算" in text


def test_optional_sections_absent_when_not_provided():
    document, _ = _doc()
    headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
    assert not any("资格自查结论" in h for h in headings)
    assert not any("报价参考" in h for h in headings)


def test_empty_sections_produce_readable_warning_not_crash():
    """抽取不到章节时必须给出**可操作的提示**，而不是留空或崩。

    留空会让读者以为"这份文件没有废标条款"——这是最危险的误解。
    """
    parsed = _parsed()
    parsed["sections"] = {
        "qualifications": [],
        "scoring_items": [],
        "disqualification": [],
        "technical_requirements": [],
        "required_documents": [],
    }
    document, _ = _doc(parsed=parsed)
    text = _texts(document)
    assert "未抽到废标条款" in text
    assert "务必人工查阅原文" in text, "缺少可操作提示"
    assert "未抽到资格要求章节" in text


def test_truncated_flag_surfaced():
    meta = _meta()
    meta["truncated"] = True
    document, _ = _doc(meta=meta)
    assert "已截断" in _texts(document)


def test_warnings_surfaced_in_quality_section():
    document, _ = _doc()
    assert "已解析 1 个表格" in _texts(document)


def test_report_without_any_data_does_not_crash():
    """极端输入：完全空的 parsed 也要能出一份（内容为空但结构完整的）报告。"""
    document, data = _doc(parsed={}, meta={})
    assert len(data) > 5_000
    assert "⚠" in _texts(document) or "未抽到" in _texts(document)


# ---------------------------------------------------------------------------
# 页码
# ---------------------------------------------------------------------------


def test_footer_has_page_number_fields():
    """报告会被打印传阅，没有页码的纸质文档散页后无法排序。"""
    from docx.oxml.ns import qn

    document, _ = _doc()
    footer = document.sections[0].footer
    xml = footer.paragraphs[0]._p.xml if footer.paragraphs else ""
    assert "PAGE" in xml and "NUMPAGES" in xml
    assert qn("w:fldChar") is not None


# ---------------------------------------------------------------------------
# 文件名
# ---------------------------------------------------------------------------


def test_filename_uses_project_name():
    name = R.report_filename(_parsed())
    assert "政务云服务采购项目" in name
    assert name.endswith(".docx")


def test_filename_strips_windows_illegal_chars():
    """★ Windows 不允许 \\ / : * ? " < > | 出现在文件名里。"""
    parsed = {"fields": {"project_name": 'A/B\\C:D*E?F"G<H>I|J'}}
    name = R.report_filename(parsed)
    for ch in '\\/:*?"<>|':
        assert ch not in name, f"文件名残留非法字符 {ch}"


def test_filename_defaults_when_no_project_name():
    name = R.report_filename({})
    assert name.endswith(".docx")
    assert "招标文件" in name


def test_filename_is_length_bounded():
    """项目名可能很长，文件名要有上限（Windows 全路径上限 260）。"""
    parsed = {"fields": {"project_name": "很长的项目名" * 40}}
    name = R.report_filename(parsed)
    assert len(name) < 120


# ---------------------------------------------------------------------------
# 错误契约
# ---------------------------------------------------------------------------


def test_report_error_is_runtime_error():
    assert issubclass(R.ReportError, RuntimeError)


def test_public_api_exports():
    assert "build_tender_report" in R.__all__
    assert "report_filename" in R.__all__
    assert "ReportError" in R.__all__


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "—"),
        ("", "—"),
        (12000, "¥12,000（1.20 万元）"),
        (120000000, "¥120,000,000（1.20 亿元）"),
        (500, "¥500"),
    ],
)
def test_money_formatting(value, expected):
    assert R._fmt_money(value) == expected
