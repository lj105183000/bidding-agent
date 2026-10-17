"""投标决策报告导出（Word）。

为什么是"决策报告"而不是"数据导出"
--------------------------------
用户拿到分析结果后要**交给领导或团队**，所以要的不是 JSON，而是一份能直接
传阅、能打印、能逐条打勾的文档。故本模块按**投标工作流**组织内容，而不是
按数据表结构：

    项目概况 → 资格自查表 → 评分标准 → 废标红线 → 时间节点
             → 需提交材料(勾选框) → 报价参考 → 数据质量说明

几个刻意的设计
-------------
1. **废标条款排在资格要求之后、最显眼位置**，并用红色标题。
   漏看一条废标条款的代价远大于报价高低。
2. **需提交材料带 ☐ 勾选框**：这份文档的主要用途是打印出来核对，
   没有勾选框就只是一份"说明"。
3. **数据质量与局限单独成章且不省略**：抽取完整度低、有警告时必须让读者
   知道结论的适用范围。把局限藏起来比没有报告更糟。
4. **中文字体显式设置 w:eastAsia**：只设 `run.font.name` 在 Word 里
   中文会回退到默认字体，排版会变形。

依赖 `python-docx`（已在 pyproject.toml 中）。
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# 中文字体。西文用 Calibri、中文用宋体/黑体——
# 只设 font.name 不够，必须同时设 w:eastAsia，否则 Word 对中文会回退到默认字体。
_BODY_FONT = "宋体"
_HEADING_FONT = "黑体"
_MONO_FONT = "Consolas"


class ReportError(RuntimeError):
    """报告生成失败。"""


def _require_docx():
    try:
        import docx  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ReportError("缺少 python-docx，请运行：uv add python-docx") from exc
    return docx


def _set_font(run, name: str, size: float | None = None, bold: bool | None = None, color=None):
    """设置 run 字体，**中英文都设**。

    `w:eastAsia` 必须显式设置：python-docx 的 `font.name` 只作用于西文，
    中文会回退到主题默认字体，导致同一段落中英文字体不一致。
    """
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _fit_table(table, widths_cm: list[float], document=None) -> None:
    """给表格设置**显式列宽**并关闭自动伸缩。

    ★ 为什么必须显式设宽（实测踩坑）
    -----------------------------
    第一版我用 `table.columns[0].width = None` 想让 Word 自适应，结果在
    LibreOffice/Word 里宽度失控：

        - 表1（2 列）第二列被压到很窄，"安徽正阳工程咨询有限公司" 折成 3 行
        - 表3（4 列）整体**超出右页边距**，"☐ 不满足" 被裁掉 → 打印缺字

    正确做法是三件事一起做（缺一不可）：
        1. `table.autofit = False` —— 关掉自动伸缩
        2. 逐**单元格**设 `cell.width` —— python-docx 的 `column.width` 在
           autofit 关闭时不一定生效，必须落到每个 cell
        3. 设置 `tblLayout` 为 `fixed` —— 否则 Word 仍会按内容重算列宽
    """
    from docx.shared import Cm

    table.autofit = False
    try:
        # 固定布局：告诉 Word/LibreOffice 按我们给的宽度渲染，不要按内容重算
        from docx.oxml.ns import qn

        tblPr = table._tbl.tblPr
        layout = tblPr.find(qn("w:tblLayout"))
        if layout is None:
            layout = tblPr.makeelement(qn("w:tblLayout"), {})
            tblPr.append(layout)
        layout.set(qn("w:type"), "fixed")
    except Exception:  # pragma: no cover - 布局属性不可用时仍设单元格宽度
        pass

    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(widths_cm):
                cell.width = Cm(widths_cm[i])
    for i, col in enumerate(table.columns):
        if i < len(widths_cm):
            col.width = Cm(widths_cm[i])


def _add_page_numbers(document) -> None:
    """在页脚加"第 X 页 / 共 Y 页"。

    报告会被打印传阅，没有页码的纸质文档散页后无法排序。
    用域代码（PAGE / NUMPAGES）而非固定数字——文档内容变化时页码要能自动更新。
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    section = document.sections[0]
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def _field(instr: str) -> None:
        run = p.add_run()
        _set_font(run, _BODY_FONT, 9)
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr_el = OxmlElement("w:instrText")
        instr_el.set(qn("xml:space"), "preserve")
        instr_el.text = instr
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr_el)
        run._r.append(end)

    label = p.add_run("第 ")
    _set_font(label, _BODY_FONT, 9)
    _field("PAGE")
    mid = p.add_run(" 页 / 共 ")
    _set_font(mid, _BODY_FONT, 9)
    _field("NUMPAGES")
    tail = p.add_run(" 页")
    _set_font(tail, _BODY_FONT, 9)


def _para(container, text: str, size: float = 10.5, bold: bool = False, font: str | None = None,
          color=None, space_after: float = 4, indent: float = 0):
    """加一个普通段落并统一字体。

    ★ `text` 里的 Markdown 粗体标记（`**…**`）会被**渲染成真粗体**而不是原样打印。
    ------------------------------------------------------------------
    实测踩坑：招标文件抽取结果与我们自己的提示文案里常含 `**必须对照原文复核**`
    这类 Markdown 标记（它们在网页上正常显示，因为前端用 react-markdown 渲染）。
    但 Word **不解析 Markdown**，直接输出会变成字面的星号：

        本报告由 AI 自动抽取生成，废标条款**必须对照原文复核**后再用

    故这里做一次最小解析：把 `**…**` 拆成独立的粗体 run。
    """
    import re as _re

    from docx.shared import Pt

    p = container.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    p.paragraph_format.space_after = Pt(space_after)

    # 按 `**bold**` 切分；奇数段是粗体内容
    parts = _re.split(r"\*\*(.+?)\*\*", text)
    for i, seg in enumerate(parts):
        if not seg:
            continue
        run = p.add_run(seg)
        _set_font(run, font or _BODY_FONT, size, bold or (i % 2 == 1), color)
    if not parts:
        run = p.add_run("")
        _set_font(run, font or _BODY_FONT, size, bold, color)
    return p


def _strip_md(text: str) -> str:
    """去掉 Markdown 标记，供**表格单元格**使用（单元格里不做粗体拆分）。

    表格内混排粗体会让列宽计算复杂化且视觉杂乱，故直接去标记。
    """
    import re as _re

    return _re.sub(r"\*\*(.+?)\*\*", r"\1", str(text or ""))


def _heading(document, text: str, level: int = 1, color=None):
    """用 Heading 样式加标题，再统一字体。

    用样式而非手工加粗：这样 Word 的导航窗格与自动目录才能识别。
    """
    from docx.shared import Pt

    h = document.add_heading(level=level)
    h.paragraph_format.space_before = Pt(10 if level <= 1 else 6)
    h.paragraph_format.space_after = Pt(4)
    run = h.add_run(text)
    _set_font(run, _HEADING_FONT, {0: 18, 1: 14, 2: 12, 3: 11}.get(level, 11), True, color)
    return h


def _fmt_money(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if n >= 100_000_000:
        return f"¥{n:,.0f}（{n / 100_000_000:.2f} 亿元）"
    if n >= 10_000:
        return f"¥{n:,.0f}（{n / 10_000:.2f} 万元）"
    return f"¥{n:,.0f}"


def build_tender_report(
    parsed: dict[str, Any],
    meta: dict[str, Any] | None = None,
    *,
    qualification: dict[str, Any] | None = None,
    price_profile: dict[str, Any] | None = None,
    competitor: dict[str, Any] | None = None,
) -> bytes:
    """生成投标决策报告（DOCX 字节）。

    参数都可选但**至少要有 parsed**。传入的可选分析会各自成章：
        `qualification`  资格审查结果 → 资格自查表
        `price_profile`  报价策略结果 → 报价参考
        `competitor`     竞争格局结果 → 竞争格局
    """
    docx = _require_docx()
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import RGBColor

    RED = RGBColor(0xC0, 0x00, 0x00)
    GRAY = RGBColor(0x60, 0x60, 0x60)

    parsed = parsed or {}
    fields = parsed.get("fields") or {}
    sections = parsed.get("sections") or {}
    meta = meta or {}

    document = docx.Document()

    # ------------------------------------------------------------------
    # 封面
    # ------------------------------------------------------------------
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    trun = title.add_run(_strip_md(fields.get("project_name") or "招标文件"))
    _set_font(trun, _HEADING_FONT, 22, True)

    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    srun = sub.add_run("投标决策报告")
    _set_font(srun, _HEADING_FONT, 15, False, GRAY)

    _para(
        document,
        f"项目编号：{fields.get('project_id') or '—'}　|　采购人：{fields.get('purchaser') or '—'}",
        size=10, color=GRAY, space_after=2,
    )
    _para(
        document,
        f"生成时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}"
        f"　|　抽取模型：{meta.get('model') or '—'}"
        f"　|　字段完整度：{meta.get('completeness', 0):.0%}",
        size=9, color=GRAY, space_after=10,
    )

    # 局限声明放封面：读者第一眼就该知道这份报告的可靠边界。
    # 拆成两段——实测单段过长会超出右页边距（Word 不会自动缩字号）。
    _para(
        document,
        "⚠ 本报告由 AI 从招标文件中自动抽取生成。",
        10, True, color=RED, space_after=1,
    )
    _para(
        document,
        "废标条款、资格要求等关键内容**必须对照原文复核**后再用于投标决策。",
        10, True, color=RED, space_after=10,
    )

    document.add_page_break()

    # ------------------------------------------------------------------
    # 一、项目概况
    # ------------------------------------------------------------------
    _heading(document, "一、项目概况", 1)
    overview_rows = [
        ("项目名称", fields.get("project_name") or "—"),
        ("项目编号", fields.get("project_id") or "—"),
        ("采购人", fields.get("purchaser") or "—"),
        ("代理机构", fields.get("agency") or "—"),
        ("预算金额", _fmt_money(fields.get("budget"))),
        ("最高限价", _fmt_money(fields.get("max_price"))),
        ("投标保证金", _fmt_money(fields.get("deposit"))),
        ("评标办法", fields.get("evaluation_method") or "—"),
        ("价格分权重", f"{fields.get('price_weight')} 分" if fields.get("price_weight") else "—"),
        ("服务期/工期", fields.get("duration") or "—"),
        ("投标有效期", fields.get("bid_validity") or "—"),
        ("付款方式", fields.get("payment_terms") or "—"),
        ("联系方式", fields.get("contact") or "—"),
    ]
    t = document.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    for k, v in overview_rows:
        row = t.add_row().cells
        kr = row[0].paragraphs[0].add_run(_strip_md(k))
        _set_font(kr, _BODY_FONT, 10, True)
        vr = row[1].paragraphs[0].add_run(_strip_md(v))
        _set_font(vr, _BODY_FONT, 10)
    # 标签列窄、内容列宽：16cm 是 A4 减去左右各 2.5cm 页边距后的可用宽度
    _fit_table(t, [4.2, 11.8])

    # ------------------------------------------------------------------
    # 二、关键时间节点（投标必须卡点）
    # ------------------------------------------------------------------
    _heading(document, "二、关键时间节点", 1)
    deadline = fields.get("bid_deadline") or ""
    open_time = fields.get("open_time") or ""
    if deadline or open_time:
        tt = document.add_table(rows=0, cols=2)
        tt.style = "Table Grid"
        for k, v in (("投标截止时间", deadline or "—"), ("开标时间", open_time or "—")):
            row = tt.add_row().cells
            kr = row[0].paragraphs[0].add_run(_strip_md(k))
            _set_font(kr, _BODY_FONT, 10, True)
            vr = row[1].paragraphs[0].add_run(_strip_md(str(v).replace("T", " ")))
            _set_font(vr, _BODY_FONT, 10, True, RED if k == "投标截止时间" else None)
        _fit_table(tt, [4.2, 11.8])
        _para(
            document,
            "※ 以上时间取自招标文件自动抽取结果，请与原文核对；"
            "逾期送达的投标文件将被拒收。",
            size=9, color=GRAY,
        )
    else:
        _para(document, "未抽到时间节点——请务必人工查阅原文确认投标截止时间。", 10, True, color=RED)

    # ------------------------------------------------------------------
    # 三、资格要求自查表
    # ------------------------------------------------------------------
    quals = sections.get("qualifications") or []
    _heading(document, f"三、资格要求自查表（{len(quals)} 条）", 1)
    if quals:
        qt = document.add_table(rows=1, cols=4)
        qt.style = "Table Grid"
        hdr = ["序号", "资格要求", "类型", "是否满足"]
        for i, h in enumerate(hdr):
            hr = qt.rows[0].cells[i].paragraphs[0].add_run(_strip_md(h))
            _set_font(hr, _BODY_FONT, 10, True)
        for i, q in enumerate(quals):
            cells = qt.add_row().cells
            for j, val in enumerate(
                [
                    str(i + 1),
                    str(q.get("text") or ""),
                    "硬性门槛" if q.get("is_mandatory") else "加分项",
                    "☐ 满足　☐ 不满足",
                ]
            ):
                vr = cells[j].paragraphs[0].add_run(_strip_md(val))
                _set_font(vr, _BODY_FONT, 10, j == 2 and bool(q.get("is_mandatory")))
        # 序号 1.2 / 要求 8.4 / 类型 2.4 / 勾选 4.0 = 16cm
        _fit_table(qt, [1.1, 7.9, 2.2, 4.8])
        _para(document, "※ 硬性门槛不满足即无法投标（或投标无效）。", size=9, color=GRAY)
    else:
        _para(document, "未抽到资格要求章节——请人工查阅原文。", 10, True, color=RED)

    # 资格审查结果（可选）
    if qualification:
        summary = qualification.get("summary") or {}
        _heading(document, "3.1 资格自查结论", 2)
        # 结论与计数**分两行**：实测拼成一行会超出右页边距被裁掉
        _para(
            document,
            str(qualification.get("overall_label") or ""),
            11, True,
            color=RED if qualification.get("overall") == "unqualified" else None,
            space_after=1,
        )
        _para(
            document,
            f"满足 {summary.get('met', 0)} 项 / 不满足 {summary.get('unmet', 0)} 项"
            f" / 需人工确认 {summary.get('uncertain', 0)} 项",
            10, color=GRAY,
        )
        if summary.get("conclusion"):
            _para(document, str(summary["conclusion"]), 10)
        gaps = qualification.get("gaps") or []
        if gaps:
            _para(document, f"硬性要求缺口（{len(gaps)} 项）：", 10, True, color=RED)
            for g in gaps:
                # 拆两行：原文 + 理由。长句拼成一行折行后难以对应
                _para(document, f"• {g.get('raw_text')}", 10, indent=12, space_after=0)
                _para(document, f"　理由：{g.get('reason')}", 9.5, indent=24)
        for c in summary.get("caveats") or []:
            _para(document, f"※ {c}", 9, color=GRAY)

    # ------------------------------------------------------------------
    # 四、评分标准
    # ------------------------------------------------------------------
    scoring = sections.get("scoring_items") or []
    _heading(document, f"四、评分标准（{len(scoring)} 项）", 1)
    if scoring:
        total = 0.0
        st = document.add_table(rows=1, cols=4)
        st.style = "Table Grid"
        for i, h in enumerate(["序号", "评分项", "分值", "类别"]):
            hr = st.rows[0].cells[i].paragraphs[0].add_run(_strip_md(h))
            _set_font(hr, _BODY_FONT, 10, True)
        for i, s in enumerate(scoring):
            score = s.get("score")
            if isinstance(score, (int, float)):
                total += float(score)
            cells = st.add_row().cells
            for j, val in enumerate(
                [
                    str(i + 1),
                    str(s.get("text") or ""),
                    f"{score} 分" if score not in (None, "") else "—",
                    str(s.get("category") or "—"),
                ]
            ):
                vr = cells[j].paragraphs[0].add_run(_strip_md(val))
                _set_font(vr, _BODY_FONT, 10)
        _fit_table(st, [1.2, 9.4, 2.4, 3.0])
        if total:
            _para(document, f"※ 以上分值合计 {total:g} 分（仅统计能识别出分值的项）。", size=9, color=GRAY)
    else:
        _para(document, "未抽到评分标准章节——报价策略需人工评估。", 10, True, color=RED)

    # ------------------------------------------------------------------
    # 五、废标条款（红线）—— 用红色标题强调
    # ------------------------------------------------------------------
    disq = sections.get("disqualification") or []
    _heading(document, f"五、废标红线（{len(disq)} 条）", 1, color=RED)
    _para(
        document,
        "以下任一条被触发，投标将被直接否决（废标）。请逐条核对投标文件。",
        10, True, color=RED,
    )
    if disq:
        for i, d in enumerate(disq):
            _para(document, f"{i + 1}. ☐ {d.get('text')}", 10.5, indent=6)
    else:
        _para(
            document,
            "未抽到废标条款。注意：废标条款常分散在「投标人须知」「投标文件的编制」"
            "等多个章节，**务必人工查阅原文**，不要因为没有抽出就认为不存在。",
            10, True, color=RED,
        )

    # ------------------------------------------------------------------
    # 六、技术要点
    # ------------------------------------------------------------------
    tech = sections.get("technical_requirements") or []
    if tech:
        _heading(document, f"六、技术要点（{len(tech)} 条）", 1)
        for i, x in enumerate(tech):
            _para(document, f"{i + 1}. {x.get('text')}", 10.5)

    # ------------------------------------------------------------------
    # 七、需提交材料清单（带勾选框）
    # ------------------------------------------------------------------
    docs = sections.get("required_documents") or []
    if docs:
        _heading(document, f"七、需提交材料清单（{len(docs)} 项）", 1)
        _para(document, "打印后逐项核对，齐全打勾：", 10, color=GRAY)
        for x in docs:
            _para(document, f"☐ {x.get('text')}", 10.5, indent=6)

    # ------------------------------------------------------------------
    # 八、报价参考（可选）
    # ------------------------------------------------------------------
    if price_profile:
        _heading(document, "八、报价参考", 1)
        rng = price_profile.get("suggested_range") or {}
        if rng:
            _para(
                document,
                f"建议报价区间：{_fmt_money(rng.get('low'))} ~ {_fmt_money(rng.get('high'))}",
                11, True,
            )
        _para(
            document,
            f"匹配历史记录 {price_profile.get('count', 0)} 条"
            f"（标的物：{price_profile.get('subject') or '—'}）",
            10, color=GRAY,
        )
        for k, label in (("median", "中位数"), ("p25", "P25"), ("p75", "P75")):
            v = (price_profile.get("stats") or {}).get(k)
            if v is not None:
                _para(document, f"　{label}：{_fmt_money(v)}", 10, indent=12)
        _para(
            document,
            "※ 建议基于历史中标价分位数，**不含成本测算与竞争态势**，仅供参考。",
            9, color=GRAY,
        )

    # ------------------------------------------------------------------
    # 九、数据质量与局限（不省略）
    # ------------------------------------------------------------------
    _heading(document, "九、数据质量与局限", 1)
    _para(
        document,
        f"文件：{meta.get('filename') or '—'}　|　类型：{str(meta.get('kind') or '—').upper()}"
        f"　|　页数：{meta.get('pages') or '—'}　|　字符数：{meta.get('total_chars') or '—'}",
        10,
    )
    _para(
        document,
        f"字段完整度：{(meta.get('completeness') or 0):.0%}"
        f"　|　抽取批次：{meta.get('batch_count') or '—'}"
        f"（成功 {meta.get('succeeded_batches') or '—'}）",
        10,
    )
    if meta.get("truncated"):
        _para(document, "⚠ 原文件过长，已截断，后文内容未参与抽取。", 10, True, color=RED)
    warnings = meta.get("warnings") or []
    if warnings:
        _para(document, "抽取过程提示：", 10, True)
        for w in warnings:
            _para(document, f"• {w}", 9.5, indent=12)
    _para(
        document,
        "※ 上述完整度与提示反映自动抽取的可靠性边界。"
        "关键条款（尤其废标条款与资格要求）请务必对照原文复核。",
        9, color=GRAY,
    )

    _add_page_numbers(document)

    buffer = io.BytesIO()
    document.save(buffer)
    logger.info("生成投标决策报告：%d 字节", buffer.tell())
    return buffer.getvalue()


def report_filename(parsed: dict[str, Any] | None = None) -> str:
    """生成下载文件名。

    用项目名而非固定名：用户一次可能导出多份报告，同名会互相覆盖。
    同时过滤 Windows 文件名的非法字符（\\ / : * ? " < > |）。
    """
    import re

    name = ((parsed or {}).get("fields") or {}).get("project_name") or "招标文件"
    safe = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(name)).strip()[:60]
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{safe}_投标决策报告_{stamp}.docx"


__all__ = ["ReportError", "build_tender_report", "report_filename"]
