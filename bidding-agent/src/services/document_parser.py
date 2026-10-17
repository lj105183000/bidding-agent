"""招标文件解析：PDF / Word → 结构化文本（含章节切分）。

为什么需要"章节切分"而不是把全文丢给模型
--------------------------------------
实测招标文件常见 **2 万~20 万字**，远超任何模型的可靠抽取范围：

    - 一次塞进去会超出上下文（本地 Ollama 只有 32K）
    - 即使塞得下，模型对长文本中段的注意力显著下降（"lost in the middle"），
      中间的资质要求、末尾的废标条款最容易漏抽

故本模块把正文切成带**章节标题**的块，抽取时按块喂给模型再合并结果。
好处有三：
    1. 每块都在模型注意力范围内
    2. 保留章节归属（"这条资质要求出自第 3 章"），便于人工核对
    3. 抽取失败可定位到具体块，而不是"整篇抽不出来"

★ 一个必须如实告知的局限
--------------------
PDF 是**版式文件**：扫描件没有文字层，`get_text()` 会返回空。
本模块遇到这种情况会明确报错并提示改用截图 + 视觉模型，
**不会返回空字符串让上层以为"文件里没内容"**。

Word 表格同理：正文段落与表格是分开存的，只读 `paragraphs` 会漏掉
表格里的资格要求（招标文件极常见）。故两者都要读。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# 单块目标长度（字符）。取值依据：
#   - 太小（<1000）会切碎章节语义，且增加 LLM 调用次数与费用
#   - 太大（>6000）接近本地模型 32K 上下文下"中段注意力衰减"的区间
# 3000 字符约等于 1500 汉字，覆盖一个完整小节，是常见招标文件章节的典型长度。
CHUNK_TARGET_CHARS = 3000

# 单块硬上限：遇到超长无标题段落（如密集的评分表）时的兜底切分点
CHUNK_MAX_CHARS = 6000

# 全文上限：防止误传巨型文件（如整本装订的招标文件合集，几十万字）
# 超过则截断并明确告知，而不是静默丢内容
MAX_TOTAL_CHARS = 400_000

# 章节标题识别
#
# 分两档，**强弱有别**（区别很重要，见下）：
#
#   strong —— "第X章/第X节"、"附件一"。这些几乎不可能是正文行首，
#             匹配即认定为标题。
#   weak   —— "1.1"、"一、"、"（一）"。⚠️ **正文条目也用这套编号**
#             （招标文件的废标条款、资格要求、评分项极常用"（一）…"列举），
#             故弱模式必须配合 `_weak_heading_ok()` 的额外约束，
#             否则会把正文条目误判成标题。
_STRONG_HEADING_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("chapter", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*[章节篇部分]\s*[、.．:：]?\s*\S")),
    ("appendix", re.compile(r"^\s*附\s*[件表录]\s*[一二三四五六七八九十\d]*\s*[、.．:：]?\s*\S")),
)
_WEAK_HEADING_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("numbered", re.compile(r"^\s*\d+(\.\d+){1,3}\s*\S")),
    ("numbered_space", re.compile(r"^\s*\d+\s+[\u4e00-\u9fa5]")),
    ("chinese_num", re.compile(r"^\s*[一二三四五六七八九十]+\s*[、.．]\s*\S")),
    ("paren_num", re.compile(r"^\s*[（(]\s*[一二三四五六七八九十\d]+\s*[)）]\s*\S")),
)

# 章节标题长度上限：真正的标题通常短；超过这个长度多半是正文恰好以数字开头
_HEADING_MAX_CHARS = 60

# 弱标题额外排除的句末标点。命中即判为**正文**而非标题。
#
# ★ 这是修复"整段内容静默消失"的关键（实测踩坑）
# ------------------------------------------------
# 原实现把 `（一）投标文件未按要求密封的。` 这类**正文条目**当成章节标题，
# 而连续多行都是这种条目时，每次 flush() 的 body 都是空的 ——
# 这些行既没进正文、也没被保留成标题，**整段消失**。
#
# 实锤后果：一份废标条款用"（一）…（九）"编号的招标文件，
# 抽取出的 `disqualification` 条目数为 **0**（用"1."编号时是 8），
# LLM 根本没看到这些条款；下游的变更追踪还会把 8 条报成"删除"、
# 把真实新增漏掉。
#
# 招标文件里"（一）"编号在废标条款/资格要求/评分项中极常见，属"整章丢失"级别。
_SENTENCE_END_PUNCT = "。；;！!？?"

# 与投标决策强相关的章节关键字 → 内部 key
#
# 用关键字把章节标题映射到我们关心的类别，好处是抽取时可以**优先送相关块**，
# 而不必把全文都喂一遍（省 token、也提高相关章节的注意力权重）。
SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "qualifications": ("资格", "资质", "投标人条件", "合格投标人", "资格审查"),
    "scoring_items": ("评标", "评分", "评审", "打分", "综合评分"),
    "disqualification": ("废标", "投标无效", "否决投标", "不予受理", "无效投标"),
    "technical_requirements": ("技术要求", "技术规格", "技术标准", "服务要求", "采购需求"),
    "required_documents": ("投标文件", "须提交", "应提交", "材料清单", "编制要求"),
}


class DocumentParseError(RuntimeError):
    """解析失败（格式不支持 / 文件损坏 / 扫描件无文字层）。"""


@dataclass
class Chunk:
    """一个文本块。"""

    index: int
    heading: str  # 所属章节标题（空表示无标题）
    text: str
    # 命中的关心章节类别（用于抽取时优先送相关块）
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "heading": self.heading,
            "chars": len(self.text),
            "categories": self.categories,
            # 预览用于前端展示与人工核对，不返回全文（可能很大）
            "preview": self.text[:160].replace("\n", " "),
        }


@dataclass
class ParsedDocument:
    """解析结果。"""

    filename: str
    kind: str  # "pdf" | "docx"
    pages: int
    total_chars: int
    truncated: bool
    text: str
    chunks: list[Chunk]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "kind": self.kind,
            "pages": self.pages,
            "total_chars": self.total_chars,
            "truncated": self.truncated,
            "chunk_count": len(self.chunks),
            "warnings": list(self.warnings),
            # 只给元信息，正文由调用方按需取（可能几万字）
            "chunks": [c.to_dict() for c in self.chunks],
            "headings": [c.heading for c in self.chunks if c.heading][:60],
        }


# ---------------------------------------------------------------------------
# 文本提取
# ---------------------------------------------------------------------------


def _extract_pdf(data: bytes, filename: str) -> tuple[str, int, list[str]]:
    """提取 PDF 文本。返回 `(文本, 页数, 警告)`。"""
    try:
        import pymupdf  # PyMuPDF ≥1.24 的新模块名
    except ImportError:  # pragma: no cover - 兼容旧版
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:
            raise DocumentParseError(
                "缺少 PDF 解析库，请运行：uv add pymupdf"
            ) from exc

    warnings: list[str] = []
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError(f"PDF 打开失败：{type(exc).__name__}") from exc

    try:
        pages = doc.page_count
        parts: list[str] = []
        empty_pages = 0
        for i, page in enumerate(doc):
            try:
                # sort=True 按阅读顺序排列文本块；不做这一步，双栏排版的
                # 招标文件会出现左右栏交错，句子读不通
                txt = page.get_text(sort=True)
            except Exception:
                txt = ""
            if not txt.strip():
                empty_pages += 1
            parts.append(f"\n<<<第 {i + 1} 页>>>\n{txt}")
    finally:
        doc.close()

    text = "".join(parts)

    # 扫描件检测：全部页都没有文字层 → 明确报错而不是返回空
    if empty_pages == pages and pages > 0:
        raise DocumentParseError(
            "该 PDF 没有文字层（很可能是扫描件或纯图片版）。"
            "请改用「截图 + 上传图片」的方式，由视觉模型识别。"
        )
    if empty_pages:
        warnings.append(
            f"{empty_pages}/{pages} 页没有提取到文字（可能是图表页或扫描页），"
            "这些页的内容未参与抽取"
        )
    return text, pages, warnings


def _extract_docx(data: bytes, filename: str) -> tuple[str, int, list[str]]:
    """提取 Word 文本。返回 `(文本, 段落数, 警告)`。

    **段落与表格都要读**：招标文件的资格要求、评分表经常放在表格里，
    只读 `paragraphs` 会整段漏掉。
    """
    try:
        import docx
    except ImportError as exc:
        raise DocumentParseError("缺少 Word 解析库，请运行：uv add python-docx") from exc

    import io

    try:
        doc = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(
            f"Word 打开失败：{type(exc).__name__}（仅支持 .docx，不支持旧版 .doc）"
        ) from exc

    parts: list[str] = []
    table_count = 0

    # 按文档顺序遍历段落与表格：python-docx 的 paragraphs / tables 是两个独立
    # 列表，直接先后拼接会打乱顺序（表格内容跑到文末），影响章节切分与抽取。
    # 故遍历 body 子元素按真实顺序取出。
    try:
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        body = doc.element.body
        for child in body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                text = Paragraph(child, doc).text
                if text.strip():
                    parts.append(text)
            elif tag == "tbl":
                table_count += 1
                table = Table(child, doc)
                rows = []
                for row in table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    parts.append("\n".join(rows))
    except Exception as exc:
        # 顺序遍历失败时退回简单方式，至少不丢内容
        logger.warning("按序遍历 Word 失败，退回分段读取: %s", exc)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for t in doc.tables:
            table_count += 1
            for row in t.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

    text = "\n".join(parts)
    warnings: list[str] = []
    if table_count:
        warnings.append(f"已解析 {table_count} 个表格（内容已按行列拼接为文本）")
    if not text.strip():
        raise DocumentParseError("Word 文档没有提取到文字内容")
    return text, len(parts), warnings


# ---------------------------------------------------------------------------
# 章节切分
# ---------------------------------------------------------------------------


def _weak_heading_ok(line: str) -> bool:
    """弱编号行（`一、` / `（一）` / `1.1`）是否**真的**是章节标题而非正文条目。

    ★ 判断依据（实测教训）
    --------------------
    招标文件里"（一）（二）…"既可能出现在**章节标题**（"（一）总则"），
    也可能出现在**正文列举**（"（一）投标文件未按要求密封的。"）。
    单看编号区分不了，但有个可靠信号：**正文条目以句末标点结尾**。

    故加两条排除：
        ① 以 。；！？ 等句末标点结尾 → 正文
        ② 同段内还有别的编号标记（说明是连续列举）→ 正文

    这两条是修复"整段内容静默消失"的关键。
    """
    s = line.strip()
    if not s:
        return False
    if s[-1] in _SENTENCE_END_PUNCT:
        return False
    # 同行出现 2 个及以上编号标记 → 是并列列举正文，不是标题
    markers = sum(
        1
        for pat in (
            r"[（(]\s*[一二三四五六七八九十\d]+\s*[)）]",
            r"(?:^|\s)[一二三四五六七八九十]+\s*[、.．]",
            r"(?:^|\s)\d+(?:\.\d+)*\s*[、.．]\s*",
        )
        if re.search(pat, s)
    )
    return markers <= 1


def _is_heading(line: str) -> tuple[bool, str]:
    """判断一行是否为章节标题，返回 `(是否标题, 命中的模式名)`。

    强模式（第X章 / 附件）直接命中；弱模式（编号）需再过 `_weak_heading_ok()`。
    """
    s = line.strip()
    if not s or len(s) > _HEADING_MAX_CHARS:
        return False, ""
    for name, pattern in _STRONG_HEADING_PATTERNS:
        if pattern.match(s):
            return True, name
    for name, pattern in _WEAK_HEADING_PATTERNS:
        if pattern.match(s) and _weak_heading_ok(s):
            return True, name
    return False, ""


def _categorize(heading: str) -> list[str]:
    """章节标题 → 关心的类别（可能多个）。"""
    hits = []
    for key, keywords in SECTION_KEYWORDS.items():
        if any(kw in heading for kw in keywords):
            hits.append(key)
    return hits


def split_into_chunks(text: str) -> list[Chunk]:
    """按章节标题切分，并受 `CHUNK_TARGET_CHARS` 约束。

    策略：
        1. 逐行扫描，遇到标题就开新块
        2. 块长超过 `CHUNK_TARGET_CHARS` 时，在**段落边界**（空行/换行）处切分
        3. 单个无标题超长段落超过 `CHUNK_MAX_CHARS` 时硬切（兜底）

    为什么在段落边界切：句子中间截断会让模型看到半句话，
    极容易抽错（例如"投标保证金为 5 万"被切成"5"和"万"两块）。
    """
    # 页码标记清理：`<<<第 N 页>>>` 是我们自己插的，切块时应去掉
    cleaned = re.sub(r"<<<第 \d+ 页>>>", "\n", text or "")

    lines = cleaned.splitlines()
    chunks: list[Chunk] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        """把当前累积的正文落成一个块。

        ★ 兜底保护：若标题行没有出现在正文里，把它补回正文开头。
        ------------------------------------------------
        这是"即使标题误判也不丢内容"的第二道防线。

        第一道防线是 `_weak_heading_ok()`（不让正文条目被当成标题）。
        但两道防线都要有——因为漏判标题只是"章节归属不准"，
        而**丢掉正文是内容损失**，后者严重得多。
        """
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if current_heading and current_heading not in body:
            body = f"{current_heading}\n{body}".strip()
        if body:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    heading=current_heading,
                    text=body,
                    categories=_categorize(current_heading),
                )
            )
        current_lines = []

    for line in lines:
        is_head, _ = _is_heading(line)
        if is_head:
            flush()
            current_heading = line.strip()
            continue
        current_lines.append(line)
        # 达到目标长度且当前行是段落末尾（空行或行尾标点）→ 切开
        if sum(len(x) for x in current_lines) >= CHUNK_TARGET_CHARS and (
            not line.strip() or line.rstrip().endswith(("。", "；", "：", ".", "!", "！"))
        ):
            flush()
            # 续块沿用同一标题，避免后续内容被当成"无章节"
            continue

    # 最后一个块：把标题补进去（flush 内的兜底只覆盖已 flush 的块）
    if current_lines or current_heading:
        flush()

    # 兜底：把仍然超长的块硬切
    result: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= CHUNK_MAX_CHARS:
            result.append(Chunk(len(result), c.heading, c.text, c.categories))
            continue
        for i in range(0, len(c.text), CHUNK_MAX_CHARS):
            piece = c.text[i : i + CHUNK_MAX_CHARS]
            result.append(
                Chunk(len(result), c.heading, piece, c.categories)
            )
    return result


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def detect_kind(filename: str, data: bytes) -> str:
    """判断文件类型：优先看**内容魔数**，再看扩展名。

    只看扩展名不可靠——用户可能把 .doc 改名成 .docx，
    或上传没有扩展名的文件。魔数才是可信来源。
    """
    if data[:4] == b"%PDF":
        return "pdf"
    # DOCX 是 ZIP：50 4B 03 04
    if data[:4] == b"PK\x03\x04":
        if filename.lower().endswith(".docx") or b"word/" in data[:4000]:
            return "docx"
        # 也可能是 xlsx/pptx，交由上层拒绝
        return "zip"
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    return "unknown"


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """解析 PDF / Word，返回带章节分块的结构化结果。

    失败抛 `DocumentParseError`（上层转 400/415），错误信息面向用户可读。
    """
    if not data:
        raise DocumentParseError("文件内容为空")

    kind = detect_kind(filename, data)
    if kind == "pdf":
        text, pages, warnings = _extract_pdf(data, filename)
    elif kind == "docx":
        text, pages, warnings = _extract_docx(data, filename)
    elif kind == "zip":
        raise DocumentParseError(
            "该文件是压缩包格式但不是 .docx（可能是 .xlsx/.pptx 改名的），"
            "请上传 PDF 或 Word 文档"
        )
    else:
        raise DocumentParseError(
            f"不支持的文件类型：{filename}。目前支持 PDF 与 Word(.docx)"
        )

    total = len(text)
    truncated = False
    if total > MAX_TOTAL_CHARS:
        text = text[:MAX_TOTAL_CHARS]
        truncated = True
        warnings.append(
            f"文档过长（{total:,} 字），已截断为前 {MAX_TOTAL_CHARS:,} 字参与抽取。"
            "如需完整解析，请拆分后分批上传"
        )

    chunks = split_into_chunks(text)
    if not chunks:
        raise DocumentParseError("未能从文档中切分出有效内容（可能是空文档）")

    logger.info(
        "解析文档 %s：%s，%d 页/段，%d 字，%d 块",
        filename, kind, pages, total, len(chunks),
    )

    return ParsedDocument(
        filename=filename,
        kind=kind,
        pages=pages,
        total_chars=total,
        truncated=truncated,
        text=text,
        chunks=chunks,
        warnings=warnings,
    )


def select_relevant_chunks(parsed: ParsedDocument, max_chars: int = 60_000) -> list[Chunk]:
    """挑选与投标决策最相关的块，供抽取时优先送模型。

    排序策略（而非简单截断）：
        1. **命中关心章节**的块最优先（资格/评分/废标/技术/材料）
        2. 其次是**开头部分**——招标文件的项目概况、时间节点、
           预算与保证金几乎都在前几块
        3. 其余按原序补齐直到 `max_chars`

    为什么不直接送全文：省 token，且让相关章节在上下文里更集中，
    实测能明显提高资质要求与废标条款的检出率。
    """
    priority: list[Chunk] = []
    head: list[Chunk] = []
    rest: list[Chunk] = []

    for c in parsed.chunks:
        if c.categories:
            priority.append(c)
        elif c.index < 5:
            head.append(c)
        else:
            rest.append(c)

    chosen: list[Chunk] = []
    used = 0
    for group in (priority, head, rest):
        for c in group:
            if used + len(c.text) > max_chars:
                continue
            chosen.append(c)
            used += len(c.text)
        if used >= max_chars:
            break

    # 按原始顺序返回：模型读乱序内容容易串章节
    return sorted(chosen, key=lambda x: x.index)


__all__ = [
    "CHUNK_MAX_CHARS",
    "CHUNK_TARGET_CHARS",
    "Chunk",
    "DocumentParseError",
    "MAX_TOTAL_CHARS",
    "ParsedDocument",
    "SECTION_KEYWORDS",
    "detect_kind",
    "parse_document",
    "select_relevant_chunks",
    "split_into_chunks",
]
