"""招标文件结构化抽取：解析后的文本 → 按 schema 抽字段与列表章节。

与 `services/extract.py`（标讯公告抽取）的关系
--------------------------------------------
两者共用同一套"调 LLM → 解析 JSON → 归一化"的骨架，但**输入形态不同**：

    extract.py          输入是一小段公告文本，一次调用抽完
    tender_extractor.py 输入是几十页的招标文件，**必须分块多次抽取再合并**

分块合并是本模块的核心复杂度，也是它单独成文件的原因。

为什么要分块（而不是一次抽完）
----------------------------
实测招标文件 2 万~20 万字，直接送模型有两个硬问题：
    1. 超出上下文（本地 Ollama 32K）
    2. 即使塞得下，"lost in the middle" 会让中段与末尾的
       资质要求、废标条款被漏抽——而这两项恰恰是投标决策最需要的

合并策略（按字段类型区分，不能一刀切）
----------------------------------
    - **标量字段**：先抽到的非空值优先。因为招标文件的项目信息、
      时间节点几乎都在开头，前面的块更权威；后面的块若给出不同值
      （罕见），记为冲突并保留首个值 + 警告。
    - **列表章节**：**全部合并并去重**。资质要求可能分散在
      "资格要求"章、"资格审查表"附表、投标文件编制要求里，
      任何一块都不能丢——这正是分块抽取的价值所在。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterator

from src.constants.tender_schema import (
    TENDER_FIELDS,
    TENDER_SECTIONS,
    completeness,
    extraction_prompt,
    normalize_tender,
)
from src.logging_config import get_logger
from src.services.document_parser import ParsedDocument, select_relevant_chunks

logger = get_logger(__name__)

# 单次送给模型的字符上限。分块后每块约 3000 字，
# 这里允许一次送多块以**减少 LLM 调用次数**（调用次数直接等于耗时与费用）。
# 12000 字约 6000 汉字，留足输出空间后仍在多数模型可靠范围内。
BATCH_CHARS = 12_000

# 最多抽多少批。防止超长文档把 LLM 调用次数放大到不可接受
# （每批一次调用，云端约 10s，本地 Ollama 约 60s）
MAX_BATCHES = 8

# 单个列表章节保留条目上限：防止模型把整章原文逐句拆进来
MAX_ITEMS_PER_SECTION = 80


class TenderExtractError(RuntimeError):
    """抽取失败（模型不可用 / 输出无法解析 / 无有效内容）。"""


# ---------------------------------------------------------------------------
# JSON 解析（与 extract.py 同策略：三级兜底）
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    if len(parts) >= 3:
        body = parts[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        return body.strip()
    return t


def parse_json_object(text: str) -> dict[str, Any]:
    """解析模型输出为 dict。策略与 `services/extract.parse_json_object` 一致：

    先判整体是否合法 JSON（是数组就拒绝），再做"截取大括号"兜底。
    顺序不能反——否则顶层是数组的响应会被从中间截出对象、当成合法结果。
    """
    body = _strip_code_fence(text)
    try:
        whole = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        whole = None
    if whole is not None:
        if isinstance(whole, dict):
            return whole
        raise TenderExtractError(
            f"模型返回的 JSON 顶层是 {type(whole).__name__}，本接口要求对象"
        )

    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(body[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

    raise TenderExtractError("模型输出无法解析为 JSON")


# ---------------------------------------------------------------------------
# 分块与批次
# ---------------------------------------------------------------------------

def _build_batches(parsed: ParsedDocument, batch_chars: int = BATCH_CHARS) -> list[str]:
    """把相关块拼成若干批次文本，每批不超过 `batch_chars`。

    相关块由 `select_relevant_chunks` 排序给出（命中关心章节的优先），
    故即使因 `MAX_BATCHES` 截断，丢掉的也是最不相关的尾部内容。

    ★ 标题去重：`split_into_chunks` 会把标题补进块正文开头
    （那是为了防止"标题误判导致正文丢失"的兜底保护），
    故这里的 `【标题】` 前缀要先判断正文是否已含该标题，避免重复喂给模型。
    重复标题本身无害，但会浪费 token 并让模型看到两份一样的标题。
    """
    chunks = select_relevant_chunks(parsed, max_chars=batch_chars * MAX_BATCHES)
    batches: list[str] = []
    buf: list[str] = []
    size = 0
    for c in chunks:
        if c.heading and not c.text.startswith(c.heading):
            piece = f"【{c.heading}】\n{c.text}"
        else:
            piece = c.text
        if size + len(piece) > batch_chars and buf:
            batches.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(piece)
        size += len(piece)
    if buf:
        batches.append("\n\n".join(buf))
    return batches[:MAX_BATCHES]


# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------

def _merge_scalars(target: dict[str, Any], incoming: dict[str, Any], warnings: list[str]) -> None:
    """标量字段合并：先到优先；冲突时保留首值并记录警告。

    为什么"先到优先"而不是"后者覆盖"：招标文件的项目名称、编号、预算、
    时间节点几乎都在开头的招标公告里。后文出现的同名内容通常是**引用**
    （如合同条款里再提一次金额），若用后者覆盖，反而引入偏差。
    """
    for f in TENDER_FIELDS:
        new = incoming.get(f.key)
        if new in (None, "", []):
            continue
        cur = target.get(f.key)
        if cur in (None, "", []):
            target[f.key] = new
        elif str(cur) != str(new):
            # 同字段不同值：可能是"预算"与"最高限价"被模型混填，
            # 也可能是真的前后不一致（值得让人看一眼）
            warnings.append(
                f"字段「{f.label}」在不同章节取值不同："
                f"已采用「{cur}」，另见「{new}」"
            )


def _normalize_item_text(text: str) -> str:
    """条目文本归一（用于去重）：去空白、去序号前缀、转小写。"""
    t = re.sub(r"\s+", "", text or "")
    t = re.sub(r"^[（(]?[一二三四五六七八九十\d]+[)）.、．:：]\s*", "", t)
    return t.lower()


def _merge_sections(
    target: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
) -> int:
    """列表章节合并：全部并集 + 按归一化文本去重。返回新增条数。

    去重按"归一化后的文本"而非原文：同一要求在不同章节常带不同序号
    （"1. 具有…"与"（一）具有…"），按原文去重会重复计入。
    """
    added = 0
    for s in TENDER_SECTIONS:
        cur = target.setdefault(s.key, [])
        seen = {_normalize_item_text(i.get("text", "")) for i in cur}
        for item in incoming.get(s.key) or []:
            key = _normalize_item_text(item.get("text", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            cur.append(item)
            added += 1
            if len(cur) >= MAX_ITEMS_PER_SECTION:
                break
    return added


def merge_batches(results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """把多批抽取结果合并成一份。返回 `(解析结果, 警告列表)`。"""
    fields: dict[str, Any] = {f.key: (None if f.type == "money" else "") for f in TENDER_FIELDS}
    sections: dict[str, list[dict[str, Any]]] = {s.key: [] for s in TENDER_SECTIONS}
    warnings: list[str] = []

    for r in results:
        _merge_scalars(fields, r.get("fields") or {}, warnings)
        _merge_sections(sections, r.get("sections") or {})

    # 同一冲突可能被多批重复记录，去重后更可读
    deduped = list(dict.fromkeys(warnings))
    return {"fields": fields, "sections": sections}, deduped


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _get_llm_client():
    from src.clients.llm_factory import get_llm_client
    from src.config import settings

    provider = (settings.llm_provider or "deepseek").strip()
    client = get_llm_client(provider)
    if client is None or not getattr(client, "available", lambda: False)():
        raise TenderExtractError(f"LLM 客户端不可用（provider={provider}）")
    return client, provider


def extract_from_parsed(
    parsed: ParsedDocument,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """对解析好的文档做结构化抽取。

    `progress(done, total)` 可选回调，用于 SSE 上报进度——
    分块抽取要多次调用 LLM，全程 1~3 分钟，没有进度反馈用户会以为卡死。
    """
    batches = _build_batches(parsed)
    if not batches:
        raise TenderExtractError("文档没有可用于抽取的内容")

    client, provider = _get_llm_client()
    system_prompt = extraction_prompt()

    raw_results: list[dict[str, Any]] = []
    batch_errors: list[str] = []

    for i, batch in enumerate(batches):
        if progress:
            progress(i, len(batches))
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"这是招标文件的第 {i + 1}/{len(batches)} 部分，"
                    "请抽取**本部分出现**的信息（本部分没有的字段留空）：\n\n" + batch
                ),
            },
        ]
        try:
            raw = client.chat_raw(messages)
        except Exception as exc:
            # 单批失败不中断整体：其余批次仍可能抽到关键信息
            logger.warning("第 %d/%d 批抽取失败：%s", i + 1, len(batches), exc)
            batch_errors.append(f"第 {i + 1} 批调用失败：{type(exc).__name__}")
            continue

        if not (raw or "").strip():
            batch_errors.append(f"第 {i + 1} 批返回空内容")
            continue

        try:
            parsed_json = parse_json_object(raw)
        except TenderExtractError as exc:
            batch_errors.append(f"第 {i + 1} 批输出无法解析：{exc}")
            continue

        raw_results.append(normalize_tender(parsed_json))

    if progress:
        progress(len(batches), len(batches))

    if not raw_results:
        raise TenderExtractError(
            "所有批次都抽取失败。" + ("；".join(batch_errors) if batch_errors else "")
        )

    merged, warnings = merge_batches(raw_results)

    # 全部批次都没抽出必填字段时给出明确提示，而不是返回一份空壳
    f = merged["fields"]
    if not any(f.get(k) for k in ("project_name", "project_id")):
        warnings.append(
            "未抽出项目名称与编号——可能该文档不是招标文件正文，"
            "或扫描件无文字层。请确认文件内容"
        )

    total_items = sum(len(v) for v in merged["sections"].values())
    logger.info(
        "招标文件抽取完成：%d 批，%d 条列表项，完整度 %.2f",
        len(raw_results), total_items, completeness(merged),
    )

    return {
        "parsed": merged,
        "meta": {
            "provider": provider,
            "model": getattr(client, "model", ""),
            "filename": parsed.filename,
            "kind": parsed.kind,
            "pages": parsed.pages,
            "total_chars": parsed.total_chars,
            "truncated": parsed.truncated,
            "chunk_count": len(parsed.chunks),
            "batch_count": len(batches),
            "succeeded_batches": len(raw_results),
            "completeness": completeness(merged),
            "section_item_counts": {k: len(v) for k, v in merged["sections"].items()},
            "warnings": list(parsed.warnings) + warnings + batch_errors,
        },
    }


def extract_from_document(filename: str, data: bytes) -> dict[str, Any]:
    """一步到位：文件字节 → 解析 → 抽取。"""
    from src.services.document_parser import parse_document

    parsed = parse_document(filename, data)
    return extract_from_parsed(parsed)


def stream_extract(
    filename: str, data: bytes
) -> Iterator[tuple[str, Any]]:
    """生成器版抽取，逐步 yield `(事件类型, 数据)`。

    事件序列：
        ("parse", {…})      解析完成（页数/字数/块数）
        ("progress", {…})   每批抽取前上报进度
        ("done", {…})       最终结果
        ("error", {…})      失败

    为什么要流式：分块抽取要调 LLM 多次，云端 1~2 分钟、本地 Ollama 可能 5 分钟以上。
    没有进度反馈的话前端只能一直转圈，用户无法判断是"在跑"还是"卡死"。
    """
    from src.services.document_parser import DocumentParseError, parse_document

    try:
        parsed = parse_document(filename, data)
    except DocumentParseError as exc:
        yield "error", {"content": str(exc)}
        return

    yield "parse", {
        "filename": parsed.filename,
        "kind": parsed.kind,
        "pages": parsed.pages,
        "total_chars": parsed.total_chars,
        "chunk_count": len(parsed.chunks),
        "truncated": parsed.truncated,
        "headings": [c.heading for c in parsed.chunks if c.heading][:40],
        "warnings": list(parsed.warnings),
    }

    batches = _build_batches(parsed)
    if not batches:
        yield "error", {"content": "文档没有可用于抽取的内容"}
        return

    try:
        client, provider = _get_llm_client()
    except TenderExtractError as exc:
        yield "error", {"content": str(exc)}
        return

    system_prompt = extraction_prompt()
    raw_results: list[dict[str, Any]] = []
    batch_errors: list[str] = []

    for i, batch in enumerate(batches):
        yield "progress", {
            "done": i,
            "total": len(batches),
            "message": f"正在抽取第 {i + 1}/{len(batches)} 部分…",
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"这是招标文件的第 {i + 1}/{len(batches)} 部分，"
                    "请抽取**本部分出现**的信息（本部分没有的字段留空）：\n\n" + batch
                ),
            },
        ]
        try:
            raw = client.chat_raw(messages)
        except Exception as exc:
            logger.warning("第 %d 批抽取失败：%s", i + 1, exc)
            batch_errors.append(f"第 {i + 1} 批调用失败：{type(exc).__name__}")
            continue
        if not (raw or "").strip():
            batch_errors.append(f"第 {i + 1} 批返回空内容")
            continue
        try:
            raw_results.append(normalize_tender(parse_json_object(raw)))
        except TenderExtractError as exc:
            batch_errors.append(f"第 {i + 1} 批输出无法解析：{exc}")

    if not raw_results:
        yield "error", {
            "content": "所有批次都抽取失败。" + ("；".join(batch_errors) if batch_errors else "")
        }
        return

    merged, warnings = merge_batches(raw_results)
    f = merged["fields"]
    if not any(f.get(k) for k in ("project_name", "project_id")):
        warnings.append(
            "未抽出项目名称与编号——可能该文档不是招标文件正文，或扫描件无文字层"
        )

    yield "done", {
        "parsed": merged,
        "meta": {
            "provider": provider,
            "model": getattr(client, "model", ""),
            "filename": parsed.filename,
            "kind": parsed.kind,
            "pages": parsed.pages,
            "total_chars": parsed.total_chars,
            "truncated": parsed.truncated,
            "chunk_count": len(parsed.chunks),
            "batch_count": len(batches),
            "succeeded_batches": len(raw_results),
            "completeness": completeness(merged),
            "section_item_counts": {k: len(v) for k, v in merged["sections"].items()},
            "warnings": list(parsed.warnings) + warnings + batch_errors,
        },
    }


__all__ = [
    "BATCH_CHARS",
    "MAX_BATCHES",
    "TenderExtractError",
    "extract_from_document",
    "extract_from_parsed",
    "merge_batches",
    "parse_json_object",
    "stream_extract",
]
