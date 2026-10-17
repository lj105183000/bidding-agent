"""合规检查工具：把招标文件条款逐条对照法规规则。

★ 为什么单独成模块而不塞进 rag_tools
--------------------------------
`rag_tools.py` 已 1421 行，且它的五个工具都是**检索类**；
本模块是**规则判定类**（确定性检测，不检索、不调 LLM）——
职责不同，混在一起会让那个文件继续膨胀且难以定位。

★ 为什么这个工具值得对 Agent 暴露（P4）
------------------------------------
项目有 `POST /api/compliance/check` 端点，但**Agent 调不到**。
于是当用户问「这条要求合规吗」，Agent 只能用检索拼一个**原则性判断**，
做不到「逐条比对 + 引用条款原文」——`docs/34` 记过这一点：
"用检索重建只能给原则性判断，**名不副实**"。

暴露之后技能 `regulation-citation-audit` 才真正可执行：
规则检出（确定性）+ 法规依据（条文原文）两层都有。

★ 依赖隔离
--------
与 `rag_tools` 同一策略：**延迟导入**服务层。
顶层 import 会让"服务层任何一处缺依赖"直接把整个工具注册表打挂，
五个已有工具一起消失——那是最坏的失败模式。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: 条款切分：中文序号 / 阿拉伯序号 / 换行。
#:
#: 为什么要切分：合规检查必须**逐条**做，整篇当一段会导致
#: "命中了一条却说不出是哪一条"，用户无法核对。
#: 与 `compliance._iter_clauses` 的口径一致——那里也要求逐条传入。
_CLAUSE_SPLIT_RE = re.compile(r"(?=第[一二三四五六七八九十百零\d]+[条章])|(?=\d+[.、])|\n+")

#: 单次检查的最大条款数。超出的截断并如实告知。
#:
#: 取值依据：`compliance.RULES` 有 ~15 条规则，逐条 × 逐规则正则匹配。
#: 200 条时单次约 1~2 秒；再大用户等待感明显，且招标文件的
#: **关键条款**（资格要求、商务条款、评标办法）通常在前 200 条内。
MAX_CLAUSES = 200

#: 单条条款的最大字符数。超长条款多为表格/目录残留，截断避免规则误匹配。
MAX_CLAUSE_CHARS = 600


def _split_clauses(text: str) -> list[str]:
    """把招标文件文本切成条款列表。"""
    if not text:
        return []
    raw = _CLAUSE_SPLIT_RE.split(text)
    out: list[str] = []
    for piece in raw:
        s = piece.strip()
        if len(s) < 4:
            continue
        out.append(s[:MAX_CLAUSE_CHARS])
        if len(out) >= MAX_CLAUSES:
            break
    return out


def check_tender_compliance(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """工具执行器：招标文件条款合规检查。

    参数：
        text            招标文件条款原文（必填）
        include_law_basis 是否附带法规依据原文，默认 True

    返回 `(给模型看的文本, sources)`，与其它工具同一契约。
    """
    text = str((arguments or {}).get("text") or "").strip()
    include_law = bool((arguments or {}).get("include_law_basis", True))

    if not text:
        return (
            "【合规检查失败】未提供招标文件条款文本（text 参数为空）。"
            "请把需要检查的条款原文作为 text 传入。",
            [],
        )

    clauses = _split_clauses(text)
    if not clauses:
        return "【合规检查失败】未能从文本中切分出可检查的条款。", []

    try:
        from src.services import compliance as compliance_service
    except Exception as exc:
        logger.warning("合规服务不可用: %s", exc)
        return (
            f"【合规检查不可用】服务未就绪（{type(exc).__name__}）。"
            "请改用自己的法规知识给出原则性判断，并明确说明未做逐条规则检出。",
            [],
        )

    # 构造 `_iter_clauses` 认识的结构：sections 下挂逐条文本。
    # 这里**必须**用条款切分而不是整篇传入，否则命中了也说不出是哪一条。
    parsed: dict[str, Any] = {
        "sections": {"招标文件条款": [{"text": c} for c in clauses]},
    }

    try:
        result = compliance_service.check_compliance(
            parsed, include_law_basis=include_law
        )
    except Exception as exc:
        logger.warning("合规检查执行失败: %s", exc, exc_info=True)
        return f"【合规检查失败】{type(exc).__name__}: {exc}", []

    return _format_result(result, len(clauses)), _to_sources(result)


def _format_result(result: dict, checked: int) -> str:
    """把检查结果渲染成给模型看的文本。"""
    findings = result.get("findings") or []
    summary = result.get("summary") or {}
    lines = [
        f"【合规检查结果】已逐条检查 {checked} 条条款，"
        f"命中疑似问题 {len(findings)} 条。",
    ]
    if summary:
        parts = [f"{k}={v}" for k, v in summary.items()]
        lines.append("分档统计：" + "、".join(parts))

    if not findings:
        lines.append(
            "未命中任何已知违规特征。**这不等于合规**——"
            "规则只覆盖已知的违规表述模式，未命中只说明没有明显问题。"
        )
    else:
        lines.append("命中明细（按严重度排序）：")
        for i, f in enumerate(findings[:20], 1):
            lines.append(
                f"\n{i}. [{f.get('severity')}] {f.get('title')}"
                f"（条款位置：{f.get('section')} #{f.get('item_index')}）"
            )
            clause = str(f.get("clause") or "").strip()
            if clause:
                lines.append(f"   原文：{clause[:200]}")
            if f.get("why"):
                lines.append(f"   为什么可疑：{f['why']}")
            if f.get("verify"):
                lines.append(f"   怎么核实：{f['verify']}")
            for law in (f.get("law_basis") or [])[:3]:
                src = law.get("law") or law.get("law_short") or ""
                art = law.get("article") or ""
                # ⚠️ 键名是 `excerpt` 不是 `text`。`_fetch_law_basis` 把检索结果的
                # `text` 摘成 `excerpt`（见 compliance.py），取错键会静默显示空正文——
                # 实测踩过：来源卡片全部只有法规名、正文为空。
                body = str(law.get("excerpt") or law.get("text") or "").strip()
                lines.append(f"   法规依据：《{src}》{art} {body[:160]}")

    disclaimer = result.get("disclaimer")
    if disclaimer:
        lines.append(f"\n⚠️ {disclaimer}")
    return "\n".join(lines)


def _to_sources(result: dict) -> list[dict]:
    """把命中的法规依据转成来源卡片。

    与 `rag_tools._rag_sources` 同一契约（`{question, answer, score, relevance}`），
    这样前端来源面板与核对器都能直接复用，不需要为本工具加分支。
    """
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for f in (result.get("findings") or []):
        for law in (f.get("law_basis") or []):
            name = str(law.get("law") or law.get("law_short") or "").strip()
            art = str(law.get("article") or "").strip()
            key = (name, art)
            if not name or key in seen:
                continue
            seen.add(key)
            body = str(law.get("excerpt") or law.get("text") or "").strip()
            sources.append({
                "question": f"《{name}》{art}",
                "answer": body,
                # 规则命中是确定性的，没有相似度概念，故 score 记 1.0 并标注口径
                "score": 1.0,
                "relevance": 1.0,
                "_source_kind": "compliance_rule",
            })
            if len(sources) >= 20:
                return sources
    return sources


#: 工具元信息。加进 `rag_tools._TOOL_SPECS` 由后者聚合注册。
TOOL_SPEC = {
    "name": "check_tender_compliance",
    "description": (
        "招标文件条款合规检查：逐条比对已知违规特征（地域限制、指定品牌、"
        "注册资本门槛、保证金超限、评分未量化等），并给出法规依据原文。"
        "**只传入用户提供的招标文件条款原文，不要自己编造条款**"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "招标文件条款原文（可多行，工具会自动切分逐条检查）。"
                    "必须是用户提供的原文，不要改写或概括"
                ),
            },
            "include_law_basis": {
                "type": "boolean",
                "description": "是否附带法规依据原文，默认 true",
            },
        },
        "required": ["text"],
    },
    "executor": check_tender_compliance,
}


__all__ = ["MAX_CLAUSES", "TOOL_SPEC", "check_tender_compliance"]
