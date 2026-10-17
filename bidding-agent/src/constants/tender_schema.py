"""招标文件的结构化字段 schema（抽取 / 前端 / 变更追踪三处共用）。

与 `bid_schema`（标讯公告）的区别
--------------------------------
两者都是招投标领域，但**抽取对象与用途完全不同**，不能合并：

    `bid_schema`    针对**中标结果公告**——一页纸，17 个平铺字段，
                    用途是入库做统计分析（金额、供应商、频次）
    `tender_schema` 针对**招标文件正文**——几十页，含多个**列表型章节**
                    （资质要求、评分标准、废标条款、时间节点），
                    用途是投标决策（我能不能投、怎么投、报价多少）

所以本模块的字段分两类：
    - `TenderField`      标量字段（项目名称、预算、开标时间…）
    - `TenderSection`    **列表型章节**（资质要求、评分项、废标条款…）

后者是本模块存在的理由：把"资格要求"压成一个字符串毫无用处，
必须逐条拆开，才能与"企业资质表"逐条比对（见 ⑤ 资格审查辅助）。

为什么把"废标条款"与"评分标准"单列
--------------------------------
这两个章节是**投标决策中风险最高、最容易漏看**的部分：

    - 废标条款：漏看一条 = 直接出局（比报价高低的代价大得多）
    - 评分标准：决定报价策略（价格分占比多少、技术分怎么给）

早期设计把它们混在"其他要求"里，实测抽取时几乎总被漏掉。
单列后提示词会专门强调，且前端能单独高亮。

与 ⑧ 变更追踪的关系
------------------
变更追踪靠**逐字段 diff** 同一项目的两份文件（原文件 vs 澄清/更正公告）。
故字段必须**稳定且可比较**：同一含义只能有一个 key，不能出现
"budget" 与 "budget_amount" 并存——那样 diff 会误报为"变更"。
`field_keys()` / `section_keys()` 就是给 diff 用的稳定标识。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 字段类型：text 原样 / money 转数值 / datetime 归一日期
FieldType = str  # "text" | "money" | "datetime"


@dataclass(frozen=True)
class TenderField:
    """招标文件的标量字段。"""

    key: str
    label: str
    llm_name: str
    type: FieldType = "text"
    hint: str = ""
    required: bool = False


@dataclass(frozen=True)
class TenderSection:
    """招标文件的列表型章节（每项是一条独立要求/评分项/废标情形）。"""

    key: str  # 英文键（有对应表列时与列名一致）
    label: str
    llm_name: str
    hint: str = ""
    # 每项额外抽取的子字段（如资质要求的「是否硬性门槛」、评分项的「分值」）
    item_fields: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 标量字段
# ---------------------------------------------------------------------------

TENDER_FIELDS: tuple[TenderField, ...] = (
    TenderField("project_name", "项目名称", "项目名称", hint="去掉「招标文件」等后缀的正式项目名", required=True),
    TenderField("project_id", "项目编号", "项目编号", hint="招标编号/项目编号", required=True),
    TenderField("purchaser", "采购人", "采购人", hint="招标人/采购单位全称"),
    TenderField("agency", "代理机构", "代理机构", hint="招标代理机构，没有则留空"),
    TenderField("budget", "预算金额", "预算金额", "money", hint="只填数字，单位元；含最高限价时也填这里"),
    TenderField("max_price", "最高限价", "最高限价", "money", hint="如与预算不同则单独填，只填数字"),
    TenderField("deposit", "投标保证金", "投标保证金", "money", hint="只填数字，单位元"),
    TenderField("bid_deadline", "投标截止时间", "投标截止时间", "datetime", hint="格式 YYYY-MM-DD HH:MM"),
    TenderField("open_time", "开标时间", "开标时间", "datetime", hint="格式 YYYY-MM-DD HH:MM"),
    TenderField("bid_open_place", "开标地点", "开标地点"),
    TenderField("contact", "联系方式", "联系方式", hint="联系人姓名与电话"),
    TenderField("duration", "服务期/工期", "服务期", hint="如 12 个月 / 90 日历天"),
    TenderField("payment_terms", "付款方式", "付款方式", hint="如 验收合格后 30 日内付 95%"),
    TenderField("bid_validity", "投标有效期", "投标有效期", hint="如 90 日历天"),
    # 下面两个是**投标决策的核心**，单列以免被漏抽
    TenderField("evaluation_method", "评标办法", "评标办法", hint="如 综合评分法 / 最低评标价法"),
    TenderField("price_weight", "价格分权重", "价格分权重", hint="如「价格分 30 分」中的 30，只填数字"),
)


# ---------------------------------------------------------------------------
# 列表型章节（本模块的重点）
# ---------------------------------------------------------------------------

TENDER_SECTIONS: tuple[TenderSection, ...] = (
    TenderSection(
        "qualifications",
        "资格要求",
        "资格要求",
        hint="投标人必须满足的资质与条件，逐条拆分",
        item_fields=("text", "category", "is_mandatory"),
    ),
    TenderSection(
        "scoring_items",
        "评分标准",
        "评分标准",
        hint="评分项逐条拆分，注明分值",
        item_fields=("text", "score", "category"),
    ),
    TenderSection(
        "disqualification",
        "废标条款",
        "废标条款",
        hint="导致投标无效/废标的情形，逐条拆分。**这一项最容易被漏，务必仔细找**",
        item_fields=("text",),
    ),
    TenderSection(
        "technical_requirements",
        "技术要求",
        "技术要求",
        hint="技术规格、参数、服务要求要点，逐条拆分",
        item_fields=("text",),
    ),
    TenderSection(
        "required_documents",
        "需提交材料",
        "需提交材料",
        hint="投标文件须包含的材料清单，逐条拆分",
        item_fields=("text",),
    ),
)


# ---------------------------------------------------------------------------
# 派生工具（抽取提示词 / 归一化 / 变更追踪共用）
# ---------------------------------------------------------------------------

def field_keys() -> tuple[str, ...]:
    return tuple(f.key for f in TENDER_FIELDS)


def section_keys() -> tuple[str, ...]:
    return tuple(s.key for s in TENDER_SECTIONS)


def text_fields_for_display() -> tuple[tuple[str, str], ...]:
    """`(key, label)` 序列，顺序即前端展示顺序。"""
    return tuple((f.key, f.label) for f in TENDER_FIELDS)


def sections_for_display() -> tuple[tuple[str, str], ...]:
    return tuple((s.key, s.label) for s in TENDER_SECTIONS)


def extraction_prompt() -> str:
    """生成招标文件抽取提示词。

    与 `bid_schema.extraction_prompt()` 同一思路：**由字段定义自动生成**，
    避免手写 prompt 漏字段。

    另外两条针对性要求：
      1. 明确要求"列表型章节逐条拆分"——否则模型倾向压成一段话
      2. 点名"废标条款最容易漏"——实测不加这句时该章节检出率明显偏低
    """
    lines = [
        "你是招投标文件结构化抽取助手。请从下面提供的**招标文件**内容中提取信息，",
        "严格只输出一个 JSON 对象，不要任何解释文字，不要用 Markdown 代码块包裹。",
        "",
        "【一、标量字段】键名必须完全一致：",
    ]
    for f in TENDER_FIELDS:
        hint = f"（{f.hint}）" if f.hint else ""
        star = "【必填】" if f.required else ""
        lines.append(f'- "{f.llm_name}"{star}：{f.label}{hint}')

    lines += [
        "",
        "【二、列表型章节】每项是**数组**，逐条拆分，不要把多条压成一条：",
    ]
    for s in TENDER_SECTIONS:
        desc = "、".join(s.item_fields)
        lines.append(f'- "{s.llm_name}"：{s.label}（{s.hint}）')
        lines.append(f'    数组元素含字段：{desc}')

    lines += [
        "",
        "抽取规则：",
        "1. 找不到的标量字段填空字符串，列表章节填 `[]`，**不要编造**；",
        "2. 金额只填数字（不含「元」、不含千分位逗号），如 1250000；",
        "3. 日期尽量补全为 YYYY-MM-DD HH:MM；只有日期时补 00:00；",
        "4. 资格要求里区分**硬性门槛**（不满足即废标，is_mandatory=true）",
        "   与**加分项**（满足可加分，is_mandatory=false）；判断不了填 true；",
        "   资格要求的 category 从「资质等级/业绩要求/财务要求/人员要求/其他」中选择；",
        "5. 评分项的 score 只填该小项的分值数字；category 从「技术/商务/价格/其他」中选择；",
        "6. **废标条款务必仔细找**，常见于「投标无效」「废标」「不予受理」「否决投标」等表述；",
        "7. 某项在文件中反复出现时只保留一处，不要重复。",
    ]
    return "\n".join(lines)


# 归一化辅助

_MONEY_VALUE_RE = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def _to_money(value: Any) -> int | float | None:
    """金额归一：**只取第一个数值**，按就近的中文单位换算。

    与 `bid_schema._to_money` 必须行为一致——两处若不同，
    同一笔金额在标讯与招标文件里会变成不同数值，跨模块比对会误报变更。

    ★ 实测踩坑：原实现"删掉所有非数字字符再拼接"，含多个数字的文本会拼出荒谬金额：
        '25000元（含税13%）'           → 2500013
        '5 万元（保证金 50,000.00 元）' → 5,500,000,000（55 亿）
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    match = _MONEY_VALUE_RE.search(text)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None

    # 单位只看数值**之后**紧邻的一小段，避免把句尾另一个金额的单位套上来
    tail = text[match.end() : match.end() + 3]
    if "亿" in tail:
        number *= 100_000_000
    elif "万" in tail:
        number *= 10_000
    else:
        prefix = text[: match.start()].rstrip()
        if prefix.endswith("亿"):
            number *= 100_000_000
        elif prefix.endswith("万"):
            number *= 10_000

    if abs(number - round(number)) < 1e-9:
        return int(round(number))
    return round(number, 2)


def _to_datetime(value: Any) -> str:
    """日期归一成 `YYYY-MM-DD HH:MM:SS`。

    ★ 必须保留时分（实测踩坑）
    ------------------------
    第一版对中文日期只捕获年月日，于是 `2026年11月20日 09:30`
    被归一成 `2026-11-20 00:00:00` —— **时分被丢掉**。

    对招标文件而言这是严重问题：投标截止时间差几小时就意味着废标。
    故现在中英文两种写法都单独剥离时间部分。

    认不出的原样返回（保留原文比丢信息更好排查）。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    # 通用时间部分：HH:MM 或 HH:MM:SS（半角/全角冒号都收）
    time_re = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})(?:\s*[:：]\s*(\d{2}))?")

    def _with_time(date_part: str, rest: str) -> str:
        tm = time_re.search(rest)
        if tm:
            hh, mm, ss = tm.group(1), tm.group(2), tm.group(3) or "00"
            return f"{date_part} {int(hh):02d}:{mm}:{ss}"
        return f"{date_part} 00:00:00"

    # ① ISO 风格：2026-11-20 / 2026-11-20 09:30 / 2026-11-20T09:30:00
    #    月份与日**允许未补零**（2026-1-5）——实测模型会这么输出，
    #    而 `strptime("%m")` 与 `fromisoformat` 都不接受未补零，
    #    若不处理会原样返回字符串，与补零写法比对时误报为"变更"。
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(.*)$", text)
    if m:
        y, mo, d, rest = m.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest)

    # ② 中文日期：2026年11月20日（可带时间）
    m2 = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?(.*)$", text)
    if m2:
        y, mo, d, rest = m2.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest or "")

    # ③ 斜杠/点分隔：2026/11/20
    m3 = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})(.*)$", text)
    if m3:
        y, mo, d, rest = m3.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest or "")

    # ④ 只有年月：2026年11月
    m4 = re.match(r"^(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*月?$", text)
    if m4:
        y, mo = m4.groups()
        return f"{int(y):04d}-{int(mo):02d}-01 00:00:00"

    # ⑤ 兜底：交给 fromisoformat（能处理 2026-11-20T09:30:00+08:00 这类）
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


def normalize_tender(raw: dict[str, Any]) -> dict[str, Any]:
    """把模型输出的中文键 JSON 归一成稳定结构。

    返回 `{"fields": {...}, "sections": {key: [item, ...]}}`。

    为什么把 fields 与 sections 分开：两者在 diff（变更追踪）里的比较方式不同——
    fields 逐键比字符串，sections 要按条目做集合比对。
    混在一个 dict 里会让 diff 逻辑到处判类型。
    """
    fields: dict[str, Any] = {}
    for f in TENDER_FIELDS:
        value = raw.get(f.llm_name)
        if value is None:
            value = raw.get(f.key)
        if f.type == "money":
            fields[f.key] = _to_money(value)
        elif f.type == "datetime":
            fields[f.key] = _to_datetime(value)
        else:
            fields[f.key] = "" if value is None else str(value).strip()

    sections: dict[str, list[dict[str, Any]]] = {}
    for s in TENDER_SECTIONS:
        value = raw.get(s.llm_name)
        if value is None:
            value = raw.get(s.key)
        items: list[dict[str, Any]] = []
        if isinstance(value, list):
            for raw_item in value:
                if isinstance(raw_item, dict):
                    item = {k: str(raw_item.get(k, "") or "").strip() for k in s.item_fields}
                    # is_mandatory 保持布尔语义，便于前端区分硬性门槛/加分项
                    if "is_mandatory" in item:
                        v = item["is_mandatory"].lower()
                        item["is_mandatory"] = v not in ("false", "0", "否", "no", "")
                    if "score" in item:
                        # 分值常被模型写成 "10分" / "10 分" / "满分10分"。
                        # 原实现 `float()` 失败就吞成 None，前端只显示空白且无告警——
                        # 而分值是评分标准里最关键的数值，静默丢失会让用户以为文件没写分值。
                        # 故回退取**前导数字**；确实没有数字才置 None。
                        raw_score = str(item["score"]).strip() if item["score"] else ""
                        if raw_score:
                            sm = re.search(r"\d+(?:\.\d+)?", raw_score)
                            item["score"] = float(sm.group(0)) if sm else None
                        else:
                            item["score"] = None
                    if item.get("text"):
                        items.append(item)
                elif isinstance(raw_item, str) and raw_item.strip():
                    # 模型偶尔只给字符串数组，补成同构 dict
                    items.append({k: (raw_item.strip() if k == "text" else "") for k in s.item_fields})
        sections[s.key] = items

    return {"fields": fields, "sections": sections}


def completeness(parsed: dict[str, Any]) -> float:
    """字段完整度 0~1。必填字段权重翻倍。"""
    fields = parsed.get("fields") or {}
    sections = parsed.get("sections") or {}
    total = 0.0
    got = 0.0
    for f in TENDER_FIELDS:
        w = 2.0 if f.required else 1.0
        total += w
        if fields.get(f.key) not in (None, "", []):
            got += w
    # 三个关键章节各算一份权重：评标办法/资格要求/废标条款缺失会实质影响决策
    for s in TENDER_SECTIONS:
        total += 1.0
        if sections.get(s.key):
            got += 1.0
    return round(got / total, 3) if total else 0.0


__all__ = [
    "TENDER_FIELDS",
    "TENDER_SECTIONS",
    "TenderField",
    "TenderSection",
    "completeness",
    "extraction_prompt",
    "field_keys",
    "normalize_tender",
    "section_keys",
    "sections_for_display",
    "text_fields_for_display",
]
