"""招投标标讯的共享字段 schema（抽取 / 存储 / 前端三处共用）。

为什么要单独抽出来
------------------
项目里有三处需要"同一份字段定义"：

    1. **抽取**：LLM / 视觉模型按这份 schema 输出结构化 JSON
    2. **存储**：写入 PostgreSQL 的 `bidding_records` 表
    3. **前端**：表格展示 17 个字段、标中文表头

早期这三处各写一份（`to_postgresql.py` 的 `COLUMN_NAME_MAP`、前端 `Extract.tsx` 的
`FIELDS`、以及散在各处的 prompt），结果是**改一处忘一处**——例如前端加了字段、
抽取 prompt 没跟上，就永远抽不出那一列。

本模块提供**唯一事实来源**，包含：
    - `BID_FIELDS`：字段清单（英文键 / 中文标签 / LLM 抽取名 / 类型 / 说明）
    - `column_map()`：中文表头 → 表列名（供 to_postgresql 与抽取结果入库复用）
    - `extraction_prompt()`：按 schema 生成的抽取提示词（避免手写漏字段）
    - `normalize_record()`：把 LLM 输出归一成可入库的 dict（类型转换 + 空值处理）

字段与 `bidding_records` 表的对应关系见 `COLUMN_NAME_MAP`（与
`src/database/to_postgresql.py` 的同名映射保持一致，本模块是它的上移版本）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# 字段类型：text 原样存 / money 转 Decimal / datetime 转时间戳
FieldType = str  # "text" | "money" | "datetime"


@dataclass(frozen=True)
class BidField:
    """一个标讯字段的定义。"""

    key: str  # 英文键（与 bidding_records 列名一致）
    label: str  # 中文标签（前端表头 / 抽取提示词）
    llm_name: str  # 抽取时用的中文键名（与原始表头一致，模型对中文表头更准）
    type: FieldType = "text"
    hint: str = ""  # 给模型的补充说明（易混字段必填）
    required: bool = False  # 是否为核心字段（缺失会降低整体置信度）


# 17 个字段，顺序与前端 `Extract.tsx` 的 FIELDS 一致。
# `llm_name` 特意用中文：模型在中文语料上对中文键名的抽取准确率更高，
# 且与 `data/processed/*.xlsx` 的原始表头一致，便于人工比对。
BID_FIELDS: tuple[BidField, ...] = (
    BidField("title", "标题", "标题", hint="公告标题原文", required=True),
    BidField("category", "类别", "类别", hint="只能是：货物 / 服务 / 工程"),
    BidField("source", "来源", "来源", hint="只能是：政府采购 / 企业采购 / 招标采购"),
    BidField("publish_time", "发布时间", "发布时间", "datetime", hint="格式 YYYY-MM-DD"),
    BidField("province", "省份", "省份", hint="如 安徽省"),
    BidField("location_city", "市区", "市区", hint="如 合肥市"),
    BidField("county", "县城", "县城", hint="如 庐江县，没有则留空"),
    BidField("project_id", "项目编号", "项目编号", hint="公告中的项目编号/招标编号", required=True),
    BidField("project_name", "项目名称", "项目名称", hint="去掉公告后缀（如中标结果公告）的正式项目名", required=True),
    BidField("purchaser", "采购人", "采购人", hint="采购单位/招标人全称"),
    BidField("agency", "代理机构", "代理机构", hint="招标代理机构，没有则留空"),
    BidField("budget", "预算", "预算", "money", hint="只填数字，单位元；无则留空"),
    BidField("project_address", "项目地址", "项目地址"),
    BidField("duration", "周期", "周期", hint="服务期/工期，如 12个月"),
    BidField("supplier", "中标人", "中标人", hint="中标供应商全称"),
    BidField("amount", "中标金额", "中标金额", "money", hint="只填数字，单位元；无则留空"),
    BidField("bid_time", "中标时间", "中标时间", "datetime", hint="格式 YYYY-MM-DD"),
)


# ---------------------------------------------------------------------------
# 列映射（与 to_postgresql.py 的 COLUMN_NAME_MAP 同源）
# ---------------------------------------------------------------------------

def column_map() -> dict[str, str]:
    """中文表头 → 表列名。供 xlsx 导入与抽取结果入库共用。"""
    mapping = {f.label: f.key for f in BID_FIELDS}
    # 与 to_postgresql.py 保持兼容的别名（同义列指向同一目标名）
    mapping.update(
        {
            "供应商": "supplier",
            "交易额": "amount",
            "核心标的物": "subject_matter",
            "标的物": "subject_matter",
        }
    )
    return mapping


# ---------------------------------------------------------------------------
# 抽取提示词（按 schema 生成，避免手写漏字段）
# ---------------------------------------------------------------------------

def extraction_prompt() -> str:
    """生成标讯抽取提示词。

    生成的提示词输出**中文键名**（与 llm_name 一致），解析侧用
    `normalize_record` 转成英文列名——这样模型只面对它最熟悉的中文表头。
    """
    lines = [
        "你是招投标标讯结构化抽取助手。请从下面提供的公告内容中提取字段，",
        "严格只输出一个 JSON 对象，不要任何解释文字，不要用 Markdown 代码块包裹。",
        "",
        "字段定义（键名必须完全一致）：",
    ]
    for f in BID_FIELDS:
        hint = f"（{f.hint}）" if f.hint else ""
        star = "【必填】" if f.required else ""
        lines.append(f'- "{f.llm_name}"{star}：{f.label}{hint}')

    lines += [
        "",
        "抽取规则：",
        "1. 找不到的字段一律填空字符串，**不要编造、不要推测**；",
        "2. 金额只填数字（不含「元」字、不含千分位逗号），如 1684000；",
        "3. 日期尽量补全为 YYYY-MM-DD；只有年月时填 YYYY-MM；",
        "4. 类别/来源必须从给定的枚举值中选择，无法判断则留空；",
        "5. 项目名称去掉「中标结果公告」「招标公告」等后缀，只保留项目正式名称。",
    ]
    return "\n".join(lines)


def json_keys() -> tuple[str, ...]:
    """抽取结果的 JSON 键名（中文），供解析侧校验。"""
    return tuple(f.llm_name for f in BID_FIELDS)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

# 金额抽取：**只取第一个数值**，并正确处理千分位。
#
# ★ 为什么必须"只取第一个"（实测踩坑，后果严重）
# --------------------------------------------
# 原实现是"删掉所有非数字字符再拼接"，于是含多个数字的文本会拼出荒谬金额：
#
#     '25000元（含税13%）'           → 2500013        （25000 与 13 被拼起来）
#     '5 万元（保证金 50,000.00 元）' → 5,500,000,000  （55 亿！）
#     '预算 120 万，限价 108 万'      → 1,201,080,000  （120 与 108 相乘叠加）
#
# 这些数字会直接进库，污染所有基于金额的统计与报价建议。
# 公告与招标文件里"金额 + 税率 + 限价 + 保证金"同句出现极常见，不是边缘情况。
#
# 千分位用 `(?:\d{1,3}(?:,\d{3})+|\d+)` 一次匹配完，避免把 "1,250,000" 拆成 "1"。
_MONEY_VALUE_RE = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

# 中文数量级单位（取**第一个**出现的，与金额数字就近匹配）
_MONEY_UNIT_RE = re.compile(r"[亿万]")


def _to_money(value: Any) -> int | float | None:
    """金额归一：**只取第一个数值**，按就近的中文单位换算。

    整数返回 int（与 xlsx 导入的 int64 对齐）；无有效数字返回 None——
    **不是 0**。0 会被下游当成"预算为零"这种荒谬结论。
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

    # 单位只看**数值之后紧跟的那一小段**，避免把句尾另一个金额的单位
    # 套到本数值上（"5 万元（保证金 50,000 元）"里 5 的单位才是"万"）
    tail = text[match.end() : match.end() + 3]
    if "亿" in tail:
        number *= 100_000_000
    elif "万" in tail:
        number *= 10_000
    elif "亿" in text[: match.start()] or "万" in text[: match.start()]:
        # 单位写在数字前（如"人民币万元 120"）的少数写法
        prefix = text[: match.start()]
        if prefix.rstrip().endswith("亿"):
            number *= 100_000_000
        elif prefix.rstrip().endswith("万"):
            number *= 10_000

    if abs(number - round(number)) < 1e-9:
        return int(round(number))
    return round(number, 2)


def _to_datetime(value: Any) -> str:
    """日期归一成 `YYYY-MM-DD HH:MM:SS`（与 xlsx 导入的格式一致）。

    ★ 必须与 `tender_schema._to_datetime` **行为完全一致**
    ----------------------------------------------------
    两个模块处理的是同一批招投标文本，同一日期若得到不同结果，
    跨模块比对（如变更追踪 diff）会把"格式差异"误报成"内容变更"。
    实测已踩坑：本模块原实现写死 `\\d{2}`，`2026-1-5 09:30` 匹配不上，
    落到中文日期分支只取年月日 → **时分被丢掉**，而招标文件那边保留了时分。

    为什么必须补全时分秒：`bidding_records.publish_time` 是 TIMESTAMPTZ，
    只给 `2023-10-13` 也能插进去，但与 xlsx 导入的 `2023-10-13 00:00:00`
    格式不一致，按时间排序/分组时会出现"同一天的两条记录看起来不同"。

    支持：ISO（含未补零）、中文日期、斜杠/点分隔、带时区、只有年月。
    认不出的原样返回——**宁可保留原文也不要丢信息**。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    # 通用时间部分：HH:MM 或 HH:MM:SS（半角/全角冒号都收）
    time_re = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})(?:\s*[:：]\s*(\d{2}))?")

    def _with_time(date_part: str, rest: str) -> str:
        tm = time_re.search(rest or "")
        if tm:
            hh, mm, ss = tm.group(1), tm.group(2), tm.group(3) or "00"
            return f"{date_part} {int(hh):02d}:{mm}:{ss}"
        return f"{date_part} 00:00:00"

    # ① ISO 风格（月份与日允许未补零）
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(.*)$", text)
    if m:
        y, mo, d, rest = m.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest)

    # ② 中文日期：2023年10月13日（可带时间）
    m2 = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?(.*)$", text)
    if m2:
        y, mo, d, rest = m2.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest or "")

    # ③ 斜杠/点分隔
    m3 = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})(.*)$", text)
    if m3:
        y, mo, d, rest = m3.groups()
        return _with_time(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", rest or "")

    # ④ 只有年月
    m4 = re.match(r"^(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*月?$", text)
    if m4:
        y, mo = m4.groups()
        return f"{int(y):04d}-{int(mo):02d}-01 00:00:00"

    # ⑤ 兜底：交给 fromisoformat（能处理带时区等写法）
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


_CATEGORY_ALIASES = {
    "货物": "货物",
    "服务": "服务",
    "工程": "工程",
    "goods": "货物",
    "service": "服务",
    "works": "工程",
}
_SOURCE_ALIASES = {
    "政府采购": "政府采购",
    "企业采购": "企业采购",
    "招标采购": "招标采购",
    "gov": "政府采购",
}


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 输出的中文键 JSON 归一成可入库的英文键 dict。

    - 未知键丢弃（模型偶尔会加自己的字段）
    - 缺失键补空字符串（下游 SQL 不必到处判 None）
    - 金额/日期按类型转换
    - 类别/来源做枚举归一（模型可能返回同义词）
    """
    out: dict[str, Any] = {}
    for f in BID_FIELDS:
        value = raw.get(f.llm_name)
        if value is None:
            # 兼容模型偶尔用英文键返回
            value = raw.get(f.key)

        if f.type == "money":
            out[f.key] = _to_money(value)
        elif f.type == "datetime":
            out[f.key] = _to_datetime(value)
        else:
            text = "" if value is None else str(value).strip()
            if f.key == "category":
                text = _CATEGORY_ALIASES.get(text, text)
            elif f.key == "source":
                text = _SOURCE_ALIASES.get(text, text)
            out[f.key] = text
    return out


def completeness(record: dict[str, Any]) -> float:
    """计算字段完整度（0-1），用于前端提示"这次抽取质量如何"。

    权重：必填字段缺失扣得更重——一份连项目名称、编号都没抽出来的结果，
    即使其他字段都填了也不可信。
    """
    if not BID_FIELDS:
        return 0.0
    total = 0.0
    got = 0.0
    for f in BID_FIELDS:
        weight = 2.0 if f.required else 1.0
        total += weight
        value = record.get(f.key)
        if value not in (None, "", []):
            got += weight
    return round(got / total, 3) if total else 0.0


def text_fields_for_display() -> tuple[tuple[str, str], ...]:
    """前端展示用的 `(key, label)` 序列（顺序即展示顺序）。"""
    return tuple((f.key, f.label) for f in BID_FIELDS)


__all__ = [
    "BID_FIELDS",
    "BidField",
    "column_map",
    "completeness",
    "extraction_prompt",
    "json_keys",
    "normalize_record",
    "text_fields_for_display",
]
