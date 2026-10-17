"""招标文件变更追踪：原招标文件 vs 澄清/更正公告 → 按后果分级的变更清单。

业务场景（本模块存在的理由）
--------------------------
招标项目在投标截止前常发 **澄清公告 / 更正公告 / 补遗书**，写法通常是
"原招标文件第 3 章第 2 条修改为……"。投标员最怕的失误链是：

    漏看澄清公告 → 按旧文件编标 → 开标时被认定"未响应招标文件" → **废标**

废标的代价远大于报价高低，所以本模块的输出不是"两段文本的差异"，
而是**按后果分级**的变更清单：哪些必须重做标书、哪些只需评估、哪些只是信息更新。

与既有模块的边界
---------------
    document_parser   文件 → 文本 + 章节分块        （不管"改了什么"）
    tender_extractor  文本 → 结构化字段 / 列表章节
    change_tracker    两份结构化结果 → 变更清单      ← 本模块

本模块**不调 LLM、不解析文件**，那是上面两个模块的职责。故 `diff_parsed()`
是纯函数，能脱离 LLM 与文件独立单测——这是刻意设计：变更判定是否漏报，
必须靠稳定的单测锁住，而 LLM 输出不能当回归基线。

关于"复用 extract_from_document"
-------------------------------
`extract_from_document()` 就是 `parse_document()` + `extract_from_parsed()`。
本模块要把"解析"与"抽取"拆成两个事件上报（前端要分别显示"已解析 12 页"
与"正在抽取第 2/5 部分"），故直接调用这两个函数——**抽取逻辑仍然只有
tender_extractor 一份实现**，这里不重复批循环。

判级依据（为什么不是"文本不同就报警"）
------------------------------------
把所有差异一律标红等于没有分级：投标员会失去判断力，进而忽略全部告警。
故按**后果**分三级，依据是招投标实务：

    critical  钱、时间、资格、废标条款、评分标准变了 → 标书必须重做
              （报价算错或资格不符 = 直接出局，没有补救机会）
    major     工期、付款、有效期、评标办法、技术要求变了 → 需评估影响
              （影响成本与方案，但通常可通过澄清或说明处理）
    minor     联系人、开标地点、单位名称等 → 信息性，改一下抬头即可

同一项变更对报价的影响还有**方向**：最高限价**下调**（利润空间被压缩）比
上调严重得多；投标截止**提前**比延后危险。方向不改变 severity（钱变了就是
critical），但会写进 `impact_notes` —— 投标员据此决定"重新报价"还是
"争取更多时间"。

归一化：为什么 diff 之前还要再归一一次
------------------------------------
`normalize_tender()` 已把金额转成数值、日期补成 `YYYY-MM-DD HH:MM:SS`，
但 `diff_parsed()` 的输入不保证来自抽取器：也可能是数据库里的历史快照、
手工构造的测试数据或前端回传的 JSON。两个入口归一程度不一致，就会出现
"`125万元` vs `1250000` 被判成变更"这类误报；误报多了用户就不看告警了，
比不报更危险。故这里对输入**再做一次幂等归一**（复用同一个 `normalize_tender`，
不另写一套），保证比较基准唯一。

比对范围的诚实说明：只比 schema 覆盖的 16 个标量字段与 5 个列表章节。
附件、图纸、格式文件、评标细则的隐含口径不在比对范围内。
"""

from __future__ import annotations

import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterator

from src.constants.tender_schema import (
    TENDER_FIELDS,
    TENDER_SECTIONS,
    normalize_tender,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class ChangeTrackError(RuntimeError):
    """变更追踪失败（文件解析失败 / 抽取失败 / 待比对的数据结构非法）。"""


# ---------------------------------------------------------------------------
# 严重度定义与判级规则
# ---------------------------------------------------------------------------

CRITICAL = "critical"
MAJOR = "major"
MINOR = "minor"

_SEVERITY_RANK = {CRITICAL: 3, MAJOR: 2, MINOR: 1}

# 键 → 严重度。key 同时覆盖**标量字段**与**列表章节**（两者在 diff 里
# 都是"变更主体"，前端也按同一套颜色渲染）。没列出的键走关键词兜底
# （`_severity_by_keyword`），故将来 schema 新增字段不会静默降级成 minor。
_SEVERITY: dict[str, str] = {
    # —— critical：钱 / 时间 / 资格 / 废标 / 评分 ——
    # 截止时间与开标时间变了，编制排期、保证金到账时间、授权文件签署日期全部跟着变，
    # 错过时间点直接失去投标资格，故与"钱"同级。
    "bid_deadline": CRITICAL,
    "open_time": CRITICAL,
    # 超过最高限价即废标（投标报价超过限价是最常见的废标情形之一），
    # 且它直接决定报价上限，属"必须重做报价"的变更。
    "max_price": CRITICAL,
    "budget": CRITICAL,
    # 保证金金额变了要重新走财务流程；少缴或形式不符会被否决。
    "deposit": CRITICAL,
    # 价格分权重决定报价策略（权重越高越要压价），与评分标准同源，故同样 critical。
    "price_weight": CRITICAL,
    # 资格门槛变了，先要回答"我方还能不能投"；废标条款变了，标书要逐条重核。
    "qualifications": CRITICAL,
    "disqualification": CRITICAL,
    "scoring_items": CRITICAL,
    # —— major：影响成本与方案，但一般不直接出局 ——
    "duration": MAJOR,
    "payment_terms": MAJOR,
    "bid_validity": MAJOR,
    "evaluation_method": MAJOR,
    "technical_requirements": MAJOR,
    # 漏交材料确实会废标，但它属于"照单核对就能解决"，与资格不符不同，
    # 故列 major：需要逐项核对，而不必推翻方案重做。
    "required_documents": MAJOR,
    # —— minor：信息性，改抬头/联系方式即可 ——
    "contact": MINOR,
    "bid_open_place": MINOR,
    "purchaser": MINOR,
    "agency": MINOR,
    # 名称/编号一般是"写错了更正"。若真换了项目，`_identity_notes` 会额外告警，
    # 因为那种情况下本次比对结果不成立。
    "project_name": MINOR,
    "project_id": MINOR,
}

# 关键词兜底（仅用于 schema 之外的键）。原则：**宁可报高不报低**——
# 漏报一个 critical 的代价是废标，误报一个的代价只是让人多看一眼。
_KEYWORD_SEVERITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        CRITICAL,
        (
            "deadline", "open_time", "price", "amount", "budget", "deposit",
            "qualification", "disqualif", "scoring", "score",
            "截止", "开标", "限价", "预算", "保证金", "资格", "资质", "废标",
            "无效", "否决", "评分", "分值", "价格分", "实质性",
        ),
    ),
    (
        MAJOR,
        (
            "duration", "payment", "validity", "evaluat", "technical", "requirement",
            "工期", "服务期", "付款", "有效期", "评标", "技术", "需求", "材料",
        ),
    ),
)

# 展示/排序用的派生表
_FIELD_TYPE = {f.key: f.type for f in TENDER_FIELDS}
_FIELD_ORDER = {f.key: i for i, f in enumerate(TENDER_FIELDS)}
_FIELD_KEYS = frozenset(f.key for f in TENDER_FIELDS)
_SECTION_LABEL = {s.key: s.label for s in TENDER_SECTIONS}
_SECTION_ITEM_FIELDS = {s.key: s.item_fields for s in TENDER_SECTIONS}


def classify_severity(key: str, old: Any = None, new: Any = None) -> str:
    """判定某个字段/章节变更的严重度，返回 `critical` / `major` / `minor`。

    `old` / `new` 参与判定不是摆设：

      1. **已知键**直接查规则表，涨跌不改变结论——最高限价上调与下调都是
         critical（都让报价上限变了），"更严重"的那一侧由 `impact_notes`
         的方向描述体现（"下调…需重新核算报价"）。
      2. **未知键**（schema 演进出新字段、或调用方传了别的键名）用关键词兜底，
         此时要看变更内容：新值里出现"废标""限价""保证金"等字样的一律按
         critical 处理。宁可报高不报低——漏报 critical 的代价是废标。

    故本函数对同一 key 可以稳定返回同一级别，便于前端按颜色分组。
    """
    name = (key or "").strip()
    known = _SEVERITY.get(name)
    if known is not None:
        return known
    return _severity_by_keyword(name, f"{_plain(old)} {_plain(new)}")


def _severity_by_keyword(key: str, text: str = "") -> str:
    haystack = f"{key} {text}".lower()
    for severity, words in _KEYWORD_SEVERITY:
        if any(word in haystack for word in words):
            return severity
    return MINOR


# ---------------------------------------------------------------------------
# 归一化（比较专用的"比较键"）
# ---------------------------------------------------------------------------

# 全角 ASCII（０-９、（）、：、，…）→ 半角。Word 导出的招标文件全角标点极常见，
# 不统一会让 "（1）" 与 "(1)"、"１２" 与 "12" 被判成两条不同要求。
_FULLWIDTH = str.maketrans({chr(c): chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)})
# 中文标点单独映射（U+3000 系列不在上面那个区间里）
_CJK_PUNCT = str.maketrans({"、": ",", "。": ".", "；": ";", "：": ":", "　": " "})

# 千分位：比较前去掉，"1,200,000" 与 "1200000" 是同一笔钱。
# 必须先于"剥序号"执行，否则 "1,200,000 元" 的前导 "1," 会被当成序号前缀。
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

# 序号前缀。招标文件的编号体系非常乱，澄清公告里更是 "第3.2条"、"3.2.1"、
# "（二）"、"2、"、"1）" 混用；不剥掉的话"同一要求换了编号"会全是误报。
#
# 两处刻意的取舍：
#   - 末尾的 `(?!\d)` 防止把 "99.95%可用性" 的 "99." 当序号剥掉（那会让
#     "99.95%可用性" 与 "99.9%可用性" 都被剥成同一个键 → **漏报**）
#   - 不剥 "★"、"▲"、"#"：招标文件用它们标注"实质性要求"，
#     星号被去掉本身就是实质性变化，不能归一掉
_ORDINAL_RE = re.compile(
    r"^(?:"
    r"第\s*[一二三四五六七八九十百零〇\d.]+\s*[章节条款项部分]"
    r"|[（(]\s*[一二三四五六七八九十百零〇\d]+\s*[)）]"
    r"|[一二三四五六七八九十]{1,3}\s*[、.．,，:：]"
    r"|\d+(?:\.\d+)+\s*(?=[\u4e00-\u9fa5])"
    r"|\d+(?:\.\d+)*\s*[、.．,，:：)）](?!\d)"
    r")"
)

# 尾部的句读差异不是变更："具有独立法人资格。" 与 "…资格" 是同一要求。
_TRAILING_PUNCT = "。.；;，,、:：!！?？/\\|"

# 占位值等价于"未填"：抽取器对缺失字段可能给 null、""、"暂无"、"无"…
# 若把 None 与 "" 当成变更，diff 会被这类噪音淹没。
_PLACEHOLDERS = frozenset(
    {"", "-", "--", "—", "－", "/", "无", "暂无", "未提供", "不详", "待定",
     "n/a", "na", "null", "none", "nil", "未填写"}
)

# 日期时间比较键的识别式。允许未补零（2026-1-5）与空格/T 分隔。
#
# 注意这里**在去空白之前**匹配：`_to_datetime()` 认不出的写法（如 "2026-1-5"）
# 会原样返回，若先去空白会变成 "2026-1-509:30:00"，日与小时黏在一起没法拆。
_DATETIME_PARTS_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?"
)


def _norm_text(value: Any) -> str:
    """把任意值归一成"比较键"字符串（**只用于比较，输出里仍给原文**）。

    归一顺序有依赖，不能随意调换：
        全角→半角 → 去千分位 → 去空白 → 剥序号（最多两层）→ 去尾部标点 → 小写
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.translate(_FULLWIDTH).translate(_CJK_PUNCT)
    text = _THOUSANDS_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    for _ in range(2):  # "（一）1. 具有…" 这类双层编号
        stripped = _ORDINAL_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    text = text.strip(_TRAILING_PUNCT).lower()
    return "" if text in _PLACEHOLDERS else text


def _datetime_key(value: Any) -> str:
    """日期时间的比较键：补齐零与缺省时分后再比。

    为什么要单独做：`_to_datetime()` 对 "2026-1-5" 这类未补零写法认不出来、
    会原样返回，于是 "2026-1-5" 与 "2026-01-05" 会被当成两个不同的时间
    ——投标截止时间因此报出一条假变更，正是本模块最不该犯的错。
    没给时间的按 00:00:00 处理（与 `_to_datetime()` 的补全规则一致）。
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = text.translate(_FULLWIDTH).translate(_CJK_PUNCT)
    match = _DATETIME_PARTS_RE.match(text)
    if not match:
        # 认不出的写法（"另行通知"）退回文本归一，至少能消掉标点与空白差异
        return _norm_text(value)
    year, month, day, hour, minute, second = match.groups()
    # 没给时分的补 00:00:00：`_to_datetime()` 对 "2026-11-20" 就是补成
    # "2026-11-20 00:00:00"，比较键必须与它一致，否则同一天会因写法不同报变更
    return (
        f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
        f"{int(hour or 0):02d}:{minute or '00'}:{second or '00'}"
    )


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(_norm_text(value).lstrip("¥￥$"))
    except (TypeError, ValueError):
        return None


def _scalar_key(key: str, value: Any) -> str:
    """标量字段的"比较键"。

    金额按**数值**比（`int 1250000` 与 `float 1250000.0`、`"125万元"` 归一后
    同键），日期按补零后的字符串比，其余按 `_norm_text` 比。
    返回空串表示"未填"——这是空值等价（None / "" / "暂无"）的唯一落点。
    """
    kind = _FIELD_TYPE.get(key, "text")
    if kind == "money":
        number = _as_number(value)
        return "" if number is None else f"{number:.2f}"
    if kind == "datetime":
        return _datetime_key(value)
    return _norm_text(value)


def _plain(value: Any, limit: int = 200) -> str:
    """给关键词兜底用的纯文本（截断，避免拿整段技术要求去匹配）。"""
    if value is None:
        return ""
    return str(value)[:limit]


def _short(value: Any, limit: int = 60) -> str:
    """`impact_notes` 里的值展示：压空白 + 截断，避免整段技术要求淹没提示。"""
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _display(value: Any) -> Any:
    """给前端看的原值：缺失统一成空串（前端不必再判 null），
    整数值的 float 收成 int（1250000.0 → 1250000，避免显示成小数）。"""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _fmt_number(value: Any) -> str:
    number = _as_number(value)
    if number is None:
        return _short(value)
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.2f}"


# ---------------------------------------------------------------------------
# 输入规整：统一成 `{"fields": {...}, "sections": {...}}` 的归一形态
# ---------------------------------------------------------------------------


def _coerce_sections(raw_sections: dict[str, Any], role: str) -> dict[str, Any]:
    """章节值必须是条目数组；容忍"整章只写了一条要求"的字符串简写。

    结构非法时**明确报错而不是当成空**：静默当成空会得出"该章节没有变更"，
    在这个业务里这是最危险的方向（漏报变更 → 废标）。
    """
    coerced: dict[str, Any] = {}
    for key, value in raw_sections.items():
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                coerced[key] = [value]
            continue
        if isinstance(value, (list, tuple)):
            coerced[key] = list(value)
            continue
        raise ChangeTrackError(
            f"{role}的章节「{key}」必须是条目数组，实际是 {type(value).__name__}"
        )
    return coerced


def _canonical(parsed: Any, role: str) -> dict[str, Any]:
    """把一份抽取结果规整成可比较的归一形态。

    复用 `normalize_tender`（幂等）而不是自写一套归一：两套归一逻辑迟早会漂移，
    而"抽取出来的值"与"diff 认为相等的值"必须是同一个定义。
    """
    if not isinstance(parsed, dict):
        raise ChangeTrackError(
            f"{role}的抽取结果必须是 dict，实际是 {type(parsed).__name__}"
        )

    raw_fields = parsed.get("fields")
    if raw_fields is None:
        # 容忍直接给一层字段快照的调用方式（如从数据库取出的历史值）
        raw_fields = {k: v for k, v in parsed.items() if k in _FIELD_KEYS}
    if not isinstance(raw_fields, dict):
        raise ChangeTrackError(f"{role}的 fields 必须是 dict")

    raw_sections = parsed.get("sections")
    if raw_sections is None:
        raw_sections = {}
    if not isinstance(raw_sections, dict):
        raise ChangeTrackError(f"{role}的 sections 必须是 dict")

    canonical = normalize_tender(raw_fields)
    canonical["sections"] = normalize_tender(
        _coerce_sections(raw_sections, role)
    )["sections"]
    return canonical


# ---------------------------------------------------------------------------
# 字段 diff
# ---------------------------------------------------------------------------


def _diff_fields(
    old_fields: dict[str, Any],
    new_fields: dict[str, Any],
    *,
    new_is_patch: bool,
) -> list[dict[str, Any]]:
    """逐字段比对。空值等价与格式差异都在比较键里消掉，不会进结果。"""
    changes: list[dict[str, Any]] = []
    for field in TENDER_FIELDS:
        old_value = old_fields.get(field.key)
        new_value = new_fields.get(field.key)
        old_key = _scalar_key(field.key, old_value)
        new_key = _scalar_key(field.key, new_value)
        if old_key == new_key:
            # 两边都没填（None / "" / "暂无" 等价），或归一后完全相同
            # （"125万元" vs "1250000"、"张工 0551…" vs "张工0551…"）
            continue
        if not new_key:
            # 补遗公告只列改动过的东西，"新文件没提"不等于"被删掉了"
            if new_is_patch:
                continue
            change_type = "removed"
        elif not old_key:
            change_type = "added"
        else:
            change_type = "modified"
        changes.append(
            {
                "key": field.key,
                "label": field.label,
                "old": _display(old_value),
                "new": _display(new_value),
                "change_type": change_type,
                "severity": classify_severity(field.key, old_value, new_value),
            }
        )
    # 危险项排前面：投标员按顺序读，先看到必须重做标书的那几条
    changes.sort(
        key=lambda c: (
            -_SEVERITY_RANK.get(c["severity"], 0),
            _FIELD_ORDER.get(c["key"], len(_FIELD_ORDER)),
        )
    )
    return changes


# ---------------------------------------------------------------------------
# 列表章节 diff
# ---------------------------------------------------------------------------


def _index_items(items: Any) -> dict[str, dict[str, Any]]:
    """按归一化文本给章节条目建索引（键 → 条目）。

    同一要求重复出现只保留第一条：抽取器已按近似规则去重，这里再兜一次，
    避免"同一句话出现两遍"被算成两条变更。
    """
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return index
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _norm_text(item.get("text", ""))
        if not key:
            continue  # 空白 / "无" 这类占位条目不是一条要求
        index.setdefault(key, item)
    return index


def _score_changed(old_item: dict[str, Any], new_item: dict[str, Any]) -> bool:
    """同一要求的**分值**是否变了。

    只在两边都抽到分值时判定：一边抽到 10 分、另一边没抽到分值是抽取噪音
    （远比重分调整常见），把它报成"评分标准变更"会制造大量假警报。
    """
    old_score = _as_number(old_item.get("score"))
    new_score = _as_number(new_item.get("score"))
    if old_score is None or new_score is None:
        return False
    return abs(old_score - new_score) > 1e-9


def _pick(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """只输出该章节声明的子字段，保证前端能按固定列渲染。"""
    return {name: _display(item.get(name)) for name in fields}


def _diff_sections(
    old_sections: dict[str, Any],
    new_sections: dict[str, Any],
    *,
    new_is_patch: bool,
) -> dict[str, dict[str, Any]]:
    """逐章节做**条目集合**比对（不是文本比对）。

    条目以"归一化文本"为身份，故同一要求换了编号（"1. 具有…" vs "（一）具有…"）
    不会报变更。
    """
    result: dict[str, dict[str, Any]] = {}
    for section in TENDER_SECTIONS:
        item_fields = _SECTION_ITEM_FIELDS[section.key]
        old_index = _index_items(old_sections.get(section.key))
        new_index = _index_items(new_sections.get(section.key))
        added: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        unchanged = 0

        for key, new_item in new_index.items():
            old_item = old_index.get(key)
            if old_item is None:
                added.append(_pick(new_item, item_fields))
            elif "score" in item_fields and _score_changed(old_item, new_item):
                # 文本一字未改、只有分值变了（"技术方案 25 分" → "20 分"）：
                # 这是评分标准的实质变更，但纯文本集合比对会漏掉它。
                # 报成"旧条删除 + 新条新增"，并在 impact_notes 里点名"分值调整"。
                removed.append(_pick(old_item, item_fields))
                added.append(_pick(new_item, item_fields))
            else:
                unchanged += 1

        if not new_is_patch:
            for key, old_item in old_index.items():
                if key not in new_index:
                    removed.append(_pick(old_item, item_fields))

        result[section.key] = {
            "added": added,
            "removed": removed,
            "unchanged_count": unchanged,
        }
    return result


# ---------------------------------------------------------------------------
# 汇总与影响提示
# ---------------------------------------------------------------------------

_SECTION_ACTIONS: dict[str, str] = {
    "qualifications": "先确认我方是否仍满足全部资格要求，新增门槛若不满足应尽快决定是否放弃投标",
    "disqualification": "废标条款直接决定生死，必须逐条比对标书，踩中任何一条都会被否决",
    "scoring_items": "评分标准变动会改变得分结构，报价策略与技术方案要点需重新设计",
    "technical_requirements": "需评估现有方案能否满足，必要时重新编制技术方案并调整报价",
    "required_documents": "须同步调整标书目录与装订内容，漏交材料会导致废标",
}

_FIELD_ACTIONS: dict[str, str] = {
    "evaluation_method": "评标办法改变会改变中标逻辑（如综合评分法 ↔ 最低评标价法），报价与技术方案策略都要重做",
    "duration": "服务期/工期变化需重新核算人力与成本，并同步修改实施方案与报价明细",
    "payment_terms": "付款方式变化影响现金流与资金占用成本，需重新核算报价",
    "bid_validity": "投标有效期变化须同步修改投标函中的承诺期限，不足会被否决",
    "technical_requirements": "需评估现有方案能否满足，必要时重做技术方案并调整报价",
    "required_documents": "须同步调整标书目录与装订内容，漏交材料会导致废标",
    "contact": "更新投标函、授权文件中的联系人信息，无需重做标书",
    "bid_open_place": "注意开标（或文件递交）地点变化，避免走错场地",
    "purchaser": "单位名称变化：核对投标函、报价文件抬头是否需同步",
    "agency": "代理机构变化：核对投标文件送达地址与联系渠道是否需同步",
    "project_name": "项目名称变化：确认是更正笔误还是换了项目",
    "project_id": "项目编号变化：确认是更正笔误还是换了项目",
    "price_weight": "价格分权重直接决定报价策略（权重越高越要压价），需重新测算报价",
}

_MONEY_ACTIONS: dict[str, str] = {
    "max_price": "报价上限随之变化，须重新核算成本与报价——超过最高限价的投标会被直接否决（废标）",
    "budget": "预算变化影响项目规模判断与报价策略，须重新核算",
    "deposit": "保证金须按新金额缴纳，金额或形式不符会被直接否决（废标）",
}

_TIME_ACTIONS: dict[str, str] = {
    "bid_deadline": "编制与递交时间随之变化，请立即调整排期并确认保证金到账时间",
    "open_time": "开标时间随之变化，请同步通知授权代表与相关人员",
}


def _bump(summary: dict[str, int], severity: str) -> None:
    summary["total"] += 1
    summary[severity] = summary.get(severity, 0) + 1


def _summarize(
    field_changes: list[dict[str, Any]], section_changes: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """计数口径：`total` 是**变更条目数**——字段一条算一条，
    章节里每条新增/删除各算一条。同一个章节改 3 条要求就是 3 项变更，
    因为投标员确实要改 3 个地方。"""
    summary = {"total": 0, CRITICAL: 0, MAJOR: 0, MINOR: 0}
    for change in field_changes:
        _bump(summary, change["severity"])
    for key, delta in section_changes.items():
        severity = classify_severity(key)
        for _ in range(len(delta["added"]) + len(delta["removed"])):
            _bump(summary, severity)
    return summary


def _direction(old: Any, new: Any) -> str:
    """涨/跌方向（金额与数量类字段用）。判断不了返回"变更"。"""
    old_number = _as_number(old)
    new_number = _as_number(new)
    if old_number is None or new_number is None or old_number == new_number:
        return "变更"
    return "上调" if new_number > old_number else "下调"


def _money_note(change: dict[str, Any]) -> str:
    key, label = change["key"], change["label"]
    old, new, change_type = change["old"], change["new"], change["change_type"]
    action = _MONEY_ACTIONS.get(key, "金额变化需重新核算报价")
    if change_type == "added":
        return f"{label}新增为 {_fmt_number(new)} 元：{action}"
    if change_type == "removed":
        return f"{label}被删除（原 {_fmt_number(old)} 元）：{action}，请确认是否并入其他条款"
    delta = abs((_as_number(new) or 0) - (_as_number(old) or 0))
    direction = _direction(old, new)
    return (
        f"{label}{direction} {_fmt_number(delta)} 元"
        f"（{_fmt_number(old)} → {_fmt_number(new)} 元）：{action}"
    )


def _time_note(change: dict[str, Any]) -> str:
    key, label = change["key"], change["label"]
    old, new, change_type = change["old"], change["new"], change["change_type"]
    action = _TIME_ACTIONS.get(key, "时间节点变化请同步调整排期")
    if change_type == "added":
        return f"{label}新增为 {_short(new)}：{action}"
    if change_type == "removed":
        return f"{label}被删除（原 {_short(old)}）：{action}，请确认是否改为另行通知"
    gap = _time_gap(old, new)
    direction = _time_direction(old, new)
    body = f"{label}{direction}"
    if gap:
        body += f" {gap}"
    return f"{body}（{_short(old)} → {_short(new)}）：{action}"


def _time_gap(old: Any, new: Any) -> str:
    """时间差的可读描述（"3 天" / "6 小时"）。解析不了就返回空串。"""
    try:
        old_dt = datetime.fromisoformat(str(old))
        new_dt = datetime.fromisoformat(str(new))
    except (TypeError, ValueError):
        return ""
    seconds = abs((new_dt - old_dt).total_seconds())
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时"
    return f"{int(seconds // 86400)} 天"


def _time_direction(old: Any, new: Any) -> str:
    try:
        old_dt = datetime.fromisoformat(str(old))
        new_dt = datetime.fromisoformat(str(new))
    except (TypeError, ValueError):
        return "变更"
    if new_dt == old_dt:
        return "变更"
    return "延后" if new_dt > old_dt else "提前"


def _compose(label: str, change: dict[str, Any], action: str) -> str:
    old, new, change_type = change["old"], change["new"], change["change_type"]
    if change_type == "added":
        return f"{label}新增为「{_short(new)}」：{action}"
    if change_type == "removed":
        return f"{label}被删除（原「{_short(old)}」）：{action}"
    return f"{label}变更（{_short(old)} → {_short(new)}）：{action}"


def _field_note(change: dict[str, Any]) -> str:
    key = change["key"]
    if key in _MONEY_ACTIONS:
        return _money_note(change)
    if key in _TIME_ACTIONS:
        return _time_note(change)
    action = _FIELD_ACTIONS.get(key)
    if action is None:
        action = {
            CRITICAL: "属实质性变更，必须复核并重做标书相应内容",
            MAJOR: "需评估对成本与方案的影响",
            MINOR: "信息性更新，一般无需重做标书",
        }[change["severity"]]
    return _compose(change["label"], change, action)


def _score_adjustments(delta: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """找出"文本相同、只有分值变了"的评分项（它们在 added/removed 里各占一条）。"""
    removed_by_text = {
        _norm_text(item.get("text", "")): item for item in delta["removed"]
    }
    adjustments: list[tuple[str, Any, Any]] = []
    for item in delta["added"]:
        old_item = removed_by_text.get(_norm_text(item.get("text", "")))
        if old_item is not None and _as_number(old_item.get("score")) != _as_number(
            item.get("score")
        ):
            adjustments.append(
                (str(item.get("text", "")), old_item.get("score"), item.get("score"))
            )
    return adjustments


def _section_note(key: str, delta: dict[str, Any]) -> str:
    added, removed = delta["added"], delta["removed"]
    if not added and not removed:
        return ""
    label = _SECTION_LABEL.get(key, key)
    action = _SECTION_ACTIONS.get(key, "需逐条核对并修改标书相应内容")

    # "文本没改、只有分值变了"的评分项在 added/removed 里各占一条，
    # 计数时要还原成"1 项分值调整"，否则用户会以为新增了一条又删了一条。
    adjustments = _score_adjustments(delta)
    adjusted_texts = {_norm_text(text) for text, _, _ in adjustments}
    added_only = [i for i in added if _norm_text(i.get("text", "")) not in adjusted_texts]
    removed_only = [i for i in removed if _norm_text(i.get("text", "")) not in adjusted_texts]

    parts: list[str] = []
    if added_only:
        parts.append(f"新增 {len(added_only)} 条")
    if removed_only:
        parts.append(f"删除 {len(removed_only)} 条")
    if adjustments:
        detail = "；".join(
            f"{_short(text, 20)} {_fmt_number(old)} 分 → {_fmt_number(new)} 分"
            for text, old, new in adjustments[:3]
        )
        parts.append(f"{len(adjustments)} 项分值调整（{detail}）")
    if not parts:  # 只可能在 added / removed 全被判定为分值调整时出现
        return ""

    samples = [f"新增「{_short(i.get('text', ''), 26)}」" for i in added_only[:2]]
    samples += [f"删除「{_short(i.get('text', ''), 26)}」" for i in removed_only[:2]]
    preview = f"（{'；'.join(samples)}）" if samples else ""
    return f"{label}：{'、'.join(parts)}{preview}：{action}"


def _identity_notes(field_changes: list[dict[str, Any]]) -> list[str]:
    """项目身份校验：编号/名称变了要先怀疑"这两份文件不是同一个项目"。

    这不是一条"变更提示"，而是整份比对结果是否成立的前提，故单列并放在最后。
    """
    by_key = {change["key"]: change for change in field_changes}
    notes: list[str] = []
    for key, what in (("project_id", "项目编号"), ("project_name", "项目名称")):
        change = by_key.get(key)
        if change and change["change_type"] == "modified":
            notes.append(
                f"注意：{what}不同（{_short(change['old'])} → {_short(change['new'])}）。"
                "请先确认两份文件属于同一项目；若属不同项目，本次比对结果不成立"
            )
    return notes


def _impact_notes(
    field_changes: list[dict[str, Any]],
    section_changes: dict[str, dict[str, Any]],
    *,
    new_is_patch: bool,
) -> list[str]:
    notes: list[str] = []
    for change in field_changes:
        note = _field_note(change)
        if note:
            notes.append(note)
    for key, delta in section_changes.items():
        note = _section_note(key, delta)
        if note:
            notes.append(note)

    if not notes:
        return [
            "未检出变更：两份文件在 schema 覆盖的字段与章节上一致"
            "（附件、图纸、格式文件等非文本内容不在比对范围内）"
        ]

    if new_is_patch:
        notes.insert(
            0,
            "本次按「补遗/澄清公告」模式比对：新文件未提及的内容不记为删除，"
            "故下方只列出新增与修改；如需复核被删除的内容，请按全文模式重新比对",
        )
    notes.extend(_identity_notes(field_changes))
    return notes


# ---------------------------------------------------------------------------
# 主入口 ①：纯函数 diff
# ---------------------------------------------------------------------------


def diff_parsed(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    new_is_patch: bool = False,
) -> dict[str, Any]:
    """对两份**已抽取结果**做 diff，不碰 LLM / 文件 / 数据库，可独立单测。

    参数
    ----
    old / new
        均为 `{"fields": {key: value}, "sections": {key: [item, ...]}}`
        （即 `extract_from_document()` 返回的 `parsed`）。
        传入未归一的原始值也能工作（内部会再过一次 `normalize_tender`）。
    new_is_patch
        新文件是否只是"只列变更条目"的补遗/澄清公告。置 True 时**不把
        "新文件未提及"当作删除**——补遗公告本来就不会重复全文，否则会冒出
        几十条假的"删除"。默认为 False（按重发的全文比对）。

    返回
    ----
    `{"field_changes": [...], "section_changes": {...}, "summary": {...}, "impact_notes": [...]}`

        field_changes    每项 `{key, label, old, new, change_type, severity}`，
                         `change_type ∈ {added, removed, modified}`，
                         `severity ∈ {critical, major, minor}`；按严重度倒序
        section_changes  每个章节 `{added: [条目], removed: [条目], unchanged_count: int}`；
                         条目里是该章节声明的子字段（text / score / category / is_mandatory）
        summary          `{total, critical, major, minor}`，total 为变更条目数
        impact_notes     人类可读的影响提示，方向（上调/下调、提前/延后）在此体现
    """
    old_canonical = _canonical(old, "原文件")
    new_canonical = _canonical(new, "澄清文件")

    field_changes = _diff_fields(
        old_canonical["fields"], new_canonical["fields"], new_is_patch=new_is_patch
    )
    section_changes = _diff_sections(
        old_canonical["sections"],
        new_canonical["sections"],
        new_is_patch=new_is_patch,
    )
    return {
        "field_changes": field_changes,
        "section_changes": section_changes,
        "summary": _summarize(field_changes, section_changes),
        "impact_notes": _impact_notes(
            field_changes, section_changes, new_is_patch=new_is_patch
        ),
    }


# ---------------------------------------------------------------------------
# 主入口 ②：编排（文件 → 解析 → 抽取 → diff）
# ---------------------------------------------------------------------------

# 新文件信息量不到原文件这个比例时，判定为"补遗/澄清公告"。取值 1/2 的依据：
# 重发的全文（哪怕抽取不全）通常保留多数字段与章节；只列变更条目的通知则
# 只有寥寥几条，实测差距远大于 2 倍，留出余量避免边界抖动。
_PATCH_VOLUME_RATIO = 0.5


def _content_volume(parsed: dict[str, Any]) -> int:
    """结构化信息量粗估：非空标量字段数 + 去重后的章节条目数。"""
    fields = parsed.get("fields") or {}
    sections = parsed.get("sections") or {}
    volume = sum(1 for key, value in fields.items() if _scalar_key(key, value))
    volume += sum(len(_index_items(items)) for items in sections.values())
    return volume


def _looks_like_patch(old_parsed: dict[str, Any], new_parsed: dict[str, Any]) -> bool:
    """新文件是否只是"只列变更条目"的补遗/澄清公告，而不是重发的全文。

    判据只用信息量比例，**不**用"有没有抽到项目名称"这类信号：抽取偶发漏抽
    项目名很常见，用它判断会把重发的全文误判成补遗，进而**隐藏真实的删除项**。
    信息量比例误判的代价则相反且可控（多出一些假的"删除"，用户能一眼看出
    "这是重发的全文"，且我们会把该判断明确写进 warnings）。
    """
    old_volume = _content_volume(old_parsed)
    new_volume = _content_volume(new_parsed)
    if old_volume < 4:  # 原文件本身几乎没抽出内容，比例判断不可靠
        return False
    return new_volume < old_volume * _PATCH_VOLUME_RATIO


def _run_stage(
    stage: str,
    parse_event: str,
    extract_event: str,
    filename: str,
    data: bytes,
) -> Iterator[tuple[str, Any]]:
    """解析 + 抽取**一份**文件，逐个 yield 事件；`yield from` 的返回值是抽取结果。

    为什么抽取要放进线程：`extract_from_parsed()` 是阻塞调用（批与批之间等 LLM），
    批进度是通过 `progress` 回调上报的，而回调里没法 yield。故用"后台线程 + 队列"
    把回调转成生成器事件：线程跑抽取，回调往队列塞进度，主生成器边取边 yield。
    这是不侵入抽取器内部、又能上报批进度的唯一做法——另一种做法是在这里重写一遍
    批循环，那会和抽取器逻辑漂移，不能接受。

    抽取失败/解析失败一律转成 `("error", {...})` 事件（流式接口的约定），
    返回 None 表示"这一阶段失败，调用方应停止"。
    """
    from src.services.document_parser import DocumentParseError, parse_document
    from src.services.tender_extractor import extract_from_parsed

    try:
        parsed = parse_document(filename, data)
    except DocumentParseError as exc:
        yield "error", {"content": f"{stage}解析失败：{exc}"}
        return None

    summary = parsed.summary()
    yield parse_event, {
        "stage": stage,
        "filename": filename,
        "kind": summary["kind"],
        "pages": summary["pages"],
        "total_chars": summary["total_chars"],
        "chunk_count": summary["chunk_count"],
        "truncated": summary["truncated"],
        "headings": summary["headings"],
        "warnings": summary["warnings"],
    }

    events: queue.Queue = queue.Queue()
    box: dict[str, Any] = {}

    def _work() -> None:
        try:
            box["result"] = extract_from_parsed(
                parsed, progress=lambda done, total: events.put(("tick", done, total))
            )
        except Exception as exc:  # noqa: BLE001 - 线程内异常必须带回主线程
            box["error"] = exc
        finally:
            events.put(("end", 0, 0))

    worker = threading.Thread(target=_work, name=f"change-track-{stage}", daemon=True)
    worker.start()

    total = 0
    while True:
        kind, done, total = events.get()
        if kind == "end":
            break
        yield extract_event, {
            "stage": stage,
            "filename": filename,
            "phase": "progress",
            "done": done,
            "total": total,
            # 抽取器在最后还会回调一次 (total, total)；那之后是合并与归一化，
            # 文案要跟着变，否则前端会一直显示"正在抽取第 1/1 部分"像卡住了
            "message": (
                f"{stage}：正在抽取第 {done + 1}/{total} 部分…"
                if done < total
                else f"{stage}：批次抽取完成，正在合并结果…"
            ),
        }
    worker.join()

    error = box.get("error")
    if error is not None:
        yield "error", {
            "content": f"{stage}抽取失败：{type(error).__name__}: {error}"
        }
        return None
    result = box.get("result")
    if result is None:
        yield "error", {"content": f"{stage}抽取未返回结果"}
        return None

    yield extract_event, {
        "stage": stage,
        "filename": filename,
        "phase": "done",
        "done": total,
        "total": total,
        "message": f"{stage}抽取完成",
        "result": result,
    }
    return result


def _stream_track(
    old_filename: str,
    old_bytes: bytes,
    new_filename: str,
    new_bytes: bytes,
    new_is_patch: bool | None,
) -> Iterator[tuple[str, Any]]:
    started = time.monotonic()
    warnings: list[str] = []

    if old_bytes == new_bytes:
        warnings.append(
            "两份文件内容完全相同——请确认是否误传了同一份文件"
            "（若该公告确实未修改任何内容，则本次无变更属正常结果）"
        )

    old_result = yield from _run_stage(
        "原招标文件", "parse_old", "extract_old", old_filename, old_bytes
    )
    if old_result is None:
        return
    new_result = yield from _run_stage(
        "澄清/更正文件", "parse_new", "extract_new", new_filename, new_bytes
    )
    if new_result is None:
        return

    old_parsed = old_result["parsed"]
    new_parsed = new_result["parsed"]

    if new_is_patch is None:
        new_is_patch = _looks_like_patch(old_parsed, new_parsed)
        if new_is_patch:
            warnings.append(
                "澄清文件的结构化信息量不到原文件的一半，已按「补遗/澄清公告"
                "（只列变更条目）」处理：新文件未提及的字段与章节不记为删除。"
                "若该文件其实是重发的全文（只是抽取不全），请用 diff_parsed(old, new) "
                "复核被删除的内容"
            )

    try:
        diff = diff_parsed(old_parsed, new_parsed, new_is_patch=bool(new_is_patch))
    except ChangeTrackError as exc:
        yield "error", {"content": f"变更比对失败：{exc}"}
        return

    yield "diff", diff

    warnings.extend(f"[原招标文件] {w}" for w in old_result["meta"].get("warnings") or [])
    warnings.extend(f"[澄清文件] {w}" for w in new_result["meta"].get("warnings") or [])
    if diff["summary"][CRITICAL]:
        warnings.append(
            f"检出 {diff['summary'][CRITICAL]} 项实质性（critical）变更："
            "须据此重做标书相应内容，否则有废标风险"
        )

    yield "done", {
        "diff": diff,
        "old": old_result,
        "new": new_result,
        "meta": {
            "old_filename": old_filename,
            "new_filename": new_filename,
            "old_kind": old_result["meta"].get("kind"),
            "new_kind": new_result["meta"].get("kind"),
            "summary": diff["summary"],
            "requires_rebid": diff["summary"][CRITICAL] > 0,
            "new_treated_as_patch": bool(new_is_patch),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "warnings": warnings,
        },
    }


def stream_track(
    old_filename: str,
    old_bytes: bytes,
    new_filename: str,
    new_bytes: bytes,
    *,
    new_is_patch: bool | None = None,
) -> Iterator[tuple[str, Any]]:
    """生成器版变更追踪，逐步 yield `(事件类型, 数据)`。

    事件序列（前端按此显示进度）：
        ("parse_old",   {…})   原文件解析完成（页数/字数/块数/章节标题）
        ("extract_old", {…})   原文件抽取进度，每批一条（phase=progress）；
                               最后一条 phase=done 并带 `result`
        ("parse_new",   {…})   澄清文件解析完成
        ("extract_new", {…})   澄清文件抽取进度，同上
        ("diff",        {…})   变更清单（= `diff_parsed()` 的返回值）
        ("done",        {…})   最终结果：`{diff, old, new, meta}`
        ("error",       {"content": "…"})

    为什么要流式：两份文件都要分块调 LLM 抽取，全程 1~4 分钟。没有中间事件，
    前端只能一直转圈，用户无法判断是"在跑"还是"卡死"——而这类工具一旦让人
    怀疑卡死，用户就会退回去用旧文件编标，功能等于没做。

    `new_is_patch=None`（默认）时自动判断新文件是"补遗公告"还是"重发的全文"，
    判断结果与理由写进 `done.meta`；显式传 True/False 可跳过自动判断。

    所有异常都收敛成 `("error", ...)` 事件，本生成器不向外抛异常。
    """
    try:
        yield from _stream_track(
            old_filename, old_bytes, new_filename, new_bytes, new_is_patch
        )
    except ChangeTrackError as exc:
        yield "error", {"content": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 流式接口不允许把异常抛给调用方
        logger.exception("变更追踪异常")
        yield "error", {"content": f"变更追踪失败：{type(exc).__name__}: {exc}"}


def track_changes(
    old_filename: str,
    old_bytes: bytes,
    new_filename: str,
    new_bytes: bytes,
    *,
    new_is_patch: bool | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """一步到位：原招标文件 + 澄清/更正文件 → 变更清单。

    返回 `{"diff": {...}, "old": {"parsed", "meta"}, "new": {...}, "meta": {...}}`：

        result["diff"]["summary"]        `{total, critical, major, minor}`
        result["diff"]["field_changes"]  逐字段变更（含 severity 与 old/new）
        result["diff"]["section_changes"]逐章节条目增删
        result["diff"]["impact_notes"]   人类可读的影响提示（含涨跌方向）
        result["meta"]["requires_rebid"] 是否必须重做标书（= critical > 0）
        result["meta"]["warnings"]       解析/抽取警告与判断说明

    `progress(event, data)` 可选回调，事件名与 `stream_track()` 相同，
    供非流式接入方（如 FastAPI 后台任务）上报进度。

    实现上直接消费 `stream_track()`，保证流式与非流式两条路径的判定结果
    完全一致（包括"补遗公告"的自动判断）。
    """
    result: dict[str, Any] | None = None
    for event, data in stream_track(
        old_filename, old_bytes, new_filename, new_bytes, new_is_patch=new_is_patch
    ):
        if event == "error":
            raise ChangeTrackError(str(data.get("content") or "变更追踪失败"))
        if progress is not None and event != "done":
            progress(event, data)
        if event == "done":
            result = data
    if result is None:
        raise ChangeTrackError("变更追踪既没有产出结果也没有报错，属于异常状态")
    return result


def extract_from_document(filename: str, data: bytes) -> dict[str, Any]:
    """（转发）单份文件的"解析 + 抽取"，与 `tender_extractor` 同一实现。

    本模块不重复实现抽取；保留这个转发只是为了给"只更新一侧文件"的场景
    （如先解析原文件、稍后再传澄清文件）一个统一入口。
    """
    from src.services.tender_extractor import extract_from_document as _extract

    try:
        return _extract(filename, data)
    except ChangeTrackError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一收敛成本模块的异常类型
        raise ChangeTrackError(f"抽取失败：{type(exc).__name__}: {exc}") from exc


__all__ = [
    "CRITICAL",
    "MAJOR",
    "MINOR",
    "ChangeTrackError",
    "classify_severity",
    "diff_parsed",
    "extract_from_document",
    "stream_track",
    "track_changes",
]
