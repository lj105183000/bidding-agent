"""资格审查辅助：把「招标文件的资格要求」与「企业资质档案」逐条比对，直接回答「我能不能投这个标」。

本模块是 ⑤，位于 ①②③④ 之后：前面的模块产出「招标文件写了什么」，
本模块产出「这些要求我满足几条、还差什么」。

为什么必须有第三个状态「需人工确认」
------------------------------------
二值判定（满足 / 不满足）放在资格审查里是**有害**的：

    - 把「不确定」判成**不满足** → 用户白白放弃一个本来能投的标
    - 把「不确定」判成**满足**   → 用户投出去，开标时因资格不符被废标

规则能可靠覆盖的只有少数几种表述：资质等级（"二级及以上"）、金额门槛
（"不低于 500 万元"）、认证证书（ISO9001）、年限（"近三年"）、失信记录。
剩下大量条文（"财务状况良好""具有独立承担民事责任的能力"）**无法**从档案
字段推出结论。此时唯一诚实的输出是 `uncertain`，并在 `reason` 里点名
**缺什么信息才能判**——而不是替用户猜一个结论。

故本模块的默认倾向是：**没把握就说没把握**。

`overall` 的判定必须保守
------------------------
    - 任一**硬性**要求 unmet           → `unqualified`（有实质缺口）
    - 否则只要有 uncertain             → `need_review`（不允许报 qualified）
    - 全部 met                          → `qualified`

即：不确定绝不等于通过。`gaps` 只列 unmet 的**硬性**要求——加分项没满足
不影响投标资格，混进 gaps 会让用户误以为标投不了。

判定为什么要「更宽松地」解释模糊口径
------------------------------------
招标文件里"近三年""单项合同金额"这类口径本身就有歧义。遇到歧义时本模块
**不会**直接判 unmet，而是：

    - 年份口径按**宽松侧**取（"近三年"从当前年份往回推 3 年，覆盖更早的业绩）
    - 单份合同不达标但**累计**达标 → `uncertain` 并说明是"累计口径可能满足"
    - 项目类型关键词没命中 → `uncertain`（关键词是猜的，用户可能只是项目名不同）

理由是**两类错误的代价不对称**：多报一次 uncertain 只是让用户多看一眼；
误判 unmet 却可能让用户放弃一个能投的标。

时间基准（"近三年"相对谁算）
---------------------------
本项目的历史数据是 2023 年的，但资格要求里的"近三年"必须相对**当前时间**
判断——招标文件的"近三年"永远指"从我投标这天往回数三年"，用数据集的年份
去推算会得出完全错误的窗口（这正是本项目最容易踩的坑之一）。

所有对外函数都接受 `reference_date` 参数：默认取 `date.today()`，
单测里传固定日期以保证结果确定（见 `tests/test_qualification.py`）。

与企业资质档案存储的关系
------------------------
档案（`company_profiles` 表）的读写也在本模块：资格比对与档案结构是同一份
契约的两端，分开两个文件会让"档案里到底该有什么字段"散落两处。
建表由 `_ensure_table()` 在每次读写时幂等执行，使用者拿到代码即可运行，
不需要先手工建表。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Callable, Iterable

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)

# 企业资质档案表名。本模块独占该表，无其他模块读写
TABLE = "company_profiles"

# ---------------------------------------------------------------------------
# 三个状态 / 三个总结论（对外契约，前端与测试都按字面值比对）
# ---------------------------------------------------------------------------

STATUS_MET = "met"            # 满足
STATUS_UNMET = "unmet"        # 不满足
STATUS_UNCERTAIN = "uncertain"  # 需人工确认（**不是**不满足）

STATUS_LABELS = {
    STATUS_MET: "满足",
    STATUS_UNMET: "不满足",
    STATUS_UNCERTAIN: "需人工确认",
}

OVERALL_QUALIFIED = "qualified"        # 可以投
OVERALL_UNQUALIFIED = "unqualified"    # 有硬性缺口，不能投
OVERALL_NEED_REVIEW = "need_review"    # 需人工确认后再定

OVERALL_LABELS = {
    OVERALL_QUALIFIED: "满足全部硬性资格要求，可以投标",
    OVERALL_UNQUALIFIED: "存在硬性资格缺口，按现状投标会被废标",
    OVERALL_NEED_REVIEW: "存在无法自动判定的要求，需人工确认后再决定",
}

# ---------------------------------------------------------------------------
# 要求类别（kind）——本模块自己的分类，比抽取时的 category 更细
#
# 为什么不用 LLM 给的 `category` 直接判：实测抽取结果的 category 只有
# 「资质等级/业绩要求/财务要求/人员要求/其他」五类，粒度不足以驱动规则
# （认证证书会被归到"资质等级"或"其他"，两者判法完全不同）。
# 故 `category` 只作为**兜底**，主分类按条文关键词做（_classify）。
# ---------------------------------------------------------------------------

KIND_QUALIFICATION = "qualification"    # 企业资质（专业承包/总承包 + 等级）
KIND_CERTIFICATION = "certification"    # 体系认证（ISO9001 / CMMI / 等保…）
KIND_PERFORMANCE = "performance"        # 业绩（金额 / 年限 / 项目类型 / 数量）
KIND_FINANCIAL = "financial"            # 财务（营业收入 / 审计报告 / 注册资本）
KIND_PERSONNEL = "personnel"            # 人员（项目负责人及其证书、职称）
KIND_CREDIT = "credit"                  # 信用（失信被执行人 / 重大税收违法）
KIND_BASIC = "basic"                    # 主体资格（营业执照 / 独立法人 / 依法纳税）
KIND_OTHER = "other"                    # 规则判不了 → 一律 uncertain

KIND_LABELS = {
    KIND_QUALIFICATION: "企业资质",
    KIND_CERTIFICATION: "体系认证",
    KIND_PERFORMANCE: "业绩要求",
    KIND_FINANCIAL: "财务要求",
    KIND_PERSONNEL: "人员要求",
    KIND_CREDIT: "信用记录",
    KIND_BASIC: "主体资格",
    KIND_OTHER: "其他要求",
}

# 抽取结果里的 category → kind（仅当关键词分类给不出结果时使用）
CATEGORY_TO_KIND = {
    "资质等级": KIND_QUALIFICATION,
    "认证": KIND_CERTIFICATION,
    "体系认证": KIND_CERTIFICATION,
    "业绩要求": KIND_PERFORMANCE,
    "业绩": KIND_PERFORMANCE,
    "财务要求": KIND_FINANCIAL,
    "财务": KIND_FINANCIAL,
    "人员要求": KIND_PERSONNEL,
    "人员": KIND_PERSONNEL,
    "信用": KIND_CREDIT,
    "信誉": KIND_CREDIT,
}


class QualificationError(RuntimeError):
    """资格审查失败（档案不存在 / 数据库不可用 / 招投标文档解析失败）。

    本模块对外只抛这一种异常：调用方（API 层）只需处理一类错误，
    原始异常类型与细节通过 `from exc` 保留在日志里。
    """


# ---------------------------------------------------------------------------
# 等级比较
# ---------------------------------------------------------------------------

# 等级 → (体系, 秩)。秩越大等级越高；**不同体系之间不可比**。
#
# 为什么把「体系」单独存出来：中国资质等级有两套并行命名法——
#     - 施工 / 承包类用数字：特级 > 一级 > 二级 > 三级
#     - 设计 / 监理类用天干：甲级 > 乙级 > 丙级 > 丁级
# 两套都能排秩，但"二级"与"乙级"谁高谁低没有公认答案（不同资质序列里
# 二者甚至不对应），所以**体系不同时必须返回"无法判断"**，不能硬比秩。
#
# 人员职称（高级 > 中级 > 初级）复用同一张表但自成体系：招标文件里
# "信息系统项目管理师（高级）"的"高级"与"电子与智能化二级"的"二级"
# 是两种完全不同的东西，混在一起比会得出荒谬结论。
LEVEL_RANKS: dict[str, tuple[str, int]] = {
    "特级": ("numeric", 100),
    "综合": ("numeric", 100),   # 设计类"综合资质"相当于最高级
    "一级": ("numeric", 80),
    "二级": ("numeric", 60),
    "三级": ("numeric", 40),
    "四级": ("numeric", 20),
    "甲级": ("grade", 80),
    "乙级": ("grade", 60),
    "丙级": ("grade", 40),
    "丁级": ("grade", 20),
    "正高级": ("seniority", 100),
    "副高级": ("seniority", 75),
    "高级": ("seniority", 80),
    "中级": ("seniority", 60),
    "初级": ("seniority", 40),
}

# 按长度倒序：**必须先匹配长词**，否则"副高级"会被"高级"截胡（导致秩变大）
_LEVEL_TOKENS: tuple[str, ...] = tuple(sorted(LEVEL_RANKS, key=len, reverse=True))

# 等级比较符：默认按">="（资质等级越高越能满足要求，是招投标行业惯例，
# 招标文件写"二级"时一级企业同样可以投）。显式写"仅限/只限"时改为精确匹配。
_LEVEL_EQ_MARKERS = ("仅限", "只限", "唯一", "限为", "限定")


def parse_level(text: str) -> tuple[str | None, str | None, int | None]:
    """从文本中解析等级，返回 `(等级原文, 体系, 秩)`；解析不出返回 `(None, None, None)`。

    长词优先扫描（`_LEVEL_TOKENS` 已按长度倒序），保证"副高级"不会被
    "高级"吃掉——这是一个真实的坑：秩从 75 变 80 会让"副高级"错误地
    满足"正高级"要求。
    """
    s = _norm(text)
    for token in _LEVEL_TOKENS:
        if token in s:
            system, rank = LEVEL_RANKS[token]
            return token, system, rank
    return None, None, None


def compare_level(
    required_level: str | None,
    owned_level: str | None,
    owned_text: str = "",
    op: str = ">=",
) -> bool | None:
    """比较两个等级。返回 `True`/`False`，**无法比较时返回 `None`**。

    `None` 的三种情形（都必须由调用方转成 `uncertain`，不能当成 False）：
        1. 要求里没有等级 → 由调用方处理（本函数返回 None 表示"无需比较"）
        2. 企业侧没写等级 → 无法判断
        3. 两侧等级属于**不同体系**（如要求"二级"、档案写"乙级"）→ 无法判断

    `owned_text` 是等级之外的补充文本（人员证书名、资质全名）：招标文件里的
    等级有时是**嵌在证书名里**的（"一级建造师"），此时企业档案的 `level`
    字段可能是空的，但证书名本身就带等级——故先在 `owned_text` 里找同一个
    等级词，找到即认为满足对应等级要求。
    """
    if not required_level:
        return None  # 没提等级 = 只要求"有这个资质/证书"，比较不适用
    req_system, req_rank = LEVEL_RANKS.get(required_level, (None, None))
    if req_rank is None:
        return None

    own_level, own_system, own_rank = parse_level(owned_level or "")
    if own_rank is None:
        # 档案没写等级时退一步：等级常常**嵌在资质/证书名里**（"一级建造师"），
        # 此时把名称里的等级解析出来照样能比。
        embedded_level, embedded_system, embedded_rank = parse_level(owned_text)
        if embedded_rank is not None:
            if embedded_system != req_system:
                return None  # 如要求"高级"、证书名带"一级" —— 两套体系，不可比
            if op == "==":
                return embedded_rank == req_rank
            return embedded_rank >= req_rank
        if required_level in _norm(owned_text):
            return True
        return None

    if req_system != own_system:
        # "二级" vs "乙级"：两套命名体系，谁高谁低没有依据 → 交给人工
        return None
    if op == "==":
        return own_rank == req_rank
    return own_rank >= req_rank


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------

# 全角 → 半角（数字/字母/括号/标点）。招标文件里全角字符极常见，
# 不做这一步时 "ＩＳＯ９００１" 与 "ISO9001" 会匹配不上。
#
# 源串与目标串逐字符对齐，长度必须相等——`str.maketrans` 会替我们校验，
# 写错一个字符就在 import 阶段炸掉，比"某些全角字符没被转换"这种静默错误好查。
_FULLWIDTH_SRC = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ（）【】［］，；：、。"
_FULLWIDTH_DST = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "()[][]"   # （ ） 【 】 ［ ］
    ",;:,."    # ， ； ： 、 。
)
_FULLWIDTH = str.maketrans(_FULLWIDTH_SRC, _FULLWIDTH_DST)


def _norm(text: Any) -> str:
    """归一化：去空白、全角转半角、转大写。所有字符串比较先过这里。"""
    return re.sub(r"\s+", "", str(text or "").translate(_FULLWIDTH)).upper()


# 名称比对前要剔掉的噪声词（长词在前，避免"资质证书"被"资质"截断后留下"证书"）
_NAME_NOISE: tuple[str, ...] = (
    "及以上", "或以上", "以上", "不低于", "不少于", "资质证书", "资格证书",
    "资质", "证书", "等级", "有效的", "有效期内的", "有效", "副本", "复印件",
)

# 条文开头的动词/序号，解析"资质名称""证书名"前要剥掉
_LEAD_NOISE_RE = re.compile(r"^\s*(?:[（(]?[一二三四五六七八九十\d]+[)）.、．:：]\s*)+")
_LEAD_VERBS: tuple[str, ...] = (
    "须具有", "应具有", "须具备", "应具备", "须持有", "应持有", "须提供", "应提供",
    "具有", "具备", "持有", "拥有", "提供", "取得", "获得", "通过",
)


def _strip_lead(text: str) -> str:
    """剥掉条文开头的序号与动词，剩下的更接近"要求的东西本身"。

    例：`"2. 具有电子与智能化工程专业承包二级及以上资质。"`
        → `"电子与智能化工程专业承包二级及以上资质。"`
    """
    s = _LEAD_NOISE_RE.sub("", str(text or "").strip())
    for _ in range(3):  # 少数条文会连用（"须提供…证明材料"），叠剥几轮
        for verb in _LEAD_VERBS:
            if s.startswith(verb):
                s = s[len(verb):]
                break
        else:
            break
    return s.strip()


def _core_name(text: str) -> str:
    """把资质/证书名归一成"核心名"，供包含比对。

    剔除等级词（二级/甲级/高级…）与"资质/证书"等噪声——招标文件写
    "电子与智能化工程专业承包二级及以上资质"，档案里写"电子与智能化工程
    专业承包"，两者的核心名都应是"电子与智能化工程专业承包"。
    """
    s = _norm(text)
    for word in _NAME_NOISE:
        s = s.replace(word, "")
    for token in _LEVEL_TOKENS:
        s = s.replace(token, "")
    return re.sub(r"[()\[\],.;:、/\\\-—_]", "", s)


def name_match(required: str, owned: str) -> str:
    """资质/证书名比对，返回 `"exact"` / `"match"` / `"weak"` / `"none"`。

    用"包含"而不是"相等"：档案里的名称常带后缀（"…专业承包（二级）"、"…资质"），
    严格相等会大面积误判为"没有该资质"。

    `"weak"` 用于长度悬殊的包含关系（如要求"施工总承包"、档案"建筑工程施工总承包"）：
    要求写得太笼统时，命中可能只是**碰巧共享了通用词**，不足以判定满足，
    由调用方转成 `uncertain`。
    """
    a, b = _core_name(required), _core_name(owned)
    if not a or not b:
        return "none"
    if a == b:
        return "exact"
    if a in b or b in a:
        ratio = min(len(a), len(b)) / max(len(a), len(b))
        return "match" if ratio >= 0.6 else "weak"
    return "none"


# ---------------------------------------------------------------------------
# 金额 / 年限解析
# ---------------------------------------------------------------------------

# 金额：可选比较词 + 数字 + 可选单位。
#
# ★ 只认「带单位」或「带比较词」的数字（_parse_amount 里的判断），
# 否则 "ISO9001" 里的 9001、项目编号里的数字都会被当成金额门槛。
# 这是实测踩过的坑：第一版不判单位，于是"ISO9001 认证"解析出金额 9001 元。
_AMOUNT_RE = re.compile(
    r"(不低于|不少于|不小于|不低於|大于等于|不高于|不超过|高于|低于|超过|达到|≥|<=|>=|<|>)?"
    r"\s*([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(亿元|万元|亿|万|元)?"
)

_UNIT_FACTORS = {"元": 1.0, "万": 10_000.0, "万元": 10_000.0, "亿": 100_000_000.0, "亿元": 100_000_000.0}

_UPPER_BOUND_OPS = ("不高于", "不超过", "低于", "<=", "<")

# 中文数字（"近三年"里的"三"），只用于年限解析
_CN_NUMBERS = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 时间窗口："近三年" "最近 3 年" "前两年"
_WINDOW_RE = re.compile(r"(?:近|最近|前)\s*([0-9]+|[一二两三四五六七八九十])\s*(?:个)?\s*年")

# 绝对起点："自 2023 年 1 月 1 日起" "2024 年以来"
_SINCE_RE = re.compile(r"(\d{4})\s*年(?:\s*\d{1,2}\s*月)?\s*(?:以来|以后|起|至今|后)")


def _to_int(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _CN_NUMBERS.get(token)


def parse_amount(text: str) -> tuple[float | None, str]:
    """解析金额门槛，返回 `(金额(元), 比较符)`；解析不出返回 `(None, ">=")`。

    支持的表述：`不低于 500 万元` / `不少于500万` / `≥1000万元` / `500 万元及以上`。
    中文数字金额（"伍佰万元"）不解析——解析不出时上层会转成 `uncertain`，
    这比猜一个数字安全。
    """
    s = _norm(text)
    for m in _AMOUNT_RE.finditer(s):
        op, number, unit = m.group(1), m.group(2), m.group(3)
        # 必须带单位或带比较词，否则是编号/标准号里的数字（见 _AMOUNT_RE 注释）
        if not unit and not op:
            continue
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            continue
        value *= _UNIT_FACTORS.get(unit or "元", 1.0)
        # 表达上"及以上"常跟在单位后面（"500 万元及以上"），比较符要补齐
        tail = s[m.end(): m.end() + 6]
        if op is None and ("及以上" in tail or "以上" in tail):
            op = "不低于"
        if op in _UPPER_BOUND_OPS:
            # 上限表述出现在资格要求里极少（多是"不超过 X 元的项目不计"），
            # 判不了就不判，返回 None 交给 uncertain
            return None, ">="
        return (int(value) if abs(value - round(value)) < 1e-9 else round(value, 2)), ">="
    return None, ">="


def parse_years(text: str, reference_date: date) -> tuple[int | None, int | None]:
    """解析业绩/财务的时间窗口，返回 `(年限, 起始年份)`。

    两种表述都支持：相对（"近三年"）与绝对（"自 2023 年 1 月 1 日起"）。

    ★ 年份口径按**宽松侧**取：`"近三年"` 在 2026 年 → 起始年份取 2023
    （2026 - 3），即覆盖"2023 年及以后"。行业里也有"近三年 = 前三个完整年度
    （2023~2025）"的口径，两者起点相同；若按"不含当年"再往前推一年，
    会把 2023 年的真实业绩排除掉——那正是本项目数据集所在的年份。
    歧义一律往"多收业绩"的方向落，理由见模块文档（误判 unmet 的代价更大）。
    """
    years: int | None = None
    m = _WINDOW_RE.search(text)
    if m:
        years = _to_int(m.group(1))
    since: int | None = None
    m2 = _SINCE_RE.search(text)
    if m2:
        since = int(m2.group(1))
    if since is None and years:
        since = reference_date.year - years
    return years, since


# ---------------------------------------------------------------------------
# 证书（体系认证）识别
# ---------------------------------------------------------------------------

# 认证 → 识别模式。ISO 编号与其中文名视为**同一个**证书（招标文件常两种混写）。
# 模式串在 _norm 之后的文本上匹配（已去空格、转大写）。
_CERT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ISO9001", r"ISO9001|GB/T19001|质量管理体系"),
    ("ISO14001", r"ISO14001|GB/T24001|环境管理体系"),
    ("ISO45001", r"ISO45001|OHSAS18001|GB/T45001|职业健康安全管理体系"),
    ("ISO27001", r"ISO27001|信息安全管理体系"),
    ("ISO20000", r"ISO20000|信息技术服务管理体系"),
    ("CMMI", r"CMMI"),
    ("ITSS", r"ITSS|信息技术服务标准"),
    ("等保", r"等保|信息安全等级保护|等级保护"),
    ("涉密信息系统集成", r"涉密信息系统集成|涉密资质"),
    ("高新技术企业", r"高新技术企业"),
    ("AAA信用", r"AAA|3A|信用等级证书|信用等级AAA"),
)

# 泛化的 ISO 编号（未收录在上面表里的），如 ISO22301 / ISO27701
_ISO_GENERIC_RE = re.compile(r"ISO(\d{4,5})")


def extract_certifications(text: str) -> list[str]:
    """从文本中抽出认证标识（归一化后的 id 列表）。

    等保会带上等级（"等保三级"）：等级保护二/三/四级是**不同的**测评结论，
    不带等级比会把"过了等保二级"错认成满足"等保三级"。
    """
    s = _norm(text)
    found: list[str] = []
    for cert_id, pattern in _CERT_PATTERNS:
        if re.search(pattern, s):
            if cert_id == "等保":
                for token in ("四级", "三级", "二级", "一级"):
                    if token in s:
                        cert_id = f"等保{token}"
                        break
            found.append(cert_id)
    for m in _ISO_GENERIC_RE.finditer(s):
        generic = f"ISO{m.group(1)}"
        if not any(generic == f or generic in f for f in found):
            found.append(generic)
    return found


def _cert_overlap(required: Iterable[str], owned: Iterable[str]) -> tuple[str, str] | None:
    """两组认证 id 是否有交集（允许包含关系："等保" ⊂ "等保三级"）。"""
    req_list, own_list = list(required), list(owned)
    for a in req_list:
        for b in own_list:
            if a == b or (len(a) >= 2 and (a in b or b in a)):
                return a, b
    return None


# ---------------------------------------------------------------------------
# 要求解析（条文 → 可比对的结构）
# ---------------------------------------------------------------------------

# 分类关键词。**顺序即优先级**，故这里用 tuple 而不是 dict：
#   - 信用/认证先判：条文里有"失信""ISO"时结论唯一，不该被"资质"等泛词带走
#   - 人员先于资质：人员条文常同时含"资格"二字
#   - 业绩先于资质：业绩条文常含"项目""合同金额"
_KIND_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (KIND_CREDIT, ("失信", "被执行人", "税收违法", "重大违法", "违法记录", "黑名单", "行贿", "严重违法", "经营异常")),
    (KIND_CERTIFICATION, ("ISO", "认证证书", "体系认证", "CMMI", "ITSS", "等保", "等级保护", "涉密", "高新技术企业")),
    (KIND_PERSONNEL, ("项目负责人", "项目经理", "技术负责人", "项目总监", "人员", "建造师", "工程师",
                      "会计师", "职称", "社保", "社会保险", "项目管理师", "持证",
                      "注册监理", "注册造价", "注册电气", "注册结构", "注册咨询")),
    (KIND_FINANCIAL, ("财务", "审计", "营业收入", "营业额", "净资产", "资产负债", "注册资本", "银行资信", "授信")),
    # "完成"与"项目"是业绩条文的通用词：正文常写"近三年完成 2 个类似项目"而不写
    # "完成过…业绩"。放在 performance（而非更靠前的类别）是安全的——人员/财务/认证
    # 的关键词都在它前面判过，带"项目负责人"的条文不会落到这里。
    (KIND_PERFORMANCE, ("业绩", "完成过", "完成", "类似项目", "同类项目", "合同金额",
                        "项目经验", "案例", "供货业绩", "业绩证明", "项目")),
    (KIND_QUALIFICATION, ("资质", "专业承包", "总承包", "许可证", "资格等级", "施工资质", "等级证书")),
    (KIND_BASIC, ("营业执照", "独立承担民事责任", "独立法人", "依法缴纳", "社会保障资金", "纳税", "法人资格")),
)

# 已知人员证书名（长词在前）。命中则直接用，避免通用正则截出一堆噪声。
_KNOWN_PERSONNEL_CERTS: tuple[str, ...] = tuple(
    sorted(
        (
            "信息系统项目管理师", "系统集成项目管理工程师", "系统分析师", "系统架构设计师",
            "软件设计师", "网络工程师", "信息安全工程师", "数据库系统工程师",
            "注册信息安全专业人员", "CISP", "CISAW", "PMP",
            "一级建造师", "二级建造师", "注册监理工程师", "注册造价工程师",
            "一级造价工程师", "二级造价工程师", "注册电气工程师", "注册结构工程师",
            "注册咨询工程师", "咨询工程师", "注册会计师", "高级工程师", "中级工程师",
        ),
        key=len,
        reverse=True,
    )
)

# 通用人员证书抽取：动词后面跟一串中文/字母，直到"资格/证书/职称"或句读
_PERSONNEL_CERT_RE = re.compile(
    r"(?:具备|具有|持有|拥有|取得|须有|应有)\s*"
    r"([\u4e00-\u9fa5A-Za-z0-9（）()·]{2,30}?)"
    r"\s*(?:资格|资格证书|证书|职称|执业资格)?\s*(?:[，。；;,、]|$)"
)

# 人员角色
_ROLE_RE = re.compile(r"(项目负责人|项目经理|技术负责人|项目总监|总工程师|项目组成员|项目团队|拟派人员)")

# 业绩项目类型："500 万元的政务云项目" → 政务云
_PROJECT_KIND_RE = re.compile(
    r"(?:亿元|万元|万|元)\s*(?:及以上|以上)?\s*的?\s*([\u4e00-\u9fa5A-Za-z0-9]{2,12}?)\s*项目"
)

# 业绩/人员数量："不少于 2 个" "至少完成 2 项" "2 名以上"。
# 单位词与数字之间允许几个字的填充（"至少**完成** 2 个"），否则最自然的写法反而漏抽。
_COUNT_RE = re.compile(r"(?:不少于|不低于|至少|多于|≥)[\u4e00-\u9fa5]{0,4}?([0-9]+)\s*(?:个|项|名|人|份)")

# 项目类型关键词的停用词：这些不是"项目类型"，只是泛称
_KIND_STOPWORDS = ("类似", "同类", "相关", "以上", "的", "类似或", "类似及以上")

# 信用条目关键词 → 档案字段
_CREDIT_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dishonest", ("失信被执行人", "失信", "被执行人")),
    ("tax_violation", ("重大税收违法", "税收违法", "偷税", "逃税")),
    ("serious_violation", ("重大违法记录", "严重违法", "重大违法")),
    ("blacklist", ("黑名单", "经营异常", "政府采购严重违法")),
)

# 否定词：条文里"未被列入…"要求企业侧对应字段为 False
_NEGATION_WORDS = ("未被", "未列入", "没有", "无", "不存在", "不属于", "未曾", "不在")

# 财务"良好"类模糊表述 → 档案 financial.healthy。
#
# 这类词判不出结论，故只在**该条文没有其他可判指标时**才用它（见 _match_financial）：
# 招标文件的财务条款几乎都写着"财务状况良好，提供近一年经审计的财务报告"，
# 若逢"良好"就制造一个 uncertain，用户会在每条财务要求上看到"需人工确认"，
# 真正需要关注的不确定项反而被淹没。可判指标（审计报告、营业收入）优先。
_HEALTHY_WORDS = ("财务状况良好", "无亏损", "未亏损", "资信良好", "财务状况正常")

# 主体资格关键词 → 档案字段（任一为真即认为"已登记主体资格"）
_BASIC_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("business_license", ("营业执照", "统一社会信用代码")),
    ("legal_entity", ("独立法人", "独立承担民事责任", "法人资格")),
    ("tax_compliance", ("依法缴纳", "税收", "纳税", "社会保障资金", "社保")),
)


def _classify(text: str, category: str = "") -> str:
    """条文 → kind。关键词优先，category 兜底。"""
    s = _norm(text)
    for kind, keywords in _KIND_KEYWORDS:
        if any(_norm(kw) in s for kw in keywords):
            return kind
    return CATEGORY_TO_KIND.get(_norm(category), KIND_OTHER)


def _extract_constraints(text: str, kind: str, reference_date: date) -> dict[str, Any]:
    """按 kind 抽取结构化约束（本模块的核心"归一化"步骤）。"""
    s = _norm(text)
    level, level_system, level_rank = parse_level(s)
    op = "==" if any(m in s for m in _LEVEL_EQ_MARKERS) else ">="
    amount, amount_op = parse_amount(s)
    years, since_year = parse_years(s, reference_date)
    count_m = _COUNT_RE.search(s)
    count = int(count_m.group(1)) if count_m else None

    if kind == KIND_QUALIFICATION:
        name = _core_name(_strip_lead(text))
        return {
            "name": _strip_lead(text),   # 原文名（含等级），便于人读
            "core_name": name,           # 归一化名，供比对
            "level": level,
            "level_rank": level_rank,
            "level_system": level_system,
            "level_op": op,
        }

    if kind == KIND_CERTIFICATION:
        return {"certs": extract_certifications(s), "level": level}

    if kind == KIND_PERFORMANCE:
        kw_m = _PROJECT_KIND_RE.search(s)
        keyword = ""
        if kw_m:
            candidate = kw_m.group(1)
            if candidate not in _KIND_STOPWORDS:
                keyword = candidate
        return {
            "amount": amount,
            "amount_op": amount_op,
            "years": years,
            "since_year": since_year,
            "count": count,
            "keyword": keyword,
        }

    if kind == KIND_FINANCIAL:
        is_capital = "注册资本" in s
        audited: bool | None = None
        if "审计" in s or "审计报告" in s:
            audited = True
        healthy: bool | None = True if any(_norm(w) in s for w in _HEALTHY_WORDS) else None
        return {
            "capital": amount if is_capital else None,
            "revenue": None if is_capital else amount,
            "amount_op": amount_op,
            "years": years,
            "since_year": since_year,
            "audited": audited,
            "healthy": healthy,
        }

    if kind == KIND_PERSONNEL:
        cert = ""
        for known in _KNOWN_PERSONNEL_CERTS:
            if _norm(known) in s:
                cert = known
                break
        if not cert:
            m = _PERSONNEL_CERT_RE.search(text)
            if m:
                raw = m.group(1)
                raw = re.sub(r"[（(].*?[)）]", "", raw)  # 去掉"（高级）"这类等级括号
                cert = _strip_lead(raw)
        role_m = _ROLE_RE.search(s)
        return {
            "role": role_m.group(1) if role_m else "",
            "cert": cert,
            "level": level,
            "level_rank": level_rank,
            "level_system": level_system,
            "count": count,
        }

    if kind == KIND_CREDIT:
        items = [key for key, kws in _CREDIT_ITEMS if any(_norm(kw) in s for kw in kws)]
        expected = not any(_norm(w) in s for w in _NEGATION_WORDS)
        return {"items": items, "expected": expected}

    if kind == KIND_BASIC:
        items = [key for key, kws in _BASIC_FIELDS if any(_norm(kw) in s for kw in kws)]
        return {"items": items or ["business_license"]}

    return {}


def _analyse(text: str, category: str, reference_date: date) -> tuple[str, dict[str, Any]]:
    """条文 → `(kind, constraints)`。解析与比对共用同一入口，保证两侧口径一致。"""
    kind = _classify(text, category)
    return kind, _extract_constraints(text, kind, reference_date)


def parse_qualification_requirements(
    parsed: dict,
    reference_date: date | str | datetime | None = None,
) -> list[dict]:
    """从抽取结果中取出资格要求，归一成可比对的结构。

    参数 `parsed` 接受三种形态（都从 P0 的抽取链路来）：
        1. `{"fields": {...}, "sections": {"qualifications": [...]}}`（`normalize_tender` 的输出）
        2. `{"parsed": {...}, "meta": {...}}`（`extract_from_document` 的完整输出）
        3. 直接给 qualifications 数组

    返回列表，每项：
        {
          "index": int,                 # 在资格要求里的序号
          "raw_text": str,              # 原文（人工核对时唯一可信的东西）
          "category": str,              # 抽取时给的类别（原样保留）
          "is_mandatory": bool,         # 硬性门槛 / 加分项
          "kind": str,                  # 本模块的细分类（见 KIND_*）
          "kind_label": str,
          "constraints": {...},         # 结构化约束（资质名/等级/金额/年限/证书/…）
        }

    为什么保留 `raw_text`：规则判定只是辅助，最终拍板的必须是人；
    展示时"原文 + 我的判定 + 判定理由"三者并排，用户才能自己复核。
    """
    items = _as_requirement_items(parsed)
    ref = _as_date(reference_date)

    out: list[dict] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("raw_text") or "").strip()
        if not text:
            continue
        category = str(item.get("category") or "").strip()
        # 抽取结果里 is_mandatory 已被 normalize_tender 转成布尔；手工构造的
        # dict 可能是字符串，这里再兜一次（畸形值按"硬性"处理——宁可多提醒）
        raw_flag = item.get("is_mandatory", True)
        if isinstance(raw_flag, str):
            is_mandatory = _norm(raw_flag) not in ("FALSE", "0", "否", "NO", "", "N")
        else:
            is_mandatory = bool(raw_flag)

        kind, constraints = _analyse(text, category, ref)
        out.append(
            {
                "index": i,
                "raw_text": text,
                "category": category,
                "is_mandatory": is_mandatory,
                "kind": kind,
                "kind_label": KIND_LABELS.get(kind, kind),
                "constraints": constraints,
            }
        )
    logger.debug("解析资格要求 %d 条", len(out))
    return out


def _as_requirement_items(parsed: Any) -> list:
    """从各种形态的输入里取出"资格要求条目"列表。"""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, ValueError):
            raise QualificationError("资格要求输入不是合法 JSON") from None
    if not isinstance(parsed, dict):
        raise QualificationError(f"无法识别的资格要求输入类型：{type(parsed).__name__}")
    inner = parsed.get("parsed")
    if isinstance(inner, (dict, list)):
        return _as_requirement_items(inner)
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        return sections.get("qualifications") or []
    if "qualifications" in parsed:
        return parsed.get("qualifications") or []
    return []


def _as_date(value: date | str | datetime | None) -> date:
    """归一化参考日期：`None` → 今天（见模块文档"时间基准"）。"""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise QualificationError(f"无法解析参考日期：{value!r}（建议用 YYYY-MM-DD）")


# ---------------------------------------------------------------------------
# 单条比对（纯函数，不碰数据库 / LLM，可独立单测）
# ---------------------------------------------------------------------------

def _result(status: str, reason: str, evidence: str = "") -> dict:
    return {"status": status, "reason": reason, "evidence": evidence}


def _met(reason: str, evidence: str = "") -> dict:
    return _result(STATUS_MET, reason, evidence)


def _unmet(reason: str, evidence: str = "") -> dict:
    return _result(STATUS_UNMET, reason, evidence)


def _uncertain(reason: str, evidence: str = "") -> dict:
    return _result(STATUS_UNCERTAIN, reason, evidence)


def _as_profile(profile: Any) -> dict:
    """档案归一化：`None` → `{}`；JSON 字符串 → dict（TEXT 列兼容）。"""
    if profile is None:
        return {}
    if isinstance(profile, str):
        text = profile.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            raise QualificationError("企业资质档案不是合法 JSON") from None
        return data if isinstance(data, dict) else {}
    if isinstance(profile, dict):
        return profile
    raise QualificationError(f"企业资质档案应为 dict，实际是 {type(profile).__name__}")


def _profile_list(profile: dict, key: str) -> list[dict]:
    """取档案里的列表字段，容忍 None / 单个 dict / 列表里混了字符串。"""
    value = profile.get(key)
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, str):
        value = [v.strip() for v in re.split(r"[;；,，]", value) if v.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip()})
    return out


def _demo_qualification(q: dict) -> str:
    parts = [str(q.get("name") or q.get("qualification") or "?")]
    if q.get("level"):
        parts.append(f"（{q['level']}）")
    if q.get("cert_no"):
        parts.append(f" 证书号 {q['cert_no']}")
    if q.get("valid_until"):
        parts.append(f" 有效期至 {q['valid_until']}")
    return "".join(parts)


def _window_start(c: dict, ref: date) -> int | None:
    """比对本条要求时的时间窗口起点（年份）。

    ★ 相对量（"近三年"）在**比对时**按传入的 `reference_date` 现算，而不是直接用
    解析时算好的 `since_year`：一个 req 字典可能被拿去用不同基准日复核
    （例如招标文件是几个月前上传的），若沿用解析时的窗口，"近三年"就会停留在
    旧基准上，同一份档案在不同日期得出互相矛盾的结论。
    `years` 是纯相对量，`since_year` 只在条文写死了年份（"自 2023 年以来"）时才是权威值。
    """
    years = c.get("years")
    if years:
        return ref.year - int(years)
    return c.get("since_year")


def _within_window(entry: dict, since_year: int | None) -> bool | None:
    """业绩是否落在时间窗口内。返回 `None` 表示"没写年份，判不了"。"""
    if since_year is None:
        return True
    raw = entry.get("year") or entry.get("date") or entry.get("completed_at")
    if raw in (None, ""):
        return None
    m = re.search(r"(\d{4})", str(raw))
    if not m:
        return None
    return int(m.group(1)) >= since_year


def _entry_amount(entry: dict) -> float | None:
    raw = entry.get("amount")
    if raw in (None, ""):
        return None
    value, _ = parse_amount(str(raw))
    if value is not None:
        return value
    try:
        return float(str(raw).replace(",", ""))
    except ValueError:
        return None


def _match_qualification_kind(text: str, c: dict, profile: dict, ref: date) -> dict:
    """企业资质：名称比对 + 等级比较。"""
    items = _profile_list(profile, "qualifications")
    required_name = str(c.get("name") or text)
    if not items:
        return _uncertain(
            f"企业资质档案中未登记任何资质，无法判断是否具备「{c.get('core_name') or required_name}」",
            "请在档案的 qualifications 中补录资质名称、等级、证书编号与有效期",
        )

    matched: dict | None = None
    weak_hits: list[dict] = []
    for q in items:
        quality = name_match(required_name, str(q.get("name") or q.get("qualification") or ""))
        if quality in ("exact", "match"):
            matched = q
            break
        if quality == "weak":
            weak_hits.append(q)

    recorded = "；".join(_demo_qualification(q) for q in items)
    if matched is None:
        if weak_hits:
            return _uncertain(
                f"档案中的「{weak_hits[0].get('name')}」与要求「{c.get('core_name')}」只是部分重合，"
                "无法确认是同一项资质（要求写得太笼统）",
                f"已登记资质：{recorded}",
            )
        return _unmet(
            f"企业资质档案中没有「{c.get('core_name') or required_name}」这项资质",
            f"已登记资质：{recorded}。若企业实际持有，请在档案中补录后重新评估",
        )

    evidence = _demo_qualification(matched)
    level_op = str(c.get("level_op") or ">=")

    # 条文没写等级 = 只要求"有这个资质"，此时不做等级比较
    # （compare_level 在"要求无等级"时返回 None，直接当成"判不了"会把满足误报成待确认）
    required_level = c.get("level")
    if required_level:
        ok = compare_level(
            required_level,
            matched.get("level"),
            owned_text=str(matched.get("name") or ""),
            op=level_op,
        )
        if ok is None:
            return _uncertain(
                f"已找到资质「{matched.get('name')}」，但无法确认其等级是否达到要求"
                f"（要求等级：{required_level}，档案等级：{matched.get('level') or '未填写'}）",
                f"{evidence}。需人工核对资质证书原件上的等级",
            )
        if not ok:
            rel = "等于" if level_op == "==" else "不低于"
            return _unmet(
                f"资质等级不足：要求{rel}「{required_level}」，档案为「{matched.get('level')}」",
                evidence,
            )

    # 有效期：只在档案填了 valid_until 时校验——过期的资质证书在评标时按无效处理
    valid_until = matched.get("valid_until")
    if valid_until:
        m = re.search(r"(\d{4})[-/年.]?(\d{1,2})?[-/月.]?(\d{1,2})?", str(valid_until))
        if m:
            try:
                end = date(int(m.group(1)), int(m.group(2) or 12), int(m.group(3) or 1))
            except ValueError:
                end = None
            if end and end < ref:
                return _unmet(
                    f"资质证书已于 {end.isoformat()} 过期（参考日期 {ref.isoformat()}）",
                    evidence,
                )
    return _met(
        f"已具备「{matched.get('name')}」"
        + (f"，等级「{matched.get('level')}」满足要求" if required_level and matched.get("level") else ""),
        evidence,
    )


def _match_certification(text: str, c: dict, profile: dict, ref: date) -> dict:
    """体系认证：证书集合比对（含有效期）。"""
    required_ids = list(c.get("certs") or [])
    if not required_ids:
        return _uncertain(
            "未能从条文中识别出具体认证名称，无法与档案比对",
            "请人工核对该条文要求的认证种类，并在档案 certifications 中补录",
        )

    raw_certs = profile.get("certifications") or profile.get("certs") or []
    if isinstance(raw_certs, str):
        raw_certs = [v.strip() for v in re.split(r"[;；,，]", raw_certs) if v.strip()]
    if not raw_certs:
        return _uncertain(
            f"企业资质档案中未登记任何体系认证，无法判断是否具备 {'、'.join(required_ids)}",
            "请在档案的 certifications 中补录认证名称（如 ISO9001、ISO27001）",
        )

    # 档案里的认证可以是字符串（"ISO9001"）或字典（带有效期）：
    # `owned_ids` 统一成 `(该条目识别出的认证 id 列表, 条目原文, 条目字典或 None)`
    owned_ids: list[tuple[list[str], str, dict | None]] = []
    for item in raw_certs:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("cert") or "")
            owned_ids.append((extract_certifications(name), name, item))
        else:
            name = str(item)
            owned_ids.append((extract_certifications(name), name, None))

    hit = _cert_overlap(required_ids, [i for ids, _, _ in owned_ids for i in ids])
    if hit is None:
        return _unmet(
            f"档案中没有要求的认证（{'、'.join(required_ids)}）",
            "已登记认证：" + "、".join(name for _, name, _ in owned_ids),
        )

    _req_id, owned_id = hit
    entry_meta = next((meta for ids, _, meta in owned_ids if owned_id in ids), None)
    if entry_meta and entry_meta.get("valid_until"):
        m = re.search(r"(\d{4})[-/年.]?(\d{1,2})?[-/月.]?(\d{1,2})?", str(entry_meta["valid_until"]))
        if m:
            try:
                end = date(int(m.group(1)), int(m.group(2) or 12), int(m.group(3) or 1))
            except ValueError:
                end = None
            if end and end < ref:
                return _unmet(
                    f"认证「{owned_id}」已于 {end.isoformat()} 过期（参考日期 {ref.isoformat()}）",
                    str(entry_meta.get("name") or owned_id),
                )
    return _met(f"已具备认证「{owned_id}」", "档案 certifications 中已登记")


def _match_performance(text: str, c: dict, profile: dict, ref: date) -> dict:
    """业绩：金额门槛 / 数量门槛 / 时间窗口 / 项目类型（逐层放宽，拿不准就 uncertain）。"""
    amount_threshold = c.get("amount")
    count_threshold = c.get("count")
    since_year = _window_start(c, ref)
    keyword = c.get("keyword") or ""

    if amount_threshold is None and count_threshold is None:
        return _uncertain(
            "未能从条文中解析出明确的业绩门槛（金额或数量），无法自动比对",
            "请人工核对该业绩要求的具体口径",
        )

    entries = _profile_list(profile, "performance")
    if not entries:
        return _uncertain(
            "企业业绩档案为空，无法判断是否满足业绩门槛",
            "请在档案的 performance 中补录项目名称、合同金额与完成年份",
        )

    # ① 项目类型关键词：关键词是"猜"出来的，一条都没命中时不能直接判 unmet
    if keyword:
        hits = [e for e in entries if keyword in str(e.get("project") or e.get("name") or "")]
        if not hits:
            names = "、".join(str(e.get("project") or e.get("name") or "?") for e in entries[:5])
            return _uncertain(
                f"没有业绩的项目名称包含「{keyword}」，无法确认是否有同类项目经验"
                "（也可能只是项目命名不同）",
                f"已登记业绩：{names}。需人工核对项目内容是否属于同类",
            )
        entries = hits

    # ② 时间窗口
    in_window: list[dict] = []
    unknown_year: list[dict] = []
    for e in entries:
        flag = _within_window(e, since_year)
        if flag is True:
            in_window.append(e)
        elif flag is None:
            unknown_year.append(e)

    window_desc = f"{since_year} 年及以后" if since_year else "不限"
    if not in_window:
        if unknown_year:
            return _uncertain(
                f"档案中的业绩未登记完成年份，无法判断是否落在「{window_desc}」内",
                "请在业绩记录中补录年份（如 year: 2024）",
            )
        years_seen = "、".join(str(e.get("year") or "?") for e in entries)
        return _unmet(
            f"没有落在「{window_desc}」时间窗口内的业绩（档案中的年份：{years_seen}）",
            "需提供窗口期内的业绩证明",
        )

    # ③ 金额门槛
    if amount_threshold is not None:
        qualified = [e for e in in_window if (_entry_amount(e) or 0) >= amount_threshold]
        if qualified:
            best = max(qualified, key=lambda e: _entry_amount(e) or 0)
            return _met(
                f"业绩金额满足门槛（要求不低于 {amount_threshold:,.0f} 元）",
                f"{best.get('project')}：{(_entry_amount(best) or 0):,.0f} 元，{best.get('year')} 年",
            )
        total = sum(_entry_amount(e) or 0 for e in in_window)
        best_amount = max((_entry_amount(e) or 0) for e in in_window)
        if total >= amount_threshold:
            return _uncertain(
                f"单项业绩金额均低于门槛（最大 {best_amount:,.0f} 元 < {amount_threshold:,.0f} 元），"
                f"但窗口内业绩累计 {total:,.0f} 元已达到门槛",
                "「单项合同金额」通常按单份合同计，若招标文件允许累计业绩则可满足——需人工确认口径",
            )
        if unknown_year:
            return _uncertain(
                "部分业绩未登记完成年份，可能落在时间窗口内并满足金额门槛",
                "请补录这些业绩的年份",
            )
        return _unmet(
            f"业绩金额不达标：要求不低于 {amount_threshold:,.0f} 元，"
            f"窗口内最大单项 {best_amount:,.0f} 元（合计 {total:,.0f} 元）",
            f"时间窗口：{window_desc}",
        )

    # ④ 数量门槛（金额门槛缺省时）
    if count_threshold is not None and len(in_window) < count_threshold:
        return _unmet(
            f"业绩数量不足：要求不少于 {count_threshold} 个，窗口内仅有 {len(in_window)} 个",
            f"时间窗口：{window_desc}",
        )
    return _met(
        f"窗口内有 {len(in_window)} 个业绩记录，满足要求",
        f"时间窗口：{window_desc}" + (f"；项目类型关键词：{keyword}" if keyword else ""),
    )


def _match_financial(text: str, c: dict, profile: dict, ref: date) -> dict:
    """财务：注册资本 / 营业收入 / 审计报告 / 财务状况良好。"""
    fin = profile.get("financial") if isinstance(profile.get("financial"), dict) else {}
    checks: list[dict] = []

    for label, threshold, field_names, source in (
        ("注册资本", c.get("capital"), ("registered_capital", "capital"), (profile, fin)),
        ("营业收入", c.get("revenue"), ("revenue", "annual_revenue", "income"), (fin, profile)),
    ):
        if threshold is None:
            continue
        main, fallback = source
        value = None
        for src in (main, fallback):
            for name in field_names:
                if src.get(name) not in (None, ""):
                    value = _entry_amount({"amount": src.get(name)})
                    break
            if value is not None:
                break
        if value is None:
            checks.append(
                _uncertain(
                    f"档案未登记{label}，无法判断是否达到 {threshold:,.0f} 元",
                    f"请在档案中补录 {field_names[0]}（单位：元）",
                )
            )
        elif value >= threshold:
            checks.append(_met(f"{label} {value:,.0f} 元 ≥ 要求 {threshold:,.0f} 元"))
        else:
            checks.append(_unmet(f"{label}不足：档案 {value:,.0f} 元 < 要求 {threshold:,.0f} 元"))

    if c.get("audited"):
        audited = fin.get("audited", profile.get("audited"))
        if audited is None:
            checks.append(
                _uncertain(
                    "无法判断是否具备经审计的财务报告",
                    "请在档案的 financial.audited 中填 true/false",
                )
            )
        elif audited:
            checks.append(_met("档案标记为已具备经审计的财务报告"))
        else:
            checks.append(_unmet("档案标记为没有经审计的财务报告"))

    healthy = c.get("healthy")
    if healthy and not checks:
        value = fin.get("healthy", profile.get("healthy"))
        if value is None:
            checks.append(
                _uncertain(
                    "条文只要求「财务状况良好」，无法从档案字段自动判定",
                    "请在档案的 financial.healthy 中填 true/false，或人工核对审计报告结论",
                )
            )
        elif value:
            checks.append(_met("档案标记为财务状况良好"))
        else:
            checks.append(_unmet("档案标记为财务状况不满足要求"))

    if not checks:
        return _uncertain(
            "未能从条文中解析出可比的财务指标，无法自动判定",
            "请人工核对该财务要求",
        )
    unmet = [r for r in checks if r["status"] == STATUS_UNMET]
    if unmet:
        return _unmet(unmet[0]["reason"], unmet[0].get("evidence", ""))
    uncertain = [r for r in checks if r["status"] == STATUS_UNCERTAIN]
    if uncertain:
        return _uncertain(uncertain[0]["reason"], uncertain[0].get("evidence", ""))
    return _met("；".join(r["reason"] for r in checks))


def _match_personnel(text: str, c: dict, profile: dict, ref: date) -> dict:
    """人员：证书名比对 + 等级（职称）比较 + 数量。"""
    people = _profile_list(profile, "personnel")
    cert = str(c.get("cert") or "")
    role = str(c.get("role") or "")

    if not people:
        return _uncertain(
            f"企业人员档案为空，无法判断是否具备「{cert or role or '该人员要求'}」",
            "请在档案的 personnel 中补录人员姓名、证书名称与等级",
        )

    def _person_cert(p: dict) -> str:
        return str(p.get("cert") or p.get("certificate") or p.get("qualification") or p.get("name") or "")

    if cert:
        candidates = [p for p in people if name_match(cert, _person_cert(p)) != "none"]
        if not candidates:
            return _unmet(
                f"人员档案中没有持有「{cert}」的人员",
                "已登记人员：" + "；".join(f"{p.get('name') or '?'}({_person_cert(p)})" for p in people),
            )
    else:
        candidates = people

    if c.get("count") and len(candidates) < c["count"]:
        return _unmet(
            f"持证人员数量不足：要求不少于 {c['count']} 名，档案中有 {len(candidates)} 名",
            "；".join(f"{p.get('name') or '?'}({_person_cert(p)})" for p in candidates),
        )

    evidence = "；".join(f"{p.get('name') or '?'}：{_person_cert(p)}" for p in candidates)
    level = c.get("level")
    if not level:
        return _met(f"已具备要求的人员{'（' + cert + '）' if cert else ''}", evidence)

    # 逐人比等级：任意一人达到即可（招标文件通常只要求"拟派项目负责人"具备）
    undecidable: list[str] = []
    for p in candidates:
        ok = compare_level(
            level,
            str(p.get("level") or ""),
            owned_text=_person_cert(p),
            op=">=",
        )
        if ok is True:
            return _met(
                f"「{p.get('name') or '?'}」持 {_person_cert(p)}（{level}）满足要求",
                evidence,
            )
        if ok is None:
            undecidable.append(str(p.get("name") or "?"))
    if undecidable:
        return _uncertain(
            f"已找到持「{cert}」的人员，但其等级是否达到「{level}」无法确认"
            f"（{'、'.join(undecidable)} 未登记等级）",
            f"{evidence}。需人工核对证书等级",
        )
    return _unmet(
        f"人员等级不达标：要求「{level}」，档案中人员等级均低于该要求",
        evidence,
    )


def _match_credit(text: str, c: dict, profile: dict, ref: date) -> dict:
    """信用记录：失信被执行人 / 重大税收违法 / 严重违法 / 黑名单。"""
    credit = profile.get("credit") if isinstance(profile.get("credit"), dict) else {}
    items = list(c.get("items") or [])
    expected = bool(c.get("expected", False))
    if not items:
        return _uncertain("未能识别出具体的信用核查项，无法自动判定", "请人工核对该信用要求")

    unmet: list[dict] = []
    uncertain: list[dict] = []
    for key in items:
        value = credit.get(key)
        if value is None:
            uncertain.append(
                _uncertain(
                    f"档案未登记信用项「{key}」，无法确认是否符合要求",
                    f"请在档案的 credit.{key} 中填 true/false（true = 存在该不良记录）",
                )
            )
            continue
        if bool(value) != expected:
            unmet.append(
                _unmet(
                    f"存在第「{key}」项不良记录，不满足该信用要求",
                    "投标前须先处理该记录，否则按招标文件通常会被废标",
                )
            )
    if unmet:
        return unmet[0]
    if uncertain:
        return uncertain[0]
    return _met("档案标记为无相关不良信用记录", "、".join(items))


def _match_basic(text: str, c: dict, profile: dict, ref: date) -> dict:
    """主体资格（营业执照 / 独立法人 / 依法纳税）：档案里登记了就算满足。"""
    items = list(c.get("items") or [])
    for key in items:
        if profile.get(key) not in (None, "", False):
            return _met(
                f"档案已登记「{key}」，具备主体资格",
                f"{key} = {profile.get(key)}",
            )
    if profile.get("company_name"):
        return _uncertain(
            "档案只有企业名称，无法确认营业执照与独立法人资格",
            "请在档案中补录 business_license（统一社会信用代码），或人工核对营业执照副本",
        )
    return _uncertain(
        "企业档案未登记主体资格信息（营业执照 / 独立法人）",
        "请在档案中补录 company_name 与 business_license",
    )


def _match_other(text: str, c: dict, profile: dict, ref: date) -> dict:
    """规则覆盖不了的条文：**一律 uncertain**，并说明需要人工核对什么。"""
    return _uncertain(
        "该条文无法用规则自动判定（属于需要人工解释的要求）",
        f"需人工核对原文：「{str(text)[:80]}」",
    )


_MATCHERS: dict[str, Callable[[str, dict, dict, date], dict]] = {
    KIND_QUALIFICATION: _match_qualification_kind,
    KIND_CERTIFICATION: _match_certification,
    KIND_PERFORMANCE: _match_performance,
    KIND_FINANCIAL: _match_financial,
    KIND_PERSONNEL: _match_personnel,
    KIND_CREDIT: _match_credit,
    KIND_BASIC: _match_basic,
    KIND_OTHER: _match_other,
}


def match_qualification(
    req: dict,
    profile: dict,
    reference_date: date | str | datetime | None = None,
) -> dict:
    """判定单条资格要求是否满足。**纯函数**：不碰数据库、不调 LLM，可独立单测。

    参数：
        req    —— 一条要求。可以是 `parse_qualification_requirements` 的产物，
                  也可以只给 `{"raw_text": "具有…二级及以上资质"}`（内部会现解析）。
        profile —— 企业资质档案（结构见模块文档 / `save_profile`）。
                  `None` 与 `{}` 等价——**空档案会得到大量 uncertain 而不是 unmet**，
                  因为"没填"与"没有"是两件事。
        reference_date —— "近三年"这类时间窗口的基准日，默认今天。
                  单测必须传固定日期，否则结果随运行日期漂移。

    返回 `{"status": "met"|"unmet"|"uncertain", "reason": str, "evidence": str}`。
    """
    if not isinstance(req, dict):
        raise QualificationError(f"资格要求应为 dict，实际是 {type(req).__name__}")
    ref = _as_date(reference_date)
    data = _as_profile(profile)

    text = str(req.get("raw_text") or req.get("text") or "").strip()
    if not text:
        raise QualificationError("资格要求缺少 raw_text / text，无法判定")

    kind = str(req.get("kind") or "").strip()
    constraints = req.get("constraints")
    if kind not in _MATCHERS or not isinstance(constraints, dict):
        kind, parsed_constraints = _analyse(text, str(req.get("category") or ""), ref)
        if not isinstance(constraints, dict) or not constraints:
            constraints = parsed_constraints

    matcher = _MATCHERS.get(kind, _match_other)
    try:
        return matcher(text, constraints, data, ref)
    except QualificationError:
        raise
    except Exception as exc:  # 单条判定失败不该拖垮整份报告
        logger.exception("资格要求判定失败：%s", text[:60])
        return _uncertain(
            f"判定过程出错（{type(exc).__name__}），无法给出结论",
            "需人工核对该条文",
        )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

# 缺口整改建议：按 kind 给"下一步该做什么"——只说"不满足"对用户没有价值
_GAP_SUGGESTIONS: dict[str, str] = {
    KIND_QUALIFICATION: "核实企业是否持有该资质；如持有请在资质档案中补录等级与证书编号，"
                        "并确认证书在投标截止日仍在有效期内",
    KIND_CERTIFICATION: "确认是否已取得该认证；未取得的需评估取证周期（体系认证通常 1~3 个月），"
                        "赶不上投标截止日则不建议投",
    KIND_PERFORMANCE: "核对窗口期内的业绩合同与验收报告；金额或年限确实不满足时，"
                      "可考虑联合体投标或放弃该项目",
    KIND_FINANCIAL: "调取近一年审计报告，确认营业收入/审计结论是否达到要求",
    KIND_PERSONNEL: "确认拟派人员的证书与职称等级，必要时更换人员配置",
    KIND_CREDIT: "先处理失信/违法记录（修复后通常需要一定时间才在公示系统更新），"
                 "未处理前投标会被废标",
    KIND_BASIC: "补齐营业执照等主体资格材料",
    KIND_OTHER: "人工核对原文并评估是否满足",
}


def check_qualifications(
    tender_parsed: dict,
    profile: dict,
    reference_date: date | str | datetime | None = None,
) -> dict:
    """逐条比对并汇总，输出"我能不能投这个标"。

    返回：
        {
          "overall": "qualified" | "unqualified" | "need_review",
          "overall_label": str,          # 中文结论，前端直接展示
          "requirements": [ {..., "status", "reason", "evidence", "suggestion"} ],
          "gaps": [ ... ],               # **只含 unmet 的硬性要求**
          "summary": {
             "total", "met", "unmet", "uncertain",
             "mandatory_total", "mandatory_unmet", "mandatory_uncertain",
             "optional_unmet",
             "by_kind": {...},
             "reference_date": "YYYY-MM-DD",
             "conclusion": str,          # 一句话结论
             "caveats": [str, ...],      # 必须随结论一起展示的局限说明
          },
        }

    `overall` 的保守判定（顺序不可调换）：
        1. 任一 `is_mandatory` 要求 unmet → `unqualified`
        2. 否则只要有 uncertain      → `need_review`（**不允许**报 qualified）
        3. 其余（全部 met）          → `qualified`

    加分项（`is_mandatory=False`）不满足**不影响** overall，也不进 gaps——
    把它算进缺口会让用户误以为标投不了。
    """
    ref = _as_date(reference_date)
    requirements = parse_qualification_requirements(tender_parsed, ref)
    data = _as_profile(profile)

    results: list[dict] = []
    for req in requirements:
        verdict = match_qualification(req, data, ref)
        item = dict(req)
        item.update(verdict)
        item["status_label"] = STATUS_LABELS.get(verdict["status"], verdict["status"])
        item["suggestion"] = (
            _GAP_SUGGESTIONS.get(req["kind"], _GAP_SUGGESTIONS[KIND_OTHER])
            if verdict["status"] == STATUS_UNMET
            else ""
        )
        results.append(item)

    mandatory = [r for r in results if r["is_mandatory"]]
    optional = [r for r in results if not r["is_mandatory"]]
    mandatory_unmet = [r for r in mandatory if r["status"] == STATUS_UNMET]
    mandatory_uncertain = [r for r in mandatory if r["status"] == STATUS_UNCERTAIN]

    if mandatory_unmet:
        overall = OVERALL_UNQUALIFIED
    elif any(r["status"] == STATUS_UNCERTAIN for r in results):
        overall = OVERALL_NEED_REVIEW
    else:
        overall = OVERALL_QUALIFIED

    # gaps 只列硬性 unmet：这是用户最需要立刻知道的部分
    gaps = [
        {
            "raw_text": r["raw_text"],
            "kind": r["kind"],
            "kind_label": r["kind_label"],
            "category": r["category"],
            "reason": r["reason"],
            "suggestion": r["suggestion"],
        }
        for r in mandatory_unmet
    ]

    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_kind.setdefault(r["kind"], {"total": 0, STATUS_MET: 0, STATUS_UNMET: 0, STATUS_UNCERTAIN: 0})
        bucket["total"] += 1
        bucket[r["status"]] += 1

    counts = {
        "total": len(results),
        STATUS_MET: sum(1 for r in results if r["status"] == STATUS_MET),
        STATUS_UNMET: sum(1 for r in results if r["status"] == STATUS_UNMET),
        STATUS_UNCERTAIN: sum(1 for r in results if r["status"] == STATUS_UNCERTAIN),
        "mandatory_total": len(mandatory),
        "mandatory_unmet": len(mandatory_unmet),
        "mandatory_uncertain": len(mandatory_uncertain),
        "optional_unmet": sum(1 for r in optional if r["status"] == STATUS_UNMET),
        "by_kind": by_kind,
        "reference_date": ref.isoformat(),
    }

    if overall == OVERALL_UNQUALIFIED:
        conclusion = (
            f"有 {len(mandatory_unmet)} 项硬性资格要求不满足，按现状投标会被废标——"
            "请先处理缺口，或放弃该项目"
        )
    elif overall == OVERALL_NEED_REVIEW:
        n_uncertain = counts[STATUS_UNCERTAIN]
        conclusion = (
            f"硬性要求中没有自动判定的缺口，但有 {n_uncertain} 项**需人工确认**"
            f"（另有 {counts[STATUS_MET]} 项已满足）——"
            "确认这些项之前不能认为可以投标"
        )
    else:
        conclusion = f"{counts[STATUS_MET]} 项资格要求全部满足，可以投标"

    caveats = [
        "「需人工确认」不等于「不满足」：多为档案信息缺失或条文口径模糊，"
        "按 reason 补齐信息后重新评估即可",
        "本结论只覆盖**资格要求**章，不含废标条款、技术要求与评分标准——"
        "投标前仍需逐条核对那几章",
        f"时间窗口以 {ref.isoformat()} 为基准推算（「近三年」= 起始年份不早于 {ref.year - 3}），"
        "年份口径存在行业歧义时按宽松侧取值",
        "判定来自规则匹配，企业档案由用户自行录入；档案不全时结论会偏保守（更多需人工确认）",
    ]

    logger.info(
        "资格审查完成：overall=%s，共 %d 条（met=%d/unmet=%d/uncertain=%d）",
        overall, counts["total"], counts[STATUS_MET], counts[STATUS_UNMET], counts[STATUS_UNCERTAIN],
    )

    return {
        "overall": overall,
        "overall_label": OVERALL_LABELS[overall],
        "requirements": results,
        "gaps": gaps,
        "summary": {**counts, "conclusion": conclusion, "caveats": caveats},
    }


def check_from_document(
    filename: str,
    data: bytes,
    profile: dict,
    reference_date: date | str | datetime | None = None,
) -> dict:
    """一步到位：招标文件字节流 → 解析 → LLM 抽取 → 资格比对。

    返回 `check_qualifications` 的全部字段，另加：
        "tender": 项目名称 / 编号 / 采购人 / 预算 / 截止时间（结果页顶部信息条）
        "parsed": 抽取出的完整结构化结果——**换一份企业档案重算时不必再调一次 LLM**
        "meta":   抽取元信息（模型、批次数、完整度、警告）

    解析与抽取的异常统一收敛成 `QualificationError`：本模块对调用方只暴露一种
    异常类型（见模块文档）。
    """
    from src.services.tender_extractor import TenderExtractError, extract_from_document

    try:
        extracted = extract_from_document(filename, data)
    except TenderExtractError as exc:
        raise QualificationError(f"招标文件抽取失败：{exc}") from exc
    except Exception as exc:  # 含 DocumentParseError（扫描件/格式不支持）
        raise QualificationError(
            f"招标文件解析失败：{exc}"
        ) from exc

    parsed = extracted.get("parsed") or {}
    check = check_qualifications(parsed, profile, reference_date)
    fields = parsed.get("fields") or {}
    check["tender"] = {
        "project_name": fields.get("project_name", ""),
        "project_id": fields.get("project_id", ""),
        "purchaser": fields.get("purchaser", ""),
        "budget": fields.get("budget"),
        "bid_deadline": fields.get("bid_deadline", ""),
    }
    check["parsed"] = parsed
    check["meta"] = extracted.get("meta") or {}
    return check


# ---------------------------------------------------------------------------
# 企业资质档案的存取（company_profiles 表）
# ---------------------------------------------------------------------------

def _dsn() -> dict[str, Any]:
    """从 .env 读连接参数（与 `analysis._dsn` 同策略：**绝不硬编码密码**）。"""
    load_dotenv(PROJECT_ROOT / ".env")
    import os

    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        raise QualificationError("未配置 PostgreSQL（POSTGRES_HOST 为空）")
    return {
        "host": host,
        "port": (os.getenv("POSTGRES_PORT") or "5432").strip(),
        "user": (os.getenv("POSTGRES_USER") or "postgres").strip(),
        # 密码末尾可能带 \r（Windows 上编辑 .env 常见），必须清掉
        "password": (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
        "dbname": (os.getenv("POSTGRES_DB") or "chatbot").strip(),
    }


class _Cursor:
    """轻量上下文管理器：连库 → 用 → **提交** → 必关。

    与 `analysis._Cursor` 的唯一差别是退出时显式 `commit()`：本模块有写路径
    （档案 CRUD），而 psycopg 关连接时未提交的事务会被回滚——只读模块可以不提交，
    写模块不提交就是"接口返回成功、数据没进库"。异常时回滚，不留半截数据。
    """

    def __enter__(self):
        try:
            self._conn = psycopg.connect(**_dsn())
        except Exception as exc:
            raise QualificationError(f"数据库连接失败：{type(exc).__name__}") from exc
        self._cur = self._conn.cursor()
        return self._cur

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        except Exception:  # pragma: no cover
            pass
        finally:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass
        return False


# 建表语句。**在服务里建表、不依赖手工 DDL**：使用者拿到代码就能跑。
#
#   - name 唯一：档案按名称寻址，同名只应有一条（重复档案会让"用哪份"变成随机）
#   - profile 用 JSONB 而不是 TEXT：档案本就是要按字段读的结构
#     （`profile->>'company_name'`、`jsonb_array_length(profile->'qualifications')`），
#     存成字符串后想按内容筛选只能全表 LIKE
#   - created_at / updated_at 由库填：写入方不该关心时间，也不该能写入一个没有时间的档案
_DDL: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id         SERIAL PRIMARY KEY,
        name       TEXT NOT NULL UNIQUE,
        profile    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 列表按 updated_at 倒序取，没有索引就是全表扫描 + 排序
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated_at ON {TABLE} (updated_at DESC)",
)


def _ensure_table(cur) -> None:
    """幂等建表（每次读写前执行）。

    为什么每次都建而不是缓存一个"已建好"标志：一条 `CREATE TABLE IF NOT EXISTS`
    在 PG 里只是查一次系统目录，开销远小于它的收益——服务重启、库被清空、
    测试里 drop 了表，下一条读写都会自愈。资格审查是低频操作（用户点一次按钮），
    这点开销换"永远能跑"很划算。
    """
    for statement in _DDL:
        cur.execute(statement)


def _jsonable(value: Any) -> Any:
    """时间戳 → ISO 字符串（返回值要进 JSON / SSE）。"""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _clean_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        raise QualificationError("企业档案名称不能为空")
    if len(text) > 200:
        raise QualificationError("企业档案名称过长（上限 200 字符）")
    return text


def save_profile(name: str, profile: dict) -> dict:
    """新增或更新企业资质档案（同名覆盖），返回档案元信息。

    用 `INSERT ... ON CONFLICT (name) DO UPDATE` 而不是"先查再写"：后者在并发下
    会撞唯一约束，且两次往返之间档案可能已被改动。`(xmax = 0)` 用来区分本次是
    新增还是更新——返回值里的 `created` 让调用方能给出"已创建 / 已更新"的提示。

    档案结构（各字段都可缺省，缺得越多判定越保守）：
        {
          "company_name": "某某科技有限公司",
          "business_license": "91340100MA2TXXXXXX",
          "qualifications": [{"name": "电子与智能化工程专业承包", "level": "二级",
                              "cert_no": "D234...", "valid_until": "2027-06-30"}],
          "certifications": ["ISO9001", "ISO27001"],
          "performance": [{"project": "某政务云项目", "amount": 5800000, "year": 2024}],
          "financial": {"revenue": 120000000, "audited": true, "healthy": true,
                        "registered_capital": 20000000},
          "personnel": [{"name": "张工", "cert": "信息系统项目管理师", "level": "高级"}],
          "credit": {"dishonest": false, "tax_violation": false}
        }
    """
    key = _clean_name(name)
    data = _as_profile(profile)
    if not isinstance(profile, dict):
        raise QualificationError("企业资质档案应为 dict（可为空 dict，但不能是 None/字符串）")

    sql = f"""
    INSERT INTO {TABLE} (name, profile) VALUES (%s, %s)
    ON CONFLICT (name) DO UPDATE
        SET profile = EXCLUDED.profile, updated_at = now()
    RETURNING id, name, created_at, updated_at, (xmax = 0) AS created
    """
    with _Cursor() as cur:
        _ensure_table(cur)
        cur.execute(sql, (key, Jsonb(data)))
        row = cur.fetchone()
    if row is None:  # pragma: no cover - INSERT ... RETURNING 必有行
        raise QualificationError("保存企业档案失败：数据库未返回结果")
    return {
        "id": row[0],
        "name": row[1],
        "created": bool(row[4]),
        "created_at": _jsonable(row[2]),
        "updated_at": _jsonable(row[3]),
    }


def load_profile(name: str) -> dict:
    """按名称读取企业资质档案；不存在时抛 `QualificationError`。

    ★ 不存在时**抛异常而不是返回 `{}`**：空字典与"企业什么都没填"无法区分，
    若静默返回空，用户打错一个档案名就会得到"全部需人工确认"的报告，
    却看不出是自己选错了档案。异常信息里带上现有档案名，便于纠正。
    """
    key = _clean_name(name)
    with _Cursor() as cur:
        _ensure_table(cur)
        cur.execute(f"SELECT profile, created_at, updated_at FROM {TABLE} WHERE name = %s", (key,))
        row = cur.fetchone()
        if row is None:
            cur.execute(f"SELECT name FROM {TABLE} ORDER BY name LIMIT 20")
            existing = [r[0] for r in cur.fetchall()]
            hint = f"现有档案：{'、'.join(existing)}" if existing else "当前还没有任何企业档案"
            raise QualificationError(f"企业档案「{key}」不存在（{hint}）")
    data = row[0]
    if isinstance(data, str):  # 理论上 JSONB 出来就是 dict，兜一层防 TEXT 列
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            raise QualificationError(f"企业档案「{key}」的内容不是合法 JSON") from None
    out = dict(data or {})
    out.setdefault("_meta", {})
    out["_meta"].update(
        {"name": key, "created_at": _jsonable(row[1]), "updated_at": _jsonable(row[2])}
    )
    return out


def list_profiles() -> list[dict]:
    """列出全部企业档案（最近更新的在前），**不返回完整档案**。

    返回 `[{"name", "company_name", "qualification_count", "certification_count",
    "performance_count", "personnel_count", "created_at", "updated_at"}]`。

    为什么不返回完整 profile：列表页只需要摘要，而档案里的业绩明细可能有几十条，
    全量序列化进列表接口会让首屏变慢。要完整内容再调 `load_profile`。
    """
    sql = f"""
    SELECT name,
           COALESCE(profile->>'company_name', '') AS company_name,
           CASE WHEN jsonb_typeof(profile->'qualifications') = 'array'
                THEN jsonb_array_length(profile->'qualifications') ELSE 0 END AS qualification_count,
           CASE WHEN jsonb_typeof(profile->'certifications') = 'array'
                THEN jsonb_array_length(profile->'certifications') ELSE 0 END AS certification_count,
           CASE WHEN jsonb_typeof(profile->'performance') = 'array'
                THEN jsonb_array_length(profile->'performance') ELSE 0 END AS performance_count,
           CASE WHEN jsonb_typeof(profile->'personnel') = 'array'
                THEN jsonb_array_length(profile->'personnel') ELSE 0 END AS personnel_count,
           created_at, updated_at
    FROM {TABLE}
    ORDER BY updated_at DESC, name
    """
    with _Cursor() as cur:
        _ensure_table(cur)
        cur.execute(sql)
        rows = cur.fetchall()
    return [
        {
            "name": r[0],
            "company_name": r[1],
            "qualification_count": r[2],
            "certification_count": r[3],
            "performance_count": r[4],
            "personnel_count": r[5],
            "created_at": _jsonable(r[6]),
            "updated_at": _jsonable(r[7]),
        }
        for r in rows
    ]


def delete_profile(name: str) -> bool:
    """删除企业档案，返回是否真的删掉了一条（不存在返回 `False`，不抛异常）。

    与 `load_profile` 的处理相反：删除是幂等操作，"删一个已经不在的档案"目标
    已经达成，报错反而让调用方要额外判一次。
    """
    key = _clean_name(name)
    with _Cursor() as cur:
        _ensure_table(cur)
        cur.execute(f"DELETE FROM {TABLE} WHERE name = %s RETURNING id", (key,))
        deleted = cur.fetchone() is not None
    if deleted:
        logger.info("已删除企业档案：%s", key)
    return deleted


__all__ = [
    "CATEGORY_TO_KIND",
    "KIND_BASIC",
    "KIND_CERTIFICATION",
    "KIND_CREDIT",
    "KIND_FINANCIAL",
    "KIND_LABELS",
    "KIND_OTHER",
    "KIND_PERFORMANCE",
    "KIND_PERSONNEL",
    "KIND_QUALIFICATION",
    "LEVEL_RANKS",
    "OVERALL_LABELS",
    "OVERALL_NEED_REVIEW",
    "OVERALL_QUALIFIED",
    "OVERALL_UNQUALIFIED",
    "QualificationError",
    "STATUS_LABELS",
    "STATUS_MET",
    "STATUS_UNCERTAIN",
    "STATUS_UNMET",
    "TABLE",
    "check_from_document",
    "check_qualifications",
    "compare_level",
    "delete_profile",
    "extract_certifications",
    "list_profiles",
    "load_profile",
    "match_qualification",
    "name_match",
    "parse_amount",
    "parse_level",
    "parse_qualification_requirements",
    "parse_years",
    "save_profile",
]
