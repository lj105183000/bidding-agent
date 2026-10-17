"""招标文件合规性检查：检出涉嫌违规的条款并引用法规依据。

为什么用"规则 + 法规引用"两层，而不是纯靠大模型判断
--------------------------------------------------
纯让模型"看看这份招标文件合不合规"会得到一个**看起来专业但无法追溯**的结论，
而合规判断是要担责任的。故设计成两层：

    第一层：**规则检出**（确定性）
        从招标文件文本里匹配已知的违规特征模式，例如
        "要求投标人在本地设有分支机构"、"注册资本不低于 5000 万元"。
        规则命中是确定的、可解释的、可复现的——用户能看到**为什么**被标记。

    第二层：**法规引用**（依据）
        为每条命中的规则挂上法规条文（`bid_regulations` 语料），
        让用户能自己去核对原文。

这样即使法规语料不全，第一层仍然能工作；语料只是让结论更有依据。

⚠️ 本模块给出的是"**疑似**"而非"违法认定"
----------------------------------------
同一个表述在不同项目背景下合法性可能不同（例如"要求本地业绩"，
若确因项目需要现场服务能力，可能被认定为合理）。
故：
    - 输出用 `suspected`（疑似）而非 `violation`（违规）
    - 每条都给出 `why`（为什么可疑）与 `how_to_verify`（怎么核实）
    - 汇总里明确写"本结论不构成法律意见"

把"疑似"包装成"违规认定"会让用户拿去做错误决策——那比不做检查更糟。

⚠️ 法规检索的已知短板（必须如实交代）
-----------------------------------
法规条文用的是**稠密向量相似度**检索，而同一主题下常有多条**都确实相关**的
法条，向量相似度区分不了它们的适用差别。实测例子：

    查询"投标保证金比例" → 第二条会串到《招标投标法实施条例》第五十八条
    （**履约**保证金 10%），而《政府采购法实施条例》第三十三条
    （采购预算 2%）可能掉出前 3。

    根因不是排序算法差，而是"哪一类项目、哪一类保证金"需要读懂语义，
    超出向量相似度的能力范围。语料方在 15 条标注查询上做过权重扫描，
    提升有限。

应对方式（本模块已采用）：
    1. **把区分信息写进查询词**（"投标保证金 不得超过 预算金额 2%
       退还 非现金形式 银行保函"），实测能把正确条文提到 top-1/2
    2. **取 top_k=3 而非 top-1**，给用户多条依据自己判断

    若将来要进一步提高精度，正确方向是"取 top_k=5~8 + 让 LLM 判读哪条适用"，
    而不是继续调检索权重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.logging_config import get_logger

logger = get_logger(__name__)

# 严重度分档
#   high   直接限制竞争，实践中常被财政部门/监管部门认定违规
#   medium 存在争议，需结合项目背景判断
#   low    表述不规范或可能引起误解，建议澄清
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class ComplianceError(RuntimeError):
    """合规检查失败。"""


@dataclass(frozen=True)
class ComplianceRule:
    """一条合规检查规则。

    `detect` 接收**单条文本**，命中返回证据片段（供展示），未命中返回 None。
    规则作用于"逐条"而不是"整篇"：招标文件的条款是按条列出的，
    逐条判断才能精确定位到"第几条有问题"。
    """

    key: str
    title: str
    severity: str  # high / medium / low
    # 规则说明：为什么这种表述可疑
    why: str
    # 建议怎么核实 / 怎么处理
    how_to_verify: str
    detect: Callable[[str], str | None]
    # 检索法规时用的查询词（用于取法规依据）
    law_query: str = ""
    # 该规则关注的主题标签（与法规语料的 topics 对应）
    topics: tuple[str, ...] = field(default_factory=tuple)


# 各规则的金额/数量阈值
#
# 阈值经过一轮实测调整：第一版注册资本阈值定 1000 万，结果
# `注册资本人民币 1000 万元` 被标记——但 1000 万在实务中很常见，
# 用它做门槛通常与项目规模匹配。**假阳性会摧毁整个检查的可信度**：
# 用户看几条无关紧要的提示后就再也不认真看了。
#
# 故只保留「单看数字就明显偏高」的档位，并把真正的判据
# （与项目预算比对）写进规则的 how_to_verify，交给用户核对。
CAPITAL_THRESHOLD = 50_000_000            # 注册资本 5000 万
SCALE_THRESHOLD = 50_000_000              # 营业收入 / 资产总额 5000 万
EXPERIENCE_AMOUNT_THRESHOLD = 10_000_000  # 单项合同业绩 1000 万
EXPERIENCE_COUNT_THRESHOLD = 5            # 同类业绩个数


def _cn_num(text: str) -> int | None:
    """把中文数字转成整数（支持"三""十五""三十"这类写法）。

    ★ 为什么必须支持中文数字（实测漏报）
    ---------------------------------
    招标文件里"近**三**年内完成过 6 个以上项目"极常见，而第一版正则只认
    `\\d+`，于是这类条款被**静默漏过**——漏报比误报危险得多，因为用户会
    以为"检查过了没问题"。

    只支持到 99（业绩年限、个数不会更大），不引入完整中文数字库。
    """
    s = (text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(s) == 1 and s in digits:
        return digits[s]
    return None


def _to_amount(raw: str, unit: str) -> float | None:
    """把「5000 万元」这类金额文本换算成元。"""
    try:
        amount = float(raw.replace(",", "").replace("，", ""))
    except ValueError:
        return None
    if unit == "万":
        amount *= 10_000
    elif unit == "亿":
        amount *= 100_000_000
    return amount


def _hit(pattern: str, text: str, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# 规则库
#
# 每条规则都对应实践中最常见的"以不合理条件限制/排斥潜在投标人"情形。
# 依据主要来自《招标投标法实施条例》第 32 条、《政府采购法实施条例》第 20 条、
# 财政部 87 号令第 17/55 条等（具体条文由法规语料层提供）。
# ---------------------------------------------------------------------------


def _detect_local_requirement(text: str) -> str | None:
    """本地化要求：要求在本地注册/设分支机构/本地业绩。"""
    return (
        _hit(r"(在|于)?[^，。；]{0,6}(本市|本地|当地|省内|县内)[^，。；]{0,8}"
             r"(注册|设立|设有|具有|拥有)[^，。；]{0,10}(分支机构|分公司|办事处|办公场所|经营场所)", text)
        or _hit(r"(投标人|供应商|申请人)[^，。；]{0,12}(须|应|必须|需)[^，。；]{0,8}"
                r"(在|于)[^，。；]{0,10}(本地|本市|当地)[^，。；]{0,8}(注册|纳税|缴纳社保)", text)
        or _hit(r"(本地|本市|当地)[^，。；]{0,6}(业绩|案例|合同)", text)
    )


def _money_pattern(label: str) -> str:
    """构造"<标签>…<金额>"的匹配模式，**优先让单位紧跟数字**。

    ★ 为什么顺序重要（实测误报）
    --------------------------
    第一版写的是 `{label}[^，。；]{0,12}?([\\d,，.]+)\\s*(万|亿)?元?`，
    于是 `注册资本人民币 1000 万元` 里的 **"币"** 被 `[\\d,，.]+` 吃进去，
    捕获组变成 `"币 1000"` → 解析失败返回 None → 实际是**漏报**；
    而它同时又会把 `1000 万` 前缀误读……两种错法都不对。

    正解是让"单位紧跟数字"的分支**排在前面**：
        ① `数字 + (万|亿) + 元` —— 最明确，无歧义
        ② `数字 + 元`         —— 无单位
    这样 `[^，。；]` 的模糊匹配只在两条明确分支都不成立时才兜底。
    """
    return (
        rf"{label}[^，。；]{{0,12}}?"
        r"([\d,，.]+)\s*(万|亿)\s*元"
        rf"|{label}[^，。；]{{0,12}}?([\d,，.]+)\s*元"
    )


def _extract_amount(text: str, label: str) -> tuple[str, float] | None:
    """从文本里取出 `<label> … <金额>`，返回 `(证据片段, 金额元)`。"""
    m = re.search(_money_pattern(label), text)
    if not m:
        return None
    # 分支①有 unit，分支②没有
    raw = m.group(1) or m.group(3) or ""
    unit = m.group(2) or ""
    if not raw:
        return None
    amount = _to_amount(raw, unit)
    if amount is None:
        return None
    return m.group(0), amount


def _detect_registered_capital(text: str) -> str | None:
    """注册资本门槛。**单看数字明显偏高**时才标记。

    ★ 阈值为什么从 1000 万提到 5000 万（实测调过一轮）
    ----------------------------------------------
    第一版阈值定 1000 万，于是 `注册资本人民币 1000 万元` 被标记。
    调试确认解析是对的（正确识别为 10,000,000 元）——**错的是阈值**：
    1000 万注册资本在招投标实务中很常见，用它做门槛通常与项目规模匹配。

    报出这类"看着像但不是问题"的项，会让用户看几条无关紧要的提示后
    就再也不认真看了——**假阳性会摧毁整个检查的可信度**。

    真正的判据是「与项目预算比对」，但那需要预算信息（单条文本里不一定有），
    故不在规则里瞎猜，而是写进 `how_to_verify` 让用户按指引核对。
    """
    got = _extract_amount(text, "注册资本")
    if got is None:
        return None
    evidence, amount = got
    return evidence if amount >= CAPITAL_THRESHOLD else None


def _detect_specific_brand(text: str) -> str | None:
    """指定品牌 / 型号 / 专利，且未写"或同等"。"""
    hit = _hit(r"(品牌|商标)[^，。；]{0,6}(须|应|必须|需|为)[^，。；]{0,20}", text)
    if not hit:
        hit = _hit(r"(指定|要求)[^，。；]{0,4}(品牌|商标|型号|专利)", text)
    if not hit:
        return None
    # 写了"或同等"/"相当于"的属于合规写法，不标记
    if re.search(r"(或同等|或相当于|同等档次|不低于该档次|参考品牌)", text):
        return None
    return hit


def _detect_excessive_deposit(text: str) -> str | None:
    """投标保证金超过招标项目预算金额的 2%。"""
    m = re.search(r"投标保证金[^，。；]{0,20}?([\d,，.]+)\s*(万|亿)?元?", text)
    return m.group(0) if m else None


def _detect_specific_certificate(text: str) -> str | None:
    """以特定证书/奖项/协会会员作为资格或加分条件。"""
    return (
        _hit(r"(获得|具有|提供)[^，。；]{0,10}(鲁班奖|詹天佑奖|优质工程奖|省级以上[^，。；]{0,6}奖)", text)
        or _hit(r"(须|应|必须|需|要求)[^，。；]{0,6}(加入|是)[^，。；]{0,8}(行业协会|商会|学会)", text)
        or _hit(r"(省级|部级|国家级)[^，。；]{0,6}(奖项|荣誉|表彰)[^，。；]{0,10}(加分|作为[^，。；]{0,4}条件)", text)
    )


def _detect_ownership_discrimination(text: str) -> str | None:
    """所有制歧视：要求国有/特定所有制，排斥外资或民营。"""
    return (
        _hit(r"(须|应|必须|要求)[^，。；]{0,6}(国有|国营|公有制)[^，。；]{0,6}(企业|单位|控股)", text)
        or _hit(r"(不接受|拒绝|排除)[^，。；]{0,8}(外资|外商|民营企业|个体工商户|分支机构)", text)
        or _hit(r"(限|仅限)[^，。；]{0,10}(国有企业|国企|央企)", text)
    )


def _detect_scale_discrimination(text: str) -> str | None:
    """规模歧视：以营业收入/资产总额等超出项目需要的指标设门槛。"""
    for label in ("营业收入", "营业额", "资产总额", "净资产"):
        got = _extract_amount(text, label)
        if got is None:
            continue
        evidence, amount = got
        if amount >= SCALE_THRESHOLD:
            return evidence
    return None


def _detect_unquantified_scoring(text: str) -> str | None:
    """评审因素未量化：只有"优/良/一般"这类主观档而没有分值区间。"""
    hit = _hit(r"(优|良|中|一般|较差)[^，。；]{0,4}(得|计|赋)[^，。；]{0,4}分", text)
    if not hit:
        hit = _hit(r"(根据|视)[^，。；]{0,10}(情况|程度)[^，。；]{0,6}(酌情|自行)[^，。；]{0,4}(打分|评分|赋分)", text)
    if not hit:
        hit = _hit(r"(综合|整体)[^，。；]{0,6}(印象|评价)[^，。；]{0,6}(打分|评分)", text)
    if not hit:
        return None
    # 若同行给出了具体分值区间，则属于合规写法
    if re.search(r"\d+\s*[~～\-至]\s*\d+\s*分|\d+\s*分", text):
        return None
    return hit


def _detect_payment_terms_unfair(text: str) -> str | None:
    """付款条件明显不合理：要求垫资、验收后长期不付、无预付款且账期过长。"""
    return (
        _hit(r"(完工|验收合格)[^，。；]{0,10}(后|之日起)[^，。；]{0,6}(\d{3,})\s*(个)?(工作)?日[^，。；]{0,6}(内)?[^，。；]{0,4}支付", text)
        or _hit(r"(无预付款|不予支付预付款)[^，。；]{0,20}(垫资|全额垫资)", text)
        or _hit(r"(须|应|必须)[^，。；]{0,6}(全额)?垫资", text)
    )


def _detect_deposit_only_cash(text: str) -> str | None:
    """保证金只接受现金/转账，不接受保函等非现金形式。"""
    hit = _hit(r"投标保证金[^，。；]{0,20}(仅|只)[^，。；]{0,6}(接受|限)[^，。；]{0,8}(现金|电汇|转账|银行转账)", text)
    if not hit:
        hit = _hit(r"(不接受|不认可)[^，。；]{0,6}(保函|银行保函|保险保单|担保函)", text)
    return hit


def _detect_excessive_experience(text: str) -> str | None:
    """业绩门槛过高：要求近 X 年内 N 个以上同类项目，或单个合同额门槛过高。

    数字**同时支持阿拉伯与中文**——“近三年内完成过 6 个以上项目”这类写法
    在招标文件里极常见，只认 `\\d+` 会静默漏报。
    """
    # 中文数字与阿拉伯数字都要认
    num = r"(\d+|[一二两三四五六七八九十]+)"
    m = re.search(
        rf"近\s*{num}\s*年[^，。；]{{0,20}}?{num}\s*(个|项|份)[^，。；]{{0,10}}(业绩|合同|项目)",
        text,
    )
    if m:
        n = _cn_num(m.group(2))
        if n is not None and n >= EXPERIENCE_COUNT_THRESHOLD:
            return m.group(0)

    m2 = re.search(
        r"(单项|单个)?(合同|项目)[^，。；]{0,6}(金额|额)[^，。；]{0,8}"
        r"(不低于|不少于|≥)\s*([\d,，.]+)\s*(万|亿)\s*元",
        text,
    )
    if m2:
        amount = _to_amount(m2.group(5), m2.group(6))
        if amount is not None and amount >= EXPERIENCE_AMOUNT_THRESHOLD:
            return m2.group(0)
    # 无单位写法（金额通常很小，一般不触发门槛，故放在后面）
    m3 = re.search(
        r"(单项|单个)?(合同|项目)[^，。；]{0,6}(金额|额)[^，。；]{0,8}"
        r"(不低于|不少于|≥)\s*([\d,，.]+)\s*元",
        text,
    )
    if m3:
        amount = _to_amount(m3.group(5), "")
        if amount is not None and amount >= EXPERIENCE_AMOUNT_THRESHOLD:
            return m3.group(0)
    return None


def _detect_unclear_evaluation(text: str) -> str | None:
    """评标办法含糊：没有明确评标方法或价格分权重。"""
    return _hit(r"(评标办法|评审办法)[^，。；]{0,6}(由|由招标人|另行|见)[^，。；]{0,10}(确定|通知|公布)", text)


RULES: tuple[ComplianceRule, ...] = (
    ComplianceRule(
        key="local_requirement",
        title="以本地化条件限制投标人",
        severity="high",
        why="要求在本地注册、设立分支机构或以本地业绩作为条件，属于以不合理条件"
            "限制、排斥潜在投标人，是最典型的违规情形之一。",
        how_to_verify="核对原文是否确有此要求。若确因项目需要现场服务能力，"
                      "合规写法应是『中标后承诺在本地设立服务机构』，而不是把它作为投标资格条件。",
        detect=_detect_local_requirement,
        law_query="以不合理条件限制排斥潜在投标人 本地业绩 分支机构",
        topics=("差别待遇", "资格条件"),
    ),
    ComplianceRule(
        key="registered_capital",
        title="注册资本门槛过高",
        severity="medium",
        why="以超出项目实际需要的注册资本作为资格条件，会排斥中小企业。"
            "注册资本与履约能力并无必然联系。",
        how_to_verify="对照**项目预算金额**判断：门槛远高于预算即为不合理"
                      "（例如预算 100 万却要求注册资本 5000 万）。"
                      "本规则只在单看数字已明显偏高时提示，未提示不代表合理——"
                      "仍建议与预算比对一次。",
        detect=_detect_registered_capital,
        law_query="不得以注册资本 营业收入 资产总额等规模条件限制供应商",
        topics=("规模条件", "资格条件"),
    ),
    ComplianceRule(
        key="specific_brand",
        title="指定特定品牌或型号",
        severity="high",
        why="指定品牌、商标、型号且未写『或同等』，会直接排斥其他品牌产品。"
            "依法应当以技术参数描述采购需求。",
        how_to_verify="若确需参照某品牌，合规写法是『参考品牌：XX，或同等档次产品』，"
                      "并给出可量化的技术参数。",
        detect=_detect_specific_brand,
        law_query="招标文件不得要求或者标明特定的生产供应者 含有倾向或者排斥潜在投标人的其他内容",
        topics=("指定品牌", "采购需求"),
    ),
    ComplianceRule(
        key="specific_certificate",
        title="以特定奖项或协会会员作为条件",
        severity="high",
        why="以特定奖项、荣誉或行业协会会员资格作为资格条件或加分项，"
            "与采购项目的实际需要无关，属于差别待遇。",
        how_to_verify="核对是否写入了具体奖项名称或协会名称。"
                      "依法可以要求的是与履约能力直接相关的资质证书。",
        detect=_detect_specific_certificate,
        law_query="将特定行政区域的业绩、奖项作为加分条件或者中标条件属于以不合理条件限制排斥潜在投标人",
        topics=("差别待遇", "评审因素"),
    ),
    ComplianceRule(
        key="ownership_discrimination",
        title="所有制歧视",
        severity="high",
        why="要求国有企业身份，或排斥外资、民营、个体工商户，"
            "违反公平竞争原则与优化营商环境要求。",
        how_to_verify="核对原文。供应商依法平等参与，不得因所有制形式受到差别待遇。",
        detect=_detect_ownership_discrimination,
        law_query="不得因所有制形式 差别待遇 歧视 外资 民营企业",
        topics=("差别待遇", "公平竞争"),
    ),
    ComplianceRule(
        key="scale_discrimination",
        title="以营业收入/资产总额设置高门槛",
        severity="medium",
        why="以营业收入、资产总额等规模指标设置远超项目需要的门槛，"
            "会排斥中小企业参与。",
        how_to_verify="对照项目预算判断门槛合理性。规模指标应当与合同履行能力相关且有依据。",
        detect=_detect_scale_discrimination,
        law_query="不得将投标人的注册资本 资产总额 营业收入 从业人员 利润 纳税额等规模条件作为资格要求或者评审因素",
        topics=("规模条件", "资格条件"),
    ),
    ComplianceRule(
        key="excessive_experience",
        title="业绩门槛偏高",
        severity="medium",
        why="要求过多的同类业绩数量或过高的单项合同额门槛，"
            "可能超出项目实际需要，排斥潜在投标人。",
        how_to_verify="对照项目规模判断。业绩要求应当与项目复杂程度相当，"
                      "且不得限定为特定行政区域或特定行业的业绩。",
        detect=_detect_excessive_experience,
        law_query="不得以特定行政区域或者特定行业的业绩、奖项作为资格条件 以不合理条件限制排斥潜在投标人",
        topics=("业绩要求", "资格条件"),
    ),
    ComplianceRule(
        key="deposit_only_cash",
        title="保证金仅接受现金形式",
        severity="medium",
        why="只接受现金或转账、不接受银行保函等非现金形式，会增加供应商资金占用，"
            "与优化营商环境要求不符。",
        how_to_verify="依法应当允许以银行保函、保险保单等非现金形式提交保证金。",
        detect=_detect_deposit_only_cash,
        law_query="投标保证金应当以 支票 汇票 本票 银行保函 等非现金形式提交 不得限定现金",
        topics=("保证金",),
    ),
    ComplianceRule(
        key="excessive_deposit",
        title="投标保证金（需核对比例）",
        severity="medium",
        why="投标保证金不得超过招标项目预算金额的 2%。"
            "本项为**提示核对**——条款里出现了保证金金额，需与预算金额比对。",
        how_to_verify="用该保证金金额 ÷ 预算金额，若超过 2% 则涉嫌超标。",
        detect=_detect_excessive_deposit,
        law_query="投标保证金 不得超过 预算金额 2% 退还 非现金形式 银行保函",
        topics=("保证金",),
    ),
    ComplianceRule(
        key="unquantified_scoring",
        title="评审因素未量化",
        severity="medium",
        why="评审因素只有『优/良/一般』这类主观档次而没有明确分值区间，"
            "会导致评审自由裁量过大。依法评审因素应当尽可能量化。",
        how_to_verify="合规写法应为『优：8-10 分；良：5-7 分；一般：0-4 分』这类明确区间。",
        detect=_detect_unquantified_scoring,
        law_query="评审因素应当量化 细化 主观评分",
        topics=("评审因素",),
    ),
    ComplianceRule(
        key="unclear_evaluation",
        title="评标办法未明确",
        severity="low",
        why="评标办法表述含糊（如『由招标人另行确定』），"
            "会让投标人无法判断评分标准，也影响公平性。",
        how_to_verify="评标办法应当在招标文件中明确载明。",
        detect=_detect_unclear_evaluation,
        law_query="评标办法应当在招标文件中载明 评标标准",
        topics=("评审因素",),
    ),
    ComplianceRule(
        key="payment_terms_unfair",
        title="付款条件可能不合理",
        severity="medium",
        why="要求垫资，或验收后账期过长（超过 90 日），会显著增加供应商资金压力，"
            "对中小企业影响尤大。",
        how_to_verify="核对付款节点与账期。政府采购合同一般应在验收合格后及时支付。",
        detect=_detect_payment_terms_unfair,
        law_query="政府采购合同应当及时支付 不得要求供应商垫资 付款期限 验收合格后支付",
        topics=("付款条件",),
    ),
)


def _iter_clauses(parsed: dict[str, Any]) -> list[tuple[str, str, int]]:
    """把抽取结果摊平成 `(章节key, 文本, 序号)` 列表。

    合规检查要**逐条**做，而不是把整篇当一段——只有逐条才能定位到
    "第 3 条资格要求有问题"。故标量字段与列表章节都摊平。
    """
    clauses: list[tuple[str, str, int]] = []
    fields = parsed.get("fields") or {}
    for key, value in fields.items():
        if value not in (None, "", []):
            clauses.append((key, str(value), 0))
    for sec_key, items in (parsed.get("sections") or {}).items():
        for i, item in enumerate(items or []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text:
                clauses.append((sec_key, str(text), i))
    return clauses


def _fetch_law_basis(rule: ComplianceRule, text: str) -> list[dict[str, Any]]:
    """为命中的规则取法规依据。

    语料不可用（未入库/未启动）时返回空列表而**不是报错**——
    规则检出本身仍有价值，法规依据只是增强。

    失败只在**首次**记日志（模块级标记）：实测同一份文件会命中多条规则，
    若每次都打日志会刷出十几行同样的"不可用"，把真正的信号淹没。

    ★ 主题过滤为空时**必须退回无过滤检索**（实测踩坑）
    ------------------------------------------------
    `regulation_store.search_regulations` 的 `topics` 是**先于排序的硬过滤**，
    故主题名一旦与语料词表不一致，就会**静默返回 0 条**。

    实测：规则写 `topics=("品牌限制",)`，而语料标签是 `"指定品牌"`，
    于是「指定特定品牌或型号」这条规则永远拿不到依据——而正确条文
    （《招标投标法》第二十条）明明在库里。这种失败**不报错、不打日志**，
    只能靠逐条核对输出才能发现。

    故这里做两级检索：先带主题过滤（更精准），为空则退回无过滤（保证有依据）。

    ★ 低于分数门槛的依据宁可不要（实测踩坑）
    ----------------------------------------
    实测「付款条件可能不合理」规则检索出的 top-3 分数只有 0.52~0.56，
    内容分别是"谈判程序""质疑答复期限"——**与付款条件无关**。

    给出一条分数很低的法规**比不给更糟**：用户看到"《XX法》第X条"会以为
    规则有依据，实际上那条法条说的是别的事。故设 `_MIN_LAW_SCORE` 门槛，
    低于门槛就返回空，让界面如实显示"未附法规条文"。
    """
    global _LAW_BASIS_WARNED

    try:
        from src.services import regulation_store

        query = rule.law_query or rule.title

        # 两个来源的候选**合并后统一过滤**。
        #
        # ★ 为什么不是"主题检索为空才退回无过滤"（实测 bug）
        # --------------------------------------------------
        # 第一版写成 if 主题检索有结果就用它、否则才无过滤。问题是主题检索的结果
        # **没有经过分数门槛**——实测「指定特定品牌」规则拿回一条 0.59 的
        # 《非招标采购方式管理办法》第十条（讲的是谈判文件制定，与指定品牌无关），
        # 却因为"主题检索有结果"而绕过了门槛。
        #
        # 合并后再过滤，才能保证输出到界面的每一条依据都真的够相关。
        candidates: list[dict] = []
        seen: set[tuple] = set()
        sources = []
        if rule.topics:
            sources.append(regulation_store.search_regulations(
                query, top_k=3, topics=list(rule.topics)
            ))
        sources.append(regulation_store.search_regulations(query, top_k=3))
        for hits in sources:
            for h in hits:
                key = (h.get("law"), h.get("article"))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(h)

        # 依据的取舍：**主题命中放行，分数只作为补充通道**
        #
        # ★ 为什么不能只按分数卡（实测数据）
        # ---------------------------------
        # 原实现统一用 `score >= _MIN_LAW_SCORE`(0.60) 过滤。
        # 实测发现这会把**正确条文也挡掉**：
        #
        #     「指定品牌」规则的候选依据              分数     topics 含「指定品牌」？
        #     《招标投标法》第二十条                  0.495   ✓（原文正是"招标文件不得
        #                                                     要求或者标明特定的生产供应者"）
        #     《招标投标法实施条例》第三十二条         0.439   ✓
        #     《政府采购法实施条例》第二十条           0.425   ✓
        #
        # 三条的 `topics` 里都明确含 `指定品牌`——**人工标注的主题标签是比
        # 余弦相似度更强的相关性信号**，却被分数门槛一票否决。
        #
        # 实测分数分布（40 条候选）：最高 0.774、中位 0.533，**只有 30% 能过 0.60**。
        # 即门槛挡掉了七成候选，其中包含大量真正对应的条文。
        #
        # 故改为**双通道**：
        #     通道一（主）  该条文的 topics 与规则的 topics 有交集 → 放行
        #     通道二（补）  分数 ≥ `_MIN_LAW_SCORE` → 放行
        #
        # 这样既救回主题精确对应的条文，又保留"分数很高但主题标签没覆盖"的兜底。
        # 原注释担心的"低分误导"仍被挡住：**主题不匹配且分数低**的依据依旧不显示。
        rule_topics = {str(t).strip() for t in (rule.topics or ())}

        def _relevant(hit: dict) -> bool:
            hit_topics = {str(t).strip() for t in (hit.get("topics") or ())}
            if rule_topics & hit_topics:
                return True
            score = hit.get("score")
            return isinstance(score, (int, float)) and score >= _MIN_LAW_SCORE

        kept = [h for h in candidates if _relevant(h)]
        kept.sort(key=lambda h: float(h.get("score") or 0), reverse=True)
        kept = kept[:3]
        if candidates and not kept:
            logger.info(
                "规则 %s 的法规检索最高分 %.3f 低于门槛 %.2f，不附依据（避免误导）",
                rule.key,
                max(float(h.get("score") or 0) for h in candidates),
                _MIN_LAW_SCORE,
            )

        return [
            {
                "law": h.get("law"),
                "article": h.get("article"),
                "excerpt": str(h.get("text") or "")[:220],
                "score": h.get("score"),
            }
            for h in kept
        ]
    except Exception as exc:
        if not _LAW_BASIS_WARNED:
            logger.info(
                "法规检索不可用（规则检出不受影响，后续同类失败不再重复记日志）：%s", exc
            )
            _LAW_BASIS_WARNED = True
        return []


# 法规依据的最低余弦相似度门槛。
#
# 实测取值依据：逐条核对全部规则的检索结果，
#   - 相关条文的分数落在 0.62 ~ 0.71（如保证金规则 top-1 = 0.680）
#   - 不相关条文的分数落在 0.52 ~ 0.56（如付款条件规则的三条）
# 0.60 落在两簇之间，且离两簇都有余量。
#
# ⚠️ 这是**保守**选择：宁可漏掉一条勉强相关的依据，也不要给出会误导用户的依据。
# 该门槛基于当前 180 条语料校准，语料显著扩充后应重新校准。
_MIN_LAW_SCORE = 0.60


_LAW_BASIS_WARNED = False


def check_compliance(
    parsed: dict[str, Any],
    *,
    include_law_basis: bool = True,
) -> dict[str, Any]:
    """对招标文件抽取结果做合规检查。

    返回：
        findings    命中的疑似问题（按严重度排序），每条含原文、为什么可疑、
                    怎么核实、法规依据
        passed      未命中任何规则的条款数（用来说明"检查过多少条"）
        summary     分档统计
        disclaimer  **免责说明**（必读）
    """
    if not isinstance(parsed, dict):
        raise ComplianceError("parsed 必须是字典（招标文件抽取结果）")

    clauses = _iter_clauses(parsed)
    if not clauses:
        raise ComplianceError("没有可检查的内容（抽取结果为空）")

    findings: list[dict[str, Any]] = []
    matched_clause_idx: set[int] = set()

    for idx, (sec_key, text, item_idx) in enumerate(clauses):
        for rule in RULES:
            try:
                evidence = rule.detect(text)
            except Exception:  # pragma: no cover - 规则自身异常不应中断整体
                logger.warning("规则 %s 在条款上抛异常，已跳过", rule.key, exc_info=True)
                continue
            if not evidence:
                continue
            matched_clause_idx.add(idx)
            findings.append(
                {
                    "rule": rule.key,
                    "title": rule.title,
                    "severity": rule.severity,
                    "section": sec_key,
                    "item_index": item_idx,
                    "clause": text,
                    "evidence": evidence,
                    "why": rule.why,
                    "how_to_verify": rule.how_to_verify,
                    "topics": list(rule.topics),
                    "law_basis": _fetch_law_basis(rule, text) if include_law_basis else [],
                }
            )

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9), f["section"]))

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1

    # 结论：保守表述——用"疑似"而非"违规"
    if counts["high"]:
        conclusion = (
            f"检出 {counts['high']} 项**高风险**疑似条款，"
            f"建议在投标前向采购人或代理机构提出书面质疑/澄清"
        )
    elif counts["medium"]:
        conclusion = (
            f"检出 {counts['medium']} 项需核实的疑似条款，"
            "建议对照原文与法规依据逐项确认后再决定是否投标"
        )
    elif counts["low"]:
        conclusion = "仅检出少量表述不规范项，风险较低"
    else:
        conclusion = "未检出规则库覆盖范围内的疑似违规条款"

    return {
        "findings": findings,
        "summary": {
            "checked_clauses": len(clauses),
            "flagged_clauses": len(matched_clause_idx),
            "counts": counts,
            "total": len(findings),
            "by_rule": by_rule,
            "rules_total": len(RULES),
            "conclusion": conclusion,
        },
        "disclaimer": (
            "本检查基于**规则匹配**，只覆盖规则库已收录的违规特征，"
            "**未命中的条款不代表合规**。"
            "同一表述在不同项目背景下的合法性可能不同，"
            "故结论均为「**疑似**」而非违规认定。"
            "本结果**不构成法律意见**，正式质疑或投诉前请核对法规原文，"
            "必要时咨询专业律师或向财政部门/监管部门咨询。"
        ),
    }


def rules_catalog() -> dict[str, Any]:
    """返回规则清单，供前端展示"我们检查哪些方面"。"""
    return {
        "rules": [
            {
                "key": r.key,
                "title": r.title,
                "severity": r.severity,
                "why": r.why,
                "how_to_verify": r.how_to_verify,
                "topics": list(r.topics),
            }
            for r in RULES
        ],
        "count": len(RULES),
        "severity_levels": {
            "high": "实践中常被认定为违规，直接影响竞争",
            "medium": "存在争议，需结合项目背景判断",
            "low": "表述不规范，建议要求澄清",
        },
    }


__all__ = [
    "RULES",
    "SEVERITY_ORDER",
    "ComplianceError",
    "ComplianceRule",
    "check_compliance",
    "rules_catalog",
]
