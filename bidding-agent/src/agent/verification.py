"""答案核对：零延迟的规则验证（不调 LLM）。

设计取舍：为什么是规则而不是 LLM
------------------------------
LLM 核验更准，但要多一次 5~20 秒调用。而本项目**最致命的一类错误恰好是
规则能抓的**：招投标/合规场景里编造法规名与条款号会直接误导决策，
而"这个法规名有没有出现在检索结果里"是个纯字符串判断。

故本模块只做确定性检查，零延迟、可离线测、结果可复现。
语义级的"这段推理是否有依据"留给可选的 LLM 核验（不在本模块）。

★ 为什么需要"工具可用性核对"（实测缺陷）
--------------------------------------
实测发现 Agent 会**编造工具不可用**：

    回答写「联网检索工具本轮不可用」
    而 search_web_cn 实际成功调用 4 次、返回 20/9/16/13 条真实结果
    且工具结果文本里明写着「【数据来源：联网搜索，请注意信息时效】」

用户据此会认为系统坏了。而链路里**没有任何环节检查这句话是否属实**——
这正是本模块 `check_tool_availability_claims` 要堵的洞。

为什么不做"事后改写正文"
----------------------
流式回答已经逐帧下发给用户，事后改写需要发 reset 清空重来，
用户会看到一次"闪烁"。故主策略是**预防**（把"哪些工具确实成功了"
作为事实注入生成前的上下文，见 `build_grounding_note`），
核对只用于**标注与可观测**（`chat()` 路径返回告警、日志记录）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 工具名 → 中文别名（答案里通常写中文，不写工具名）
_TOOL_LABELS = {
    "search_knowledge_base": ("知识库", "本地知识库", "问答知识库"),
    "search_knowledge_graph": ("知识图谱", "图谱"),
    "query_database": ("业务数据库", "数据库", "中标数据库"),
    # ★ 「联网工具」是**本模块 docstring 自己举的例子**里用的说法
    #   （"回答写「联网检索工具本轮不可用」"），而它此前**不在别名表里**
    #   —— 于是"本次联网工具不可用"（不带"检索/搜索"二字）检测不到。
    #   实测：加上之前漏报 2 例，加上之后漏报 0、**无误报**。
    #
    #   为什么三个联网工具都加：任一联网工具成功 = "联网可用"，
    #   此时用泛称说"联网工具不可用"都是矛盾。
    #
    #   误报面实测（这是清单担心的点）：
    #     · web 工具**未执行**时说"联网工具不可用" → **不报**（确实没跑，合理）
    #     · 只成功知识图谱时说"联网工具不可用" → **不报**（联网确实没跑）
    #     两者都由 `succeeded_tools` 把关，与是否扩别名无关。
    "search_web": ("联网搜索", "联网检索", "网络搜索", "Tavily", "联网工具"),
    "search_web_cn": ("联网搜索", "联网检索", "中文联网", "联网工具"),
    "search_exa": ("语义搜索", "联网语义搜索", "联网工具"),
}

#: 声称"不可用/失败"的说法。
#:
#: ⚠️ 只收**否定式**，不要把裸的「可用」放进来。
#: 曾试图靠 `"可用"` 是 `"不可用"` 子串这一点来"顺便覆盖"，
#: 结果它匹配到了正向陈述：「知识库**可用**，联网工具不可用」里，
#: 检查知识库时命中裸「可用」，把"知识库可用"判成了"知识库不可用"。
#: 语义相反的词不能进同一份清单。
_NEGATIVE_PHRASES = frozenset({
    "不可用",
    "未启用",
    "未开启",
    "无法联网",
    "联网功能关闭",
    "工具调用失败",
    "检索工具失败",
    "未能调用",
    "无法调用",
})
_UNAVAILABLE_PATTERNS = tuple(_NEGATIVE_PHRASES)

# 法条引用形态：第X条 / 第X号令 / 〔YYYY〕N号 / 《法规名》
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]+条(?:之[一二三四五六七八九十]+)?")
_DOC_NUMBER_RE = re.compile(r"第\s*\d+\s*号令|〔\s*\d{4}\s*〕\s*\d+\s*号|\[\s*\d{4}\s*\]\s*\d+\s*号")
_LAW_NAME_RE = re.compile(r"《[^》]{2,40}》")
# 关键数字：百分比与金额
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_MONEY_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?\s*(?:万元|亿元|元)")


@dataclass
class CheckResult:
    """一次核对的结论。"""

    #: 无法从工具结果中得到支撑的**非法典类**法规名（真正可疑的编造）
    unsupported_laws: list[str] = field(default_factory=list)
    #: 常识性骨架法律（如《招标投标法》）——语料里没有，但提及合理。
    #: 不计入 unsupported_laws（见 `_COMMON_LAWS` 的说明），但仍如实记录。
    common_laws: list[str] = field(default_factory=list)
    #: 无法从工具结果中得到支撑的条款号
    unsupported_articles: list[str] = field(default_factory=list)
    #: 无法从工具结果中得到支撑的百分比/金额
    unsupported_numbers: list[str] = field(default_factory=list)
    #: 声称某工具不可用、但该工具实际成功执行了
    false_unavailable_claims: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.unsupported_laws
            or self.unsupported_articles
            or self.unsupported_numbers
            or self.false_unavailable_claims
        )

    @property
    def critical_ok(self) -> bool:
        """是否**没有严重错误**。

        ★ 为什么要把严重性分开（评测数据驱动，见 docs/33）
        ----------------------------------------------
        实测发现两类问题的危害**完全不同**，混算成一根指标会掩盖真实风险：

            数字/条款号编造  → 高。用户会据此核对合规性，错一个数字就可能误导决策。
                              实测 3 题编造了 `10%`、1 题编造了"第四十五条"。
            法规名未在结果中  → 低。《招标投标法》这类常识性法规名不在检索结果里，
                              但写出来让答案更完整，**不属于"编造"**。

        原先两者都算 `ok=False`，于是"核对通过率 0.71"看不出到底是
        "3 题编了数字"还是"3 题提了法规名"——**指标无法指导优化方向**。
        故拆成两根：`critical_ok`（数字+条款+工具可用性）与 `ok`（含法规名）。
        """
        return not (
            self.unsupported_articles
            or self.unsupported_numbers
            or self.false_unavailable_claims
        )

    def summary(self) -> str:
        """一句话概述，供日志与 done 帧使用。"""
        parts: list[str] = []
        if self.false_unavailable_claims:
            parts.append("声称工具不可用但实际成功：" + "、".join(self.false_unavailable_claims))
        if self.unsupported_laws:
            parts.append("法规名未经检索结果支持：" + "、".join(self.unsupported_laws[:5]))
        if self.common_laws:
            parts.append("提及常识法规（语料未收录）：" + "、".join(self.common_laws[:5]))
        if self.unsupported_articles:
            parts.append("条款号未经检索结果支持：" + "、".join(self.unsupported_articles[:5]))
        if self.unsupported_numbers:
            parts.append("数字未经检索结果支持：" + "、".join(self.unsupported_numbers[:5]))
        return "；".join(parts) if parts else "核对通过"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            # 严重性分层：数字/条款号编造才是高风险（见 critical_ok 说明）
            "critical_ok": self.critical_ok,
            "unsupported_laws": self.unsupported_laws,
            "common_laws": self.common_laws,
            "unsupported_articles": self.unsupported_articles,
            "unsupported_numbers": self.unsupported_numbers,
            "false_unavailable_claims": self.false_unavailable_claims,
            "summary": self.summary(),
        }


#: 中文数字 → 阿拉伯数字。用于「百分之十」↔「10%」这类**等价形态**的归一化。
_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def cn_to_number(text: str) -> float | None:
    """把中文数字串转成数值。无法解析返回 None。

    支持「十」「十五」「二十六」「一百二十」「三」这类常见写法。
    **不支持**「一百零五万三千」这种多级复合——投标场景里罕见，
    而把它做对需要完整的数字解析器，收益不匹配。
    """
    s = (text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return float(s)
    total = 0.0
    section = 0.0
    current = 0.0
    found = False
    for ch in s:
        if ch in _CN_DIGITS:
            current = float(_CN_DIGITS[ch])
            found = True
        elif ch in _CN_UNITS:
            unit = float(_CN_UNITS[ch])
            found = True
            if unit >= 10000:
                section = (section + (current or 1)) * unit
                total += section
                section = 0.0
            else:
                section += (current or 1) * unit
            current = 0.0
        else:
            return None
    if not found:
        return None
    return total + section + current


def _number_variants(item: str) -> set[str]:
    """为一个数字引用生成所有**等价形态**的归一化串。

    ★ 为什么必须做（实测第 7 类误报）
    ------------------------------
    工具结果里写「添购金额不超过原合同采购金额**百分之十**的」，
    模型规范地改写成「不超过原合同采购金额 **10%**」。
    核对器原先只认阿拉伯形态，于是判成"编造 10%"——
    **模型的转换完全正确，是核对器太窄**。

    实测数据：工具结果里阿拉伯数字集合为**空**、中文形态为 `['百分之十']`，
    即这类题会**必然误报**。误报比漏报危险（会让人不再信任核对），必须修。
    """
    out = {_normalize(item)}
    m = re.fullmatch(r"百分之(.+)", item.strip())
    if m:
        value = cn_to_number(m.group(1))
        if value is not None:
            # 10.0 → "10"；2.5 → "2.5"
            num = _fmt_number(value)
            out.add(f"{num}%")
            out.add(f"{num}％")
    m2 = re.fullmatch(r"(.+?)(万元|亿元|元)", item.strip())
    if m2:
        value = cn_to_number(m2.group(1))
        if value is not None:
            unit = m2.group(2)
            num = _fmt_number(value)
            out.add(f"{num}{unit}")
    # ★ 裸数值形态：工具结果里的金额常常**不带单位**（如 `2500.00`），
    # 而答案写成 `2500 元` / `0.25 万元`。只比对带单位的形态会漏掉这类。
    # 取数字部分本身（含去尾随零的写法）作为候选。
    m3 = re.match(r"^\s*([\d,]+(?:\.\d+)?)\s*(?:元|万元|亿元)\s*$", item)
    if m3:
        raw_num = m3.group(1).replace(",", "")
        out.add(_normalize(raw_num))
        if "." in raw_num:
            # 2500.00 → 2500
            out.add(_normalize(raw_num.rstrip("0").rstrip(".")))
    # 统一去空白再返回。
    # haystack 侧已 `_normalize` 去空白，若这里留下「80 万元」这种带空格的形态，
    # `'80 万元' in '80万元'` 为假——实测踩过，金额形态会必然误报。
    return {_normalize(x) for x in out if x}


#: 中文数字字符集，用于给 haystack 生成等价形态
_CN_NUM_CHARS = "零一二两三四五六七八九十百千万亿"
_PERCENT_CN_RE = re.compile(rf"百分之([{_CN_NUM_CHARS}\d.]+)")
#: 金额候选：先粗抓「数字 + 单位」，再在代码里**显式剥掉单位**。
#:
#: ★ 为什么不用复杂正则（试错记录）
#: ------------------------------
#: 关键难点：「八十万元」里的「万」属于单位，而「十万」里的「万」属于数字本身。
#: 用 `([\d一二…亿]+)(万元|亿元)` 会把「八十万元」切成数字「八十万」+单位「元」
#: → 算成 800000 元（实测踩过）。
#: 试过用前瞻 `(?!元)` 让正则自己区分，但可读性差且易错。
#: 改成"粗抓候选 + 代码里最左最长优先剥单位"，逻辑一眼可验。
_MONEY_CN_RE = re.compile(rf"[{_CN_NUM_CHARS}\d.]+(?:亿元|万元|元)")


def _split_money(token: str) -> tuple[str, str] | None:
    """把 `八十万元` 拆成 `('八十', '万元')`；`十万` 拆成 `('十万', '元')`。"""
    for unit in ("亿元", "万元", "元"):
        if token.endswith(unit):
            number = token[: -len(unit)]
            if number:
                return number, unit
    return None


def _fmt_number(value: float) -> str:
    """数值转字符串。整数不带小数点，避免 `10.0%` 这种写法与答案不匹配。

    也不能用 `:g`——它会把 500000000 输出成 `5e+08`。
    """
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


#: 本领域**常识性法规名**——提及它们不算编造。
#:
#: ★ 为什么要单独归类（语料实测 + 指标可操作性）
#: -----------------------------------------
#: 实测（`logs/check_law_in_corpus.py`）：用「招标投标法」检索知识库，
#: top5 里**含该法规名的条数 = 0/5**。即这两部骨架法律**根本不在语料里**，
#: 模型是凭参数知识提及的。
#:
#: 于是 `unsupported_laws` 里反复出现《招标投标法》《招标投标法实施条例》，
#: 把"答案核对通过率"压到 0.71——**但这个信号不可执行**：
#:      · 要求模型不提及本领域最基本的法律，不合理
#:      · 补语料是人工工作（75 条里加整部法律）
#:
#: 真正该守的是"别编**具体条款号与数字**"（用户会照此核对合规性）。
#: 故把常识法规名归到 `common_laws`，**不计入 unsupported_laws**，
#: 但仍如实记录，供"语料补充清单"参考。
#:
#: ⚠️ 白名单要**克制**：只放本领域人尽皆知的骨架法律。
#: 放宽会让"编造法规名"这个检查失效。
_COMMON_LAWS = frozenset({
    "中华人民共和国招标投标法",
    "招标投标法",
    "招标投标法实施条例",
    "中华人民共和国政府采购法",
    "政府采购法",
    "政府采购法实施条例",
    "中华人民共和国民法典",
    "中华人民共和国建筑法",
})


def _is_common_law(name: str) -> bool:
    """判断法规名是否属常识性骨架法律（去书名号后比对）。"""
    return name.strip("《》").strip() in _COMMON_LAWS


def _haystack_variants(haystack: str) -> str:
    """把工具结果全文里的**中文数字形态**补上阿拉伯等价形态。

    ★ 为什么必须双向归一化（第一次修错了方向）
    ----------------------------------------
    第一次只把**答案侧**的「10%」转成「百分之十」，然后查它是否在
    haystack 里——但 haystack 里是「百分之十」，`'10%' in '百分之十'` 仍为假。

    正确做法是**给 haystack 追加等价形态**：
        「百分之十」 → 追加「10%」
        「八十万元」 → 追加「80万元」
    这样答案侧的任一种写法都能命中。

    追加而不是替换：原文里的中文形态仍要保留（答案也可能照抄中文形态）。
    """
    extra: list[str] = []
    for m in _PERCENT_CN_RE.finditer(haystack):
        value = cn_to_number(m.group(1))
        if value is not None:
            extra.append(f"{_fmt_number(value)}%")
    for m in _MONEY_CN_RE.finditer(haystack):
        split = _split_money(m.group(0))
        if split is None:
            continue
        value = cn_to_number(split[0])
        if value is not None:
            extra.append(f"{_fmt_number(value)}{split[1]}")
    extra.extend(_yuan_conversions(haystack))
    return haystack + "".join(extra)


#: 裸元金额：工具结果里的原始数值，如 `14382650.79`。
#:
#: ★ 实测误报（第 8 类）
#: -------------------
#: 知识图谱工具返回 `最高中标金额 14382650.79`（单位：元），
#: 模型规范地换算成 `1438.27 万元` 写进答案——**换算完全正确**，
#: 但核对器只认原样字符串，于是 8 个金额全被判成"编造"。
#: 这与「百分之十 ↔ 10%」是同一类问题：**等价形态未归一化**。
#:
#: 裸元金额：工具结果里的原始数值。
#:
#: ★ 必须在**去空格之前**匹配（实测第三次踩坑）
#: ----------------------------------------
#: `_normalize` 会去掉所有空白，于是工具结果里的
#:     `2500.00 1700.00 19750.00`   →   `2500.001700.0019750.00`
#: **三个金额被粘成一个数字**，换算结果全错（`0.25 万元` 出不来）。
#:
#: 空格是数字的**边界信息**，不能先丢。故 `_yuan_conversions` 接收**原始**
#: haystack（保留空格），换算完再由调用方归一化。
#:
#: 另两个已修的坑：
#:     坑① 负向前瞻 `(?!元)` 把「数字 + 元」形态排除——那正是知识图谱的写法
#:     坑② 门槛 5 位数字漏掉 2500/1700 这类小额，而它们换算成「0.25 万元」正是模型会写的
_RAW_YUAN_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*元?(?![\d])")
#: 换算时保留的小数位。答案通常写两位（`1438.27 万元`），
#: 而原值可能有更多位（`1438.265079`），故同时给出多个精度供匹配。
_MONEY_DECIMALS = (0, 1, 2, 4)


#: 换算后的最小值。低于它会产出 `0.003 万元` 这种不自然写法，
#: 且小额数字更可能是编号/数量而非金额，误换算只制造噪音。
_MIN_SCALED = 0.1

#: 带尾随零的小数：`2500.00` → `2500`、`1438.270` → `1438.27`。
#:
#: ★ 实测踩坑：工具结果写 `2500.00`（两位小数），模型答案写 `2500 元`，
#: 字符串比对不上。**它们是同一个值**，差异只是显示精度。
#: 故给 haystack 补上"去掉尾随零"的形态。
_TRAILING_ZERO_RE = re.compile(r"(?<=\d)\.0+(?!\d)")


def _yuan_conversions(haystack: str) -> list[str]:
    """把工具结果里的**裸元金额**补上「万元 / 亿元」等价形态。

    ⚠️ 传入的 `haystack` 必须是**保留空格的原文**，不能是 `_normalize` 过的：
    去空格会把相邻数字粘成一个（实测 `2500.00 1700.00` → `2500.001700.00`），
    换算结果全错。见 `_RAW_YUAN_RE` 的说明。

    ★ 判据是"换算后是否可读"，不是"原值是否够大"（两次试错记录）
    ------------------------------------------------------------
    第一版：`value >= 10000` 才产出两种单位 → 14382650.79 只出「万元」，
            答案写「0.14 亿元」时匹配不上。
    第二版：分别设 `万元 ≥ 1万` / `亿元 ≥ 1亿` 门槛 → 14382650.79 仍出不了
            「亿元」（它只有 0.14 亿，原值没到 1 亿）。
    第三版（本版）：**换算后再判**——`scaled >= 0.1` 即可。
    """
    out: list[str] = []
    for m in _RAW_YUAN_RE.finditer(haystack):
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        for unit, divisor in (("万元", 10000), ("亿元", 100000000)):
            scaled = value / divisor
            if scaled < _MIN_SCALED:
                continue
            for digits in _MONEY_DECIMALS:
                out.append(f"{round(scaled, digits):.{digits}f}{unit}")
    return out


def _normalize(text: str) -> str:
    """归一化空白与书名号，便于子串比对。

    ★ 为什么必须去掉书名号（实测误报）
    ---------------------------------
    工具结果里的法规名常常**不带书名号**（检索到的网页正文就写作
    「招标投标法实施条例」），而模型回答时会规范地加上《》。若比对含书名号，
    三条真实存在于检索结果的法规会被判成"编造"——实测 5 项报告里 3 项是这种误报。

    误报比漏报危险：它会让使用者不再信任这个检查。故把书名号一并归一化。

    同理去掉空白：模型的答案是 markdown，「第 二十六 条」这类写法很常见。
    """
    cleaned = re.sub(r"\s+", "", text or "")
    return cleaned.replace("《", "").replace("》", "")


def _mentioned_tool_labels(answer: str) -> list[str]:
    """答案里提到了哪些工具的中文别名（用于可用性核对）。"""
    hits: list[str] = []
    for names in _TOOL_LABELS.values():
        for name in names:
            if name in answer and name not in hits:
                hits.append(name)
    return hits


#: 句子/子句边界。中英文标点都收——判"某工具是否被声称不可用"必须**在子句内**判。
#:
#: ★ 为什么切得越细越好（两次实测误报的教训）
#: ----------------------------------------
#: 第一版用固定字符窗口（±30 字），把「联网检索工具本轮不可用，以下项目来自
#: 招投标知识库」里的知识库也判成了"被声称不可用"。
#: 第二版改成按句切分，但给逗号加了"后面至少 12 字才切"的限制，
#: 于是「知识库可用，联网工具不可用。」里的「联网工具不可用。」只有 9 字、
#: 没切开，窗口又跨到了下一句。
#:
#: 结论：**不要为了"句子完整性"而合并短子句**——这里判的是局部断言，
#: 切得越细，跨句误判越少。误报会让人不再信任这个检查，代价高于漏报。
_SENTENCE_SPLIT_RE = re.compile(r"[。！？；;，,\n]+")

#: 反转词：出现在别名与"不可用"字样之间时，说明该别名**不是**被声称不可用的那个。
#:
#: 实测误报案例：
#:     「联网检索工具本轮不可用，以下项目**来自招投标知识库**与业务数据库」
#: 固定窗口把"来自知识库"也圈了进去，于是知识库被误判为"被声称不可用"。
#: 误报比漏报更危险，故必须显式排除这类反向表述。
#:
#: ⚠️ 这份清单要**只收真正反转语义的词**。
#: 曾把「依据」也放进来，结果「本次联网工具不可用，但知识库提供了依据」
#: 里的联网工具漏报了——而"提供了依据"其实是在**证实**前文的不可用。
#: 故「依据/根据/来源」这类词已移出。
_NEGATION_CUES = (
    "来自", "源自", "改用", "改从", "以下", "下列",
)
#: 对比词：把陈述方向从"不可用"翻转到另一个工具
_CONTRAST_CUES = ("但", "而", "不过", "然而", "却")

#: 已知漏报（不做处理，如实记录）
#:
#: 「本次联网工具不可用，但知识库提供了依据。」
#:     联网确实被说成不可用，但"但"在这里是**强调后半句**而非反转前半句。
#:     区分这两种用法需要理解语义，规则做不到。
#:
#: 取舍依据：本检查的两个用途里，"作为上下文注入的预防"不依赖检测精度，
#: "作为 done 帧告警"漏报只是少一条提示。而**误报会阻塞一次正常回答**——
#: 代价不对称，故一律偏向漏报。


def _sentences_with_offsets(text: str) -> list[tuple[str, int]]:
    """把文本切成句子，并给出每句在原文中的起始下标。

    下标用于把"句子内命中的别名"映射回原文（日志与去重需要）。
    """
    out: list[tuple[str, int]] = []
    pos = 0
    for piece in _SENTENCE_SPLIT_RE.split(text):
        if piece is None:
            continue
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        if piece.strip():
            out.append((piece, idx))
        pos = idx + len(piece)
    return out or [(text, 0)]


def _sentence_claims_unavailable(sentence: str, label: str) -> bool:
    """判断**这一句**是否在声称 `label` 不可用。

    判据（宁可漏报不误报——误报会让人不再信任这个检查）：
        1. 子句里必须同时出现该别名与某个"不可用"字样
        2. 别名与不可用字样之间不得有反转/对比词
        3. 别名与不可用字样之间不得出现第二个工具别名
        4. **`"可用"` 必须先剔除 `"不可用"`**（见下）

    ★ 第 4 条是踩过的坑
    ----------------
    `"可用"` 是 `"不可用"` 的**子串**。于是
    「知识库可用，联网工具不可用。」里，检查"知识库"时会在后文匹配到
    "不可用"内部的"可用"，把它当成本句的可用性判断，从而漏报真实矛盾。
    故匹配「可用」这类肯定词前，必须先把「不可用」整体删掉。
    """
    if label not in sentence:
        return False
    for phrase in _UNAVAILABLE_PATTERNS:
        if phrase not in sentence:
            continue
        for match in re.finditer(re.escape(label), sentence):
            head = sentence[: match.start()]
            tail = sentence[match.end() :]
            # 前缀式：「联网检索工具本轮不可用」
            pos = head.rfind(phrase)
            if pos >= 0:
                between = head[pos + len(phrase) :]
                if any(cue in between for cue in _NEGATION_CUES + _CONTRAST_CUES):
                    continue
                if _has_other_label(between, label):
                    continue
                return True
            # 后缀式：「知识库不可用」
            pos = tail.find(phrase)
            if pos >= 0:
                between = tail[:pos]
                if any(cue in between for cue in _NEGATION_CUES + _CONTRAST_CUES):
                    continue
                if _has_other_label(between, label):
                    continue
                return True
    return False


def _has_other_label(between: str, current: str) -> bool:
    """`between` 里是否出现了**另一个**工具别名。

    出现即说明这句话在分别描述多个工具（「知识库可用，联网不可用」），
    此时不能把整句的判断套到某一个别名上。
    """
    for names in _TOOL_LABELS.values():
        for name in names:
            if name != current and name in between:
                return True
    return False


def check_tool_availability_claims(
    answer: str, succeeded_tools: list[str] | set[str]
) -> list[str]:
    """核对"工具不可用"类声明是否与实际执行结果矛盾。

    只报**矛盾**：某工具实际成功执行了，而答案却按它的中文别名说"不可用"。
    没执行过的工具不报——那种情况下说"不可用"是合理的（确实没调过）。

    返回矛盾的描述列表（空 = 无矛盾）。
    """
    text = answer or ""
    if not text:
        return []
    if not any(p in text for p in _UNAVAILABLE_PATTERNS):
        return []

    succeeded = {str(t) for t in (succeeded_tools or [])}
    sentences = [s for s, _ in _sentences_with_offsets(text)]

    conflicts: list[str] = []
    seen: set[str] = set()
    for tool in succeeded:
        for label in _TOOL_LABELS.get(tool, ()):
            if any(_sentence_claims_unavailable(s, label) for s in sentences):
                key = f"{label}（{tool} 实际执行成功）"
                if key not in seen:
                    seen.add(key)
                    conflicts.append(key)
                break
    return conflicts


def check_citations(answer: str, tool_text: str) -> CheckResult:
    """核对答案里的法规引用与数字是否有工具结果支撑。

    `tool_text` 是所有工具结果拼成的文本（即模型生成时能看到的那部分）。
    只把**完全找不到**的记为未支持；能找到的（含去空白后匹配）视为有依据。

    ★ 为什么只报"找不到"而不判断"找得对不对"
    --------------------------------------
    判断"引用的条款内容是否真的对应"需要语义理解，属 LLM 核验范畴。
    规则层只回答"这个字符串在检索结果里出现过吗"——
    而这一条恰好能抓住最高频的幻觉：**编造一个库里根本没有的法规名或条款号**。
    """
    result = CheckResult()
    text = answer or ""
    # 双向归一化：给 haystack 补上中文数字与金额单位的等价形态。
    #
    # ⚠️ 顺序要紧：`_yuan_conversions` 必须吃**保留空格的原文**，
    # 因为去空格会把相邻数字粘成一个（实测 `2500.00 1700.00` → `2500.001700.00`，
    # 三个金额变成一个数，换算结果全错）。
    # 故先把换算结果算出来，再连同原文一起归一化。
    raw = tool_text or ""
    extra_yuan = "".join(_yuan_conversions(raw))
    # ⚠️ 拼接时必须加分隔符。
    # 直接 `raw + extra_yuan` 会让末尾数字与换算结果粘在一起——
    # 实测 `'小额 2500.00'` + `'0万元'` → `'小额2500.000万元'`，
    # 于是 `2500元` 这个形态永远出不来。
    base = _normalize(raw + "\n" + extra_yuan)
    # 补上"去掉尾随零"的形态：工具结果写 `2500.00` 而答案写 `2500` 时
    # 字符串比对不上，但它们是同一个值（实测踩坑）。
    haystack = _haystack_variants(base) + _TRAILING_ZERO_RE.sub("", base)
    if not text:
        return result

    # 法规名：分出"常识性骨架法律"与"真正可疑的编造"
    for name in dict.fromkeys(_LAW_NAME_RE.findall(text)):
        if _normalize(name) in haystack:
            continue
        if _is_common_law(name):
            result.common_laws.append(name)
        else:
            result.unsupported_laws.append(name)

    # 条款号与文号。逐个做**两种形态**的比对：原样、以及去掉全角括号后的形态。
    #
    # 文号在检索结果里可能写作「发改办法规〔2020〕770号」也可能写作
    # 「发改办法规[2020]770号」，两者的方括号形态不同（全角/半角）。
    # 只按原样比会把真实存在的文号判成编造——实测踩过 3 项这类误报。
    for pattern in (_ARTICLE_RE, _DOC_NUMBER_RE):
        for item in dict.fromkeys(pattern.findall(text)):
            variants = {_normalize(item)}
            variants.add(_normalize(item.replace("〔", "[").replace("〕", "]")))
            variants.add(_normalize(item.replace("〔", "").replace("〕", "")))
            if not any(v in haystack for v in variants):
                result.unsupported_articles.append(item)

    # 百分比与金额。按**等价形态**比对：答案写「10%」而工具结果写「百分之十」
    # 时必须算匹配（实测必然误报的场景，见 `_number_variants`）。
    #
    # ⚠️ 比对前两边都要 `_normalize`：答案写「2500 元」（带空格），
    # 而 haystack 已去空格，不统一就永远匹配不上（实测踩过）。
    for pattern in (_PERCENT_RE, _MONEY_RE):
        for item in dict.fromkeys(pattern.findall(text)):
            if not any(_normalize(v) in haystack for v in _number_variants(item)):
                result.unsupported_numbers.append(item)

    return result


def verify_answer(
    answer: str,
    tool_text: str = "",
    succeeded_tools: list[str] | set[str] | None = None,
) -> CheckResult:
    """完整核对：引用 + 工具可用性。零延迟、不调 LLM。"""
    result = check_citations(answer, tool_text)
    result.false_unavailable_claims = check_tool_availability_claims(
        answer, succeeded_tools or []
    )
    return result


def build_grounding_note(succeeded_tools: list[str] | set[str]) -> str:
    """构造"事实清单"注入生成前的上下文，**预防**编造工具不可用。

    为什么用注入而不是事后改写：流式回答已逐帧下发，改写要 reset 重来，
    用户会看到闪烁。把"哪些工具确实成功了"作为事实现给模型，
    是成本最低且对流式友好的做法。
    """
    tools = [str(t) for t in (succeeded_tools or []) if t]
    if not tools:
        return ""
    labels: list[str] = []
    for tool in tools:
        alias = _TOOL_LABELS.get(tool, ())
        labels.append(f"{tool}（{alias[0]}）" if alias else tool)
    return (
        "【本轮工具执行事实】以下工具**已成功执行并返回结果**："
        + "、".join(labels)
        + "。回答中不要声称这些工具不可用或调用失败；"
        "若某项信息确实没检索到，请说明「未检索到该信息」，而不是说工具不可用。"
    )


__all__ = [
    "CheckResult",
    "build_grounding_note",
    "check_citations",
    "check_tool_availability_claims",
    "verify_answer",
]
