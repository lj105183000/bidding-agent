"""答案核对模块单测（零延迟规则验证，不调 LLM）。

为什么单独测它
-------------
1. **误报会阻塞一次正常回答**，代价高于漏报——故大量用例是"不该报"的负例
2. 判据涉及中文子串陷阱（`"可用"` 是 `"不可用"` 的子串）与跨句窗口，
   这些坑都实际踩过，必须钉死
3. 它要进 done 帧与生成前上下文，形状变了前端与 agent 都会受影响
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import verification as V  # noqa: E402


# ===========================================================================
# 工具可用性核对
# ===========================================================================


class TestToolAvailabilityClaims:
    """实测缺陷：Agent 会编造"工具不可用"，而工具其实成功返回了真实数据。"""

    def test_detects_real_contradiction(self):
        """★ 原始缺陷：回答说联网不可用，而它成功调用过 4 次。"""
        ans = "联网检索工具本轮不可用，以下项目来自招投标知识库与业务数据库。"
        got = V.check_tool_availability_claims(ans, ["search_web_cn", "search_knowledge_base"])
        assert any("联网" in g for g in got)

    def test_does_not_flag_the_tool_it_says_is_working(self):
        """★ 误报防护：同句里"来自知识库"不该被判成"知识库不可用"。

        实测踩过：固定字符窗口把"来自知识库"也圈进了不可用的邻域。
        """
        ans = "联网检索工具本轮不可用，以下项目来自招投标知识库与业务数据库。"
        got = V.check_tool_availability_claims(ans, ["search_knowledge_base"])
        assert got == [], f"知识库明明被说成可用，不该报：{got}"

    def test_no_claim_means_no_report(self):
        assert V.check_tool_availability_claims(
            "根据知识库与业务数据库的结果，投标保证金为 2%。", ["search_knowledge_base"]
        ) == []

    def test_missing_information_is_not_unavailability(self):
        """★ 「未检索到该信息」≠「工具不可用」。

        前者是诚实说明，后者是对系统状态的错误陈述。混淆会导致大量误报。
        """
        assert V.check_tool_availability_claims(
            "联网搜索已成功返回结果，本次未检索到该企业的记录。", ["search_web_cn"]
        ) == []
        assert V.check_tool_availability_claims(
            "知识库中未检索到该条款，联网搜索也未找到相关内容。",
            ["search_knowledge_base", "search_web_cn"],
        ) == []

    def test_tool_not_called_is_not_a_contradiction(self):
        """没执行过的工具，说它"不可用"是合理的。"""
        assert V.check_tool_availability_claims(
            "联网搜索工具本轮不可用。", ["search_knowledge_base"]
        ) == []

    def test_contrast_sentence_flags_the_right_tool(self):
        """「知识库可用，联网搜索不可用」——只该报联网。

        别名既认全称（联网搜索 / 联网检索）也认泛称（联网工具）——
        泛称此前漏报，已补（见 `test_alias_short_form_is_now_detected`）。
        """
        got = V.check_tool_availability_claims(
            "知识库可用，联网搜索不可用。", ["search_knowledge_base", "search_web_cn"]
        )
        assert len(got) == 1, f"应只报联网：{got}"
        assert "联网" in got[0]

    def test_alias_short_form_is_now_detected(self):
        """泛称「联网工具」现在**能**检测到（原来是已知漏报，已修）。

        ★ 为什么原来漏、为什么现在敢补
        -------------------------
        原缺口：本模块 docstring **自己举的例子**就是
        「回答写「联网检索工具本轮不可用」」，而别名表里只有
        `联网搜索 / 联网检索 / 网络搜索 / Tavily` ——
        于是「本次**联网工具**不可用」（不带"检索/搜索"二字）检测不到。

        原来不敢补的理由是"会扩大误报面"。本轮**实测了这个担心**：

            web 工具**未执行** → 说"联网工具不可用" → **不报**（合理）
            只成功知识图谱     → 说"联网工具不可用" → **不报**（联网确实没跑）
            说"联网工具可用"   → **不报**（不是否定式）

        即误报面由 `succeeded_tools` 把关，**与是否扩别名无关**。
        加上之后：漏报 2 → 0，误报 0。
        """
        got = V.check_tool_availability_claims("联网工具不可用。", ["search_web_cn"])
        assert got, "泛称「联网工具」应被检出（联网实际成功却说不可用）"

    def test_alias_short_form_not_flagged_when_web_did_not_run(self):
        """★ 上一条的**误报面**对照：联网没跑时说"不可用"是合理的，不能报。"""
        assert V.check_tool_availability_claims(
            "联网工具不可用。", ["search_knowledge_base"]
        ) == []

    def test_alias_full_forms_are_detected(self):
        """正式别名（联网搜索 / 联网检索）都能检测到。"""
        for phrase in ("联网搜索不可用。", "联网检索工具本轮不可用。"):
            assert V.check_tool_availability_claims(phrase, ["search_web_cn"]), phrase

    def test_positive_word_is_not_matched_inside_negative_word(self):
        """★ 回归哨兵：`"可用"` 是 `"不可用"` 的子串。

        上例曾因这个子串关系漏报：检查"知识库"时在后文匹配到"不可用"内部的
        "可用"，把它当成本句的可用性判断。
        """
        assert V.check_tool_availability_claims(
            "知识库不可用。", ["search_knowledge_base"]
        ) != []

    @pytest.mark.parametrize(
        "sentence",
        [
            "知识库不可用，无法检索。",
            "未能调用联网检索工具，以下结论仅基于知识库。",
            "本次联网搜索不可用，未能获取实时公告。",
            "知识库不可用，联网也不可用。",
        ],
    )
    def test_detects_various_phrasings(self, sentence):
        got = V.check_tool_availability_claims(
            sentence, ["search_knowledge_base", "search_web_cn"]
        )
        assert got, f"应报出矛盾：{sentence}"

    def test_is_flagged_when_negation_and_alias_share_a_clause(self):
        """★ 更正：这一句**会报**，而且一直都会报 —— 「但」从未起抑制作用。

        实测分句：

            本次联网检索不可用，但知识库提供了依据。
            → ['本次联网检索不可用']   ← 否定词"不可用" + 别名"联网检索"**同句**
            → ['但知识库提供了依据']   ← 无关

        判据是「逐句 · 别名 + 否定词」，故它必然命中。

        ★ 这条测试原来叫 `..._is_known_miss` 并断言 `got == []`，
        但那个"漏报"的唯一原因是它用了**简称**「联网工具」（别名没覆盖）。
        简称已补，所以它现在报出来 —— 这是**修好了**，不是判据变宽。

        同时更正原 docstring 的说法：「但」在这里不需要被"理解"，
        因为真正决定命中与否的是**分句边界**，不是转折词。
        """
        got = V.check_tool_availability_claims(
            "本次联网检索不可用，但知识库提供了依据。",
            ["search_web_cn", "search_knowledge_base"],
        )
        assert got, "否定词与别名同句 → 应命中（与「但」无关）"

    def test_negation_in_a_later_clause_is_not_linked_to_alias(self):
        """★ 这才是**真正不处理**的情形：否定词与别名**落在不同分句**。

            「联网检索没有返回结果，但这不是工具不可用。」

        前半句有别名、后半句有否定词，中间隔着逗号。要判定"后半句的
        「不可用」是在否定前半句的联网工具"需要理解指代与转折，
        规则做不到 → **不报**。

        取舍：漏报只少一条告警；误报会阻塞正常回答。故偏向漏报。
        """
        got = V.check_tool_availability_claims(
            "联网检索没有返回结果，但这不是工具不可用。",
            ["search_web_cn", "search_knowledge_base"],
        )
        assert got == [], (
            "若这条开始报出，说明开始跨分句关联否定词与别名了 —— "
            "请重新评估误报风险（这是刻意不做的）"
        )

    def test_concessive_but_short_form_is_still_detected(self):
        """★ 但**泛称**那句现在会报 —— 不是因为处理了「但」，而是因为
        「联网工具」这个别名补齐了。

        两件事必须分开钉：混在一起时，"修了别名"与"处理了语义"看起来一样。
        """
        got = V.check_tool_availability_claims(
            "本次联网工具不可用，但知识库提供了依据。",
            ["search_web_cn", "search_knowledge_base"],
        )
        assert got, "泛称下应报（别名生效），这与「但」的语义无关"

    def test_empty_inputs(self):
        assert V.check_tool_availability_claims("", ["search_web"]) == []
        assert V.check_tool_availability_claims("联网不可用", []) == []


# ===========================================================================
# 引用核对
# ===========================================================================


class TestCitationCheck:
    def test_flags_fabricated_law_name(self):
        """★ 招投标场景最致命的一类幻觉：编造法规名。

        ⚠️ 用**不在常识白名单里**的法规名。
        《招标投标法》这类骨架法律语料里没有（实测 top5 命中 0/5），
        但提及合理，已归入 `common_laws` 不计入未支持——
        详见 `TestCommonLawWhitelist`。
        """
        r = V.check_citations(
            "依据《某省招标投标条例》第二十六条的规定……",
            "投标保证金不得超过招标项目估算价的2%。",
        )
        assert "《某省招标投标条例》" in r.unsupported_laws
        assert "第二十六条" in r.unsupported_articles

    def test_empty_tool_text_flags_all_citations(self):
        """没有工具结果时，任何条款引用都无依据——这正是"凭参数记忆作答"的检测。"""
        r = V.check_citations("依据《某省采购条例》第七十一条……", "")
        assert r.unsupported_laws
        assert not r.ok

    def test_no_flag_when_present_in_tool_text(self):
        r = V.check_citations(
            "依据《招标投标法实施条例》第二十六条，不得超 2%。",
            "《招标投标法实施条例》第二十六条：投标保证金不得超过2%。",
        )
        assert r.ok, r.summary()

    def test_ignores_book_title_marks(self):
        """★ 回归哨兵：工具结果常**不带书名号**，答案会规范地加上。

        实测误报：5 项报告里 3 项是这类形态差异——
        「《招标投标法实施条例》」被判编造，而检索结果里就写着
        「招标投标法实施条例」（网页正文常不带书名号）。
        误报比漏报危险，会让使用者不再信任这个检查。
        """
        r = V.check_citations(
            "依据《招标投标法实施条例》的规定。",
            "招标投标法实施条例 第二十六条 投标保证金不得超过2%",
        )
        assert r.unsupported_laws == [], f"书名号差异不该算未支持：{r.unsupported_laws}"

    def test_ignores_document_number_bracket_style(self):
        """文号的全角/半角方括号形态差异同样不该算未支持。"""
        r = V.check_citations(
            "见发改办法规〔2020〕770号文。",
            "依据发改办法规[2020]770号规定……",
        )
        assert r.unsupported_articles == [], f"括号形态差异不该算未支持：{r.unsupported_articles}"

    def test_still_flags_truly_absent_document_number(self):
        """★ 归一化不能把真幻觉一起放过——这是上一条的对照。"""
        r = V.check_citations(
            "见发改办法规〔2099〕999号文。",
            "依据发改办法规[2020]770号规定……",
        )
        assert r.unsupported_articles, "真幻觉必须仍被报出"

    def test_tolerates_whitespace_differences(self):
        """答案常写成「第 二十六 条」，比对前必须去空白。"""
        r = V.check_citations(
            "见《招标投标法实施条例》第 二十六 条。",
            "《招标投标法实施条例》第二十六条",
        )
        assert r.ok, r.summary()

    def test_flags_unsupported_percent(self):
        r = V.check_citations("投标保证金不得超过 3%。", "投标保证金不得超过2%")
        assert "3%" in r.unsupported_numbers

    def test_flags_unsupported_money(self):
        r = V.check_citations("最高不超过 80 万元。", "履约保证金不超过中标合同金额的10%")
        assert any("80" in n for n in r.unsupported_numbers)

    def test_empty_answer_is_ok(self):
        assert V.check_citations("", "随便什么").ok


    def test_ok_property_and_summary(self):
        r = V.CheckResult()
        assert r.ok
        assert r.summary() == "核对通过"
        r2 = V.CheckResult(unsupported_laws=["《X法》"])
        assert not r2.ok
        assert "《X法》" in r2.summary()

    def test_to_dict_shape(self):
        d = V.check_citations("《X法》第一条", "").to_dict()
        for key in (
            "ok", "unsupported_laws", "unsupported_articles",
            "unsupported_numbers", "false_unavailable_claims", "summary",
        ):
            assert key in d, f"缺字段 {key}"


# ===========================================================================
# 组合入口与上下文注入
# ===========================================================================


class TestChineseNumeralEquivalence:
    """★ 第 7 类误报：答案写「10%」而工具结果写「百分之十」。

    实测取证（`logs/forensic_number_10pct.py`）：

        工具结果里的数字形态: 阿拉伯数字 = []      中文形态 = ['百分之十']
        工具结果里有 '10' 这个子串吗: False

    **模型的转换完全正确，是核对器只认阿拉伯数字形态。**
    这类题在旧实现下**必然误报**——实测 3 题全中，把"无严重编造比例"
    从 1.00 压到 0.79，指向了错误的优化方向。
    """

    def test_percent_chinese_vs_arabic(self):
        r = V.check_citations(
            "添购金额不超过原合同采购金额 10% 的。",
            "添购金额不超过原合同采购金额百分之十的。",
        )
        assert r.unsupported_numbers == [], f"等价形态不该算未支持：{r.unsupported_numbers}"

    def test_money_chinese_vs_arabic(self):
        """★ 「八十万元」的「万」属单位，不能切成数字「八十万」。"""
        assert V.check_citations(
            "最高不超过 80 万元。", "最高不超过八十万元"
        ).unsupported_numbers == []
        assert V.check_citations(
            "预算 2 亿元。", "预算两亿元"
        ).unsupported_numbers == []

    def test_whitespace_in_money_variant(self):
        """答案常写「80 万元」（带空格），工具结果写「80万元」。

        带空格的形态若不去空白，`'80 万元' in '80万元'` 为假——实测踩过。
        """
        assert V.check_citations(
            "最高不超过 80 万元。", "最高不超过80万元"
        ).unsupported_numbers == []

    @pytest.mark.parametrize(
        "answer,tool_text",
        [
            ("添购不超过 30%。", "添购不超过百分之十的。"),
            ("最高不超过 100 万元。", "最高不超过八十万元"),
            ("预算 3 亿元。", "预算两亿元"),
            ("占比 5%。", "占比百分之二"),
        ],
    )
    def test_real_fabrication_still_caught(self, answer, tool_text):
        """★ 归一化不能把真编造一起放过——这是上面几条的对照。"""
        assert V.check_citations(answer, tool_text).unsupported_numbers

    def test_cn_to_number_basics(self):
        assert V.cn_to_number("十") == 10
        assert V.cn_to_number("十五") == 15
        assert V.cn_to_number("二十六") == 26
        assert V.cn_to_number("一百二十") == 120
        assert V.cn_to_number("三") == 3
        assert V.cn_to_number("abc") is None
        assert V.cn_to_number("") is None

    def test_large_number_not_scientific_notation(self):
        """`{value:g}` 会把 200000000 输出成 `2e+08`，与答案的「2 亿元」匹配不上。"""
        out = V._haystack_variants(V._normalize("预算两亿元"))
        assert "2亿元" in out
        assert "e+08" not in out


class TestYuanUnitConversion:
    """★ 第 8 类误报：工具结果给原始「元」，答案换算成「万元/亿元」。

    实测取证（`logs/verify_tool_text_accumulation.py`）：

        工具结果（知识图谱）: 最高中标金额 14382650.79      ← 单位：元
        模型答案:            1438.27 万元                  ← 规范换算

    **换算完全正确**，但核对器只认原样字符串，于是 8 个金额全被判成"编造"。
    与「百分之十 ↔ 10%」是同一类问题：等价形态未归一化。
    """

    TOOL = "中移系统集成有限公司（最高中标金额 14382650.79）"
    BIG = "项目总投资 350000000 元"

    @pytest.mark.parametrize(
        "answer",
        [
            "中标金额 1438.27 万元",     # 两位小数（答案最常见写法）
            "中标金额 1438 万元",        # 取整
            "中标金额 1438.2651 万元",   # 四位小数
            "中标金额 0.14 亿元",        # 换算成亿元
        ],
    )
    def test_yuan_conversions_match(self, answer):
        assert V.check_citations(answer, self.TOOL).unsupported_numbers == []

    @pytest.mark.parametrize(
        "answer",
        ["项目投资 3.5 亿元", "项目投资 35000 万元"],
    )
    def test_billion_conversions_match(self, answer):
        assert V.check_citations(answer, self.BIG).unsupported_numbers == []

    @pytest.mark.parametrize(
        "answer,tool_text,expect",
        [
            ("中标金额 9999 万元", TOOL, "9999 万元"),      # 真编造
            ("项目投资 99 亿元", BIG, "99 亿元"),           # 真编造
            ("合同金额 8000 元", "合同金额 5000 元", "8000 元"),  # 小额真编造
        ],
    )
    def test_real_fabrication_still_caught(self, answer, tool_text, expect):
        """★ 归一化不能把真编造一起放过——这是上面几条的对照。"""
        assert expect in V.check_citations(answer, tool_text).unsupported_numbers

    def test_small_amount_passes_through_unchanged(self):
        """小额（<0.1 万元）不产出换算形态，避免 `0.05 万元` 这种噪音。"""
        assert V.check_citations("合同金额 5000 元", "合同金额 5000 元").unsupported_numbers == []

    def test_conversion_threshold_is_on_scaled_value(self):
        """★ 回归哨兵：门槛判的是**换算后的值**，不是原始值。

        两次试错都栽在这里：
            第一版 `value >= 10000` 才出两种单位 → 14382650.79 出不了「亿元」
            第二版 分别设 1万/1亿 门槛 → 它只有 0.14 亿，原值没到 1 亿，仍出不了
        正确判据：换算后 `>= 0.1` 即可。
        """
        out = V._yuan_conversions(V._normalize("金额 14382650.79"))
        assert "1438.27万元" in out, "应产出万元形态"
        assert "0.14亿元" in out, "应产出亿元形态（换算后 0.14 ≥ 0.1）"


class TestYuanConversionEdgeCases:
    """★ 实测工具结果的**真实格式**（`logs/forensic_competition_amounts.py`）。

    取证结论——工具返回三种形态，我的实现最初只处理了其中一种：
        `14382650.79`        —— 纯数字
        `14382650.79 元`     —— 数字 + 空格 + 元（**知识图谱的写法**）
        `2500.00` `1700.00`  —— 小额两位小数

    三个坑逐个钉住，它们都是"看起来能用但实际必然误报"的类型。
    """

    TOOL = "中移系统集成 14382650.79 元 小额 2500.00 1700.00 19750.00"

    @pytest.mark.parametrize(
        "answer",
        [
            "中标金额 1438.27 万元",   # 元 → 万元
            "中标金额 0.14 亿元",      # 元 → 亿元
            "中标金额 14382650.79 元",  # 原样（工具里的写法）
            "中标金额 0.25 万元",      # 小额 2500 → 0.25 万元
            "中标金额 0.17 万元",      # 小额 1700 → 0.17 万元
            "中标金额 1.98 万元",      # 19750 → 1.98 万元
        ],
    )
    def test_real_tool_formats_are_normalized(self, answer):
        assert V.check_citations(answer, self.TOOL).unsupported_numbers == []

    @pytest.mark.parametrize(
        "answer,expect",
        [
            ("中标金额 9999 万元", "9999 万元"),
            ("中标金额 0.99 万元", "0.99 万元"),
            ("中标金额 88 元", "88 元"),
        ],
    )
    def test_real_fabrication_still_caught(self, answer, expect):
        assert expect in V.check_citations(answer, self.TOOL).unsupported_numbers

    def test_digits_with_unit_suffix_are_converted(self):
        """★ 坑①：负向前瞻 `(?!元)` 曾把「数字 + 元」形态**排除**——
        而那正是知识图谱的写法，导致大额金额完全不做换算。"""
        out = V._yuan_conversions("最高中标金额 14382650.79 元")
        assert "1438.27万元" in out

    def test_small_amounts_are_converted(self):
        """★ 坑②：门槛设 5 位数字漏掉 2500/1700，
        而它们换算成「0.25 万元」正是模型会写的形态。"""
        out = V._yuan_conversions("金额 2500.00 1700.00")
        assert "0.25万元" in out
        assert "0.17万元" in out

    def test_conversion_runs_before_whitespace_stripping(self):
        """★ 坑③：`_normalize` 去空格会把相邻数字**粘成一个**。

        实测 `2500.00 1700.00 19750.00` → `2500.001700.0019750.00`，
        于是换算出的是无意义的大数。空格是数字的边界信息，不能先丢。
        """
        out = V._yuan_conversions("2500.00 1700.00 19750.00")
        assert "0.25万元" in out, "若这里为空说明空格被提前去掉、数字粘在一起了"
        # 粘成一个大数时会出现这种荒谬结果，必须不出现
        assert not any(x.startswith("2500001700") for x in out)

    def test_conversion_gets_raw_text_not_normalized(self):
        """★ 真正的修法：`_yuan_conversions` 吃**原始带空格文本**。

        ⚠️ 这里曾有一个**错误的归因**：我先以为问题是"拼接时缺分隔符"，
        并写了对照测试。实测发现加不加分隔符结果**完全一样**——
        因为 `_normalize` 会把换行也去掉。

        真正起作用的是**调用顺序**：若先把整段 `_normalize`（去空格），
        `2500.00 1700.00 19750.00` 就粘成 `2500.001700.0019750.00`，
        换算出来全是荒谬的大数。故必须先拿原文做换算。

        本测试直接钉住这个顺序依赖。
        """
        joined = "2500.00 1700.00 19750.00"
        raw_out = V._yuan_conversions(joined)
        assert "0.25万元" in raw_out, "原文换算应产出 0.25 万元"
        assert "0.17万元" in raw_out
        assert "1.98万元" in raw_out

        # 对照：先归一化（去空格）再换算 → 数字粘连成 `2500.001700.0019750.00`，
        # 只有最前面那个能算出有意义的值，**后面的全丢**。
        #
        # ⚠️ 断言要用 `0.17万元`，不能用 `0.25万元`：
        # 粘连后 `2500.00` 仍是完整的（它在最前面），`0.25万元` 反而凑巧出现。
        # 实测踩过——用错断言会让这个测试永远通不过，白查一轮。
        bad_out = V._yuan_conversions(V._normalize(joined))
        assert "0.17万元" not in bad_out, "该用例失去意义：去空格后居然还能正确换算"
        assert "1.98万元" not in bad_out

    @pytest.mark.parametrize(
        "answer",
        ["中标金额 0.25 万元", "中标金额 2500.00 元"],
    )
    def test_service_path_normalizes_common_forms(self, answer):
        """端到端确认 `check_citations` 走的是"先原文换算、再归一化"的路径。"""
        assert V.check_citations(answer, "小额 2500.00").unsupported_numbers == []

    def test_bare_amount_without_unit_is_matched(self):
        """★ 工具结果里的金额常常**不带单位**（裸数字 `2500.00`），
        而答案写成 `2500 元`。

        最初只比对带单位的形态，这类必然匹配不上。
        修法：`_number_variants` 额外产出**数字部分本身**（含去尾随零的写法）。
        """
        assert V.check_citations(
            "中标金额 2500 元", "小额 2500.00"
        ).unsupported_numbers == []
        assert V.check_citations(
            "中标金额 2500.00 元", "小额 2500.00"
        ).unsupported_numbers == []


class TestCommonLawWhitelist:
    """★ 常识性骨架法律单独归类，不计入「未支持法规」。

    实测依据（`logs/check_law_in_corpus.py`）：用「招标投标法」检索知识库，
    **top5 里含该法规名的条数 = 0/5**——这两部骨架法律根本不在语料里，
    模型是凭参数知识提及的。

    于是它们反复出现在 `unsupported_laws` 里，把"答案核对通过率"压到 0.71，
    但这个信号**不可执行**：要求模型不提及本领域最基本的法律并不合理。
    真正该守的是"别编具体条款号与数字"。故归入 `common_laws`。

    ⚠️ 白名单必须克制——放宽会让"编造法规名"这个检查失效，
    故下面同时钉住"非白名单法规名仍要报"。
    """

    @pytest.mark.parametrize(
        "law",
        ["《招标投标法》", "《招标投标法实施条例》", "《政府采购法》",
         "《中华人民共和国招标投标法》"],
    )
    def test_common_laws_are_not_unsupported(self, law):
        r = V.check_citations(f"依据{law}相关规定。", "无关内容")
        assert law in r.common_laws, f"{law} 应归入常识法规"
        assert law not in r.unsupported_laws
        assert r.ok, "只提常识法规名时不该判未通过"

    @pytest.mark.parametrize(
        "law",
        ["《某省招标投标条例》", "《XX市采购管理办法》", "《不存在的法》"],
    )
    def test_non_common_laws_still_flagged(self, law):
        """★ 回归哨兵：白名单不能宽到让编造法规名漏网。"""
        r = V.check_citations(f"依据{law}相关规定。", "无关内容")
        assert law in r.unsupported_laws
        assert law not in r.common_laws
        assert not r.ok

    def test_common_law_recorded_not_dropped(self):
        """不计入未支持 ≠ 丢弃。仍要记录，供「语料补充清单」参考。"""
        r = V.check_citations("依据《招标投标法》相关规定。", "无关内容")
        assert r.common_laws == ["《招标投标法》"]
        assert "常识法规" in r.summary()
        assert r.to_dict()["common_laws"] == ["《招标投标法》"]

    def test_common_law_does_not_mask_number_fabrication(self):
        """★ 关键：白名单只放宽法规名，**不能**顺带放过数字编造。"""
        r = V.check_citations("依据《招标投标法》，保证金为 10%。", "无关内容")
        assert r.ok is False
        assert r.critical_ok is False
        assert r.unsupported_numbers == ["10%"]

    def test_law_in_tool_text_is_not_reported_at_all(self):
        """工具结果里有该法规名时，两个列表都不该出现它。"""
        r = V.check_citations(
            "依据《招标投标法》相关规定。", "《招标投标法》第二条 本法的适用范围"
        )
        assert r.common_laws == []
        assert r.unsupported_laws == []
        assert r.ok


class TestSeverityLevels:
    """★ 严重性分层：数字/条款号编造 vs 法规名未在结果中。

    实测发现两类危害完全不同，混算成一根指标会掩盖真实风险：
        数字/条款号编造 → 高（用户据此核对合规性）
        法规名未在结果中 → 低（《招标投标法》这类常识名，写出来更完整）
    """

    def test_law_name_alone_is_not_critical(self):
        """非白名单的法规名仍报未支持，但**不算严重**（严重只看数字与条款号）。"""
        r = V.check_citations("依据《某省招标投标条例》相关规定。", "无关内容")
        assert not r.ok, "仍应报未支持"
        assert r.critical_ok, "但法规名属低风险，不算严重"

    @pytest.mark.parametrize(
        "answer",
        [
            "保证金为 10%。",          # 数字
            "见第四十五条。",           # 条款号
        ],
    )
    def test_number_and_article_are_critical(self, answer):
        r = V.check_citations(answer, "无关内容")
        assert not r.critical_ok

    def test_false_unavailable_is_critical(self):
        r = V.verify_answer("联网搜索不可用。", "", succeeded_tools=["search_web_cn"])
        assert not r.critical_ok

    def test_clean_answer_is_both_ok(self):
        r = V.check_citations("投标保证金不得超过 2%。", "投标保证金不得超过2%")
        assert r.ok and r.critical_ok

    def test_to_dict_exposes_critical_ok(self):
        d = V.check_citations("《X法》第一条", "").to_dict()
        assert "critical_ok" in d
        assert d["critical_ok"] is False
        d2 = V.check_citations("《X法》", "").to_dict()
        assert d2["critical_ok"] is True


class TestVerifyAnswer:
    def test_combines_both_checks(self):
        """两个检查都生效（用正式别名，见上面关于简称的说明）。"""
        r = V.verify_answer(
            "联网搜索不可用。依据《X法》第一条。",
            "",
            succeeded_tools=["search_web_cn"],
        )
        assert r.false_unavailable_claims, r.summary()
        assert r.unsupported_laws
        assert not r.ok

    def test_passes_on_grounded_answer(self):
        r = V.verify_answer(
            "投标保证金不得超过 2%。",
            "投标保证金不得超过2%",
            succeeded_tools=["search_knowledge_base"],
        )
        assert r.ok, r.summary()


class TestGroundingNote:
    def test_lists_succeeded_tools_with_labels(self):
        note = V.build_grounding_note(["search_web_cn", "search_knowledge_base"])
        assert "search_web_cn" in note
        assert "联网搜索" in note
        assert "search_knowledge_base" in note
        assert "知识库" in note

    def test_instructs_not_to_claim_unavailable(self):
        note = V.build_grounding_note(["search_web_cn"])
        assert "不要声称" in note
        assert "未检索到" in note

    def test_empty_when_no_tools(self):
        assert V.build_grounding_note([]) == ""
        assert V.build_grounding_note(None) == ""
