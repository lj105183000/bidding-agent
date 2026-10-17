"""回答数据可视化的测试：图表提取器。

★ 本文件的核心是**负例**
==================
图表画错数字比不画图表有害得多 —— 用户会当真的看。
故这里对"什么算可图表数据"设了严格门槛，**绝大多数测试是"必须拒绝"**：

    五列明细表（项目|标的|单位|供应商|金额）→ 拒绝
        它不是"类别+数值"的排行语义，画成条形图会变成
        "按项目名的金额排行"，数字对但语义错。
    值列带单位（`23996.00 元`）→ 拒绝
    单行表 → 拒绝（画不出分布）
    正文里的编号列举 → 拒绝

真实工具输出的三种形态各有用例（都取自实测）：
    ① Markdown 表格   数据库排行类（top_subject / location_dist / …）
    ② 括号排名         知识图谱 top_traded（`1. 空调（交易频次 175）`）
    ③ 键值列表         图谱规模 / 数据库统计概览（`- 采购单位：3577`）
"""

from __future__ import annotations

import pytest

from src.agent.charts import MAX_BARS, MAX_CHARTS, extract_charts


def one(text: str, tool: str = "t") -> dict | None:
    charts = extract_charts(text, tool)
    return charts[0] if charts else None


# ===========================================================================
# 一、真实形态：必须提取到
# ===========================================================================


class TestRealShapes:
    """三种真实输出形态。数据取自实测（`logs/audit_db_shapes*.py`）。"""

    def test_markdown_table(self):
        """① 数据库排行类：标题 + 两列表 + 分隔行。"""
        text = (
            "业务数据库查询结果（共 6 条）：\n"
            "标的物 | 数量\n"
            "--------\n"
            "空调 | 175\n"
            "打印/复印纸的网上超市采购 | 160\n"
            "粉盒 | 150\n"
        )
        c = one(text)
        assert c is not None
        assert c["kind"] == "bar"
        assert c["title"] == "业务数据库查询结果（共 6 条）"
        assert c["label_label"] == "标的物"
        assert c["value_label"] == "数量"
        assert [d["label"] for d in c["data"]] == [
            "空调", "打印/复印纸的网上超市采购", "粉盒"]
        assert [d["value"] for d in c["data"]] == [175.0, 160.0, 150.0]

    def test_paren_rank_list(self):
        """② 知识图谱：`1. 空调（交易频次 175）`。

        ★ 这是与表格**不同**的第三种形态（由不同工具生成）。
        第一版只认表格与 `键：值`，实测漏掉了它。
        """
        text = (
            "采购频次最高的标的物（前 5 名）：\n"
            "1. 空调（交易频次 175）\n"
            "2. 打印/复印纸的网上超市采购（交易频次 160）\n"
            "3. 粉盒（交易频次 150）\n"
        )
        c = one(text)
        assert c is not None
        assert c["title"] == "采购频次最高的标的物（前 5 名）"
        assert [d["label"] for d in c["data"]] == ["空调", "打印/复印纸的网上超市采购", "粉盒"]
        assert [d["value"] for d in c["data"]] == [175.0, 160.0, 150.0]

    def test_kv_list(self):
        """③ 键值列表：图谱规模。"""
        text = (
            "知识图谱规模：\n"
            "- 代理机构：325\n"
            "- 地区：38\n"
            "- 采购单位：3577\n"
            "- 供应商：4138\n"
        )
        c = one(text)
        assert c is not None
        assert c["title"] == "知识图谱规模"
        assert [d["label"] for d in c["data"]] == ["代理机构", "地区", "采购单位", "供应商"]
        assert [d["value"] for d in c["data"]] == [325.0, 38.0, 3577.0, 4138.0]

    def test_db_stats_overview(self):
        """键值列表：数据库统计概览（含大数值）。"""
        text = (
            "业务数据库整体统计：\n"
            "- 记录总数：8746\n"
            "- 中标金额合计：21856705663.93\n"
            "- 标的物种类数：2529\n"
            "- 采购单位数：3584\n"
        )
        c = one(text)
        assert c is not None
        assert c["data"][1]["value"] == 21856705663.93

    def test_halfwidth_colon_in_kv(self):
        """`.env`/工具里半角冒号也出现过，两种都要认。"""
        c = one("统计：\n- 甲: 10\n- 乙: 20\n- 丙: 30\n")
        assert c is not None
        assert [d["value"] for d in c["data"]] == [10.0, 20.0, 30.0]

    def test_thousand_separator(self):
        c = one("标的物 | 数量\n------\n空调 | 1,234\n粉盒 | 987\n")
        assert c is not None
        assert c["data"][0]["value"] == 1234.0


# ===========================================================================
# 二、负例：必须拒绝（本文件最重要的部分）
# ===========================================================================


class TestMustReject:
    def test_five_column_detail_table(self):
        """★ 五列明细表必须拒绝。

        它的语义是"项目明细"，不是"类别+数值的排行"。
        画成条形图会变成"按项目名的金额排行"—— 数字对但**语义错**，
        比不画更有害。
        """
        text = (
            "业务数据库查询结果（共 3 条）：\n"
            "项目名称 | 标的物 | 采购单位 | 供应商 | 中标金额\n"
            "------------------------------\n"
            "某学校空调采购项目 | 空调 | 某学校 | 某公司 | 23996.00\n"
            "某财政局空调采购项目 | 空调 | 某财政局 | 另一公司 | 13850.00\n"
            "某合作社空调采购项目 | 空调配件 | 某合作社 | 第三公司 | 21000.00\n"
        )
        assert extract_charts(text) == []

    def test_value_with_unit(self):
        """值列含「元」→ 整表拒绝。

        ★ 为什么"整表拒绝"而不是"提取出数字就行"
        ------------------------------------
        放宽到"搜到第一串数字就算"的话，`2023-10`、`第二十七条第 2 款`
        都会被当成数值。既然无法可靠区分"数值"与"含数字的文本"，
        就在**整表**层面拒绝，而不是逐行猜。
        """
        text = (
            "业务数据库查询结果（共 3 条）：\n"
            "项目 | 中标金额\n"
            "--------\n"
            "甲项目 | 23996.00 元\n"
            "乙项目 | 13850.00 元\n"
            "丙项目 | 21000.00 元\n"
        )
        assert extract_charts(text) == []

    def test_single_row_table(self):
        """单行表画不出"分布"。"""
        text = "业务数据库查询结果（共 1 条）：\n标的物 | 数量\n--------\n空调 | 175\n"
        assert extract_charts(text) == []

    def test_two_kv_rows(self):
        """键值行少于 3 行 → 更像正文里的偶然短语。"""
        assert extract_charts("说明：\n- 甲：10\n- 乙：20\n") == []

    def test_two_rank_rows(self):
        assert extract_charts("排名：\n1. 空调（频次 175）\n2. 粉盒（频次 150）\n") == []

    def test_prose_numbered_list(self):
        """正文里的编号列举：数字在冒号后、不在括号里 → 不匹配。"""
        text = (
            "根据《政府采购法》第三十一条，可以采用单一来源方式采购的情形包括：\n"
            "- 只能从唯一供应商处采购的：1\n"
            "- 发生了不可预见的紧急情况的：2\n"
        )
        assert extract_charts(text) == []

    def test_plain_prose(self):
        assert extract_charts("投标保证金不得超过招标项目估算价的 2%。") == []

    def test_kb_search_result(self):
        """知识库检索结果是"问/答/相关度"，不是统计结构 → 无图。"""
        text = (
            "知识库检索到 3 条相关资料：（匹配度良好，可直接引用）\n"
            "[1] 问：投标保证金的有效期怎么规定？\n"
            "    答：《招标投标法实施条例》第二十六条…\n"
            "    相关度：0.68\n"
        )
        assert extract_charts(text) == []

    def test_empty_and_none(self):
        assert extract_charts("") == []
        assert extract_charts(None) == []  # type: ignore[arg-type]

    def test_malformed_never_raises(self):
        """提取器**永不抛异常**——图表是增益项，不能影响问答。"""
        for bad in ("|", "||\n||", "：\n：", "- ：", "a|b\n" + "-" * 100, "| |\n|---|\n| |"):
            extract_charts(bad)  # 不应抛


# ===========================================================================
# 三、边界与上限
# ===========================================================================


class TestLimits:
    def test_truncated_to_max_bars_with_note(self):
        """超过上限要截断，**并且说明**——不能让用户以为图里是全部。"""
        text = "标的物 | 数量\n--------\n" + "\n".join(
            f"标的{i} | {100 - i}" for i in range(1, 21)
        )
        c = one(text)
        assert c is not None
        assert len(c["data"]) == MAX_BARS
        assert c["note"], "截断时必须给出说明"
        assert str(MAX_BARS) in c["note"]

    def test_at_limit_has_no_note(self):
        """恰好等于上限时不截断，note 应为空（否则是假提示）。"""
        text = "标的物 | 数量\n--------\n" + "\n".join(
            f"标的{i} | {100 - i}" for i in range(1, MAX_BARS + 1)
        )
        c = one(text)
        assert c is not None
        assert len(c["data"]) == MAX_BARS
        assert c["note"] == ""

    def test_max_charts_cap(self):
        """一段文本里多个可图表结构 → 最多 MAX_CHARTS 张。"""
        text = "\n".join(
            f"表{i}：\n类别 | 数量\n--------\n甲 | {i}\n乙 | {i + 1}\n丙 | {i + 2}"
            for i in range(1, 7)
        )
        charts = extract_charts(text)
        assert len(charts) == MAX_CHARTS

    def test_source_tool_recorded(self):
        """记录来源工具名：便于排查"哪个工具的格式变了导致提取不到"。"""
        c = one("标的物 | 数量\n------\n甲 | 1\n乙 | 2\n", tool="query_database")
        assert c is not None
        assert c["source_tool"] == "query_database"

    def test_no_title_does_not_fabricate(self):
        """取不到标题时返回空串，**不编造**。

        图表上宁可不显示标题，也不要写一个与数据不符的名字。
        """
        c = one("标的物 | 数量\n------\n甲 | 1\n乙 | 2\n")
        assert c is not None
        assert c["title"] == ""

    def test_output_is_json_serializable(self):
        """要随 SSE 下发，必须可 JSON 序列化。"""
        import json

        text = (
            "知识图谱规模：\n- 甲：1\n- 乙：2\n- 丙：3\n"
            "标的物 | 数量\n------\n空调 | 175\n粉盒 | 150\n"
        )
        charts = extract_charts(text)
        json.dumps(charts, ensure_ascii=False)  # 不应抛


# ===========================================================================
# 四、接入 Agent 的契约
# ===========================================================================


class TestAgentIntegration:
    """图表随 `done` 帧下发。"""

    def test_done_frame_has_charts_field(self):
        """★ 即使没有图表，`charts` 字段也必须存在（空数组）。

        前端据此判断"有没有图"，字段缺失与空数组对前端是两种情况
        （`undefined` 会让 `.map` 之前的分支判断更啰嗦）。
        """
        import json

        from src.agent import BiddingAgent

        class FakeLLM:
            provider = "fake"
            model_name = "fake"
            chain: list = []
            last_used_provider = "fake"
            last_reasoning = ""
            trace_recorder = None

            def set_thinking(self, *a, **k):
                pass

            def supports_reasoning(self):
                return False

            def available(self):
                return True

            def chat_raw(self, messages, tools=None, **kw):
                self.last_reasoning = ""
                # 调一次 query_database，返回可图表的数据
                return json.dumps(
                    {"tool_calls": [{"name": "query_database", "arguments": {
                        "query_type": "top_subject", "limit": 3}}]},
                    ensure_ascii=False,
                )

            def chat_stream(self, messages, **kw):
                yield "根据数据…"

        # 让工具执行器返回可图表文本（不打真实数据库）
        import src.tools.rag_tools as rt

        original = rt.TOOL_EXECUTORS.get("query_database")
        rt.TOOL_EXECUTORS["query_database"] = lambda args, q="": (
            "业务数据库查询结果（共 3 条）：\n标的物 | 数量\n--------\n"
            "空调 | 175\n粉盒 | 150\n硒鼓 | 112\n",
            [],
        )
        import src.agent.react_loop as rl

        rl.TOOL_EXECUTORS["query_database"] = rt.TOOL_EXECUTORS["query_database"]
        try:
            agent = BiddingAgent(llm_client=FakeLLM())
            done: dict = {}
            for et, payload in agent._chat_events("统计一下采购最多的标的物"):
                if et == "done":
                    done = payload
            assert "charts" in done, "done 帧必须带 charts 字段"
            assert len(done["charts"]) >= 1, f"应提取到图表，实得 {done['charts']}"
            c = done["charts"][0]
            assert c["data"][0]["label"] == "空调"
            assert c["data"][0]["value"] == 175.0
        finally:
            if original is not None:
                rt.TOOL_EXECUTORS["query_database"] = original
                rl.TOOL_EXECUTORS["query_database"] = original
