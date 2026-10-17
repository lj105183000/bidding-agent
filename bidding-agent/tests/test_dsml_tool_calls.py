"""DSML / XML 风格工具调用的检测与解析测试。

★ 为什么需要这个文件
================
`tests/test_tool_text_parsing.py` 覆盖的是 JSON 风格泄漏。而实测发现
**DSML / XML 风格的泄漏完全没被覆盖**：

    原 `_looks_like_tool_text` 只查 3 个子串：
        "<tool_call>" / `tool_calls` 关键字 / `{"name":`
    而项目文档（docs/组件工作机制.md §8）声称检测范围含
        "XML 标签: tool / invoke / function_call、DSML 标记、裸函数名"
    —— **文档描述的是设想的系统，代码只实现了一个子集。**

后果：模型输出 DSML 风格的调用时，检测不到 → 不会 reset → 不会兜底执行
→ **那串标记会原样展示给用户**（用户实际遇到的就是这个）。

★ 样本来源
========
`dsml_samples.USER_SAMPLE` 是用户实际遇到的泄漏输出（结构与内容原样保留）。
其余变体来自"真实可能出现的写法"，逐个都有存在理由（见样本文件注释）。
"""

from __future__ import annotations

import json

import pytest

from src.agent.tool_defense import (
    _looks_like_tool_text,
    _normalize_tool_text,
    _parse_tool_text,
)

from dsml_samples import (  # noqa: E402
    BENIGN,
    USER_SAMPLE,
    VARIANTS,
)

# ===========================================================================
# ① 用户实际遇到的泄漏
# ===========================================================================


class TestUserReportedLeak:
    """用户上报的真实泄漏样本。**这是本文件的回归基线。**"""

    def test_detected(self):
        assert _looks_like_tool_text(USER_SAMPLE), (
            "用户上报的 DSML 泄漏样本必须被检测到——"
            "检测不到就不会触发 reset 与兜底执行，标记会直接展示给用户"
        )

    def test_parsed_into_two_calls(self):
        calls = _parse_tool_text(USER_SAMPLE)
        assert len(calls) == 2, f"应解析出两个调用，实得 {len(calls)}"
        assert all(c["name"] == "search_knowledge_graph" for c in calls)

    def test_arguments_are_clean(self):
        """参数值必须是干净的关键词，不含 DSML 标记或空白污染。"""
        calls = _parse_tool_text(USER_SAMPLE)
        keywords = [c["arguments"].get("keyword") for c in calls]
        assert keywords == ["管理局", "中学"], f"关键词解析错误: {keywords}"
        for c in calls:
            assert c["arguments"].get("query_type") == "search_entity"
            for value in c["arguments"].values():
                assert "DSML" not in value, f"参数值被标记污染: {value!r}"
                assert value == value.strip(), f"参数值有首尾空白: {value!r}"

    def test_user_visible_reason(self):
        """说明为什么这必须修：标记会原样出现在回答里。

        这里不是断言实现细节，而是把**危害**固化成测试文档——
        将来有人想删掉这段逻辑时，会先看到这条测试的说明。
        """
        # 未修复时：检测 False → 不 reset → 标记原样下发
        assert _looks_like_tool_text(USER_SAMPLE) is True
        # 修复后：能解析出真实调用 → 可以真的去执行工具
        assert _parse_tool_text(USER_SAMPLE), "能检测但解析不出 → 仍会兜底失败"


# ===========================================================================
# ② 变体覆盖
# ===========================================================================


class TestVariants:
    """各变体都要既检测到、又解析出正确调用。

    `VARIANTS` 是"形态清单"：新增一种写法就往清单里加一条。
    ★ 这种写法的价值：**漏掉某个变体会在测试里暴露**，而不是等到线上
    用户看到一串标记（原实现漏掉 DSML 就是这个模式导致的）。
    """

    @pytest.mark.parametrize(
        "label,text,count,name",
        VARIANTS,
        ids=[v[0] for v in VARIANTS],
    )
    def test_detected_and_parsed(self, label, text, count, name):
        assert _looks_like_tool_text(text), f"[{label}] 未被检测到"
        calls = _parse_tool_text(text)
        assert len(calls) == count, f"[{label}] 应解析 {count} 个，实得 {len(calls)}"
        assert calls[0]["name"] == name, f"[{label}] 工具名错误: {calls[0]['name']}"

    def test_truncated_invoke_has_no_closing_tag(self):
        """★ invoke 未闭合也必须能解析。

        真实场景：模型输出被 max_tokens 截断，收尾标签丢了。
        要求闭合标签会让这类泄漏**恰好**解析不出来——而它正是最需要
        兜底执行的情况（用户已经看到半截标记了）。
        """
        text = dict((v[0], v[1]) for v in VARIANTS)["invoke 未闭合"]
        assert "</invoke>" not in text, "样本本身就不该有收尾标签"
        calls = _parse_tool_text(text)
        assert calls and calls[0]["arguments"].get("keyword") == "管理局"

    def test_namespace_prefix_parsed(self):
        text = dict((v[0], v[1]) for v in VARIANTS)["命名空间前缀"]
        calls = _parse_tool_text(text)
        assert calls and calls[0]["name"] == "search_knowledge_base"
        assert calls[0]["arguments"] == {"query": "投标保证金"}

    def test_multi_param_all_kept(self):
        text = dict((v[0], v[1]) for v in VARIANTS)["多参数"]
        calls = _parse_tool_text(text)
        assert calls[0]["arguments"] == {"sql": "SELECT 1", "limit": "10"}


# ===========================================================================
# ③ 防误伤
# ===========================================================================


class TestNoFalsePositives:
    """正常回答里提到工具名不该被判为工具调用。

    ★ 为什么这条很重要
    --------------
    误判的代价不是"多跑一次工具"，而是**正常回答被 reset 清空重来**——
    用户会看到回答闪一下然后变了。故检测必须"疑高"但不能"滥"。
    """

    @pytest.mark.parametrize("text", BENIGN, ids=lambda t: t[:16])
    def test_benign_not_detected(self, text):
        assert not _looks_like_tool_text(text), f"误判为工具调用: {text!r}"

    @pytest.mark.parametrize("text", BENIGN, ids=lambda t: t[:16])
    def test_benign_parses_to_nothing(self, text):
        assert _parse_tool_text(text) == [], f"误解析出调用: {text!r}"

    def test_mentioning_tool_name_without_call_syntax(self):
        """提到 `search_knowledge_base` 但后面没跟括号 → 不是调用。

        `_BARE_CALL_RE` 只在"紧跟左括号"时才算，正是为了避免这一条误判。
        """
        assert not _looks_like_tool_text("可以用 search_knowledge_base 检索。")
        assert not _looks_like_tool_text("search_knowledge_base 是四个基础工具之一。")


# ===========================================================================
# ④ 原有形态未退化
# ===========================================================================


class TestExistingFormatsStillWork:
    """JSON / 代码围栏 / `<tool_call>` 标签三种原有形态不能被改坏。

    ★ 为什么必须单独测
    --------------
    新增 XML 解析是**在解析入口最前面**插入的一次尝试。若它写得太宽松
    （例如把任意含 `name=` 的文本都吃掉），会抢在 JSON 解析之前返回错误结果。
    故这里把原有形态逐条锁住。
    """

    CASES = [
        ('{"tool_calls": [{"name": "search_knowledge_base", "arguments": {"query": "x"}}]}',
         "search_knowledge_base"),
        ('{"name": "search_knowledge_graph", "arguments": {"keyword": "y"}}',
         "search_knowledge_graph"),
        ('```json\n{"tool_calls": [{"name": "query_database", "arguments": {}}]}\n```',
         "query_database"),
        ('<tool_call>{"name": "search_knowledge_base", "arguments": {"query": "z"}}</tool_call>',
         "search_knowledge_base"),
    ]

    @pytest.mark.parametrize("text,expect", CASES, ids=lambda v: str(v)[:24])
    def test_still_parsed(self, text, expect):
        calls = _parse_tool_text(text)
        assert calls, f"原有形态解析失败: {text[:50]}"
        assert calls[0]["name"] == expect

    def test_string_arguments_still_double_parsed(self):
        """部分模型把 arguments 序列化成字符串，原有二次解析逻辑不能丢。"""
        text = '{"name": "query_database", "arguments": "{\\"q\\": \\"金额\\"}"}'
        calls = _parse_tool_text(text)
        assert calls[0]["arguments"] == {"q": "金额"}

    def test_dedup_still_works(self):
        """同名同参去重（模型偶发重复输出，重复执行会放大后端压力）。"""
        text = ('{"tool_calls": ['
                '{"name": "a", "arguments": {"x": 1}},'
                '{"name": "a", "arguments": {"x": 1}}]}')
        assert len(_parse_tool_text(text)) == 1

    def test_dedup_applies_to_xml_too(self):
        """去重对 XML 形态同样生效。"""
        from dsml_samples import FW

        one = (f'<{FW}DSML{FW}invoke name="search_knowledge_base">'
               f'<{FW}DSML{FW}parameter name="query">同一问题</parameter></invoke>')
        assert len(_parse_tool_text(one + "\n" + one)) == 1


# ===========================================================================
# ⑤ 归一化
# ===========================================================================


class TestNormalization:
    def test_dsml_marker_removed(self):
        """归一化后不该残留 DSML 标记（否则参数值会被污染）。"""
        from dsml_samples import FW_SINGLE

        cleaned = _normalize_tool_text(FW_SINGLE)
        assert "DSML" not in cleaned, f"标记未清除: {cleaned[:80]!r}"
        assert "invoke" in cleaned, "标签名应保留（解析器要用）"

    def test_existing_behavior_unchanged(self):
        fenced = '```json\n{"name": "x", "arguments": {}}\n```'
        assert _normalize_tool_text(fenced) == '{"name": "x", "arguments": {}}'
        tagged = '前缀<tool_call>{"name": "x"}</tool_call>后缀'
        assert _normalize_tool_text(tagged) == '{"name": "x"}'

    def test_parse_result_is_json_serializable(self):
        """解析结果要能直接进 `meta` 与日志（不能含不可序列化对象）。"""
        calls = _parse_tool_text(USER_SAMPLE)
        json.dumps(calls, ensure_ascii=False)  # 不应抛
