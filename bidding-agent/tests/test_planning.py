"""Plan-and-Solve 规划模块单测。

重点在**容错**：规划轮是一次额外的 LLM 调用，它的输出不可控。
而实测模型即便被明确要求"只输出 JSON"，仍会包 markdown 围栏或加解释文字。
解析失败是**静默的**（计划为空 → ReAct 退化成原样），很难发现，
所以容错路径必须有测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import planning as P  # noqa: E402


def _plan_json(*steps) -> str:
    return json.dumps(
        {"needs": [{"info": i, "tool": t, "query": q} for i, t, q in steps]},
        ensure_ascii=False,
    )


class TestParsePlan:
    def test_parses_clean_json(self):
        plan = P.parse_plan(_plan_json(("安徽省安装类项目", "query_database", "安徽 安装")))
        assert plan.ok
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "query_database"
        assert plan.steps[0].query == "安徽 安装"

    def test_strips_markdown_fence(self):
        """★ 实测模型常把 JSON 包在 ```json 里，尽管提示词明确禁止。"""
        raw = "```json\n" + _plan_json(("x", "query_database", "")) + "\n```"
        assert P.parse_plan(raw).ok

    def test_tolerates_surrounding_prose(self):
        """模型常在 JSON 前后加一句说明。"""
        raw = "好的，分析如下：\n" + _plan_json(("x", "query_database", "")) + "\n以上。"
        assert P.parse_plan(raw).ok

    def test_returns_not_ok_on_garbage(self):
        plan = P.parse_plan("这不是 JSON，我只是在闲聊。")
        assert plan.ok is False
        assert plan.reason

    def test_returns_not_ok_on_empty(self):
        assert P.parse_plan("").ok is False
        assert P.parse_plan(None).ok is False  # type: ignore[arg-type]

    def test_returns_not_ok_on_empty_needs(self):
        assert P.parse_plan(json.dumps({"needs": []})).ok is False
        assert P.parse_plan(json.dumps({"needs": "不是列表"})).ok is False

    def test_returns_not_ok_when_needs_missing(self):
        assert P.parse_plan(json.dumps({"other": 1})).ok is False

    def test_skips_malformed_steps(self):
        """条目不是 dict、或 info 为空时跳过，不能让整份计划失败。"""
        raw = json.dumps({
            "needs": [
                "裸字符串",
                {"tool": "query_database"},            # 缺 info
                {"info": "  ", "tool": "query_database"},  # info 空白
                {"info": "有效需求", "tool": "query_database"},
            ]
        })
        plan = P.parse_plan(raw)
        assert plan.ok
        assert len(plan.steps) == 1
        assert plan.steps[0].info == "有效需求"

    def test_caps_at_max_steps(self):
        """★ 超过 MAX_PLAN_STEPS 要截断：轮次上限也是 4，
        需求多于轮次时后面几条必然执行不到，列出来只会让模型纠结先做哪个。"""
        raw = _plan_json(*[(f"需求{i}", "query_database", "") for i in range(10)])
        plan = P.parse_plan(raw)
        assert len(plan.steps) == P.MAX_PLAN_STEPS

    def test_filters_tools_not_exposed(self):
        """★ 计划里出现未暴露的工具必须过滤。

        用户没开联网却规划了 search_web，模型会去调一个本轮不存在的工具，
        白浪费一轮——这与工具白名单门禁是同一道防线，只是提前到规划阶段。
        """
        raw = _plan_json(
            ("查法规", "search_knowledge_base", ""),
            ("查最新公告", "search_web_cn", ""),
        )
        plan = P.parse_plan(raw, allowed_tools={"search_knowledge_base", "query_database"})
        assert plan.ok
        assert [s.tool for s in plan.steps] == ["search_knowledge_base"]
        assert "search_web_cn" in plan.reason

    def test_all_filtered_means_no_plan(self):
        raw = _plan_json(("查最新公告", "search_web_cn", ""))
        plan = P.parse_plan(raw, allowed_tools={"search_knowledge_base"})
        assert plan.ok is False

    def test_keeps_steps_without_tool(self):
        """`tool` 可以为空（模型只说了要什么信息、没说用什么工具）——
        这种步骤仍有价值，不该丢。"""
        raw = json.dumps({"needs": [{"info": "需要项目清单"}]})
        plan = P.parse_plan(raw, allowed_tools={"query_database"})
        assert plan.ok
        assert plan.steps[0].info == "需要项目清单"
        assert plan.steps[0].tool == ""

    def test_no_filter_when_allowed_is_none(self):
        raw = _plan_json(("x", "任意工具名", ""))
        assert P.parse_plan(raw, allowed_tools=None).ok


class TestPlanTools:
    def test_dedupes_preserving_order(self):
        plan = P.Plan(steps=[
            P.PlanStep("a", "query_database"),
            P.PlanStep("b", "search_knowledge_base"),
            P.PlanStep("c", "query_database"),
        ], ok=True)
        assert plan.tools() == ["query_database", "search_knowledge_base"]

    def test_to_dict_shape(self):
        plan = P.parse_plan(_plan_json(("x", "query_database", "q")))
        d = plan.to_dict()
        for key in ("ok", "reason", "steps"):
            assert key in d
        assert set(d["steps"][0]) == {"info", "tool", "query"}


class TestBuildPlanNote:
    def test_renders_steps_with_tool_and_query(self):
        plan = P.parse_plan(_plan_json(("安徽省安装类项目", "query_database", "安徽 安装")))
        note = P.build_plan_note(plan)
        assert "安徽省安装类项目" in note
        assert "query_database" in note
        assert "安徽 安装" in note

    def test_renders_step_without_tool(self):
        plan = P.parse_plan(json.dumps({"needs": [{"info": "需要清单"}]}))
        note = P.build_plan_note(plan)
        assert "需要清单" in note

    def test_says_suggestion_not_command(self):
        """★ 措辞必须是"建议"而非"必须"。

        计划看不到任何检索结果，对"库里有没有"完全没有概念。
        用命令式会压制 ReAct 的纠错能力——而那正是它相对 Plan-and-Solve 的优势。
        """
        plan = P.parse_plan(_plan_json(("x", "query_database", "")))
        note = P.build_plan_note(plan)
        assert "建议" in note
        assert "不要为了对齐计划而硬凑" in note

    def test_empty_for_failed_plan(self):
        assert P.build_plan_note(P.Plan(ok=False)) == ""
        assert P.build_plan_note(P.Plan(ok=True, steps=[])) == ""


# ===========================================================================
# 接线：规划必须真的接进 ReAct 循环
# ===========================================================================


class TestPlanningWiring:
    """只测 `planning` 模块不够——若 ReAct 忘了用它，规划就是死代码，单测照样全绿。"""

    def test_append_plan_note_puts_text_in_system(self):
        from src.agent.react_loop import ReActMixin

        msgs = [{"role": "system", "content": "BASE"}, {"role": "user", "content": "q"}]
        out = ReActMixin._append_plan_note(msgs, "【建议的检索计划】1. 查法规")
        assert out[0]["role"] == "system"
        assert "BASE" in out[0]["content"]
        assert "建议的检索计划" in out[0]["content"]
        # 不得就地修改入参
        assert msgs[0]["content"] == "BASE"

    def test_append_plan_note_inserts_system_when_absent(self):
        from src.agent.react_loop import ReActMixin

        out = ReActMixin._append_plan_note([{"role": "user", "content": "q"}], "计划")
        assert out[0]["role"] == "system"
        assert len(out) == 2

    def test_planner_failure_does_not_break_qa(self):
        """★ 规划是**增益项**：调用失败必须按"无计划"继续，不能影响问答。"""

        class _BoomClient:
            def chat_raw(self, messages, tools=None, **kwargs):
                raise RuntimeError("规划轮炸了")

        from src.agent.react_loop import ReActMixin

        meta: dict = {}
        note = ReActMixin()._run_planner("问题", _BoomClient(), meta, [])
        assert note == ""
        assert meta["plan"]["ok"] is False
        assert "失败" in meta["plan"]["reason"]

    def test_planner_garbage_output_is_safe(self):
        """模型返回非 JSON 时同样降级，且把原因写进 meta 便于排查。"""

        class _GarbageClient:
            def chat_raw(self, messages, tools=None, **kwargs):
                return "我不太确定该怎么规划，先随便说说。"

        from src.agent.react_loop import ReActMixin

        meta: dict = {}
        note = ReActMixin()._run_planner("问题", _GarbageClient(), meta, [])
        assert note == ""
        assert meta["plan"]["ok"] is False

    def test_planner_filters_unexposed_tools_from_schemas(self):
        """★ 工具白名单必须从 tool_schemas 推导，而不是写死全部工具。

        否则用户没开联网、规划却给出 search_web_cn，模型会去调一个本轮
        不存在的工具，白浪费一轮。
        """

        class _Client:
            def chat_raw(self, messages, tools=None, **kwargs):
                return _plan_json(
                    ("查法规", "search_knowledge_base", ""),
                    ("查最新公告", "search_web_cn", ""),
                )

        from src.agent.react_loop import ReActMixin

        schemas = [
            {"function": {"name": "search_knowledge_base"}},
            {"function": {"name": "query_database"}},
        ]
        meta: dict = {}
        note = ReActMixin()._run_planner("问题", _Client(), meta, schemas)
        assert "search_knowledge_base" in note
        assert "search_web_cn" not in note
        assert [s["tool"] for s in meta["plan"]["steps"]] == ["search_knowledge_base"]
