"""「联网工具」别名扩展的回归测试（局限②）。

★ 锁住的行为
==========
`check_tool_availability_claims` 的别名表里**必须有泛称「联网工具」**。

原缺口（清单里的局限②）：模块 docstring **自己举的例子**就是
    「回答写「联网检索工具本轮不可用」」
而别名表里只有 `联网搜索 / 联网检索 / 网络搜索 / Tavily` ——
于是「本次**联网工具**不可用」（不带"检索/搜索"二字）检测不到。

实测（本轮）：
    · 未加别名：漏报 2 例
    · 加上之后：漏报 0、**误报 0**

误报面（这是清单担心扩别名的主要理由）由 `succeeded_tools` 把关：
    · web 工具**未执行** → 说"联网工具不可用"合理 → 不报
    · 只成功知识图谱   → 联网确实没跑 → 不报
    · 说"联网工具可用" → 不是否定式 → 不报
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.verification import (  # noqa: E402
    _TOOL_LABELS,
    check_tool_availability_claims,
)

WEB_TOOLS = ("search_web", "search_web_cn", "search_exa")


class TestGenericWebAlias:
    def test_generic_alias_present_on_every_web_tool(self):
        """泛称「联网工具」必须挂在**每个**联网工具上。

        只挂一个的话：若成功的是 `search_web` 而别名只加在 `search_web_cn` 上，
        就漏了。任一联网工具成功都意味着"联网可用"。
        """
        for tool in WEB_TOOLS:
            assert "联网工具" in _TOOL_LABELS[tool], (
                f"{tool} 的别名缺「联网工具」—— "
                "答案里写泛称时检测不到（这是清单记录的漏报）"
            )

    def test_detects_generic_wording_when_web_succeeded(self):
        """★ 核心断言：说「联网工具不可用」而联网**实际成功** → 必须报矛盾。"""
        for text in (
            "本次联网工具不可用，请以知识库为准。",
            "联网工具未启用，以下来自知识库。",
            "知识库可用，联网工具不可用。",
        ):
            got = check_tool_availability_claims(text, {"search_web_cn"})
            assert got, f"应检出矛盾但未检出：{text}"

    @pytest.mark.parametrize("text", [
        "本次联网工具不可用。",
        "联网工具未启用，以下来自知识库。",
    ])
    def test_no_false_positive_when_web_not_run(self, text):
        """★ 误报面：联网**没跑**时说"不可用"是**合理**的，不能报。

        这是扩别名的主要风险 —— 由 `succeeded_tools` 把关。
        """
        assert check_tool_availability_claims(text, {"search_knowledge_base"}) == []

    def test_no_false_positive_when_only_graph_ran(self):
        """只成功知识图谱 → 联网确实没跑 → 不该报。"""
        got = check_tool_availability_claims(
            "联网工具不可用，来自知识图谱。", {"search_knowledge_graph"}
        )
        assert got == []

    def test_positive_statement_not_flagged(self):
        """说「联网工具已成功调用」不是否定式 → 不报。"""
        got = check_tool_availability_claims("联网工具已成功调用。", {"search_web_cn"})
        assert got == []
