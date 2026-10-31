"""`check_tender_compliance` 工具 + 工具注册契约的测试。

★ 为什么需要一个"注册契约"测试
----------------------------
本轮实测连续踩了三个**静默**缺陷，每一个都让新工具"看起来注册了、实际用不了"：

    ① Qdrant `topics` 缺 payload 索引 → 带主题过滤的检索 400
       → 被上层当成"语料不可用"静默降级 → 法规依据恒为空
    ② `TOOL_EXECUTORS` 含新工具，但 `get_tool_schemas` 仍只遍历旧列表
       → **模型看不到工具，永远不会调用**；注册表查起来却一切正常
    ③ `TOOLS`（BaseTool 列表）同样漏了新工具 → 契约测试抓到

②③ 属同一类：**注册表与暴露面用了两个来源**。本文件的
`TestToolRegistryContract` 把"注册的必须暴露、暴露的必须注册"钉死。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools import compliance_tools as CT  # noqa: E402
from src.tools.rag_tools import TOOL_EXECUTORS, TOOLS, get_tool_schemas  # noqa: E402

#: 用户会贴进来的典型条款（命中多种违规特征）
SAMPLE = """第三条 投标人必须在本市注册满三年，且注册资本不低于 5000 万元。
第五条 投标保证金为项目估算价的 5%。
第六条 本项目指定使用华为品牌设备。"""


class TestSplitClauses:
    """条款切分：合规检查必须**逐条**做，否则命中了也说不出是哪一条。"""

    def test_splits_numbered_clauses(self):
        clauses = CT._split_clauses(SAMPLE)
        assert len(clauses) >= 3
        assert any("本市注册" in c for c in clauses)
        assert any("华为品牌" in c for c in clauses)

    def test_splits_by_newline(self):
        clauses = CT._split_clauses("要求投标人具备甲级资质\n要求提供本地业绩\n")
        assert len(clauses) == 2

    def test_drops_tiny_fragments(self):
        """过短片段（如单字、标点）不作为条款——它们只会制造规则误匹配。"""
        clauses = CT._split_clauses("好\n第三条 投标人应当具备有效营业执照并满足全部资格条件\n")
        assert all(len(c) >= 4 for c in clauses)

    def test_caps_clause_length(self):
        clauses = CT._split_clauses("第三条 " + "长" * 5000)
        assert clauses
        assert all(len(c) <= CT.MAX_CLAUSE_CHARS for c in clauses)

    def test_caps_clause_count(self):
        text = "\n".join(f"第{i}条 投标人应当具备相应的资格条件与履约能力" for i in range(500))
        assert len(CT._split_clauses(text)) <= CT.MAX_CLAUSES

    def test_empty_input(self):
        assert CT._split_clauses("") == []
        assert CT._split_clauses(None) == []  # type: ignore[arg-type]


class TestExecutor:
    def test_rejects_empty_text(self):
        text, sources = CT.check_tender_compliance({})
        assert "未提供" in text
        assert sources == []

    def test_detects_known_violations(self):
        """★ 核心能力：规则检出（确定性，不依赖 LLM/网络）。"""
        text, _ = CT.check_tender_compliance({"text": SAMPLE})
        assert "命中疑似问题" in text
        # 三类典型违规都应命中
        assert "本地化条件" in text or "本地" in text
        assert "品牌" in text
        assert "保证金" in text or "注册资本" in text

    def test_says_rule_coverage_is_limited(self):
        """★ 必须说明"未命中不等于合规"。

        规则只覆盖已知的违规表述模式。若输出让用户以为"没命中就是合规"，
        会给出虚假的安全感——那比不检查更危险。
        """
        text, _ = CT.check_tender_compliance({"text": "第三条 投标人应当具备营业执照。"})
        assert "不代表合规" in text or "不等于合规" in text

    def test_includes_disclaimer(self):
        text, _ = CT.check_tender_compliance({"text": SAMPLE})
        assert "疑似" in text

    @pytest.mark.integration
    def test_returns_sources_with_text(self):
        """★ 来源卡片必须带正文。

        实测踩过：`_fetch_law_basis` 用的键是 `excerpt` 而非 `text`，
        取错键会静默产出**只有法规名、正文为空**的卡片。
        """
        _, sources = CT.check_tender_compliance({"text": SAMPLE})
        assert sources, "应产出法规来源"
        for s in sources:
            for key in ("question", "answer", "score", "relevance"):
                assert key in s, f"来源卡片缺字段 {key}"
        assert all(str(s["question"]).startswith("《") for s in sources)
        # 至少大部分卡片带正文（法规库检索不可用时可能为空，但正常路径应有）
        assert any(len(str(s["answer"])) > 10 for s in sources), "来源卡片正文全为空"

    def test_never_raises_on_garbage(self):
        for bad in ({}, {"text": ""}, {"text": "   "}, {"text": "短"},
                    {"text": SAMPLE, "include_law_basis": "yes"}):
            CT.check_tender_compliance(bad)  # 不该抛

    def test_include_law_basis_false(self):
        text, sources = CT.check_tender_compliance(
            {"text": SAMPLE, "include_law_basis": False}
        )
        assert "命中疑似问题" in text
        assert sources == []


class TestToolRegistryContract:
    """★ 注册表与暴露面必须同源——本轮三个静默缺陷都出在这里。"""

    def test_compliance_tool_is_registered(self):
        assert "check_tender_compliance" in TOOL_EXECUTORS

    def test_every_registered_tool_is_exposed_in_schemas(self):
        """★ 核心哨兵：注册了但没暴露 = 模型永远看不到 = 永远不会调用。

        实测踩过：`TOOL_EXECUTORS` 已含合规工具，而 `get_tool_schemas`
        仍遍历旧的 `_TOOL_SPECS`，于是"注册成功"是假象。
        注册表查起来完全正常，只有断言 schema 列表才能发现。
        """
        exposed = {s["function"]["name"] for s in get_tool_schemas()}
        registered = set(TOOL_EXECUTORS)
        assert registered == exposed, (
            f"注册与暴露不一致：仅在注册表 {registered - exposed}，"
            f"仅在暴露面 {exposed - registered}"
        )

    def test_tools_list_matches_executors(self):
        """`TOOLS`（BaseTool 列表）也必须同源——既有契约测试抓过这一条。"""
        assert sorted(t.name for t in TOOLS) == sorted(TOOL_EXECUTORS)

    def test_compliance_tool_is_in_base_names(self):
        """合规工具应**常驻暴露**（不依赖联网开关）。

        它与三个检索类工具并列：不发起外部请求、不需要联网开关，
        而用户问「这条要求合规吗」时**必须先有它**才能给出可核对的结论。
        """
        from src.agent.constants import BASE_TOOL_NAMES

        assert "check_tender_compliance" in BASE_TOOL_NAMES

    def test_schema_shape(self):
        schemas = {s["function"]["name"]: s for s in get_tool_schemas()}
        spec = schemas["check_tender_compliance"]["function"]
        assert spec["parameters"]["required"] == ["text"]
        assert "text" in spec["parameters"]["properties"]
        assert spec["description"]

    @pytest.mark.parametrize(
        "name",
        ["search_knowledge_base", "search_knowledge_graph", "query_database",
         "check_tender_compliance"],
    )
    def test_base_tools_have_executors(self, name):
        from src.agent.constants import BASE_TOOL_NAMES

        assert name in BASE_TOOL_NAMES
        assert name in TOOL_EXECUTORS
