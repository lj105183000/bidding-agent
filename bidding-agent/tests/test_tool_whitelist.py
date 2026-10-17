"""工具执行白名单门禁单测。

★ 这个门禁是**安全边界**，不是便利性
----------------------------------
`_resolve_tool_schemas` 只控制"把哪些工具的定义发给模型"，
而 `_execute_tool` 用全局注册表 `TOOL_EXECUTORS.get(name)`——
于是模型"凭记忆"输出一个未暴露的工具名，调用照样成功。

实测：用户在「关闭联网」（web_search_mode=off）下提问，
模型仍自行调用了 `search_web_cn`，返回 30 条联网来源。
**用户明确说不联网，系统却联网了**——既是行为违背，也可能外泄内网查询。

故必须有执行侧校验，且要覆盖两条路径：
    主循环（结构化 tool_calls）与文本兜底（`_fallback_execute`）。
"""

from __future__ import annotations

import pytest

from src.agent.react_loop import BASE_TOOL_NAMES, WEB_TOOL_NAMES, ReActMixin

#: ★ _execute_tool 现在是**实例方法**（它要把 RAG 计时经 self 带出，
#: 见实现处的说明）。测试里的直接调用故需要一个实例。
#:
#: 为什么不再改回 @staticmethod：计时是在**子线程**里 begin() 的，
#: 而 copy_context() 的改动不会传播回父线程——静态方法拿不到 self，
#: 就没有父线程读得到、又不改冻结契约的通道了。
_MIXIN = ReActMixin()


WEB_TOOLS = sorted(WEB_TOOL_NAMES)
BASE_TOOLS = sorted(BASE_TOOL_NAMES)


# ---------------------------------------------------------------------------
# 允许集合的计算
# ---------------------------------------------------------------------------


def test_web_tools_included_when_enabled():
    allowed = ReActMixin._allowed_tool_names({"web_search_enabled": True})
    assert allowed == set(BASE_TOOL_NAMES) | set(WEB_TOOL_NAMES)
    for name in WEB_TOOLS:
        assert name in allowed


def test_web_tools_excluded_when_disabled():
    allowed = ReActMixin._allowed_tool_names({"web_search_enabled": False})
    assert allowed == set(BASE_TOOL_NAMES)
    for name in WEB_TOOLS:
        assert name not in allowed


def test_web_tools_excluded_by_default():
    """缺省保守：没写明开关时不暴露联网工具。"""
    allowed = ReActMixin._allowed_tool_names({})
    for name in WEB_TOOLS:
        assert name not in allowed


def test_missing_meta_key_is_treated_as_disabled():
    allowed = ReActMixin._allowed_tool_names({"sources": [], "tool_name": ""})
    assert allowed == set(BASE_TOOL_NAMES)


def test_base_tools_always_allowed():
    for enabled in (True, False):
        allowed = ReActMixin._allowed_tool_names({"web_search_enabled": enabled})
        for name in BASE_TOOLS:
            assert name in allowed


def test_web_and_base_are_disjoint():
    """两组不能重叠，否则"关联网"关不干净。"""
    assert not (set(BASE_TOOL_NAMES) & set(WEB_TOOL_NAMES))


# ---------------------------------------------------------------------------
# 执行侧拒绝（核心）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WEB_TOOLS)
def test_execute_rejects_web_tool_when_disabled(name):
    """★ 关闭联网时，联网工具必须被**拒绝执行**（而不是仅不暴露）。"""
    text, sources, ok = _MIXIN._execute_tool(
        name, {"query": "测试"}, allowed=set(BASE_TOOL_NAMES)
    )
    assert ok is False, f"{name} 在关闭联网时仍被执行了"
    assert sources == [], "被拒绝的工具不应产生任何来源"
    assert "不可用" in text


def test_rejection_message_tells_model_to_switch_tool():
    """拒绝信息要引导模型换工具，而不是让它重试同一个（浪费轮次）。"""
    text, _, _ = _MIXIN._execute_tool(
        "search_web", {"query": "x"}, allowed=set(BASE_TOOL_NAMES)
    )
    assert "改用其他工具" in text or "其他工具" in text


def test_execute_rejects_any_unregistered_name_even_when_allowed():
    """允许集合里有、但注册表里没有的工具同样要拒（双向校验）。"""
    text, sources, ok = _MIXIN._execute_tool(
        "not_a_real_tool", {}, allowed={"not_a_real_tool"}
    )
    assert ok is False
    assert sources == []


def test_allowed_none_skips_gate():
    """allowed=None 表示不校验（仅供直接调用/测试）——用于确认门禁确实由 allowed 控制。"""
    # 用一个必然不存在的工具名：跳过门禁后会走到"未注册"分支而非"未暴露"分支
    text, _, ok = _MIXIN._execute_tool("not_a_real_tool", {}, allowed=None)
    assert ok is False
    assert "未注册" in text or "暂不可用" in text


def test_execute_with_empty_allowlist_rejects_everything():
    text, sources, ok = _MIXIN._execute_tool(
        BASE_TOOLS[0] if BASE_TOOLS else "search_knowledge_base", {}, allowed=set()
    )
    assert ok is False
    assert sources == []


# ---------------------------------------------------------------------------
# 与 meta 的契约
# ---------------------------------------------------------------------------


def test_meta_key_used_by_gate_is_the_one_core_sets():
    """门禁读的键必须与 core 写入的键一致，否则门禁形同虚设。"""
    import inspect

    from src.agent.core import BiddingAgent

    src = inspect.getsource(BiddingAgent._chat_events)
    assert '"web_search_enabled"' in src
    assert 'meta["web_search_enabled"] = True' in src, "自动兜底开启联网时未更新 meta"


def test_auto_fallback_sets_meta_flag_so_gate_allows_web_tools():
    """自动兜底开启联网后，门禁必须放行联网工具——否则兜底等于没做。"""
    import inspect

    from src.agent.core import BiddingAgent

    src = inspect.getsource(BiddingAgent._chat_events)
    # 兜底分支里同时设置了 meta 与 web_search_enabled
    assert 'meta["web_search_enabled"] = True' in src
    assert "web_search_enabled = True" in src
