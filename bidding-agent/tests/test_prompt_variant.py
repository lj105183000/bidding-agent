"""提示词变体机制的单测。

背景（数据驱动，见 docs/21-RAGAS评测改造与数据驱动优化.md）：
    ragas 评测发现"几乎原题"的忠实度反而最低（0.494 vs 库内缺失题的 0.872）。
    排查确认原因在生成侧——库内答案是"一句话法条"，原提示词却要求"条理清晰"，
    模型于是补出解释与推断，这些内容在参考里没有，Faithfulness 判低分。

故新增 strict / minimal 两个更严的变体，用 `RAG_PROMPT_VARIANT` 选择。
这些测试保证：变体可切换、非法值不炸服务、切换机制不被破坏。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_with(monkeypatch, value: str | None):
    """在指定环境变量下重新加载 pipeline 模块。"""
    if value is None:
        monkeypatch.delenv("RAG_PROMPT_VARIANT", raising=False)
    else:
        monkeypatch.setenv("RAG_PROMPT_VARIANT", value)
    import src.rag.pipeline as mod

    return importlib.reload(mod)


@pytest.fixture(autouse=True)
def _restore_module():
    """测完把模块恢复成默认变体，避免污染同进程的其它测试。

    pipeline 是模块级单例，`importlib.reload` 会改全局状态——
    不还原的话，后续测试可能跑在 strict 提示词下，产生难查的连带失败。
    """
    yield
    import os

    os.environ.pop("RAG_PROMPT_VARIANT", None)
    import src.rag.pipeline as mod

    importlib.reload(mod)


def test_variants_registered():
    import src.rag.pipeline as mod

    assert set(mod._PROMPT_VARIANTS) == {"default", "balanced", "strict", "minimal"}


def test_default_is_active_without_env(monkeypatch):
    mod = _reload_with(monkeypatch, None)
    assert mod.PROMPT_VARIANT == "default"
    assert mod.ACTIVE_SYSTEM_PROMPT == mod.SYSTEM_PROMPT


def test_strict_variant_selectable(monkeypatch):
    mod = _reload_with(monkeypatch, "strict")
    assert mod.PROMPT_VARIANT == "strict"
    assert mod.ACTIVE_SYSTEM_PROMPT == mod.SYSTEM_PROMPT_STRICT
    assert mod.ACTIVE_SYSTEM_PROMPT != mod.SYSTEM_PROMPT


@pytest.mark.parametrize("value", ["STRICT", "Strict", " strict ", "strict\n"])
def test_variant_name_is_case_and_space_insensitive(monkeypatch, value):
    """环境变量常被手写，大小写与前后空格不该导致回落。"""
    mod = _reload_with(monkeypatch, value)
    assert mod.PROMPT_VARIANT == "strict"


def test_unknown_variant_falls_back_without_crashing(monkeypatch):
    """★ 非法值只回落告警，**不抛异常**。

    拼错一个环境变量就让整个问答服务起不来，是把配置错误的代价放大到
    不可用级别——那比"用了默认提示词"严重得多（与 RERANK_ENABLED 同口径）。
    """
    mod = _reload_with(monkeypatch, "bogus")
    assert mod.PROMPT_VARIANT == "default"
    assert mod.ACTIVE_SYSTEM_PROMPT == mod.SYSTEM_PROMPT


def test_balanced_prompt_forbids_elaboration():
    """★ balanced 必须同样禁止三类越界——它是 strict 的替代品，不是放宽版。

    它相对 strict 放宽的是"完整性"（要求把参考里有的讲完），
    而**不是**放宽"不许超纲"。若有人误把它当成"温和版 strict"而删掉禁止条款，
    忠实度会退回改造前水平。
    """
    import src.rag.pipeline as mod

    text = mod.SYSTEM_PROMPT_BALANCED
    assert "补充说明" in text or "要点提示" in text
    assert "推断" in text
    assert "条款号" in text or "法规名称" in text
    assert "常识" in text


def test_balanced_prompt_requires_completeness():
    """★ balanced 相对 strict 的关键增量：明确要求**完整作答**。

    这是修复"相关性掉 0.097"的核心——strict 的第 3 条
    （"参考内容较短时直接给出结论即可"）被模型过度执行，
    导致个案相关性从 0.937 掉到 0.362（217 字 → 40 字）。
    balanced 必须把"完整"写成正面要求。
    """
    import src.rag.pipeline as mod

    text = mod.SYSTEM_PROMPT_BALANCED
    assert "完整" in text
    # 必须区分"完整"与"扩写"，否则模型又会走回 strict 的老路
    assert "扩写" in text or "一句话" in text


def test_balanced_handles_partial_coverage_explicitly():
    """参考只覆盖部分问题时要有明确指引，而不是让模型自行取舍。

    第 6 题的失败正是这种情况：参考只回答了"是否可以写明"，
    模型要么漏答、要么用常识补上。balanced 要求"答完覆盖到的部分 +
    明确指出未涉及的部分"——两者都不做越界。
    """
    import src.rag.pipeline as mod

    text = mod.SYSTEM_PROMPT_BALANCED
    assert "未涉及" in text or "未覆盖" in text


def test_strict_prompt_forbids_elaboration():
    """★ strict 必须显式禁止三类越界——这是它存在的全部理由。

    若有人"精简"提示词时删掉这些约束，忠实度会退回改造前的水平，
    而外表上完全看不出来（提示词还是一段中文）。
    """
    import src.rag.pipeline as mod

    text = mod.SYSTEM_PROMPT_STRICT
    # ① 禁止自撰解读段落
    assert "补充说明" in text or "要点提示" in text
    # ② 禁止从参考推断新结论
    assert "推断" in text
    # ③ 禁止参考未提及的法规名称/条款号
    assert "法规名称" in text or "条款号" in text
    # ④ 参考不足时要求停止，而不是用常识补充
    assert "常识" in text


def test_minimal_prompt_is_shorter_than_balanced():
    """minimal 是 A/B 的下界参照，应当比 balanced 更短。"""
    import src.rag.pipeline as mod

    assert len(mod.SYSTEM_PROMPT_MINIMAL) < len(mod.SYSTEM_PROMPT_BALANCED)


def test_build_messages_uses_active_prompt(monkeypatch):
    """★ build_messages 必须读 `ACTIVE_SYSTEM_PROMPT`，而不是 `SYSTEM_PROMPT`。

    这是最容易失效的一环：若它读死 `SYSTEM_PROMPT`，选任何变体都不会
    改变实际行为，A/B 三组会产出相同分数——让人误以为"提示词没用"。
    本测试直接检查进到 messages 里的 system 内容。
    """
    from src.rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    sources = [{"question": "q", "answer": "a"}]

    import src.rag.pipeline as mod

    original = mod.ACTIVE_SYSTEM_PROMPT
    try:
        mod.ACTIVE_SYSTEM_PROMPT = "【测试用提示词】"
        messages = pipe.build_messages("问题", sources, None)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "【测试用提示词】"
    finally:
        mod.ACTIVE_SYSTEM_PROMPT = original


def test_ab_presets_cover_all_variants():
    """评测预置里必须有为每个变体准备的配置，否则 A/B 无从写起。"""
    from eval import ragas_eval as R

    covered = {
        c.prompt_variant
        for c in R._PRESETS.values()
        if c.prompt_variant
    }
    assert covered == {"default", "balanced", "strict", "minimal"}


def test_ab_presets_hold_top_k_constant():
    """★ 提示词 A/B 必须保持 top_k 与精排一致，否则它是多变量实验。

    若三组配置的 top_k 不同，分数差异就分不清是提示词还是检索深度造成的——
    这种实验得不出任何可执行的结论。
    """
    from eval import ragas_eval as R

    prompt_presets = [c for c in R._PRESETS.values() if c.prompt_variant]
    assert len({c.top_k for c in prompt_presets}) == 1
    assert len({c.rerank for c in prompt_presets}) == 1
