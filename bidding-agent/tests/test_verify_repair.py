"""核对失败后的**定向修复轮**测试（Reflection：从「只上报」到「能纠正」）。

★ 设计要点（每条都有对应用例）
========================
    ① 默认关闭：不开开关时行为与改动前**完全一致**（不能悄悄多花一次 LLM 调用）
    ② 真的能修：核对失败 → status + reset + 重新流式输出修正后的答案
    ③ 修完要复核：`verification_repair` 里同时留 before/after，A/B 才有依据
    ④ **失败不能更糟**：LLM 报错 / 输出过短 / 没有工具结果时，
       一律保留原答案，事件流不产出任何 reset（否则用户会看到空白）
"""

import json

import pytest

import src.agent.core as core_mod
from src.agent import BiddingAgent


class _FakeClient:
    """只实现修复轮需要的 `chat_raw`。"""

    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls: list[list] = []

    def chat_raw(self, messages, tools=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


def _failing_verification():
    return {
        "ok": False,
        "critical_ok": False,
        "unsupported_articles": ["第二十四条"],
        "unsupported_numbers": ["10%"],
        "unsupported_laws": [],
        "false_unavailable_claims": [],
    }


def _meta(answer="原始回答。依据《招标投标法》第二十四条，保证金不得超过 10%。"):
    return {
        "answer_text": answer,
        "tool_output_text": ["【工具 search_knowledge_base 结果】投标保证金不得超过招标项目估算价的2%。"],
        "succeeded_tools": ["search_knowledge_base"],
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# ① 默认关闭
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_VERIFY_REPAIR", raising=False)
    assert BiddingAgent._repair_enabled() is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "随便写的"])
def test_invalid_values_keep_it_off(monkeypatch, value):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", value)
    assert BiddingAgent._repair_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_explicit_on(monkeypatch, value):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", value)
    assert BiddingAgent._repair_enabled() is True


def test_off_produces_no_events(monkeypatch):
    """★ 关闭时必须**一个事件都不产**（行为与改动前一致）。"""
    monkeypatch.delenv("AGENT_VERIFY_REPAIR", raising=False)
    agent = BiddingAgent(llm_client=_FakeClient("修好的回答"))
    events = list(agent._repair_answer_events(_meta(), _failing_verification(),
                                             _FakeClient("修好的回答")))
    assert events == []


# ---------------------------------------------------------------------------
# ② 真的能修
# ---------------------------------------------------------------------------


def test_repair_emits_status_reset_and_tokens(monkeypatch):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    fixed = "修正后的回答。依据检索结果，投标保证金不得超过招标项目估算价的2%，" * 4
    client = _FakeClient(fixed)
    agent = BiddingAgent(llm_client=client)
    meta = _meta()

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))
    kinds = [k for k, _ in events]

    assert kinds[0] == "status", "应先告诉用户「正在修正引用」"
    assert "reset" in kinds, "必须先作废已流出的正文，否则新旧答案会叠在一起"
    assert kinds.index("reset") < kinds.index("token"), "reset 必须在重发之前"
    assert kinds.count("token") > 1, "修正后的答案要按帧重发"
    # 拼回来的正文应与修复输出一致（只有切帧，没有丢字）
    assert "".join(p for k, p in events if k == "token") == fixed
    # meta 里的答案正文也要跟着更新（否则 done 帧/后续核对拿到的是旧答案）
    assert meta["answer_text"] == fixed


def test_prompt_contains_issues_and_tool_text(monkeypatch):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    client = _FakeClient("修正后的回答" * 20)
    agent = BiddingAgent(llm_client=client)

    list(agent._repair_answer_events(_meta(), _failing_verification(), client))

    assert client.calls, "修复轮没有发起 LLM 调用"
    prompt = json.dumps(client.calls[0], ensure_ascii=False)
    assert "第二十四条" in prompt, "问题清单里必须点名具体条款号"
    assert "10%" in prompt, "数字问题也要点名"
    assert "2%" in prompt, "必须带上检索结果（否则模型只能凭记忆再编一次）"
    assert "不要改变结论" in prompt, "任务边界要写清：只改引用、不改结论"


# ---------------------------------------------------------------------------
# ③ 修完复核
# ---------------------------------------------------------------------------


def test_records_before_and_after(monkeypatch):
    """修复后必须**再核对一次**，并把 before/after 都留下。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    fixed = "修正后的回答，不再提及任何条款号与百分比。" * 6
    agent = BiddingAgent(llm_client=_FakeClient(fixed))
    meta = _meta()

    list(agent._repair_answer_events(meta, _failing_verification(), _FakeClient(fixed)))

    repair = meta.get("verification_repair")
    assert repair and repair["applied"] is True
    assert repair["before"]["unsupported_articles"] == ["第二十四条"]
    assert "after" in repair, "没有 after 就无法判断修复有没有生效"
    # 修复后的答案里去掉了条款号与 10%，核对应通过
    assert repair["after"]["ok"] is True
    assert meta["verification"] == repair["after"]


def test_no_change_output_is_not_treated_as_repaired(monkeypatch):
    """★ 模型逐字返回原文时：不 reset、不重发，如实记 `no_change`。

    实测踩到：第一版无论改没改都 reset + 重发全文 —— 用户白等一次 LLM 调用、
    界面还要闪一下（清空再逐字重画同一段）。故"没改"必须与"改好了"区分开。
    """
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    # 用够长的答案（生产里是几百到上千字），避免先被"输出过短"的保守闸拦下
    long_answer = ("原始回答。依据《招标投标法》第二十四条，投标保证金不得超过估算价的 10%，"
                   "这一比例在实务中常被引用，但需以招标文件为准。") * 4
    meta = _meta(long_answer)
    original = meta["answer_text"]
    client = _FakeClient(f"  {original}  ")   # 逐字返回原文（只多首尾空白）
    agent = BiddingAgent(llm_client=client)

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))

    assert events == [], "一字未改却重发了全文（会白等 + 界面闪烁）"
    assert meta["answer_text"] == original
    assert meta["verification_repair"]["applied"] is False
    assert meta["verification_repair"]["reason"] == "no_change"


def test_too_short_records_reason(monkeypatch):
    """被保守闸拦下时也要记原因，否则 A/B 分不清"模型不肯改"与"被闸拦下"。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    meta = _meta("原始回答。" * 40)
    client = _FakeClient("太短")
    agent = BiddingAgent(llm_client=client)

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))

    assert events == []
    assert meta["verification_repair"]["reason"] == "too_short"


def test_issues_only_list_present_fields():
    text = BiddingAgent._format_repair_issues({
        "unsupported_articles": ["第二十四条"],
        "unsupported_numbers": [],
        "unsupported_laws": [],
        "false_unavailable_claims": [],
    })
    assert "第二十四条" in text
    assert "数字" not in text, "没有问题的类别不该出现在清单里"


def test_no_issues_means_no_repair(monkeypatch):
    """核对失败但没有可点名的项（异常情况）→ 不修，避免瞎改。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    client = _FakeClient("随便改改")
    agent = BiddingAgent(llm_client=client)

    events = list(agent._repair_answer_events(
        _meta(), {"ok": False, "unsupported_articles": [], "unsupported_numbers": [],
                  "unsupported_laws": [], "false_unavailable_claims": []}, client))

    assert events == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# ④ 失败不能更糟
# ---------------------------------------------------------------------------


def test_llm_failure_keeps_original_answer(monkeypatch):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    client = _FakeClient(error=RuntimeError("模型挂了"))
    agent = BiddingAgent(llm_client=client)
    meta = _meta()
    original = meta["answer_text"]

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))

    assert events == [], "修复失败时不能产出 reset（否则用户看到空白）"
    assert meta["answer_text"] == original
    assert "verification_repair" not in meta or not meta["verification_repair"].get("applied")


def test_too_short_output_is_rejected(monkeypatch):
    """修复输出明显过短 → 当失败处理，保留原答案（防止把好答案换成残句）。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    client = _FakeClient("太短了")
    agent = BiddingAgent(llm_client=client)
    meta = _meta()
    original = meta["answer_text"]

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))

    assert events == []
    assert meta["answer_text"] == original


def test_missing_tool_text_skips_repair(monkeypatch):
    """没有工具结果就没有依据可依 → 不该修（那只会让模型再编一次）。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    client = _FakeClient("修正后的回答" * 20)
    agent = BiddingAgent(llm_client=client)
    meta = _meta()
    meta["tool_output_text"] = []

    events = list(agent._repair_answer_events(meta, _failing_verification(), client))

    assert events == []
    assert client.calls == []


def test_missing_client_skips_repair(monkeypatch):
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    agent = BiddingAgent(llm_client=_FakeClient("x"))
    events = list(agent._repair_answer_events(_meta(), _failing_verification(), None))
    assert events == []


# ---------------------------------------------------------------------------
# ⑤ 与主流程的接线
# ---------------------------------------------------------------------------


def test_chat_events_triggers_repair_when_verification_fails(monkeypatch):
    """核对失败时，`_chat_events` 必须把修复事件透出去（接线正确）。"""
    monkeypatch.setenv("AGENT_VERIFY_REPAIR", "1")
    fixed = "修正后的回答，依据检索结果给出结论，不再引用未支持的条款号。" * 5

    class _ScriptedLLM(_FakeClient):
        def __init__(self):
            super().__init__(fixed)
            self.raw_calls = 0

        def chat_raw(self, messages, tools=None):
            self.raw_calls += 1
            self.calls.append(messages)
            # 第一轮决策（无工具）→ 返回空表示直接回答；其后是修复轮
            return "" if self.raw_calls == 1 else fixed

        def chat_stream(self, messages):
            yield "原始回答，依据《招标投标法》第二十四条。"

    agent = BiddingAgent(llm_client=_ScriptedLLM())

    def _fake_verify(self, answer, meta):
        """让核对必然失败，并补上"有工具结果"这一前提。

        ★ 为什么必须补：`_repair_answer_events` 在没有工具结果时会**主动跳过**
        （没有依据可依，修只会让模型再编一次）。而本用例用的是脚本化 LLM，
        没有真的调工具，故 `tool_output_text` 是空的 —— 不补的话
        触发不了修复，测试会误判成"接线断了"。
        """
        meta["tool_output_text"] = ["【工具 search_knowledge_base 结果】保证金不超过估算价的2%。"]
        meta["succeeded_tools"] = ["search_knowledge_base"]
        meta["tool_called"] = True
        meta["verification"] = _failing_verification()
        return meta["verification"]

    monkeypatch.setattr(core_mod.BiddingAgent, "_verify_final_answer", _fake_verify)

    events = list(agent._chat_events("投标保证金上限是多少？"))
    kinds = [k for k, _ in events]

    assert "reset" in kinds, "接线断了：核对失败后没有触发修复"
    assert kinds.count("token") >= 2
    done = next(p for k, p in events if k == "done")
    assert done.get("verification_repair"), "done 帧里必须带上修复记录（A/B 依据）"
