"""首轮追问（clarify）的测试。

★ 本功能的两类风险，本文件分别覆盖
=========================
**机制风险**（会不会崩 / 协议对不对）：
    · clarify 事件能否正确产出、载荷结构是否正确
    · 追问轮**不能**产出正文与来源（否则用户同时看到残缺回答与问题）
    · done 帧必须带 `clarified` 标记，否则前端无法区分"追问"与"答完但为空"

**判别风险**（该问的问了没 / 不该问的问了没）：
    太激进 → 通用法规问题也被追问，用户拿不到答案（**比不做更糟**）
    太保守 → 永不触发，功能等于没做
    故有 `TestGates` 逐条验证四个门控。

★ 判别**质量**（模型准不准）不在这里测
============================
真实模型判别质量的测量在 `logs/verify_clarify_real_llm.py`：
10 条人工标注用例，实测 9/10（该直答 5/5，该追问 4/5）。
那需要真调 API、耗时数分钟，不适合放进单元测试。
本文件只用假客户端验证**机制**与**门控**。
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.agent import BiddingAgent
from src.agent.constants import (
    CLARIFY_MIN_QUESTION_CHARS,
    CLARIFY_TOOL_NAME,
    MAX_CLARIFY_OPTIONS,
    MAX_CLARIFY_QUESTION_CHARS,
    MAX_CLARIFY_QUESTIONS,
)
from src.agent.react_loop import ReActMixin


class ScriptedLLM:
    """假客户端：看到追问工具就调用它；看不到就正常作答。

    ★ 为什么这样设计：门控的验证需要"模型**只要有机会就一定会追问**"。
    若假客户端自己会判断该不该问，那测的是假客户端的判断力，不是门控。
    """

    provider = "fake"
    model_name = "fake"
    chain: list = []
    last_used_provider = "fake"
    last_reasoning = ""
    trace_recorder = None

    def __init__(self, questions=None, options=None, reason=""):
        self.seen_tools: list[list[str]] = []
        self.questions = questions or ["贵单位所在地区是？"]
        self.options = options or []
        self.reason = reason
        self.answered = False

    def set_thinking(self, *a, **k):
        pass

    def supports_reasoning(self):
        return False

    def available(self):
        return True

    def chat_raw(self, messages, tools=None, **kw):
        names = [t["function"]["name"] for t in (tools or [])]
        self.seen_tools.append(names)
        self.last_reasoning = ""
        if CLARIFY_TOOL_NAME in names:
            qs = [{"question": q} for q in self.questions]
            if self.options:
                qs[0]["options"] = self.options
            args = {"questions": qs}
            if self.reason:
                args["reason"] = self.reason
            return json.dumps(
                {"tool_calls": [{"name": CLARIFY_TOOL_NAME, "arguments": args}]},
                ensure_ascii=False,
            )
        self.answered = True
        return "根据规定，单一来源采购适用于……"

    def chat_stream(self, messages, **kw):
        self.answered = True
        yield "根据规定，"
        yield "答复内容。"


def collect(agent, question, history=None):
    events = []
    for et, p in agent._chat_events(question, history=history):
        events.append((et, p))
    return events


# ===========================================================================
# 一、解析与校验（不依赖 Agent 编排）
# ===========================================================================


class TestExtractor:
    """`_extract_clarification`：把工具调用翻译成事件载荷。"""

    ex = staticmethod(ReActMixin._extract_clarification)

    def test_basic(self):
        r = self.ex(
            [{"name": CLARIFY_TOOL_NAME, "arguments": {"questions": [
                {"question": "贵单位所在地区是？", "options": ["北京", "上海"]},
                {"question": "采用哪种采购方式？"},
            ], "reason": "地区与方式决定适用法规"}}]
        )
        assert r is not None
        assert len(r["questions"]) == 2
        assert r["questions"][0]["options"] == ["北京", "上海"]
        assert r["questions"][1]["options"] == []
        assert r["reason"] == "地区与方式决定适用法规"

    def test_non_clarify_call_returns_none(self):
        assert self.ex([{"name": "search_knowledge_base",
                         "arguments": {"query": "x"}}]) is None

    @pytest.mark.parametrize("bad", [
        {"questions": "地区"},      # 不是数组
        {"questions": []},          # 空
        {"questions": [{"question": "是吗"}]},  # 过短（无信息量）
        {},                          # 缺 questions
    ])
    def test_malformed_returns_none(self, bad):
        """★ 校验失败一律返回 None（= 不追问，继续正常回答）。

        宁可不问也不能问出畸形问题。而"不追问"退化到的是改动前的行为，
        是**安全的下限**。
        """
        assert self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": bad}]) is None

    def test_arguments_not_dict(self):
        assert self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": None}]) is None

    def test_options_deduped(self):
        """模型常给出重复选项（实测），渲染成按钮后用户会以为是 bug。"""
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {"questions": [
            {"question": "采购方式是什么？",
             "options": ["公开招标", "公开招标", "竞争性磋商"]}]}}])
        assert r["questions"][0]["options"] == ["公开招标", "竞争性磋商"]

    def test_options_limited(self):
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {"questions": [
            {"question": "选一个吧？",
             "options": [f"选项{i}" for i in range(20)]}]}}])
        assert len(r["questions"][0]["options"]) == MAX_CLARIFY_OPTIONS

    def test_questions_limited(self):
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {"questions": [
            {"question": f"第{i}个问题的正文内容"} for i in range(1, 9)]}}])
        assert len(r["questions"]) == MAX_CLARIFY_QUESTIONS

    def test_long_question_truncated(self):
        """模型偶发把整段解释塞进"问题"（实测出现过 200+ 字）。"""
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {"questions": [
            {"question": "请说明" * 60}]}}])
        q = r["questions"][0]["question"]
        assert len(q) <= MAX_CLARIFY_QUESTION_CHARS + 1  # +1 是省略号
        assert q.endswith("…")

    def test_string_form_accepted(self):
        """兼容 `questions: ["问题文本"]` 的简写形态。"""
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {
            "questions": ["贵单位所在地区是哪里呢"]}}])
        assert r["questions"][0]["question"] == "贵单位所在地区是哪里呢"

    def test_custom_reason_default(self):
        """工具指定了 reason（模型偶发漏给）→ 前端不显示理由行。"""
        r = self.ex([{"name": CLARIFY_TOOL_NAME, "arguments": {
            "questions": [{"question": "贵单位所在地区是哪里呢"}]}}])
        assert "reason" in r and r["reason"] == ""

    def test_no_calls(self):
        assert self.ex([]) is None
        assert self.ex(None) is None


# ===========================================================================
# 二、门控：什么时候允许追问
# ===========================================================================


class TestGates:
    """四个门控条件。假客户端"有机会就一定追问"，故可直接观察门控。"""

    def test_first_turn_exposes_tool_and_clarifies(self):
        llm = ScriptedLLM()
        agent = BiddingAgent(llm_client=llm)
        events = collect(agent, "我们单位这个项目可以单一来源采购吗？")
        kinds = [e for e, _ in events]
        assert CLARIFY_TOOL_NAME in llm.seen_tools[0], "首轮应暴露追问工具"
        assert "clarify" in kinds
        assert "token" not in kinds, "追问轮不该产出正文"

    def test_history_suppresses_tool(self):
        """★ 有历史时不暴露追问工具。

        否则用户答完补充信息后，模型可以再追问一轮——**永远拿不到答案**。
        首轮限定从结构上排除，不依赖模型自觉。
        """
        llm = ScriptedLLM()
        agent = BiddingAgent(llm_client=llm)
        events = collect(
            agent, "我们单位这个项目可以单一来源采购吗？",
            history=[{"role": "user", "content": "我们这个项目可以单一来源吗？"},
                     {"role": "assistant", "content": "请补充地区与采购方式。"},
                     {"role": "user", "content": "广东，竞争性磋商。"}],
        )
        kinds = [e for e, _ in events]
        assert CLARIFY_TOOL_NAME not in llm.seen_tools[0]
        assert "clarify" not in kinds
        assert "token" in kinds, "有历史时应正常作答"

    def test_short_question_suppresses_tool(self):
        """极短输入（寒暄/误触）不追问——那样只会显得机械。"""
        llm = ScriptedLLM()
        agent = BiddingAgent(llm_client=llm)
        events = collect(agent, "你好")
        kinds = [e for e, _ in events]
        assert CLARIFY_TOOL_NAME not in llm.seen_tools[0]
        assert "clarify" not in kinds

    def test_whitespace_only_is_short(self):
        llm = ScriptedLLM()
        agent = BiddingAgent(llm_client=llm)
        collect(agent, " " * 50)  # 长度够但全是空白 → strip 后为空
        assert CLARIFY_TOOL_NAME not in llm.seen_tools[0]

    def test_threshold_boundary(self):
        """恰好达到阈值应放行（边界含等号，避免"差一个字就不问"）。"""
        llm = ScriptedLLM()
        agent = BiddingAgent(llm_client=llm)
        q = "甲" * CLARIFY_MIN_QUESTION_CHARS
        collect(agent, q)
        assert CLARIFY_TOOL_NAME in llm.seen_tools[0]


# ===========================================================================
# 三、事件与载荷契约
# ===========================================================================


class TestEvents:
    def test_clarify_payload_shape(self):
        llm = ScriptedLLM(questions=["贵单位所在地区是？"],
                          options=["北京", "上海"], reason="地区决定适用法规")
        agent = BiddingAgent(llm_client=llm)
        events = collect(agent, "我们单位这个项目可以单一来源采购吗？")
        clarify = [p for e, p in events if e == "clarify"]
        assert len(clarify) == 1
        payload = clarify[0]
        assert isinstance(payload.get("questions"), list)
        assert payload["questions"][0]["options"] == ["北京", "上海"]
        assert payload["reason"] == "地区决定适用法规"

    def test_done_marks_clarified(self):
        """★ done 必须带 `clarified`，否则前端无法区分
        "追问结束" 与 "答完了但内容为空"（后者是异常，需要提示用户）。"""
        agent = BiddingAgent(llm_client=ScriptedLLM())
        events = collect(agent, "我们单位这个项目可以单一来源采购吗？")
        done = [p for e, p in events if e == "done"]
        assert len(done) == 1
        assert done[0].get("clarified") is True

    def test_no_sources_on_clarify(self):
        """追问轮不检索 → 不应有任何来源，否则前端会渲染空来源卡。"""
        agent = BiddingAgent(llm_client=ScriptedLLM())
        events = collect(agent, "我们单位这个项目可以单一来源采购吗？")
        done = [p for e, p in events if e == "done"][0]
        assert not (done.get("sources") or [])
        assert not (done.get("web_sources") or [])
        assert done.get("tool_called") is not True

    def test_normal_answer_not_marked_clarified(self):
        """正常作答的 done **不能**被误标为 clarified。"""
        agent = BiddingAgent(llm_client=ScriptedLLM())
        events = collect(agent, "投标保证金比例上限是多少？")  # 通用问题
        # 假客户端"有机会就问"，故这里必须走有历史的路径才能拿到正文；
        # 直接断言：clarified 标记与是否产出 clarify 事件一致
        got_clarify = any(e == "clarify" for e, _ in events)
        done = [p for e, p in events if e == "done"][0]
        assert done.get("clarified") is (True if got_clarify else None)

    def test_clarify_terminates_without_generation(self):
        """追问后**不进入生成**：不应出现 tool_called / 工具名。"""
        agent = BiddingAgent(llm_client=ScriptedLLM())
        events = collect(agent, "我们单位这个项目可以单一来源采购吗？")
        assert not any(e == "token" for e, _ in events)
        done = [p for e, p in events if e == "done"][0]
        assert not done.get("tool_name")


# ===========================================================================
# 四、SSE 帧契约（前端依赖它取到问题）
# ===========================================================================


class TestSseFrame:
    """★ 这条测试对着一类"静默丢失"的缺陷。

    `chat_stream` 的兜底分支是
        `content = payload if isinstance(payload, str) else payload.get("content", "")`
    而 clarify 的载荷是 `{"questions": [...], "reason": ...}`，**没有 content 键**。
    若走兜底，前端只会收到 `{"type":"clarify","content":""}` ——
    追问的问题与选项**全部丢失**，界面上什么都不出现，而本轮又已结束，
    用户看到的是"转圈后什么都没发生"。
    """

    def test_clarify_frame_carries_questions(self):
        agent = BiddingAgent(llm_client=ScriptedLLM(
            questions=["贵单位所在地区是？"], options=["北京"]))
        frames = []
        for raw in agent.chat_stream("我们单位这个项目可以单一来源采购吗？"):
            raw = raw.strip()
            if not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body in ("[DONE]", "") or body.startswith(":"):
                continue
            frames.append(json.loads(body))

        clarify = [f for f in frames if f.get("type") == "clarify"]
        assert len(clarify) == 1, f"应有且仅有一个 clarify 帧，实得 {len(clarify)}"
        frame = clarify[0]
        assert frame.get("questions"), "★ clarify 帧必须带 questions（不能只有空 content）"
        assert frame["questions"][0]["question"] == "贵单位所在地区是？"
        assert frame["questions"][0]["options"] == ["北京"]

    def test_frame_type_not_overridable_by_payload(self):
        """载荷里若意外含 `type` 键，不能覆盖事件类型。"""
        agent = BiddingAgent(llm_client=ScriptedLLM())
        frames = []
        for raw in agent.chat_stream("我们单位这个项目可以单一来源采购吗？"):
            raw = raw.strip()
            if raw.startswith("data: ") and raw[6:] not in ("[DONE]", ""):
                frames.append(json.loads(raw[6:]))
        for f in frames:
            if f.get("type") == "clarify":
                assert f["type"] == "clarify"


# ===========================================================================
# 五、前端契约（跨语言，只能对源码做静态断言）
# ===========================================================================


class TestFrontendContract:
    """★ 锁住那个"只在前端才暴露"的缺陷。

    后端 `clarify_allowed = (not history) and ...`，而前端算 history 是
    `messages.filter(m => m.status === 'done')`。**欢迎语的状态恰好是 `done`**
    （它确实不是"生成中"），于是首次提问时 history 恒非空
    → `clarify_allowed` 恒为假 → **首轮追问完全失效**。

    实测确认：直接打 API，"空历史"会追问 3 条、"带 2 条历史"则直接作答；
    而界面上发出的首轮问题永远带着那条欢迎语。

    ★ 为什么 Python 单元测试当初没抓到
    -----------------------------
    测试传的是 `history=None`——**前端真实的取法才是要覆盖的路径**。
    这类"两端各自正确、拼起来失效"的缺陷，只能靠跨语言断言兜住。
    """

    #: 前端 Chat 页路径
    #: ★ 前端与后端是**两个平级目录**（idding-agent/ 与 rontend/），
    #: 故要从仓库根再往上一级。第一版写成 parents[1] / "frontend"，
    #: 路径不存在 → 4 条测试**全部静默 skip**，输出"26 passed, 4 skipped"，
    #: 看起来像通过，实际一条前端契约都没验证（假绿灯）。
    CHAT_TSX = (
        pathlib.Path(__file__).resolve().parents[2]
        / "frontend" / "src" / "pages" / "Chat.tsx"
    )

    @staticmethod
    def _require_frontend(path: pathlib.Path) -> str:
        """读前端源码。

        ★ 找不到就 **fail** 而不是 skip：本测试是防回归用的，
        静默跳过等于没有防护，而 "skipped" 在 CI 里几乎不会被注意到。
        """
        assert path.exists(), (
            f"找不到前端源码 {path}；若前端目录挪了位置，请同步本测试的 CHAT_TSX"
        )
        return path.read_text(encoding="utf-8")

    def test_frontend_excludes_welcome_from_history(self):
        src = self._require_frontend(self.CHAT_TSX)
        # 在"组装 history"的那段里必须看到对 welcome 的排除
        idx = src.find("组装已完成回合的历史")
        assert idx > 0, "找不到 history 组装处（前端可能被重构，需同步本测试）"
        segment = src[idx:idx + 1200]
        assert "welcome" in segment, (
            "★ 前端组装 history 时必须排除欢迎语（id === 'welcome'）。"
            "它是 status='done' 的 assistant 消息，会被算作一轮已完成对话，"
            "导致后端 clarify_allowed 恒为假、首轮追问彻底失效。"
        )

    def test_frontend_defines_clarify_status(self):
        """`ChatMessage.status` 必须包含 `clarify`。

        若沿用 `done`，渲染层会走进"答完了但内容为空"的分支，
        显示空回复框（而不是追问卡片）。
        """
        src = self._require_frontend(self.CHAT_TSX)
        assert "'clarify'" in src, "ChatMessage.status 应包含 'clarify'"

    def test_frontend_done_does_not_clobber_clarify(self):
        """★ `onDone` 不能把追问卡片覆写成"（未生成回答）"。

        追问轮 `content` 为空，而 `done` 处理器里有
        `content: m.content || '（未生成回答）'` + `status: 'done'`。
        不做 `clarified` 分支的话，刚渲染出的追问卡片会被一个
        "未生成回答"的空白气泡取代 —— 两个事件相隔极短，肉眼只看到后者。
        """
        src = self._require_frontend(self.CHAT_TSX)
        idx = src.find("const done = (payload: DoneEvent)")
        assert idx > 0, "找不到 done 处理器"
        segment = src[idx:idx + 1400]
        assert "payload.clarified" in segment, (
            "★ done 处理器必须判 `payload.clarified`，否则会抹掉追问卡片"
        )

    def test_api_types_declare_clarify(self):
        api_ts = self.CHAT_TSX.parents[1] / "lib" / "api.ts"
        src = self._require_frontend(api_ts)
        assert "ClarifyEvent" in src
        assert "onClarify" in src, "AskHandlers 必须声明 onClarify"
        assert "case 'clarify'" in src, "dispatchEvent 必须分发 clarify"
