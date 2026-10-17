"""历史消息定期压缩（滚动摘要）。

★ 要解决的缺口：现有实现只**硬截断**，更早的信息永久丢失
----------------------------------------------------
`_truncate_history` 只保留最近 `MAX_HISTORY_ROUNDS`（5）轮，超出部分**直接丢弃**。
后果是一个真实且常见的场景：

    第 1 轮：我在跟"华中电力建设公司"的中标情况
    第 3 轮：那它 2024 年呢？
    第 9 轮：刚才那家公司再查一下省份分布      ← 第 1 轮已被丢掉
             模型不知道"刚才那家"是谁

截断是**静默的**——模型不会说"我忘了前文"，它会猜或反问，用户只觉得"变笨了"。

★ 设计：滚动摘要（不是每轮都压）
-----------------------------
朴素做法是"每轮把全部历史压成摘要"，问题是**摘要会反复被摘要**，
信息层层衰减（第 10 轮时第 1 轮的内容已被压了 10 次）。

故采用**滚动**策略：把"最旧的一批轮次"压成一段摘要，它固定下来，
后续只在需要时把**更新的**旧轮次并入摘要。压缩频率随会话长度下降：

    轮次  1……8        9        10……16      17      18……
    动作  不压      压一次     不压        再压    不压

实现上用一个"已覆盖到第几条"的水位线（`covered`）：
只有当未覆盖的历史超过 `COMPACT_TRIGGER_ROUNDS` 轮时才重新压缩，
且每次把水位线推进到"保留窗口之前"。

★ 为什么用 LLM 做摘要，而不是规则截取
----------------------------------
规则截取（如"取每轮前 50 字"）会把关键约束切碎：
「投标保证金**不得超过 2%**」截成「投标保证金不得超过」就失去了结论。
摘要需要理解哪些是**需要带走的**：实体名、用户约束、已确认结论、未决问题。

★ 摘要提示词的设计要点
-------------------
    1. **只保留会跨越后续轮次还有用的**：实体、约束、结论、未决项
    2. **明确丢弃**：寒暄、工具调用细节、被推翻的中间结论
    3. **不臆造**：原文没说的不要补
    4. 短——摘要是为了省 token，写成一大段就失去意义
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from src.logging_config import get_logger

logger = get_logger(__name__)

#: 触发压缩的历史轮数。低于它不压——短对话硬截断已够，压缩是多一次 LLM 调用。
COMPACT_TRIGGER_ROUNDS = 9

#: 压缩时**保留不压**的最近轮数。
#:
#: 为什么留 4 轮而不是留 1 轮：最近几轮是当前话题的上下文，
#: 摘要成一段话会丢掉细节（如"那 2024 年呢"依赖的上一轮具体问法）。
#: 留 4 轮既能保住近期细节，又能让摘要覆盖更早的内容。
COMPACT_KEEP_RECENT_ROUNDS = 4

#: 摘要文本长度上限（字符）。摘要是为了省 token，不设上限就失去意义。
MAX_SUMMARY_CHARS = 800

#: 单轮内容截断长度：压缩输入时，超长回答只取前这么多字。
#:
#: 为什么需要：历史回答实测平均 480 字，10 轮就是 4800 字，
#: 整份塞进摘要请求会吃掉大量 token。取前 400 字保留结论部分
#: （招投标回答的结论通常在前半段），截掉的细节本就不是摘要要保留的。
COMPACT_INPUT_TRUNCATE = 400

COMPACT_PROMPT = """你是对话摘要助手。

请把下面这段**早期对话**压缩成一段简短摘要，供后续对话继续使用。

必须保留（若原文有）：
- 提到的具体实体名（公司、项目、地区、法规名称）
- 用户的约束与偏好（如"只看安徽的""要 2024 年之后的"）
- 已确认的结论与数据
- 尚未解决的问题

必须丢弃：
- 寒暄与客套
- 工具调用的过程细节（只留结论）
- 被后续推翻的中间结论

要求：
1. 不要臆造原文没有的内容。
2. 用中文，300 字以内，直接给摘要正文，不要任何前后缀或标题。

早期对话：
"""


@dataclass
class Compaction:
    """一次压缩的结果。"""

    summary: str = ""
    #: 摘要覆盖到了第几条消息（水位线，用于滚动更新）
    covered: int = 0
    #: 是否真的压缩了（False 表示未达触发条件，或压缩失败）
    compacted: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "compacted": self.compacted,
            "covered": self.covered,
            "summary_len": len(self.summary),
            "reason": self.reason,
        }


@dataclass
class _SessionState:
    """单个会话的压缩状态（进程内）。"""

    summary: str = ""
    covered: int = 0


@dataclass
class _StateStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, _SessionState] = field(default_factory=dict)


_store = _StateStore()

#: 进程内保留的会话数上限。
#:
#: 超限时按插入顺序淘汰最旧的——压缩状态是**可重建的**（丢了就重新压一次），
#: 故不需要持久化，但要防止长跑进程把内存吃光。
MAX_TRACKED_SESSIONS = 500


def _rounds(history: list) -> int:
    """历史里有几轮（一条 user 消息算一轮）。

    按 **user 消息数**而不是消息总条数：历史里可能混入 system/工具消息，
    按总条数算会让"轮数"虚高，压缩触发得过早。
    """
    return sum(
        1 for m in (history or [])
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
    )


def _valid_messages(history: list) -> list[dict]:
    """过滤出可用的对话消息（与 `_truncate_history` 同一口径）。"""
    return [
        m for m in (history or [])
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]


def _compaction_window(history: list, covered: int) -> tuple[int, int]:
    """返回可压缩区间 `[start, keep_from)`；`start >= keep_from` 表示无可压内容。

    抽成独立函数是**为了消除"判断与行为不一致"**（实测踩过）：
    `needs_compaction` 与 `compact_history` 曾各自实现一遍判据，
    结果一个说"该压"、另一个实际不压。现在两者都调这一个函数，
    只有一处口径，不可能漂移。
    """
    messages = _valid_messages(history)
    keep_from = max(0, len(messages) - COMPACT_KEEP_RECENT_ROUNDS * 2)
    start = max(0, covered)
    return start, keep_from


def needs_compaction(history: list, covered: int | None = None,
                     session_id: str = "") -> bool:
    """是否需要（重新）压缩。

    ★ 三个条件都要满足（实测踩坑）
    ----------------------------
        ① 未覆盖的轮数 > `COMPACT_TRIGGER_ROUNDS`
        ② 未覆盖区间**超出保留窗口**（有真正可压的部分）
        ③ 待压缩内容非空

    第一版只看条件 ①，于是 11~14 轮时返回 True 但 `compact_history` 实际不压
    （可压区间为空）——**判断与行为不一致**。这个函数是对外暴露的，
    调用方若据它决定"要不要提示用户正在压缩"，就会出现"说要压缩却没压"。

    修正后仍不一致，因为调用方传 `covered=0`，而 `compact_history` 用的是
    **会话内的真实水位**——两者没在比同一个条件。

    故现在：`covered=None` 时**从会话状态读水位**（与 `compact_history` 完全同源），
    且两者共用 `_compaction_window` 算区间。口径只有一处。
    """
    if covered is None:
        covered = _get_state(session_id).covered if session_id else 0
    messages = _valid_messages(history)
    if len(messages) <= covered:
        return False
    if _rounds(messages[covered:]) <= COMPACT_TRIGGER_ROUNDS:
        return False
    start, keep_from = _compaction_window(history, covered)
    return start < keep_from


def _render_for_prompt(messages: list[dict]) -> str:
    """把待压缩的消息渲染成提示词输入。"""
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        content = str(m.get("content") or "").strip()
        if len(content) > COMPACT_INPUT_TRUNCATE:
            content = content[:COMPACT_INPUT_TRUNCATE] + "…（略）"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def compact_history(
    history: list,
    client,
    *,
    session_id: str = "",
    force: bool = False,
) -> Compaction:
    """按需把最旧的一批轮次压成摘要。**永不抛异常**。

    返回 `Compaction`：`compacted=False` 表示这次没压（未达条件或压缩失败），
    此时调用方应原样使用原有摘要（若有）。

    `client` 需提供 `chat_raw(messages)`；传 None 时直接返回未压缩。

    ★ 为什么失败不抛异常
    ------------------
    压缩是**增益项**（让模型记得更早的内容）。它失败最多退化成现有的硬截断，
    不该让整轮问答失败——那比"记不住前文"严重得多。
    """
    if client is None:
        return Compaction(reason="无可用 LLM 客户端，跳过压缩")

    messages = _valid_messages(history)
    state = _get_state(session_id)
    covered = state.covered if session_id else 0

    if not force and not needs_compaction(messages, covered):
        # 复用已有摘要：这是滚动策略的关键——不满足条件时不重复压缩
        return Compaction(
            summary=state.summary if session_id else "",
            covered=covered,
            compacted=False,
            reason="未达压缩触发条件",
        )

    # 待压缩区间：从头到"保留窗口"之前。与 `needs_compaction` 共用同一函数，
    # 保证"判断该压"与"实际能压"永远一致。
    start, keep_from = _compaction_window(messages, covered)
    if start >= keep_from:
        return Compaction(summary=state.summary, covered=covered,
                          compacted=False, reason="没有新的可压缩区间")

    pending = messages[start:keep_from]
    if not pending:
        return Compaction(summary=state.summary, covered=covered,
                          compacted=False, reason="可压缩区间为空")

    # 滚动并入：已有摘要 + 新增旧轮次，一起重压成一段
    body = _render_for_prompt(pending)
    if state.summary:
        body = f"（已有摘要）\n{state.summary}\n\n（新增的较早对话）\n{body}"

    try:
        raw = client.chat_raw([{"role": "user", "content": COMPACT_PROMPT + body}])
    except Exception:
        logger.warning("历史压缩调用失败，退化为原有摘要/硬截断", exc_info=True)
        return Compaction(summary=state.summary, covered=covered,
                          compacted=False, reason="压缩调用失败")

    summary = str(raw or "").strip()[:MAX_SUMMARY_CHARS]
    if not summary:
        return Compaction(summary=state.summary, covered=covered,
                          compacted=False, reason="压缩返回空摘要")

    if session_id:
        _set_state(session_id, summary, keep_from)
    logger.info(
        "历史已压缩：覆盖前 %d 条消息，摘要 %d 字（会话 %s）",
        keep_from, len(summary), session_id or "<无>",
    )
    return Compaction(summary=summary, covered=keep_from, compacted=True,
                      reason=f"已压缩前 {keep_from} 条消息")


def build_compaction_note(compaction: Compaction) -> str:
    """把摘要渲染成注入 system 的文本。

    措辞要点：标明这是**早期对话的摘要**（不是本轮检索所得，也不是当前问题的答案），
    并提示它可能损失细节——这样模型在需要精确信息时会更倾向重新查证，
    而不是凭摘要里的模糊记忆作答。
    """
    if not compaction.summary:
        return ""
    return (
        "【早期对话摘要】以下是你与用户**更早轮次**对话的摘要"
        "（原始消息已因长度限制移出上下文）：\n"
        f"{compaction.summary}\n"
        "摘要可能损失细节：若当前问题需要摘要里提到的精确数值、条款号或原文，"
        "请用工具重新检索确认，不要仅凭摘要作答。"
    )


# ===========================================================================
# 进程内会话状态
# ===========================================================================


def _get_state(session_id: str) -> _SessionState:
    if not session_id:
        return _SessionState()
    with _store.lock:
        return _store.sessions.get(session_id, _SessionState())


def _set_state(session_id: str, summary: str, covered: int) -> None:
    if not session_id:
        return
    with _store.lock:
        _store.sessions[session_id] = _SessionState(summary=summary, covered=covered)
        while len(_store.sessions) > MAX_TRACKED_SESSIONS:
            # dict 保序：弹出最早插入的
            _store.sessions.pop(next(iter(_store.sessions)))


def reset_session(session_id: str) -> None:
    """清掉某会话的压缩状态（会话删除时调用，避免状态泄漏）。"""
    if not session_id:
        return
    with _store.lock:
        _store.sessions.pop(session_id, None)


def clear_all() -> None:
    """清空全部压缩状态（测试与诊断用）。"""
    with _store.lock:
        _store.sessions.clear()


def compaction_enabled() -> bool:
    """是否启用历史压缩（`AGENT_COMPACT`，默认开启）。

    为什么默认开：它**只在长会话触发**（> 9 轮），短对话零成本、零延迟；
    而对长会话，它补的是"更早内容永久丢失"这个静默缺陷。
    与 `AGENT_MEMORY` 同一取舍逻辑——没有固定成本的能力默认开；
    `AGENT_PLANNING` 那种每轮多一次调用的才默认关。
    """
    raw = (os.getenv("AGENT_COMPACT") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


__all__ = [
    "COMPACT_KEEP_RECENT_ROUNDS",
    "COMPACT_TRIGGER_ROUNDS",
    "MAX_SUMMARY_CHARS",
    "Compaction",
    "build_compaction_note",
    "clear_all",
    "compact_history",
    "compaction_enabled",
    "needs_compaction",
    "reset_session",
]
