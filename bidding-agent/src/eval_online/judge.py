"""逐轮 LLM 裁判（RAGAS 风格 faithfulness / relevancy）。

★ 与离线 `eval/ragas_eval.py` 的分工
============================
    `eval/ragas_eval.py`  → 用 **ragas 库**跑全量数据集，批量、可复现、慢（实测一次 83 分钟）
    本模块（在线）        → 单轮轻量判分，只为看板提供"这一轮答得好不好"的信号

★ 为什么不在 API 进程里 import ragas
============================
ragas 会连带拉入 datasets / sentence-transformers 等重依赖，
在 Web 进程里导入会显著拖慢启动并常驻大量内存。
故这里**自己写一次判分调用**（一次 LLM、JSON 输出、强制 0~1 分），
指标口径对齐 RAGAS，但不宣称"用的就是 ragas 库"。

★ 隐私约束：文本只在内存过一遍
====================
在线样本**落盘时不存任何文本**（只有 `question_chars` 这类长度）。
本模块拿到的 question/answer/contexts 只存在于**内存队列**里，
判分完成后由调用方丢弃；写盘的只有 `{sample_id, faithfulness, relevancy, ...}`。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: 裁判模型（可用环境变量覆盖）。默认沿用离线 RAGAS 的裁判模型。
ENV_JUDGE_MODEL = "RAGAS_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "glm-5.3-flash"

#: 单次判分的超时（秒）。超时就记为失败，不拖住队列。
JUDGE_TIMEOUT_SECONDS = 60.0

#: 上下文送裁判时的截断长度（字符）。太长会显著增加 token 与延迟。
MAX_CONTEXT_CHARS = 4000
MAX_ANSWER_CHARS = 2000

_PROMPT = """你是招投标采购领域的答案质量评审。请只输出一行 JSON，不要任何解释。

【用户问题】
{question}

【参考材料（检索结果，可能不完整）】
{contexts}

【系统给出的回答】
{answer}

请按下面两个维度打分（0.0~1.0 的两位小数）：

1. faithfulness（忠实度）：回答中的事实性内容有多少能被【参考材料】支持。
   - 1.0 = 全部有据可依
   - 0.5 = 部分可支持，另有部分材料里找不到
   - 0.0 = 回答与材料无关，或明显是材料外的编造
   - 若材料为空而回答给出了具体条款/数字，应给低分。

2. relevancy（相关性）：回答是否真正在回应用户问题（而不是答了别的）。
   - 1.0 = 直接回答
   - 0.5 = 沾边但没答到点上
   - 0.0 = 答非所问

只输出 JSON，格式：{{"faithfulness": 0.00, "relevancy": 0.00}}"""


def judge_model_name() -> str:
    return (os.getenv(ENV_JUDGE_MODEL, "") or "").strip() or DEFAULT_JUDGE_MODEL


def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit] + "\n…（已截断）"


def _parse_scores(text: str) -> tuple[float, float] | None:
    """从模型输出里抠出两个 0~1 分数。抠不到返回 None。"""
    if not text:
        return None
    # 优先取 JSON
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            f = float(obj.get("faithfulness"))
            r = float(obj.get("relevancy"))
            return (_clamp(f), _clamp(r))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    # 兜底：两个 "key: 0.xx"
    fs = re.findall(r"faithfulness\D{0,12}([0-9]*\.?[0-9]+)", text, re.I)
    rs = re.findall(r"relevancy\D{0,12}([0-9]*\.?[0-9]+)", text, re.I)
    if fs and rs:
        try:
            return (_clamp(float(fs[0])), _clamp(float(rs[0])))
        except ValueError:
            return None
    return None


def _clamp(v: float) -> float:
    if v != v:          # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def _build_contexts(sources: Any) -> str:
    """把 `done["sources"]` 拼成裁判可读的参考材料。

    每个 source 形如 `{question, answer, relevance}` —— 那是知识库里检索到的
    问答对，作为"参考材料"喂给裁判。
    """
    if not isinstance(sources, (list, tuple)):
        return ""
    parts: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        a = str(s.get("answer") or "").strip()
        if not a:
            continue
        q = str(s.get("question") or "").strip()
        parts.append(f"- {q}\n  {a}" if q else f"- {a}")
    return _truncate("\n".join(parts), MAX_CONTEXT_CHARS)


def judge_turn(
    *,
    question: str,
    answer: str,
    sources: Any = None,
    client: Any = None,
) -> dict[str, Any]:
    """对一轮问答打分。**永不抛异常**。

    返回 `{"ok": bool, "faithfulness": float|None, "relevancy": float|None,
           "judge_model": str, "error": str}`
    """
    out: dict[str, Any] = {
        "ok": False, "faithfulness": None, "relevancy": None,
        "judge_model": judge_model_name(), "error": "",
    }
    try:
        q = (question or "").strip()
        a = (answer or "").strip()
        if len(q) < 4 or len(a) < 8:
            out["error"] = "问题或回答过短，跳过判分"
            return out
        ctx = _build_contexts(sources)
        prompt = _PROMPT.format(
            question=_truncate(q, 800),
            contexts=ctx or "（本轮没有检索到任何材料）",
            answer=_truncate(a, MAX_ANSWER_CHARS),
        )
        text, real_model = _call_llm(prompt, client=client)
        # ★ 标签必须反映**实际跑的模型**，不能用"期望的"模型名
        #
        # 踩过：`DEFAULT_JUDGE_MODEL = "glm-5.3-flash"` 而 provider 是 deepseek，
        # 于是记录里写着 glm、实际是 deepseek —— 评测数据的模型标签错，
        # 比没有标签更糟。故一律以客户端实例上的真实 model 为准。
        out["judge_model"] = real_model or out["judge_model"]
        scores = _parse_scores(text)
        if scores is None:
            out["error"] = f"裁判输出无法解析：{text[:120]}"
            return out
        out.update(ok=True, faithfulness=scores[0], relevancy=scores[1])
        return out
    except Exception as exc:  # noqa: BLE001 - 判分是旁路，绝不外抛
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _call_llm(prompt: str, *, client: Any = None) -> str:
    """调用裁判模型。复用项目统一的 LLM 客户端工厂。

    ★ 签名必须对齐（我第一版写错了，这里记下来免得再错）
    --------------------------------------------------
        `get_llm_client(provider: str)` 收的是 **provider**（deepseek/zhipu/…），
          **不是** model 名；
        `chat_stream(messages: list[dict], **kwargs)` 收的是**消息列表**，
          **不是**裸字符串。

    模型名通过 `RAGAS_JUDGE_MODEL` 覆盖时，由 `_judge_provider()` 决定走哪家
    （模型名只影响该家请求体里的 `model` 字段，由客户端自己读配置；
    这里不强行改 model，避免与客户端内部校验冲突）。

    用**独立客户端实例**，不复用 Agent 的（避免污染主链路的 provider 状态与降级冷却）。
    """
    if client is None:
        from src.clients.llm_factory import get_llm_client  # 延迟 import

        client = get_llm_client(_judge_provider())

    # 真实模型名：客户端上优先 `model`，退而 `_model`，再不行给空
    real_model = ""
    for attr in ("model", "_model", "model_name"):
        val = getattr(client, attr, None)
        if isinstance(val, str) and val.strip():
            real_model = val.strip()
            break

    messages = [{"role": "user", "content": prompt}]
    chunks: list[str] = []
    deadline_hit = threading.Event()

    timer = threading.Timer(JUDGE_TIMEOUT_SECONDS, deadline_hit.set)
    timer.daemon = True
    timer.start()
    try:
        for piece in client.chat_stream(messages):
            if deadline_hit.is_set():
                logger.debug("在线裁判超时（已收集 %d 片段）", len(chunks))
                break
            if isinstance(piece, str):
                chunks.append(piece)
    finally:
        timer.cancel()
    return "".join(chunks).strip(), real_model


#: 环境变量：裁判走哪家 provider（默认 deepseek，与主链路同源以便复用凭据）
ENV_JUDGE_PROVIDER = "RAGAS_JUDGE_PROVIDER"
DEFAULT_JUDGE_PROVIDER = "deepseek"


def _judge_provider() -> str:
    return (os.getenv(ENV_JUDGE_PROVIDER, "") or "").strip() or DEFAULT_JUDGE_PROVIDER


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_PROVIDER",
    "ENV_JUDGE_MODEL",
    "ENV_JUDGE_PROVIDER",
    "judge_model_name",
    "judge_turn",
]
