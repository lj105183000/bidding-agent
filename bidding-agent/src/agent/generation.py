"""最终生成（GenerationMixin）：流式 / 深度思考 / RAG 降级。

职责边界：ReActMixin 负责工具循环与决策，本 mixin 负责「最后一公里」——
把最终回答流式发给用户（含防泄漏缓冲），以及 Agent 整体失败时降级为直接 RAG 问答。
done 事件由 core 统一收尾产出，本模块不产出 done。
"""

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from src.agent.constants import MAX_TOKENS_DEEP, STREAM_FLUSH_CHARS
from src.agent.prompts import DEEP_THINKING_INJECT, SYSTEM_PROMPT
from src.agent.utils import _pace_stream_chunks

logger = logging.getLogger(__name__)

RAG_TOP_K = 5  # RAG 降级路径的检索条数，与 /api/ask 默认值保持一致


class GenerationMixin:

    if TYPE_CHECKING:
        # 仅供 IDE/类型检查解析：这些方法分别来自 ToolDefenseMixin / ReActMixin / BiddingAgent，
        # 由 BiddingAgent 组合各 mixin 后才真正存在，运行时此处不产生任何属性
        def _detect_tool_text(self, text: str) -> bool: ...
        def _parse_tool_text(self, text: str) -> list: ...
        def _fallback_execute(self, leaked_text: str, meta: dict) -> tuple: ...
        def _merge_sources(self, existing: list, new: list) -> list: ...
        def _get_llm_client(self, provider: str = ""): ...
        def _get_rag_pipeline(self): ...

    def _generate_final(self, messages: list, client, meta: dict,
                        deep_thinking: bool = False) -> Iterator[tuple]:
        """最终流式生成。产出 "status"|"token"|"thinking" 事件，不产出 done。

        深度思考：客户端实现 chat_stream_thinking 时走思考流（kind ∈ thinking/content），
        否则回退提示词注入（把推理引导追加进 system 消息）。

        防泄漏缓冲：token 先攒入 pending，攒满 STREAM_FLUSH_CHARS 且未检出
        工具调用格式才切帧下发；一旦确认泄漏，立即发 reset 清空已流正文，
        走兜底执行后重新生成。
        """
        yield "status", "正在生成回答..."
        t0 = time.monotonic()

        use_thinking = deep_thinking and hasattr(client, "chat_stream_thinking")
        if deep_thinking and not use_thinking:
            # 回退提示词注入：引导模型先推理再作答（推理不单独成流）
            messages = list(messages)
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n" + DEEP_THINKING_INJECT,
            }

        # 统一流格式：chat_stream 产出纯字符串，chat_stream_thinking 产出 (kind, 文本)，
        # 此处归一为 (kind, 文本)，kind 为 None 表示普通正文
        if use_thinking:
            stream = client.chat_stream_thinking(messages, max_tokens=MAX_TOKENS_DEEP)
        else:
            stream = ((None, chunk) for chunk in client.chat_stream(messages))

        pending = ""
        for kind, chunk in stream:
            if kind == "thinking":
                yield "thinking", chunk
                continue
            pending += chunk
            if self._detect_tool_text(pending):
                calls = self._parse_tool_text(pending)
                if calls:
                    # 确认泄漏：reset 清空已流正文 → 兜底执行 → 重新生成
                    logger.warning("最终生成检出工具调用文本泄漏，触发 reset 与兜底执行")
                    yield "reset", None
                    yield from self._reset_and_fallback(pending, client, meta)
                    meta["phase_times"].append(
                        ("生成回答", int((time.monotonic() - t0) * 1000)))
                    return
                if len(pending) >= STREAM_FLUSH_CHARS * 4:
                    # 疑似片段迟迟解析不出完整调用：按普通文本下发，避免吞掉正常回答
                    for small in _pace_stream_chunks(pending):
                        meta["streamed"] = True
                        yield "token", small
                    pending = ""
                continue
            if len(pending) >= STREAM_FLUSH_CHARS:
                for small in _pace_stream_chunks(pending):
                    meta["streamed"] = True
                    yield "token", small
                pending = ""

        if pending:
            if self._parse_tool_text(pending):
                # 流结束时缓冲里仍是完整调用：同样 reset + 兜底
                logger.warning("流结束时检出工具调用文本泄漏，触发 reset 与兜底执行")
                yield "reset", None
                yield from self._reset_and_fallback(pending, client, meta)
            else:
                for small in _pace_stream_chunks(pending):
                    meta["streamed"] = True
                    yield "token", small

        meta["phase_times"].append(("生成回答", int((time.monotonic() - t0) * 1000)))

    def _reset_and_fallback(self, leaked_text: str, client, meta: dict) -> Iterator[tuple]:
        """泄漏后的兜底流程：执行解析出的工具 → 用结果重新生成一次（非流式转小帧下发）。"""
        result_text, calls = self._fallback_execute(leaked_text, meta)
        if not calls:
            return
        base = meta.get("last_stream_messages") or [{"role": "system", "content": SYSTEM_PROMPT}]
        retry_messages = [
            base[0],
            {
                "role": "user",
                "content": (
                    f"你刚才的回答里混入了工具调用格式的文本，已为你执行该工具。工具结果：\n"
                    f"{result_text}\n\n"
                    "请基于以上信息直接输出最终中文回答，不要再输出任何工具调用格式。"
                ),
            },
        ]
        try:
            answer = client.chat_raw(retry_messages)
        except Exception as exc:
            logger.warning("兜底重生成失败: %s", exc)
            answer = ""
        if self._detect_tool_text(answer or ""):
            answer = ""
        answer = (answer or "").strip() or "抱歉，本次回答生成失败，请重试。"
        for small in _pace_stream_chunks(answer):
            meta["streamed"] = True
            yield "token", small

    def _rag_fallback_events(self, question: str, meta: dict) -> Iterator[tuple]:
        """Agent/LLM 整体失败时的 RAG 降级路径：直接检索知识库并生成。

        status 序列：正在检索知识库... → 正在生成回答...（见开发文档 §6.3）。
        知识库也失败时产出 error 事件，由 core 决定是否收尾。
        """
        yield "status", "正在检索知识库..."
        t0 = time.monotonic()

        pipeline = self._get_rag_pipeline()
        if pipeline is None:
            yield "error", {"content": "知识库暂时不可用，请稍后重试。"}
            return
        try:
            sources = pipeline.search(question, top_k=RAG_TOP_K)
        except Exception as exc:
            # 完整异常只进日志；用户可见提示不泄露内部细节
            logger.warning("RAG 降级检索失败: %s", exc)
            yield "error", {"content": "知识库检索失败，请稍后重试。"}
            return
        if not sources:
            yield "error", {"content": "知识库中暂未找到相关内容，请换个问法试试。"}
            return
        meta["sources"] = self._merge_sources(meta["sources"], sources)
        meta["phase_times"].append(("检索", int((time.monotonic() - t0) * 1000)))

        yield "status", "正在生成回答..."
        t1 = time.monotonic()
        context = "\n\n".join(
            f"问：{s.get('question', '')}\n答：{s.get('answer', '')}" for s in sources
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考以下知识库内容回答问题。\n\n{context}\n\n问题：{question}"},
        ]
        try:
            for chunk in self._get_llm_client().chat_stream(messages):
                for small in _pace_stream_chunks(chunk):
                    meta["streamed"] = True
                    yield "token", small
        except Exception as exc:
            logger.warning("RAG 降级生成失败，直接拼接来源: %s", exc)
            answer = "\n\n".join(
                f"问：{s.get('question', '')}\n答：{s.get('answer', '')}" for s in sources
            )
            for small in _pace_stream_chunks(answer):
                meta["streamed"] = True
                yield "token", small
        meta["phase_times"].append(("生成", int((time.monotonic() - t1) * 1000)))