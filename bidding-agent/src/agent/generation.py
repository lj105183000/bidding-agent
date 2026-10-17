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
from src.agent.verification import build_grounding_note

logger = logging.getLogger(__name__)

RAG_TOP_K = 5  # RAG 降级路径的检索条数，与 /api/ask 默认值保持一致


class GenerationMixin:

    @staticmethod
    def _inject_grounding(messages: list, meta: dict) -> list:
        """把"本轮工具执行事实"追加到 system 消息末尾，返回新的 messages。

        独立成方法而不是内联在 `_generate_final` 里，是为了**可测**：
        内联时探针只能在 `_generate_final` 之前抓 messages，
        而注入发生在函数内部的局部副本上，外部观察不到——实测因此误判过一次
        "注入没生效"。

        追加到首条 system 的末尾而不是另起一条 system：部分厂商只认首条 system。
        """
        grounding = build_grounding_note(meta.get("succeeded_tools") or [])
        if not grounding:
            return messages
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": str(messages[0].get("content") or "") + "\n\n" + grounding,
            }
        else:
            messages.insert(0, {"role": "system", "content": grounding})
        return messages

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

        # 生成前注入"本轮工具执行事实"：预防模型编造"工具不可用"。
        #
        # 实测缺陷：回答写「联网检索工具本轮不可用」，而 search_web_cn 实际
        # 成功调用 4 次、返回 20/9/16/13 条真实结果，工具结果文本里还明写着
        # 「【数据来源：联网搜索，请注意信息时效】」。用户据此会以为系统坏了。
        #
        # 为什么用注入而不是事后改写：流式回答已逐帧下发，改写要 reset 清空重来，
        # 用户会看到一次闪烁。把事实摆在模型面前是最低成本且对流式友好的做法。
        messages = self._inject_grounding(messages, meta)

        # ★ 走思考流还是提示词注入：看**客户端开关状态**与**协议能力**
        # -----------------------------------------------------
        # 判定依据是"本轮客户端有没有被开启思考模式"，而不是模型名比较。
        #
        # 为什么改成这样：DeepSeek 发布 V4 后取消了独立推理模型名，思考模式由
        # 请求参数 `{"thinking": {"type": "enabled"}}` 控制（见
        # `BaseLLMClient.set_thinking` 的说明）。同一个模型名既可以有思维链
        # 也可以没有，"模型名"不再是判定依据——`core._thinking_model` 只对
        # **开了深度思考的请求**把开关打开（在浅拷贝上，不影响并发请求）。
        #
        # 历史缺陷回顾：原实现是 `hasattr(client, "chat_stream_thinking")`，
        # 而基类实现了该方法（只为统一接口），于是判定**恒为真**、
        # 回退分支永不执行 —— 没配推理模型时开关打开后既没有思维链
        # （普通模型无 reasoning_content）也没有分析指令，等于什么都没做。
        enabled = bool(getattr(client, "thinking_enabled", lambda: False)())
        capable = bool(getattr(client, "supports_reasoning", lambda: False)())
        use_thinking = deep_thinking and enabled and capable

        if deep_thinking and not use_thinking:
            reason = (
                "未配置思考模式（或客户端不支持）" if not enabled
                else "客户端不支持思维链"
            )
            logger.info("深度思考回退提示词注入（%s）", reason)
            messages = list(messages)
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n" + DEEP_THINKING_INJECT,
            }

        # 统一流格式：chat_stream 产出纯字符串，chat_stream_thinking 产出 (kind, 文本)，
        # 此处归一为 (kind, 文本)，kind 为 None 表示普通正文
        #
        # ★ `max_tokens` 两条路径都要传
        # -------------------------
        # `OpenAICompatClient._payload()` 默认**不带** `max_tokens`，即使用厂商
        # 默认值。而深度思考的输出明显更长（要有拆解、多角度分析、结论），
        # 厂商默认常常不够，会出现"分析到一半被截断"。
        #
        # 原实现只给推理模型那条路径传了 `max_tokens=MAX_TOKENS_DEEP`，而
        # **提示词注入这条路径完全没传** —— 恰恰是最需要放宽的那条（模型没有
        # 原生思维链，全部内容都算在正文预算里）。故两条路径都传。
        if use_thinking:
            stream = client.chat_stream_thinking(messages, max_tokens=MAX_TOKENS_DEEP)
        else:
            stream = (
                (None, chunk)
                for chunk in client.chat_stream(
                    messages,
                    **({"max_tokens": MAX_TOKENS_DEEP} if deep_thinking else {}),
                )
            )

        # ★ 归一化：三种产出形态都要支持（这里曾有一个静默吞掉思维链的 bug）
        # --------------------------------------------------------------
        #     ① `{"type": "thinking"|"content", "content": ...}` 字典
        #        —— `chat_stream_thinking` 的形态（链展开后就是字典）
        #     ② `(kind, chunk)` 元组 —— 早期的归一化形态，测试替身与第三方实现可能用
        #     ③ 纯字符串 —— `chat_stream` 的形态
        #
        # 原实现写的是 `for kind, chunk in stream:`。对**字典**迭代得到的是**键**：
        #
        #     for kind, chunk in {"type": "thinking", "content": "We"}:
        #         第1轮 kind='type'     chunk='thinking'
        #         第2轮 kind='content'  chunk='We'
        #
        # 于是 `kind == "thinking"` **永远不成立**：思维链的**内容字符串**
        # （'thinking' / 'We' / …）被当成正文下发，而 `thinking` 事件恒为 0
        # ——前端折叠块永远空白，同时正文里混进思维链。
        #
        # 实测证据：在循环里打印 kind 的取值集合只得到 `{'type': 713}`。
        #
        # 注意别用 `isinstance(chunk, "dict")` 这种写法：字符串形式的注解会让
        # 判断永远为假，反而把纯字符串送进 `.get()` 而抛 AttributeError
        # （实测踩过，导致 13 个既有测试失败）。
        def _normalize(item):
            if isinstance(item, dict):
                return item.get("type"), item.get("content", "")
            if isinstance(item, str):
                return None, item
            try:
                kind, chunk = item
            except (TypeError, ValueError):
                return None, str(item)
            return kind, chunk

        pending = ""
        for _item in stream:
            kind, chunk = _normalize(_item)
            if kind == "thinking":
                yield "thinking", chunk or ""
                continue
            pending += chunk
            if self._detect_tool_text(pending):
                calls = self._parse_tool_text(pending)
                if calls:
                    # 确认泄漏：reset 清空已流正文 → 兜底执行 → 重新生成
                    logger.warning("最终生成检出工具调用文本泄漏，触发 reset 与兜底执行")
                    yield "reset", None
                    yield from self._reset_and_fallback(pending, client, meta)
                    self._mark_phase(meta, "生成回答", t0)
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

        self._mark_phase(meta, "生成回答", t0)

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
        # ★ 降级路径也要采集 RAG 耗时（实测发现的缺口）
        #
        # 原实现只调 `pipeline.search()` 而不读计时，于是**走降级时
        # `done` 帧的 `rag_timing` 恒为 null** —— 而"Agent 失败了、退回纯 RAG"
        # 恰恰是最需要看检索性能的时候（用户会问"为什么这次这么慢"）。
        #
        # 计时本身由 `search()` 写在单例的 `last_timing` 上（见 `src/rag/timing.py`
        # 里"为什么必须从单例读"的说明），故这里只需取回并塞进 meta。
        from src.rag.timing import reset_marks, take_marks

        reset_marks()   # 先清，避免读到上一次的旧值
        try:
            sources = pipeline.search(question, top_k=RAG_TOP_K)
        except Exception as exc:
            # 完整异常只进日志；用户可见提示不泄露内部细节
            logger.warning("RAG 降级检索失败: %s", exc)
            yield "error", {"content": "知识库检索失败，请稍后重试。"}
            return
        _rag = take_marks()
        if _rag is not None:
            meta.setdefault("rag_timings", []).append(_rag)
        if not sources:
            yield "error", {"content": "知识库中暂未找到相关内容，请换个问法试试。"}
            return
        meta["sources"] = self._merge_sources(meta["sources"], sources)
        self._mark_phase(meta, "检索", t0)

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
        self._mark_phase(meta, "生成", t1)