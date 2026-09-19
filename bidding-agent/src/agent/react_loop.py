"""ReAct 多轮工具循环（ReActMixin）。

事件流架构：mixin 方法均为 generator，产出事件元组——
  "status"|"token"|"thinking", 文本 / "reset", None / "done"|"error", dict
core._chat_events 统一调度，chat / chat_stream 两条路径共用同一流程。

工具执行契约（冻结，见 分配说明.md §5）：TOOL_EXECUTORS[name] 接收一个
arguments dict，返回 (格式化文本, sources 列表)；任何失败都返回结构化
错误文本而非抛异常，Agent 永不因后端宕机崩溃。

注意：本模块与 src.agent.core 各持有一份 TOOL_EXECUTORS 模块级绑定
（此处用于文本兜底过滤），测试替换时必须同时替换两处。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING
from collections.abc import Iterator

from src.agent.constants import (
    MAX_TOOL_ROUNDS,
    TOOL_TIMEOUT_SECONDS,
    WEB_TOOL_NAMES,
)

try:
    # 工具层（成员 C）尚未合入时以空集运行，测试经 monkeypatch 注入
    from src.tools.rag_tools import TOOL_EXECUTORS
except ImportError:  # pragma: no cover - 阶段一工具层未落地
    TOOL_EXECUTORS = {}

logger = logging.getLogger(__name__)

# 约束重试提示：决策轮检出畸形工具文本时，要求模型重新输出
_TOOL_CONSTRAINT = (
    "你刚才的输出中混入了工具调用格式的文本。"
    "请重新输出：要么只输出工具调用，要么直接给出回答正文，不要混合。"
)
# 最终生成约束：追加在消息末尾，防止模型复读工具调用过程或输出函数格式
_FINAL_CONSTRAINT = (
    "请基于以上工具结果直接输出最终中文回答，"
    "不要输出任何函数调用格式，不要复述你的分析过程。"
)


class ReActMixin:
    """ReAct 多轮工具循环：决策（chat_raw）→ 解析调用 → 执行 → 结果回填，
    直到模型不再调用工具或达到 MAX_TOOL_ROUNDS 上限，转入最终流式生成。"""

    if TYPE_CHECKING:
        # 仅供 IDE/类型检查解析：这些方法分别来自 ToolDefenseMixin / GenerationMixin，
        # 由 BiddingAgent 组合各 mixin 后才真正存在，运行时此处不产生任何属性
        def _detect_tool_text(self, text: str) -> bool: ...
        def _parse_tool_text(self, text: str) -> list: ...
        def _generate_final(self, messages: list, client, meta: dict,
                            deep_thinking: bool = False) -> Iterator[tuple]: ...

    # --- 工具执行 ---

    @staticmethod
    def _execute_tool(name: str, arguments: dict):
        """执行单个工具。返回 (文本, sources, ok)；失败返回结构化错误文本。"""
        executor = TOOL_EXECUTORS.get(name)
        if executor is None:
            return f"工具 {name} 暂不可用（未注册或后端未就绪）。", [], False

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
        try:
            future = pool.submit(lambda: executor(arguments or {}))
            result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            logger.warning("工具 %s 执行超时（%ss）", name, TOOL_TIMEOUT_SECONDS)
            return "工具执行超时，请稍后重试或缩小查询范围。", [], False
        except Exception as exc:
            logger.warning("工具 %s 执行失败: %s", name, exc)
            return "工具执行失败，已跳过该数据源，请结合其他信息回答。", [], False
        finally:
            # 不等待超时任务：该线程随后自行结束，不阻塞回答流程
            pool.shutdown(wait=False)

        if isinstance(result, tuple) and len(result) == 2:
            text, sources = result
        else:
            # 执行器未按契约返回二元组：降级为纯文本，不丢弃内容
            text, sources = result, []
        if not isinstance(sources, list):
            sources = []
        return str(text), sources, True

    @staticmethod
    def _validate_result(name: str, text: str, ok: bool) -> str:
        """按工具类型附加质量标签，帮助用户判断数据可信度。新增工具在此补分支。"""
        if not ok:
            return text
        if name == "search_knowledge_graph":
            return text + "\n\n【数据来源：知识图谱】"
        if name == "query_database":
            return text + "\n\n【数据来源：业务数据库】"
        if name in WEB_TOOL_NAMES:
            return text + "\n\n【数据来源：联网搜索，请注意信息时效】"
        return text

    @staticmethod
    def _merge_sources(existing: list, new: list) -> list:
        """跨轮来源合并去重：按 (question, answer, url) 组合键判断重复。"""
        seen = {
            (s.get("question"), s.get("answer"), s.get("url"))
            for s in existing
            if isinstance(s, dict)
        }
        merged = list(existing)
        for source in new or []:
            if not isinstance(source, dict):
                continue
            key = (source.get("question"), source.get("answer"), source.get("url"))
            if key not in seen:
                seen.add(key)
                merged.append(source)
        return merged

    def _record_tool_meta(self, name: str, sources: list, meta: dict):
        """工具执行成功后回填 done 帧元数据。tool_name 恒为最后一个执行完成的工具（单字符串）。"""
        meta["tool_called"] = True
        meta["tool_name"] = name
        if name in WEB_TOOL_NAMES:
            meta["web_sources"] = self._merge_sources(meta["web_sources"], sources)
        else:
            meta["sources"] = self._merge_sources(meta["sources"], sources)

    # --- 最终生成准备 ---

    @staticmethod
    def _clean_for_final(messages: list) -> list:
        """构造最终生成消息：system 必须在最前（messages[0]），末尾追加约束消息。

        约束消息的位置是测试断言点：最后一条必须是 user 角色的约束指令。
        """
        cleaned = [messages[0]] + [
            m for m in messages[1:] if m.get("role") != "system"
        ]
        cleaned.append({"role": "user", "content": _FINAL_CONSTRAINT})
        return cleaned

    def _fallback_execute(self, leaked_text: str, meta: dict):
        """文本兜底执行：从泄漏文本解析工具调用并执行（流式轮检出的调用走这里）。

        只执行第一个可解析调用，避免模型一次泄漏多个调用时放大后端压力。
        返回 (工具结果文本, calls)。
        """
        calls = self._parse_tool_text(leaked_text)
        if not calls:
            return "", []
        name = calls[0]["name"]
        text, sources, ok = self._execute_tool(name, calls[0]["arguments"])
        text = self._validate_result(name, text, ok)
        if ok:
            self._record_tool_meta(name, sources, meta)
        return f"[工具 {name} 结果]\n{text}", calls

    # --- 主循环 ---

    def _chat_stream_tools(self, messages: list, client, meta: dict, deep_thinking: bool = False):
        """ReAct 多轮循环主体。产出 status 事件，最终转入 _generate_final 产出 token 事件。

        每轮：chat_raw 决策 → 解析调用 → 无调用则进入最终生成 / 有调用则执行并回填结果。
        决策轮检出畸形工具文本时做一次约束重试；仍无法解析则视为无调用。
        """
        conversation = list(messages)
        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            # 状态序列对齐开发文档 §6.3：
            # 检索与搜索 → 分析检索结果 → 补充检索（第N轮）→ ... → 生成回答
            if round_no == 1:
                yield "status", "正在检索与搜索..."
            else:
                yield "status", "正在分析检索结果..."

            t0 = time.monotonic()
            decision = client.chat_raw(conversation)
            meta["phase_times"].append(
                ("首轮分析" if round_no == 1 else "分析检索结果",
                 int((time.monotonic() - t0) * 1000))
            )

            calls = self._parse_tool_text(decision)
            if not calls and self._detect_tool_text(decision):
                # 疑似工具调用但解析失败（畸形 JSON）：约束重试一次
                logger.warning("决策轮输出疑似工具格式但无法解析，约束重试")
                conversation.append({"role": "assistant", "content": decision})
                conversation.append({"role": "user", "content": _TOOL_CONSTRAINT})
                decision = client.chat_raw(conversation)
                if self._detect_tool_text(decision):
                    # 约束重试仍输出工具格式：放弃本轮调用，按无调用处理
                    logger.warning("约束重试后仍输出工具格式文本，按无调用处理")
                    decision = ""
                calls = self._parse_tool_text(decision) if decision else []
            if not calls:
                conversation.append({"role": "assistant", "content": decision or ""})
                break

            if round_no > 1:
                yield "status", f"正在补充检索（第{round_no - 1}轮）..."
            t1 = time.monotonic()
            results = []
            for call in calls:
                name = call["name"]
                text, sources, ok = self._execute_tool(name, call["arguments"])
                text = self._validate_result(name, text, ok)
                if ok:
                    self._record_tool_meta(name, sources, meta)
                results.append(f"[工具 {name} 结果]\n{text}")
            meta["phase_times"].append(
                ("检索与搜索" if round_no == 1 else f"补充检索（第{round_no - 1}轮）",
                 int((time.monotonic() - t1) * 1000))
            )

            conversation.append({"role": "assistant", "content": decision})
            conversation.append({
                "role": "user",
                "content": (
                    "工具结果：\n" + "\n\n".join(results)
                    + "\n请继续：若信息足够请直接回答，若仍需查询请再次调用工具。"
                ),
            })
        else:
            # 达到轮次上限：不再决策，直接带着已有工具结果进入最终生成
            logger.info("ReAct 循环达到 %d 轮上限，进入最终生成", MAX_TOOL_ROUNDS)

        final_messages = self._clean_for_final(conversation)
        meta["last_stream_messages"] = final_messages
        yield from self._generate_final(final_messages, client, meta, deep_thinking)