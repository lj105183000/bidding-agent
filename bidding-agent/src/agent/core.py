"""BiddingAgent 主体：生命周期、_chat_events 编排入口、chat / chat_stream。

两条输出路径共用同一事件流（开发文档 §8.1）：
  _chat_events  产出六类事件元组（status/token/thinking/reset/done/error）
  chat_stream   序列化为 SSE 帧（api/server.py 的 /api/chat/stream 使用）
  chat          收集为 dict（非流式接口与 eval/agent_eval.py 使用）
"""

import logging
import os
import time

from src.agent.constants import THINKING_MODEL_ENV, llm_client_lock
from src.agent.generation import GenerationMixin
from src.agent.react_loop import ReActMixin
from src.agent.skills import _load_skills, _match_skills
from src.agent.tool_defense import ToolDefenseMixin
from src.agent.utils import _sse, _truncate_history, SSE_DONE_MARK, SSE_PADDING
from src.agent.prompts import SYSTEM_PROMPT

try:
    # 工具层（成员 C）尚未合入时以空集运行；此绑定用于阶段二接 ToolRunner（runner 构造）
    from src.tools.rag_tools import TOOL_EXECUTORS
except ImportError:  # pragma: no cover - 阶段一工具层未落地
    TOOL_EXECUTORS = {}

try:
    # LLM 客户端工厂（成员 D）尚未合入时不可用，测试经注入 fake client 自测
    from src.clients.llm_factory import get_llm_client
except ImportError:  # pragma: no cover - 阶段一客户端未落地
    get_llm_client = None

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "deepseek"


class AgentError(Exception):
    """Agent 内部错误基类。完整异常只进日志，用户可见文案由 _chat_events 统一给出。"""


class LLMUnavailableError(AgentError):
    """LLM 客户端不可用（工厂未就绪或 provider 缺少配置）。"""


class BiddingAgent(ReActMixin, GenerationMixin, ToolDefenseMixin):
    """编排核心。llm_client / rag_pipeline 可注入（测试与阶段二联调前自测）。"""

    def __init__(self, llm_client=None, rag_pipeline=None):
        self._llm_client = llm_client
        self._rag_pipeline = rag_pipeline
        self._provider_clients = {}  # provider -> client 缓存，受 llm_client_lock 保护
        self.skills = _load_skills()
        # 健康检查的 agent_ready 依据：初始化走到这里即视为就绪
        self.ready = True
        logger.info("BiddingAgent 初始化完成，加载技能 %d 个", len(self.skills))

    # --- 组件获取（懒加载 + 缓存） ---

    def _get_llm_client(self, provider: str = ""):
        """按 provider 取 LLM 客户端。注入的 llm_client 优先（测试/特殊部署），
        否则经 llm_factory 创建并按 provider 缓存（并发首用只创建一份）。"""
        if self._llm_client is not None:
            return self._llm_client
        provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        with llm_client_lock:
            client = self._provider_clients.get(provider)
            if client is None:
                if get_llm_client is None:
                    raise LLMUnavailableError("LLM 客户端工厂未就绪")
                client = get_llm_client(provider)
                self._provider_clients[provider] = client
            return client

    def _get_rag_pipeline(self):
        """取 RAG 流水线单例（成员 B）。未合入时返回 None，RAG 降级路径据此报错。"""
        if self._rag_pipeline is not None:
            return self._rag_pipeline
        if getattr(self, "_rag_probed", False):  # 已探测过且不可用，避免每次请求重复 import
            return None
        self._rag_probed = True
        try:
            from src.rag.pipeline import rag_pipeline
        except ImportError:
            logger.info("RAG 流水线尚未合入，降级路径不可用")
            return None
        self._rag_pipeline = rag_pipeline
        return self._rag_pipeline

    def _thinking_model(self, client, provider: str = ""):
        """深度思考模型切换：配置了 THINKING_MODEL 且客户端支持则切换；
        否则原样返回，由最终生成回退提示词注入。新增 provider 在 THINKING_MODEL_ENV 补充。"""
        env_key = THINKING_MODEL_ENV.get(
            provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER))
        model_name = os.getenv(env_key, "").strip() if env_key else ""
        if model_name and hasattr(client, "switch_thinking_model"):
            logger.info("深度思考启用思考模型 %s", model_name)
            client.switch_thinking_model(model_name)
        return client

    # --- 上下文构建 ---

    def _build_context(self, question: str, history: list = None,
                       web_search_enabled: bool = False) -> list:
        """构建首轮消息：system（角色 + 技能注入）+ 截断后的历史 + 当前问题。

        返回列表首条必须是 system 消息（_clean_for_final 依赖 messages[0]）。
        """
        system = SYSTEM_PROMPT
        matched = _match_skills(question, self.skills)
        if matched:
            blocks = "\n\n".join(f"### 技能：{s['name']}\n{s['body']}" for s in matched)
            system += "\n\n## 已启用技能（按其步骤执行）\n" + blocks
        if web_search_enabled:
            system += "\n\n已开启联网搜索：search_web / search_exa 工具可用，时效性问题优先联网。"
        messages = [{"role": "system", "content": system}]
        messages.extend(_truncate_history(history))
        messages.append({"role": "user", "content": question})
        return messages

    # --- 编排入口 ---

    def _chat_events(self, question: str, history: list = None,
                     web_search_enabled: bool = False, provider: str = "",
                     deep_thinking_enabled: bool = False):
        """编排入口。产出事件元组：

        ("status"|"token"|"thinking", 文本)  ("reset", None)
        ("done", {sources, web_sources, tool_called, tool_name, elapsed_ms, phase_times})
        ("error", {"content": 用户可见文案})

        Agent 主路径失败时自动降级 RAG；降级也失败产出 error，不产出 done。
        """
        t_start = time.monotonic()
        meta = {
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",       # 单字符串：最后一个执行完成的工具名（前端徽标 + eval 依赖）
            "phase_times": [],
            "last_stream_messages": None,
            "streamed": False,     # 是否已有正文流出：决定失败时走 RAG 降级还是直接报错
        }
        yield "status", "正在初始化..."

        error_event = None
        try:
            client = self._get_llm_client(provider)
            if deep_thinking_enabled:
                client = self._thinking_model(client, provider)
                yield "status", "正在深度思考..."
            messages = self._build_context(question, history, web_search_enabled)
            yield from self._chat_stream_tools(messages, client, meta, deep_thinking_enabled)
        except Exception:
            logger.exception("Agent 编排失败，尝试降级")
            if meta["streamed"]:
                # 正文已部分流出：不再降级重答（避免重复内容），直接报错由前端兜底
                error_event = {"content": "回答生成中断，请稍后重试。"}
            else:
                for event_type, payload in self._rag_fallback_events(question, meta):
                    if event_type == "error":
                        error_event = payload
                    else:
                        yield event_type, payload

        if error_event is not None:
            yield "error", error_event
        else:
            yield "done", self._build_done(meta, t_start)

    @staticmethod
    def tool_names() -> list:
        """当前注册的工具名列表（健康检查/诊断用）。"""
        return list(TOOL_EXECUTORS)

    @staticmethod
    def _build_done(meta: dict, t_start: float) -> dict:
        """组装 done 帧（契约见开发文档 §6.4，字段结构与 tool_name 单字符串语义勿改）。"""
        return {
            "sources": meta["sources"],
            "web_sources": meta["web_sources"],
            "tool_called": meta["tool_called"],
            "tool_name": meta["tool_name"],
            "elapsed_ms": int((time.monotonic() - t_start) * 1000),
            "phase_times": meta["phase_times"],
        }

    # --- 对外两条路径 ---

    def chat(self, question: str, history: list = None, web_search_enabled: bool = False,
             provider: str = "", deep_thinking_enabled: bool = False) -> dict:
        """非流式入口：拉平事件流收集为 dict。eval/agent_eval.py 依赖 tool_name 口径。"""
        result = {
            "answer": "",
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",
            "elapsed_ms": 0,
            "phase_times": [],
        }
        for event_type, payload in self._chat_events(
                question, history, web_search_enabled, provider, deep_thinking_enabled):
            if event_type == "token":
                result["answer"] += payload
            elif event_type == "reset":
                # 与前端语义一致：清空已流正文，等后续 token 重来
                result["answer"] = ""
            elif event_type == "done":
                result.update(payload)
            elif event_type == "error":
                result["answer"] += f"\n\n【错误】{payload['content']}"
        return result

    def chat_stream(self, question: str, history: list = None,
                    web_search_enabled: bool = False, provider: str = "",
                    deep_thinking_enabled: bool = False):
        """SSE 流式入口：事件元组 → SSE 帧。首帧 2KB padding 强制代理 flush。"""
        yield SSE_PADDING
        for event_type, payload in self._chat_events(
                question, history, web_search_enabled, provider, deep_thinking_enabled):
            if event_type == "reset":
                yield _sse({"type": "reset"})
            elif event_type == "done":
                frame = {"type": "done"}
                frame.update(payload)
                yield _sse(frame)
            else:
                content = payload if isinstance(payload, str) else payload.get("content", "")
                yield _sse({"type": event_type, "content": content})
        yield SSE_DONE_MARK
