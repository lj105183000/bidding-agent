"""招投标采购 Agent API 服务。

端点契约见 docs/开发文档.md §5；SSE 事件协议见 §6。
错误码：400 参数 / 413 图片过大 / 422 枚举校验 / 429 限流 / 500 处理异常 / 503 依赖未就绪。
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.clients.llm_factory import get_llm_client
from src.config import settings
from src.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# 模块级单例：限流器
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_max,
    window_seconds=settings.rate_limit_window,
)

# SSE 首帧 2KB padding：强制代理/缓冲区立即 flush（见 §6.1）
_SSE_PADDING = ":" + " " * 2048 + "\n\n"

# 图片 base64 上限（约 3MB），超限返回 413
_MAX_IMAGE_BASE64_CHARS = 4_000_000


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """对话接口请求参数（见 docs/开发文档.md §5.1）。"""

    question: str = Field(..., description="用户问题，空串 400")
    history: list[dict[str, str]] | None = Field(
        default=None, description="历史对话 [{role, content}]，后端仅保留最近 5 轮"
    )
    web_search_enabled: bool = Field(
        default=False, description="开启后动态挂载联网搜索工具"
    )
    provider: str | None = Field(
        default=None, description="指定 LLM 服务商；空 = 取配置 LLM_PROVIDER"
    )
    deep_thinking_enabled: bool = Field(
        default=False, description="是否启用深度思考"
    )


class AskRequest(BaseModel):
    """直接 RAG 问答请求（不走 Agent 工具）。"""

    question: str = Field(..., description="用户问题，空串 400")
    top_k: int = Field(default=5, ge=1, le=50)


class VisionRequest(BaseModel):
    """图片理解请求。"""

    image_base64: str = Field(..., min_length=1)
    prompt: str = Field(default="请详细描述这张图片的内容")


class ConversationRequest(BaseModel):
    """会话保存请求。"""

    session_id: str = Field(..., min_length=1)
    title: str = Field(default="")
    messages: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """用户反馈请求。"""

    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., description='枚举 "up" / "down"')

    @field_validator("rating")
    @classmethod
    def _validate_rating(cls, v: str) -> str:
        if v not in ("up", "down"):
            # 非法值由全局异常处理器转 422
            raise ValueError('rating 必须为 "up" 或 "down"')
        return v


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def normalize_history(
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """只保留最近 5 轮对话。"""
    if not history:
        return []
    return history[-5:]


def build_messages(
    question: str,
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """构造发送给 LLM 的标准消息列表。"""
    messages = normalize_history(history)
    messages.append({"role": "user", "content": question})
    return messages


def sse_event(event: str, data: dict[str, Any]) -> str:
    """将结构化事件序列化为 SSE JSON 帧。"""
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False)}\n\n"


def _client_ip(request: Request) -> str:
    """从请求中提取客户端 IP，未知时回退 unknown。"""
    return request.client.host if request.client else "unknown"


def _require_question(question: str) -> None:
    """question 为空或纯空白时返回 400。"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")


_pg_unavailable_logged = False


def _get_postgres_client():
    """懒加载 PostgreSQL 客户端（C 组成员负责实现）。

    模块未就绪或连接失败时返回 None，调用方据此返回 503。
    """
    global _pg_unavailable_logged
    try:
        from src.database.postgresql_client import PostgreSQLClient  # type: ignore

        return PostgreSQLClient()
    except (ImportError, ModuleNotFoundError):
        # C 组尚未实现，属预期情况，仅首次打印提示
        if not _pg_unavailable_logged:
            logger.warning(
                "PostgreSQL 客户端（src.database.postgresql_client）尚未实现，"
                "会话/反馈接口将返回 503。"
            )
            _pg_unavailable_logged = True
        return None
    except Exception:
        logger.exception("PostgreSQL 客户端初始化失败")
        return None


# ---------------------------------------------------------------------------
# 应用与中间件
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭生命周期。"""
    logger.info("API 服务启动，监听 %s:%s", settings.api_host, settings.api_port)
    yield
    logger.info("API 服务关闭")


app = FastAPI(title="招投标采购 Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """对 /api/chat* 前缀端点按客户端 IP 滑动窗口限流（30 次 / 60 秒）。"""
    path = request.url.path
    if path.startswith("/api/chat"):
        ip = _client_ip(request)
        if not rate_limiter.allow(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
            )
    return await call_next(request)


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Pydantic 校验失败统一 422。"""
    return JSONResponse(
        status_code=422,
        content={"detail": "请求参数校验失败，请检查字段类型与取值"},
    )


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """返回 API 及依赖组件的当前状态。

    各组件就绪状态由对应模块（B/C/A）在实现后回填；
    当前骨架阶段除 API 本身外均为未就绪。
    """
    return {
        "ready": False,  # RAG 流水线未初始化（B 组实现后回填）
        "agent_ready": False,  # Agent 未初始化（A 组实现后回填）
        "graph_ready": False,  # Neo4j 未连接（C 组实现后回填）
        "pg_ready": _get_postgres_client() is not None,
        "points_count": 0,
        "latencies": {
            "neo4j_ms": -1,
            "postgres_ms": -1,
            "qdrant_check_ms": -1,
        },
    }


# ---------------------------------------------------------------------------
# 对话类端点
# ---------------------------------------------------------------------------


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式对话（推荐）。事件协议见 docs/开发文档.md §6。"""
    _require_question(request.question)
    messages = build_messages(request.question, request.history)

    async def event_generator():
        # 首帧 2KB padding，强制代理立即 flush
        yield _SSE_PADDING

        try:
            client = get_llm_client(request.provider)

            yield sse_event("status", {"content": "正在生成回答"})

            if request.deep_thinking_enabled:
                async for chunk in client.chat_stream_thinking(messages):
                    if chunk["type"] == "thinking":
                        yield sse_event("thinking", {"content": chunk["content"]})
                    elif chunk["type"] == "content":
                        yield sse_event("token", {"content": chunk["content"]})
            else:
                async for content in client.chat_stream(messages):
                    yield sse_event("token", {"content": content})

            yield sse_event(
                "done",
                {
                    "sources": [],
                    "web_sources": [],
                    "tool_called": False,
                    "tool_name": "",
                    "elapsed_ms": 0,
                    "phase_times": {},
                },
            )
        except Exception:
            # 异常只进日志，不向用户泄露内部地址 / 原始异常
            logger.exception("流式对话失败")
            yield sse_event("error", {"content": "回答生成失败，请稍后重试"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """非流式对话（已废弃，保留兼容）。"""
    _require_question(request.question)
    messages = build_messages(request.question, request.history)

    try:
        client = get_llm_client(request.provider)

        if request.deep_thinking_enabled:
            content_parts: list[str] = []
            async for chunk in client.chat_stream_thinking(messages):
                if chunk["type"] == "content":
                    content_parts.append(chunk["content"])
            answer = "".join(content_parts)
        else:
            answer = await client.chat(messages)

        return {
            "answer": answer,
            "sources": [],
            "tool_called": False,
            "tool_name": "",
            "provider": client.provider,
            "model": client.model_name,
        }
    except Exception:
        logger.exception("非流式对话失败")
        raise HTTPException(
            status_code=500,
            detail="回答生成失败，请稍后重试",
        )


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    """直接问答（暂走 LLM 直答，RAG 由 B 组接入后回填 sources）。"""
    _require_question(request.question)
    messages = [{"role": "user", "content": request.question}]

    try:
        client = get_llm_client()
        answer = await client.chat(messages)
        return {
            "answer": answer,
            "sources": [],
        }
    except Exception:
        logger.exception("ask 接口失败")
        raise HTTPException(
            status_code=500,
            detail="回答生成失败，请稍后重试",
        )


# ---------------------------------------------------------------------------
# 视觉类端点
# ---------------------------------------------------------------------------


@app.post("/api/vision")
async def vision(request: VisionRequest) -> dict[str, Any]:
    """图片理解。base64 超限 413，视觉服务未就绪 503。"""
    if len(request.image_base64) > _MAX_IMAGE_BASE64_CHARS:
        raise HTTPException(
            status_code=413,
            detail="图片过大，请压缩后重新上传",
        )

    if not settings.zhipu_api_key:
        raise HTTPException(
            status_code=503,
            detail="视觉服务未就绪，请配置 ZHIPU_API_KEY",
        )

    try:
        from src.clients.vision import VisionClient

        client = VisionClient()
        analysis = await client.analyze(request.image_base64, request.prompt)
        return {
            "analysis": analysis,
            "model": client.model_name,
        }
    except Exception:
        logger.exception("视觉接口失败")
        raise HTTPException(
            status_code=500,
            detail="图片分析失败，请稍后重试",
        )


# ---------------------------------------------------------------------------
# 会话与反馈（PostgreSQL）
# ---------------------------------------------------------------------------


@app.post("/api/conversations")
async def save_conversation(request: ConversationRequest) -> dict[str, str]:
    """保存会话（新会话插入，已存在则更新；空 title 保留库中原值）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        pg.upsert_conversation(
            session_id=request.session_id,
            title=request.title,
            messages=request.messages,
        )
        return {"status": "ok"}
    except Exception:
        logger.exception("保存会话失败")
        raise HTTPException(status_code=503, detail="会话保存失败")


@app.get("/api/conversations")
async def list_conversations(q: str = "") -> dict[str, Any]:
    """获取会话列表；q 非空时全文搜索。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conversations = pg.list_conversations(q=q or None)
        return {"conversations": conversations}
    except Exception:
        logger.exception("获取会话列表失败")
        raise HTTPException(status_code=503, detail="获取会话列表失败")


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str) -> dict[str, Any]:
    """获取单个会话；不存在时 messages 为 []。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conv = pg.get_conversation(session_id)
        if conv is None:
            return {"session_id": session_id, "messages": []}
        return conv
    except Exception:
        logger.exception("获取会话失败")
        raise HTTPException(status_code=503, detail="获取会话失败")


@app.delete("/api/conversations")
async def clear_conversations() -> dict[str, str]:
    """清空全部会话。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        pg.clear_conversations()
        return {"status": "ok"}
    except Exception:
        logger.exception("清空会话失败")
        raise HTTPException(status_code=503, detail="清空会话失败")


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str) -> dict[str, str]:
    """删除单个会话。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        pg.delete_conversation(session_id)
        return {"status": "ok"}
    except Exception:
        logger.exception("删除会话失败")
        raise HTTPException(status_code=503, detail="删除会话失败")


@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest) -> dict[str, str]:
    """保存用户反馈。rating 非法值由校验器拦截返回 422。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="反馈服务未就绪")

    try:
        pg.insert_feedback(
            session_id=request.session_id,
            question=request.question,
            answer=request.answer,
            rating=request.rating,
        )
        return {"status": "ok"}
    except Exception:
        logger.exception("保存反馈失败")
        raise HTTPException(status_code=503, detail="反馈保存失败")
