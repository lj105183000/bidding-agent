"""招投标采购 Agent API 服务。

端点契约见 docs/开发文档.md §5；SSE 事件协议见 §6。
错误码：400 参数 / 413 图片过大 / 422 枚举校验 / 429 限流 / 500 处理异常 / 503 依赖未就绪。

SSE 帧统一由 `BiddingAgent.chat_stream` 产出（`type` 键，done 帧含
sources / web_sources / tool_called / tool_name / elapsed_ms / phase_times，
tool_name 为单字符串）——本文件只负责透传与包装，不再自行拼 event 帧。
会话 / 反馈端点按接口联调 spec §7 适配 PostgresClient 的逐条消息存储模型。
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# 模块级单例：限流器
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_max,
    window_seconds=settings.rate_limit_window,
)

# SSE 首帧 2KB padding：强制代理/缓冲区立即 flush（见 §6.1）。
# 由 BiddingAgent.chat_stream 产出，此处仅保留一个错误帧构造辅助。
_SSE_PADDING = ":" + " " * 2048 + "\n\n"

# 图片 base64 上限（约 3MB），超限返回 413
_MAX_IMAGE_BASE64_CHARS = 4_000_000

# BiddingAgent 单例（懒构造：初始化失败记录日志并以 None 兜底，chat* 端点据此返回 503）
try:
    from src.agent import BiddingAgent

    bidding_agent: BiddingAgent | None = BiddingAgent()
except Exception:  # pragma: no cover - 初始化失败属部署异常
    logger.exception("BiddingAgent 初始化失败，对话端点将返回 503")
    bidding_agent = None


def _error_sse(content: str) -> str:
    """构造单条 error 帧（type 键口径，与 §6 协议一致）。"""
    return "data: " + json.dumps(
        {"type": "error", "content": content}, ensure_ascii=False
    ) + "\n\n"


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
    """会话保存请求（接口联调 spec §7：messages 为**本轮新增**的消息）。"""

    session_id: str = Field(..., min_length=1)
    title: str = Field(default="")
    messages: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """用户反馈请求（spec §7：按 session_id 解析会话，关联 assistant 消息 id）。"""

    session_id: str = Field(..., min_length=1)
    message_id: int = Field(..., description="assistant 消息 id（保存会话时返回的 ids 中取）")
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


def _client_ip(request: Request) -> str:
    """从请求中提取客户端 IP，未知时回退 unknown。"""
    return request.client.host if request.client else "unknown"


def _require_question(question: str) -> None:
    """question 为空或纯空白时返回 400。"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")


_pg_unavailable_logged = False


def _get_postgres_client():
    """懒加载 PostgreSQL 客户端（src.database.postgresql_client）。

    模块未就绪或连接失败时返回 None，调用方据此返回 503。
    注意实际类名是 `PostgresClient`（非骨架假设的 PostgreSQLClient）。
    """
    global _pg_unavailable_logged
    try:
        from src.database.postgresql_client import PostgresClient

        return PostgresClient()
    except (ImportError, ModuleNotFoundError, ValueError):
        # 未实现或缺必需配置，属预期情况，仅首次打印提示
        if not _pg_unavailable_logged:
            logger.warning(
                "PostgreSQL 客户端（src.database.postgresql_client.PostgresClient）"
                "不可用，会话/反馈接口将返回 503。"
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

    agent_ready 反映 BiddingAgent 单例是否初始化成功；pg_ready 反映 Postgres
    客户端能否构造（真正连通性由会话端点按需判断）。
    """
    return {
        "ready": bidding_agent is not None,
        "agent_ready": bidding_agent is not None,
        "graph_ready": False,  # Neo4j 未接入本次联调范围
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
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式对话（推荐）。事件协议见 docs/开发文档.md §6。

    事件帧由 `bidding_agent.chat_stream` 直接产出（type 键、首帧 padding、
    末帧 [DONE] 均已含），本端点只透传；Agent 未就绪时返回首帧即 error。
    """
    _require_question(request.question)

    def event_generator():
        if bidding_agent is None:
            yield _SSE_PADDING
            yield _error_sse("Agent 未就绪，请稍后重试")
            return
        try:
            yield from bidding_agent.chat_stream(
                question=request.question,
                history=request.history,
                web_search_enabled=request.web_search_enabled,
                provider=request.provider or "",
                deep_thinking_enabled=request.deep_thinking_enabled,
            )
        except Exception:
            # 异常只进日志，不向用户泄露内部地址 / 原始异常
            logger.exception("流式对话失败")
            yield _error_sse("回答生成失败，请稍后重试")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """非流式对话（评测 / 脚本调用，保留兼容）。"""
    _require_question(request.question)
    if bidding_agent is None:
        raise HTTPException(status_code=503, detail="Agent 未就绪，请稍后重试")

    try:
        return bidding_agent.chat(
            question=request.question,
            history=request.history,
            web_search_enabled=request.web_search_enabled,
            provider=request.provider or "",
            deep_thinking_enabled=request.deep_thinking_enabled,
        )
    except Exception:
        logger.exception("非流式对话失败")
        raise HTTPException(status_code=500, detail="回答生成失败，请稍后重试")


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    """直接问答（暂走 LLM 直答，RAG 由 B 组接入后回填 sources）。"""
    _require_question(request.question)
    messages = [{"role": "user", "content": request.question}]

    try:
        from src.clients.llm_factory import get_llm_client

        client = get_llm_client(settings.llm_provider)
        answer = "".join(client.chat_stream(messages))
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
# 会话与反馈（PostgreSQL，接口联调 spec §7 契约适配）
# ---------------------------------------------------------------------------

# 允许落库的 role / 单条消息保留字段：thinking 与计时不落库（PersistedMessage 契约）
_ALLOWED_ROLES = ("user", "assistant")


@app.post("/api/conversations")
def save_conversation(request: ConversationRequest) -> dict[str, Any]:
    """保存本轮新增消息。

    messages 语义为**本轮新增**的 user/assistant 消息（前端每回合只发两条）。
    服务端按 session_id 找到会话（找不到则 create_conversation 新建），
    逐条 save_message；响应带每条消息的 id（前端取 assistant 消息 id 用于反馈）。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conversation_id = pg.find_id_by_session(request.session_id)
        if conversation_id < 0:
            conversation_id = pg.create_conversation(
                title=request.title, session_id=request.session_id
            )
            if conversation_id < 0:
                logger.error("创建会话失败：session_id=%s", request.session_id)
                raise HTTPException(status_code=503, detail="会话保存失败")

        ids: list[int] = []
        for msg in request.messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in _ALLOWED_ROLES or not content or not str(content).strip():
                continue
            mid = pg.save_message(
                conversation_id=conversation_id,
                role=role,
                content=str(content),
                sources=msg.get("sources"),
                tool_name=msg.get("toolName"),
            )
            if mid >= 0:
                ids.append(mid)
        return {"status": "ok", "ids": ids}
    except HTTPException:
        raise
    except Exception:
        logger.exception("保存会话失败")
        raise HTTPException(status_code=503, detail="会话保存失败")


@app.get("/api/conversations")
def list_conversations() -> dict[str, Any]:
    """获取会话列表（新建的在前）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conversations = pg.list_conversations()
        return {"conversations": conversations}
    except Exception:
        logger.exception("获取会话列表失败")
        raise HTTPException(status_code=503, detail="获取会话列表失败")


@app.get("/api/conversations/{session_id}")
def get_conversation(session_id: str) -> dict[str, Any]:
    """获取单个会话的全部消息；不存在时 messages 为 []。

    `tool_name` 映射为前端契约的 `toolName`；sources 原样返回。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conversation_id = pg.find_id_by_session(session_id)
        if conversation_id < 0:
            return {"session_id": session_id, "messages": []}
        rows = pg.load_messages(conversation_id)
        messages = [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "sources": row.get("sources"),
                "toolName": row.get("tool_name"),
            }
            for row in rows
        ]
        return {"session_id": session_id, "messages": messages}
    except Exception:
        logger.exception("获取会话失败")
        raise HTTPException(status_code=503, detail="获取会话失败")


@app.delete("/api/conversations/{session_id}")
def delete_conversation(session_id: str) -> dict[str, str]:
    """删除单个会话（含其消息与反馈）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")

    try:
        conversation_id = pg.find_id_by_session(session_id)
        if conversation_id >= 0:
            pg.delete_conversation(conversation_id)
        return {"status": "ok"}
    except Exception:
        logger.exception("删除会话失败")
        raise HTTPException(status_code=503, detail="删除会话失败")


@app.post("/api/feedback")
def save_feedback(request: FeedbackRequest) -> dict[str, str]:
    """保存用户反馈。rating 非法值由校验器拦截返回 422。

    按 session_id 解析 conversation_id，再调 save_feedback(cid, mid, rating)。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="反馈服务未就绪")

    try:
        conversation_id = pg.find_id_by_session(request.session_id)
        if conversation_id < 0:
            return {"status": "ok"}  # 会话未落库：无对象可反馈，静默成功
        pg.save_feedback(
            conversation_id=conversation_id,
            message_id=request.message_id,
            rating=request.rating,
        )
        return {"status": "ok"}
    except Exception:
        logger.exception("保存反馈失败")
        raise HTTPException(status_code=503, detail="反馈保存失败")
