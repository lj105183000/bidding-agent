from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from src.clients.llm_factory import get_llm_client
from src.rate_limiter import RateLimiter
from fastapi.responses import StreamingResponse
import json

class ChatRequest(BaseModel):
    """聊天接口请求参数。"""

    question: str = Field(...,min_length=1, description="用户问题")
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="历史对话",
    )
    provider: str | None = Field(
        default=None,
        description="指定 LLM 服务商",
    )
    deep_thinking: bool = Field(
        default=False,
        description="是否启用深度思考",
    )


app = FastAPI(title="招投标采购 Agent API")


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



def sse_event(event: str, data: dict) -> str:
    """将结构化事件序列化为 SSE JSON 帧。"""
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False)}\n\n"

@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
) -> dict:
    """处理同步聊天请求。"""
    client_ip = http_request.client.host if http_request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
        )

    messages = build_messages(
        request.question,
        request.history,
    )

    try:
        client = get_llm_client(request.provider)

        if request.deep_thinking:
            thinking_parts: list[str] = []
            content_parts: list[str] = []

            async for chunk in client.chat_stream_thinking(messages):
                if chunk["type"] == "thinking":
                    thinking_parts.append(chunk["content"])
                elif chunk["type"] == "content":
                    content_parts.append(chunk["content"])

            answer = "".join(content_parts)
        else:
            answer = await client.chat(messages)
        return {
            "answer": answer,
            "provider": client.provider,
            "model": client.model_name,
        }
    
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 请求失败: {exc}",
        ) from exc


rate_limiter = RateLimiter()


@app.get("/api/health")
def health_check() -> dict:
    """返回 API 及依赖组件的当前状态。"""
    return {
        "ready": True,
        "agent_ready": False,
        "graph_ready": False,
        "pg_ready": False,
        "points_count": 0,
    }


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
) -> StreamingResponse:
    """处理流式聊天请求。"""
    client_ip = http_request.client.host if http_request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
        )

    messages = build_messages(
        request.question,
        request.history,
    )

    async def event_generator():
        try:
            client = get_llm_client(request.provider)

            yield sse_event(
                "status",
                {"message": "正在生成回答"},
            )

            if request.deep_thinking:
                async for chunk in client.chat_stream_thinking(messages):
                    if chunk["type"] == "thinking":
                        yield sse_event(
                            "thinking",
                            {"content": chunk["content"]},
                        )
                    elif chunk["type"] == "content":
                        yield sse_event(
                            "token",
                            {"content": chunk["content"]},
                        )
            else:
                async for content in client.chat_stream(messages):
                    yield sse_event(
                        "token",
                        {"content": content},
                    )

            yield sse_event(
                "done",
                {
                    "sources": [],
                    "web_sources": [],
                    "tool_name": "",
                    "elapsed_ms": 0,
                    "phase_times": {},
                },
            )
        except Exception as exc:
            yield sse_event(
                "error",
                {"message": str(exc)},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ask")
async def ask(
    request: ChatRequest,
    http_request: Request,
) -> dict:
    """处理简单问答请求。"""
    client_ip = http_request.client.host if http_request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
        )

    messages = build_messages(
        request.question,
        request.history,
    )

    try:
        client = get_llm_client(request.provider)
        answer = await client.chat(messages)

        return {
            "answer": answer,
            "provider": client.provider,
            "model": client.model_name,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 请求失败: {exc}",
        ) from exc

'''
@app.get("/api/test-limit")
def test_limit(request: Request) -> dict:
    """临时接口，用于验证限流逻辑。"""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    return {"success": True}

'''