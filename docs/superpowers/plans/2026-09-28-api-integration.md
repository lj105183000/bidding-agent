# 前后端接口联调实施计划（Chat 全链路）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 后端 merge feat/api-llm 并补齐为调用 BiddingAgent 的真实 API（SSE `type` 键口径），前端 Vite SPA 建 SSE 网络层并改造 Chat 页为流式问答 + 会话/反馈持久化。

**Architecture:** 后端 `api/server.py` 只做编排（懒加载单例 + StreamingResponse 包 `agent.chat_stream`，agent 已产出完整 SSE 帧）；前端 `lib/sse.ts`（纯解析器）+ `lib/api.ts`（唯一网络层）+ Chat.tsx 重写。直连 `:8001`，后端 CORS 白名单加 `:5173`。

**Tech Stack:** FastAPI / TestClient / pytest（同步）、Vite 5 + React 18 + react-markdown ^9 + remark-gfm ^4 + vitest

**Spec:** `docs/superpowers/specs/2026-09-28-api-integration-design.md`（含 §7 契约适配，**必须先读**）

**已核实的代码事实（写本计划时逐一确认过，执行者不要凭假设推翻）：**

- `BiddingAgent.chat_stream(question, history, web_search_enabled, provider, deep_thinking_enabled)`（`src/agent/core.py:215`）**已产出完整 SSE 帧**：首帧 padding、`{"type": ...}` JSON 帧、`SSE_DONE_MARK`。helper 在 `src/agent/utils.py`（`_sse` / `SSE_PADDING` / `SSE_DONE_MARK`）。server 端**不要**再自己拼帧。
- done 帧字段：`sources / web_sources / tool_called / tool_name / elapsed_ms / phase_times`（`core.py:177-186`），`tool_name` 单字符串，`phase_times` 为 `[[名, ms], ...]`。
- dev 的 PG 客户端类名是 **`PostgresClient`**（`src/database/postgresql_client.py:240`），方法：`create_conversation(title, session_id) -> int`（-1 哨兵）、`list_conversations(limit) -> [{id, session_id, title, created_at}]`、`delete_conversation(conversation_id: int) -> bool`、`save_feedback(conversation_id, message_id, rating, comment=None) -> bool`、`save_message(conversation_id, role, content, sources, tool_name) -> int`（-1 哨兵）、`load_messages(conversation_id, limit) -> [{id, role, content, sources, tool_name, created_at}]`、`health() -> {"ok": bool, "error": str|None}`、`ensure_schema() -> bool`。
- `src/rag/pipeline.py:392` 有模块级单例 `rag_pipeline`，属性 `ready: bool`、`ready_error: str|None`、`store`（`VectorStore.count() -> int`）。
- Neo4j/PG 的 `health()` 都返回 `{"ok": bool, "error": str|None}` 且不抛异常。
- feat/api-llm 的 `src/config.py`（140 行 dataclass 版）包含 api/cors/rate-limit 键；**dev 的 `src/config.py`（36 行）有 `rerank_enabled` 与 `qdrant_distance` 两个 feat 版没有的键**，且被 `src/rag/pipeline.py` 读取——merge 解决冲突时必须保留。
- `tests/test_sse_events.py` 是 SSE `type` 键口径的权威测试，全程必须保持绿。
- 前端 `frontend/src/pages/Chat.tsx`（265 行，纯 mock）、`src/contexts/model.tsx`（`MODEL_OPTIONS = ['GPT-4o','Claude-3.5','通义千问']`，被 `components/Layout.tsx` 与 `pages/Extract.tsx` 引用）、`components/` 有 Button(icon prop)/Card(title,subtitle)/PageHeader/Toast(useToast)。
- 约定：全部端点**同步 `def`**；提交信息 `类型(模块): 中文描述`；不动队友未提交的 `frontend/package-lock.json`（本次会在 frontend 装依赖会产生新的 lock 变更，属预期，正常提交）。

---

### Task 1: merge origin/feat/api-llm 进 dev 并解决冲突

**Files:**
- Modify: `bidding-agent/src/config.py`（冲突解决）
- New: `bidding-agent/api/server.py`、`bidding-agent/main.py`、`bidding-agent/src/rate_limiter.py`、`bidding-agent/tests/test_api_server.py`（若 feat 分支带）

- [ ] **Step 1: 确认工作区干净（除他人改动）**

```bash
cd D:/bidding-agent && git status --short
```

预期：只有 `M frontend/package-lock.json`。若有其他未提交改动，停下来报告。

- [ ] **Step 2: 执行 merge**

```bash
cd D:/bidding-agent && git merge origin/feat/api-llm
```

预期：`api/`、`main.py`、`src/rate_limiter.py` 等无冲突合入；`bidding-agent/src/config.py` 冲突（两边都改过）。

- [ ] **Step 3: 解决 `bidding-agent/src/config.py` 冲突**

以 **feat 版（140 行 dataclass 版）为基底**，然后做两处补齐（把 dev 版独有的键加回去，放"检索与重排配置"小节）：

```python
    # ---- 检索与重排配置 ----
    # 对应 pipeline.py 里的 settings.rerank_enabled（dev 分支独有，rag 模块读取）
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() in ("true", "1", "yes")
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "Cosine")
```

同时把 dev 版 `load_dotenv` 的写法统一为 feat 版顶部已有的 `load_dotenv(override=False)`（不要 dev 的 `PROJECT_ROOT / ".env"` 版本）。

- [ ] **Step 4: 其余冲突一律以 dev 口径为准**（特别是任何测试文件冲突——`tests/test_sse_events.py` 的 `type` 键口径不可动）。逐个 `git status` 列出的冲突文件解决后 `git add`。

- [ ] **Step 5: 全量测试验证 merge 未破坏 dev**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest -q 2>&1 | tail -3
```

预期：全绿。若有失败且与 merge 相关，报告 BLOCKED。

- [ ] **Step 6: 提交 merge**

```bash
cd D:/bidding-agent && git commit --no-edit
```

（merge 提交保留默认信息；若 Step 2 因无冲突直接完成则跳过本步。）

---

### Task 2: CORS 默认白名单加 :5173 + .env.example

**Files:**
- Modify: `bidding-agent/src/config.py`（`cors_origin_list` property）
- Modify: `bidding-agent/.env.example`

- [ ] **Step 1: 修改 `cors_origin_list` 默认值**

```python
# 旧
        if not raw:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
# 新
        if not raw:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
```

- [ ] **Step 2: `.env.example` 增加（放在 API 服务小节附近）**

```bash
# 前端跨域白名单（逗号分隔；* 允许所有；留空 = localhost:3000/5173）
#CORS_ORIGINS=
```

- [ ] **Step 3: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/src/config.py bidding-agent/.env.example && git commit -m "$(cat <<'EOF'
feat(api): CORS 默认白名单加入 Vite dev 端口 5173

前端为 Vite SPA（:5173）且直连 :8001，不进白名单会被 CORSMiddleware 拦截。
EOF
)"
```

---

### Task 3: PostgresClient.find_id_by_session（TDD）

**Files:**
- Modify: `bidding-agent/src/database/postgresql_client.py`（`list_conversations` 之后）
- Test: `bidding-agent/tests/test_postgres_session_lookup.py`

- [ ] **Step 1: 写失败测试**

```python
"""find_id_by_session：按 session_id 反查会话 id（server 端会话/反馈路由依赖）。"""

from src.database.postgresql_client import PostgresClient


def test_find_id_by_session_returns_id(monkeypatch):
    monkeypatch.setattr(
        PostgresClient, "_query",
        lambda self, sql, params: [{"id": 7}],
    )
    assert PostgresClient().find_id_by_session("s-1") == 7


def test_find_id_by_session_returns_negative_one_when_missing(monkeypatch):
    monkeypatch.setattr(
        PostgresClient, "_query",
        lambda self, sql, params: [],
    )
    assert PostgresClient().find_id_by_session("s-none") == -1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_postgres_session_lookup.py -q
```

预期：FAIL（AttributeError: find_id_by_session）。

- [ ] **Step 3: 实现（放在 `list_conversations` 方法之后）**

```python
    def find_id_by_session(self, session_id: str) -> int:
        """按 session_id 反查会话 id（同 session 多行时取最新），找不到返回 -1。

        server 的会话/反馈路由只有前端给的 session_id，没有数字主键；
        幂等写入与 feedback 归属都靠它先换成 conversation_id。
        """
        rows = self._query(
            "SELECT id FROM conversations WHERE session_id = %s ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        return int(rows[0]["id"]) if rows else -1
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_postgres_session_lookup.py -q
```

预期：2 passed。

- [ ] **Step 5: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/src/database/postgresql_client.py bidding-agent/tests/test_postgres_session_lookup.py && git commit -m "$(cat <<'EOF'
feat(database): PostgresClient 增加 find_id_by_session 会话反查

API 层只有前端 session_id，需要换成 conversation_id 才能幂等落库与归属反馈。
EOF
)"
```

---

### Task 4: server.py 对话端点接入 BiddingAgent + 真实健康检查（TDD）

**Files:**
- Modify: `bidding-agent/api/server.py`
- Test: `bidding-agent/tests/test_api_server.py`（新建或重构 feat 版）

- [ ] **Step 1: 写失败测试（新建 `tests/test_api_server.py`；若 feat 分支自带旧测试，整体替换为本文件）**

```python
"""API 服务测试：SSE 帧序列、chat 汇总、健康检查。全部同步、文件内局部 fake。"""

import json

from fastapi.testclient import TestClient

import api.server as server
from src.agent.utils import SSE_DONE_MARK, SSE_PADDING, _sse


class _FakeAgent:
    """脚本化 fake：chat_stream 按参数产出事件帧。"""

    def __init__(self):
        self.calls = []

    def chat_stream(self, question, history=None, web_search_enabled=False,
                    provider="", deep_thinking_enabled=False):
        self.calls.append({
            "question": question, "history": history,
            "web_search_enabled": web_search_enabled, "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
        })
        yield SSE_PADDING
        yield _sse({"type": "status", "content": "正在初始化..."})
        yield _sse({"type": "token", "content": "答案"})
        yield _sse({"type": "done", "sources": [], "web_sources": [],
                    "tool_called": False, "tool_name": "",
                    "elapsed_ms": 5, "phase_times": []})
        yield SSE_DONE_MARK


def _client_with(fake_agent):
    server.get_bidding_agent = lambda: fake_agent
    return TestClient(server.app)


def _frames(text):
    out = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    out.append(json.loads(payload))
    return out


def test_chat_stream_frames_and_passthrough_params():
    fake = _FakeAgent()
    resp = _client_with(fake).post("/api/chat/stream", json={
        "question": "如何质疑单一来源公示？",
        "history": [{"role": "user", "content": "hi"}],
        "provider": "zhipu", "web_search_enabled": True,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _frames(resp.text)
    assert frames[0]["type"] == "status"
    assert frames[-1]["type"] == "done"
    assert isinstance(frames[-1]["tool_name"], str)
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert fake.calls[0]["provider"] == "zhipu"
    assert fake.calls[0]["web_search_enabled"] is True
    assert fake.calls[0]["history"] == [{"role": "user", "content": "hi"}]


def test_chat_stream_empty_question_400():
    resp = _client_with(_FakeAgent()).post("/api/chat/stream", json={"question": "  "})
    assert resp.status_code == 400


def test_chat_non_stream_collects_answer():
    fake = _FakeAgent()

    class _CollectAgent(_FakeAgent):
        def chat(self, **kwargs):
            return {"answer": "答案", "sources": [], "web_sources": [],
                    "tool_called": False, "tool_name": "", "elapsed_ms": 5,
                    "phase_times": []}

    resp = _client_with(_CollectAgent()).post("/api/chat", json={"question": "q"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "答案"
    assert isinstance(body["tool_name"], str)


def test_health_reports_pipeline_and_deps(monkeypatch):
    class _FakePipeline:
        ready = True
        ready_error = None

        class store:
            @staticmethod
            def count():
                return 3

    class _FakePg:
        def health(self):
            return {"ok": True, "error": None}

    monkeypatch.setattr(server, "rag_pipeline", _FakePipeline())
    monkeypatch.setattr(server, "_get_postgres_client", lambda: _FakePg())
    resp = TestClient(server.app).get("/api/health")
    body = resp.json()
    assert body["ready"] is True
    assert body["points_count"] == 3
    assert body["pg_ready"] is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_api_server.py -q
```

预期：FAIL（`api.server` 无 `get_bidding_agent` / `rag_pipeline` 属性，或端点仍为骨架行为）。

- [ ] **Step 3: 修改 `api/server.py`**

3a. 顶部 import 区改为：

```python
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.agent import BiddingAgent
from src.clients.llm_factory import get_llm_client
from src.config import settings
from src.database.neo4j_client import Neo4jClient
from src.database.postgresql_client import PostgresClient
from src.logging_config import setup_logging  # noqa: F401  (保持与 main.py 一致的导入面)
from src.rag.pipeline import rag_pipeline
from src.rate_limiter import RateLimiter
```

注意：`get_llm_client` 仅剩 `/api/ask` 在用（本次不动 ask/vision）。

3b. 在 `rate_limiter` 单例之后加懒加载单例（双检锁，CLAUDE.md 约定）：

```python
# 模块级单例：Agent（懒加载 + 双检锁，并发首用只构造一份）
_bidding_agent: BiddingAgent | None = None
_bidding_agent_lock = threading.Lock()


def get_bidding_agent() -> BiddingAgent:
    global _bidding_agent
    if _bidding_agent is None:
        with _bidding_agent_lock:
            if _bidding_agent is None:
                _bidding_agent = BiddingAgent()
    return _bidding_agent
```

3c. **删除**骨架版辅助函数 `sse_event` / `normalize_history` 之外的全部死代码：`build_messages`（改为保留 `normalize_history`，`build_messages` 删）、以及 chat 端点里对 `get_llm_client` 的直接调用逻辑。

同时确认/补齐以下三项（骨架版已有则跳过）：

```python
logger = logging.getLogger(__name__)


def _require_question(question: str) -> None:
    if not (question or "").strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
```

并确认 `ChatRequest` 模型含全部 5 个字段 `question / history / provider / web_search_enabled / deep_thinking_enabled`（缺 `deep_thinking_enabled` 则补 `deep_thinking_enabled: bool = False`；`provider` 缺省为 `""`）。

3d. 用下面的实现**整体替换** `/api/chat/stream` 与 `/api/chat` 两个端点（含装饰器；注意是同步 `def`）：

```python
@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式对话。agent.chat_stream 已产出完整 SSE 帧（padding/type 帧/[DONE]）。"""
    _require_question(request.question)
    agent = get_bidding_agent()
    return StreamingResponse(
        agent.chat_stream(
            question=request.question,
            history=normalize_history(request.history),
            web_search_enabled=request.web_search_enabled,
            provider=request.provider or "",
            deep_thinking_enabled=request.deep_thinking_enabled,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """非流式对话（拉平事件流，eval 依赖 tool_name 口径）。"""
    _require_question(request.question)
    try:
        result = get_bidding_agent().chat(
            question=request.question,
            history=normalize_history(request.history),
            web_search_enabled=request.web_search_enabled,
            provider=request.provider or "",
            deep_thinking_enabled=request.deep_thinking_enabled,
        )
        return result
    except Exception:
        logger.exception("非流式对话失败")
        raise HTTPException(status_code=500, detail="回答生成失败，请稍后重试")
```

3e. 用下面的实现**整体替换** `/api/health` 端点：

```python
@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """返回 API 及依赖组件的当前状态；任何探测失败都降级为未就绪，不让 /api/health 崩。"""
    ready = False
    ready_error: str | None = "RAG 流水线未初始化"
    points_count = 0
    try:
        ready = rag_pipeline.ready
        ready_error = rag_pipeline.ready_error
        if ready:
            points_count = rag_pipeline.store.count()
    except Exception:
        logger.exception("健康检查：RAG 探测失败")

    pg_ok = False
    try:
        pg = _get_postgres_client()
        pg_ok = bool(pg and pg.health()["ok"])
    except Exception:
        logger.exception("健康检查：PostgreSQL 探测失败")

    graph_ok = False
    try:
        graph_ok = bool(Neo4jClient().health()["ok"])
    except Exception:
        logger.exception("健康检查：Neo4j 探测失败")

    return {
        "ready": ready,
        "ready_error": ready_error,
        "agent_ready": True,  # get_bidding_agent 懒加载，构造即视为可用
        "graph_ready": graph_ok,
        "pg_ready": pg_ok,
        "points_count": points_count,
        "latencies": {
            "neo4j_ms": -1,
            "postgres_ms": -1,
            "qdrant_check_ms": -1,
        },
    }
```

3f. `rate_limit_middleware` 与 422 处理器、lifespan、CORS 中间件**保持不动**。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_api_server.py tests/test_sse_events.py -q
```

预期：全绿（含既有 SSE 口径测试）。

- [ ] **Step 5: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/api/server.py bidding-agent/tests/test_api_server.py && git commit -m "$(cat <<'EOF'
feat(api): chat/stream 与 chat 接入 BiddingAgent，健康检查接真实组件

骨架版直连 LLM 且 done 帧恒空；现由 agent.chat_stream 产出完整 SSE 帧
（type 键口径与 tests/test_sse_events.py 一致），done 帧字段满足 CLAUDE.md 红线。
EOF
)"
```

---

### Task 5: server.py 会话/反馈端点适配 PostgresClient（TDD）

**Files:**
- Modify: `bidding-agent/api/server.py`
- Test: `bidding-agent/tests/test_api_server.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
class _FakePg:
    """PostgresClient 会话面 fake。"""

    def __init__(self, existing_id=-1):
        self.existing_id = existing_id
        self.saved = []
        self.feedback = []
        self.deleted = []

    def find_id_by_session(self, session_id):
        return self.existing_id

    def create_conversation(self, title, session_id=None):
        return 11

    def save_message(self, conversation_id, role, content, sources=None, tool_name=None):
        self.saved.append((conversation_id, role, content, sources, tool_name))
        return len(self.saved) * 10

    def load_messages(self, conversation_id, limit=100):
        return [
            {"id": 1, "role": "user", "content": "q", "sources": None, "tool_name": None, "created_at": "2026-01-01"},
            {"id": 2, "role": "assistant", "content": "a", "sources": [{"question": "q", "answer": "a", "score": 0.9}], "tool_name": "rag_search", "created_at": "2026-01-01"},
        ]

    def delete_conversation(self, conversation_id):
        self.deleted.append(conversation_id)
        return True

    def save_feedback(self, conversation_id, message_id, rating, comment=None):
        self.feedback.append((conversation_id, message_id, rating))
        return True


def test_save_conversation_appends_messages_and_returns_ids():
    pg = _FakePg(existing_id=-1)
    server._get_postgres_client = lambda: pg
    resp = TestClient(server.app).post("/api/conversations", json={
        "session_id": "s-1", "title": "第一问",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "sources": [{"question": "q", "answer": "a", "score": 1}], "toolName": "rag_search"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ids"] == [10, 20]
    assert pg.saved[0][1] == "user"
    assert pg.saved[1][4] == "rag_search"      # toolName -> tool_name
    assert pg.saved[1][3] is not None          # sources 透传


def test_save_conversation_503_without_pg():
    server._get_postgres_client = lambda: None
    resp = TestClient(server.app).post("/api/conversations", json={
        "session_id": "s-1", "messages": [{"role": "user", "content": "q"}],
    })
    assert resp.status_code == 503


def test_get_conversation_maps_tool_name():
    server._get_postgres_client = lambda: _FakePg(existing_id=11)
    resp = TestClient(server.app).get("/api/conversations/s-1")
    body = resp.json()
    assert body["messages"][1]["toolName"] == "rag_search"


def test_feedback_routes_to_save_feedback():
    pg = _FakePg(existing_id=11)
    server._get_postgres_client = lambda: pg
    resp = TestClient(server.app).post("/api/feedback", json={
        "session_id": "s-1", "message_id": 20, "rating": "up",
    })
    assert resp.status_code == 200
    assert pg.feedback == [(11, 20, "up")]


def test_feedback_without_pg_503():
    server._get_postgres_client = lambda: None
    resp = TestClient(server.app).post("/api/feedback", json={
        "session_id": "s-1", "message_id": 20, "rating": "up",
    })
    assert resp.status_code == 503
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_api_server.py -q
```

预期：新增用例 FAIL（骨架端点形状不匹配），Task 4 用例仍 PASS。

- [ ] **Step 3: 修改 `api/server.py`**

3a. 请求模型区：`ConversationRequest` 替换为：

```python
class MessageIn(BaseModel):
    """前端落库的一条消息（PersistedMessage 契约；thinking/计时不落库）。"""

    role: str = Field(..., description="user / assistant")
    content: str = Field(..., min_length=1)
    sources: Any = Field(default=None, description="参考来源列表或 null")
    toolCalled: bool = Field(default=False)
    toolName: str | None = Field(default=None)
    image: str | None = Field(default=None, description="暂无存储列，服务端忽略")
    imageName: str | None = Field(default=None, description="暂无存储列，服务端忽略")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role 必须为 user 或 assistant")
        return v


class ConversationRequest(BaseModel):
    """追加保存会话消息（messages 为本轮新增，通常 user+assistant 两条）。"""

    session_id: str = Field(..., min_length=1)
    title: str = Field(default="")
    messages: list[MessageIn] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """用户反馈：按 session 定位会话、message 定位消息。"""

    session_id: str = Field(..., min_length=1)
    message_id: int = Field(..., ge=1)
    rating: str = Field(..., description='枚举 "up" / "down"')

    @field_validator("rating")
    @classmethod
    def _validate_rating(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError('rating 必须为 "up" 或 "down"')
        return v
```

3b. **整体替换**"会话与反馈"一节的 5 个端点为：

```python
@app.post("/api/conversations")
def save_conversation(request: ConversationRequest) -> dict[str, Any]:
    """追加保存消息；会话不存在则先建。返回各消息的落库 id（feedback 要用 assistant 的）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")
    try:
        cid = pg.find_id_by_session(request.session_id)
        if cid < 0:
            cid = pg.create_conversation(title=request.title, session_id=request.session_id)
            if cid < 0:
                raise HTTPException(status_code=503, detail="会话保存失败")
        ids = [
            pg.save_message(
                cid, role=m.role, content=m.content, sources=m.sources,
                tool_name=(m.toolName if m.toolCalled else None),
            )
            for m in request.messages
        ]
        return {"status": "ok", "ids": ids}
    except HTTPException:
        raise
    except Exception:
        logger.exception("保存会话失败")
        raise HTTPException(status_code=503, detail="会话保存失败")


@app.get("/api/conversations")
def list_conversations(q: str = "") -> dict[str, Any]:
    """会话列表；q 非空时按标题过滤（列表量小，内存过滤足够）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")
    try:
        conversations = pg.list_conversations()
        if q:
            conversations = [c for c in conversations if q in (c.get("title") or "")]
        return {"conversations": conversations}
    except Exception:
        logger.exception("获取会话列表失败")
        raise HTTPException(status_code=503, detail="获取会话列表失败")


@app.get("/api/conversations/{session_id}")
def get_conversation(session_id: str) -> dict[str, Any]:
    """按 session 读回消息（tool_name 映射为前端契约的 toolName）；不存在时 messages 为 []。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")
    try:
        cid = pg.find_id_by_session(session_id)
        if cid < 0:
            return {"session_id": session_id, "messages": []}
        rows = pg.load_messages(cid)
        messages = [
            {"id": r.get("id"), "role": r["role"], "content": r["content"],
             "sources": r.get("sources"), "toolName": r.get("tool_name")}
            for r in rows
        ]
        return {"session_id": session_id, "messages": messages}
    except Exception:
        logger.exception("获取会话失败")
        raise HTTPException(status_code=503, detail="获取会话失败")


@app.delete("/api/conversations/{session_id}")
def delete_conversation(session_id: str) -> dict[str, str]:
    """删除单个会话（消息/反馈随级联与显式删除清理）。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="会话服务未就绪")
    try:
        cid = pg.find_id_by_session(session_id)
        if cid >= 0:
            pg.delete_conversation(cid)
        return {"status": "ok"}
    except Exception:
        logger.exception("删除会话失败")
        raise HTTPException(status_code=503, detail="删除会话失败")


@app.post("/api/feedback")
def save_feedback(request: FeedbackRequest) -> dict[str, str]:
    """保存反馈：session -> conversation_id，message_id 由前端从落库响应取得。"""
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="反馈服务未就绪")
    try:
        cid = pg.find_id_by_session(request.session_id)
        if cid < 0:
            raise HTTPException(status_code=503, detail="反馈保存失败")
        if not pg.save_feedback(cid, request.message_id, request.rating):
            raise HTTPException(status_code=503, detail="反馈保存失败")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("保存反馈失败")
        raise HTTPException(status_code=503, detail="反馈保存失败")
```

注意：删除骨架版的 `DELETE /api/conversations`（清空全部）端点，以及 `_get_postgres_client` 中对 `PostgreSQLClient` 的引用——改为：

```python
    try:
        return PostgresClient()
    except Exception:
        logger.exception("PostgreSQL 客户端初始化失败")
        return None
```

（去掉 `ImportError` 分支与 `_pg_unavailable_logged` 全局——dev 分支客户端必然存在，连接失败走 `health()` 探测。）

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest tests/test_api_server.py -q && uv run pytest -q 2>&1 | tail -2
```

预期：全绿。

- [ ] **Step 5: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/api/server.py bidding-agent/tests/test_api_server.py && git commit -m "$(cat <<'EOF'
feat(api): 会话/反馈端点适配 dev 的 PostgresClient 逐条消息存储

骨架按"整包 JSONB upsert"假设编写；实际客户端为逐条消息表。
POST /api/conversations 改为追加本轮消息并返回落库 id，feedback 按 session+message 定位。
契约变化将同步开发文档 §5。
EOF
)"
```

---

### Task 6: 前端依赖 + SSE 解析器（TDD, vitest）

**Files:**
- Modify: `frontend/package.json`（依赖 + test script）
- Create: `frontend/src/lib/sse.ts`
- Test: `frontend/src/lib/sse.test.ts`

- [ ] **Step 1: 安装依赖**

```bash
cd D:/bidding-agent/frontend && npm install react-markdown@^9.0.0 remark-gfm@^4.0.1 && npm install -D vitest@^2.1.0
```

- [ ] **Step 2: `package.json` scripts 加 `"test": "vitest run"`（在 `"preview"` 后）**

- [ ] **Step 3: 写失败测试 `src/lib/sse.test.ts`**

```ts
import { describe, expect, it, vi } from 'vitest'
import { createSseParser } from './sse'

describe('createSseParser', () => {
  it('解析单帧 JSON', () => {
    const onFrame = vi.fn()
    const p = createSseParser(onFrame)
    p.feed('data: {"type":"token","content":"你好"}\n\n')
    expect(onFrame).toHaveBeenCalledWith({ type: 'token', content: '你好' })
  })

  it('跨 chunk 拼接半帧', () => {
    const onFrame = vi.fn()
    const p = createSseParser(onFrame)
    p.feed('data: {"type":"to')
    p.feed('ken","content":"a"}\n\n')
    expect(onFrame).toHaveBeenCalledWith({ type: 'token', content: 'a' })
  })

  it('跳过 2KB padding 注释帧与 [DONE]', () => {
    const onFrame = vi.fn()
    const p = createSseParser(onFrame)
    p.feed(': ' + ' '.repeat(2048) + '\n\n')
    p.feed('data: [DONE]\n\n')
    expect(onFrame).not.toHaveBeenCalled()
  })

  it('坏帧 JSON 不抛异常、不回调', () => {
    const onFrame = vi.fn()
    const p = createSseParser(onFrame)
    p.feed('data: {broken json}\n\ndata: {"type":"reset"}\n\n')
    expect(onFrame).toHaveBeenCalledTimes(1)
    expect(onFrame).toHaveBeenCalledWith({ type: 'reset' })
  })
})
```

- [ ] **Step 4: 跑测试确认失败**

```bash
cd D:/bidding-agent/frontend && npm test
```

预期：FAIL（sse.ts 不存在）。

- [ ] **Step 5: 实现 `src/lib/sse.ts`**

```ts
/**
 * SSE 帧解析器（纯函数，便于单测）。
 * 协议见后端 docs/开发文档.md §6：data: {json}\n\n，首帧 2KB 注释 padding，尾帧 data: [DONE]。
 */
export type SseHandler = (frame: Record<string, unknown>) => void

export function createSseParser(onFrame: SseHandler) {
  let buf = ''

  const processBlock = (block: string) => {
    const dataLine = block.split('\n').find((l) => l.startsWith('data:'))
    if (!dataLine) return // 注释 padding（":" 开头）等无 data 行的块
    const payload = dataLine.slice(5).trim()
    if (!payload || payload === '[DONE]') return
    try {
      const json = JSON.parse(payload) as Record<string, unknown>
      if (json && typeof json === 'object' && 'type' in json) onFrame(json)
    } catch {
      // 坏帧静默丢弃：流式 JSON 截断不应打断整个会话
    }
  }

  return {
    feed(chunk: string) {
      buf += chunk
      let idx: number
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        processBlock(block)
      }
    },
  }
}
```

- [ ] **Step 6: 跑测试确认通过**

```bash
cd D:/bidding-agent/frontend && npm test
```

预期：4 passed。

- [ ] **Step 7: 提交**

```bash
cd D:/bidding-agent && git add frontend/package.json frontend/package-lock.json frontend/src/lib/sse.ts frontend/src/lib/sse.test.ts && git commit -m "$(cat <<'EOF'
feat(frontend): SSE 帧解析器 + vitest（padding/[DONE]/坏帧容错/跨 chunk 拼接）
EOF
)"
```

---

### Task 7: 前端网络层 `lib/api.ts`

**Files:**
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: 实现完整文件**

```ts
/**
 * 唯一网络层：后端 FastAPI :8001（SSE 直连，不走 Vite 代理）。
 * 端点契约见后端 docs/开发文档.md §5；SSE 协议见 §6。
 */
import { createSseParser } from './sse'

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8001'

export interface SourceItem {
  question: string
  answer: string
  score: number
}

export interface WebSourceItem {
  title?: string
  url?: string
  [key: string]: unknown
}

export interface DonePayload {
  sources: SourceItem[]
  web_sources: WebSourceItem[]
  tool_called: boolean
  tool_name: string
  elapsed_ms: number
  phase_times: [string, number][]
}

export type SseFrame =
  | { type: 'status'; content: string }
  | { type: 'token'; content: string }
  | { type: 'thinking'; content: string }
  | { type: 'reset' }
  | ({ type: 'done' } & DonePayload)
  | { type: 'error'; content: string }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** 状态码 -> 用户友好文案（不泄露内部细节） */
function friendlyError(status: number): string {
  if (status === 429) return '提问太频繁了，请稍等片刻再试。'
  if (status === 503) return '服务还未就绪（知识库或存储未连接），请稍后再试。'
  if (status === 400) return '请求参数有误。'
  return '服务暂时不可用，请稍后重试。'
}

export interface AskStreamOptions {
  question: string
  history: { role: string; content: string }[]
  web_search_enabled: boolean
  deep_thinking_enabled: boolean
  provider: string
  signal: AbortSignal
  onFrame: (frame: SseFrame) => void
}

export async function askStream(opts: AskStreamOptions): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: opts.question,
      history: opts.history,
      web_search_enabled: opts.web_search_enabled,
      deep_thinking_enabled: opts.deep_thinking_enabled,
      provider: opts.provider,
    }),
    signal: opts.signal,
  })
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
  if (!res.body) throw new ApiError(0, '连接中断，请重试。')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  const parser = createSseParser((raw) => opts.onFrame(raw as SseFrame))
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    parser.feed(decoder.decode(value, { stream: true }))
  }
}

export interface HealthInfo {
  ready: boolean
  ready_error?: string | null
  agent_ready: boolean
  graph_ready: boolean
  pg_ready: boolean
  points_count: number
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
  return res.json()
}

export interface PersistedMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[] | null
  toolCalled?: boolean
  toolName?: string | null
  image?: string | null
  imageName?: string | null
}

export interface ConversationItem {
  id: number
  session_id: string
  title: string
  created_at: string
}

export async function saveMessages(
  sessionId: string,
  title: string,
  messages: PersistedMessage[],
): Promise<{ status: string; ids: number[] }> {
  const res = await fetch(`${API_BASE}/api/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, title, messages }),
  })
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
  return res.json()
}

export async function listConversations(): Promise<{ conversations: ConversationItem[] }> {
  const res = await fetch(`${API_BASE}/api/conversations`)
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
  return res.json()
}

export interface StoredMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[] | null
  toolName?: string | null
}

export async function loadConversation(
  sessionId: string,
): Promise<{ session_id: string; messages: StoredMessage[] }> {
  const res = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`)
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
  return res.json()
}

export async function deleteConversation(sessionId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' },
  )
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
}

export async function sendFeedback(
  sessionId: string,
  messageId: number,
  rating: 'up' | 'down',
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message_id: messageId, rating }),
  })
  if (!res.ok) throw new ApiError(res.status, friendlyError(res.status))
}
```

- [ ] **Step 2: 类型检查**

```bash
cd D:/bidding-agent/frontend && npx tsc -b --noEmit 2>&1 | head -5 || npx tsc --noEmit -p tsconfig.app.json
```

预期：无错误（`import.meta.env` 需要 `vite-env.d.ts`，已有）。

- [ ] **Step 3: 提交**

```bash
cd D:/bidding-agent && git add frontend/src/lib/api.ts && git commit -m "$(cat <<'EOF'
feat(frontend): 网络层 api.ts（SSE 流式/健康/会话/反馈，统一错误文案）
EOF
)"
```

---

### Task 8: model.tsx 换真实 provider

**Files:**
- Modify: `frontend/src/contexts/model.tsx`

- [ ] **Step 1: 整体替换为**

```tsx
/* eslint-disable react-refresh/only-export-components -- 按设计规格: Provider/常量/hook 同文件 */
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

/** 后端支持的 provider（逐请求参数，见 ChatRequest.provider） */
export const MODEL_OPTIONS = ['deepseek', 'zhipu', 'vllm', 'ollama'] as const

const STORAGE_KEY = 'chat_llm_provider'

function loadInitial(): string {
  const saved = localStorage.getItem(STORAGE_KEY)
  return saved && (MODEL_OPTIONS as readonly string[]).includes(saved) ? saved : MODEL_OPTIONS[0]
}

interface ModelContextValue {
  model: string
  setModel: (m: string) => void
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState(loadInitial)
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, model)
  }, [model])
  return <ModelContext.Provider value={{ model, setModel }}>{children}</ModelContext.Provider>
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext)
  if (!ctx) throw new Error('useModel 必须在 <ModelProvider> 内使用')
  return ctx
}
```

（`MODEL_OPTIONS` 值变了：Layout 头部选择器与 Extract 页自动跟随，无需改它们。）

- [ ] **Step 2: 类型检查 + 提交**

```bash
cd D:/bidding-agent/frontend && npx tsc --noEmit -p tsconfig.app.json && cd D:/bidding-agent && git add frontend/src/contexts/model.tsx && git commit -m "$(cat <<'EOF'
feat(frontend): provider 选项换为后端真实四项并持久化 localStorage
EOF
)"
```

---

### Task 9: MarkdownMessage 组件 + Chat 页全量改造

**Files:**
- Create: `frontend/src/components/MarkdownMessage.tsx`
- Modify: `frontend/src/pages/Chat.tsx`（整体重写）

- [ ] **Step 1: 新建 `src/components/MarkdownMessage.tsx`**

```tsx
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/** AI 回答渲染：GFM 表格/列表/代码块；链接强制新开 + noopener。流式半截 markdown 也可安全渲染。 */
export default function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="text-[14px] leading-6 text-ink-700 [&_a]:text-brand-600 [&_a]:underline [&_code]:rounded [&_code]:bg-ink-50 [&_code]:px-1 [&_li]:ml-4 [&_ol]:list-decimal [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-ink-50 [&_pre]:p-3 [&_table]:mt-2 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-slate-200 [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-slate-200 [&_th]:bg-ink-50 [&_th]:px-2 [&_th]:py-1 [&_ul]:list-disc">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
```

- [ ] **Step 2: 整体重写 `src/pages/Chat.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react'
import {
  Copy, Globe, Plus, Send, Sparkles, Square, ThumbsDown, ThumbsUp, Trash2,
} from 'lucide-react'
import Button from '../components/Button'
import Card from '../components/Card'
import MarkdownMessage from '../components/MarkdownMessage'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'
import { useModel } from '../contexts/model'
import { QA_DATA } from '../data/qa'
import {
  ApiError, askStream, deleteConversation, fetchHealth, listConversations,
  loadConversation, saveMessages, sendFeedback,
  type ConversationItem, type DonePayload, type HealthInfo, type SourceItem,
  type WebSourceItem,
} from '../lib/api'

interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  status: 'pending' | 'typing' | 'done' | 'error'
  sources?: SourceItem[]
  webSources?: WebSourceItem[]
  toolCalled?: boolean
  toolName?: string
  thinking?: string
  elapsedMs?: number
  error?: string
  dbId?: number // 落库后的消息 id，反馈上报用
}

const WELCOME =
  '您好！我是招采智脑，可以为您解答招投标全流程问题。试试问我：单一来源采购公示被质疑怎么办？'
const HISTORY_MESSAGES = 10 // 最近 5 轮
const SESSION_KEY = 'chat_active_session' // 刷新后仍定位到上次会话

function genSessionId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}-${Math.random()}`
}

function Dots() {
  return (
    <span className="flex items-center gap-1 py-1.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-400"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </span>
  )
}

function ThinkingBlock({ text }: { text: string }) {
  return (
    <details className="mb-2 rounded-md border border-slate-200 bg-ink-50 px-3 py-2">
      <summary className="cursor-pointer text-[11px] font-medium text-ink-400">
        深度思考过程
      </summary>
      <div className="mt-1 whitespace-pre-wrap text-xs leading-5 text-ink-400">{text}</div>
    </details>
  )
}

function AiBubble({
  m, onUpdate, onFeedback,
}: {
  m: ChatMessage
  onUpdate: () => void
  onFeedback: (messageId: number, rating: 'up' | 'down') => void
}) {
  const toast = useToast()
  const [rated, setRated] = useState<boolean | null>(null)

  if (m.status === 'pending') {
    return (
      <div className="flex gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
          <Sparkles className="h-3.5 w-3.5 text-white" />
        </div>
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-2">
          <Dots />
        </div>
      </div>
    )
  }

  const copy = () => {
    navigator.clipboard
      .writeText(m.content)
      .then(() => toast.show('已复制到剪贴板', 'success'))
      .catch(() => toast.show('复制失败', 'error'))
  }

  const rate = (rating: 'up' | 'down') => {
    if (rated !== null) return
    if (!m.dbId) {
      toast.show('该回答未成功落库，暂时无法记录反馈', 'info')
      return
    }
    setRated(rating === 'up')
    onFeedback(m.dbId, rating)
  }

  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
        <Sparkles className="h-3.5 w-3.5 text-white" />
      </div>
      <div className="max-w-[75%] rounded-xl border border-slate-200 bg-white px-4 py-3">
        {m.thinking && <ThinkingBlock text={m.thinking} />}
        <MarkdownMessage content={m.content} />
        {m.status === 'typing' && (
          <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-brand-600 align-middle" />
        )}
        {m.status === 'error' && m.error && (
          <div className="mt-2 rounded-md bg-danger/10 px-3 py-2 text-xs text-danger">
            {m.error}
          </div>
        )}

        {m.status === 'done' && m.toolCalled && m.toolName && (
          <div className="mt-2 inline-flex items-center gap-1 rounded-md bg-brand-50 px-2 py-0.5 text-[11px] text-brand-700">
            <Globe className="h-3 w-3" />
            已调用工具：{m.toolName}
          </div>
        )}

        {m.status === 'done' && (m.sources?.length || m.webSources?.length) ? (
          <div className="mt-3 space-y-1 rounded-md border-l-2 border-brand-500 bg-brand-50 px-3 py-2">
            <div className="text-[11px] font-medium text-brand-700">参考来源</div>
            {m.sources?.map((s, i) => (
              <div key={i} className="truncate text-xs text-ink-700">
                [{i + 1}] {s.question}
              </div>
            ))}
            {m.webSources?.map((w, i) => (
              <a
                key={i}
                href={w.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block truncate text-xs text-brand-600 underline"
              >
                {w.title || w.url}
              </a>
            ))}
          </div>
        ) : null}

        {m.status === 'done' && (
          <div className="mt-3 flex items-center gap-1 border-t border-slate-100 pt-2">
            <button
              onClick={copy}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-ink-400 transition-colors hover:bg-ink-50 hover:text-ink-700"
            >
              <Copy className="h-3 w-3" />
              复制
            </button>
            {m.elapsedMs ? (
              <span className="px-2 text-[11px] text-ink-300">{(m.elapsedMs / 1000).toFixed(1)}s</span>
            ) : null}
            <button
              onClick={() => rate('up')}
              className={`ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors hover:bg-ink-50 ${
                rated === true ? 'text-ok' : 'text-ink-400 hover:text-ink-700'
              }`}
            >
              <ThumbsUp className="h-3 w-3" />
              有用
            </button>
            <button
              onClick={() => rate('down')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors hover:bg-ink-50 ${
                rated === false ? 'text-danger' : 'text-ink-400 hover:text-ink-700'
              }`}
            >
              <ThumbsDown className="h-3 w-3" />
              没帮助
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Chat() {
  const { model } = useModel()
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 0, role: 'assistant', content: WELCOME, status: 'done' },
  ])
  const [webSearch, setWebSearch] = useState(false)
  const [deepThinking, setDeepThinking] = useState(false)
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [activeSession, setActiveSession] = useState<string>(
    () => localStorage.getItem(SESSION_KEY) || genSessionId(),
  )
  const [currentSessionTitle, setCurrentSessionTitle] = useState('')

  const idRef = useRef(1)
  const streamIdRef = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const toast = useToast()

  useEffect(() => {
    localStorage.setItem(SESSION_KEY, activeSession)
  }, [activeSession])

  const scrollToBottom = () => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }

  const refreshConversations = () => {
    listConversations()
      .then((r) => setConversations(r.conversations))
      .catch(() => {}) // 会话列表拉取失败不阻塞对话
  }

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null))
    refreshConversations()
  }, [])

  const patch = (id: number, p: Partial<ChatMessage>) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...p } : m)))

  const persist = (
    sessionId: string,
    title: string,
    question: string,
    answer: string,
    done: DonePayload | null,
    aid: number,
  ) => {
    const pair = [
      { role: 'user' as const, content: question },
      {
        role: 'assistant' as const,
        content: answer,
        sources: done?.sources?.length ? done.sources : null,
        toolCalled: !!done?.tool_called,
        toolName: done?.tool_called ? done.tool_name : null,
      },
    ]
    saveMessages(sessionId, title, pair)
      .then((res) => {
        const dbId = res.ids?.[1]
        if (dbId && dbId > 0) patch(aid, { dbId })
        refreshConversations()
      })
      .catch(() => toast.show('会话保存失败，本次回答不会被记录', 'info'))
  }

  const send = (raw: string) => {
    const text = raw.trim()
    if (!text || streamIdRef.current !== null) return
    const uid = idRef.current++
    const aid = idRef.current++
    streamIdRef.current = aid
    const isFirstTurn = messages.filter((m) => m.id !== 0).length === 0
    if (isFirstTurn) setCurrentSessionTitle(text.slice(0, 30))
    setMessages((prev) => [
      ...prev,
      { id: uid, role: 'user', content: text, status: 'done' },
      { id: aid, role: 'assistant', content: '', status: 'pending' },
    ])
    setInput('')

    const history = messages
      .filter((m) => m.id !== 0 && m.status === 'done' && m.content)
      .slice(-HISTORY_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content }))

    const controller = new AbortController()
    abortRef.current = controller

    let content = ''
    let thinking = ''
    let donePayload: DonePayload | null = null

    askStream({
      question: text,
      history,
      web_search_enabled: webSearch,
      deep_thinking_enabled: deepThinking,
      provider: model,
      signal: controller.signal,
      onFrame: (f) => {
        if (f.type === 'token') {
          content += f.content
          patch(aid, { content, status: 'typing' })
        } else if (f.type === 'thinking') {
          thinking += f.content
          patch(aid, { thinking })
        } else if (f.type === 'reset') {
          content = ''
          thinking = ''
          patch(aid, { content: '', thinking: '' })
        } else if (f.type === 'done') {
          donePayload = f
          patch(aid, {
            sources: f.sources,
            webSources: f.web_sources,
            toolCalled: f.tool_called,
            toolName: f.tool_called ? f.tool_name : undefined,
            elapsedMs: f.elapsed_ms,
          })
        } else if (f.type === 'error') {
          patch(aid, { status: 'error', error: f.content })
        }
      },
    })
      .then(() => {
        if (!donePayload) {
          // 流正常结束但没有 done 帧（如服务端异常关闭）：给出兜底提示
          if (!content) patch(aid, { status: 'error', error: '连接中断，请重试。' })
        }
        patch(aid, { status: 'done' })
        persist(
          activeSession,
          currentSessionTitle || text.slice(0, 30),
          text, content, donePayload, aid,
        )
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') {
          patch(aid, { status: 'done' })
          persist(activeSession, currentSessionTitle || text.slice(0, 30), text, content, donePayload, aid)
        } else {
          const msg = err instanceof ApiError ? err.message : '网络异常，请检查后端服务是否启动。'
          patch(aid, { status: 'error', error: msg, content })
        }
      })
      .finally(() => {
        streamIdRef.current = null
        abortRef.current = null
      })
  }

  const stop = () => abortRef.current?.abort()

  const switchSession = (sid: string) => {
    if (streamIdRef.current !== null) return
    loadConversation(sid)
      .then((r) => {
        setActiveSession(sid)
        const restored: ChatMessage[] = r.messages.map((m) => ({
          id: idRef.current++,
          role: m.role,
          content: m.content,
          status: 'done' as const,
          sources: m.sources ?? undefined,
          toolCalled: !!m.toolName,
          toolName: m.toolName ?? undefined,
          dbId: m.id,
        }))
        setMessages([{ id: 0, role: 'assistant', content: WELCOME, status: 'done' }, ...restored])
        setCurrentSessionTitle(
          conversations.find((c) => c.session_id === sid)?.title ?? '',
        )
      })
      .catch(() => toast.show('会话读取失败', 'error'))
  }

  const newSession = () => {
    if (streamIdRef.current !== null) return
    setActiveSession(genSessionId())
    setCurrentSessionTitle('')
    setMessages([{ id: 0, role: 'assistant', content: WELCOME, status: 'done' }])
  }

  const removeSession = (sid: string) => {
    deleteConversation(sid)
      .then(() => {
        refreshConversations()
        if (sid === activeSession) newSession()
      })
      .catch(() => toast.show('删除失败', 'error'))
  }

  const feedback = (messageId: number, rating: 'up' | 'down') => {
    sendFeedback(activeSession, messageId, rating)
      .then(() => toast.show('感谢反馈，已记录', 'info'))
      .catch(() => toast.show('反馈保存失败', 'info'))
  }

  const streaming = streamIdRef.current !== null

  return (
    <div>
      <PageHeader title="智能问答" description="基于企业知识库的招投标采购智能问答助手" />

      {health && !health.ready && (
        <div className="mb-3 rounded-lg bg-warn/10 px-4 py-2 text-xs text-warn">
          知识库未就绪（{health.ready_error || '未知原因'}），当前提问可能无法检索到参考资料。
        </div>
      )}

      <div className="flex h-[calc(100vh-160px)] min-h-[520px] gap-4">
        <div className="flex flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white">
          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-5">
            {messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[75%] whitespace-pre-wrap rounded-xl bg-brand-600 px-4 py-2.5 text-[14px] leading-6 text-white">
                    {m.content}
                  </div>
                </div>
              ) : (
                <AiBubble
                  key={m.id}
                  m={m}
                  onUpdate={scrollToBottom}
                  onFeedback={feedback}
                />
              ),
            )}
          </div>

          <div className="border-t border-slate-200 bg-white p-4">
            <div className="mb-2 flex items-center gap-4 text-xs text-ink-500">
              <label className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={webSearch}
                  onChange={(e) => setWebSearch(e.target.checked)}
                  className="accent-brand-600"
                />
                联网搜索
              </label>
              <label className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={deepThinking}
                  onChange={(e) => setDeepThinking(e.target.checked)}
                  className="accent-brand-600"
                />
                深度思考
              </label>
              <span className="ml-auto text-ink-300">模型：{model}</span>
            </div>
            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send(input)
                  }
                }}
                rows={2}
                placeholder="请输入您的问题，Enter 发送，Shift+Enter 换行"
                className="max-h-32 flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-[13px] leading-5 text-ink-900 outline-none transition-colors placeholder:text-ink-300 focus:border-brand-500"
              />
              {streaming ? (
                <Button onClick={stop} icon={<Square className="h-3.5 w-3.5" />}>
                  停止
                </Button>
              ) : (
                <Button onClick={() => send(input)} icon={<Send className="h-3.5 w-3.5" />}>
                  发送
                </Button>
              )}
            </div>
          </div>
        </div>

        <aside className="w-[300px] shrink-0 space-y-4 overflow-y-auto">
          <Card
            title="历史会话"
            subtitle="点击切换，刷新不丢"
            action={
              <button
                onClick={newSession}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-brand-600 hover:bg-brand-50"
              >
                <Plus className="h-3 w-3" />
                新会话
              </button>
            }
          >
            <div className="space-y-1">
              {conversations.length === 0 && (
                <div className="py-2 text-center text-xs text-ink-300">暂无历史会话</div>
              )}
              {conversations.map((c) => (
                <div
                  key={c.session_id}
                  className={`group flex items-center gap-1 rounded-lg border px-3 py-2 text-xs transition-colors ${
                    c.session_id === activeSession
                      ? 'border-brand-300 bg-brand-50 text-brand-700'
                      : 'border-slate-200 text-ink-700 hover:border-brand-200 hover:bg-brand-50'
                  }`}
                >
                  <button
                    onClick={() => switchSession(c.session_id)}
                    className="flex-1 truncate text-left"
                    title={c.title}
                  >
                    {c.title || '未命名会话'}
                  </button>
                  <button
                    onClick={() => removeSession(c.session_id)}
                    className="hidden shrink-0 rounded p-1 text-ink-300 hover:text-danger group-hover:block"
                    title="删除会话"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </Card>

          <Card title="推荐问题" subtitle="点击自动提问">
            <div className="space-y-2">
              {QA_DATA.slice(0, 6).map((q) => (
                <button
                  key={q.id}
                  onClick={() => send(q.question)}
                  className="block w-full rounded-lg border border-slate-200 px-3 py-2 text-left text-xs leading-5 text-ink-700 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700"
                >
                  {q.question}
                </button>
              ))}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  )
}
```

注意：
- 原"知识库覆盖"卡片删除（mock 数据，被真实健康横幅替代）。
- 原 `useTypewriter` hook 与 `lib/matcher.ts` 不再被 Chat 使用；`matcher.ts` 若再无其他引用则删除，`useTypewriter` 若无其他引用同样删除（执行时 grep 确认）。
- `Card` 组件若无 `action` prop：给 `frontend/src/components/Card.tsx` 增加可选 prop `action?: ReactNode`，渲染在标题行右侧（读原文件后最小改动）。
- 用到的 Tailwind 类 `text-ok`/`text-danger`/`bg-danger/10`/`text-warn`/`bg-warn/10` 若主题未定义，检查 `tailwind.config.js` 中对应色名，替换为既有等价类（如 `text-emerald-600`/`text-red-500`/`bg-amber-50 text-amber-700`），保持视觉一致。

- [ ] **Step 3: 类型检查与构建**

```bash
cd D:/bidding-agent/frontend && npx tsc --noEmit -p tsconfig.app.json && npm run build 2>&1 | tail -5
```

预期：无类型错误，build 产物生成。

- [ ] **Step 4: 提交**

```bash
cd D:/bidding-agent && git add frontend/src/components/MarkdownMessage.tsx frontend/src/pages/Chat.tsx frontend/src/components/Card.tsx frontend/src/lib frontend/src/hooks frontend/src/components/Layout.tsx && git commit -m "$(cat <<'EOF'
feat(frontend): Chat 页接入 SSE 流式问答与会话/反馈持久化

替换 mock 匹配逻辑：流式渲染（GFM 表格）、thinking 折叠、来源/工具徽标、
provider 逐请求携带、联网/深度思考开关、历史会话侧栏、点赞点踩上报、停止按钮。
EOF
)"
```

---

### Task 10: 端到端联调验证

- [ ] **Step 1: 起后端**

```bash
cd D:/bidding-agent/bidding-agent && uv run python main.py api
```

（后台运行；日志确认 `Uvicorn running on http://0.0.0.0:8001`。若 `.env` 缺必需配置按 CLAUDE.md 排障表处理。）

- [ ] **Step 2: curl 验证健康与 SSE**

```bash
curl -s http://localhost:8001/api/health
curl -s -N -X POST http://localhost:8001/api/chat/stream -H "Content-Type: application/json" -d '{"question":"单一来源采购公示被质疑怎么办？"}' | head -c 2000
```

预期：health 返回 JSON；SSE 有 `type: status/token` 帧、中文 token 流、`type: done` 帧（sources/tool_name/elapsed_ms/phase_times 齐全）与 `data: [DONE]`。

- [ ] **Step 3: 起前端**

```bash
cd D:/bidding-agent/frontend && npm run dev
```

- [ ] **Step 4: 浏览器黄金路径核对（用 gstack 无头浏览器或请用户人工核验）**

逐项：流式逐字输出 → GFM 表格渲染 → 来源/工具徽标 → 耗时 → 历史会话出现并可切换/删除 → 刷新后会话还在 → 点赞后 POST /api/feedback 200 → 停止按钮生效 → provider 切换后请求体 provider 变化（DevTools Network 确认）→ 联网开关打开后请求体 `web_search_enabled: true`。

- [ ] **Step 5: 回归**

```bash
cd D:/bidding-agent/bidding-agent && uv run pytest -q 2>&1 | tail -2 && cd ../frontend && npm test && npm run build 2>&1 | tail -2
```

预期：全绿。

---

### Task 11: 文档同步

**Files:**
- Modify: `bidding-agent/docs/技术栈.md`、`bidding-agent/docs/开发文档.md`、`CLAUDE.md`

- [ ] **Step 1: 《技术栈.md》§1/§8 加回渲染与测试依赖**（`react-markdown ^9.0.0 + remark-gfm ^4.0.1`、devDependencies `vitest ^2.1.0`），§8 的"尚未接入后端 API"引注改为"已接入后端 :8001（SSE 直连）"。

- [ ] **Step 2: 《开发文档.md》§5**：会话/反馈端点契约改为本计划 Task 5 的实现（POST /api/conversations 追加语义 + `ids` 响应、DELETE 清空端点删除、POST /api/feedback 改 `{session_id, message_id, rating}`、GET conversations/{session_id} 的 `toolName` 映射）；**§2.3 配置表**补 `CORS_ORIGINS` 键。

- [ ] **Step 3: 《CLAUDE.md》**：状态段"前端……尚未接入后端 API"改为"已接入"；排障表"前端流式整段一次显示（接入 SSE 后）"条目改为现状表述。

- [ ] **Step 4: 提交**

```bash
cd D:/bidding-agent && git add CLAUDE.md bidding-agent/docs/技术栈.md bidding-agent/docs/开发文档.md && git commit -m "$(cat <<'EOF'
docs: 同步接口联调后的实际形态（前端已接入、会话/反馈契约适配、CORS 键）
EOF
)"
```

---

## 执行注意

- Task 1 的 merge 冲突解决是全程风险最高的一步；config.py 必须保留 dev 的 `rerank_enabled`/`qdrant_distance`（`src/rag/pipeline.py` 读取，缺了会 AttributeError）。
- 全程在 dev 分支；**不要 push**（用户明确要求联调完成后再推）。
- 前端 `npm install` 会更新 `package-lock.json`——那是本次的合法变更，正常提交；但开始前工作区已有的 lock 变更（队友的）会被覆盖合并，若 Step 遇到 lock 冲突，报告并请用户确认。
