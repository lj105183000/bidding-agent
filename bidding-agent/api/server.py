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
import os
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.api_access import require_logs_access
from src.config import settings
from src.health import build_health, mark_warming_started, mark_warmup_done
from src.monitoring import build_metrics, render_prometheus, safe_record
from src.logging_config import (
    bind_log_context,
    log_buffer,
    reset_log_context,
)
from src.rate_limiter import (
    build_limiters,
    check_request,
)

logger = logging.getLogger(__name__)

# 模块级单例：限流器
def _rate_overrides() -> dict[str, tuple[int, int]]:
    """从环境变量读配额覆盖。

    ★ 为什么允许覆盖
    -------------
    不同部署的合理配额差别很大：本地单机演示希望宽松（一个人调试），
    公网多租户必须收紧。把三个数字写死在代码里，运维只能改代码。

    只覆盖**显式设置了**的层；未设置时用 `src.rate_limiter.DEFAULT_TIERS`
    的默认值。窗口与请求数分开读，允许"只调请求数、保持 60 秒窗口"。
    """
    overrides: dict[str, tuple[int, int]] = {}
    for tier, default_window in (("heavy", 60), ("llm", 60), ("context", 60)):
        raw_max = os.getenv(f"RATE_LIMIT_{tier.upper()}", "").strip()
        raw_win = os.getenv(f"RATE_LIMIT_{tier.upper()}_WINDOW", "").strip()
        if not raw_max and not raw_win:
            continue
        try:
            max_req = int(raw_max) if raw_max else {"heavy": 10, "llm": 30, "context": 120}[tier]
            window = int(raw_win) if raw_win else default_window
        except ValueError:
            logger.warning(
                "限流配额环境变量取值非法（RATE_LIMIT_%s=%r, _WINDOW=%r），该层用默认值",
                tier.upper(), raw_max, raw_win,
            )
            continue
        if max_req > 0 and window > 0:
            overrides[tier] = (max_req, window)
    return overrides


#: 分层限流器：`{层名: RateLimiter}`。`none` 层不在其中（= 不限流）。
#:
#: 环境变量覆盖（都可选，逗号分隔的数字）：
#:     RATE_LIMIT_HEAVY / RATE_LIMIT_HEAVY_WINDOW
#:     RATE_LIMIT_LLM   / RATE_LIMIT_LLM_WINDOW
#:     RATE_LIMIT_CONTEXT / RATE_LIMIT_CONTEXT_WINDOW
rate_limiters = build_limiters(_rate_overrides())

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

# 图谱结构化查询服务（供前端"知识图谱"页直查，不经过 LLM）。
# 单独 try：它只是可选能力，导入失败不该影响对话等主链路。
try:
    from src.database import graph_data
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("图谱查询服务导入失败，/api/graph/* 端点将返回 503")
    graph_data = None

# 图片比价服务（视觉提取商品 → 多平台检索报价 → 择优建议）
try:
    from src.services import price_compare as price_compare_service
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("比价服务导入失败，/api/price-compare 端点将返回 503")
    price_compare_service = None

# 标讯智能提取服务（公告文本 / 截图 → 17 个结构化字段）
try:
    from src.services import extract as extract_service
    from src.constants import bid_schema
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("标讯提取服务导入失败，/api/extract 端点将返回 503")
    extract_service = None
    bid_schema = None

# 集采分析服务（PostgreSQL 多维聚合：分位数 / 排行 / 报价策略 / 数据质量）
try:
    from src.services import analysis as analysis_service
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("集采分析服务导入失败，/api/analysis/* 端点将返回 503")
    analysis_service = None

# 子图抽取（供前端知识图谱关系可视化）
try:
    from src.database import subgraph as subgraph_service
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("子图服务导入失败，/api/graph/subgraph 端点将返回 503")
    subgraph_service = None

# 竞争对手分析（供应商格局 / 单供应商画像 / 采购人格局 / 市场集中度）
try:
    from src.services import competitor as competitor_service
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("竞争对手分析服务导入失败，/api/competitor/* 端点将返回 503")
    competitor_service = None

# 物料识别（物料名归一化 / 模糊匹配 / 详情 / 标准化建议）
try:
    from src.services import material as material_service
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("物料识别服务导入失败，/api/material/* 端点将返回 503")
    material_service = None

# 招标文件解析与抽取（PDF/Word → 章节分块 → 结构化字段）
try:
    from src.services import document_parser as document_parser_service
    from src.services import tender_extractor as tender_extractor_service
    from src.services import tender_store as tender_store_service
    from src.services import change_tracker as change_tracker_service
    from src.services import qualification as qualification_service
    from src.services import report as report_service
    from src.services import compliance as compliance_service
    from src.services import web_fallback as web_fallback_service
    from src.constants import tender_schema
except Exception:  # pragma: no cover - 模块未就绪
    logger.exception("招标文件解析服务导入失败，/api/tender/* 端点将返回 503")
    document_parser_service = None
    tender_extractor_service = None
    tender_store_service = None
    change_tracker_service = None
    qualification_service = None
    report_service = None
    compliance_service = None
    web_fallback_service = None
    tender_schema = None


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
    session_id: str = Field(
        default="",
        description=(
            "会话标识（可选）。用于**历史压缩**的滚动水位线分桶——"
            "同一 session_id 跨轮复用同一份早期对话摘要，避免每轮重复压缩。"
            "留空时压缩仍可用，只是状态按「无会话」共享。"
        ),
    )
    web_search_enabled: bool = Field(
        default=False, description="开启后动态挂载联网搜索工具（旧字段，保留兼容）"
    )
    web_search_mode: str = Field(
        default="auto",
        description=(
            "联网搜索模式：auto=本地知识库不够用时自动联网（默认）；"
            "on=强制联网；off=只用本地知识库"
        ),
    )

    @field_validator("web_search_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        v = (v or "auto").strip().lower()
        if v not in ("auto", "on", "off"):
            raise ValueError('web_search_mode 只能是 "auto" / "on" / "off"')
        return v
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


class GraphQueryRequest(BaseModel):
    """知识图谱直查请求（前端"知识图谱"页使用，不经过 LLM）。

    与 `/api/chat/stream` 的区别：那条路让 Agent 自己决定查什么；
    这条路是用户在界面上**显式指定**查询类型与参数，返回结构化 JSON 供表格渲染。

    `query_type` 不用 Literal 枚举：合法值来自 rag_tools._GRAPH_QUERY_TYPES，
    写死在模型里会与那边漂移；且非法值由服务层抛出带"可用清单"的错误，
    比 Pydantic 的通用 422 文案对使用者更有帮助。
    """

    query_type: str = Field(..., min_length=1, description="查询类型，见 GET /api/graph/types")
    entity: str = Field(default="", description="实体名称（search_entity / items_by_entity / entity_detail / supplier_by_subject 必填）")
    limit: int = Field(default=20, ge=1, le=200, description="返回条数上限")
    keyword: str = Field(default="", description="entity 的别名，便于前端复用同一个输入框")


class PriceCompareRequest(BaseModel):
    """图片比价请求。

    与 `/api/vision` 的区别：那条路只做「图片 → 文字解析」；
    这条路是「图片 → 商品信息 → 多平台报价 → 择优建议」的完整编排。

    `hint` 用于补充图片没体现的信息（如"这是包装盒侧面，型号是 XXX"），
    会追加到视觉模型的提示词里。
    """

    image_base64: str = Field(..., min_length=1, description="图片 base64 或完整 data URL")
    hint: str = Field(default="", description="用户补充说明（可选）")


class ExtractRequest(BaseModel):
    """标讯提取请求（公告文本或公告截图 → 17 个结构化字段）。

    `source` 决定走哪条路：
        text  → 大模型按 schema 抽 JSON（快、准，适合从网站复制的公告正文）
        image → 视觉模型抽 JSON（适合截图，但慢一些）

    `save=True` 时把结果写入 `bidding_records`。默认不入库——
    提取是探索性操作，用户往往要先看清楚结果再决定是否保存。
    """

    source: str = Field(default="text", description='输入类型："text" 或 "image"')
    text: str = Field(default="", description="source=text 时的公告文本")
    image_base64: str = Field(default="", description="source=image 时的图片 base64 / data URL")
    hint: str = Field(default="", description="图片模式的补充说明（可选）")
    save: bool = Field(default=False, description="是否写入 bidding_records 表")


class PriceProfileRequest(BaseModel):
    """报价策略查询请求。

    `exclude_suspicious=True` 时排除品类与金额明显不匹配的可疑记录
    （如服务类项目记成数亿元），避免结论被脏数据带偏。
    """

    subject: str = Field(..., min_length=1, description="标的物关键词，支持模糊匹配")
    exclude_suspicious: bool = Field(default=True, description="是否排除可疑金额记录")
    top_n: int = Field(default=10, ge=1, le=50, description="返回的历史案例条数")


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


def _resolve_web_mode(request: "ChatRequest") -> bool:
    """把三态模式解析成"是否强制挂载联网工具"的布尔值。

    auto → False（先不挂；由 Agent 判断本地不够时自己开启）
    on   → True （强制挂载）
    off  → False（不挂；且 auto_web_fallback 也是 False，故不会自动开启）

    兼容：`web_search_mode` 缺省是 "auto"，但旧调用方只传
    `web_search_enabled=True` 时期望"强制联网"，故 on/off 之外还要看旧字段。
    """
    mode = (getattr(request, "web_search_mode", None) or "auto").strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto：尊重旧的布尔字段（旧的 True 等价于 on）
    return bool(getattr(request, "web_search_enabled", False))


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


def _preload_models() -> None:
    """后台预热本地模型，消除首次请求的冷启动停顿。

    为什么需要这一步
    ----------------
    `src/rag/embedder.py` 的两个模型都是**懒加载**（首次实际使用时才加载）。
    这个设计对测试与 CI 是对的，但在真实使用中会造成一次很难解释的停顿：

        - 嵌入模型 `bge-small-zh-v1.5`：首次加载约 20s
        - 精排模型 `bge-reranker-base`：首次加载实测约 80s（权重 1.1GB）

    而精排模型的触发点是 `rag_tools._rerank_web_results`——**每次联网搜索都会走它**，
    且不看 `RERANK_ENABLED`（那个开关只控制 RAG 检索链路）。
    结果是：用户打开「联网搜索」后的**第一次提问会卡 80 秒**，看起来像服务挂了。

    为什么放在后台线程而不是启动时同步加载
    --------------------------------------
    同步加载会让 `uvicorn` 的 startup 阻塞 100 秒，健康检查在此期间一律超时，
    部署脚本与容器探针会判定"启动失败"。放后台线程后：
      - 服务立即可用（知识库/图谱/业务库三个数据源不受影响）
      - 模型在后台就绪，第一次联网搜索通常已经能命中已加载的模型
      - 预热失败只记日志，不影响服务本身（懒加载仍会在首次使用时兜底重试）
    """
    def _warm() -> None:
        # ★ 标记预热状态：健康检查据此区分"还在预热"与"真的坏了"。
        #
        # 不标记的后果实测过：预热要 100 秒（嵌入 20s + 精排 80s），
        # 期间三个数据源其实都正常，但运维看到"接口慢"却没有任何线索
        # 说明是预热未完成——只能去翻日志猜。
        mark_warming_started()
        try:
            from src.rag.embedder import embedder, reranker

            logger.info("后台预热嵌入模型…")
            embedder.encode_query("预热")  # 触发加载，顺带验证编码链路可用
            logger.info("嵌入模型预热完成")

            logger.info("后台预热精排模型（联网搜索首次调用会用到，约 1~2 分钟）…")
            reranker.model  # noqa: B018 - 取属性即触发懒加载
            logger.info("精排模型预热完成，联网搜索首次调用不再有冷启动停顿")
        except Exception as exc:
            logger.exception("模型预热失败（不影响服务，首次使用时仍会按懒加载重试）")
            # 失败也标记结束，但带上原因：健康检查会报 degraded 而不是一直 warming
            mark_warmup_done(error=f"{type(exc).__name__}: {exc}"[:200])
            return
        mark_warmup_done()

    threading.Thread(target=_warm, name="model-preload", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭生命周期。"""
    logger.info("API 服务启动，监听 %s:%s", settings.api_host, settings.api_port)
    _preload_models()
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


# ---------------------------------------------------------------------------
# 前端静态托管（同源部署）
# ---------------------------------------------------------------------------
#
# ★ 为什么由后端托管（而不是再起一个 nginx / vite preview）
# ------------------------------------------------------
# 本项目此前**没有生产托管路径**：`docker-compose.yml` 明确写着
# "后端 uvicorn 与前端 vite 保持本地进程运行"，即前端一直跑的是 **dev server**。
# 这带来三个问题：① dev server 不做压缩/缓存协商，也不该对外；② 两个端口
# 两套来源，必须维护 CORS 白名单；③ 部署时得再多一个常驻进程。
#
# 由后端托管则一举解决：**同源**（不再需要跨域配置）、gzip 由 Starlette 的
# 静态文件响应处理、部署只有一个进程。
#
# ★ 为什么默认只在 dist 存在时启用
# -----------------------------
# 开发时前端跑 vite dev server（有 HMR），此时 `dist/` 可能是旧的或不存在。
# 若强行挂载，开发者会看到**过期页面**却以为是代码没生效 —— 这类"看起来
# 像缓存问题"的坑最难查。故：dist 不存在就完全不挂载（等于没这功能）。
#
# 可用 `FRONTEND_DIST` 指定目录；设 `FRONTEND_DIST=off` 可显式关闭。
#
# ★ 候选路径要**逐个探测**，不能写死相对层级（实测踩到并修）
# ------------------------------------------------------
# 第一版写的是 `Path(__file__).resolve().parents[1] / "frontend" / "dist"`，
# 本机实际布局是：
#
#     C:\Users\10518\bidding-agent\bidding-agent\api\server.py   ← __file__
#     parents[1] = C:\Users\10518\bidding-agent\bidding-agent      ← 工程根
#     而前端在   C:\Users\10518\bidding-agent\frontend            ← 工程根的**上一级**
#
# 于是解析成 `...\bidding-agent\bidding-agent\frontend\dist`（不存在）→
# 因为下面"dist 不存在就完全不挂载"的策略，**静默不生效**：
# 用户按说明打开 8001 看到的是 404，看起来就是"前端打不开"。
# 教训：这种"找不到就安静跳过"的兜底必须把**探测过的路径**打出来，
# 否则配置错与"没构建"这两种情况在日志里长得一模一样。
def _resolve_frontend_dist() -> tuple[Path, list[Path]]:
    """返回 (选中的目录, 探测过的候选列表)。选不中时第二个返回值用于诊断日志。"""
    env_value = (os.getenv("FRONTEND_DIST") or "").strip()
    if env_value:
        return Path(env_value), [Path(env_value)]

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "frontend" / "dist",          # 前端与后端同目录（子模块布局）
        project_root.parent / "frontend" / "dist",   # 前端在工程根的上一级（本机布局）
        Path.cwd() / "frontend" / "dist",            # 从哪个目录启动的
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate, candidates
    return candidates[0], candidates


_FRONTEND_DIST, _FRONTEND_CANDIDATES = _resolve_frontend_dist()
_FRONTEND_DISABLED = os.getenv("FRONTEND_DIST", "").strip().lower() in (
    "off", "0", "false", "no",
)
# 目录里必须有 index.html 才算可用的构建产物（空目录/半截构建都不算）
_FRONTEND_READY = not _FRONTEND_DISABLED and (_FRONTEND_DIST / "index.html").is_file()

if _FRONTEND_READY:
    logger.info("前端静态资源已就绪：%s（路由在**文件末尾**注册，见那里的说明）",
                _FRONTEND_DIST)
elif _FRONTEND_DISABLED:
    logger.info("前端静态资源托管已被 FRONTEND_DIST 显式关闭")
else:
    # WARNING 而不是 INFO：用户按文档打开 8001 却看到 404 时，
    # 这条日志要能直接回答"为什么没托管"（把探测过的路径都列出来）
    logger.warning(
        "未托管前端静态资源：以下路径都没有 index.html → %s；"
        "开发时由 vite dev server（5173）提供页面属正常，"
        "若要同源访问请在 frontend/ 下执行 npm run build 或用 FRONTEND_DIST 指定",
        "、".join(str(p) for p in _FRONTEND_CANDIDATES),
    )


@app.middleware("http")
async def log_context_middleware(request: Request, call_next):
    """给每个请求绑定 `request_id` 等结构化日志上下文。

    ★ 为什么这层比"格式改成 JSON"更重要
    --------------------------------
    只把格式改成 JSON 而不绑上下文，得到的是一堆**孤立的** JSON 行——
    仍然无法回答"这条报错属于哪个请求"。

    绑上之后，**这个请求链路里所有模块的日志都自动带上它**，
    不需要每个 `logger.info` 手动传参（322 处调用一处都不用改）。
    前端出问题时，把响应头里的 `X-Request-Id` 报过来，
    就能直接捞出该请求的完整日志。

    ★ 为什么用 contextvars 而不是 threading.local
    ------------------------------------------
    `/api/chat/stream` 是 SSE（异步生成器），且工具层是 4 线程并行。
    `threading.local` 在同一个线程服务多个协程时会串味，
    `contextvars` 对同步与异步都正确。

    ★ 顺序：这层要在限流之前
    ----------------------
    限流直接返回 429 时也要能看到"是谁被限了"，故上下文必须在它之前建立。
    FastAPI 的中间件是**后加的先执行**，故此函数定义在 `rate_limit_middleware`
    **之后**（见下方定义顺序）。

    ★ 顺序陷阱（实测踩过，务必保持）
    ------------------------------
    `reset_log_context` 必须在**所有本请求的日志都发完之后**才执行。

    第一版写成：

        try:
            response = await call_next(request)
        finally:
            reset_log_context(token)      # ← 先还原
        elapsed_ms = ...
        logger.info("请求完成", extra=extra)   # ← 再记日志 → request_id 已经没了

    结果：**访问日志里一条 request_id 都没有**（实测 0/11），
    而最小复现却是通的 —— 因为那个复现是在 finally **之前**读的上下文。
    这个 bug 很隐蔽：其它字段（status / elapsed_ms）都正常，
    只有上下文悄悄丢了，看起来像"中间件没生效"。

    故现在的结构是：访问日志在 try 内发，`finally` 只负责还原。
    """
    # 优先用上游传来的 id（网关/前端已生成时保持链路一致），否则自己生成
    request_id = (
        request.headers.get("X-Request-Id")
        or request.headers.get("X-Correlation-Id")
        or uuid.uuid4().hex[:16]
    )
    token = bind_log_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    t0 = time.monotonic()
    try:
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        # 回传 request_id：前端/用户报障时能直接把 id 给出来
        response.headers["X-Request-Id"] = request_id

        # 访问日志：只记一行汇总，不记请求体（可能含采购文件内容与图片 base64）。
        extra = {"elapsed_ms": elapsed_ms, "status": response.status_code}
        if response.status_code >= 500:
            logger.error("请求完成（服务端错误）", extra=extra)
        elif response.status_code >= 400:
            logger.warning("请求完成（客户端错误）", extra=extra)
        elif elapsed_ms >= SLOW_REQUEST_MS:
            # 慢请求单独提级：默认 INFO 会淹没在大量正常日志里，
            # 而"哪个接口慢"是移动网络下最常被问的问题
            logger.warning("慢请求", extra=extra)
        else:
            logger.info("请求完成", extra=extra)
        return response
    except Exception:
        # 未处理异常：记一条带堆栈的 error，再抛给 FastAPI 的处理器。
        # 不在这里构造响应——保持既有的错误处理链不变，避免行为漂移。
        logger.exception(
            "请求处理失败",
            extra={"elapsed_ms": int((time.monotonic() - t0) * 1000)},
        )
        raise
    finally:
        # ★ 必须在 finally 里还原，且**放在最后**。不还原则同一个 worker 的
        #   后续请求会继承上一个请求的 request_id —— 那比没有还糟：
        #   日志会指向**错误的**请求，比缺失更难排查。
        reset_log_context(token)


#: 超过这个耗时就记 WARNING。
#:
#: 取 3 秒：项目里问答类接口本身就慢（实测 P95 约 17 秒），
#: 卡太低会让所有问答都变成"慢请求"警告，反而淹没真正的异常。
#: 3 秒能筛出"本该快却慢了"的接口（列表、健康检查、图谱直查）。
SLOW_REQUEST_MS = 3000

#: 进程启动时刻（`/api/health/live` 算 uptime 用）。
#:
#: 放在模块级而不是 `src.health` 里：存活探针刻意**不 import 任何业务模块**，
#: 避免"探活本身因为 import 了重依赖而变慢或失败"——那会让存活探针
#: 误判进程已死并触发重启。
_health_started_at = time.time()


@app.get("/api/logs", dependencies=[Depends(require_logs_access)])
async def get_logs(
    level: str = "",
    logger_name: str = Query("", alias="logger"),
    request_id: str = "",
    limit: int = 100,
):
    """查询进程内最近日志（结构化后最实用的一步）。

    过滤参数都可选，可组合：
        level       最低级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        logger      按 logger 名**前缀**匹配（`src.rag` 命中 `src.rag.pipeline`）
        request_id  按请求链路过滤——排障时最常用
        limit       返回条数上限（默认 100，最多 500）

    返回**时间正序**（最早的在前），与日志输出顺序一致。
    """
    items = log_buffer.snapshot()

    if level:
        threshold = getattr(logging, level.strip().upper(), None)
        if isinstance(threshold, int):
            items = [
                it for it in items
                if getattr(logging, str(it.get("level", "INFO")), 0) >= threshold
            ]
    if logger_name:
        items = [it for it in items if str(it.get("logger", "")).startswith(logger_name)]
    if request_id:
        items = [it for it in items if it.get("request_id") == request_id]

    cap = max(1, min(int(limit or 100), log_buffer._capacity))
    return {
        "total": len(items),
        "returned": min(len(items), cap),
        # 取**最后** cap 条：排障关心最近的，而不是最早的
        "items": items[-cap:],
        "buffer_size": log_buffer._capacity,
        "format": os.getenv("LOG_FORMAT", "").strip().lower() or "auto",
    }


@app.delete("/api/logs")
async def clear_logs():
    """清空内存日志缓冲（仅影响缓冲，不删已落盘的输出）。"""
    log_buffer.clear()
    return {"ok": True}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """分层限流。按端点代价给不同配额，并回传限流响应头。

    ★ 与原实现的三点差别

    **① 从 2 个前缀扩到全站分层**
       原来只有 `/api/chat*` 与 `/api/graph*`。而 `/api/extract`、
       `/api/vision`、`/api/tender/parse`、`/api/web/search` 这些
       **重度消耗算力或按次计费**的端点完全没有保护——连点就能把额度打满。

       现在按代价分四层（见 `src/rate_limiter.TIER_RULES`）：
           none     健康/日志 —— **绝不限流**（限流探活会让编排器误判并重启）
           heavy    抽取/视觉/解析/报告 —— 10 次/分钟
           llm      问答/比对/合规/联网 —— 30 次/分钟（与原实现一致）
           context  只读统计/图谱/会话 —— 120 次/分钟（翻页切页签会打不少请求，
                                        卡太严会误伤）

    **② 真实客户端识别**（原实现的严重缺陷）
       原来 `_client_ip` 只看 `request.client.host`，**不读 `X-Forwarded-For`**。
       任何反代/网关部署下那都是**代理的 IP**，于是所有用户共享一个桶——
       30 次/分钟后**全站用户一起被拒**。这不是"限流偏严"，是限流失效且误伤所有人。

       现在只有**直连方在 `TRUSTED_PROXIES` 里**才采信转发头，
       且取 XFF 里"最右侧的非可信 IP"（最左侧是客户端自己写的，可伪造）。

    **③ 回传限流响应头**
       `X-RateLimit-Limit/Remaining/Window` + 超限时的 `Retry-After`。
       没有这些，客户端只能盲目重试——而盲目重试恰恰让限流更糟。
    """
    path = request.url.path
    # 判定逻辑在 `src.rate_limiter.check_request`（**纯函数**，可脱离应用单测）。
    # 这里只负责取请求信息、包响应。
    #
    # ★ 为什么抽成纯函数
    # ----------------
    # 中间件写在 `api/server.py` 里且是 FastAPI 装饰器形态，要测它就得起整个应用；
    # 而 import 本模块会拉起 Agent / RAG / 数据库客户端一堆重依赖。
    # 实测：用它跑一次 `/api/ask` 会触发**真实模型下载**并卡住十几分钟。
    # 抽出来之后，测试能对任意路径验证分层与配额，不启动应用、不碰外部依赖。
    verdict = check_request(
        path,
        request.client.host if request.client else "",
        request.headers.get("X-Forwarded-For", ""),
        request.headers.get("X-Real-IP", ""),
        rate_limiters,
    )
    if not verdict["limited"]:
        # `none` 层（健康检查/日志）：不参与限流，但**仍要记指标**
        # —— 健康检查的延迟本身就是重要的监控信号。
        t0 = time.monotonic()
        resp = await call_next(request)
        safe_record(path, resp.status_code, (time.monotonic() - t0) * 1000)
        return resp

    # 把真实 IP 与分层绑进日志上下文：排障时能直接看到"是谁在打、打的是哪一层"
    bind_log_context(client_ip=verdict["key"], rate_tier=verdict["tier"])

    if not verdict["allowed"]:
        logger.warning(
            "限流拒绝：tier=%s ip=%s path=%s retry_after=%ss",
            verdict["tier"], verdict["key"], path, verdict["retry_after"],
        )
        # 429 也是"请求结果"，必须计入指标 —— 否则监控里看不到限流在生效，
        # 而"被限流了多少"恰恰是要观察的关键信号之一。
        safe_record(path, 429, verdict["elapsed_ms"])
        return JSONResponse(
            status_code=429,
            content={
                "detail": "请求过于频繁，请稍后再试",
                # 让前端能显示"还需等待多久"，而不是一个没有解释的失败
                "retry_after": verdict["retry_after"],
                "tier": verdict["tier"],
            },
            headers=verdict["headers"],
        )

    t0 = time.monotonic()
    response = await call_next(request)
    safe_record(path, response.status_code, (time.monotonic() - t0) * 1000)
    # 成功路径也回传剩余配额：前端据此提前降速，而不是撞到 429 才知道
    for key, value in verdict["headers"].items():
        response.headers[key] = value
    return response


@app.get("/api/metrics", dependencies=[Depends(require_logs_access)])
def get_metrics(
    format: str = Query(
        "json",
        description="json（默认，完整的请求/系统/组件聚合）或 prometheus（文本格式，供采集器抓取）",
    ),
):
    """系统监控聚合视图。

    ★ 它回答什么问题（与另两个观测端点的分工）
    --------------------------------------
        `/api/health`    **依赖现在通不通**（连通性 + 延迟，单点快照）
        `/api/logs`      **发生了什么**（结构化日志，逐条）
        `/api/metrics`   **整体怎么样**（聚合：哪里慢、错在哪、吃多少资源）

    三者互补。只有 health 时你知道"Qdrant 通了"，但不知道"最近 5 分钟
    问答的 P95 是多少"；只有 logs 时，得自己数日志行才得出"错误率 8%"。

    ★ 返回内容
    --------
        requests    按端点的请求数 / 错误率 / P50·P95·P99 / 慢请求占比
                    （窗口内分位**与**全窗口均值都给——只看均值会被历史稀释，
                     只看分位又不知道总量）
        system      进程 CPU / 内存 / 线程 / 句柄 + 主机 CPU / 内存 / 磁盘
        components  限流各层、检索缓存命中率、LLM provider 链、日志级别分布

    ★ 为什么同时提供 Prometheus 文本格式
    ---------------------------------
    `?format=prometheus` 输出 text exposition format 0.0.4。这是**零成本的
    可扩展性**：数据已在内存里，只多一个渲染函数。将来要接
    Prometheus / VictoriaMetrics 直接把 scrape 指向这里；接之前没有开销
    （没人抓就不渲染）。

    ★ 轮询开销
    --------
    `requests` 与 `components` 是读内存快照（微秒级）；`system` 走 psutil
    但有 2 秒缓存。故前端每 5~10 秒拉一次是安全的。
    """
    if format.strip().lower() == "prometheus":
        body = render_prometheus(bidding_agent, rate_limiters)
        # Prometheus 文本格式有约定的 content type
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")
    return build_metrics(bidding_agent, rate_limiters)


@app.get("/api/eval/offline")
def get_offline_eval() -> dict:
    """离线评测结果快照（供前端 /eval 页展示）。

    ★ 与 `/api/eval/online` 的分工
    -------------------------
        离线（本端点）  固定测试集 + LLM 裁判，**有参考答案** →
                        能测"答得对不对"，含趋势与裁判一致性
        在线            真实请求采样，**无参考答案** → 代理信号

    ★ 为什么必须经后端
    ------------
    浏览器读不了磁盘。评测产物是 `eval/` 下的 JSON/JSONL，故这里读文件、
    归一成前端契约（含**指标转置**：落盘是"每次运行一行"，而画趋势线需要
    "每个指标一条序列"）。

    ★ 一个如实的局限
    ----------
    **没有逐用例明细**（`EvalReport` 只保存聚合后的 `MetricResult`）。
    响应里用 `capabilities.per_case = false` 显式声明，让前端**隐藏**
    那张表而不是画一张空表。
    """
    from src.eval_offline import build_snapshot

    return build_snapshot()


@app.get("/api/eval/online")
def get_online_eval(limit: int = 2000) -> dict:
    """RAG 在线评测：真实请求的质量信号聚合。

    ★ 与离线评测（`eval/`）的区别
    ------------------------
        离线 固定测试集 + LLM 裁判，**有参考答案** → 能测"答得对不对"
             但测试集有限，覆盖不到真实长尾
        在线 真实请求 + 已有信号，**无参考答案** → 测不了"对不对"
             但覆盖真实分布，能发现"线上某类问题一直在失败"

    ★ 为什么必须带 `caveats`
    -------------------
    在线评测没有参考答案，所有数字都是**代理信号**。不加提醒的话，
    `verify_ok_rate: 0.95` 会被读成"95% 的答案是对的"——
    而它实际只说"95% 的答案里引用的法条/数字能在检索结果里找到"。
    把口径随数据一起下发，是避免这类误读最直接的办法。
    """
    limit = max(1, min(int(limit or 2000), 20000))
    try:
        from src.eval_online import evaluator, summarize_rows

        rows = evaluator.read_rows(limit=limit)
        payload = summarize_rows(rows)
        # 采样器自身状态也要可观测（丢了多少条是"数据是否可信"的前提）
        payload["recorder"] = evaluator.snapshot()
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("在线评测聚合失败: %s", exc, exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/cache")
def get_cache_status() -> dict:
    """各缓存的命中情况。

    ★ 为什么需要它
    -----------
    项目里有**四种缓存**（检索结果 / 联网搜索×2 / Exa 结果），各自有
    TTL 与命中计数，但此前**没有任何地方能一眼看到它们的整体状态**。
    排查"为什么同样的查询很慢"时，第一个要确认的就是"缓存到底有没有命中"。

    ★ 与 `/api/logs` DELETE 的分工
    ---------------------------
    那个清日志缓冲（观测数据），这个清缓存（性能数据）。
    两者都会改变服务行为，故都做成显式动作，不自动触发。
    """
    out: dict[str, Any] = {}

    # 检索缓存
    try:
        # ★ 单例：`RAGPipeline()` 会新建实例，读到的是它自己的空缓存 ——
        # 监控会永远显示"命中 0 / 未命中 0"，看起来像"缓存从没被用过"。
        from src.rag.pipeline import rag_pipeline

        info = rag_pipeline.cache_info()
        hits, misses = info.get("hits", 0), info.get("misses", 0)
        info["hit_rate"] = round(hits / (hits + misses), 4) if (hits + misses) else 0.0
        out["retrieval"] = info
    except Exception as exc:  # noqa: BLE001
        out["retrieval"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 联网搜索：Tavily / 智谱
    #
    # ★ 这两个客户端只有模块级常量形式的 TTL，没有 `cache_info()`（本模块
    # 不为了统一而改动它们——那是 docs/46 之外的改动范围）。故这里直接读
    # 私有字段，并**显式带出 TTL 常量**，让返回结构与 Exa 那段一致。
    try:
        from src.web_search import CACHE_TTL_SECONDS as TAVILY_TTL
        from src.web_search import web_search_client

        with web_search_client._lock:
            out["web_search_tavily"] = {
                "size": len(web_search_client._cache),
                "ttl_seconds": TAVILY_TTL,
                # 这两个客户端没统计命中率，如实标 None 而不是编造 0
                "hit_rate": None,
            }
    except Exception as exc:  # noqa: BLE001
        out["web_search_tavily"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        from src.web_search_zhipu import CACHE_TTL_SECONDS as ZHIPU_TTL
        from src.web_search_zhipu import zhipu_search_client

        with zhipu_search_client._lock:
            out["web_search_zhipu"] = {
                "size": len(zhipu_search_client._cache),
                "ttl_seconds": ZHIPU_TTL,
                "hit_rate": None,
            }
    except Exception as exc:  # noqa: BLE001
        out["web_search_zhipu"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Exa（本次新增）
    try:
        from src.mcp.web_search_exa import exa_search_client

        out["web_search_exa"] = exa_search_client.cache_info()
    except Exception as exc:  # noqa: BLE001
        out["web_search_exa"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out


@app.delete("/api/cache")
def clear_caches(scope: str = Query("all", description="all / retrieval / web")) -> dict:
    """清空缓存。

    **为什么需要显式清除**：这些缓存都没有失效钩子（除了检索缓存的语料指纹），
    故"改完配置想立刻生效"或"怀疑拿到旧数据"时，此前只能重启服务。
    对一个问答服务来说，重启意味着丢失会话内已加载的模型（冷启动 100 秒）。

    `scope`：
        all        全部（默认）
        retrieval  只清检索结果缓存
        web        只清三个联网搜索缓存
    """
    scope = (scope or "all").strip().lower()
    done: list[str] = []

    if scope in ("all", "retrieval"):
        try:
            # ★ 单例：缓存挂在实例上，new 一个实例清的是它自己的空缓存，
            # 「清除」看起来成功（返回 200）但真实缓存原封不动。
            from src.rag.pipeline import rag_pipeline

            rag_pipeline.clear_cache()
            done.append("retrieval")
        except Exception as exc:  # noqa: BLE001
            logger.warning("清空检索缓存失败：%s", exc)

    if scope in ("all", "web"):
        for label, importer in (
            ("web_search_tavily", ("src.web_search", "web_search_client")),
            ("web_search_zhipu", ("src.web_search_zhipu", "zhipu_search_client")),
            ("web_search_exa", ("src.mcp.web_search_exa", "exa_search_client")),
        ):
            try:
                mod = __import__(importer[0], fromlist=[importer[1]])
                getattr(mod, importer[1]).clear_cache()
                done.append(label)
            except Exception as exc:  # noqa: BLE001
                logger.warning("清空 %s 缓存失败：%s", label, exc)

    logger.info("已清空缓存：scope=%s 实际清理=%s", scope, done)
    return {"ok": True, "scope": scope, "cleared": done}


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
def health_check(
    depth: str = Query(
        "ready",
        description="live=只报进程；ready=探本地依赖（默认，免费）；"
                    "deep=额外真实调用一次 LLM（**会消耗 API 额度**）",
    ),
    config: bool = Query(False, description="是否附带实际生效的关键配置（不含密钥值）"),
):
    """健康检查。细化后的语义见 `src/health.py` 的模块说明。

    ★ 与旧版的差别（都是实测暴露的问题）

        ① **分三层深度**。`deep` 会真实调用一次 LLM —— 故绝不是默认：
           Docker HEALTHCHECK 每 30 秒一次，一天 2880 次计费调用。
        ② **返回真实 HTTP 状态码**：`down` → 503，其余 200。
           编排器只看状态码，永远 200 等于"永远健康"。
        ③ **探 LLM 可用性**。`/api/chat` 是核心接口，
           密钥失效时旧版健康检查全绿而问答必然失败。
        ④ **区分 warming 与 down**。预热要 100 秒（嵌入 20s + 精排 80s），
           期间三个数据源其实正常，不该被判成不可用而重启。
        ⑤ **并发探测**。串行最坏 9 秒，并发后接近最慢的那个。
        ⑥ `?config=true` 暴露实际生效的配置（collection / 模型名 / 开关），
           便于核对"环境变量到底生效了没有"。**不暴露任何密钥值**。

    兼容性：旧版返回的 `ready` / `agent_ready` / `graph_ready` /
    `pg_ready` / `qdrant_ready` / `points_count` / `latencies` 字段
    **全部保留**，既有调用方与监控不会因本次细化而失效。
    """
    payload, code = build_health(
        depth=depth, include_config=config,
        # 模块级全局由这里注入：src.health 反向 import 会循环导入，
        # 且注入让测试能直接传替身
        agent=bidding_agent, graph_data=graph_data,
        postgres_getter=_get_postgres_client,
        # ★ 5 秒探测缓存（局限 2）
        #
        # ready 要并发探 Qdrant/Neo4j/Postgres/LLM，实测 1~3.3 秒（Qdrant Cloud
        # 跨洲往返占大头）。k8s 探针常配 5~10 秒间隔 —— 不缓存的话
        # 每次都付这个开销，而结果几乎不变。
        #
        # 缓存命中时响应里带 cached: true，避免运维以为这是刚测的。
        cache_seconds=5.0,
    )

    # ---- 旧字段兼容层（勿删：既有前端与监控在读数）----
    by_name = {c["name"]: c for c in payload["components"]}
    qdrant = by_name.get("qdrant")
    points = -1
    if qdrant and qdrant.get("ok"):
        # ★ 优先读**结构化字段** `count`（`probe_qdrant` 直接给的数字）
        #
        # 此前的写法是解析 detail 显示字符串：
        #     points = int(str(qdrant.get("detail", "")).split("=")[-1])
        # 而 detail 后来改成了 `points≈565`（标注为近似值），没有 `=` 可切，
        # 于是 int() 失败 → points = -1 → 前端显示「未连接向量库」。
        # 教训：**显示字符串永远不该被当作数据源解析**。
        raw = qdrant.get("count")
        if isinstance(raw, int):
            points = raw
        else:
            # 兼容旧 payload（component 里没带 count）：
            # 只认纯数字 tail，`≈` 之类符号直接放弃而不是抛异常
            tail = str(qdrant.get("detail", "")).split("=")[-1].strip("≈~ ")
            try:
                points = int(tail)
            except (TypeError, ValueError):
                points = -1

    payload["ready"] = payload["status"] in ("ok", "degraded", "warming")
    payload["agent_ready"] = bidding_agent is not None
    payload["graph_ready"] = bool(by_name.get("neo4j", {}).get("ok"))
    payload["pg_ready"] = bool(by_name.get("postgres", {}).get("ok"))
    payload["qdrant_ready"] = points >= 0
    payload["points_count"] = points
    payload["latencies"] = {
        "neo4j_ms": by_name.get("neo4j", {}).get("ms", -1) if by_name.get("neo4j", {}).get("ok") else -1,
        "postgres_ms": by_name.get("postgres", {}).get("ms", -1) if by_name.get("postgres", {}).get("ok") else -1,
        "qdrant_check_ms": by_name.get("qdrant", {}).get("ms", -1),
    }
    if qdrant and not qdrant.get("ok") and qdrant.get("error"):
        payload["qdrant_error"] = qdrant["error"]

    if code != 200:
        return JSONResponse(status_code=code, content=payload)
    return payload


@app.get("/api/health/live")
def health_live() -> dict[str, Any]:
    """存活探针：**不探任何外部依赖**，只回答"进程还在吗"。

    ★ 为什么必须与 readiness 分开
    ---------------------------
    容器编排的存活探针是用来**重启**的。若它探依赖，
    则 Qdrant 短暂抖动会让编排器重启容器——重启解决不了上游故障，
    只会让服务反复重启、日志被截断、更难排查。

    存活探针只应回答"进程是否卡死"。故它不做任何 IO（微秒级），
    可安全地以 5~10 秒间隔调用。
    """
    return {
        "status": "alive",
        "uptime_s": int(time.time() - _health_started_at),
    }


@app.get("/api/health/ready")
def health_ready() -> Response:
    """就绪探针：探本地依赖，按结果返回 200 / 503。

    它是 `/api/health?depth=ready` 的语义化别名——给编排器一个
    **不用解析 body、不用记查询参数** 的端点。
    """
    payload, code = build_health(
        depth="ready", agent=bidding_agent, graph_data=graph_data,
        postgres_getter=_get_postgres_client,
    )
    return JSONResponse(status_code=code, content=payload)


# ---------------------------------------------------------------------------
# 知识图谱直查（前端"知识图谱"页使用，不经过 LLM）
# ---------------------------------------------------------------------------


@app.get("/api/graph/types")
def graph_types() -> dict[str, Any]:
    """返回可用的图谱查询类型清单。

    前端据此渲染选择器，而不是在 TypeScript 里硬编码一份——新增查询类型时
    只需改 rag_tools._GRAPH_QUERY_TYPES，前端自动跟上。
    """
    if graph_data is None:
        raise HTTPException(status_code=503, detail="图谱查询服务未就绪")

    types = graph_data.available_query_types()
    # 每种类型需要哪些参数：前端用它决定是否显示实体输入框、是否必须填
    needs_entity = {
        "search_entity": True,
        "items_by_entity": True,
        "entity_detail": True,
        "supplier_by_subject": True,
        "top_traded": False,
        "graph_stats": False,
    }
    return {
        "types": [
            {"query_type": name, "requires_entity": needs_entity.get(name, True)}
            for name in types
        ],
        "graph_ready": graph_data.graph_available(),
    }


@app.post("/api/graph/query")
def graph_query(request: GraphQueryRequest) -> dict[str, Any]:
    """执行一次图谱查询，返回结构化 JSON。

    状态码：400 查询类型/参数不合法 ｜ 503 图谱不可用（未配置或连接失败）。
    返回 `{"query_type", "data", "count"}`——`data` 的形状随类型而定，
    与 Neo4jClient 的返回一致（graph_stats/entity_detail 为 dict，其余为 list）。
    """
    if graph_data is None:
        raise HTTPException(status_code=503, detail="图谱查询服务未就绪")

    # entity 与 keyword 是同一个语义（前端复用输入框），entity 优先
    arguments = {
        "entity": request.entity or request.keyword,
        "keyword": request.keyword or request.entity,
        "limit": request.limit,
    }
    try:
        return graph_data.run_graph_query(request.query_type, arguments)
    except graph_data.GraphQueryError as exc:
        # 客户端 bug（类型写错/漏填实体）：把可用清单回给前端，便于直接提示用户
        raise HTTPException(status_code=400, detail=str(exc))
    except graph_data.GraphUnavailableError:
        logger.exception("图谱查询失败：%s", request.query_type)
        raise HTTPException(
            status_code=503, detail="知识图谱暂时不可用，请检查 Neo4j 实例状态"
        )


# ---------------------------------------------------------------------------
# 图片比价（视觉提取商品 → 多平台检索报价 → 择优建议）
# ---------------------------------------------------------------------------


@app.post("/api/price-compare")
def price_compare(request: PriceCompareRequest) -> dict[str, Any]:
    """上传商品图片，返回多平台比价结果。

    流程（服务层编排，约 10~15 秒）：
        图片 → 视觉模型提取商品信息（JSON）→ 并行检索各平台 → 抽取报价 → 择优建议

    状态码：400 图片无法解析 / 未识别出商品 ｜ 413 图片过大 ｜
    500 未知错误 ｜ 503 服务未就绪（依赖缺失或搜索服务不可用）。

    注意：结果里的 `recommendation.caveats` 是**必读**内容——本功能基于公开
    搜索摘要而非电商官方价格 API，存在平台取不到价、价格可能属不同规格等局限，
    前端必须原样展示这些提示，不能只显示"最优平台"。
    """
    if len(request.image_base64) > _MAX_IMAGE_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="图片过大，请压缩后重新上传")

    if price_compare_service is None:
        raise HTTPException(status_code=503, detail="比价服务未就绪")

    try:
        # 服务内部用 asyncio.run 调视觉模型，在同步端点里执行不会与事件循环冲突；
        # 这个端点本身是 def（非 async），FastAPI 会把它放到线程池，故不会阻塞事件循环
        return price_compare_service.compare_by_image(
            request.image_base64, hint=request.hint, include_raw_text=False
        )
    except price_compare_service.PriceCompareError as exc:
        logger.info("比价失败：%s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        logger.info("比价入参不合法：%s", exc)
        raise HTTPException(status_code=400, detail=f"图片数据无法解析：{exc}")
    except Exception:
        logger.exception("比价服务异常")
        raise HTTPException(status_code=500, detail="比价失败，请稍后重试")


# ---------------------------------------------------------------------------
# 标讯智能提取（公告文本 / 截图 → 17 个结构化字段）
# ---------------------------------------------------------------------------


@app.get("/api/extract/schema")
def extract_schema() -> dict[str, Any]:
    """返回标讯字段清单，供前端渲染表格与表单。

    与 `/api/graph/types` 同一思路：**字段由后端定义，前端不硬编码**。
    新增字段只需改 `src/constants/bid_schema.py`，前端自动跟上。
    """
    if bid_schema is None:
        raise HTTPException(status_code=503, detail="标讯提取服务未就绪")

    return {
        "fields": [
            {
                "key": f.key,
                "label": f.label,
                "type": f.type,
                "hint": f.hint,
                "required": f.required,
            }
            for f in bid_schema.BID_FIELDS
        ],
        "count": len(bid_schema.BID_FIELDS),
    }


@app.post("/api/extract")
def extract_notice(request: ExtractRequest) -> dict[str, Any]:
    """从公告文本或公告截图提取 17 个结构化字段。

    状态码：400 输入不合法 / 模型输出无法解析 ｜ 413 图片过大 ｜
    500 模型调用失败 ｜ 503 服务未就绪。

    返回体含 `meta.completeness`（字段完整度 0-1）与 `meta.warnings`
    （被清洗的可疑值）——前端应展示，让用户知道这次抽取有多可信。
    """
    if extract_service is None or bid_schema is None:
        raise HTTPException(status_code=503, detail="标讯提取服务未就绪")

    if request.source == "image" and len(request.image_base64) > _MAX_IMAGE_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="图片过大，请压缩后重新上传")

    try:
        result = extract_service.extract(
            source=request.source,
            text=request.text,
            image_base64=request.image_base64,
            hint=request.hint,
        )
    except extract_service.ExtractError as exc:
        logger.info("标讯提取失败：%s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"图片数据无法解析：{exc}")
    except Exception:
        logger.exception("标讯提取异常")
        raise HTTPException(status_code=500, detail="提取失败，请稍后重试")

    if request.save:
        # 入库失败不当作整体失败：字段已经抽出来了，前端仍应看到结果，
        # 只是额外提示"未入库"及其原因
        result["save"] = extract_service.save_record(result["record"])

    return result


# ---------------------------------------------------------------------------
# 集采分析（PostgreSQL 多维聚合）
# ---------------------------------------------------------------------------


def _analysis_guard():
    """分析类端点的共用前置检查。"""
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="集采分析服务未就绪")


@app.get("/api/analysis/overview")
def analysis_overview(exclude_suspicious: bool = True) -> dict[str, Any]:
    """总体概览：条数、金额分位数、维度基数、数据质量提示。

    ⚠️ `data_caveats` 是**必读**字段——本数据集只有单月数据（无法做趋势）、
    金额跨 6 个数量级（均值不可用）等局限都在里面，前端必须展示。
    """
    _analysis_guard()
    try:
        return analysis_service.overview(exclude_suspicious=exclude_suspicious)
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("分析总览失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/analysis/dimensions")
def analysis_dimensions() -> dict[str, Any]:
    """可用的分组维度清单（前端据此渲染下拉框，不硬编码）。"""
    _analysis_guard()
    return {
        "dimensions": [
            {"key": k, "label": v} for k, v in analysis_service.GROUPABLE.items()
        ]
    }


@app.get("/api/analysis/ranking")
def analysis_ranking(
    dimension: str = "subject_matter",
    limit: int = 20,
    order_by: str = "count",
    exclude_suspicious: bool = True,
) -> dict[str, Any]:
    """按维度分组排行。每组给出**中位数**（组内同样可能有超大额记录）。"""
    _analysis_guard()
    try:
        return analysis_service.group_ranking(
            dimension,
            limit=limit,
            order_by=order_by,
            exclude_suspicious=exclude_suspicious,
        )
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("分组排行失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/analysis/distribution")
def analysis_distribution(exclude_suspicious: bool = True) -> dict[str, Any]:
    """金额数量级分布（柱状图数据源）。"""
    _analysis_guard()
    try:
        return analysis_service.amount_distribution(exclude_suspicious=exclude_suspicious)
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("金额分布失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.post("/api/analysis/price-profile")
def analysis_price_profile(request: PriceProfileRequest) -> dict[str, Any]:
    """报价策略：某标的物的价格分位数 + 建议报价区间 + 竞争对手。"""
    _analysis_guard()
    try:
        return analysis_service.price_profile(
            request.subject,
            exclude_suspicious=request.exclude_suspicious,
            top_n=request.top_n,
        )
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("报价策略失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/analysis/data-quality")
def analysis_data_quality() -> dict[str, Any]:
    """数据质量：字段空值率 + 可疑金额明细（含逐条判定原因）。"""
    _analysis_guard()
    try:
        return analysis_service.data_quality()
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("数据质量查询失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/analysis/recent")
def analysis_recent(limit: int = 8, exclude_suspicious: bool = True) -> dict[str, Any]:
    """最近的中标记录（工作台"最近中标动态"数据源）。

    ⚠️ 响应含 `coverage.caveat`：本数据集只覆盖单月，"最近"指**数据里**最近，
    不代表正在发生的采购。前端必须展示该说明。
    """
    _analysis_guard()
    try:
        return analysis_service.recent_records(
            limit=limit, exclude_suspicious=exclude_suspicious
        )
    except analysis_service.AnalysisError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("最近记录查询失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


# ---------------------------------------------------------------------------
# 知识图谱关系可视化（子图抽取）
# ---------------------------------------------------------------------------


def _subgraph_guard():
    if subgraph_service is None:
        raise HTTPException(status_code=503, detail="图谱子图服务未就绪")


@app.get("/api/graph/overview")
def graph_overview() -> dict[str, Any]:
    """图谱规模总览 + 可探索的枢纽实体（前端"推荐探索"列表）。"""
    _subgraph_guard()
    try:
        return subgraph_service.graph_nodes_overview()
    except subgraph_service.SubgraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("图谱总览失败")
        raise HTTPException(status_code=500, detail="图谱查询失败，请稍后重试")


@app.get("/api/graph/search")
def graph_search(keyword: str = "", limit: int = 15) -> dict[str, Any]:
    """按关键词搜索图谱实体（供前端自动补全）。"""
    _subgraph_guard()
    try:
        items = subgraph_service.search_entities(keyword, limit=limit)
        return {"items": items, "count": len(items)}
    except subgraph_service.SubgraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("实体搜索失败")
        raise HTTPException(status_code=500, detail="图谱查询失败，请稍后重试")


@app.get("/api/graph/subgraph")
def graph_subgraph(label: str, name: str, max_nodes: int = 40) -> dict[str, Any]:
    """以某实体为中心抽子图，供前端力导向图渲染。

    ⚠️ 两点必须让前端知道：
      1. **Transaction 节点不画入图**——实测 8730 个，只挂金额、无出边，
         画出来只是孤立点。中心节点的交易统计单独给出。
      2. **边不含金额**——图谱中没有「采购人→交易」的边，无法算出每条边
         对应的金额。需要按维度细分金额请用 `/api/analysis/*`。
       两条都写在响应的 `legend.notes` 里。

    超限时按度数截断并返回 `truncated=true`（不静默丢数据）。
    """
    _subgraph_guard()
    try:
        return subgraph_service.build_subgraph(label, name, max_nodes=max_nodes)
    except subgraph_service.SubgraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except subgraph_service.SubgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("子图抽取失败")
        raise HTTPException(status_code=500, detail="图谱查询失败，请稍后重试")


# ---------------------------------------------------------------------------
# 竞争对手分析
# ---------------------------------------------------------------------------


def _competitor_guard():
    if competitor_service is None:
        raise HTTPException(status_code=503, detail="竞争对手分析服务未就绪")


def _competitor_error(exc: Exception) -> HTTPException:
    """把服务层异常映射成合适的状态码。

    `GraphUnavailableError` 是 `CompetitorError` 的子类，但语义不同：
    图谱不可用时服务**会降级返回业务库结果**（`graph_enriched=False`），
    所以它通常不会走到这里；真走到这里说明业务库也挂了 → 503。
    """
    if isinstance(exc, competitor_service.GraphUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    # 参数不合法（关键词过短等）归 400，其余归 503
    msg = str(exc)
    if "关键词" in msg or "至少" in msg or "不能为空" in msg:
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(status_code=503, detail=msg)


@app.get("/api/competitor/suppliers")
def competitor_suppliers(
    subject: str, limit: int = 20, exclude_suspicious: bool = True
) -> dict[str, Any]:
    """某标的物的供应商竞争格局。

    金额一律用**中位数**：实测该数据集金额跨 6 个数量级，均值不具代表性。
    """
    _competitor_guard()
    try:
        return competitor_service.suppliers_for_subject(
            subject, limit=limit, exclude_suspicious=exclude_suspicious
        )
    except competitor_service.CompetitorError as exc:
        raise _competitor_error(exc)
    except Exception:
        logger.exception("供应商格局分析失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/competitor/profile")
def competitor_profile(name: str) -> dict[str, Any]:
    """单个供应商画像：中标规模、常投标的物、价格水平。"""
    _competitor_guard()
    try:
        return competitor_service.competitor_profile(name)
    except competitor_service.CompetitorError as exc:
        raise _competitor_error(exc)
    except Exception:
        logger.exception("供应商画像失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/competitor/purchaser")
def competitor_purchaser(purchaser: str, limit: int = 20) -> dict[str, Any]:
    """某采购人的合作供应商格局与集中度。"""
    _competitor_guard()
    try:
        return competitor_service.purchaser_landscape(purchaser, limit=limit)
    except competitor_service.CompetitorError as exc:
        raise _competitor_error(exc)
    except Exception:
        logger.exception("采购人格局分析失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


@app.get("/api/competitor/concentration")
def competitor_concentration(subject: str) -> dict[str, Any]:
    """某标的物的市场集中度（CR3 / CR5 / HHI）。

    HHI 判定标准：<1500 竞争充分 ｜ 1500~2500 中等集中 ｜ >2500 高度集中。
    """
    _competitor_guard()
    try:
        return competitor_service.market_concentration(subject)
    except competitor_service.CompetitorError as exc:
        raise _competitor_error(exc)
    except Exception:
        logger.exception("市场集中度分析失败")
        raise HTTPException(status_code=500, detail="分析失败，请稍后重试")


# ---------------------------------------------------------------------------
# 物料识别
# ---------------------------------------------------------------------------


def _material_guard():
    if material_service is None:
        raise HTTPException(status_code=503, detail="物料识别服务未就绪")


def _material_error(exc: Exception) -> HTTPException:
    msg = str(exc)
    if "关键词" in msg or "不能为空" in msg or "不存在" in msg or "至少" in msg:
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(status_code=503, detail=msg)


@app.get("/api/material/stats")
def material_stats(top_n: int = 20) -> dict[str, Any]:
    """物料库总览：标准物料数、原始写法数、归并压缩率。"""
    _material_guard()
    try:
        return material_service.material_stats(top_n=top_n)
    except material_service.MaterialError as exc:
        raise _material_error(exc)
    except Exception:
        logger.exception("物料库统计失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.get("/api/material/match")
def material_match(query: str, limit: int = 10) -> dict[str, Any]:
    """物料模糊匹配：输入任意写法，返回候选标准物料。"""
    _material_guard()
    try:
        return material_service.match_materials(query, limit=limit)
    except material_service.MaterialError as exc:
        raise _material_error(exc)
    except Exception:
        logger.exception("物料匹配失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.get("/api/material/detail")
def material_detail(name: str) -> dict[str, Any]:
    """单个标准物料的详情：分位数、主要供应商、主要采购人。"""
    _material_guard()
    try:
        return material_service.material_detail(name)
    except material_service.MaterialError as exc:
        raise _material_error(exc)
    except Exception:
        logger.exception("物料详情查询失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.get("/api/material/suggestions")
def material_suggestions(limit: int = 50) -> dict[str, Any]:
    """疑似应归并但尚未归并的物料对，供人工确认。"""
    _material_guard()
    try:
        return material_service.suggest_standardization(limit=limit)
    except material_service.MaterialError as exc:
        raise _material_error(exc)
    except Exception:
        logger.exception("标准化建议查询失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.post("/api/material/normalize-preview")
def material_normalize_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """预览物料名归一化结果（前端输入时实时提示用）。"""
    _material_guard()
    names = payload.get("names") or []
    if not isinstance(names, list):
        raise HTTPException(status_code=400, detail="names 必须是字符串数组")
    try:
        return material_service.normalize_preview([str(n) for n in names])
    except material_service.MaterialError as exc:
        raise _material_error(exc)
    except Exception:
        logger.exception("归一化预览失败")
        raise HTTPException(status_code=500, detail="处理失败，请稍后重试")


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
                web_search_enabled=_resolve_web_mode(request),
                provider=request.provider or "",
                deep_thinking_enabled=request.deep_thinking_enabled,
                auto_web_fallback=(request.web_search_mode or "auto") == "auto",
                session_id=request.session_id or "",
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
            web_search_enabled=_resolve_web_mode(request),
            provider=request.provider or "",
            deep_thinking_enabled=request.deep_thinking_enabled,
            auto_web_fallback=(request.web_search_mode or "auto") == "auto",
            # 透传 session_id：历史压缩的滚动水位线按它分桶。
            # 漏传的后果**不报错也不改行为**（只是每轮重新压缩、白花一次 LLM 调用），
            # 所以这个参数必须有测试断言（见 test_chat_stream_passes_session_id_to_agent）。
            session_id=request.session_id or "",
        )
    except Exception:
        logger.exception("非流式对话失败")
        raise HTTPException(status_code=500, detail="回答生成失败，请稍后重试")


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """直接 RAG 检索：**不经过 Agent、不调用大模型**，返回最匹配的问答原文。

    与 `/api/chat*` 的分工（见 docs/开发文档.md §5.1）：
        `/api/chat*` 走 `BiddingAgent`（ReAct 工具循环，会调 LLM 生成回答）
        `/api/ask`   只做向量检索，把知识库里最匹配的问答对原样返回

    适用场景：脚本/批处理需要"库里的标准问答原文"而非"模型润色后的回答"；
    也用于 LLM 不可用时仍能提供检索结果。

    ★ relevance 字段与相关性过滤（2026-10-12 新增）
    ---------------------------------------------
    `score` 是 RRF 融合分，只反映候选内排名，**任何查询都会有第一名**——
    实测无关问题的 RRF 分甚至高于相关问题，故不能用它判断相关性。

    `relevance` 是与查询的稠密余弦相似度，可判相关。标定结果
    （18 个相关问题 / 12 个无关问题，取每题最佳结果）：
        相关 0.5806~0.7306 ｜ 无关 0.2964~0.4256 ｜ 间隔 0.1550
    据此设阈值 0.5：低于它的结果不作为答案返回（否则"
    红烧肉怎么做才好吃"会拿到一条招投标问答冒充答案）。

    ★ 历史实现与规格相反
    ------------------
    原实现直接 `client.chat_stream(messages)` **调用大模型**，并把 `sources`
    硬编码为 `[]`，与文档"不调用大模型"的规格相反，且 `AskRequest.top_k`
    这个专为检索设计的字段从未被使用。

    响应契约：`{answer, sources: [{question, answer, score, relevance}], count, filtered}`
    """
    _require_question(request.question)

    try:
        # ★ 必须用模块级单例，不要 `RAGPipeline()`
        # ------------------------------------
        # `search_cached` 的缓存在**实例属性** `_cache` 上。现 new 一个实例
        # 等于每次都是空缓存 → 同一问题反复检索都打 Qdrant。
        # Agent 路径（`agent/core._get_rag_pipeline`）用的就是单例
        # `src.rag.pipeline.rag_pipeline`，这里必须一致，否则"有没有缓存"
        # 取决于走哪个接口（实测：这条路径的缓存在修复前从未生效过）。
        from src.rag.pipeline import rag_pipeline

        sources = rag_pipeline.search_cached(request.question, top_k=request.top_k)
    except Exception:
        logger.exception("ask 接口检索失败")
        raise HTTPException(status_code=500, detail="知识库检索失败，请稍后重试")

    # 只保留契约字段：向量库内部字段（id、payload 结构）属实现细节
    normalized = [
        {
            "question": str(s.get("question") or ""),
            "answer": str(s.get("answer") or ""),
            "score": s.get("score"),
            "relevance": s.get("relevance"),
        }
        for s in sources
    ]

    # 说明：相关性过滤**已经在 `RAGPipeline.search()` 里做过**（见该方法的
    # 「相关性过滤」小节），此处不再重复。早先这里有一层 `best < MIN_RELEVANCE`
    # 的判断，但它检查的是「返回结果里第一条的 relevance」——而能被返回的第一条
    # 必然已 >= 阈值，故那一层**永远不会触发**，是死代码，已删除。
    #
    # 阈值与"relevance 为 None 时不过滤"的口径都收敛在 pipeline 一处，
    # 避免两个入口各写一份、改一处漏一处。
    if not normalized:
        return {
            "answer": "知识库中未找到与该问题匹配的内容。可换个问法，或改用 /api/chat 让 Agent 联网检索。",
            "sources": [],
            "count": 0,
            "filtered": 0,
        }

    return {
        "answer": normalized[0]["answer"],
        "sources": normalized,
        "count": len(normalized),
        "filtered": 0,
    }


# ---------------------------------------------------------------------------
# 视觉类端点
# ---------------------------------------------------------------------------


@app.post("/api/vision")
async def vision(request: VisionRequest) -> dict[str, Any]:
    """图片理解。

    状态码：400 图片数据无法解码 ｜ 413 图片过大 ｜ 422 字段校验失败 ｜
    500 模型调用失败 ｜ 503 未配置 ZHIPU_API_KEY。
    """
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
    except ValueError as exc:
        # 客户端数据问题（base64 无法解码）：返回 4xx 让前端提示"换一张图"，
        # 而不是 500——那会让用户以为是服务故障，去重试同一张坏图
        logger.info("视觉接口入参不合法：%s", exc)
        raise HTTPException(status_code=400, detail=f"图片数据无法解析：{exc}")
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
        # ★ 本轮问答的 Langfuse trace_id（P0：让用户反馈能挂成 Score）
        #
        # 为什么在这里取而**不是**让前端传：trace_id 由 SDK 在 `begin()` 时生成、
        # 只活在内存里，前端根本不知道它。agent 在回合结束时已把它按
        # `session_id` 落到库里（见 `src/langfuse_feedback.py`），这里查回来即可。
        #
        # 只写给 `assistant` 消息：反馈针对的是回答，用户消息没有可评价内容。
        # 取不到是正常的（未配 Langfuse），故不告警、静默降级。
        _trace_id: str | None = None
        try:
            from src.langfuse_feedback import lookup_trace_id

            _trace_id = lookup_trace_id(request.session_id)
        except Exception:  # noqa: BLE001 - 观测是旁路
            logger.debug("查询 session 的 trace_id 失败", exc_info=True)

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
                trace_id=_trace_id if role == "assistant" else None,
                # ★ 图片解析文本随消息落库（"只存解析文本"：原图不入库）
                #
                # 只接受 list：契约里形如
                # `[{"name": ..., "analysis": ..., "model": ...}, ...]`。
                # 传别的形状（字符串 / dict）说明调用方写错了，直接丢弃而不是塞进
                # JSONB —— 否则读端拿到的形状不可预期（前端按数组渲染会直接报错）。
                image_analyses=(
                    msg.get("imageAnalyses")
                    if isinstance(msg.get("imageAnalyses"), list)
                    else None
                ),
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


# ---------------------------------------------------------------------------
# 对话历史：搜索 / 增强列表 / 重命名
# ---------------------------------------------------------------------------


class RenameConversationRequest(BaseModel):
    """重命名会话。"""

    title: str = Field(..., min_length=1, max_length=120, description="新标题")

    @field_validator("title")
    @classmethod
    def _strip_and_require(cls, v: str) -> str:
        """去空白后必须非空，否则 422。

        ★ 为什么必须在**校验层**挡下（实测）
        ----------------------------------
        `min_length=1` 对 `"   "` 是**通过**的（长度 3）——
        它校验的是"传了东西"，不是"传了有意义的东西"。
        于是纯空白会一路走到数据库层，被 `rename_conversation` 拒绝并返回 400，
        前端拿到的是"400 Bad Request"而不是"标题不能为空"这种可读提示。

        在这里 strip 后判空，错误在最早的环节暴露，且错误信息准确。
        返回值也顺便 strip——库里不该存前后带空格的标题。
        """
        text = (v or "").strip()
        if not text:
            raise ValueError("标题不能为空")
        return text[:120]


@app.get("/api/conversations/search")
def search_conversations_endpoint(q: str, limit: int = 30) -> dict[str, Any]:
    """按关键词搜索对话历史（**标题 + 消息内容**），按相关性降序。

    ⚠️ 路由顺序很重要：本路由必须定义在 `/api/conversations/{session_id}` **之前**，
    否则 Starlette 会把 `search` 当成一个 session_id 匹配到那条路由上，
    结果是"搜历史"永远返回 404 或空会话。

    每条结果含：
        `matched_in`  命中位置（title / message / both）—— 只给标题的话
                      用户看不出"为什么这条被搜出来"
        `snippet`     命中处的上下文片段（以关键词为中心，两端省略号）
        `relevance`   pg_trgm 相似度，用于排序
        `message_count`

    中文搜索用 `ILIKE` + pg_trgm，**不用** PostgreSQL 全文检索：
    `to_tsvector('simple', '中国大陆和香港的招投标法规')` 会把整句当成 1 个
    token（它按空格分词，中文没有空格），搜「香港」匹配不到。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="对话历史服务未就绪")
    try:
        items = pg.search_conversations(q, limit=limit)
        return {"items": items, "count": len(items), "keyword": q}
    except Exception:
        logger.exception("搜索对话历史失败")
        raise HTTPException(status_code=500, detail="搜索失败，请稍后重试")


@app.get("/api/conversations/detailed")
def list_conversations_detailed_endpoint(limit: int = 50) -> dict[str, Any]:
    """会话列表（带消息数与首条用户消息预览），供「跨会话恢复」使用。

    与 `/api/conversations` 的区别：多返回 `message_count` / `preview` /
    `last_message_at`，并按**最近一条消息时间**排序。

    为什么恢复会话需要这些：
        1. 自动生成的标题信息量低，预览才能让用户认出"就是那次问香港法规的对话"
        2. `message_count = 0` 的会话（点了新建但没提问）应当能被识别
        3. 按最后消息时间排更符合直觉——只看创建时间的话，
           昨天建、今天还在用的长会话会沉到底部

    同样必须定义在 `/{session_id}` 之前（见 search 端点的说明）。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="对话历史服务未就绪")
    try:
        items = pg.list_conversations_detailed(limit=limit)
        return {"items": items, "count": len(items)}
    except Exception:
        logger.exception("查询会话列表失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.patch("/api/conversations/{session_id}/title")
def rename_conversation_endpoint(
    session_id: str, request: RenameConversationRequest
) -> dict[str, Any]:
    """重命名会话（用户手动改标题）。

    空标题由请求模型的 `min_length=1` 挡下（422）——会话没有标题后，
    列表里就只剩 id，用户再也认不出是哪次对话。
    """
    pg = _get_postgres_client()
    if pg is None:
        raise HTTPException(status_code=503, detail="对话历史服务未就绪")
    try:
        conversation_id = pg.find_id_by_session(session_id)
        if conversation_id < 0:
            raise HTTPException(status_code=404, detail=f"未找到会话 {session_id}")
        ok = pg.rename_conversation(conversation_id, request.title)
        if not ok:
            raise HTTPException(status_code=400, detail="标题不能为空")
        return {"status": "ok", "session_id": session_id, "title": request.title.strip()}
    except HTTPException:
        raise
    except Exception:
        logger.exception("重命名会话失败")
        raise HTTPException(status_code=500, detail="重命名失败，请稍后重试")


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
                # 图片解析文本（老库 / 该列不可用时为 None，前端按"没有图片"处理）
                "imageAnalyses": row.get("image_analyses"),
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
        # 一并清掉该会话的历史压缩状态。
        #
        # 不清的后果是**隐蔽的**：压缩摘要是按 session_id 分桶的进程内状态。
        # 若前端复用了同一个 session_id（或用户"删除后重开"），
        # 新会话会读到旧会话的早期摘要——凭空多出一段不属于它的上下文，
        # 而模型会把它当作真实历史。
        try:
            from src.agent.compaction import reset_session

            reset_session(session_id)
        except Exception:  # pragma: no cover - 清理失败不影响删除本身
            logger.warning("清理历史压缩状态失败（会话已删除）", exc_info=True)
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
        # ★ P0：把用户反馈挂成 Langfuse 的 Score
        #
        # 为什么这是最重要的一处补足：Langfuse 里此前**只有耗时与 token**，
        # 没有任何"用户是否满意"的维度 —— 于是"哪一类问题被点踩"
        # 在 Langfuse 里根本筛不出来。而反馈数据**本来就有**
        # （`rating` 已经是 'up'/'down'），只是没往那边送。
        #
        # 放在落库**之后**：先保证反馈本身不丢，再接观测。
        # 整段包 try/except —— 观测失败绝不能让用户看到 503。
        try:
            from src.langfuse_feedback import score_feedback

            score_feedback(
                session_id=request.session_id,
                message_id=request.message_id,
                rating=request.rating,
                pg=pg,
            )
        except Exception:  # noqa: BLE001 - 观测是旁路
            logger.debug("Langfuse 反馈评分失败", exc_info=True)
        # 反馈变化后让会话记忆缓存失效：新采纳的问答应当能进入后续召回。
        #
        # 不做这一步的后果是**静默的**：用户点了赞，但记忆里那份缓存还是旧的，
        # 新内容要等下次重启后端才生效——而没有任何地方会提示这一点。
        try:
            from src.agent.memory import session_memory

            session_memory.invalidate()
        except Exception:  # pragma: no cover - 失效失败不影响反馈本身
            logger.warning("会话记忆缓存失效失败（反馈已保存）", exc_info=True)
        return {"status": "ok"}
    except Exception:
        logger.exception("保存反馈失败")
        raise HTTPException(status_code=503, detail="反馈保存失败")


@app.get("/api/feedback/knowledge-gaps")
def list_knowledge_gaps(limit: int = 50) -> dict[str, Any]:
    """列出被点踩的问答，作为**待补的知识缺口**清单。

    为什么单独做这个端点：现有的反馈看板只给"踩 N 次"这种数字，
    **它不告诉你该补什么**。本端点把被踩的问题原文列出来，
    运营照单补语料即可——把反馈从"指标"变成"可执行动作"。

    每条含 `question`（用户问的）、`answer`（当时的回答）、`comment`（用户评论）。
    """
    try:
        from src.agent.memory import load_knowledge_gaps

        items = load_knowledge_gaps(limit=max(1, min(limit, 500)))
        return {"items": [g.to_dict() for g in items], "count": len(items)}
    except Exception:
        logger.exception("读取知识缺口失败")
        raise HTTPException(status_code=503, detail="知识缺口读取失败")


@app.get("/api/debug/compaction")
def debug_compaction() -> dict[str, Any]:
    """诊断：查看服务端历史压缩状态表里到底有什么。

    为什么需要这个端点：压缩状态是**进程内**的，从外部完全观察不到。
    实测中"同一会话每轮都重新压缩"这类问题无法通过网络响应判断，
    只能直接读状态表——它同时也是"多进程部署会失效"这一局限的验证手段
    （若部署了多 worker，不同请求会命中不同的状态表，`pid` 会不同）。
    """
    try:
        from src.agent import compaction as C

        with C._store.lock:
            sessions = {
                sid: {"covered": st.covered, "summary_len": len(st.summary)}
                for sid, st in C._store.sessions.items()
            }
        return {
            "pid": os.getpid(),
            "store_id": id(C._store),
            "session_count": len(sessions),
            "sessions": sessions,
        }
    except Exception:
        logger.exception("读取压缩状态失败")
        raise HTTPException(status_code=503, detail="压缩状态读取失败")


@app.get("/api/feedback/accepted")
def list_accepted_examples(limit: int = 50) -> dict[str, Any]:
    """列出被采纳（点赞）的问答——即会话记忆的**数据来源**。

    做这个端点的理由：会话记忆是个隐式生效的能力，用户看不到它在用哪些历史。
    出问题时（例如某条历史把回答带偏了）需要能查到"记忆里到底有什么"。
    """
    try:
        from src.agent.memory import _load_accepted_from_db

        rows = _load_accepted_from_db(limit=max(1, min(limit, 500)))
        items = [
            {
                "question": str(r.get("question") or ""),
                "answer": str(r.get("answer") or ""),
                "rating": str(r.get("rating") or ""),
            }
            for r in rows
            if isinstance(r, dict) and r.get("question")
        ]
        return {"items": items, "count": len(items)}
    except Exception:
        logger.exception("读取已采纳问答失败")
        raise HTTPException(status_code=503, detail="已采纳问答读取失败")


# ---------------------------------------------------------------------------
# 招标文件解析与结构化抽取（PDF / Word）
# ---------------------------------------------------------------------------

# 上传文件大小上限。取值依据：实测常见招标文件 1~20MB（含图纸的会更大，
# 但那类文件本就不适合做文本抽取）。50MB 足够覆盖纯文本型招标文件，
# 同时避免把服务器内存打满（内容会整体读入内存）。
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _tender_guard():
    if tender_extractor_service is None or document_parser_service is None:
        raise HTTPException(status_code=503, detail="招标文件解析服务未就绪")


@app.get("/api/tender/schema")
def tender_schema_endpoint() -> dict[str, Any]:
    """返回招标文件的字段与章节定义，供前端渲染（不在前端硬编码）。

    与 `/api/extract/schema`（标讯 17 字段）并列但不同：
    本接口返回 16 个标量字段 + 5 个**列表章节**。
    """
    _tender_guard()
    return {
        "fields": [
            {
                "key": f.key,
                "label": f.label,
                "type": f.type,
                "hint": f.hint,
                "required": f.required,
            }
            for f in tender_schema.TENDER_FIELDS
        ],
        "sections": [
            {
                "key": s.key,
                "label": s.label,
                "hint": s.hint,
                "item_fields": list(s.item_fields),
            }
            for s in tender_schema.TENDER_SECTIONS
        ],
        "field_count": len(tender_schema.TENDER_FIELDS),
        "section_count": len(tender_schema.TENDER_SECTIONS),
    }


@app.post("/api/tender/parse")
async def tender_parse(file: UploadFile = File(...)) -> dict[str, Any]:
    """只解析文档、不抽取（快速预览：页数/字数/章节结构）。

    用途：用户上传后先确认"文件读对了"，再决定是否花时间做 LLM 抽取。
    抽取要调多次 LLM（云端 1~2 分钟、本地 Ollama 更久），
    先给一次秒级的解析预览能避免白等。

    状态码：400 文件无法解析（含扫描件无文字层）｜ 413 文件过大 ｜ 503 服务未就绪
    """
    _tender_guard()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(data) / 1048576:.1f}MB），上限 {_MAX_UPLOAD_BYTES // 1048576}MB",
        )

    try:
        parsed = document_parser_service.parse_document(file.filename or "unnamed", data)
    except document_parser_service.DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("文档解析失败")
        raise HTTPException(status_code=500, detail="文档解析失败，请稍后重试")

    return parsed.summary()


@app.post("/api/tender/extract")
async def tender_extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """解析 + 结构化抽取（非流式）。

    耗时较长（分块多次调 LLM），前端优先用 `/api/tender/extract/stream` 拿进度。
    本接口供脚本/批处理调用。

    响应含 `meta.completeness` 与 `meta.warnings`，前端必须展示——
    抽取质量与文件质量强相关，用户需要知道结果可不可信。
    """
    _tender_guard()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(data) / 1048576:.1f}MB），上限 {_MAX_UPLOAD_BYTES // 1048576}MB",
        )

    try:
        return tender_extractor_service.extract_from_document(file.filename or "unnamed", data)
    except document_parser_service.DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except tender_extractor_service.TenderExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("招标文件抽取失败")
        raise HTTPException(status_code=500, detail="抽取失败，请稍后重试")


@app.post("/api/tender/extract/stream")
async def tender_extract_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """SSE 流式抽取：边解析边抽取边上报进度。

    事件协议（`type` 键）：
        `{"type":"parse",    ...}`  解析完成（页数/字数/章节）
        `{"type":"progress", "done":i, "total":n, "message":"…"}`
        `{"type":"done",     "parsed":{…}, "meta":{…}}`
        `{"type":"error",    "content":"…"}`
        末尾固定 `data: [DONE]`

    为什么必须有流式版本：分块抽取要调多次 LLM，云端 1~2 分钟、
    本地 Ollama 可能 5 分钟以上。没有进度反馈时前端只能一直转圈，
    用户无法判断是"在跑"还是"卡死"。
    """
    _tender_guard()

    data = await file.read()
    filename = file.filename or "unnamed"

    def event_generator():
        yield _SSE_PADDING
        if not data:
            yield _error_sse("文件内容为空")
            yield "data: [DONE]\n\n"
            return
        if len(data) > _MAX_UPLOAD_BYTES:
            yield _error_sse(
                f"文件过大（{len(data) / 1048576:.1f}MB），上限 {_MAX_UPLOAD_BYTES // 1048576}MB"
            )
            yield "data: [DONE]\n\n"
            return

        try:
            for event_type, payload in tender_extractor_service.stream_extract(filename, data):
                yield "data: " + json.dumps(
                    {"type": event_type, **payload}, ensure_ascii=False
                ) + "\n\n"
        except Exception:
            # 异常只进日志，不向用户泄露内部细节
            logger.exception("招标文件流式抽取失败")
            yield _error_sse("抽取失败，请稍后重试")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 招标文件持久化与截止日期监控
# ---------------------------------------------------------------------------


class SaveTenderRequest(BaseModel):
    """保存抽取结果。

    直接把 `/api/tender/extract` 的返回体传回来即可（`parsed` + `meta`）。
    不接收文件本身——文件已在上一步解析完，重传一遍浪费带宽与时间。
    """

    parsed: dict[str, Any] = Field(..., description="抽取出的字段与章节")
    meta: dict[str, Any] = Field(default_factory=dict, description="抽取元信息（文件名/页数/模型等）")


def _tender_store_guard():
    if tender_store_service is None:
        raise HTTPException(status_code=503, detail="招标文件存储服务未就绪")


@app.post("/api/tender/save")
def tender_save(request: SaveTenderRequest) -> dict[str, Any]:
    """把抽取结果存入数据库。

    存下来之后才有两件此前做不到的事：
      1. **截止日期监控**——`bid_deadline` 是抽出来的，落库才能查"哪些快截止了"
      2. **重复上传检测**——同项目编号再传会回报 `duplicate_of`，
         供前端提示"该项目已解析过，是否做变更对比"

    同一项目编号**允许多份**（原文件 + 澄清公告），变更追踪正需要它们。
    """
    _tender_store_guard()
    try:
        return tender_store_service.save_document(
            {"parsed": request.parsed, "meta": request.meta}
        )
    except tender_store_service.TenderStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("保存招标文件失败")
        raise HTTPException(status_code=500, detail="保存失败，请稍后重试")


@app.get("/api/tender/documents")
def tender_documents(limit: int = 50, with_deadline_only: bool = False) -> dict[str, Any]:
    """已解析的招标文件列表（按上传时间倒序，含剩余天数与紧急度）。"""
    _tender_store_guard()
    try:
        items = tender_store_service.list_documents(
            limit=limit, with_deadline_only=with_deadline_only
        )
        return {"items": items, "count": len(items)}
    except tender_store_service.TenderStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("查询招标文件列表失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.get("/api/tender/documents/{doc_id}")
def tender_document_detail(doc_id: int) -> dict[str, Any]:
    """单份文件的完整抽取结果（含章节明细）。"""
    _tender_store_guard()
    try:
        doc = tender_store_service.get_document(doc_id)
    except tender_store_service.TenderStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"未找到 id={doc_id} 的招标文件")
    return doc


@app.delete("/api/tender/documents/{doc_id}")
def tender_document_delete(doc_id: int) -> dict[str, Any]:
    """删除一份文件记录。"""
    _tender_store_guard()
    try:
        ok = tender_store_service.delete_document(doc_id)
    except tender_store_service.TenderStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail=f"未找到 id={doc_id} 的招标文件")
    return {"status": "ok", "deleted": doc_id}


@app.get("/api/tender/deadlines")
def tender_deadlines(days: int = 30, include_expired: bool = False) -> dict[str, Any]:
    """未来 N 天内截止的项目（P3 截止日期监控）。

    ⚠️ `coverage` 字段是**必读**：本监控只覆盖已上传并抽到截止时间的文件，
    **不是全网项目监控**。不说明这点用户会以为系统漏报了项目。
    """
    _tender_store_guard()
    try:
        return tender_store_service.upcoming_deadlines(
            days=days, include_expired=include_expired
        )
    except tender_store_service.TenderStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("截止日期查询失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


# ---------------------------------------------------------------------------
# ⑧ 招标文件变更追踪
# ---------------------------------------------------------------------------


def _change_tracker_guard():
    if change_tracker_service is None:
        raise HTTPException(status_code=503, detail="变更追踪服务未就绪")


@app.post("/api/tender/compare")
async def tender_compare(
    old_file: UploadFile = File(..., description="原招标文件"),
    new_file: UploadFile = File(..., description="澄清/更正文件"),
    new_is_patch: bool | None = None,
) -> dict[str, Any]:
    """比对两份文件，输出结构化变更清单（非流式）。

    **业务价值**：招标项目常发澄清/更正公告。投标员最怕漏看变更 →
    按旧文件做标 → **废标**。本接口直接给出"改了哪些、哪些是实质性变更"。

    `new_is_patch` 留空时自动判断：补遗式澄清公告（信息量不足原文件一半）
    "没提到"**不等于**"被删除"，自动判断能避免把没变的字段全报成删除。
    """
    _change_tracker_guard()

    old_bytes = await old_file.read()
    new_bytes = await new_file.read()
    if not old_bytes or not new_bytes:
        raise HTTPException(status_code=400, detail="两份文件都不能为空")
    for label, blob in (("原文件", old_bytes), ("新文件", new_bytes)):
        if len(blob) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{label}过大（{len(blob) / 1048576:.1f}MB），上限 {_MAX_UPLOAD_BYTES // 1048576}MB",
            )

    try:
        return change_tracker_service.track_changes(
            old_file.filename or "old",
            old_bytes,
            new_file.filename or "new",
            new_bytes,
            new_is_patch=new_is_patch,
        )
    except Exception as exc:
        from src.services.change_tracker import ChangeTrackError

        if isinstance(exc, ChangeTrackError):
            raise HTTPException(status_code=400, detail=str(exc))
        logger.exception("变更比对失败")
        raise HTTPException(status_code=500, detail="比对失败，请稍后重试")


@app.post("/api/tender/compare/stream")
async def tender_compare_stream(
    old_file: UploadFile = File(...),
    new_file: UploadFile = File(...),
    new_is_patch: bool | None = None,
) -> StreamingResponse:
    """SSE 流式变更比对。

    要抽取**两份**文件，云端 20~40 秒、本地 Ollama 更久，必须有进度反馈。
    事件：`parse_old`/`extract_old`/`parse_new`/`extract_new`/`diff`/`done`/`error`。
    """
    _change_tracker_guard()

    old_bytes = await old_file.read()
    new_bytes = await new_file.read()
    old_name = old_file.filename or "old"
    new_name = new_file.filename or "new"

    def event_generator():
        yield _SSE_PADDING
        if not old_bytes or not new_bytes:
            yield _error_sse("两份文件都不能为空")
            yield "data: [DONE]\n\n"
            return
        try:
            for event_type, payload in change_tracker_service.stream_track(
                old_name, old_bytes, new_name, new_bytes, new_is_patch=new_is_patch
            ):
                yield "data: " + json.dumps(
                    {"type": event_type, **(payload if isinstance(payload, dict) else {"value": payload})},
                    ensure_ascii=False,
                ) + "\n\n"
        except Exception:
            logger.exception("流式变更比对失败")
            yield _error_sse("比对失败，请稍后重试")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# ⑤ 资格审查辅助
# ---------------------------------------------------------------------------


def _qualification_guard():
    if qualification_service is None:
        raise HTTPException(status_code=503, detail="资格审查服务未就绪")


class QualificationCheckRequest(BaseModel):
    """对已抽取的招标文件做资格审查（不重传文件）。"""

    parsed: dict[str, Any] = Field(..., description="招标文件抽取结果（parsed）")
    profile: dict[str, Any] = Field(default_factory=dict, description="企业资质档案")
    profile_name: str = Field(default="", description="或指定已保存的档案名")
    reference_date: str | None = Field(
        default=None, description="基准日期（YYYY-MM-DD），用于计算「近三年」，留空取今天"
    )


class ProfileSaveRequest(BaseModel):
    """保存企业资质档案。"""

    name: str = Field(..., min_length=1, description="档案名（唯一）")
    profile: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/qualification/check")
def qualification_check(request: QualificationCheckRequest) -> dict[str, Any]:
    """逐条比对资格要求，回答"**我能不能投这个标**"。

    ⚠️ `overall` 是保守判定：
        - 任一**硬性**要求不满足 → `unqualified`
        - 有"需人工确认"项    → `need_review`（**不会报 qualified**）
        - 全部满足            → `qualified`

    "需人工确认"这一档不能省：把不确定的判成"不满足"会让用户白白放弃能投的标；
    判成"满足"会让用户投一个必废的标。
    """
    _qualification_guard()

    profile = request.profile
    if request.profile_name:
        try:
            profile = qualification_service.load_profile(request.profile_name)
        except Exception as exc:
            from src.services.qualification import QualificationError

            if isinstance(exc, QualificationError):
                raise HTTPException(status_code=404, detail=str(exc))
            raise

    try:
        return qualification_service.check_qualifications(
            request.parsed, profile, reference_date=request.reference_date
        )
    except Exception as exc:
        from src.services.qualification import QualificationError

        if isinstance(exc, QualificationError):
            raise HTTPException(status_code=400, detail=str(exc))
        logger.exception("资格审查失败")
        raise HTTPException(status_code=500, detail="审查失败，请稍后重试")


@app.post("/api/qualification/check/upload")
async def qualification_check_upload(
    file: UploadFile = File(...),
    profile_name: str = "",
    reference_date: str | None = None,
) -> dict[str, Any]:
    """上传招标文件 + 档案名，一步完成解析与资格审查。"""
    _qualification_guard()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件过大")

    profile: dict[str, Any] = {}
    if profile_name:
        try:
            profile = qualification_service.load_profile(profile_name)
        except Exception as exc:
            from src.services.qualification import QualificationError

            if isinstance(exc, QualificationError):
                raise HTTPException(status_code=404, detail=str(exc))
            raise

    try:
        return qualification_service.check_from_document(
            file.filename or "unnamed", data, profile, reference_date=reference_date
        )
    except Exception as exc:
        from src.services.qualification import QualificationError

        if isinstance(exc, QualificationError):
            raise HTTPException(status_code=400, detail=str(exc))
        logger.exception("资格审查失败")
        raise HTTPException(status_code=500, detail="审查失败，请稍后重试")


@app.get("/api/qualification/profiles")
def qualification_profiles() -> dict[str, Any]:
    """已保存的企业资质档案列表。"""
    _qualification_guard()
    try:
        items = qualification_service.list_profiles()
        return {"items": items, "count": len(items)}
    except Exception:
        logger.exception("查询档案列表失败")
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")


@app.get("/api/qualification/profiles/{name}")
def qualification_profile_get(name: str) -> dict[str, Any]:
    """取单个档案的完整内容。"""
    _qualification_guard()
    from src.services.qualification import QualificationError

    try:
        return qualification_service.load_profile(name)
    except QualificationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("读取档案失败")
        raise HTTPException(status_code=500, detail="读取失败，请稍后重试")


@app.post("/api/qualification/profiles")
def qualification_profile_save(request: ProfileSaveRequest) -> dict[str, Any]:
    """新建或覆盖保存企业资质档案（同名覆盖）。"""
    _qualification_guard()
    from src.services.qualification import QualificationError

    try:
        return qualification_service.save_profile(request.name, request.profile)
    except QualificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("保存档案失败")
        raise HTTPException(status_code=500, detail="保存失败，请稍后重试")


@app.delete("/api/qualification/profiles/{name}")
def qualification_profile_delete(name: str) -> dict[str, Any]:
    """删除档案。"""
    _qualification_guard()
    from src.services.qualification import QualificationError

    try:
        ok = qualification_service.delete_profile(name)
    except QualificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("删除档案失败")
        raise HTTPException(status_code=500, detail="删除失败，请稍后重试")
    if not ok:
        raise HTTPException(status_code=404, detail=f"未找到档案「{name}」")
    return {"status": "ok", "deleted": name}


# ---------------------------------------------------------------------------
# ⑪ 报告导出（Word）
# ---------------------------------------------------------------------------


def _report_guard():
    if report_service is None:
        raise HTTPException(status_code=503, detail="报告导出服务未就绪")


class ReportExportRequest(BaseModel):
    """导出投标决策报告。

    直接传 `/api/tender/extract` 的返回体即可（`parsed` + `meta`）；
    `qualification` / `priceProfile` / `competitor` 可选，传了就各自成章。
    """

    parsed: dict[str, Any] = Field(..., description="招标文件抽取结果")
    meta: dict[str, Any] = Field(default_factory=dict, description="抽取元信息")
    qualification: dict[str, Any] | None = Field(default=None, description="资格审查结果（可选）")
    price_profile: dict[str, Any] | None = Field(default=None, description="报价策略结果（可选）")
    competitor: dict[str, Any] | None = Field(default=None, description="竞争格局结果（可选）")


@app.post("/api/report/tender")
def report_tender(request: ReportExportRequest) -> Response:
    """生成**投标决策报告**（Word），直接以附件形式下载。

    为什么是"决策报告"而不是"数据导出"：用户拿到结果后要交给领导或团队，
    需要一份能打印、能传阅、能逐条打勾的文档。故按**投标工作流**组织：
    项目概况 → 关键时间 → 资格自查表 → 评分标准 → **废标红线** →
    技术要点 → 需提交材料(勾选框) → 报价参考 → 数据质量与局限。

    「数据质量与局限」一章**不省略**：抽取完整度低或有警告时必须让读者
    知道结论的适用范围。把局限藏起来比没有报告更糟。
    """
    _report_guard()
    try:
        data = report_service.build_tender_report(
            request.parsed,
            request.meta,
            qualification=request.qualification,
            price_profile=request.price_profile,
            competitor=request.competitor,
        )
    except report_service.ReportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("生成报告失败")
        raise HTTPException(status_code=500, detail="生成报告失败，请稍后重试")

    filename = report_service.report_filename(request.parsed)
    # 中文文件名必须用 RFC 5987 的 filename* 形式，否则浏览器会拿到乱码名
    quoted = urllib.parse.quote(filename)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename=\"tender-report.docx\"; filename*=UTF-8''{quoted}",
            "Content-Length": str(len(data)),
        },
    )


@app.post("/api/report/tender/from-document")
async def report_tender_from_document(
    file: UploadFile = File(..., description="招标文件 PDF/Word"),
    include_qualification: bool = False,
    profile_name: str = "",
) -> Response:
    """一步到位：上传文件 → 解析 → 抽取 →（可选资格审查）→ 生成报告。

    `include_qualification=True` 时需给 `profile_name`，会先做资格审查再写入报告。
    """
    _report_guard()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件过大，上限 50MB")

    try:
        extracted = tender_extractor_service.extract_from_document(
            file.filename or "unnamed", raw
        )
    except document_parser_service.DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except tender_extractor_service.TenderExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    qual_result = None
    if include_qualification and qualification_service is not None:
        try:
            profile = qualification_service.load_profile(profile_name) if profile_name else {}
            qual_result = qualification_service.check_qualifications(
                extracted["parsed"], profile
            )
        except Exception:
            # 资格审查失败不应让整份报告失败——报告主体仍有价值
            logger.warning("报告中的资格审查环节失败，跳过该章节", exc_info=True)

    try:
        data = report_service.build_tender_report(
            extracted["parsed"], extracted["meta"], qualification=qual_result
        )
    except report_service.ReportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    filename = report_service.report_filename(extracted["parsed"])
    quoted = urllib.parse.quote(filename)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename=\"tender-report.docx\"; filename*=UTF-8''{quoted}",
            "Content-Length": str(len(data)),
        },
    )

# ---------------------------------------------------------------------------
# ⑨ 合规性检查
# ---------------------------------------------------------------------------


def _compliance_guard():
    if compliance_service is None:
        raise HTTPException(status_code=503, detail="合规检查服务未就绪")


class ComplianceCheckRequest(BaseModel):
    """对已抽取的招标文件做合规检查（不重传文件）。"""

    parsed: dict[str, Any] = Field(..., description="招标文件抽取结果（parsed）")
    include_law_basis: bool = Field(default=True, description="是否附带法规依据")


@app.get("/api/compliance/rules")
def compliance_rules() -> dict[str, Any]:
    """返回规则清单，供前端展示"我们检查哪些方面"（不在前端硬编码）。"""
    _compliance_guard()
    return compliance_service.rules_catalog()


@app.post("/api/compliance/check")
def compliance_check(request: ComplianceCheckRequest) -> dict[str, Any]:
    """检查招标文件中涉嫌违规的条款。

    ⚠️ 两层设计：
        第一层 **规则检出**（确定性）—— 匹配已知的违规特征模式，可解释可复现
        第二层 **法规引用**（依据）—— 挂上法规条文，让用户能自己核对原文

    规则命中是确定的；法规语料不可用时第一层仍完整工作（语料是增强而非前提）。

    ⚠️ 输出是「**疑似**」而非违规认定：同一表述在不同项目背景下合法性可能不同。
    响应里的 `disclaimer` 是必读字段。
    """
    _compliance_guard()
    try:
        return compliance_service.check_compliance(
            request.parsed, include_law_basis=request.include_law_basis
        )
    except compliance_service.ComplianceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("合规检查失败")
        raise HTTPException(status_code=500, detail="检查失败，请稍后重试")


@app.post("/api/compliance/check/upload")
async def compliance_check_upload(
    file: UploadFile = File(..., description="招标文件 PDF/Word"),
    include_law_basis: bool = True,
) -> dict[str, Any]:
    """一步到位：上传文件 → 解析 → 抽取 → 合规检查。"""
    _compliance_guard()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件过大，上限 50MB")

    try:
        extracted = tender_extractor_service.extract_from_document(
            file.filename or "unnamed", raw
        )
    except document_parser_service.DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except tender_extractor_service.TenderExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = compliance_service.check_compliance(
            extracted["parsed"], include_law_basis=include_law_basis
        )
    except compliance_service.ComplianceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 把抽取元信息一并带上：检查结论的可靠性取决于抽取质量，
    # 用户需要知道"这份检查基于多少完整度的抽取"
    result["extract_meta"] = extracted.get("meta")
    return result


@app.get("/api/compliance/regulations/stats")
def compliance_regulation_stats() -> dict[str, Any]:
    """法规语料库的规模统计（用于说明检查的法规依据覆盖程度）。

    语料未入库时返回 `available: false` 而不是报错——规则检出不依赖它。
    """
    _compliance_guard()
    try:
        from src.services import regulation_store

        stats = regulation_store.regulation_stats()
        return {"available": True, **stats}
    except Exception as exc:
        logger.info("法规语料不可用：%s", exc)
        return {
            "available": False,
            "reason": "法规语料尚未入库，当前检查只使用规则匹配（结论仍然有效，但没有法规条文引用）",
        }


# ---------------------------------------------------------------------------
# 联网搜索（统一两路数据源 + 相关性评分）
# ---------------------------------------------------------------------------


class WebSearchRequest(BaseModel):
    """直接联网搜索（不经 Agent）。"""

    query: str = Field(..., min_length=1, description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=10, description="每路返回条数")
    min_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="相关度下限：低于它的结果被丢弃；留空不过滤",
    )
    providers: list[str] | None = Field(
        default=None, description='数据源，可选 ["tavily","zhipu"]；留空两路都用'
    )


def _web_guard():
    if web_fallback_service is None:
        raise HTTPException(status_code=503, detail="联网搜索服务未就绪")


@app.get("/api/web/status")
def web_status() -> dict[str, Any]:
    """联网数据源的可用性 + 本地覆盖判定阈值（供前端提示与校准观察）。"""
    _web_guard()
    return {
        "providers": web_fallback_service.is_available(),
        "min_relevance": web_fallback_service.DEFAULT_MIN_RELEVANCE,
        "default_top_k": web_fallback_service.DEFAULT_WEB_TOP_K,
        "modes": {
            "auto": "本地知识库无匹配时自动联网（推荐）",
            "on": "强制联网，获取最新公告与法规更新",
            "off": "只用本地知识库，不联网",
        },
    }


@app.post("/api/web/search")
def web_search_endpoint(request: WebSearchRequest) -> dict[str, Any]:
    """联网搜索，返回带**相关性评分**的结果（按相关度降序）。

    两路数据源互补：Tavily（英文语料强、带 AI 摘要）+ 智谱（中文站点强、带发布时间）。
    智谱不返回分数，故用与本地检索**同一套嵌入模型**补算余弦相似度——
    好处是分数与本地知识库的 relevance **同量纲**，可以直接比较。

    **部分失败不整体失败**：一路挂了另一路的结果仍然有用，
    `errors` 字段如实列出失败的数据源与原因。
    """
    _web_guard()
    try:
        return web_fallback_service.search_web(
            request.query,
            max_results=request.max_results,
            providers=request.providers or ("tavily", "zhipu"),
            min_score=request.min_score,
        )
    except web_fallback_service.WebSearchUnavailable as exc:
        # 503 而不是 500：这是"没配密钥/数据源都不可用"，属能力缺失而非程序错误
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("联网搜索失败")
        raise HTTPException(status_code=500, detail="搜索失败，请稍后重试")


@app.post("/api/web/coverage")
def web_coverage(request: WebSearchRequest) -> dict[str, Any]:
    """判断本地知识库能否回答该问题（**两层判定**）。

    第一层相似度粗筛 + 第二层 LLM 判读"资料能否回答"。
    两层都要：实测相似度只能判断**主题相关**，不能判断**能否回答**——
    问"中国大陆和香港的招投标法规区别"时，本地全是大陆法规，
    相似度却达 0.55 过了门槛，纯相似度会把它误判为"本地够用"。

    本端点供前端解释"为什么这次联网了"。
    """
    _web_guard()
    cov = web_fallback_service.local_coverage(request.query)
    return cov.to_dict()


# ---------------------------------------------------------------------------
# 前端静态托管与 SPA 兜底（**必须注册在文件末尾**）
# ---------------------------------------------------------------------------
#
# ★ 为什么必须放在最后：FastAPI 按**注册顺序**匹配路由
# -------------------------------------------------
# 兜底路由 `/{full_path:path}` 会匹配任意路径。若注册在 `/api/*` 之前，
# 所有接口都会被它先截住 —— 表现为"前端页面正常、接口全部 404"，
# 而且错误信息毫无线索。
#
# 这一点是**实测踩到的**：第一版把兜底写在 `app = FastAPI(...)` 后面，
# 结果 `/api/health/live` 返回 404（`tests/test_frontend_hosting.py::
# test_existing_api_routes_still_work` 抓到的就是这个）。故移到文件末尾 ——
# 此位置之后**不得再注册任何路由**。
if _FRONTEND_READY:
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # 带哈希文件名的资源由 StaticFiles 提供（可长期缓存）
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):
        """SPA 兜底：非 /api 路径一律交给前端路由。

        ★ 为什么必须兜底：前端用的是 createBrowserRouter（history 路由），
        `/chat`、`/analysis` 这些路径在服务器上**没有对应文件**。
        不兜底的话，用户刷新任何子页面都会 404 —— 而 dev server 自带兜底，
        所以这个问题**只会在部署后暴露**。

        `/api/*` 不在此列：走到这里说明接口路径不存在，应如实 404，
        而不是丢一个 index.html 回去（那会让前端把"接口 404"误读成"页面正常"）。
        """
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="接口不存在")
        # 真实存在的文件（logo.svg / favicon 等）优先按文件返回
        root = _FRONTEND_DIST.resolve()
        candidate = (_FRONTEND_DIST / full_path).resolve()
        inside = str(candidate).startswith(str(root))
        if inside and candidate.is_file():
            return FileResponse(str(candidate))
        # 其余（含 /chat 这类前端路由）都回 index.html
        #
        # ★ `no-cache` 是必须的：index.html 里记着带哈希的资源名，
        # 一旦被缓存，发版后用户会一直请求**已删除的旧 chunk**。
        # （前端为此另有"页面版本已更新，请刷新"的兜底提示，但根治在响应头。）
        return FileResponse(
            str(_FRONTEND_DIST / "index.html"),
            headers={"Cache-Control": "no-cache"},
        )
