"""知识图谱结构化查询服务（供前端专用端点调用）。

为什么需要这个模块
------------------
`src/tools/rag_tools.py` 的 `search_knowledge_graph` 是给 **LLM** 用的工具，
它把结果格式化成中文 markdown 文本（对齐、加小标题、按 MAX_LIMIT 截断），
因为 LLM 只吃文本。但前端要画表格、标徽章、排序、展示完整字段时，
解析那段文本既脆弱又丢信息（金额被 `_fmt_amount` 转成带千分位的字符串、
条数被截断、字段名被本地化成中文）。

本模块直接调用 `Neo4jClient` 的**结构化方法**（本来就返回 dict/list），
把原始数据以 JSON 友好形式交给前端——展示层的事交给展示层做。

设计边界
--------
- **只读**：只调用 client 的查询方法，不做任何写入
- **不格式化**：不拼接中文文本、不做千分位、不截断（截断由路由层的 limit 控制）
- **复用校验**：查询类型白名单直接取 `rag_tools._GRAPH_QUERY_TYPES`，
  新增查询类型时两边自动同步，不会漂移
- **失败抛异常**：与工具层"失败返回可读文案"不同，这里失败必须让路由层
  转成 HTTP 状态码（前端据此展示错误态），而不是把错误文案当数据返回
"""

from __future__ import annotations

from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# 与 rag_tools 共用同一份白名单，避免两处各写一份导致漂移。
# 取不到时退化为空元组——路由层会因此拒绝所有查询类型，
# 但这属于"内部契约被破坏"的显式失败，好过静默接受未知类型。
try:
    from src.tools.rag_tools import (
        _GRAPH_QUERY_TYPES,
        _get_neo4j,
        MAX_LIMIT,
        GRAPH_DEFAULT_LIMIT,
    )
except Exception:  # pragma: no cover - 工具层未就绪
    logger.exception("无法从 rag_tools 导入图谱查询契约")
    _GRAPH_QUERY_TYPES = ()
    MAX_LIMIT = 200
    GRAPH_DEFAULT_LIMIT = 20

    def _get_neo4j():  # type: ignore[misc]
        return None


class GraphUnavailableError(RuntimeError):
    """图谱客户端不可用（未配置 NEO4J_* 或连接失败）。路由层应转 503。"""


class GraphQueryError(ValueError):
    """查询类型或参数不合法。路由层应转 400/422。"""


def available_query_types() -> list[str]:
    """供前端渲染查询类型选择器（避免前端硬编码一份可能过期的清单）。"""
    return list(_GRAPH_QUERY_TYPES)


def graph_available() -> bool:
    """图谱是否可用。

    与 web_search 的 health() 不同，这里**真的构造客户端**——因为 Neo4j 的
    失败模式是"配置缺失"或"连不上"，只有真试一次才知道。构造本身很轻
    （只建 driver，不发查询），故健康检查里调用它不会拖慢 /api/health。
    """
    return _get_neo4j() is not None


def _require_entity(arguments: dict) -> str:
    """取出必填的实体名，缺失时抛 GraphQueryError。

    与工具层 `_dispatch_graph` 的 `missing` 二元组机制不同：这里直接抛异常，
    因为前端是**显式选类型 + 填表单**，缺参数属于客户端 bug，应该明确报错，
    而不是返回一句"缺少参数"的文本让调用方去猜。
    """
    for key in ("entity", "keyword", "query", "name", "subject"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise GraphQueryError("缺少参数 entity（实体名称）")


def _safe_limit(arguments: dict) -> int:
    """limit 收敛到 [1, MAX_LIMIT]，非法值退回默认。"""
    raw = arguments.get("limit")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return GRAPH_DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


def run_graph_query(query_type: str, arguments: dict | None = None) -> dict[str, Any]:
    """执行图谱查询，返回 `{"query_type": str, "data": ..., "count": int}`。

    `data` 的形状随查询类型而定（与 Neo4jClient 的返回一致）：
        graph_stats         -> dict   {标签: 数量}
        entity_detail       -> dict   {subject, suppliers, purchasers, ...}
        其余 4 类           -> list[dict]

    `count` 是为前端方便统计而算的条目数（dict 记键数，list 记元素数）。

    异常：
        GraphQueryError       查询类型未登记 / 缺必填参数
        GraphUnavailableError 图谱客户端不可用（未配置或连不上）
    """
    arguments = arguments if isinstance(arguments, dict) else {}

    if query_type not in _GRAPH_QUERY_TYPES:
        raise GraphQueryError(
            f"未知的查询类型 {query_type!r}；可用：{'、'.join(_GRAPH_QUERY_TYPES)}"
        )

    client = _get_neo4j()
    if client is None:
        raise GraphUnavailableError("知识图谱客户端不可用（检查 NEO4J_* 配置与实例状态）")

    limit = _safe_limit(arguments)

    try:
        if query_type == "graph_stats":
            data: Any = client.graph_stats()
        elif query_type == "top_traded":
            data = client.top_traded(limit=limit)
        elif query_type == "search_entity":
            data = client.search_entity(_require_entity(arguments), limit=limit)
        elif query_type == "items_by_entity":
            data = client.items_by_entity(_require_entity(arguments), limit=limit)
        elif query_type == "entity_detail":
            data = client.entity_detail(_require_entity(arguments), limit=limit)
        elif query_type == "supplier_by_subject":
            data = client.supplier_by_subject(_require_entity(arguments), limit=limit)
        else:  # pragma: no cover - 上面的白名单校验已覆盖
            raise GraphQueryError(f"查询类型 {query_type!r} 未实现")
    except GraphQueryError:
        raise
    except Exception as exc:
        # 客户端契约是"失败返回空、绝不抛异常"，所以走到这里通常是驱动层的
        # 连接问题（实例暂停、网络抖动）。原样抛出交给路由层判 503。
        logger.warning("图谱查询失败（%s）：%s", query_type, exc)
        raise GraphUnavailableError(f"图谱查询失败：{type(exc).__name__}") from exc

    if isinstance(data, dict):
        count = len(data)
    elif isinstance(data, list):
        count = len(data)
    else:  # pragma: no cover - 客户端不会返回其他类型
        data, count = [], 0

    return {"query_type": query_type, "data": data, "count": count}


__all__ = [
    "GraphQueryError",
    "GraphUnavailableError",
    "available_query_types",
    "graph_available",
    "run_graph_query",
]
