"""Qdrant 只读适配器。

只做两件事：**数点数**、**判断集合是否存在**。质量扫描需要这两项；
目录端点不调用它（目录来自静态注册表，不连 Qdrant）。

刻意不调用 `VectorStore.ensure_collection()`：那是“建集合”的写操作，
质量扫描如果误建一个空集合，会把“集合缺失”这个故障掩盖成“0 条数据”。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.config import settings
from src.datahub.contract import DataHubUnavailableError
from src.rag.regulation_store import REGULATION_COLLECTION
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_stores: dict[str, VectorStore] = {}
_lock = threading.Lock()


def _store(collection: str) -> VectorStore:
    """按集合名缓存 VectorStore（只读；不需要嵌入模型）。"""
    with _lock:
        store = _stores.get(collection)
        if store is None:
            store = VectorStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection=collection,
                dim=settings.qdrant_vector_size,
                distance=settings.qdrant_distance,
                timeout=int(settings.qdrant_timeout),
            )
            _stores[collection] = store
        return store


def count(collection: str) -> int:
    """集合点数；失败抛 DataHubUnavailableError。"""
    try:
        return int(_store(collection).count())
    except Exception as exc:
        raise DataHubUnavailableError(
            f"Qdrant 查询失败：{type(exc).__name__}"
        ) from exc


def collection_exists(collection: str) -> bool:
    """集合是否存在；失败抛 DataHubUnavailableError。"""
    try:
        return bool(_store(collection).client.collection_exists(collection))
    except Exception as exc:
        raise DataHubUnavailableError(
            f"Qdrant 连接失败：{type(exc).__name__}"
        ) from exc


def known_collections() -> dict[str, str]:
    """逻辑名 → 物理集合名。问答集合由 QDRANT_COLLECTION 决定。"""
    return {
        "qa": settings.qdrant_collection,
        "regulations": REGULATION_COLLECTION,
    }


def clear_cache() -> None:
    """清空客户端缓存（测试与改配置后立即生效用）。"""
    with _lock:
        _stores.clear()
