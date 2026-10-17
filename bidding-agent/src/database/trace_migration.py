"""给 `messages` 表加 `trace_id` 列（幂等），并让 `save_message` 支持写入。

★ 为什么不做成"独立的映射表"
=====================
反馈表里已有 `message_id`（实测：`{'message_id': 76}`），
它指的就是 `messages.id`。把 `trace_id` 放在 `messages` 上，
反馈端点一次 join 就能拿到 —— 不需要再维护第三张表。

★ 为什么用 ALTER TABLE IF NOT EXISTS 而不是建表
=========================================
列要加在**既有表**上。PostgreSQL 9.6+ 支持
`ADD COLUMN IF NOT EXISTS`，故这条语句可以反复执行。

★ 失败不影响功能
============
这是"可选增强"：加不上列，反馈就接不上 Langfuse，
但会话保存、反馈保存本身照旧。故所有涉及点都吞异常。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_COLUMN = "trace_id"


def ensure_column(pg) -> bool:
    """给 messages 加 trace_id 列（幂等）。返回是否可用。

    ★ 必须 `fetch=False`（与 `langfuse_feedback.ensure_table` 同一个坑）
    ------------------------------------------------------------
    `_query` 默认会 `fetchall()`，而 **ALTER TABLE 不产生结果集**，
    于是 psycopg 抛 `ProgrammingError: the last operation didn't produce records`。
    列其实加上了，但异常被吞 → 这里误判为失败 → 静默降级。
    """
    try:
        pg._query(
            f"ALTER TABLE messages ADD COLUMN IF NOT EXISTS {_COLUMN} text",
            fetch=False,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.warning(
            "给 messages 加 %s 列失败（反馈将无法关联 Langfuse）", _COLUMN,
            exc_info=True,
        )
        return False
