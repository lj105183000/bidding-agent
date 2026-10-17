"""给 `messages` 表补 `image_analyses` 列（幂等）。

★ 用途（"只存解析文本"的落库方案）
==============================
图片本身**不入库**（原图体积大、要对象存储、还要处理生命周期），
只把视觉模型的解析文本随消息存下来。理由：

    · 用户真正需要回看的是"当时系统从图里读出了什么"（金额、编号、日期），
      而不是那张截图的像素
    · 解析文本是**纯文本**，进 JSONB 就能直接用 `->>` 查（例如"找出所有
      解析出过某项目编号的会话"），原图做不到这一点
    · 原图不落库也顺带避开了隐私与保留期问题

⚠️ 代价（如实记录）：刷新页面后**看不到缩略图**，只能看到解析文本。
若将来要"回看原图"，那是另一套方案（对象存储 + 引用 id），不要往这一列里塞 base64。

★ 为什么与 `trace_migration` 长得像、却不共用同一个函数
=================================================
两处都是"给 messages 加一列"的幂等 DDL，逻辑几乎一样。没有合并成一个通用
`ensure_column(pg, name, type)` 的原因是：`trace_migration` 的**源码文本**
被既有测试直接断言（`test_ensure_column_source_uses_fetch_false` 检查那个
函数体里必须出现 `fetch=False`）—— 把它改成一行委托会让那条测试失去意义，
而动别人的测试只为了少写 10 行不划算。故这里保留同样的骨架，并把那个坑
再写一遍（它值得写两遍）：

★ **必须 `fetch=False`**
====================
`PostgresClient._query` 默认走 `fetchall()`，而 `ALTER TABLE` **不产生结果集**
→ psycopg 抛 `ProgrammingError: the last operation didn't produce records`。
列其实已经加上了，异常却被 `_query` 吞掉 → 这里误判为失败 → 静默降级
（表现为"图片解析结果永远不落库，且没有任何报错"）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 列名。JSONB 而不是 TEXT：解析结果本就是结构化的
#: `[{"name": ..., "analysis": ..., "model": ...}, ...]`，存成 JSON 字符串后
#: 想按"解析出的编号"过滤只能全表 LIKE。
_COLUMN = "image_analyses"


def ensure_column(pg) -> bool:
    """给 `messages` 加 `image_analyses` 列（幂等）。返回是否可用。"""
    try:
        pg._query(
            f"ALTER TABLE messages ADD COLUMN IF NOT EXISTS {_COLUMN} jsonb",
            fetch=False,
        )
        return True
    except Exception:  # noqa: BLE001 - 加不上列不该影响会话保存
        logger.warning(
            "给 messages 加 %s 列失败（图片解析结果将不落库）", _COLUMN,
            exc_info=True,
        )
        return False


__all__ = ["ensure_column"]
