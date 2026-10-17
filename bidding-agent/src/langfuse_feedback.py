"""会话 → 最近一次 Langfuse trace 的关联存储。

★ 为什么需要它（这是接入"用户反馈 → Langfuse Score"的关键一环）
====================================================
Langfuse 的 Score 必须挂在**具体的 trace_id** 上（`create_score(trace_id=...)`）。
而两个事实让这个 id 很难拿到：

    ① trace_id 由 SDK 在 `LangfuseTagger.begin()` 时生成，**只活在内存里**，
       请求结束就没有了
    ② 用户点赞发生在**之后**（`POST /api/feedback` 是独立请求，
       只带 `session_id` + `message_id` + `rating`）

故需要一个**跨请求**的桥：把"这次问答的 trace_id"存到能按 `session_id`
找回的地方。

★ 为什么不用内存字典（那是第一反应）
==========================
内存字典的问题：
    · **多进程部署下失效** —— uvicorn 多 worker 时，写它的进程与读它的
      进程可能不是同一个，反馈会静默查不到 trace_id
    · **重启即丢** —— 用户还没点赞就重启，那条反馈就永远接不上

故**落到 PostgreSQL**（本项目已有该库，且反馈本身就落库，join 是自然的）。
另外 `messages.trace_id` 也顺带让"这条回答对应哪条 trace"变成可直接查的事实，
对排查问题本身就有用。

★ 为什么不直接写 `messages.trace_id`
=============================
`messages` 行由前端在**收到 done 帧之后**才调 `POST /api/conversations` 创建，
那时 agent 那侧早就结束了。故 agent 侧只能先写这个映射表，
再让落库端点把 trace_id 一并写进 `messages`。

★ 有界与过期
========
表会随请求增长，故：
    · 每个 session 只保留**最近一次**（`ON CONFLICT DO UPDATE`）
    · 写入时顺带清理超过 `MAX_AGE_HOURS` 的行（不做定时任务，省一个进程）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 关联行的最长保留时间。超过就清掉。
#:
#: 选 24 小时的理由：用户点赞通常在**当次会话内**（分钟级），
#: 但"回头再来点赞"也可能隔天。24 小时足够宽，且不会让表无限增长。
MAX_AGE_HOURS = 24

_TABLE = "session_traces"


def ensure_table(pg: Any) -> bool:
    """建表（幂等）。返回是否可用。

    与项目里其它"可选增强"一致：**建表失败不影响问答**，
    只是反馈接不上 Langfuse。故这里吞异常并返回 False。

    ★ 必须 `fetch=False`（实测踩到的坑）
    -------------------------------
    `PostgresClient._query` 默认 `fetch=True`，会执行 `cur.fetchall()`；
    而 **DDL 不产生结果集**，于是 psycopg 抛
    `ProgrammingError: the last operation didn't produce records`。

    后果很隐蔽：**表其实建成了**，但异常被 `_query` 吞掉并返回 `[]`，
    我这里的 `except` 又把它当失败 → `ensure_table` 返回 False，
    "映射写入" 也返回 False —— 全程**没有任何报错**，
    只有"反馈接不上 Langfuse"这一个结果。
    `_query` 的 docstring 早就写了 "`fetch=False` 用于 DDL"，是我漏用。
    """
    try:
        pg._query(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                session_id  text PRIMARY KEY,
                trace_id    text NOT NULL,
                updated_at  timestamptz NOT NULL DEFAULT now()
            )
            """,
            fetch=False,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.warning("创建 %s 表失败（反馈将无法关联 Langfuse）", _TABLE, exc_info=True)
        return False


def remember_session_trace(session_id: str, trace_id: str) -> bool:
    """记下"这个会话最近一次问答的 trace_id"。

    **永不抛异常**：这是观测旁路。失败只意味着反馈接不上 Langfuse，
    不该影响任何用户可见行为。

    顺带清理过期行（不做定时任务，省一个常驻进程）。
    """
    if not session_id or not trace_id:
        return False
    try:
        from src.database.postgresql_client import PostgresClient

        pg = PostgresClient()
        ensure_table(pg)
        # 这条有 `RETURNING`，故保持默认 `fetch=True`
        pg._query(
            f"""
            INSERT INTO {_TABLE} (session_id, trace_id, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (session_id)
            DO UPDATE SET trace_id = EXCLUDED.trace_id, updated_at = now()
            RETURNING session_id
            """,
            (session_id, trace_id),
        )
        # 顺手清理：一次 DELETE 比一个定时任务便宜得多，且天然"顺路"。
        # 同样不取结果，故 `fetch=False`。
        pg._query(
            f"DELETE FROM {_TABLE} WHERE updated_at < now() - make_interval(hours => %s)",
            (MAX_AGE_HOURS,),
            fetch=False,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.debug("记录 session→trace 映射失败", exc_info=True)
        return False


def lookup_trace_id(session_id: str) -> str | None:
    """按 session_id 取最近一次的 trace_id。取不到返回 None。

    取不到是**正常情况**（未配 Langfuse、或映射已过期），故不告警。
    """
    if not session_id:
        return None
    try:
        from src.database.postgresql_client import PostgresClient

        pg = PostgresClient()
        rows = pg._query(
            f"SELECT trace_id FROM {_TABLE} WHERE session_id = %s", (session_id,)
        )
        if rows and isinstance(rows[0], dict):
            v = rows[0].get("trace_id")
            return str(v) if v else None
        return None
    except Exception:  # noqa: BLE001
        logger.debug("查询 session→trace 映射失败", exc_info=True)
        return None


#: 本项目的 rating 取值 → Langfuse Score 的数值。
#:
#: ★ 为什么要映射而不是直接传 'up'/'down'
#: -----------------------------------
#: Langfuse 的 Score 支持 NUMERIC / CATEGORICAL 等类型。
#: 用 NUMERIC(-1/1) 的好处是**可以直接算均值**（"满意度趋势"），
#: 而 CATEGORICAL 只能看分布。两者都有用，但趋势更常用于发现退化。
#:
#: 取值来自实测（`SELECT rating FROM feedback` 得到 `'up'`），
#: 而 `FeedbackRequest` 的校验器只允许 `up` / `down`。
_SCORE_VALUES = {"up": 1.0, "down": -1.0}


def score_feedback(session_id: str, message_id: int, rating: str, pg=None) -> bool:
    """把用户反馈挂成 Langfuse Score。返回是否成功。

    **永不抛异常**：这是观测旁路，失败只意味着 Langfuse 里少一个评分，
    绝不能影响"反馈已保存"这个已经完成的事实。

    ★ 三条前置检查（任一不满足就静默返回 False）
    ------------------------------------
        ① 未启用 Langfuse → 没有 trace 可挂（不是错误）
        ② rating 不在映射表里 → 不猜语义（宁可少一个评分，也不写错值）
        ③ 消息上没有 trace_id → 老消息、或加列之前写入的
    """
    try:
        # ★ 用 `is_enabled()` 而不是 `is_configured()`
        # ----------------------------------------
        # 用户把 `LANGFUSE_TRACING_ENABLED` 设为 false/0/no 时，期望的是
        # "**一条数据都别往外发**"。评分虽然是"写"而不是"追踪"，但它同样是
        # 往外部服务发数据——只查 `is_configured()`（凭据齐不齐）会让总开关
        # 形同虚设。`is_enabled()` = 凭据齐 + 总开关开 + SDK 装好。
        from src.langfuse_tracing import build_client, is_enabled

        if not is_enabled():
            return False

        value = _SCORE_VALUES.get(str(rating or "").strip().lower())
        if value is None:
            logger.debug("未知的 rating %r，不写 Langfuse 评分", rating)
            return False

        if pg is None:
            from src.database.postgresql_client import PostgresClient

            pg = PostgresClient()
        trace_id = pg.find_trace_id_by_message(message_id)
        if not trace_id:
            logger.debug("消息 %s 没有 trace_id，跳过 Langfuse 评分", message_id)
            return False

        # 每次现建客户端（不复用 `langfuse_tracing.get_client()` 的缓存）：
        # 用户的 key / host 可能在运行期被换掉，评分挂到旧凭据的客户端上
        # 会表现为"评分悄悄没写进去"。构造参数集中在 build_client() 里。
        client = build_client()
        if client is None:
            return False
        # `create_score` 是 SDK 里**唯一**接受 trace_id 的写入口；
        # `score_current_trace` 需要 OTel 上下文（反馈是独立请求，没有）故不能用。
        client.create_score(
            name="user-feedback",
            value=value,
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=f"来自 /api/feedback（session={session_id[:8]}…）"
            if session_id
            else None,
        )
        # 评分是"用户点了赞/踩"的即时动作，值得立刻送出去：
        # 不 flush 的话进程重启会把它丢掉，而这条数据是人工产生的、无法重放。
        try:
            from src.langfuse_tracing import flush_client

            flush_client(client, 3.0)
        except Exception:  # noqa: BLE001
            pass
        logger.info("已把反馈 %s 挂到 trace %s", rating, trace_id)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("写入 Langfuse 评分失败", exc_info=True)
        return False
