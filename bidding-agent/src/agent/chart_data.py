"""从**工具手里的结构化数据**产出图表数据（替代解析工具文本）。

★ 为什么必须改（原方案的可靠性问题）
==========================
原实现是"**正则解析格式化后的文本**"（`charts.py` 316 行）：
工具先把数据拼成给 LLM 看的文本，我们再从那段文本里把数字**猜**回来。

三个固有缺陷：
    ① **格式即契约**：工具的展示格式一改（加个空格、换个分隔符），
       解析就静默失效 —— 图表消失，且没有任何报错
    ② **单元测试只能测"我编的样本"**：真实输出与样本的差异无法覆盖
       （实测踩到过：`| a | b |` 与 `a | b` 差一对竖线就完全解析不出来）
    ③ **数字被格式化后不可逆**：`_fmt_amount` 把金额变成 `1.2 万元`，
       解析回来就丢了精度；`_truncate` 还会把长字段截断

改为**工具把手里的结构化数据直接交出来**：
    · 数据在格式化**之前**就是 `list[dict]`，交出来零成本
    · 格式怎么变都不影响图表
    · 数字保持原值（不经格式化）

★ 与 `charts.py` 的关系
==================
`charts.py` **保留作为兜底**，不删：
    · 联网类工具（`search_web*`）的结果本就是自然语言，没有结构化数据可交
    · 逐个改造 8 个工具有风险，先让"有结构数据的"走新路
调用方优先用本模块的产出，没有再退回解析。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 与 `charts.py` 保持一致的展示上限（超过会截断并在 `note` 里说明）。
MAX_BARS = 12

#: 最多几张图。与 `src/agent/constants.MAX_CHARTS` 同值。
MAX_CHARTS = 3


def _num(value: Any) -> float | None:
    """尽量转成数值。转不了返回 None（**不补 0**）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("，", "")
        # 去掉常见单位后缀：5.2万 / 1.2亿 / 30%
        mult = 1.0
        if s.endswith("%"):
            s = s[:-1]
        elif s.endswith("万"):
            s, mult = s[:-1], 10000.0
        elif s.endswith("亿"):
            s, mult = s[:-1], 100000000.0
        try:
            return float(s) * mult
        except ValueError:
            return None
    return None


def _dedupe_labels(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """同名标签去重（保留第一个）。

    ★ 为什么必须去重：横条图的 Y 轴用标签当 key，重名会让 React 报
    duplicate key，且视觉上两条同名条无法区分 —— 而"同名"在真实数据里
    很常见（不同地区的同名标的物）。
    """
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for label, value in pairs:
        key = label.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((key, value))
    return out


def _bars(title: str, pairs: list[tuple[str, float]], *, value_label: str,
          label_label: str, source_tool: str) -> list[dict]:
    """把 (标签, 数值) 组装成图表数据。"""
    pairs = _dedupe_labels(pairs)
    if len(pairs) < 2:
        return []          # 单条数据画不成有意义的对比
    # 只保留数值最大的前 MAX_BARS 条（横条图从大到小更易读）
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    note = ""
    if len(pairs) > MAX_BARS:
        note = f"仅显示前 {MAX_BARS} 项（共 {len(pairs)} 项）"
        pairs = pairs[:MAX_BARS]
    return [{
        "kind": "bar",
        "title": title,
        "value_label": value_label,
        "label_label": label_label,
        "data": [{"label": lb, "value": v} for lb, v in pairs],
        "note": note,
        "source_tool": source_tool,
    }]


# ---------------------------------------------------------------------------
# 业务数据库
# ---------------------------------------------------------------------------

#: 哪些列适合当"数值"。按优先级取第一个命中的。
_PG_NUMERIC_HINTS = (
    "cnt", "count", "total", "amount", "sum", "num", "qty", "quantity",
    "次数", "数量", "金额", "笔数",
)
#: 哪些列适合当"标签"。
_PG_LABEL_HINTS = (
    "name", "subject", "subject_matter", "project", "unit", "supplier",
    "region", "标的物", "采购单位", "供应商", "地区", "项目",
)


def _pick_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """从列名里挑 (标签列, 数值列)。

    ★ 为什么按名字挑而不是"第一列当标签、第二列当数值"
    ------------------------------------------------
    `query_database` 的列由**查询模板**决定，顺序不固定；
    按位置挑会把 `count` 当标签、把标的物名当数值 —— 图就画反了。
    按名字挑至少有语义依据，挑不到就**不画**（宁可没有图，也不要画错的图）。
    """
    label = next(
        (c for hint in _PG_LABEL_HINTS for c in columns if hint in str(c).lower()),
        None,
    )
    value = next(
        (c for hint in _PG_NUMERIC_HINTS for c in columns if hint in str(c).lower()),
        None,
    )
    return label, value


def chart_from_rows(rows: Any, *, title: str = "业务数据库查询结果") -> list[dict]:
    """从 `query_database` 的行数据产图。

    `rows` 是 `list[dict]`（`_fmt_pg` 收的那种）；单个 dict（统计概览）
    也认，但那种更适合 KPI 卡而非条形图 —— 这里只在有标签/数值对时才画。
    """
    try:
        if isinstance(rows, dict):
            pairs: list[tuple[str, float]] = []
            for k in rows:
                v = _num(rows[k])
                if v is not None:
                    pairs.append((str(k), v))
            return (
                _bars(title, pairs, value_label="数值", label_label="项目",
                      source_tool="query_database")
                if len(pairs) >= 2
                else []
            )

        records = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        if len(records) < 2:
            return []
        columns: list[str] = []
        for r in records:
            for k in r:
                if k not in columns:
                    columns.append(k)
        label_col, value_col = _pick_columns(columns)
        if not label_col or not value_col:
            logger.debug("行数据里找不到标签/数值列（列=%s），不产图", columns)
            return []
        pairs: list[tuple[str, float]] = []
        for r in records:
            v = _num(r.get(value_col))
            if v is not None:
                pairs.append((str(r.get(label_col) or "").strip(), v))
        return _bars(
            f"{title}（共 {len(records)} 条）", pairs,
            value_label=str(value_col), label_label=str(label_col),
            source_tool="query_database",
        )
    except Exception:  # noqa: BLE001 - 产图是增强项，绝不能影响工具返回
        logger.debug("从业务库行数据产图失败", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# 知识图谱
# ---------------------------------------------------------------------------

def chart_from_top_traded(rows: Any) -> list[dict]:
    """从 `client.top_traded()` 的返回产图。

    形状实测为 `[{"name": ..., "tradeFrequency": ...}]`。
    """
    try:
        records = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        if len(records) < 2:
            return []
        pairs: list[tuple[str, float]] = []
        for r in records:
            name = str(r.get("name") or "").strip()
            # 兼容几种同义键：图谱客户端的字段名改过
            value = None
            for k in ("tradeFrequency", "trade_frequency", "frequency", "cnt", "count"):
                value = _num(r.get(k))
                if value is not None:
                    break
            if name and value is not None:
                pairs.append((name, value))
        return _bars("交易频次排行", pairs, value_label="交易频次",
                     label_label="标的物", source_tool="search_knowledge_graph")
    except Exception:  # noqa: BLE001
        logger.debug("从 top_traded 产图失败", exc_info=True)
        return []


def chart_from_graph_stats(data: Any) -> list[dict]:
    """从 `client.graph_stats()` 的返回产图（节点/关系计数）。"""
    try:
        if not isinstance(data, dict) or len(data) < 2:
            return []
        pairs: list[tuple[str, float]] = []
        for k in data:
            v = _num(data[k])
            if v is not None:
                pairs.append((str(k), v))
        return _bars("知识图谱规模", pairs, value_label="数量",
                     label_label="类型", source_tool="search_knowledge_graph")
    except Exception:  # noqa: BLE001
        logger.debug("从 graph_stats 产图失败", exc_info=True)
        return []


def cap_charts(charts: list[dict]) -> list[dict]:
    """最多保留 `MAX_CHARTS` 张（与 `constants.MAX_CHARTS` 同口径）。"""
    out = [c for c in charts if isinstance(c, dict) and c.get("data")]
    return out[:MAX_CHARTS]
