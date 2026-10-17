"""知识图谱子图抽取（供前端力导向图渲染）。

为什么单独做一层，而不是复用 `graph_data.run_graph_query`
----------------------------------------------------------
`graph_data` 面向"表格展示"：返回行列表，每行是扁平的记录。
力导向图需要的是**节点 + 边**结构，且必须严格控制规模——
实测 `SubjectMatter(空调)` 有 480 个直接邻居，`Transaction` 节点多达 8730 个，
**原样返回会让浏览器渲染卡死**。

本模块的三个关键设计
--------------------

1. **跳过 Transaction 节点**。图谱是星型结构，`Transaction` 只挂
   `amount`/`id` 两个属性且**没有出边**，画在图上只是 8730 个孤立点。
   它的信息（金额）改为**聚合到边上**（`amount_sum` / `amount_median`），
   图更干净且信息不丢。

2. **节点数硬上限**（`MAX_NODES`，默认 60）。力导向图超过 ~100 个节点就
   基本看不清；且布局计算是 O(n²) 级别的开销。超限时**按度数截断**
   （保留连接最多的节点）并回报 `truncated=True`，而不是静默丢数据。

3. **每个节点带 `degree`**（原始度数，非子图内度数）。前端据此调整半径，
   也便于用户理解"这个实体在原图里有多大"。

边上的金额用**中位数**而非均值：实测该数据集金额跨 6 个数量级
（办公用品几千元 ↔ 基建工程数亿元），均值会被超大额记录带偏。
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase

from src.logging_config import get_logger

logger = get_logger(__name__)

# 节点数上限：力导向图超过 ~100 个节点基本不可读，且布局计算开销上升
MAX_NODES = 60
DEFAULT_NODES = 40

# 每个节点最多展开的邻居数
MAX_NEIGHBORS_PER_NODE = 25

# 允许出现的节点标签（白名单）。`Transaction` 刻意排除，理由见模块文档。
ALLOWED_LABELS = ("SubjectMatter", "Purchaser", "Supplier", "Agency", "Location")

# 标签中文名（前端图例用）
LABEL_CN = {
    "SubjectMatter": "标的物",
    "Purchaser": "采购单位",
    "Supplier": "供应商",
    "Agency": "代理机构",
    "Location": "地区",
    "Transaction": "交易",
}

# 关系中文名
RELATION_CN = {
    "PURCHASED_BY": "被采购于",
    "SUPPLIED_BY": "由…供应",
    "LOCATED_AT": "位于",
    "AGENT_BY": "代理机构",
    "HAS_TRANSACTION": "包含交易",
}


class SubgraphError(RuntimeError):
    """子图抽取失败。"""


class SubgraphUnavailable(SubgraphError):
    """Neo4j 不可用（未启动 / 配置错误）。"""


def _driver():
    uri = (os.getenv("NEO4J_URI") or "").strip()
    if not uri:
        raise SubgraphUnavailable("未配置 NEO4J_URI")
    try:
        d = GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=15,
        )
        d.verify_connectivity()
        return d
    except Exception as exc:
        raise SubgraphUnavailable(f"Neo4j 连接失败：{type(exc).__name__}") from exc


def _db_name() -> str:
    return (os.getenv("NEO4J_DATABASE") or "neo4j").strip()


def _clamp(n: int, lo: int, hi: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = DEFAULT_NODES
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 子图查询
# ---------------------------------------------------------------------------

# 说明：Cypher 不允许在 MATCH 里参数化标签，故标签用白名单校验后拼接。
# 关系方向统一按"从中心节点出发"取，图里方向不敏感（前端画无向边）。
_NEIGHBOR_QUERY = """
MATCH (center:{label} {{name: $name}})
OPTIONAL MATCH (center)-[r]-(nb)
WHERE NOT nb:Transaction
WITH center, r, nb,
     // 金额聚合：交易关系下没有直接金额，改从 Transaction 汇总
     CASE WHEN nb IS NULL THEN 0 ELSE 1 END AS has_nb
RETURN
    coalesce(center.name, '')       AS center_name,
    labels(center)[0]               AS center_label,
    CASE WHEN nb IS NULL THEN NULL ELSE coalesce(nb.name, '') END AS nb_name,
    CASE WHEN nb IS NULL THEN NULL ELSE labels(nb)[0] END         AS nb_label,
    CASE WHEN r IS NULL THEN NULL ELSE type(r) END                AS rel_type,
    has_nb
LIMIT $limit
"""


def _collect_neighbors(
    driver, label: str, name: str, limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """取某个实体的直接邻居。返回 `(中心节点, 邻居列表)`。"""
    query = _NEIGHBOR_QUERY.format(label=label)
    with driver.session(database=_db_name()) as s:
        recs = list(s.run(query, name=name, limit=limit))

    if not recs:
        raise SubgraphError(f"未找到 {LABEL_CN.get(label, label)}「{name}」")

    first = recs[0]
    center = {
        "id": f"{first['center_label']}:{first['center_name']}",
        "label": first["center_label"],
        "label_cn": LABEL_CN.get(first["center_label"], first["center_label"]),
        "name": first["center_name"],
        "is_center": True,
    }

    neighbors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in recs:
        if not r["nb_name"]:
            continue
        nid = f"{r['nb_label']}:{r['nb_name']}"
        if nid in seen:
            continue
        seen.add(nid)
        neighbors.append(
            {
                "id": nid,
                "label": r["nb_label"],
                "label_cn": LABEL_CN.get(r["nb_label"], r["nb_label"]),
                "name": r["nb_name"],
                "rel_type": r["rel_type"],
                "rel_cn": RELATION_CN.get(r["rel_type"], r["rel_type"]),
            }
        )
    return center, neighbors


def _center_stats(driver, center_label: str, center_name: str) -> dict[str, Any]:
    """取中心实体的交易金额统计（笔数 / 合计 / 中位数）。

    ★ 为什么只做"中心节点"的统计，不做"每条边"的统计
    -----------------------------------------------
    第一版尝试把金额聚合到每条边上，结果是**40 条边金额完全相同**——
    因为图谱是**星型结构**：

        (SubjectMatter)-[:HAS_TRANSACTION]->(Transaction {amount})
        (SubjectMatter)-[:PURCHASED_BY]->(Purchaser)
        (SubjectMatter)-[:SUPPLIED_BY]->(Supplier)

    `Transaction` 只与 `SubjectMatter` 相连，**没有"采购人 → 交易"的边**，
    所以"某采购人在某标的物上花了多少钱"这个信息**在图谱里不存在**，
    只能回 PostgreSQL 按 (subject_matter, purchaser) 分组才能算出来。
    强行在图上算，每个邻居都会得到中心节点的总额，是**看起来精确但完全错误**的数据。

    故这里只返回中心节点的聚合统计，并在响应里注明"边不含金额、
    需要分维度金额请用集采分析接口"。宁可少给信息，也不给错的信息。
    """
    if center_label != "SubjectMatter":
        return {}
    try:
        with driver.session(database=_db_name()) as s:
            rec = s.run(
                """
                MATCH (sm:SubjectMatter {name: $name})-[:HAS_TRANSACTION]->(t:Transaction)
                WITH count(t) AS txn, sum(coalesce(t.amount, 0)) AS total,
                     collect(coalesce(t.amount, 0)) AS amounts
                RETURN txn, total, amounts
                """,
                name=center_name,
            ).single()
            if not rec:
                return {}
            amounts = sorted(a for a in (rec["amounts"] or []) if a)
            median = amounts[len(amounts) // 2] if amounts else None
            return {
                "txn_count": rec["txn"],
                "amount_sum": float(rec["total"]) if rec["total"] is not None else None,
                # 中位数而非均值：实测金额跨 6 个数量级，均值会被超大额记录带偏
                "amount_median": float(median) if median is not None else None,
            }
    except Exception as exc:  # 统计失败不影响图本身
        logger.warning("中心节点金额统计失败（不影响图形）：%s", exc)
        return {}


def build_subgraph(
    label: str,
    name: str,
    max_nodes: int = DEFAULT_NODES,
    include_stats: bool = True,
) -> dict[str, Any]:
    """以某实体为中心抽子图。

    返回：
        {
          "center": {...},
          "nodes": [{id, label, label_cn, name, degree, is_center}...],
          "edges": [{source, target, rel_type, rel_cn, txn_count, amount_sum, amount_median}...],
          "truncated": bool,
          "total_neighbors": int,
          "legend": {...}   # 前端图例
        }

    超限时按**度数**截断（保留连接最多的节点）并设 `truncated=True`——
    静默丢弃会让用户以为图就是全部。
    """
    if label not in ALLOWED_LABELS:
        raise SubgraphError(
            f"不支持的节点类型 {label!r}，可用：{'、'.join(ALLOWED_LABELS)}"
        )
    clean_name = (name or "").strip()
    if not clean_name:
        raise SubgraphError("请提供实体名称")

    limit = _clamp(max_nodes, 5, MAX_NODES) * 4  # 多取一些再去重截断
    driver = _driver()
    try:
        center, neighbors = _collect_neighbors(driver, label, clean_name, limit)

        # 取每个邻居的度数，用于排序与前端半径
        neighbor_names = [n["name"] for n in neighbors]
        degrees: dict[str, int] = {}
        if neighbor_names:
            with driver.session(database=_db_name()) as s:
                for r in s.run(
                    """
                    MATCH (n)-[r]-()
                    WHERE coalesce(n.name,'') IN $names
                    RETURN coalesce(n.name,'') AS name, count(r) AS deg
                    """,
                    names=neighbor_names,
                ):
                    degrees[r["name"]] = r["deg"]

        for n in neighbors:
            n["degree"] = degrees.get(n["name"], 1)

        total_neighbors = len(neighbors)
        # 按度数降序截断
        neighbors.sort(key=lambda x: x["degree"], reverse=True)
        keep = neighbors[: _clamp(max_nodes, 5, MAX_NODES)]
        truncated = total_neighbors > len(keep)

        stats = _center_stats(driver, center["label"], center["name"]) if include_stats else {}

        center_degree = 0
        with driver.session(database=_db_name()) as s:
            rec = s.run(
                f"MATCH (c:{label} {{name: $name}})-[r]-() RETURN count(r) AS deg",
                name=clean_name,
            ).single()
            if rec:
                center_degree = rec["deg"]
        center["degree"] = center_degree
        if stats:
            center.update(stats)

        nodes = [center] + keep
        # 边只描述"关系"，不带金额——图里确实没有 per-relationship 金额，
        # 硬算会得出"每条边都等于中心节点总额"的错误结果（见 _center_stats 说明）
        edges = [
            {
                "source": center["id"],
                "target": n["id"],
                "rel_type": n["rel_type"],
                "rel_cn": n["rel_cn"],
            }
            for n in keep
        ]

        return {
            "center": center,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "total_neighbors": total_neighbors,
            "shown_nodes": len(nodes),
            "max_nodes": _clamp(max_nodes, 5, MAX_NODES),
            "legend": {
                "node_labels": [{"key": k, "cn": LABEL_CN[k]} for k in ALLOWED_LABELS],
                "notes": [
                    "Transaction（交易）节点未画入图中——实测有 8730 个，只挂金额属性、"
                    "无出边，画出来只是孤立点。中心节点的交易统计已单独给出",
                    "边不含金额：图谱中 Transaction 只与标的物相连，"
                    "没有「采购人→交易」的边，故无法算出每条边对应的金额。"
                    "需要按供应商/采购人细分金额，请用「集采分析」页",
                ],
            },
        }
    finally:
        driver.close()


def graph_nodes_overview() -> dict[str, Any]:
    """图谱整体规模 + 可探索的枢纽实体（供前端"推荐探索"列表）。"""
    driver = _driver()
    try:
        with driver.session(database=_db_name()) as s:
            by_label = {}
            for r in s.run(
                "MATCH (n) WHERE NOT n:Transaction "
                "RETURN labels(n)[0] AS lbl, count(*) AS c ORDER BY c DESC"
            ):
                by_label[r["lbl"]] = r["c"]

            # 度数最高的实体（排除 Transaction）——这些最适合作为探索入口
            hubs = []
            for r in s.run(
                """
                MATCH (n)-[r]-()
                WHERE NOT n:Transaction AND coalesce(n.name,'') <> ''
                RETURN labels(n)[0] AS lbl, n.name AS name, count(r) AS deg
                ORDER BY deg DESC LIMIT 15
                """
            ):
                hubs.append(
                    {
                        "label": r["lbl"],
                        "label_cn": LABEL_CN.get(r["lbl"], r["lbl"]),
                        "name": r["name"],
                        "degree": r["deg"],
                    }
                )

            total_nodes = sum(by_label.values())
            rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        return {
            "total_nodes": total_nodes,
            "total_relationships": rels,
            "by_label": [
                {"key": k, "cn": LABEL_CN.get(k, k), "count": v}
                for k, v in sorted(by_label.items(), key=lambda x: -x[1])
            ],
            "hub_entities": hubs,
            "allowed_labels": [
                {"key": k, "cn": LABEL_CN[k]} for k in ALLOWED_LABELS
            ],
        }
    finally:
        driver.close()


def search_entities(keyword: str, limit: int = 15) -> list[dict[str, Any]]:
    """按关键词搜索实体（供前端自动补全）。"""
    kw = (keyword or "").strip()
    if not kw:
        return []
    n = _clamp(limit, 1, 50)
    driver = _driver()
    try:
        out = []
        with driver.session(database=_db_name()) as s:
            for r in s.run(
                """
                MATCH (n)
                WHERE NOT n:Transaction
                  AND coalesce(n.name,'') CONTAINS $kw
                OPTIONAL MATCH (n)-[r]-()
                RETURN labels(n)[0] AS lbl, n.name AS name, count(r) AS deg
                ORDER BY deg DESC LIMIT $limit
                """,
                kw=kw,
                limit=n,
            ):
                out.append(
                    {
                        "label": r["lbl"],
                        "label_cn": LABEL_CN.get(r["lbl"], r["lbl"]),
                        "name": r["name"],
                        "degree": r["deg"],
                    }
                )
        return out
    finally:
        driver.close()


__all__ = [
    "ALLOWED_LABELS",
    "LABEL_CN",
    "MAX_NODES",
    "SubgraphError",
    "SubgraphUnavailable",
    "build_subgraph",
    "graph_nodes_overview",
    "search_entities",
]
