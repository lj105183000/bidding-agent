"""轻量数据质量扫描。

设计原则
--------
- **单条规则失败不拖垮整次扫描**：规则抛异常 → 该条记为 `skip`，其余照跑。
  质量报告的价值在于“哪里有问题”，而不是“第一条规则炸了就什么都没有”。
- **不重复实现判据**：可疑金额复用 `datahub.checks.suspicious_sql`；
  空值率复用 `datahub.checks.empty_rates`。与 `analysis.data_quality()` 同源。
- **扫描只读**：本模块不写业务表，只追加 `data_quality_reports`。
- **默认持久化**：`run_quality_scan(persist=True)` 写 PG；命令行
  `python -m src.datahub.quality` 不持久化，只打印 JSON 供人工核对。

规则状态：
    pass   通过
    warn   可接受但需要注意
    fail   必须处理
    skip   本次无法检查（依赖不可用/表不存在）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import PROJECT_ROOT, settings
from src.constants.regulations import all_articles
from src.datahub import neo4j as neo4j_adapter
from src.datahub import qdrant as qdrant_adapter
from src.datahub.checks import suspicious_sql
from src.datahub.contract import DataHubError
from src.datahub.pg import query_dicts
from src.datahub.registry import ASSETS
from src.database.neo4j_client import ALLOWED_LABELS
from src.rag.regulation_store import REGULATION_COLLECTION

logger = logging.getLogger(__name__)

TABLE = "bidding_records"
_STATUSES = ("pass", "warn", "fail", "skip")


# ---------------------------------------------------------------------------
# 上下文：把三类外部依赖收成可注入的函数，规则本身不关心连接细节
# ---------------------------------------------------------------------------


class QualityContext:
    """质量规则可用的只读访问函数。

    默认实现走 `src/datahub` 适配器；测试通过注入 fake 函数绕开真实服务。
    """

    def __init__(
        self,
        *,
        pg_rows: Callable[..., list[dict[str, Any]]] | None = None,
        neo4j_rows: Callable[..., list[dict[str, Any]]] | None = None,
        qdrant_count: Callable[[str], int] | None = None,
        qdrant_exists: Callable[[str], bool] | None = None,
    ) -> None:
        self._pg_rows = pg_rows or (lambda sql, params=(): query_dicts(sql, params))
        self._neo4j_rows = neo4j_rows or (
            lambda cypher, **params: neo4j_adapter.run(
                cypher, raise_on_error=True, error_cls=DataHubError, **params
            )
        )
        self._qdrant_count = qdrant_count or qdrant_adapter.count
        self._qdrant_exists = qdrant_exists or qdrant_adapter.collection_exists

    def pg_rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return self._pg_rows(sql, params)

    def neo4j_rows(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return self._neo4j_rows(cypher, **params)

    def qdrant_count(self, collection: str) -> int:
        return int(self._qdrant_count(collection))

    def qdrant_exists(self, collection: str) -> bool:
        return bool(self._qdrant_exists(collection))


@dataclass(frozen=True)
class QualityRule:
    name: str
    asset: str
    severity: str  # warn / fail（该规则可能达到的最高级别）
    description: str
    check: Callable[[QualityContext], dict[str, Any]]


def _result(
    status: str,
    value: Any = None,
    threshold: Any = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "threshold": threshold,
        "message": message,
    }


# ---------------------------------------------------------------------------
# bidding_records
# ---------------------------------------------------------------------------


def _check_row_count(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(f"SELECT count(*) AS n FROM {TABLE}")
    total = int(rows[0]["n"]) if rows else 0
    if total == 0:
        return _result("fail", total, 1, "bidding_records 没有数据，请先执行 PostgreSQL 导入")
    return _result("pass", total, 1, f"共 {total} 行")


def _field_empty_rates(ctx: QualityContext, fields: tuple[str, ...]) -> tuple[int, dict[str, float]]:
    expressions = ", ".join(
        f"count(*) FILTER (WHERE {field} IS NULL OR {field}::text = '') AS {field}"
        for field in fields
    )
    rows = ctx.pg_rows(
        f"SELECT count(*) AS total, {expressions} FROM {TABLE}"
    )
    row = rows[0] if rows else {}
    total = int(row.get("total") or 0)
    if total == 0:
        return 0, {field: 0.0 for field in fields}
    return total, {
        field: round(int(row.get(field) or 0) / total * 100, 2) for field in fields
    }


def _check_required_empty_rate(ctx: QualityContext) -> dict[str, Any]:
    fields = ("project_id", "project_name", "purchaser", "supplier", "amount")
    total, rates = _field_empty_rates(ctx, fields)
    if total == 0:
        return _result("skip", rates, None, "无数据，跳过必填字段空值率检查")
    worst = max(rates.values())
    if worst > 20:
        status, threshold = "fail", 20
    elif worst > 5:
        status, threshold = "warn", 5
    else:
        status, threshold = "pass", 5
    return _result(status, rates, threshold, f"必填字段最高空值率 {worst}%")


def _check_optional_sparse_rate(ctx: QualityContext) -> dict[str, Any]:
    fields = ("budget", "duration", "agency", "project_address")
    total, rates = _field_empty_rates(ctx, fields)
    if total == 0:
        return _result("skip", rates, None, "无数据，跳过稀疏字段检查")
    # 这几个字段的稀疏是**已知且可接受**的（budget 空值率约 99.5%），只记录不判失败。
    return _result("pass", rates, None, "稀疏字段空值率（仅记录，不判定失败）")


def _check_duplicate_project_amount(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        f"""
        SELECT count(*) AS groups FROM (
            SELECT project_id, amount FROM {TABLE}
            WHERE project_id IS NOT NULL AND amount IS NOT NULL
            GROUP BY project_id, amount HAVING count(*) > 1
        ) duplicated
        """
    )
    groups = int(rows[0]["groups"]) if rows else 0
    status = "warn" if groups > 0 else "pass"
    return _result(status, groups, 0, f"{groups} 组 (project_id, amount) 重复")


def _check_invalid_amount(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        f"SELECT count(*) AS n FROM {TABLE} WHERE amount IS NOT NULL AND amount <= 0"
    )
    count = int(rows[0]["n"]) if rows else 0
    status = "fail" if count > 0 else "pass"
    return _result(status, count, 0, f"{count} 条金额 <= 0")


def _check_suspicious_amount(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        f"SELECT count(*) AS n FROM {TABLE} WHERE {suspicious_sql()}"
    )
    count = int(rows[0]["n"]) if rows else 0
    status = "warn" if count > 0 else "pass"
    return _result(status, count, 0, f"{count} 条命中金额可疑规则")


def _check_time_freshness(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        f"""
        SELECT max(bid_time) AS max_bid_time,
               count(DISTINCT to_char(bid_time, 'YYYY-MM')) AS months
        FROM {TABLE}
        """
    )
    row = rows[0] if rows else {}
    raw = row.get("max_bid_time")
    months = int(row.get("months") or 0)
    if not raw:
        return _result("fail", None, None, "bid_time 全为空，无法判断时效性")

    try:
        latest = datetime.fromisoformat(str(raw))
    except ValueError:
        return _result("skip", raw, None, "bid_time 无法解析")
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - latest).days
    if age_days > 365:
        return _result("warn", age_days, 365, f"最新中标时间距今 {age_days} 天")
    if months <= 1:
        return _result("warn", months, 1, "数据只覆盖 1 个月，时间趋势不可用")
    return _result("pass", {"age_days": age_days, "months": months}, None, "时效性正常")


# ---------------------------------------------------------------------------
# 会话 / 反馈
# ---------------------------------------------------------------------------


def _check_feedback_orphans(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        """
        SELECT count(*) AS n FROM feedback f
        LEFT JOIN conversations c ON f.conversation_id = c.id
        WHERE f.conversation_id IS NOT NULL AND c.id IS NULL
        """
    )
    count = int(rows[0]["n"]) if rows else 0
    status = "warn" if count > 0 else "pass"
    return _result(status, count, 0, f"{count} 条反馈找不到对应会话")


def _check_session_traces_stale(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.pg_rows(
        "SELECT count(*) AS n FROM session_traces "
        "WHERE updated_at < now() - interval '24 hours'"
    )
    count = int(rows[0]["n"]) if rows else 0
    status = "warn" if count > 0 else "pass"
    return _result(status, count, 0, f"{count} 条 trace 映射超过 24 小时未更新")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def _vocab_n_docs() -> int | None:
    path = Path(PROJECT_ROOT) / "data" / "vocab.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("n_docs")
        return int(value) if value is not None else None
    except Exception:  # noqa: BLE001 - 词表缺失/损坏时只跳过一致性比对
        return None


def _check_bid_qa_points(ctx: QualityContext) -> dict[str, Any]:
    collection = settings.qdrant_collection
    points = ctx.qdrant_count(collection)
    if points == 0:
        return _result("fail", points, 1, f"问答集合 {collection} 为空")
    expected = _vocab_n_docs()
    if expected is not None and expected != points:
        return _result(
            "warn",
            {"points": points, "vocab_n_docs": expected},
            expected,
            "问答集合点数与 data/vocab.json 的 n_docs 不一致（换数据后需重新 ingest）",
        )
    return _result("pass", points, None, f"问答集合 {collection} 共 {points} 点")


def _check_bid_regulations_points(ctx: QualityContext) -> dict[str, Any]:
    points = ctx.qdrant_count(REGULATION_COLLECTION)
    if points == 0:
        return _result("fail", points, 1, f"法规集合 {REGULATION_COLLECTION} 为空")
    corpus = len(all_articles())
    if corpus and points != corpus:
        return _result(
            "warn",
            {"points": points, "corpus_articles": corpus},
            corpus,
            "法规库点数与内置语料不一致，需重新执行法规增量入库",
        )
    return _result("pass", points, None, f"法规集合共 {points} 点")


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


def _check_graph_node_counts(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.neo4j_rows(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"
    )
    counts = {str(row.get("label")): int(row.get("count") or 0) for row in rows}
    total = sum(counts.values())
    if total == 0:
        return _result("fail", counts, 1, "知识图谱为空")
    missing = [
        label
        for label in ("SubjectMatter", "Purchaser", "Supplier")
        if counts.get(label, 0) == 0
    ]
    if missing:
        return _result("warn", counts, None, f"图谱缺少关键标签：{'、'.join(missing)}")
    return _result("pass", counts, None, f"图谱共 {total} 个节点")


def _check_graph_orphans(ctx: QualityContext) -> dict[str, Any]:
    rows = ctx.neo4j_rows(
        "MATCH (n) WHERE NOT (n)--() RETURN count(*) AS n"
    )
    count = int(rows[0]["n"]) if rows else 0
    status = "warn" if count > 0 else "pass"
    return _result(status, count, 0, f"{count} 个孤立节点（无任何关系）")


# ---------------------------------------------------------------------------
# 目录漂移
# ---------------------------------------------------------------------------


def _check_catalog_drift(ctx: QualityContext) -> dict[str, Any]:
    expected_pg = {
        asset.name.split(".", 1)[1]
        for asset in ASSETS
        if asset.kind == "postgres_table" and asset.status == "active"
    }
    try:
        rows = ctx.pg_rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    except Exception as exc:  # noqa: BLE001 - 规则层记 skip，不拖垮扫描
        return _result("skip", None, None, f"无法读取 PostgreSQL 目录：{type(exc).__name__}")
    actual_pg = {str(row.get("table_name")) for row in rows}
    missing_pg = sorted(expected_pg - actual_pg)

    expected_collections = {
        "qdrant.bid_qa": settings.qdrant_collection,
        "qdrant.bid_regulations": REGULATION_COLLECTION,
    }
    missing_collections: list[str] = []
    for asset_name, collection in expected_collections.items():
        try:
            if not ctx.qdrant_exists(collection):
                missing_collections.append(asset_name)
        except Exception as exc:  # noqa: BLE001
            return _result(
                "skip",
                None,
                None,
                f"无法连接 Qdrant：{type(exc).__name__}",
            )

    missing_labels: list[str] = []
    try:
        rows = ctx.neo4j_rows("CALL db.labels() YIELD label RETURN label")
        actual_labels = {str(row.get("label")) for row in rows}
        missing_labels = sorted(ALLOWED_LABELS - actual_labels)
    except Exception:  # noqa: BLE001 - 图谱不可用时不让目录规则整体失败
        missing_labels = []

    value = {
        "missing_pg_tables": missing_pg,
        "missing_collections": missing_collections,
        "missing_neo4j_labels": missing_labels,
    }
    if missing_pg or missing_collections:
        return _result("fail", value, None, "注册表声明的资产在实库中缺失")
    if missing_labels:
        return _result("warn", value, None, "图谱缺少注册表中的标签")
    return _result("pass", value, None, "注册表与实库一致")


# ---------------------------------------------------------------------------
# 规则注册表
# ---------------------------------------------------------------------------

RULES: tuple[QualityRule, ...] = (
    QualityRule(
        "bidding_records.row_count",
        "pg.bidding_records",
        "fail",
        "业务表非空",
        _check_row_count,
    ),
    QualityRule(
        "bidding_records.required_empty_rate",
        "pg.bidding_records",
        "fail",
        "必填字段空值率（>5% warn，>20% fail）",
        _check_required_empty_rate,
    ),
    QualityRule(
        "bidding_records.optional_sparse_rate",
        "pg.bidding_records",
        "warn",
        "稀疏字段空值率（仅记录）",
        _check_optional_sparse_rate,
    ),
    QualityRule(
        "bidding_records.duplicate_project_amount",
        "pg.bidding_records",
        "warn",
        "(project_id, amount) 重复组",
        _check_duplicate_project_amount,
    ),
    QualityRule(
        "bidding_records.invalid_amount",
        "pg.bidding_records",
        "fail",
        "金额 <= 0",
        _check_invalid_amount,
    ),
    QualityRule(
        "bidding_records.suspicious_amount",
        "pg.bidding_records",
        "warn",
        "命中金额可疑规则",
        _check_suspicious_amount,
    ),
    QualityRule(
        "bidding_records.time_freshness",
        "pg.bidding_records",
        "warn",
        "中标时间时效性与覆盖月数",
        _check_time_freshness,
    ),
    QualityRule(
        "feedback.orphans",
        "pg.feedback",
        "warn",
        "反馈找不到对应会话",
        _check_feedback_orphans,
    ),
    QualityRule(
        "session_traces.stale",
        "pg.session_traces",
        "warn",
        "trace 映射超过 24 小时未更新",
        _check_session_traces_stale,
    ),
    QualityRule(
        "bid_qa.points",
        "qdrant.bid_qa",
        "fail",
        "问答集合非空且与 vocab 一致",
        _check_bid_qa_points,
    ),
    QualityRule(
        "bid_regulations.points",
        "qdrant.bid_regulations",
        "fail",
        "法规集合非空且与内置语料一致",
        _check_bid_regulations_points,
    ),
    QualityRule(
        "graph.node_counts",
        "neo4j.graph",
        "fail",
        "图谱节点计数",
        _check_graph_node_counts,
    ),
    QualityRule(
        "graph.orphan_nodes",
        "neo4j.graph",
        "warn",
        "孤立节点",
        _check_graph_orphans,
    ),
    QualityRule(
        "catalog.drift",
        "catalog",
        "fail",
        "注册表与实库一致性",
        _check_catalog_drift,
    ),
)


# ---------------------------------------------------------------------------
# 扫描入口
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_quality_scan(
    source: str = "manual",
    persist: bool = True,
    *,
    context: QualityContext | None = None,
) -> dict[str, Any]:
    """执行全部规则并返回报告。单条规则异常只记 skip，**本函数永不抛**。"""
    started = time.perf_counter()
    ctx = context or QualityContext()
    checks: list[dict[str, Any]] = []

    for rule in RULES:
        rule_started = time.perf_counter()
        try:
            outcome = rule.check(ctx) or {}
            status = str(outcome.get("status") or "pass")
            if status not in _STATUSES:
                status = "skip"
            checks.append(
                {
                    "name": rule.name,
                    "asset": rule.asset,
                    "severity": rule.severity,
                    "description": rule.description,
                    "status": status,
                    "value": outcome.get("value"),
                    "threshold": outcome.get("threshold"),
                    "message": outcome.get("message", ""),
                    "elapsed_ms": max(
                        0, int((time.perf_counter() - rule_started) * 1000)
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - 单条规则失败不拖垮扫描
            logger.warning("质量规则 %s 执行失败", rule.name, exc_info=True)
            checks.append(
                {
                    "name": rule.name,
                    "asset": rule.asset,
                    "severity": rule.severity,
                    "description": rule.description,
                    "status": "skip",
                    "value": None,
                    "threshold": None,
                    "message": f"检查失败：{type(exc).__name__}",
                    "elapsed_ms": max(
                        0, int((time.perf_counter() - rule_started) * 1000)
                    ),
                }
            )

    summary = {
        status: sum(1 for check in checks if check["status"] == status)
        for status in _STATUSES
    }
    overall = "fail" if summary["fail"] else (
        "degraded" if summary["warn"] or summary["skip"] else "ok"
    )
    report: dict[str, Any] = {
        "overall": overall,
        "generated_at": _now_iso(),
        "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "source": source,
        "summary": summary,
        "checks": checks,
    }

    if persist:
        try:
            from src.datahub.quality_store import save_report

            report["report_id"] = save_report(report, source=source)
        except Exception:  # noqa: BLE001 - 持久化失败不影响本次结果
            logger.exception("质量报告持久化失败（本次扫描结果仍返回）")

    return report


def latest_report() -> dict[str, Any] | None:
    """最近一份已持久化的报告；表不存在/无数据返回 None。"""
    from src.datahub.quality_store import latest_report as _latest

    return _latest()


if __name__ == "__main__":  # pragma: no cover - 手工核对入口
    print(
        json.dumps(
            run_quality_scan(source="cli", persist=False),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
