"""静态数据血缘。

为什么是静态的
--------------
本项目的数据流转几乎全部由**脚本步骤**决定（batch 五步、两个导入脚本、ingest），
不是运行时动态生成的任务图。把这些边写进代码有三个好处：

1. 目录端点不依赖数据库，三库全挂也能看血缘；
2. 边是“人确认过的”，不会被运行时的一次临时查询污染；
3. 测试可以枚举校验：每条边的两端都必须是注册表里的资产。

不足：运行时的 `save=true`、会话写入、反馈写入只能按“入口 → 表”的粗粒度描述，
做不到字段级血缘。这是轻量方案的边界，已在 `docs/开发文档.md` 写清。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.datahub.registry import ASSET_INDEX, find_asset


@dataclass(frozen=True)
class LineageEdge:
    """一条血缘边：`source -> target`，`via` 说明由谁完成。"""

    source: str
    target: str
    via: str
    kind: str  # batch / importer / runtime / scheduler / code
    schedulable: bool = False


def _e(
    source: str,
    target: str,
    via: str,
    kind: str,
    *,
    schedulable: bool = False,
) -> LineageEdge:
    return LineageEdge(
        source=source,
        target=target,
        via=via,
        kind=kind,
        schedulable=schedulable,
    )


EDGES: tuple[LineageEdge, ...] = (
    # ---- batch 五步管道：原始 xlsx → 标的物 xlsx / 图谱 CSV ----
    _e("file.data.raw.training_xlsx", "file.data.processed.dedup_xlsx", "batch/去重.py", "batch"),
    _e(
        "file.data.processed.dedup_xlsx",
        "file.data.batch.requests_jsonl",
        "batch/提数据_编JSONL.py",
        "batch",
    ),
    _e(
        "file.data.batch.requests_jsonl",
        "file.data.batch.results_raw_jsonl",
        "batch/提交batch任务.py",
        "batch",
    ),
    _e(
        "file.data.batch.results_raw_jsonl",
        "file.data.batch.results_processed_jsonl",
        "batch/按custom_id排序.py",
        "batch",
    ),
    _e(
        "file.data.batch.results_processed_jsonl",
        "file.data.processed.subjects_xlsx",
        "batch/提数据_插标的物.py",
        "batch",
    ),
    _e(
        "file.data.processed.subjects_xlsx",
        "file.data.processed.frequency_xlsx",
        "batch/统计数据_插交易频次.py",
        "batch",
    ),
    _e(
        "file.data.processed.subjects_xlsx",
        "file.data.processed.graph_frequency_csv",
        "batch/准备图谱CSV.py",
        "batch",
    ),
    _e(
        "file.data.processed.subjects_xlsx",
        "file.data.processed.graph_transactions_csv",
        "batch/准备图谱CSV.py",
        "batch",
    ),
    # ---- 图谱导入 ----
    _e(
        "file.data.processed.graph_frequency_csv",
        "neo4j.graph",
        "src/database/to_neo4j.py",
        "importer",
    ),
    _e(
        "file.data.processed.graph_transactions_csv",
        "neo4j.graph",
        "src/database/to_neo4j.py",
        "importer",
    ),
    _e("neo4j.graph", "neo4j.SubjectMatter", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.graph", "neo4j.Transaction", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.graph", "neo4j.Purchaser", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.graph", "neo4j.Supplier", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.graph", "neo4j.Agency", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.graph", "neo4j.Location", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.SubjectMatter", "neo4j.HAS_TRANSACTION", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.SubjectMatter", "neo4j.PURCHASED_BY", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.SubjectMatter", "neo4j.AGENT_BY", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.SubjectMatter", "neo4j.SUPPLIED_BY", "src/database/to_neo4j.py", "importer"),
    _e("neo4j.SubjectMatter", "neo4j.LOCATED_AT", "src/database/to_neo4j.py", "importer"),
    _e(
        "neo4j.graph",
        "tools.search_knowledge_graph",
        "src/database/neo4j_client.py",
        "code",
    ),
    _e("neo4j.graph", "service.competitor", "src/services/competitor.py", "code"),
    # ---- PostgreSQL 导入 ----
    _e(
        "file.data.processed.subjects_xlsx",
        "pg.bidding_records",
        "src/database/to_postgresql.py",
        "importer",
    ),
    _e("pg.bidding_records", "service.analysis", "src/services/analysis.py", "code"),
    _e("pg.bidding_records", "service.competitor", "src/services/competitor.py", "code"),
    _e("pg.bidding_records", "service.material", "src/services/material.py", "code"),
    _e("pg.bidding_records", "tools.query_database", "src/tools/rag_tools.py", "code"),
    _e("service.analysis", "runtime.api.analysis", "api/server.py", "code"),
    _e("service.competitor", "runtime.api.competitor", "api/server.py", "code"),
    _e("service.material", "runtime.api.material", "api/server.py", "code"),
    # ---- 问答知识库 ----
    _e("file.data.qa_xlsx", "qdrant.bid_qa", "src/rag/ingest.py", "importer"),
    _e("file.data.qa_xlsx", "file.data.vocab_json", "src/rag/ingest.py", "importer"),
    _e("qdrant.bid_qa", "rag.pipeline", "src/rag/pipeline.py", "code"),
    _e("file.data.vocab_json", "rag.pipeline", "src/rag/pipeline.py", "code"),
    _e("rag.pipeline", "tools.search_knowledge_base", "src/tools/rag_tools.py", "code"),
    _e("rag.pipeline", "runtime.api.chat", "api/server.py", "code"),
    # ---- 法规知识库 ----
    _e(
        "code.src.constants.regulations",
        "qdrant.bid_regulations",
        "scripts/ingest_regulations.py",
        "importer",
    ),
    _e(
        "scheduler.regulations_incremental",
        "qdrant.bid_regulations",
        "src/rag/regulation_store.py",
        "scheduler",
        schedulable=True,
    ),
    _e("qdrant.bid_regulations", "service.compliance", "src/services/compliance.py", "code"),
    _e(
        "service.compliance",
        "tools.check_tender_compliance",
        "src/tools/compliance_tools.py",
        "code",
    ),
    _e("service.compliance", "runtime.api.compliance", "api/server.py", "code"),
    # ---- 运行时写入 ----
    _e("runtime.api.extract", "pg.bidding_records", "src/services/extract.py", "runtime"),
    _e("runtime.api.conversations", "pg.conversations", "src/database/postgresql_client.py", "runtime"),
    _e("runtime.api.conversations", "pg.messages", "src/database/postgresql_client.py", "runtime"),
    _e("runtime.api.feedback", "pg.feedback", "src/database/postgresql_client.py", "runtime"),
    _e("runtime.api.tender_save", "pg.tender_documents", "src/services/tender_store.py", "runtime"),
    _e(
        "runtime.api.qualification_profiles",
        "pg.company_profiles",
        "src/services/qualification.py",
        "runtime",
    ),
    _e("runtime.agent.trace", "pg.session_traces", "src/langfuse_feedback.py", "runtime"),
    # ---- 调度任务 ----
    _e(
        "scheduler.quality_scan",
        "pg.data_quality_reports",
        "src/datahub/quality.py",
        "scheduler",
        schedulable=True,
    ),
    _e(
        "pg.data_quality_reports",
        "runtime.api.data_quality",
        "api/server.py",
        "code",
    ),
    _e(
        "scheduler.catalog_snapshot",
        "pg.data_catalog_snapshots",
        "src/datahub/catalog_store.py",
        "scheduler",
        schedulable=True,
    ),
)


def _edge_key(edge: LineageEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.via)


def lineage_subgraph(
    asset: str | None = None,
    *,
    direction: str = "both",
    depth: int = 2,
) -> dict[str, object]:
    """返回血缘子图。

    `asset=None` 返回整张静态图（资产目录页的“全局血缘”视图）；
    指定资产时从它出发做 BFS，`direction` 控制只走上游/下游/双向。
    """
    depth = max(1, min(5, int(depth)))
    direction = (direction or "both").strip().lower()
    if direction not in ("both", "upstream", "downstream"):
        raise ValueError("direction 只能是 both / upstream / downstream")

    selected: list[LineageEdge] = []
    if asset is None:
        selected = list(EDGES)
        nodes = {edge.source for edge in EDGES} | {edge.target for edge in EDGES}
    else:
        target = find_asset(asset)
        if target is None:
            raise KeyError(asset)
        nodes: set[str] = {target.name}
        selected_keys: set[tuple[str, str, str]] = set()
        for _ in range(depth):
            newly_reached: set[str] = set()
            for edge in EDGES:
                key = _edge_key(edge)
                if direction in ("both", "downstream") and edge.source in nodes:
                    if key not in selected_keys:
                        selected.append(edge)
                        selected_keys.add(key)
                    if edge.target not in nodes:
                        newly_reached.add(edge.target)
                if direction in ("both", "upstream") and edge.target in nodes:
                    if key not in selected_keys:
                        selected.append(edge)
                        selected_keys.add(key)
                    if edge.source not in nodes:
                        newly_reached.add(edge.source)
            if not newly_reached:
                break
            nodes |= newly_reached

    node_payloads = []
    for name in sorted(nodes):
        registered = ASSET_INDEX.get(name)
        if registered is None:
            # 正常路径不会发生（测试会保证）；保留兜底是为了让血缘图能画出“未登记节点”，
            # 而不是直接 500 —— 目录的价值在于发现缺口，不是掩盖缺口。
            node_payloads.append(
                {"name": name, "kind": "unknown", "layer": "unknown", "description": ""}
            )
            continue
        node_payloads.append(
            {
                "name": registered.name,
                "kind": registered.kind,
                "layer": registered.layer,
                "description": registered.description,
            }
        )

    edge_payloads = [asdict(edge) for edge in selected]
    return {
        "nodes": node_payloads,
        "edges": edge_payloads,
        "count": len(edge_payloads),
        "direction": direction,
        "depth": depth,
    }
