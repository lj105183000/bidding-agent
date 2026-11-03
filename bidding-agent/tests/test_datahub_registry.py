"""资产注册表与静态血缘的覆盖性测试。

这些用例守的是“目录说真话”：
    - 资产名唯一、kind/layer 合法；
    - 上下游、血缘边两端都能在索引里找到（否则前端画出一堆 unknown 节点）；
    - Neo4j 标签/关系与 `neo4j_client.ALLOWED_*` 完全一致；
    - 全量重建 / 运行时写入的资产绝不能被标成可调度。
"""

import pytest

from src.datahub.lineage import EDGES, lineage_subgraph
from src.datahub.registry import (
    ASSET_INDEX,
    ASSET_KINDS,
    ASSET_LAYERS,
    ASSETS,
    find_asset,
    list_assets,
)
from src.database.neo4j_client import ALLOWED_LABELS, ALLOWED_RELS
from src.rag.regulation_store import REGULATION_COLLECTION


def test_asset_names_unique():
    names = [asset.name for asset in ASSETS]
    assert len(names) == len(set(names))


def test_all_assets_use_known_kind_and_layer():
    for asset in ASSETS:
        assert asset.kind in ASSET_KINDS, f"{asset.name} 的 kind 未登记：{asset.kind}"
        assert asset.layer in ASSET_LAYERS, f"{asset.name} 的 layer 未登记：{asset.layer}"


def test_upstream_downstream_reference_registered_assets():
    for asset in ASSETS:
        for name in asset.upstream + asset.downstream:
            assert name in ASSET_INDEX, f"{asset.name} 引用了未登记的资产：{name}"


def test_lineage_edges_reference_registered_assets():
    for edge in EDGES:
        assert edge.source in ASSET_INDEX, f"血缘边起点未登记：{edge.source}"
        assert edge.target in ASSET_INDEX, f"血缘边终点未登记：{edge.target}"
        assert edge.kind in {"batch", "importer", "runtime", "scheduler", "code"}


def test_destructive_assets_are_not_schedulable():
    """全量重建 / 运行时写入的资产不能被调度器自动执行。

    这是安全红线：`ingest` 会删集合重建，`to_postgresql.py` 会 DROP 表，
    运行时写入路径更不该被“定时任务”碰。
    """
    for asset in ASSETS:
        if asset.modes & {"full_rebuild", "runtime_write"}:
            assert asset.schedulable is False, f"{asset.name} 不应可调度"


def test_full_rebuild_assets_declare_recreate_risk():
    for asset in ASSETS:
        if "full_rebuild" in asset.modes:
            assert asset.recreate_risk in {"drops_table", "drops_collection", "clears_graph"}, (
                f"{asset.name} 是全量重建，但没有声明重建风险"
            )


def test_neo4j_assets_cover_allowed_labels_and_relationships():
    labels = {
        asset.name.split(".", 1)[1] for asset in ASSETS if asset.kind == "neo4j_label"
    }
    relationships = {
        asset.name.split(".", 1)[1]
        for asset in ASSETS
        if asset.kind == "neo4j_relationship"
    }
    assert labels == ALLOWED_LABELS
    assert relationships == ALLOWED_RELS


def test_qdrant_assets_cover_documented_collections():
    names = {asset.name for asset in ASSETS if asset.kind == "qdrant_collection"}
    assert "qdrant.bid_qa" in names
    assert f"qdrant.{REGULATION_COLLECTION}" in names


def test_pg_assets_cover_core_tables():
    expected = {
        "pg.bidding_records",
        "pg.conversations",
        "pg.messages",
        "pg.feedback",
        "pg.tender_documents",
        "pg.company_profiles",
        "pg.session_traces",
        "pg.data_quality_reports",
        "pg.data_catalog_snapshots",
    }
    actual = {asset.name for asset in ASSETS if asset.kind == "postgres_table"}
    assert expected <= actual


def test_find_asset_supports_full_and_short_names():
    assert find_asset("pg.bidding_records").name == "pg.bidding_records"
    assert find_asset("bidding_records").name == "pg.bidding_records"
    assert find_asset("bid_qa").name == "qdrant.bid_qa"
    assert find_asset("SubjectMatter").name == "neo4j.SubjectMatter"
    assert find_asset("subjectmatter").name == "neo4j.SubjectMatter"


def test_find_asset_returns_none_for_unknown():
    assert find_asset("not.exists") is None
    assert find_asset("") is None


def test_list_assets_filters():
    pg_assets = list_assets(kind="postgres_table")
    assert pg_assets
    assert all(asset.kind == "postgres_table" for asset in pg_assets)
    raw_assets = list_assets(layer="raw")
    assert raw_assets
    assert all(asset.layer == "raw" for asset in raw_assets)


def test_lineage_subgraph_global_contains_all_edges():
    graph = lineage_subgraph(None)
    assert graph["count"] == len(EDGES)
    assert len(graph["nodes"]) >= 2


def test_lineage_subgraph_respects_direction_and_depth():
    upstream = lineage_subgraph("pg.bidding_records", direction="upstream", depth=1)
    assert upstream["count"] > 0
    # 上游只应包含“指向当前资产”的边（直接上游），不应把它的下游带进来
    for edge in upstream["edges"]:
        assert edge["target"] == "pg.bidding_records"

    downstream = lineage_subgraph("pg.bidding_records", direction="downstream", depth=1)
    assert downstream["count"] > 0
    for edge in downstream["edges"]:
        assert edge["source"] == "pg.bidding_records"


def test_lineage_subgraph_rejects_unknown_asset():
    with pytest.raises(KeyError):
        lineage_subgraph("not.exists")
