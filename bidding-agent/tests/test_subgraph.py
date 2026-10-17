"""子图抽取服务单测。

只测**不依赖 Neo4j** 的部分：参数校验、白名单、常量契约。
真实子图结果由端到端脚本验证。

重点守住：
    - `Transaction` 必须在白名单之外（8730 个孤立节点，画进图会卡死浏览器）
    - 非法标签/空名称要在**连库之前**拒绝
    - 节点上限必须存在且有上界（力导向图规模失控是真实风险）
"""

import pytest

from src.database import subgraph as SG


# ---------------------------------------------------------------------------
# 标签白名单
# ---------------------------------------------------------------------------


def test_transaction_excluded_from_allowed_labels():
    """Transaction 必须被排除——实测 8730 个，只挂金额无出边，画出来只是孤立点。"""
    assert "Transaction" not in SG.ALLOWED_LABELS


def test_allowed_labels_are_the_real_entity_labels():
    """白名单要与实测的图谱标签一致。"""
    assert set(SG.ALLOWED_LABELS) == {
        "SubjectMatter",
        "Purchaser",
        "Supplier",
        "Agency",
        "Location",
    }


def test_every_allowed_label_has_chinese_name():
    for lbl in SG.ALLOWED_LABELS:
        assert lbl in SG.LABEL_CN, f"{lbl} 缺少中文名"
        assert SG.LABEL_CN[lbl]


def test_transaction_still_has_chinese_name():
    """虽然不画进图，但仍需要中文名（用于日志与提示文案）。"""
    assert SG.LABEL_CN["Transaction"] == "交易"


def test_relation_names_cover_real_relationship_types():
    """实测的 5 种关系类型都要有中文名。"""
    for rel in ("HAS_TRANSACTION", "PURCHASED_BY", "SUPPLIED_BY", "LOCATED_AT", "AGENT_BY"):
        assert rel in SG.RELATION_CN, f"关系 {rel} 缺少中文名"


# ---------------------------------------------------------------------------
# 参数校验（必须在连库之前发生）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["Transaction", "Foo", "", "subjectmatter", "MATCH"])
def test_build_subgraph_rejects_bad_label(bad):
    """非法标签要报错，且**不应尝试连库**（连不上时会抛 SubgraphUnavailable 而非 SubgraphError）。"""
    with pytest.raises(SG.SubgraphError) as ei:
        SG.build_subgraph(bad, "空调")
    assert "不支持的节点类型" in str(ei.value)
    # 关键：这必须是我们自己的校验错误，而不是连接失败
    assert not isinstance(ei.value, SG.SubgraphUnavailable)


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_build_subgraph_rejects_empty_name(bad):
    with pytest.raises(SG.SubgraphError) as ei:
        SG.build_subgraph("SubjectMatter", bad)
    assert "实体名称" in str(ei.value)
    assert not isinstance(ei.value, SG.SubgraphUnavailable)


def test_label_validation_happens_before_name_validation():
    """两个都非法时，先报标签错（校验顺序稳定，便于前端按 400 处理）。"""
    with pytest.raises(SG.SubgraphError) as ei:
        SG.build_subgraph("Bad", "")
    assert "不支持的节点类型" in str(ei.value)


# ---------------------------------------------------------------------------
# 规模上限
# ---------------------------------------------------------------------------


def test_max_nodes_has_hard_ceiling():
    """硬上限必须存在：实测「空调」有 480 个邻居，无上限会拖垮浏览器。"""
    assert SG.MAX_NODES <= 200
    assert SG.MAX_NODES >= 20


def test_default_nodes_within_ceiling():
    assert SG.DEFAULT_NODES <= SG.MAX_NODES


def test_clamp_respects_ceiling():
    assert SG._clamp(9999, 5, SG.MAX_NODES) == SG.MAX_NODES
    assert SG._clamp(0, 5, SG.MAX_NODES) == 5
    assert SG._clamp(-10, 5, SG.MAX_NODES) == 5


def test_clamp_handles_bad_input():
    assert SG._clamp(None, 5, SG.MAX_NODES) == SG.DEFAULT_NODES
    assert SG._clamp("abc", 5, SG.MAX_NODES) == SG.DEFAULT_NODES


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------


def test_unavailable_is_subclass_of_error():
    """`SubgraphUnavailable` 要能被 `SubgraphError` 捕获，但调用方也能单独区分
    （一个映射 503、一个映射 400）。"""
    assert issubclass(SG.SubgraphUnavailable, SG.SubgraphError)
    assert issubclass(SG.SubgraphError, RuntimeError)


def test_error_types_are_distinguishable():
    assert SG.SubgraphUnavailable is not SG.SubgraphError


# ---------------------------------------------------------------------------
# 邻居上限
# ---------------------------------------------------------------------------


def test_neighbor_limit_reasonable():
    """单节点展开的邻居数要有上限，否则度数为 1410 的「合肥市」会拉回海量数据。"""
    assert 5 <= SG.MAX_NEIGHBORS_PER_NODE <= 100
