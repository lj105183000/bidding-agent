"""检索相关性过滤的单测。

背景（质量检查发现）：
    `pipeline.search()` 原先**没有相关性过滤**——RRF 融合保证任何查询都有
    top_k 条结果，于是"冰岛的首都是哪里"这种完全无关的问题也会拿到 5 条噪音，
    全部喂给生成端。

    实测分布（完全分离，故阈值 0.5 可用）：
        完全无关问题 top1   0.284 ~ 0.421
        评测题（应命中）    0.892 ~ 0.987

    不过滤的后果不是"答错"，而是**"知识库没有答案"这个判断退化成依赖大模型
    自己识破噪音**——换个弱模型或加上 ReAct 循环就会开始基于噪音作答。
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rec(rel, q="q", a="a"):
    return {"question": q, "answer": a, "score": 0.03, "relevance": rel}


class TestApplyRelevanceFilter:
    """直接测静态方法——不依赖向量库，跑得快且能覆盖边界。"""

    def test_drops_below_threshold(self):
        kept = RAGPipeline._apply_relevance_filter(
            [_rec(0.9), _rec(0.6), _rec(0.3), _rec(0.1)], min_relevance=0.5
        )
        assert [r["relevance"] for r in kept] == [0.9, 0.6]

    def test_keeps_exactly_at_threshold(self):
        """边界：恰好等于阈值要保留（`>=` 而非 `>`）。

        阈值是标定出来的"判定相关"下限，落在下限上的样本属于相关。
        """
        kept = RAGPipeline._apply_relevance_filter([_rec(0.5)], min_relevance=0.5)
        assert len(kept) == 1

    def test_keeps_none_relevance(self):
        """★ `relevance=None`（取向量失败）必须保留，不能当"不相关"。

        未知不等于不相关。基础设施出问题时宁可返回结果，
        也不要因为"算不出分数"就告诉用户"知识库没有答案"——
        那是把一次技术故障伪装成业务结论。
        与 `/api/ask` 的既有口径一致。
        """
        kept = RAGPipeline._apply_relevance_filter(
            [_rec(None), _rec(0.9)], min_relevance=0.5
        )
        assert len(kept) == 2

    def test_keeps_non_numeric_relevance(self):
        """非数值（脏数据）同样按"未知"处理，不因类型问题丢掉结果。"""
        kept = RAGPipeline._apply_relevance_filter(
            [{"question": "q", "answer": "a", "relevance": "0.9"}], min_relevance=0.5
        )
        assert len(kept) == 1

    def test_zero_threshold_disables_filter(self):
        """显式传 0 关闭过滤——语料自检需要看全量结果。"""
        records = [_rec(0.1), _rec(0.01), _rec(None)]
        assert RAGPipeline._apply_relevance_filter(records, 0.0) == records

    def test_negative_threshold_disables_filter(self):
        records = [_rec(0.1)]
        assert RAGPipeline._apply_relevance_filter(records, -1.0) == records

    def test_default_uses_min_relevance_constant(self):
        """★ 默认阈值必须取 `MIN_RELEVANCE`，不能各自写死一个数。

        阈值散落成多份（`/api/ask`、web_fallback、pipeline）时会漂移——
        同一个问题在不同入口得到"有答案/无答案"两种结论，用户无法理解。
        """
        from src.rag.vector_store import MIN_RELEVANCE

        records = [_rec(MIN_RELEVANCE - 0.001), _rec(MIN_RELEVANCE + 0.001)]
        kept = RAGPipeline._apply_relevance_filter(records, None)
        assert len(kept) == 1
        assert kept[0]["relevance"] == MIN_RELEVANCE + 0.001

    def test_all_dropped_returns_empty(self):
        """全部低于阈值 → 空列表，让上层走"知识库没有"的分支。"""
        assert RAGPipeline._apply_relevance_filter([_rec(0.1), _rec(0.2)], 0.5) == []

    def test_empty_input(self):
        assert RAGPipeline._apply_relevance_filter([], 0.5) == []

    def test_preserves_order(self):
        """过滤只做筛选，不重排——顺序由检索侧（RRF / 精排）决定。"""
        records = [_rec(0.9, "第一"), _rec(0.1, "丢"), _rec(0.7, "第二")]
        kept = RAGPipeline._apply_relevance_filter(records, 0.5)
        assert [r["question"] for r in kept] == ["第一", "第二"]

    def test_does_not_mutate_input(self):
        """不得原地修改——检索缓存返回的是浅拷贝，改元素会污染缓存。"""
        records = [_rec(0.9), _rec(0.1)]
        snapshot = [dict(r) for r in records]
        RAGPipeline._apply_relevance_filter(records, 0.5)
        assert records == snapshot


def test_search_accepts_min_relevance_kwarg():
    """`search` 必须接受 `min_relevance` 关键字参数（供自检关闭过滤）。"""
    import inspect

    sig = inspect.signature(RAGPipeline.search)
    assert "min_relevance" in sig.parameters
    # 必须是 keyword-only：位置传参会与既有的 search(q, top_k) 调用混淆
    assert sig.parameters["min_relevance"].kind is inspect.Parameter.KEYWORD_ONLY
