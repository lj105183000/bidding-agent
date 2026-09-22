"""联网结果重排测试：降序 / 分数缺失兜底 / 精排缺失与异常时的降级 / 调用契约。

桩必须照 `src/rag/embedder.py` 的 **真实** `Reranker.rerank` 写：

    rerank(self, query: str, candidates: list[dict], top_k: int, text_key: str = "question") -> list[dict]

候选是 `list[dict]`（它按 `text_key` 从字典取值配对），返回的是候选的**副本**
按 logit 降序截断后的列表，每项补 `rerank_score`（原始 logit）与 `score`（sigmoid 0-1）。
`_FakeReranker` 里逐条照抄这段实现——之前这层桩是按"元组进、分数列表出"的想当然形状
写的，结果把 `rag_tools` 与真精排器的接口不匹配全挡住了：测试全绿而功能静默失效。
改这个文件前先回 `embedder.py:139-174` 核一遍签名与返回形状。

`src/rag` 由成员 B 实现（已随 PR #3 合入 main），`_rerank_web_results` 里是**函数内导入**
且取的是**包级属性**，所以测试把带 `reranker` 属性的假包塞进 `sys.modules` 即可生效，
不必造 `src/rag/reranker.py`——那个文件不存在。

断言只针对行为：顺序、值域、条数、是否改动入参、传给对方的形状。
"""

import math
import sys
import types

from src.tools.rag_tools import _rerank_web_results


def _sigmoid(x: float) -> float:
    """照抄 embedder.py:177——真精排器用它把 logit 压成 0-1 的展示分。"""
    return 1.0 / (1.0 + math.exp(-x))


class _FakeReranker:
    """复刻 `Reranker.rerank` 的真实契约（签名与返回形状与 embedder.py 逐条对齐）。

    - `logits`：按候选顺序给的原始分；callable 时按候选条数现算。
    - `error`：设了就抛，模拟模型加载失败。
    - `shuffle`：返回**未排序**的候选，用来验证重排函数自己会排，不吃对方的承诺。
    - `drop_score`：返回的候选不带 `score` 键，模拟分数缺失。
    """

    def __init__(self, logits=None, error=None, shuffle=False, drop_score=False):
        self.logits = logits
        self.error = error
        self.shuffle = shuffle
        self.drop_score = drop_score
        self.calls = []

    def rerank(self, query, candidates, top_k, text_key="question"):
        # top_k 刻意不给默认值：真签名里它是必填位置参数，桩也不该放宽，
        # 否则调用方漏传时测试替它兜住了
        self.calls.append({"query": query, "candidates": [dict(c) for c in candidates], "top_k": top_k})
        if self.error is not None:
            raise self.error
        logits = self.logits(len(candidates)) if callable(self.logits) else self.logits
        pairs = list(zip(candidates, logits))
        if not self.shuffle:
            pairs.sort(key=lambda kv: float(kv[1]), reverse=True)
        out = []
        for item, logit in pairs[:top_k]:
            # 真实现是 `merged = dict(item)` 造副本：对象身份会丢失，
            # 调用方只能靠自己的键认回来，桩必须保留这一点
            merged = dict(item)
            merged["rerank_score"] = float(logit)
            if not self.drop_score:
                merged["score"] = _sigmoid(float(logit))
            out.append(merged)
        return out


class _RawReranker:
    """原样返回预设值，用来制造"对方返回了无法识别的形状"。"""

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def rerank(self, query, candidates, top_k, text_key="question"):
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        return self.raw


def _install_fake_reranker(monkeypatch, reranker):
    """把假精排器挂在假 `src.rag` 包的 `reranker` 属性上（对应真包的 __init__ 导出）。"""
    package = types.ModuleType("src.rag")
    package.__path__ = []
    package.reranker = reranker
    monkeypatch.setitem(sys.modules, "src.rag", package)
    return reranker


def _items(*names):
    """构造联网结果（question 承载标题，answer 承载正文）。"""
    return [{"question": name, "answer": f"{name} 的正文"} for name in names]


def _order(records):
    return [record["question"] for record in records]


class Test_重排行为:
    """精排器可用时的排序行为。"""

    def test_空列表返回空列表(self):
        """验证空输入不崩且返回空列表。"""
        assert _rerank_web_results("空调采购", []) == []

    def test_输入为None或字符串时返回空列表(self):
        """验证非列表输入被当作空结果处理。"""
        assert _rerank_web_results("空调采购", None) == []
        assert _rerank_web_results("空调采购", "不是列表") == []

    def test_按精排分数降序排列(self, monkeypatch):
        """验证结果按精排 logit 从高到低重排，而不是保持引擎顺序。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[-3.0, 5.0, 0.0]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["B", "C", "A"]

    def test_精排返回的顺序被打乱时仍按分数降序(self, monkeypatch):
        """验证重排函数自己会排序，不依赖精排器"已降序"的承诺。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[-3.0, 5.0, 0.0], shuffle=True))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["B", "C", "A"]

    def test_用精排给出的分数不再二次归一化(self, monkeypatch):
        """验证直接采用精排器的 sigmoid 分，不把它 min-max 拉伸成 0 与 1 两端。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[1.0, 2.0, 3.0]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        scores = [record["score"] for record in result]
        assert _order(result) == ["C", "B", "A"]
        assert scores[0] < 1.0 and scores[-1] > 0.0
        # 容差按展示精度给（写回时四舍五入到 4 位小数），关键是它等于精排给的分，
        # 而不是 min-max 之后必然出现的 1.0 / 0.0
        assert abs(scores[0] - _sigmoid(3.0)) < 1e-3
        assert abs(scores[-1] - _sigmoid(1.0)) < 1e-3
        assert 0.0 <= scores[-1] <= scores[0] <= 1.0

    def test_全同分时保持原顺序(self, monkeypatch):
        """验证分数全相同时不打乱原有顺序。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[1.0, 1.0, 1.0]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_top_k截断只保留前N条(self, monkeypatch):
        """验证 top_k 生效：只保留重排后的前 N 条，且取的是分数最高的那些。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[-3.0, 5.0, 0.0]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"), top_k=2)
        assert _order(result) == ["B", "C"]

    def test_不修改传入的列表与元素(self, monkeypatch):
        """验证重排返回新列表，原入参的元素与分数都不被就地改写。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[-3.0, 5.0]))
        items = _items("A", "B")
        items[0]["score"] = 0.42
        _rerank_web_results("空调采购", items)
        assert _order(items) == ["A", "B"]
        assert items[0]["score"] == 0.42
        assert "score" not in items[1]

    def test_问题为空时不做重排(self, monkeypatch):
        """验证没有用户问题时保持引擎顺序，不调用精排器。"""
        fake = _install_fake_reranker(monkeypatch, _FakeReranker(logits=[-3.0, 5.0]))
        result = _rerank_web_results("", _items("A", "B"))
        assert _order(result) == ["A", "B"]
        assert fake.calls == []

    def test_传给精排器的候选是带question的字典列表(self, monkeypatch):
        """验证调用形状符合对方契约：候选是 list[dict] 且配对字段是 question。"""
        fake = _install_fake_reranker(monkeypatch, _FakeReranker(logits=[1.0, 2.0]))
        _rerank_web_results("空调采购", _items("A", "B"))
        candidates = fake.calls[0]["candidates"]
        assert isinstance(candidates, list) and len(candidates) == 2
        assert all(isinstance(candidate, dict) for candidate in candidates)
        # 配对文本同时含标题与正文：只给标题会丢掉判断相关性最需要的正文信息
        assert "A" in candidates[0]["question"] and "正文" in candidates[0]["question"]


class Test_重排降级:
    """精排不可用或返回形状不对时的兜底：一律保持原顺序，绝不抛异常。"""

    def test_精排模块缺失时保持原顺序(self, monkeypatch):
        """验证 src.rag 导入不了时原样返回（sys.modules 置 None 让导入必然失败）。"""
        monkeypatch.setitem(sys.modules, "src.rag", None)
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_精排抛异常时保持原顺序(self, monkeypatch):
        """验证精排器报错时吞掉异常并保持引擎顺序。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(error=RuntimeError("模型加载失败")))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_分数为None时不崩且保持原顺序(self, monkeypatch):
        """验证精排返回的候选没有 score 键时降级，不出现 NaN 或崩溃。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[3.0, 2.0, 1.0], drop_score=True))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_分数为NaN时排到最后且不崩(self, monkeypatch):
        """验证 NaN 分数不被当成有效分数参与排序，落到末尾。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[0.9, float("nan"), 0.2]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "C", "B"]

    def test_精排返回空列表时保持原顺序(self, monkeypatch):
        """验证精排器返回空列表时原样返回，不把整批结果丢掉。"""
        _install_fake_reranker(monkeypatch, _RawReranker([]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_精排返回非列表时保持原顺序(self, monkeypatch):
        """验证精排器返回字符串等无法识别的形状时放弃重排。"""
        _install_fake_reranker(monkeypatch, _RawReranker("不是列表"))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_精排返回的候选认不出锚点时保持原顺序(self, monkeypatch):
        """验证返回的字典里没有回填锚点时放弃重排，而不是按下标硬套造成错位。"""
        _install_fake_reranker(monkeypatch, _RawReranker([{"question": "A", "score": 0.9}]))
        result = _rerank_web_results("空调采购", _items("A", "B", "C"))
        assert _order(result) == ["A", "B", "C"]

    def test_单条结果原样返回且不改成满分(self, monkeypatch):
        """验证只有一条结果时不重排，也不把引擎分覆盖成满分。"""
        fake = _install_fake_reranker(monkeypatch, _FakeReranker(logits=[5.0]))
        items = _items("A")
        items[0]["score"] = 0.3
        result = _rerank_web_results("空调采购", items)
        assert len(result) == 1
        assert result[0]["score"] == 0.3
        assert fake.calls == []

    def test_结果里的非字典元素被过滤(self, monkeypatch):
        """验证结果列表里混入字符串等脏数据时只保留字典元素。"""
        _install_fake_reranker(monkeypatch, _FakeReranker(logits=[5.0, 1.0]))
        mixed = [{"question": "A", "answer": "a"}, "脏数据", None, {"question": "B", "answer": "b"}]
        result = _rerank_web_results("空调采购", mixed)
        assert _order(result) == ["A", "B"]
