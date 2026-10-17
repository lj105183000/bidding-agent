"""格式化器与工具执行器的降级测试。

两层各测一件事：
- `_fmt_*` 是**最后一道防线**，输入可能是任意形状（后端宕机返回 None、字段缺失、
  列表里混着非 dict），所以直接拿畸形输入调真实实现，断言"仍然产出可读文本"，
  不 mock 内部辅助函数（`_fmt_rows` / `_truncate` 一律走真的）。
- 执行器是 Agent 与数据源之间的接缝，底层抛异常时**必须**返回 `("文案", [])`，
  绝不能让异常冒到 ReAct 循环里。这里只 mock **跨模块边界**（数据库客户端单例、
  `src.rag.pipeline`、`src.web_search`、`src.mcp.web_search_exa`）。

断言只针对行为（是否可读、条数、结构、先后关系），不比对具体文案全等。
"""

import sys
import types

import pytest

from src.tools import rag_tools
from src.tools.rag_tools import (
    MAX_OBSERVATION_CHARS,
    PG_DISPLAY_ROWS,
    TOOL_EXECUTORS,
    _fmt_graph,
    _fmt_pg,
    _fmt_rag,
    _fmt_web,
)


# --------------------------------------------------------------------------
# 局部 fake（约定见开发文档 §9.3：文件内局部类，不建全局 fixture）
# --------------------------------------------------------------------------


class _FakeRagPipeline:
    """模拟成员 B 的 RAGPipeline：返回条目或抛异常。

    ★ 必须同时提供 `search` 与 `search_cached`（契约变更）
    ------------------------------------------------
    `search_knowledge_base` 工具**已改为调 `search_cached`**
    （原先调 `search`，导致 Agent 路径完全绕过缓存——实测缺陷，见 docs/40）。

    替身只实现 `search` 时，工具会在 `AttributeError` 处进 except，
    返回「知识库检索失败」文案 —— **测试看到的是"检索失败"而非"契约不符"**，
    排查方向会被带偏（实测如此）。
    """

    def __init__(self, items=None, error=None):
        self.items = items if items is not None else []
        self.error = error
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append({"question": question, "top_k": top_k})
        if self.error is not None:
            raise self.error
        return self.items

    def search_cached(self, question, top_k=5):
        """与 `search` 同签名同返回——替身不模拟缓存语义。"""
        return self.search(question, top_k=top_k)


class _FakeGraphClient:
    """模拟 Neo4jClient：查询方法可选抛异常，`health()` 决定"没连上"还是"没查到"。"""

    def __init__(self, data=None, error=None, healthy=True):
        self.data = data
        self.error = error
        self.healthy = healthy

    def health(self):
        return {"ok": self.healthy, "error": "" if self.healthy else "连不上"}

    def _answer(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.data

    # 6 个查询方法都走同一个假实现：测试只关心"异常被吞掉"与"空结果怎么报"
    search_entity = items_by_entity = entity_detail = supplier_by_subject = top_traded = graph_stats = _answer


class _FakePgClient:
    """模拟 PostgresClient：ValueError 与其它异常要走向两条不同的文案分支。"""

    def __init__(self, data=None, error=None, healthy=True):
        self.data = data
        self.error = error
        self.healthy = healthy
        self.calls = []

    def health(self):
        return {"ok": self.healthy}

    def _answer(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return self.data

    search_by_keyword = by_purchaser = by_supplier = amount_range = location_dist = _answer
    trend = purchaser_ranking = supplier_ranking = avg_amount_by_subject = top_subject = _answer
    stats_overview = _answer


class _FakeWebSearchClient:
    """模拟联网客户端：`search()` 返回载荷或抛异常。

    参数用 `**kwargs` 收：Tavily 走 `max_results=`、Exa 走 `num_results=`，
    两个执行器的调用形状不同，桩不该替被测代码决定用哪个名字。
    """

    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else {"results": [], "answer": ""}
        self.error = error
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return self.payload


def _install(monkeypatch, name, value):
    """替换 rag_tools 里的懒加载单例工厂（执行器读的是模块级名字）。"""
    monkeypatch.setattr(rag_tools, name, lambda *a, **k: value)


# --------------------------------------------------------------------------
# _fmt_rag
# --------------------------------------------------------------------------


class Test_格式化知识库:
    """_fmt_rag 的畸形输入与截断行为。"""

    def test_空列表给出空结果文案(self):
        """验证 items=[] 时返回非空的空结果提示而不是空串。"""
        text = _fmt_rag([])
        assert isinstance(text, str) and text.strip()
        assert "未找到" in text or "未" in text

    def test_None与字符串输入不崩(self):
        """验证 items 不是列表时按空结果处理。"""
        assert _fmt_rag(None).strip()
        assert _fmt_rag("不是列表").strip()

    def test_列表里混入非字典元素被跳过(self):
        """验证裸值元素不参与渲染，真实条目照常输出。"""
        text = _fmt_rag([{"question": "Q1", "answer": "A1"}, "脏数据", None, 42])
        assert "Q1" in text and "A1" in text
        assert "脏数据" not in text

    def test_只有答案没有问题时仍渲染(self):
        """验证缺 question 的条目不会被整条丢弃。"""
        text = _fmt_rag([{"answer": "只有答案"}])
        assert "只有答案" in text

    def test_全空字典条目被跳过(self):
        """验证既无问题也无答案的条目不算作一条资料。"""
        text = _fmt_rag([{}, {"question": "", "answer": "   "}])
        assert text.strip()

    def test_相关度缺失与NaN时不显示相关度(self):
        """验证 score 为 None/NaN 时条目照常渲染且不出现相关度行。"""
        text = _fmt_rag([{"question": "Q1", "answer": "A1", "score": None}])
        assert "Q1" in text
        text_nan = _fmt_rag([{"question": "Q2", "answer": "A2", "score": float("nan")}])
        assert "Q2" in text_nan

    def test_有条目但无分数时仍给出列表头部(self):
        """验证 score 字段整体缺失时仍能渲染编号列表。"""
        text = _fmt_rag([{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}])
        assert "[1]" in text and "[2]" in text

    def test_分数高低给出不同的可信度提示(self):
        """验证高分与低分批次给出的引用建议不同。

        ★ 必须用 `relevance` 而不是 `score`（这里原先用 score，正是 bug 本身）
        ---------------------------------------------------------------
        `score` 是 **RRF 融合分**，只反映候选内排名——项目自己的注释写着
        「任何查询都会有第一名，实测无关问题的 RRF 分（最高 0.500）
        能高于相关问题（最低 0.081）」。

        而原先实现正是拿 `score` 的平均值去比 0.5 打标签，于是
        **检索良好的问题会被告知「匹配度一般，请结合其他信息判断」**，
        模型据此认为资料不可靠——把质量信号喂反了（实测：真实 relevance 0.69，
        而 score 只有 0.10）。
        """
        high = _fmt_rag([{"question": "Q", "answer": "A", "relevance": 0.9}])
        low = _fmt_rag([{"question": "Q", "answer": "A", "relevance": 0.1}])
        assert high.splitlines()[0] != low.splitlines()[0]
        assert "良好" in high.splitlines()[0]
        assert "一般" in low.splitlines()[0]

    def test_score不再驱动可信度标签(self):
        """★ 回归哨兵：只有 `score`（无 relevance）时**不得**下「匹配度良好」的结论。

        RRF 分不支持下这个判断。这条守住在有人图省事把判据改回 score 时报警。
        """
        text = _fmt_rag([{"question": "Q", "answer": "A", "score": 0.99}])
        first = text.splitlines()[0]
        assert "良好" not in first
        assert "无法判断相关性" in first

    def test_相关度标注分数口径(self):
        """两种分数量纲不同，输出里必须标口径，否则模型会拿它们互相比较。"""
        with_rel = _fmt_rag([{"question": "Q", "answer": "A", "relevance": 0.7}])
        only_score = _fmt_rag([{"question": "Q", "answer": "A", "score": 0.7}])
        assert "相关度：0.70" in with_rel
        assert "排名分(不可判相关)：0.70" in only_score

    def test_limit生效只渲染前N条(self):
        """验证 limit 参数截断条目数，而不是把全部结果都塞进上下文。"""
        items = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(3)]
        text = _fmt_rag(items, limit=2)
        assert "Q0" in text and "Q1" in text
        assert "Q2" not in text

    def test_超长答案被截断(self):
        """验证超长 answer 被截短，避免单条资料吃掉整个上下文预算。"""
        text = _fmt_rag([{"question": "Q", "answer": "长" * 900}])
        assert len(text) < 900


# --------------------------------------------------------------------------
# _fmt_graph
# --------------------------------------------------------------------------


class Test_格式化图谱:
    """_fmt_graph 的分派与兜底渲染。"""

    def test_规模统计转成中文标签(self):
        """验证 graph_stats 的英文节点标签被翻成中文，不直接暴露给用户。"""
        text = _fmt_graph("graph_stats", {"SubjectMatter": 2528, "Supplier": 4138})
        assert "标的物" in text and "供应商" in text
        assert "SubjectMatter" not in text
        assert "2528" in text

    def test_规模统计为空字典给出空结果文案(self):
        """验证 graph_stats 收到 {} 时给出可读文案而不是空串。"""
        assert _fmt_graph("graph_stats", {}).strip()

    def test_规模统计收到非字典时不崩(self):
        """验证 graph_stats 的数据形状不符时按空结果处理。"""
        assert _fmt_graph("graph_stats", ["不是字典"]).strip()

    def test_实体详情分段展示(self):
        """验证 entity_detail 的各子列表按小标题分段输出。"""
        data = {
            "subject": "空调",
            "suppliers": [{"supplier": "甲公司", "amount": 120000.5}],
            "purchasers": [{"purchaser": "乙单位"}],
            "locations": ["杭州市"],
            "transactions": [{"id": "T1", "amount": 5000}],
        }
        text = _fmt_graph("entity_detail", data)
        assert "空调" in text
        assert "供应商" in text and "甲公司" in text
        assert "杭州市" in text

    def test_实体详情空字典给出空结果文案(self):
        """验证 entity_detail 收到 {} 时给出可读文案。"""
        assert _fmt_graph("entity_detail", {}).strip()

    def test_实体详情子列表缺失或为空时不崩(self):
        """验证 entity_detail 只有部分字段时照常渲染，缺的段直接省略。"""
        text = _fmt_graph("entity_detail", {"subject": "打印机", "suppliers": [], "purchasers": None})
        assert "打印机" in text

    def test_交易频次排行带序号(self):
        """验证 top_traded 输出带序号的排行。"""
        text = _fmt_graph("top_traded", [{"name": "空调", "tradeFrequency": 175}, {"name": "打印机", "tradeFrequency": 90}])
        assert "空调" in text and "175" in text
        assert "1." in text and "2." in text

    def test_模糊搜实体带类型标签(self):
        """验证 search_entity 结果把节点类型翻成中文。"""
        text = _fmt_graph("search_entity", [{"name": "空调", "label": "SubjectMatter"}])
        assert "空调" in text and "标的物" in text
        assert "SubjectMatter" not in text

    def test_实体关联标的物带关系标签(self):
        """验证 items_by_entity 把关系类型翻成中文。"""
        text = _fmt_graph("items_by_entity", [{"subject": "空调", "relation": "PURCHASED_BY"}])
        assert "空调" in text and "采购单位" in text
        assert "PURCHASED_BY" not in text

    def test_供应商与金额按两位小数展示(self):
        """验证 supplier_by_subject 的金额被格式化，不出现原始长小数。"""
        text = _fmt_graph("supplier_by_subject", [{"supplier": "甲公司", "amount": 120000.5678}])
        assert "甲公司" in text
        assert "120000.57" in text

    def test_未知查询类型兜底渲染而不是空白(self):
        """验证新增查询类型未登记时仍按"键：值"输出，不丢结果。"""
        text = _fmt_graph("某个新类型", [{"foo": "bar", "baz": 1}])
        assert "foo" in text and "bar" in text

    def test_空结果给出空结果文案(self):
        """验证空列表 / None / 字符串输入都返回非空提示。"""
        for payload in ([], None, "不是列表"):
            assert _fmt_graph("search_entity", payload).strip()

    def test_列表里混入非字典元素被跳过(self):
        """验证裸值元素被过滤，真实行照常渲染。"""
        text = _fmt_graph("search_entity", ["脏数据", {"name": "空调", "label": "SubjectMatter"}])
        assert "空调" in text
        assert "脏数据" not in text


# --------------------------------------------------------------------------
# _fmt_pg
# --------------------------------------------------------------------------


class Test_格式化业务库:
    """_fmt_pg 的两种返回形状（dict 统计 / list 表格）与表格渲染。"""

    def test_统计字典按固定顺序展示(self):
        """验证 stats_overview 的键按登记顺序输出，同一份数据每次渲染一致。"""
        text = _fmt_pg("stats_overview", {"distinct_purchaser": 3584, "total_records": 8746})
        assert "记录总数" in text and "8746" in text
        assert text.index("记录总数") < text.index("采购单位数")

    def test_统计字典为空给出空结果文案(self):
        """验证空统计字典返回可读文案。"""
        assert _fmt_pg("stats_overview", {}).strip()

    def test_统计金额格式化成两位小数(self):
        """验证 total_amount 被格式化，不出现原始浮点长尾。"""
        text = _fmt_pg("stats_overview", {"total_amount": 1234567.891})
        assert "1234567.89" in text

    def test_表格带中文表头(self):
        """验证 list 结果的表头用中文列名。"""
        text = _fmt_pg("search_by_keyword", [{"project_name": "某项目", "purchaser": "某单位"}])
        assert "项目名称" in text and "采购单位" in text
        assert "某项目" in text

    def test_表头取所有行的键的并集(self):
        """验证某行缺列时该列仍出现在表头，且该行留空而不是错位。"""
        rows = [{"a": 1}, {"b": 2}]
        text = _fmt_pg("search_by_keyword", rows)
        body = [line for line in text.splitlines() if " | " in line]
        assert len(body) >= 2
        assert all(len(line.split(" | ")) == 2 for line in body)

    def test_空列表与None给出空结果文案(self):
        """验证空结果与非列表输入都返回非空提示。"""
        for payload in ([], None, "不是列表"):
            assert _fmt_pg("search_by_keyword", payload).strip()

    def test_列表里混入非字典元素被跳过(self):
        """验证裸值元素被过滤，真实行照常渲染。"""
        text = _fmt_pg("search_by_keyword", ["脏数据", {"project_name": "某项目"}])
        assert "某项目" in text
        assert "脏数据" not in text

    def test_超过显示上限时只渲染前20行并说明总数(self):
        """验证行数超过 PG_DISPLAY_ROWS 时截断展示，同时告知总条数。"""
        rows = [{"project_name": f"项目{i}"} for i in range(PG_DISPLAY_ROWS + 5)]
        text = _fmt_pg("search_by_keyword", rows)
        assert str(len(rows)) in text
        assert f"项目{PG_DISPLAY_ROWS}" not in text
        assert f"项目{PG_DISPLAY_ROWS - 1}" in text

    def test_超长字段被截断(self):
        """验证单元格超长时被截短，表格不会因单列过长而失去可读性。"""
        text = _fmt_pg("search_by_keyword", [{"project_name": "长" * 300}])
        assert len(text) < 300

    def test_金额列格式化成两位小数(self):
        """验证表格里的 amount 列被格式化。"""
        text = _fmt_pg("search_by_keyword", [{"supplier": "甲公司", "amount": 999.5}])
        assert "999.50" in text


# --------------------------------------------------------------------------
# _fmt_web
# --------------------------------------------------------------------------


class Test_格式化联网:
    """_fmt_web 的摘要段与结果条目渲染。"""

    def test_摘要与结果都展示(self):
        """验证摘要段与结果条目同时输出。"""
        text = _fmt_web("这是摘要", [{"title": "标题", "url": "https://a.example", "content": "正文"}])
        assert "这是摘要" in text
        assert "标题" in text and "https://a.example" in text

    def test_无摘要时不出现摘要段(self):
        """验证 answer 为空时只输出结果条目，不留空标题。"""
        text = _fmt_web("", [{"title": "标题", "url": "https://a.example"}])
        assert "标题" in text
        assert "摘要" not in text

    def test_只有摘要没有结果时仍可读(self):
        """验证结果为空但摘要有内容时不返回空结果文案。"""
        text = _fmt_web("这是摘要", [])
        assert "这是摘要" in text

    def test_摘要与结果都为空给出空结果文案(self):
        """验证两者皆空时返回可读文案。"""
        assert _fmt_web("", []).strip()
        assert _fmt_web(None, None).strip()

    def test_列表里混入非字典元素被跳过(self):
        """验证裸值元素被过滤，真实条目照常渲染。"""
        text = _fmt_web("", ["脏数据", {"title": "标题", "url": "https://a.example"}])
        assert "标题" in text
        assert "脏数据" not in text

    def test_标题缺失时回落到链接(self):
        """验证没有 title 的条目用 url 充当标题，不出现空标题。"""
        text = _fmt_web("", [{"url": "https://b.example"}])
        assert "https://b.example" in text

    def test_超长正文被截断(self):
        """验证单条正文按 WEB_SNIPPET_CHARS 截断。"""
        text = _fmt_web("", [{"title": "标题", "content": "长" * 800}])
        assert len(text) < 800

    def test_limit生效只渲染前N条(self):
        """验证 limit 参数截断展示条数，同时说明总条数。"""
        items = [{"title": f"标题{i}", "url": f"https://x{i}.example"} for i in range(4)]
        text = _fmt_web("", items, limit=2)
        assert "标题0" in text and "标题1" in text
        assert "标题3" not in text


# --------------------------------------------------------------------------
# 执行器的降级路径
# --------------------------------------------------------------------------


class Test_知识库执行器:
    """search_knowledge_base 在 RAG 未就绪 / 报错 / 无结果时的四态分离。"""

    def test_模块未就绪时降级且不抛异常(self, monkeypatch):
        """验证 RAGPipeline 拿不到时返回文案与空来源，而不是抛异常。"""
        _install(monkeypatch, "_get_rag_pipeline", None)
        text, sources = TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调采购"})[:2]
        assert isinstance(text, str) and text.strip()
        assert sources == []

    def test_检索抛异常时降级且不抛异常(self, monkeypatch):
        """验证底层检索报错时吞掉异常并返回文案。"""
        _install(monkeypatch, "_get_rag_pipeline", _FakeRagPipeline(error=RuntimeError("Qdrant 连接失败")))
        text, sources = TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调采购"})[:2]
        assert text.strip() and sources == []
        assert "Qdrant" not in text

    def test_检索无结果时降级(self, monkeypatch):
        """验证后端返回空列表时给出"未找到"类文案与空来源。"""
        _install(monkeypatch, "_get_rag_pipeline", _FakeRagPipeline(items=[]))
        text, sources = TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调采购"})[:2]
        assert text.strip() and sources == []

    def test_缺少检索内容时提示补参数(self, monkeypatch):
        """验证 query 与用户问题都为空时不调用后端，直接提示缺参数。"""
        pipeline = _FakeRagPipeline(items=[{"question": "Q", "answer": "A"}])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        text, sources = TOOL_EXECUTORS["search_knowledge_base"]({}, "")[:2]
        assert text.strip() and sources == []
        assert pipeline.calls == []

    def test_未给query时回落到用户原始问题(self, monkeypatch):
        """验证模型没猜出参数名时用用户问题当检索式。"""
        pipeline = _FakeRagPipeline(items=[{"question": "Q", "answer": "A", "score": 0.8}])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        TOOL_EXECUTORS["search_knowledge_base"]({}, "2025年空调采购情况")
        assert pipeline.calls[0]["question"] == "2025年空调采购情况"

    def test_正常检索的来源字段契约(self, monkeypatch):
        """验证 sources 元素的字段契约，且与格式化文本一致。

        2026-10 起为**四个键**：`question` / `answer` / `score` / `relevance`。

        新增 `relevance` 是有意的增量：`score` 是 RRF 排名分（任何查询都有第一名），
        **无法判断相关性**；`relevance` 才是稠密余弦相似度。原先丢掉它导致
        前端与质量判断只能看到一个无意义的 0.10。
        """
        pipeline = _FakeRagPipeline(items=[{
            "question": "Q1", "answer": "A1", "score": 0.83, "relevance": 0.71,
        }])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        text, sources = TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调采购"})[:2]
        assert "Q1" in text
        assert len(sources) == 1
        assert sorted(sources[0]) == ["answer", "question", "relevance", "score"]
        assert sources[0]["score"] == 0.83
        assert sources[0]["relevance"] == 0.71

    def test_无relevance时字段仍存在为None(self, monkeypatch):
        """检索结果没带 relevance（如旧契约的 fake）时，键要在、值为 None。

        键缺失会让下游 `s.get("relevance")` 与 `"relevance" in s` 行为不一致，
        前端拿不到稳定的形状。
        """
        pipeline = _FakeRagPipeline(items=[{"question": "Q1", "answer": "A1", "score": 0.5}])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        _, sources = TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调"})[:2]
        assert "relevance" in sources[0]
        assert sources[0]["relevance"] is None

    def test_top_k参数越界被收敛(self, monkeypatch):
        """验证 LLM 给出超范围的 top_k 时被夹到上限，不会一路传到检索层。"""
        pipeline = _FakeRagPipeline(items=[{"question": "Q", "answer": "A"}])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        TOOL_EXECUTORS["search_knowledge_base"]({"query": "空调", "top_k": 9999})
        assert pipeline.calls[0]["top_k"] == rag_tools.RAG_MAX_TOP_K


class Test_图谱执行器:
    """search_knowledge_graph 的参数容错与"没连上/没查到"的区分。"""

    def test_客户端未就绪时降级且不抛异常(self, monkeypatch):
        """验证 Neo4j 客户端拿不到时返回文案与空来源。"""
        _install(monkeypatch, "_get_neo4j", None)
        text, sources = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "graph_stats"})[:2]
        assert text.strip() and sources == []

    def test_查询抛异常时降级且不抛异常(self, monkeypatch):
        """验证客户端报错时吞掉异常并返回文案。"""
        _install(monkeypatch, "_get_neo4j", _FakeGraphClient(error=RuntimeError("Aura 已暂停")))
        text, sources = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "top_traded"})[:2]
        assert text.strip() and sources == []
        assert "Aura" not in text

    def test_空结果时按健康状态区分没连上还是没查到(self, monkeypatch):
        """验证同为空的两种原因给出不同文案：服务不可用 vs 未找到。"""
        _install(monkeypatch, "_get_neo4j", _FakeGraphClient(data=[], healthy=False))
        down, _ = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "search_entity", "entity": "空调"})[:2]
        _install(monkeypatch, "_get_neo4j", _FakeGraphClient(data=[], healthy=True))
        empty, _ = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "search_entity", "entity": "空调"})[:2]
        assert down != empty

    def test_未知查询类型回落到模糊搜索(self, monkeypatch):
        """验证模型猜错 query_type 时回落到 search_entity，而不是直接失败。"""
        client = _FakeGraphClient(data=[{"name": "空调", "label": "SubjectMatter"}])
        _install(monkeypatch, "_get_neo4j", client)
        text, _ = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "不存在的类型", "entity": "空调"})[:2]
        assert "空调" in text

    def test_缺实体名时提示补参数(self, monkeypatch):
        """验证需要实体名的查询类型在缺参数时给出可读提示。"""
        client = _FakeGraphClient(data=[{"name": "空调"}])
        _install(monkeypatch, "_get_neo4j", client)
        text, sources = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "entity_detail"}, "")[:2]
        assert text.strip() and sources == []

    def test_图谱来源恒为空(self, monkeypatch):
        """验证图谱结果不进 sources（图谱没有可引用的问答对）。"""
        _install(monkeypatch, "_get_neo4j", _FakeGraphClient(data={"SubjectMatter": 2528}))
        text, sources = TOOL_EXECUTORS["search_knowledge_graph"]({"query_type": "graph_stats"})[:2]
        assert "2528" in text
        assert sources == []


class Test_业务库执行器:
    """query_database 的参数校验分支与异常收敛。"""

    def test_客户端未就绪时降级且不抛异常(self, monkeypatch):
        """验证 PG 客户端拿不到时返回文案与空来源。"""
        _install(monkeypatch, "_get_postgres", None)
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "stats_overview"})[:2]
        assert text.strip() and sources == []

    def test_查询抛异常时降级且不抛异常(self, monkeypatch):
        """验证客户端报错时吞掉异常并返回"服务不可用"类文案。"""
        _install(monkeypatch, "_get_postgres", _FakePgClient(error=RuntimeError("password authentication failed")))
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "stats_overview"})[:2]
        assert text.strip() and sources == []
        assert "password" not in text

    def test_参数非法时回给LLM可改的提示(self, monkeypatch):
        """验证底层 ValueError 不被当成服务故障，而是提示参数有误。"""
        _install(monkeypatch, "_get_postgres", _FakePgClient(error=ValueError("group_by 只支持 month / year")))
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "trend", "group_by": "quarter"})[:2]
        assert text.strip() and sources == []
        assert "参数" in text

    def test_缺必需参数时提示补参数(self, monkeypatch):
        """验证关键词类查询缺参数时不拿空串去查，而是明确说缺什么。"""
        client = _FakePgClient(data=[])
        _install(monkeypatch, "_get_postgres", client)
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "search_by_keyword"}, "")[:2]
        assert text.strip() and sources == []
        assert client.calls == []

    def test_金额区间两个边界都缺时提示补参数(self, monkeypatch):
        """验证 amount_range 两个边界都缺时不查全表，而是要求至少给一个。"""
        client = _FakePgClient(data=[])
        _install(monkeypatch, "_get_postgres", client)
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "amount_range"}, "")[:2]
        assert text.strip() and sources == []
        assert client.calls == []

    def test_未知查询类型回落到关键词搜索(self, monkeypatch):
        """验证模型猜错 query_type 时回落到 search_by_keyword。"""
        client = _FakePgClient(data=[{"project_name": "某项目"}])
        _install(monkeypatch, "_get_postgres", client)
        text, _ = TOOL_EXECUTORS["query_database"]({"query_type": "不存在的类型", "keyword": "空调"})[:2]
        assert "某项目" in text

    def test_不支持的粒度被收敛掉而不报错(self, monkeypatch):
        """验证 group_by 给了客户端不认的值时先收敛成 month，不让它抛 ValueError。"""
        client = _FakePgClient(data=[{"period": "2025-01", "cnt": 3}])
        _install(monkeypatch, "_get_postgres", client)
        TOOL_EXECUTORS["query_database"]({"query_type": "trend", "group_by": "quarter"})
        assert client.calls[0]["kwargs"]["group_by"] == "month"

    def test_空结果时按健康状态区分没连上还是没查到(self, monkeypatch):
        """验证同为空的两种原因给出不同文案。"""
        _install(monkeypatch, "_get_postgres", _FakePgClient(data=[], healthy=False))
        down, _ = TOOL_EXECUTORS["query_database"]({"query_type": "location_dist"})[:2]
        _install(monkeypatch, "_get_postgres", _FakePgClient(data=[], healthy=True))
        empty, _ = TOOL_EXECUTORS["query_database"]({"query_type": "location_dist"})[:2]
        assert down != empty

    def test_统计总览的字典结果照常渲染(self, monkeypatch):
        """验证 stats_overview 返回单个 dict 时不被当成空结果丢掉。"""
        _install(monkeypatch, "_get_postgres", _FakePgClient(data={"total_records": 8746}))
        text, sources = TOOL_EXECUTORS["query_database"]({"query_type": "stats_overview"})[:2]
        assert "8746" in text
        assert sources == []

    def test_业务库来源恒为空(self, monkeypatch):
        """验证业务库结果不进 sources。"""
        _install(monkeypatch, "_get_postgres", _FakePgClient(data=[{"project_name": "某项目"}]))
        _, sources = TOOL_EXECUTORS["query_database"]({"query_type": "search_by_keyword", "keyword": "空调"})[:2]
        assert sources == []


class Test_联网执行器:
    """search_web / search_exa 的密钥缺失、异常与未就绪降级。"""

    def test_未配置密钥时降级且不发请求(self, monkeypatch):
        """验证缺 TAVILY_API_KEY 时给出明确提示，而不是静默走免密通道。"""
        monkeypatch.setattr("src.web_search.is_configured", lambda: False)
        text, sources = TOOL_EXECUTORS["search_web"]({"query": "2025年空调采购"})[:2]
        assert text.strip() and sources == []
        assert "TAVILY_API_KEY" in text

    def test_搜索抛异常时降级且不泄露内部细节(self, monkeypatch):
        """验证 Tavily 报错时吞掉异常，且异常里的地址与密钥不进用户可见文案。"""
        monkeypatch.setattr("src.web_search.is_configured", lambda: True)
        monkeypatch.setattr(
            "src.web_search.web_search_client",
            _FakeWebSearchClient(error=RuntimeError("连接 10.0.0.5:8080 失败 secret-token")),
        )
        text, sources = TOOL_EXECUTORS["search_web"]({"query": "2025年空调采购"})[:2]
        assert text.strip() and sources == []
        assert "10.0.0.5" not in text and "secret-token" not in text

    def test_空结果时降级(self, monkeypatch):
        """验证联网返回 0 条时给出"没搜到"类文案与空来源。"""
        monkeypatch.setattr("src.web_search.is_configured", lambda: True)
        monkeypatch.setattr("src.web_search.web_search_client", _FakeWebSearchClient({"results": [], "answer": ""}))
        text, sources = TOOL_EXECUTORS["search_web"]({"query": "2025年空调采购"})[:2]
        assert text.strip() and sources == []

    def test_正常搜索的来源字段契约(self, monkeypatch):
        """验证联网 sources 元素的字段契约。

        2026-10 起为**五个键**：`question`（标题）/ `answer`（正文）/ `score` /
        `url` / `score_kind`。新增 `score_kind` 是有意的增量：
        `engine` 引擎自带分与 `cosine` 本地补算分的**口径不同、不可横比**，
        前端要据此提示用户（不标的话用户会拿 0.0001 的引擎分与 0.71 的
        本地匹配度比较而得出错误结论）。
        """
        monkeypatch.setattr("src.web_search.is_configured", lambda: True)
        monkeypatch.setattr(
            "src.web_search.web_search_client",
            _FakeWebSearchClient(
                {"results": [{"title": "公告", "url": "https://a.example", "content": "正文"}], "answer": "摘要"}
            ),
        )
        text, sources = TOOL_EXECUTORS["search_web"]({"query": "2025年空调采购"}, "2025年空调采购")[:2]
        assert "摘要" in text and "https://a.example" in text
        assert len(sources) == 1
        assert sorted(sources[0]) == ["answer", "question", "score", "score_kind", "url"]
        assert sources[0]["url"] == "https://a.example"

    def test_未给query时回落到用户原始问题(self, monkeypatch):
        """验证模型没猜出参数名时用用户问题当搜索词。"""
        client = _FakeWebSearchClient()
        monkeypatch.setattr("src.web_search.is_configured", lambda: True)
        monkeypatch.setattr("src.web_search.web_search_client", client)
        TOOL_EXECUTORS["search_web"]({}, "2025年空调采购情况")
        assert client.calls[0]["query"] == "2025年空调采购情况"

    def test_Exa未就绪时降级且不抛异常(self, monkeypatch):
        """验证 src/mcp 未落地时给出可读提示，落地后无需改本文件。"""
        package = types.ModuleType("src.mcp")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "src.mcp", package)
        monkeypatch.delitem(sys.modules, "src.mcp.web_search_exa", raising=False)
        text, sources = TOOL_EXECUTORS["search_exa"]({"query": "语义搜索"})[:2]
        assert text.strip() and sources == []

    def test_Exa就绪时能返回结果(self, monkeypatch):
        """验证 src/mcp 落地后本工具自动挂上，无需改动 rag_tools。"""
        fake_client = _FakeWebSearchClient({"results": [{"title": "论文", "url": "https://e.example"}], "answer": ""})
        package = types.ModuleType("src.mcp")
        package.__path__ = []
        module = types.ModuleType("src.mcp.web_search_exa")
        module.exa_search_client = fake_client
        monkeypatch.setitem(sys.modules, "src.mcp", package)
        monkeypatch.setitem(sys.modules, "src.mcp.web_search_exa", module)
        text, sources = TOOL_EXECUTORS["search_exa"]({"query": "语义搜索"})[:2]
        assert "https://e.example" in text
        assert sorted(sources[0]) == ["answer", "question", "score", "score_kind", "url"]

    def test_Exa抛异常时降级且不抛异常(self, monkeypatch):
        """验证 Exa 客户端报错时吞掉异常并返回文案。"""
        module = types.ModuleType("src.mcp.web_search_exa")
        module.exa_search_client = _FakeWebSearchClient(error=RuntimeError("MCP 子进程已退出"))
        package = types.ModuleType("src.mcp")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "src.mcp", package)
        monkeypatch.setitem(sys.modules, "src.mcp.web_search_exa", module)
        text, sources = TOOL_EXECUTORS["search_exa"]({"query": "语义搜索"})[:2]
        assert text.strip() and sources == []
        assert "MCP" not in text


class Test_执行器统一契约:
    """5 个执行器共同要守的底线。"""

    @pytest.mark.parametrize("name", sorted(TOOL_EXECUTORS))
    def test_空参数不抛异常(self, name, monkeypatch):
        """验证任何执行器收到 {} 都返回 (str, list) 而不是抛异常。"""
        for factory in ("_get_rag_pipeline", "_get_neo4j", "_get_postgres"):
            _install(monkeypatch, factory, None)
        text, sources = TOOL_EXECUTORS[name]({})[:2]
        assert isinstance(text, str) and text.strip()
        assert isinstance(sources, list) and sources == []

    @pytest.mark.parametrize("name", sorted(TOOL_EXECUTORS))
    def test_参数不是字典时不抛异常(self, name, monkeypatch):
        """验证 arguments 传 None（tool_defense 解析出空参数时的形状）也能活下来。"""
        for factory in ("_get_rag_pipeline", "_get_neo4j", "_get_postgres"):
            _install(monkeypatch, factory, None)
        text, sources = TOOL_EXECUTORS[name](None)[:2]
        assert isinstance(text, str) and text.strip()
        assert isinstance(sources, list)

    @pytest.mark.parametrize("name", sorted(TOOL_EXECUTORS))
    def test_输出总量不超过上下文预算(self, name, monkeypatch):
        """验证超长结果被总量截断，不会撑爆 LLM 上下文。"""
        pipeline = _FakeRagPipeline(items=[{"question": f"Q{i}", "answer": "长" * 400} for i in range(20)])
        _install(monkeypatch, "_get_rag_pipeline", pipeline)
        _install(monkeypatch, "_get_neo4j", _FakeGraphClient(data=[{"name": "长" * 200}] * 50))
        _install(monkeypatch, "_get_postgres", _FakePgClient(data=[{"project_name": "长" * 200}] * 50))
        # 密钥若真的配了，这一条会打真实 API——单元测试必须与网络解耦，固定成"未配置"
        monkeypatch.setattr("src.web_search.is_configured", lambda: False)
        text, _ = TOOL_EXECUTORS[name]({"query": "空调", "keyword": "空调", "entity": "空调"})[:2]
        assert len(text) <= MAX_OBSERVATION_CHARS + 1

    def test_注册的工具名与常量一致(self):
        """验证 5 个键名与 constants 的 BASE_TOOL_NAMES / WEB_TOOL_NAMES 逐字相同。"""
        from src.agent.constants import BASE_TOOL_NAMES, WEB_TOOL_NAMES

        assert sorted(TOOL_EXECUTORS) == sorted(list(BASE_TOOL_NAMES) + list(WEB_TOOL_NAMES))

    def test_TOOLS列表与执行器一一对应(self):
        """验证 TOOLS 导出的 BaseTool 子类名与执行器键名一一对应。"""
        assert sorted(tool.name for tool in rag_tools.TOOLS) == sorted(TOOL_EXECUTORS)


class Test_透传图表契约:
    """工具可以**可选地**多返回一个元素：结构化图表数据。

    ★ 为什么需要这条契约测试
    -------------------
    改这条契约时（从"后端正则解析工具文本"改为"工具透传结构化数据"），
    5 个既有用例因为按二元组解包而挂掉 —— 说明**返回长度是被隐式依赖的契约**。
    故这里把它显式测出来：

        · 产图表的工具 → 3 元组，第三项是 `list[dict]` 且结构正确
        · 不产图表的工具 → 2 元组（不为它多返回空列表）
        · 失败/空结果 → 2 元组（没有数据可画）

    第三项的 schema 与 `charts.py` 的产出**必须一致**，否则前端要处理两种形状。
    """

    #: 已改造为透传的工具
    _CHART_TOOLS = {
        "query_database": {"query_type": "top_subject", "limit": 5},
        "search_knowledge_graph": {"query_type": "top_traded", "limit": 5},
    }

    @pytest.mark.integration
    def test_chart_tools_return_three_tuple(self):
        for name, args in self._CHART_TOOLS.items():
            out = TOOL_EXECUTORS[name](args)
            assert len(out) == 3, f"{name} 应返回三元组（文本, 来源, 图表）"

    @pytest.mark.integration
    def test_chart_shape_matches_schema(self):
        """★ 与 `charts.py` 产出的 schema 一致 —— 否则前端要处理两种形状。"""
        from src.agent.charts import extract_charts

        for name, args in self._CHART_TOOLS.items():
            _, _, charts = TOOL_EXECUTORS[name](args)
            if not charts:      # 库里可能没数据，此时不断言形状
                continue
            for c in charts:
                assert isinstance(c, dict)
                for key in ("kind", "title", "value_label", "label_label",
                            "data", "note", "source_tool"):
                    assert key in c, f"{name} 的图表缺字段 {key}"
                assert c["kind"] == "bar"
                assert isinstance(c["data"], list)
                for item in c["data"]:
                    assert set(item) == {"label", "value"}, (
                        f"数据项只应有 label/value，实为 {sorted(item)}"
                    )
                    assert isinstance(item["value"], (int, float))
            # 与非透传路径产出的键集合比对（两者的消费方是同一个前端组件）
            ref = extract_charts("- a：1\n- b：2\n- c：3", "query_database")
            if ref:
                assert set(charts[0]) == set(ref[0]), "两条路径的图表键必须一致"

    @pytest.mark.integration
    def test_no_chart_tools_return_two_tuple(self):
        """不产图表的工具不必多返回空列表 —— 契约是"可选第三项"。"""
        for name, args in (("search_knowledge_base", {"query": "空调"}),
                           ("search_web", {"query": "空调"})):
            if name not in TOOL_EXECUTORS:
                continue
            out = TOOL_EXECUTORS[name](args)
            assert len(out) == 2, f"{name} 应仍返回二元组"

    def test_failure_path_returns_two_tuple(self):
        """失败结果没有数据可画，返回二元组。

        ★ 这条很重要：调用方若一律按三元组解包，**失败路径会 ValueError** ——
        而失败路径恰恰是最需要稳定的时候。
        """
        out = TOOL_EXECUTORS["query_database"]({"query_type": "top_subject",
                                               "group_by": "不存在的粒度"})
        assert len(out) >= 2, "失败路径至少要能解出文本与来源"

    def test_dispatch_accepts_both_lengths(self):
        """工具返回值两种长度都要认 —— 这是「可选第三项（图表）」能成立的前提。

        ★ 改法说明（2026-10-16）
        ----------------------
        原来用**源码字符串**断言（`inspect.getsource(_execute_tool)` 里必须有
        `len(result) == 3`）。工具改为并行执行后，解包逻辑搬到了
        `_execute_tool_ex`（`_execute_tool` 变成写回实例属性的兼容入口），
        字符串断言随之失效 —— 那次失败是**测试抓对了变化**，不是回归。

        这里改成断言**行为**：装上返回三元组的执行器，就必须拿到图表；
        返回二元组的执行器，图表为空且不报错。行为断言比源码断言强：
        它不管实现怎么搬，只管契约有没有被破坏。
        """
        import src.agent.core as core_mod
        import src.agent.react_loop as loop_mod
        from src.agent import BiddingAgent

        original_loop = loop_mod.TOOL_EXECUTORS
        original_core = core_mod.TOOL_EXECUTORS
        try:
            executors = {
                "with_charts": lambda arguments, question="": (
                    "文本", [{"question": "q", "answer": "a"}],
                    [{"type": "bar", "name": "图", "data": []}],
                ),
                "without_charts": lambda arguments, question="": ("文本", []),
            }
            loop_mod.TOOL_EXECUTORS = executors
            core_mod.TOOL_EXECUTORS = executors
            agent = BiddingAgent(llm_client=None)  # type: ignore[arg-type]

            text3, sources3, ok3, obs3 = agent._execute_tool_ex("with_charts", {})
            assert ok3 is True and text3 == "文本" and sources3, "三元组未被正确解包"
            assert obs3["charts"], "三元组里的图表必须透传出来"

            text2, _sources2, ok2, obs2 = agent._execute_tool_ex("without_charts", {})
            assert ok2 is True and text2 == "文本", "二元组仍须支持"
            assert obs2["charts"] == [], "二元组没有图表数据，应为空而不是报错"
        finally:
            loop_mod.TOOL_EXECUTORS = original_loop
            core_mod.TOOL_EXECUTORS = original_core
