"""工具定义 + 执行器注册表 + 结果格式化（五路工具的落地文件）。

职责
----
把四个数据源翻译成**给 LLM 看的文本**，并把它们的来源列表交给上层：

    search_knowledge_base   → src.rag.pipeline.rag_pipeline（成员 B，未就绪时降级）
    search_knowledge_graph  → src.database.neo4j_client.Neo4jClient
    query_database          → src.database.postgresql_client.PostgresClient
    search_web              → src.web_search.web_search_client（Tavily）
    search_exa              → src.mcp.web_search_exa（成员 C 的另一部分，未落地时降级）

执行器契约（冻结，见 分配说明.md §5 / docs/组件工作机制.md §13）
--------------------------------------------------------------
`executor(arguments: dict, question: str = "") -> (格式化文本, sources 列表)`

- **永不抛异常**：失败返回 `("错误文案", [])`，让 Agent 带着"这一路没有数据"继续往下答。
  后端宕机不该让整轮问答崩掉——这是本项目对 Agent 的硬承诺。
- 契约是二元组，所以**成败只能体现在文本里**（`src/tools/base.py` 的 `_as_tool_result`
  把元组路径恒定为 `success=True`）。不要改成抛异常。
- `question` 可缺省：`src/agent/react_loop.py` 只传 arguments，`ToolRunner` 才会注入
  question（靠 `_accepts_question` 签名探测）。联网类需要它做结果重排，检索类用不上。

为什么不在顶层 import 数据源
----------------------------
1. `src/agent/constants.py`：**顶层绝不能 import**。导入链是
   `src.agent.__init__` → `core` →（try）`rag_tools` → `src.tools.base` → 回头 import
   `src.agent.constants` 时 `src.agent` 只初始化了一半，抛出的 ImportError 会被
   `core.py` 的 `except ImportError` 静默吞掉，**整个工具层悄悄变成空字典**。
   需要常量时在函数内 import。
2. `src.rag` / `src.mcp`：顶层 import 会让整个 `TOOL_EXECUTORS` 构造失败，五个工具一起
   消失；放进函数里则只有那一路降级。`src.mcp` 还没落地；`src.rag` 虽已随 PR #3 合入 main，
   但它模块级依赖成员 D 的 `src.config` / `src.logging_config` / `src.clients`，
   那三个文件还只在 `feat/api-llm` 上——**在这条分支上 import 仍然必然失败**，
   所以降级路径不是历史包袱，而是当前的真实运行状态。

参数名为什么要容错
------------------
工具调用走的是**文本解析**（`src/agent/tool_defense.py` 正则抽 `{"name","arguments"}`），
不是原生 Function Calling；系统提示里只给了工具名和一句话描述、**没有参数名**，
模型只能猜。所以每个执行器都用 `_pick` 接受多个同义键，猜错时回落到默认分支，
而不是直接回一句"缺少必填参数"。
"""

import logging
import math
import threading

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# --- 数值上限 ---

# 与两个数据库客户端的 MAX_LIMIT 对齐（neo4j_client.py:65 / postgresql_client.py:97）。
# 客户端自己也会再收敛一次，这里先挡一道是为了不让明显越界的值走完整个调用链。
MAX_LIMIT = 200

RAG_DEFAULT_TOP_K = 5      # 与 src/agent/generation.py 的 RAG_TOP_K 一致
RAG_MAX_TOP_K = 20
GRAPH_DEFAULT_LIMIT = 20
PG_DEFAULT_LIMIT = 20

WEB_DEFAULT_MAX_RESULTS = 5
WEB_MAX_RESULTS = 10
WEB_DISPLAY_LIMIT = 5      # 组件工作机制.md §13：_fmt_web 只展示前 5 条
PG_DISPLAY_ROWS = 20       # 组件工作机制.md §13：_fmt_pg 只展示前 20 行

# --- 文本截断 ---

# 工具结果会原样进 LLM 上下文，而 MAX_TOOL_ROUNDS=4 意味着最多累积四轮结果。
# 单条记录截得短、整体再兜一道，防止一次检索把上下文预算吃光。
FIELD_TRUNCATE_CHARS = 120
ANSWER_TRUNCATE_CHARS = 300
WEB_SNIPPET_CHARS = 200
PG_FIELD_TRUNCATE_CHARS = 40
MAX_OBSERVATION_CHARS = 4000

# --- 用户可见文案 ---
#
# 三条 RAG 文案与 src/agent/generation.py:145-156 的降级路径保持同一措辞：
# 同一类故障在"工具路径"与"RAG 降级路径"下必须给出一致的说法，否则排障时
# 会被当成两个不同的问题。文案里不含地址、SQL、异常类型（那些只进日志）。

_TEXT_RAG_UNAVAILABLE = "知识库暂时不可用，请稍后重试。"
_TEXT_RAG_FAILED = "知识库检索失败，请稍后重试。"
_TEXT_RAG_EMPTY = "知识库中暂未找到相关内容，请换个问法试试。"

_TEXT_GRAPH_EMPTY = "知识图谱中未找到符合条件的记录。"
_TEXT_GRAPH_DOWN = "知识图谱服务暂时不可用，已跳过该数据源。"
_TEXT_PG_EMPTY = "业务数据库中未找到符合条件的记录。"
_TEXT_PG_DOWN = "业务数据库暂时不可用，已跳过该数据源。"

_TEXT_WEB_NO_KEY = "未配置联网搜索密钥（TAVILY_API_KEY），联网搜索不可用。"
_TEXT_WEB_FAILED = "联网搜索暂时不可用，请稍后重试。"
_TEXT_WEB_EMPTY = "联网没有搜到相关结果。"
# 智谱中文联网搜索（第二路搜索数据源，与 Tavily 互补）
_TEXT_WEB_CN_NO_KEY = "未配置中文联网搜索密钥（ZHIPU_API_KEY），中文联网搜索不可用。"
_TEXT_WEB_CN_FAILED = "中文联网搜索暂时不可用，请稍后重试。"
_TEXT_WEB_CN_EMPTY = "中文联网没有搜到相关结果。"
_TEXT_EXA_UNAVAILABLE = "语义联网搜索（Exa）暂未就绪，已跳过该数据源；如需时效性信息请改用关键词联网搜索。"
_TEXT_EXA_FAILED = "语义联网搜索暂时不可用，请稍后重试。"
_TEXT_EXA_EMPTY = "语义联网搜索没有返回结果。"

_TEXT_MISSING_QUERY = "缺少检索内容，请提供关键词或问题后重试。"


# --------------------------------------------------------------------------
# 懒加载单例
# --------------------------------------------------------------------------
#
# 为什么由本文件持有而不是放在数据库模块里：两个客户端的 __all__ 只导出常量与类，
# 没有模块级实例；而每次调用都新建 Neo4jClient 会重建 driver 与连接池，
# 一轮问答里同一个工具被调多次就会反复建连（Aura 上尤其慢）。
# 与 core._get_rag_pipeline 一样用"探测一次 + 双检锁"，构造失败也缓存：
# 构造只读配置、不发网络包（见 Neo4jClient.__init__ 的说明），没有任何自愈空间。

_client_lock = threading.Lock()
_neo4j_client = None
_neo4j_probed = False
_pg_client = None
_pg_probed = False
_rag_pipeline = None
_rag_probed = False


def _get_neo4j():
    """取图谱客户端单例；配置缺失或驱动初始化失败时返回 None（不抛）。"""
    global _neo4j_client, _neo4j_probed
    if _neo4j_probed:
        return _neo4j_client
    with _client_lock:
        if _neo4j_probed:
            return _neo4j_client
        _neo4j_probed = True
        try:
            from src.database.neo4j_client import Neo4jClient

            _neo4j_client = Neo4jClient()
        except Exception:
            logger.exception("知识图谱客户端不可用，图谱工具将走降级文案")
            _neo4j_client = None
    return _neo4j_client


def _get_postgres():
    """取业务库客户端单例；配置缺失或驱动初始化失败时返回 None（不抛）。"""
    global _pg_client, _pg_probed
    if _pg_probed:
        return _pg_client
    with _client_lock:
        if _pg_probed:
            return _pg_client
        _pg_probed = True
        try:
            from src.database.postgresql_client import PostgresClient

            _pg_client = PostgresClient()
        except Exception:
            logger.exception("业务数据库客户端不可用，数据库工具将走降级文案")
            _pg_client = None
    return _pg_client


def _get_rag_pipeline():
    """取 RAG 流水线单例；成员 B 尚未合入时返回 None（不抛）。

    取的是 `src.rag.pipeline.rag_pipeline` 这个**模块级对象**而不是工厂函数——
    `src/agent/core.py:83` 就是这么取的，两边必须一致，否则 B 按一处实现、这边按另一处调用。
    """
    global _rag_pipeline, _rag_probed
    if _rag_probed:
        return _rag_pipeline
    with _client_lock:
        if _rag_probed:
            return _rag_pipeline
        _rag_probed = True
        try:
            from src.rag.pipeline import rag_pipeline

            _rag_pipeline = rag_pipeline
        except Exception:
            logger.info("RAG 流水线尚未合入，知识库工具将走降级文案")
            _rag_pipeline = None
    return _rag_pipeline


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------


def _pick(arguments, *names, default=""):
    """按候选名依次取第一个非空值（理由见模块 docstring「参数名为什么要容错」）。"""
    if not isinstance(arguments, dict):
        return default
    for name in names:
        value = arguments.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _as_text(value) -> str:
    """把任意值转成可放进文本的字符串；None 与空串统一成空串。"""
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value, default: int, maximum: int, minimum: int = 1) -> int:
    """把 LLM 给的条数收敛成合法整数：非法退回默认值，越界截断。

    与 `postgresql_client._safe_limit` 同一套路——外部数值不可信，字符串或浮点数
    直接塞进 LIMIT / Cypher 会被驱动拒绝，整条查询连"有没有结果"都问不出来。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(number, maximum))


def _truncate(text, limit: int) -> str:
    """按字符截断并补省略号；中文按字符计，不做字节换算。"""
    text = _as_text(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _finalize(text: str) -> str:
    """所有格式化结果的统一出口：去空白 + 总量兜底截断。"""
    return _truncate(text, MAX_OBSERVATION_CHARS)


def _norm_score(value):
    """把分数收敛成 0-1 的 float；缺失 / None / NaN / 非数值一律返回 None。

    为什么不用 `float(value or 0)`：那样缺失会变成 0.0，与"算出来就是 0 分"混为一谈，
    前端与质量标签都无法区分。None 表示"这个来源没有分数"。

    为什么负数也夹到 0：分数会进 done 帧的 sources，前端按 0-1 渲染进度条，
    而精排模型吐出的原始分可能是负数（logits）。夹住范围比让前端拿到越界值安全。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(number, 1.0))


def _fmt_amount(value) -> str:
    """金额显示成"1234567.00"，非数值原样返回（后端可能给 Decimal / None / 字符串）。"""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return _as_text(value) or "—"


def _fmt_rows(rows) -> str:
    """兜底渲染：按行输出"键：值"，不假设任何固定列。

    存在的意义是**新增查询类型时不会渲染成空白**——格式化器是最后一道防线，
    宁可给出一份朴素但可读的列表，也不要因为列名没登记就丢掉整份结果。
    """
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts = [
            f"{_GRAPH_FIELD_LABELS.get(key, key)}：{_truncate(value, FIELD_TRUNCATE_CHARS)}"
            for key, value in row.items()
            if _as_text(value)
        ]
        if parts:
            lines.append("；".join(parts))
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


# --------------------------------------------------------------------------
# 格式化器
# --------------------------------------------------------------------------

_GRAPH_FIELD_LABELS = {
    "label": "类型",
    "name": "名称",
    "tradeFrequency": "交易频次",
    "subject": "标的物",
    "relation": "关系",
    "entity": "实体",
    "supplier": "供应商",
    "amount": "金额",
    "count": "数量",
    "id": "编号",
    "period": "时间",
    "cnt": "数量",
}

# 图谱节点标签与关系类型的展示名：英文枚举进不了给用户的正文
_GRAPH_LABEL_LABELS = {
    "SubjectMatter": "标的物",
    "Purchaser": "采购单位",
    "Supplier": "供应商",
    "Agency": "代理机构",
    "Location": "地区",
    "Transaction": "交易记录",
}
_GRAPH_RELATION_LABELS = {
    "PURCHASED_BY": "采购单位",
    "SUPPLIED_BY": "供应商",
    "AGENT_BY": "代理机构",
    "LOCATED_AT": "所在地",
    "HAS_TRANSACTION": "交易记录",
    "SELF": "标的物本身",
}

# entity_detail 的子列表字段 → 中文小标题（顺序即展示顺序）
_ENTITY_DETAIL_SECTIONS = (
    ("suppliers", "供应商"),
    ("purchasers", "采购单位"),
    ("agencies", "代理机构"),
    ("locations", "所在地"),
    ("transactions", "交易记录"),
)

# 业务库字段 → 中文表头。未登记的字段原样显示（见 _fmt_rows 的说明）
_PG_FIELD_LABELS = {
    "project_id": "项目编号",
    "project_name": "项目名称",
    "subject_matter": "标的物",
    "purchaser": "采购单位",
    "agency": "代理机构",
    "supplier": "供应商",
    "amount": "中标金额",
    "location_city": "所在地",
    "created_at": "时间",
    "bid_time": "中标时间",
    "publish_time": "发布时间",
    "period": "时间",
    "cnt": "数量",
    "avg_amount": "平均金额",
    "match_count": "命中记录数",
    "total_records": "记录总数",
    "total_amount": "中标金额合计",
    "min_bid_time": "最早中标时间",
    "max_bid_time": "最近中标时间",
    "distinct_subject": "标的物种类数",
    "distinct_purchaser": "采购单位数",
}

# 统计类字典的展示顺序（不依赖 dict 的插入顺序，保证同一份数据每次渲染一致）
_PG_STAT_ORDER = (
    "total_records",
    "total_amount",
    "distinct_subject",
    "distinct_purchaser",
    "min_bid_time",
    "max_bid_time",
)


def _fmt_rag(items, limit: int = RAG_DEFAULT_TOP_K) -> str:
    """知识库检索结果 → 编号资料列表（问 / 答 / 相关度）。

    同时把整体匹配度写进文本：`src/agent/react_loop.py` 的 `_validate_result` 只有
    图谱 / 业务库 / 联网三个质量标签分支，没有 RAG 的；在不改动他人文件的前提下，
    由格式化器自己给出提示，用户才知道这批资料是否值得引用。

    ★ 相关度必须用 `relevance`，不是 `score`（实测踩坑）
    --------------------------------------------------
    原先这里用 `score`，并按 `平均 >= 0.5` 打「匹配度良好，可直接引用」的标签。
    但 `score` 是 **RRF 融合分**，只反映候选内排名——项目自己的注释写着：

        「RRF 只反映候选内排名，**任何查询都会有第一名**，
          实测无关问题的 RRF 分（最高 0.500）能高于相关问题（最低 0.081）」

    实测后果：一个明明检索良好的问题，工具返回的文本里写着「相关度：0.10」，
    并按 0.10 < 0.5 打上「匹配度一般，请结合其他信息判断」。

    **模型据此判断"这批资料不可靠"，而实际上资料是对的**——
    这是把质量信号喂反了，比不给信号更糟。
    与此同时，真正可判相关的 `relevance`（同一批数据实测 0.69）被整个丢弃。
    """
    blocks = []
    scores = []
    # 再切一刀 limit：检索侧已按 top_k 截过，但工具参数与检索参数未必同源，
    # 这里兜住"调用方给了个很大的 top_k"的情况
    for item in (items[:limit] if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        question = _as_text(item.get("question"))
        answer = _as_text(item.get("answer"))
        if not question and not answer:
            continue
        # 优先 relevance（稠密余弦，可判相关）；没有时才回落到 score
        relevance = _norm_score(item.get("relevance"))
        score = _norm_score(item.get("score"))
        shown = relevance if relevance is not None else score
        if shown is not None:
            scores.append(shown)
        lines = [f"[{len(blocks) + 1}] 问：{_truncate(question, FIELD_TRUNCATE_CHARS)}"]
        lines.append(f"    答：{_truncate(answer, ANSWER_TRUNCATE_CHARS)}")
        if shown is not None:
            # 标出口径：两种分数量纲不同，不标会让模型拿它们互相比较
            label = "相关度" if relevance is not None else "排名分(不可判相关)"
            lines.append(f"    {label}：{shown:.2f}")
        blocks.append("\n".join(lines))

    if not blocks:
        return _TEXT_RAG_EMPTY

    header = f"知识库检索到 {len(blocks)} 条相关资料："
    # 只有拿到 relevance 时才敢下"匹配度良好"的结论——RRF 分不支持这个判断
    usable = [
        _norm_score(i.get("relevance"))
        for i in (items[:limit] if isinstance(items, list) else [])
        if isinstance(i, dict)
    ]
    usable = [x for x in usable if x is not None]
    if usable:
        # 延迟导入：本模块顶层的注释写明「顶层 import `src.rag` 会让整个
        # TOOL_EXECUTORS 构造失败，五个工具一起消失」，故阈值也按同样口径取。
        try:
            from src.rag.vector_store import MIN_RELEVANCE

            threshold = float(MIN_RELEVANCE)
        except Exception:  # pragma: no cover - src.rag 不可用时仍有合理默认
            threshold = 0.5
        average = sum(usable) / len(usable)
        header += (
            "（匹配度良好，可直接引用）"
            if average >= threshold
            else "（匹配度一般，请结合其他信息判断）"
        )
    elif scores:
        header += "（仅有序分，无法判断相关性）"
    return _finalize(header + "\n" + "\n".join(blocks))


def _fmt_graph_stats(data: dict) -> str:
    """图谱规模统计：`{标签: 数量}` → 中文计数列表。"""
    lines = []
    for label, count in data.items():
        name = _GRAPH_LABEL_LABELS.get(str(label), str(label))
        lines.append(f"- {name}：{count}")
    if not lines:
        return _TEXT_GRAPH_EMPTY
    return _finalize("知识图谱规模：\n" + "\n".join(lines))


def _fmt_graph_detail(data: dict) -> str:
    """标的物完整关系详情：按供应商 / 采购单位 / 代理机构 / 所在地 / 交易记录分段。"""
    subject = _as_text(data.get("subject"))
    blocks = [f"标的物「{subject}」的关联信息："] if subject else ["标的物关联信息："]
    for key, title in _ENTITY_DETAIL_SECTIONS:
        values = data.get(key)
        if not isinstance(values, list) or not values:
            continue
        rendered = []
        for value in values[:MAX_LIMIT]:
            if isinstance(value, dict):
                # 子项也是 dict（如交易记录 {id, amount}），复用 _fmt_rows 的"键：值"渲染
                rendered.append(_truncate("；".join(
                    f"{_GRAPH_FIELD_LABELS.get(key, key)}：{value[key]}"
                    for key in value
                    if _as_text(value[key])
                ), FIELD_TRUNCATE_CHARS))
            else:
                rendered.append(_as_text(value))
        rendered = [item for item in rendered if item]
        if rendered:
            blocks.append(f"【{title}】共 {len(values)} 项：\n" + "\n".join(f"  - {r}" for r in rendered))
    if len(blocks) == 1 and not subject:
        return _TEXT_GRAPH_EMPTY
    return _finalize("\n".join(blocks))


def _fmt_graph(query_type: str, data) -> str:
    """图谱查询结果 → 按查询类型分派的文本。空 / 异常形状一律给可读文案。"""
    if query_type == "graph_stats" and isinstance(data, dict):
        return _fmt_graph_stats(data)
    if query_type == "entity_detail" and isinstance(data, dict):
        return _fmt_graph_detail(data)

    rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    if not rows:
        return _TEXT_GRAPH_EMPTY

    if query_type == "top_traded":
        lines = [
            f"{index}. {_as_text(row.get('name'))}（交易频次 {_as_text(row.get('tradeFrequency')) or '—'}）"
            for index, row in enumerate(rows[:MAX_LIMIT], 1)
        ]
        return _finalize(f"采购频次最高的标的物（前 {len(lines)} 名）：\n" + "\n".join(lines))

    if query_type == "search_entity":
        lines = []
        for row in rows[:MAX_LIMIT]:
            label = _GRAPH_LABEL_LABELS.get(_as_text(row.get("label")), _as_text(row.get("label")) or "实体")
            frequency = _as_text(row.get("tradeFrequency"))
            suffix = f"，交易频次 {frequency}" if frequency else ""
            lines.append(f"- {_as_text(row.get('name'))}（{label}{suffix}）")
        return _finalize("匹配到的实体：\n" + "\n".join(lines))

    if query_type == "items_by_entity":
        lines = []
        for row in rows[:MAX_LIMIT]:
            relation = _GRAPH_RELATION_LABELS.get(_as_text(row.get("relation")), _as_text(row.get("relation")))
            lines.append(f"- {_as_text(row.get('subject'))}（关联方式：{relation}）")
        return _finalize("关联到的标的物：\n" + "\n".join(lines))

    if query_type == "supplier_by_subject":
        lines = [
            f"- {_as_text(row.get('supplier'))}（最高中标金额 {_fmt_amount(row.get('amount'))}）"
            for row in rows[:MAX_LIMIT]
        ]
        return _finalize("该标的物的供应商：\n" + "\n".join(lines))

    body = _fmt_rows(rows[:MAX_LIMIT])
    return _finalize(body) if body else _TEXT_GRAPH_EMPTY


def _fmt_pg(query_type: str, rows) -> str:
    """业务库查询结果 → 中文表头表格（前 20 行，字段截断）。

    两种形状都要认：绝大多数查询返回 `list[dict]`，而 `stats_overview` 返回单个 dict。
    """
    if isinstance(rows, dict):
        if not rows:
            return _TEXT_PG_EMPTY
        ordered = [key for key in _PG_STAT_ORDER if key in rows]
        ordered += [key for key in rows if key not in _PG_STAT_ORDER]
        lines = []
        for key in ordered:
            value = rows[key]
            if key == "total_amount":
                value = _fmt_amount(value)
            lines.append(f"- {_PG_FIELD_LABELS.get(key, key)}：{value}")
        return _finalize("业务数据库整体统计：\n" + "\n".join(lines))

    records = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not records:
        return _TEXT_PG_EMPTY

    # 表头取所有行的键的保序并集：不同查询返回的列不一样，写死一份列清单会在
    # 新增查询时静默丢列
    columns: list[str] = []
    for row in records:
        for key in row:
            if key not in columns:
                columns.append(key)

    shown = records[:PG_DISPLAY_ROWS]
    header = " | ".join(_PG_FIELD_LABELS.get(key, key) for key in columns)
    lines = [header, "-" * max(8, len(header))]
    for row in shown:
        cells = []
        for key in columns:
            value = row.get(key)
            if key == "amount":
                value = _fmt_amount(value)
            cells.append(_truncate(value, PG_FIELD_TRUNCATE_CHARS))
        lines.append(" | ".join(cells))

    title = f"业务数据库查询结果（共 {len(records)} 条"
    title += f"，显示前 {len(shown)} 条）：" if len(records) > len(shown) else "）："
    return _finalize(title + "\n" + "\n".join(lines))


def _fmt_web(answer: str, items, limit: int = WEB_DISPLAY_LIMIT) -> str:
    """联网结果 → 摘要段 + 前 N 条（标题 / 链接 / 摘要）。

    摘要单独标注"AI 生成"：它是搜索服务对网页的二次概括，可能有过时或臆测成分，
    而用户往往直接引用。链接必须原文给出，供核对。
    """
    blocks = []
    answer = _as_text(answer)
    if answer:
        blocks.append("【联网摘要】" + _truncate(answer, ANSWER_TRUNCATE_CHARS))
        blocks.append("（摘要由搜索服务自动生成，引用请核对下方原始链接）")

    records = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    lines = []
    for index, item in enumerate(records[:limit], 1):
        title = _as_text(item.get("title")) or _as_text(item.get("url")) or "（无标题）"
        url = _as_text(item.get("url"))
        content = _truncate(item.get("content"), WEB_SNIPPET_CHARS)
        block = f"[{index}] {title}"
        if url:
            block += f"\n    {url}"
        if content:
            block += f"\n    {content}"
        lines.append(block)

    if lines:
        blocks.append(f"联网搜索到 {len(records)} 条结果" + (f"，显示前 {len(lines)} 条：" if len(records) > len(lines) else "："))
        blocks.append("\n".join(lines))

    if not blocks:
        return _TEXT_WEB_EMPTY
    return _finalize("\n".join(blocks))


# --------------------------------------------------------------------------
# 联网结果重排
# --------------------------------------------------------------------------

# 引擎分低于此值时视为「不可信」，改用余弦分覆盖。
#
# 实测依据：Tavily 对中文问题给出的引擎分常在 0.0001 量级，
# 而同一批结果用余弦补算是 0.59~0.73（差三个数量级）。
# 0.0001 不是「不相关」的表达，而是引擎语言适配不佳的表现；
# 显示给用户会让人以为「这些结果全都不相关」而放弃查看。
_ENGINE_SCORE_DOWNGRADE_BELOW = 0.05


def _backfill_cosine_scores(question: str, records: list, force: bool = False) -> list:
    """给联网结果补算与问题的余弦相似度。

    ★ 字段名有两种形态，必须都认（实测踩坑）
    ----------------------------------------
    本函数在 `_rerank_web_results` 的**早退分支**里被调用，那时 records 还是
    **搜索引擎的原样输出**，不是 `_web_sources` 转换后的形状：

        搜索引擎原样（Tavily / 智谱）:  title / content / score / url
        `_web_sources` 转换后:          question / answer / score / url

    第一版只读 `answer`，而智谱的正文在 `content` 里 → 条件 `r.get("answer")`
    恒为空 → **整批记录被跳过、分数永远补不上**（前端显示"无分数"）。
    而我当时的单元测试用 `answer` 做输入，恰好掩盖了这个 bug。

    故这里 `content` 与 `answer` 都读。

    ★ 为什么默认覆盖引擎分（`force=False` 也有条件覆盖）
    ------------------------------------------------
    两条数据源的分数口径实测差异极大：

        智谱    **完全不返回分数**
        Tavily  返回引擎分，但**对中文问题常低到 0.0001**
                （同一批结果走余弦补算是 0.59~0.73，差三个数量级）

    0.0001 这种值不是"不相关"的表达，而是引擎对该语言不适配的表现。
    把它显示给用户会让人以为"这些结果全都不相关"，从而放弃查看——
    而实际上里面可能有高度相关的内容。

    故策略是：**优先用余弦分**（与本地知识库同量纲、口径一致、可解释），
    余弦算不出来时才保留引擎分作为兜底。

    `force=True` 时忽略已有的 `score` 一律重算；平时只在
    `score` 为空**或小得不可信**（`_ENGINE_SCORE_DOWNGRADE_BELOW`）时才覆盖。

    失败时原样返回：算不出分数不能影响搜索结果本身。
    """
    def _body(record: dict) -> str:
        """取正文：兼容搜索引擎原样字段（content）与转换后字段（answer）。"""
        return _as_text(record.get("content")) or _as_text(record.get("answer"))

    def _needs_score(record: dict) -> bool:
        if not _body(record):
            return False
        current = record.get("score")
        if current is None:
            return True
        if force:
            return True
        # 引擎分小得不可信 → 用余弦覆盖（见 docstring 说明）
        return isinstance(current, (int, float)) and current < _ENGINE_SCORE_DOWNGRADE_BELOW

    targets = [r for r in records if _needs_score(r)]
    if not targets or not _as_text(question):
        return records

    try:
        # 函数内导入：src.services 依赖较多，未就绪时不应让联网工具整体不可用
        from src.services.web_fallback import _score_by_cosine

        scores = _score_by_cosine(question, [_body(r) for r in targets])
    except Exception:
        logger.info("联网结果补算相关度失败，保持原样（不影响搜索）", exc_info=True)
        return records

    for record, score in zip(targets, scores):
        if isinstance(score, (int, float)):
            record["score"] = round(float(score), 4)
            # 内部标记，供 `_web_sources` 归类分数口径（下划线前缀，不进最终 payload）
            record["_score_kind"] = "cosine"
    return records


def _rerank_web_results(question: str, items, top_k=None):
    """用精排模型按用户问题给联网结果打分并降序排列。

    对接 `src/rag/embedder.py` 的 `Reranker.rerank(query, candidates, top_k)`：
    候选必须是 `list[dict]`（它按 `text_key` 从字典里取值配对），返回的是**候选的副本**
    重排后的列表，每项带 `rerank_score`（原始 logit）与 `score`（sigmoid 后的 0-1）。
    它已经做完降序与归一化，这里**不再二次归一化**——min-max 会把"精排认为这批都不太相关"
    强行拉出一个满分 1.0，让用户以为首条很准。

    任何一步失败都**保持原顺序返回**：联网结果本身是可用的（引擎分照旧），
    因为精排器没就绪就把整批结果丢掉，是把可选优化当成了硬依赖。

    返回新列表，不原地改入参——上游的缓存会按引用共享同一份结果，
    就地改分数会污染后续请求（web_search 里已用深拷贝防了一层，这里再明确一次）。
    """
    records = [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    if len(records) < 2:
        # 只有一条时排序没有意义，直接原样返回；此时沿用引擎分（Tavily 的相关度本就是 0-1），
        # 不要自作主张改成 1.0——那会把"引擎认为它不太相关"伪装成满分。
        # 但**没有分数**的要补算，否则前端显示"无分数"（智谱恒无分）
        return _backfill_cosine_scores(question, records)

    question = _as_text(question)
    if not question:
        # 没有可对照的问题就无从重排，保持引擎顺序与引擎分
        # （补算也需要问题，故这里不补）
        return records

    try:
        # 函数内导入：src.rag 依赖成员 D 的 src.config 等模块，未就绪时顶层导入会让
        # 五个工具一起消失。取的是**包级属性**（src/rag/__init__.py 导出）而非子模块——
        # 精排器住在 embedder.py 里，没有 src/rag/reranker.py 这个文件
        from src.rag import reranker
    except Exception:
        logger.info("精排模块尚未就绪，联网结果保持引擎顺序")
        return _backfill_cosine_scores(question, records)

    # 只在模型**已经加载好**时才精排（非阻塞探测，见 Reranker.loaded）。
    #
    # 为什么必须加这道闸：`reranker.rerank()` 会触发懒加载，而该模型权重 1.1GB、
    # 首次加载实测约 80 秒，且是同步阻塞的。原先联网搜索无条件调用它，
    # 于是用户打开「联网搜索」后的第一次提问会**卡 80 秒**，看起来像服务挂了。
    #
    # 取舍依据：联网搜索结果条数少（默认 5~10 条），且搜索引擎本身已按相关性排序，
    # 精排在此处是"锦上添花"而非必需。让可选优化阻塞主流程，是把增益当成了硬依赖。
    # 后端启动时会用后台线程预热该模型（api/server.py 的 _preload_models），
    # 预热完成后本函数会自动开始精排，无需任何配置改动。
    #
    # `getattr(..., True)` 的双重语义很关键，别简化成 `not reranker.loaded`：
    #   - 真实精排器有 loaded 属性 → 未加载时跳过（本次修复的目标）
    #   - 测试注入的替身没有该属性 → 默认 True，照常走精排逻辑（否则测试会被误跳过）
    # 这样闸门只对真实实现生效，不改变既有可注入性契约。
    if not getattr(reranker, "loaded", True):
        # ★ 这条是最常走到的分支（精排模型 1.1GB，未预热完成时一直走这里）。
        # 原样返回会让智谱结果永远没有分数，故必须补算余弦分。
        logger.info("精排模型尚未就绪，联网结果按引擎顺序返回并补算相关度（后台预热中）")
        return _backfill_cosine_scores(question, records)

    # `_idx` 是回填锚点：对方用 `dict(item)` 造副本，对象身份会丢，只能靠自己塞的键认回来。
    # 配对文本给"标题 + 正文"而不是只给标题——招投标网页的标题常常只有项目名，
    # 判断相关性靠的其实是正文里的金额与采购单位
    candidates = [
        {"question": f"{_as_text(item.get('question'))} {_as_text(item.get('answer'))}".strip(), "_idx": index}
        for index, item in enumerate(records)
    ]
    try:
        # top_k 传满：要不要截断由调用方决定，少给几条不该由精排器替我们做主
        reranked = reranker.rerank(question, candidates, top_k=len(candidates))
    except Exception:
        logger.warning("联网结果重排失败，保留引擎顺序", exc_info=True)
        return records

    scored = []
    for entry in reranked if isinstance(reranked, list) else []:
        if not isinstance(entry, dict):
            continue
        index = entry.get("_idx")
        if not isinstance(index, int) or not 0 <= index < len(records):
            continue
        scored.append((index, _norm_score(entry.get("score"))))
    if not scored:
        logger.warning("精排返回的形状无法识别，保留引擎顺序")
        return _backfill_cosine_scores(question, records)

    # 自己再排一次，不吃对方"已降序"的承诺。没有分数的排到最后——它不等于 0 分，
    # 排到"确实不相关"前面会误导；同分按下标稳定排序，保证同一批结果每次顺序一致
    scored.sort(key=lambda pair: (-(pair[1] if pair[1] is not None else -1.0), pair[0]))

    limit = _safe_int(top_k, default=len(scored), maximum=len(scored))
    picked = []
    for index, score in scored[:limit]:
        record = records[index]
        if score is not None:
            record["score"] = round(score, 4)
        picked.append(record)
    return picked


# --------------------------------------------------------------------------
# 来源列表
# --------------------------------------------------------------------------


def _rag_sources(items) -> list[dict]:
    """知识库来源：`{question, answer, score, relevance}`。

    ★ `relevance` 必须带上（实测缺陷修复）
    ------------------------------------
    原先只输出 `{question, answer, score}`，把 `relevance` 丢了。后果是
    **前端与 Agent 的质量判断都拿不到可判相关的分数**：

        `pipeline.search` 实测返回  relevance=0.691（可判相关）
        `_rag_sources` 转换后      relevance 丢失，只剩 score=0.095（RRF 排名分）

    前端「参考来源」面板因此只能显示一个无意义的 0.10，
    `_validate_result` 对 RAG 来源也无从判断质量。

    这是**有意的契约增量**——与 `_web_sources` 输出 `score_kind` 同理：
    少了它，下游无法区分"这条到底相不相关"，只能看到一个"总会有第一名"的排名分。
    """
    sources = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question = _as_text(item.get("question"))
        answer = _as_text(item.get("answer"))
        if not question and not answer:
            continue
        sources.append({
            "question": question,
            "answer": answer,
            "score": _norm_score(item.get("score")),
            "relevance": _norm_score(item.get("relevance")),
        })
    return sources


def _web_sources(items) -> list[dict]:
    """联网来源：`{question, answer, score, url}`。

    `question` 承载标题（跨轮去重键是 `(question, answer, url)`，见 `react_loop._merge_sources`）。

    字段：`question`（标题）/ `answer`（正文）/ `score` / `url` / `score_kind`。
    `score_kind`（`engine` 引擎自带 / `cosine` 本地补算）是**有意的增量**：
    两种口径不可直接横比，前端要据此提示用户。
    """
    sources = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        url = _as_text(item.get("url"))
        title = _as_text(item.get("title"))
        if not url and not title:
            continue
        # 分数口径：余弦补算的与本地知识库同量纲可比；引擎分的口径不同。
        # 这个字段是**有意的增量**：前端要据此提示「不可直接横比」，
        # 不标的话用户会拿 0.0001 的引擎分与 0.71 的本地匹配度比较而得出错误结论。
        kind = item.get("_score_kind") or "engine"
        sources.append(
            {
                "question": title or url,
                "answer": _as_text(item.get("content")),
                "score": _norm_score(item.get("score")),
                "url": url,
                "score_kind": kind,
            }
        )
    return sources


# --------------------------------------------------------------------------
# 执行器
# --------------------------------------------------------------------------


def search_knowledge_base(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """检索招投标问答知识库（Qdrant 混合检索 + 精排）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    # 模型没给 query 时回落到用户原始问题：整句话是合法且通常更好的检索式
    query = _as_text(_pick(arguments, "query", "question", "keyword", "q", "text", "search_query")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []
    top_k = _safe_int(_pick(arguments, "top_k", "topk", "topK", "limit", "k", "num"), RAG_DEFAULT_TOP_K, RAG_MAX_TOP_K)

    pipeline = _get_rag_pipeline()
    if pipeline is None:
        return _TEXT_RAG_UNAVAILABLE, []
    try:
        # 关键字传 top_k：成员 B 的签名是 search(question, top_k=5)
        #
        # ★ 模型只给了 `query` 时，把带历史的检索提示拼进去（多轮指代消解）
        # ------------------------------------------------------------
        # 实测：Agent 的决策 LLM 一般能自己改写追问（"公示期多长" →
        # "依法必须进行招标的项目 招标人应于何时公示中标候选人 公示期多长"），
        # 但**不保证稳定**——另一例它先跑偏到上一个话题，用了三轮工具调用才修正。
        #
        # 拼上"（上文：上一轮问题）"后：改写得好则多几个相关词（无害），
        # 改写跑偏则有上文兜住，少走弯路。检索相似度实测 0.627 → 0.868。
        #
        # 只在 `question` 与 `query` 不同时才需要——两者相同说明模型没给 query，
        # 此时 `question` 本身已是用户原问题，调用方（Agent）会给带历史的版本。
        # ★ 走 **search_cached**，不是 search（实测缺陷修复）
        # -----------------------------------------------
        # `RAGPipeline.search()` **没有缓存**，`search_cached()` 才有。
        # RAG 主链路（`pipeline.chat_events`）一直用的是 `search_cached`，
        # 但**本工具从一开始调的是 `search`**——于是 Agent 路径完全绕过缓存。
        #
        # 后果在实测里看得见：同一轮问答中 `search_knowledge_base` 被重复调用
        # 4~5 次（`docs/36` 记的"打转"），每次都是一次完整的
        # 编码 + 向量检索 + 融合，而模型给的 query 往往是用户的同一句话。
        #
        # 换成 `search_cached` 是**纯收益**：
        #     · 签名兼容（同样接受关键字 `top_k`），返回类型一致
        #     · 未命中时行为与 `search` 完全相同
        #     · 命中时省掉一次编码 + 检索（实测单次约 430ms）
        #
        # 缓存失效仍由调用方负责（`clear_search_cache()`），与主链路口径一致——
        # 见 `search_cached` 的 docstring。
        if question and _as_text(question) != query:
            items = pipeline.search_cached(
                f"{query}（上文：{_as_text(question)}）", top_k=top_k
            )
        else:
            items = pipeline.search_cached(query, top_k=top_k)
    except Exception:
        logger.exception("知识库检索失败：%s", query)
        return _TEXT_RAG_FAILED, []

    records = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    if not records:
        return _TEXT_RAG_EMPTY, []
    return _fmt_rag(records, top_k), _rag_sources(records)


def search_knowledge_graph(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """查询知识图谱（标的物 / 采购单位 / 供应商 / 地区之间的关系与频次）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query_type = _as_text(_pick(arguments, "query_type", "queryType", "type", "template"))
    if query_type not in _GRAPH_QUERY_TYPES:
        # 猜错或缺失时回落到模糊搜索：这是"用户问了个名字"最常见的意图，
        # 比直接回一句"不支持的查询类型"有用得多
        if query_type:
            logger.info("图谱查询类型无法识别（%s），回落到 search_entity", query_type)
        query_type = "search_entity"

    limit = _safe_int(_pick(arguments, "limit", "top_k", "topk", "count", "num"), GRAPH_DEFAULT_LIMIT, MAX_LIMIT)
    entity = _as_text(_pick(arguments, "entity", "keyword", "name", "subject", "query", "q", "text")) or _as_text(question)

    client = _get_neo4j()
    if client is None:
        return _TEXT_GRAPH_DOWN, []

    try:
        if query_type == "graph_stats":
            # ★ 透传结构化图表数据（第三个返回值，可选）
            # 见 `src/agent/chart_data.py`：数字直接取自 `graph_stats()`
            # 的返回，不经格式化文本，因而不受展示格式变化影响。
            from src.agent.chart_data import chart_from_graph_stats

            _stats = client.graph_stats()
            return _fmt_graph(query_type, _stats), [], chart_from_graph_stats(_stats)
        if query_type == "top_traded":
            from src.agent.chart_data import chart_from_top_traded

            _rows = client.top_traded(limit=limit)
            return _fmt_graph(query_type, _rows), [], chart_from_top_traded(_rows)

        if not entity:
            return "请提供要查询的实体名称（标的物 / 采购单位 / 供应商 / 地区）。", []

        if query_type == "search_entity":
            data = client.search_entity(entity, limit=limit)
        elif query_type == "items_by_entity":
            data = client.items_by_entity(entity, limit=limit)
        elif query_type == "entity_detail":
            data = client.entity_detail(entity, limit=limit)
        else:  # supplier_by_subject
            data = client.supplier_by_subject(entity, limit=limit)
    except Exception:
        # 客户端的查询方法自己会吞异常返回空，走到这里说明是更外层的意外
        logger.exception("知识图谱查询失败：%s / %s", query_type, entity)
        return _TEXT_GRAPH_DOWN, []

    # 客户端对"没查到"与"没连上"都返回空，靠 health() 把两种情况分开报——
    # 只在空结果这条路上多花一次探活，正常路径不受影响
    if not data:
        status = client.health()
        return (_TEXT_GRAPH_DOWN if not status.get("ok") else _TEXT_GRAPH_EMPTY), []
    return _fmt_graph(query_type, data), []


def query_database(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """查询业务数据库（中标金额、采购频次、地区分布、时间趋势等统计）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query_type = _as_text(_pick(arguments, "query_type", "queryType", "type", "template"))
    if query_type not in _PG_QUERY_TYPES:
        if query_type:
            logger.info("数据库查询类型无法识别（%s），回落到 search_by_keyword", query_type)
        query_type = "search_by_keyword"

    limit = _safe_int(_pick(arguments, "limit", "top_k", "topk", "count", "num"), PG_DEFAULT_LIMIT, MAX_LIMIT)
    keyword = _as_text(_pick(arguments, "keyword", "query", "q", "text", "subject")) or _as_text(question)

    client = _get_postgres()
    if client is None:
        return _TEXT_PG_DOWN, []

    try:
        data, missing = _dispatch_pg(client, query_type, arguments, keyword, limit)
    except ValueError as exc:
        # 客户端对非法参数（金额写错、group_by 不支持）抛 ValueError，属**调用方**的错误：
        # 把原因回给 LLM，让它改参数重试，而不是报"服务不可用"
        logger.info("业务数据库参数不合法：%s", exc)
        return f"查询参数有误：{exc}", []
    except Exception:
        logger.exception("业务数据库查询失败：%s", query_type)
        return _TEXT_PG_DOWN, []

    if missing:
        return f"缺少参数：{missing}。请补充后重试。", []
    if not data:
        status = client.health()
        return (_TEXT_PG_DOWN if not status.get("ok") else _TEXT_PG_EMPTY), []
    # ★ 透传结构化图表数据（第三个返回值，可选）
    #
    # 为什么在这里产：`data` 是**格式化之前**的原始行，图表数字直接取自它，
    # 因而不受 `_fmt_amount`（金额→"1.2 万元"）与 `_truncate` 的影响。
    # 原实现是从 `_fmt_pg` 的**输出文本**里正则解析 —— 格式一改就静默失效，
    # 而且解析不回来被格式化过的数字。
    #
    # 失败路径仍返回二元组：那里没有数据可画，`_execute_tool` 两种长度都接受。
    from src.agent.chart_data import chart_from_rows

    return _fmt_pg(query_type, data), [], chart_from_rows(data)


def _dispatch_pg(client, query_type: str, arguments: dict, keyword: str, limit: int):
    """按查询类型调用对应方法，返回 `(数据, 缺失的参数说明)`。

    返回二元组而不是直接返回数据：有些查询必须带业务参数（"谁的供应商"），
    缺参数时应该明确告诉 LLM 缺什么，而不是拿空字符串去查、再回一句"没查到"。
    """
    if query_type == "search_by_keyword":
        return (client.search_by_keyword(keyword, limit=limit), "") if keyword else ([], "keyword（关键词）")
    if query_type == "by_purchaser":
        value = _as_text(_pick(arguments, "purchaser", "buyer", "entity", "name", "keyword"))
        return (client.by_purchaser(value, limit=limit), "") if value else ([], "purchaser（采购单位名称）")
    if query_type == "by_supplier":
        value = _as_text(_pick(arguments, "supplier", "vendor", "entity", "name", "keyword"))
        return (client.by_supplier(value, limit=limit), "") if value else ([], "supplier（供应商名称）")
    if query_type == "avg_amount_by_subject":
        value = _as_text(_pick(arguments, "subject", "subject_matter", "entity", "name", "keyword"))
        return (client.avg_amount_by_subject(value, limit=limit), "") if value else ([], "subject（标的物名称）")
    if query_type == "amount_range":
        # 两个边界都缺时用全表默认区间没有意义，明确要求至少给一个
        low = _pick(arguments, "min_amount", "minAmount", "min", default=None)
        high = _pick(arguments, "max_amount", "maxAmount", "max", default=None)
        if low is None and high is None:
            return [], "min_amount 或 max_amount（金额区间，至少给一个）"
        return client.amount_range(min_amount=low, max_amount=high, limit=limit), ""
    if query_type == "trend":
        # 数据里只有 month / year 两种粒度（postgresql_client._TREND_EXPRESSIONS），
        # 别的值会被客户端拒绝，这里先收敛掉
        group_by = _as_text(_pick(arguments, "group_by", "groupBy", "granularity", "period")) or "month"
        if group_by not in ("month", "year"):
            logger.info("时间粒度 %s 不支持，回落到 month", group_by)
            group_by = "month"
        return client.trend(group_by=group_by, limit=limit), ""
    if query_type == "location_dist":
        return client.location_dist(limit=limit), ""
    if query_type == "purchaser_ranking":
        return client.purchaser_ranking(limit=limit), ""
    if query_type == "supplier_ranking":
        return client.supplier_ranking(limit=limit), ""
    if query_type == "top_subject":
        return client.top_subject(limit=limit), ""
    if query_type == "stats_overview":
        return client.stats_overview(), ""
    return [], ""  # 类型已在调用方校验过，兜底返回空


def search_web(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """联网关键词搜索（Tavily），结果经精排后按用户问题排序。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query = _as_text(_pick(arguments, "query", "keyword", "q", "text", "search_query", "question")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []

    # 延迟导入：让"没配密钥"这类纯配置问题不依赖于模块能否导入，
    # 也避免 web_search 的 import 失败连带影响其他四个工具
    try:
        from src import web_search
    except Exception:
        logger.exception("联网搜索模块导入失败")
        return _TEXT_WEB_FAILED, []

    if not web_search.is_configured():
        return _TEXT_WEB_NO_KEY, []

    max_results = _safe_int(
        _pick(arguments, "max_results", "maxResults", "limit", "count", "num"), WEB_DEFAULT_MAX_RESULTS, WEB_MAX_RESULTS
    )
    depth = _as_text(_pick(arguments, "depth", "search_depth")) or "basic"

    try:
        payload = web_search.web_search_client.search(query, max_results=max_results, depth=depth)
    except Exception:
        logger.exception("联网搜索失败：%s", query)
        return _TEXT_WEB_FAILED, []

    items = payload.get("results") if isinstance(payload, dict) else None
    records = _rerank_web_results(_as_text(question) or query, items)
    if not records:
        return _TEXT_WEB_EMPTY, []
    return _fmt_web(payload.get("answer", ""), records), _web_sources(records)


def search_web_cn(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """中文联网搜索（智谱 Web Search），与 `search_web`（Tavily）互补。

    为什么要有第二个联网搜索
    ------------------------
    两者覆盖面不同，不是替代关系：

        search_web（Tavily）  ：英文语料强、带 AI 摘要；但不返回发布时间
        search_web_cn（智谱） ：中文站点覆盖好（省市政府采购网、公共资源交易中心），
                                **每条结果都带 publish_date 与来源媒体**，适合判断时效性

    招投标场景里"这条公告是不是最新的"经常是判断题的关键，而 Tavily 的结果没有日期字段，
    只能靠正文里碰运气，故引入智谱作为补充数据源。

    实现说明：Tavily 走 `src/web_search.py`，智谱走 `src/web_search_zhipu.py`，
    两个模块都只负责取数据；格式化与来源构造在本文件里**共用同一套函数**
    （智谱的返回值已被归一成与 Tavily 同形的字段集，故无需分叉）。
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    query = _as_text(_pick(arguments, "query", "keyword", "q", "text", "search_query", "question")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []

    # 延迟导入：与 search_web 同一理由——纯配置问题不应依赖模块能否导入，
    # 本模块导入失败也不该连带影响其他工具
    try:
        from src import web_search_zhipu
    except Exception:
        logger.exception("智谱联网搜索模块导入失败")
        return _TEXT_WEB_CN_FAILED, []

    if not web_search_zhipu.is_configured():
        return _TEXT_WEB_CN_NO_KEY, []

    max_results = _safe_int(
        _pick(arguments, "max_results", "maxResults", "limit", "count", "num"), WEB_DEFAULT_MAX_RESULTS, WEB_MAX_RESULTS
    )
    engine = _as_text(_pick(arguments, "search_engine", "engine", "depth")) or "search_std"

    try:
        payload = web_search_zhipu.zhipu_search_client.search(
            query, max_results=max_results, engine=engine
        )
    except Exception:
        logger.exception("智谱联网搜索失败：%s", query)
        return _TEXT_WEB_CN_FAILED, []

    items = payload.get("results") if isinstance(payload, dict) else None
    records = _rerank_web_results(_as_text(question) or query, items)
    if not records:
        return _TEXT_WEB_CN_EMPTY, []
    # answer 传空串：智谱不提供 AI 摘要，_fmt_web 会自动跳过摘要段
    return _fmt_web(payload.get("answer", ""), records), _web_sources(records)


def search_exa(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """语义联网搜索（Exa MCP）。

    `src/mcp/web_search_exa.py` 尚未落地（`src/mcp/` 目前只有一个空 `__init__.py`）。
    约定该模块导出模块级单例 `exa_search_client`，提供
    `search(query, num_results=N) -> {"results": [{title, url, content}], "answer": str}`。
    未就绪时本工具返回可读的降级文案，落地后无需改动本文件即可自动挂上。
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    query = _as_text(_pick(arguments, "query", "keyword", "q", "text", "search_query", "question")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []

    try:
        from src.mcp.web_search_exa import exa_search_client
    except Exception:
        logger.info("Exa MCP 客户端尚未就绪，语义联网搜索降级")
        return _TEXT_EXA_UNAVAILABLE, []

    num_results = _safe_int(
        _pick(arguments, "num_results", "numResults", "max_results", "limit", "count"), WEB_DEFAULT_MAX_RESULTS, WEB_MAX_RESULTS
    )
    try:
        payload = exa_search_client.search(query, num_results=num_results)
    except Exception:
        logger.exception("语义联网搜索失败：%s", query)
        return _TEXT_EXA_FAILED, []

    # 兼容两种返回形状：带 results 的 dict，或直接给列表
    items = payload.get("results") if isinstance(payload, dict) else payload
    records = _rerank_web_results(_as_text(question) or query, items)
    if not records:
        return _TEXT_EXA_EMPTY, []
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    return _fmt_web(answer, records), _web_sources(records)


# 支持的路由图谱查询类型：与 Neo4jClient 的 6 个查询方法一一对应。
# 文档 §14 里的 list_entities 在客户端中没有对应方法，不收；supplier_by_subject 则是
# 客户端有、文档漏写的，按"以已合并代码为准"补上。
_GRAPH_QUERY_TYPES = (
    "search_entity",
    "items_by_entity",
    "top_traded",
    "entity_detail",
    "supplier_by_subject",
    "graph_stats",
)

# 业务库查询类型：与 PostgresClient 的 11 个查询方法一一对应。
# 文档 §6 的 6 个旧枚举（filter_by_field / aggregate_stats / top_by_amount ...）与实际方法
# 对不上，映射会有损，故直接以方法名为枚举值，1:1 路由。
_PG_QUERY_TYPES = (
    "search_by_keyword",
    "by_purchaser",
    "by_supplier",
    "amount_range",
    "location_dist",
    "trend",
    "purchaser_ranking",
    "supplier_ranking",
    "avg_amount_by_subject",
    "top_subject",
    "stats_overview",
)


# --------------------------------------------------------------------------
# 工具元信息（给 ToolRunner 的参数校验 / API 层的工具清单用）
# --------------------------------------------------------------------------


#: 追问工具的参数上限。
#:
#: ★ 为什么必须设上限
#: ---------------
#: 模型很容易一次抛出 5~6 个问题（"您是哪个地区？预算多少？采购方式？
#: 项目类型？有没有特殊要求？"），用户面对的是一份问卷而不是一次对话。
#: 3 条以内才像"澄清"，超过就变成"填表"——而填表的转化率极低，
#: 用户往往直接放弃或随便填，反而比不问更糟。
MAX_CLARIFY_QUESTIONS = 3

#: 追问工具在工具文本里的固定标记。
#:
#: 追问的数据（问题与选项）通过 `arguments` 传给 Agent 层，本执行器**不解析它**。
#: 这里返回固定文本的理由：
#:   · 执行器契约是 `(文本, sources, ok)` 三元组（见 `_execute_tool`），
#:     没有回传结构化数据的通道；把数据塞进文本再解析回来会引入
#:     "格式化→解析"两条容易漂移的代码。
#:   · 真的走到这一句，只可能是**模型没按约定传 questions** 的兜底情况，
#:     此时给出可读文案即可（Agent 层会退化为"不追问，继续正常回答"）。
CLARIFY_MARKER = "【需要用户澄清】"


def ask_clarification(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """追问工具的执行器。

    ★ 这个工具**不检索、不产生来源**，也不该走"工具结果回传给模型"的常规路径。
    真正的处理在 `react_loop`：模型调用它时**立刻产出 `clarify` 事件并结束本轮**，
    不再进入检索与生成。见 `react_loop._chat_stream_tools` 里的拦截。

    ★ 为什么执行器仍然要存在（而不是只放个 schema 占位）
    ------------------------------------------------
    工具注册表 `TOOL_EXECUTORS` 由 `_all_specs()` 派生，schema 与执行器同源。
    若只声明 schema 不给执行器，`TOOL_EXECUTORS.get("ask_clarification")` 为 None，
    任何**绕过拦截**的路径（测试直接调 `_execute_tool`、模型在后续轮次调用）
    都会拿到"工具暂不可用（未注册或后端未就绪）"——
    那会让模型以为工具坏了并重试，而不是察觉"这轮不该追问"。
    """
    raw = arguments.get("questions") if isinstance(arguments, dict) else None
    items: list[str] = []
    if isinstance(raw, list):
        for item in raw[:MAX_CLARIFY_QUESTIONS]:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("question") or "").strip()
            else:
                text = ""
            if text:
                items.append(text)
    if not items:
        return (
            f"{CLARIFY_MARKER}（未提供问题，本次不追问，按已有信息回答）",
            [],
        )
    joined = "；".join(items)
    return f"{CLARIFY_MARKER} {joined}", []


# 每个工具的执行器与元信息声明在一处，避免"加了工具忘了注册"或两处参数名漂移
_TOOL_SPECS = (
    {
        # ★ 追问工具放在**第一位**：它是纯交互动作、不产生外部请求，
        # 且决策轮里"该不该先问清楚"应当先于"检索什么"被考虑。
        # （实测：放在列表末尾时，模型更容易直接选检索工具——
        #   工具顺序对选择概率有影响，虽非决定性。）
        "name": "ask_clarification",
        "description": (
            "当用户问题**指向其自身项目/单位的具体情形**、但缺少作出判断所必需的关键信息时，"
            "用本工具向用户追问，**不要靠猜测检索**。"
            "判断口径：\n"
            "  · 适用：问题含『我们/我这个项目/本公司』等自身指向，"
            "或需要地区、采购方式、项目类型、金额区间等前提才能给结论。"
            "例：『我们这个项目可以单一来源采购吗？』（缺项目性质与地区）\n"
            "  · 不适用：通用法规与流程类问题，知识库本身就能回答，**应当直接检索**。"
            "例：『单一来源采购的适用条件是什么？』『投标保证金比例上限是多少？』\n"
            "  · 不适用：问题已足够具体（含明确的地区/方式/对象），直接检索。\n"
            "★ 只问**确实缺失且会改变结论**的信息，最多 3 条，一次问完；"
            "能从问题本身推断的信息不要问。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "maxItems": MAX_CLARIFY_QUESTIONS,
                    "description": "要追问的问题，最多 3 条，按重要性排序",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "问题正文，一句话，具体到能直接回答",
                            },
                            "options": {
                                "type": "array",
                                "maxItems": 5,
                                "description": (
                                    "可选的候选项。当答案可以枚举时给出（如采购方式、地区），"
                                    "前端会渲染成可点选的按钮，比让用户手打更快更准；"
                                    "答案开放时（如金额）省略本字段"
                                ),
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["question"],
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "一句话说明为什么必须问清楚（会展示给用户，便于其理解）",
                },
            },
            "required": ["questions"],
        },
        "executor": ask_clarification,
    },
    {
        "name": "search_knowledge_base",
        "description": "招投标问答知识库（政策法规、办理流程、常见问题），优先使用",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索用的自然语言问题，保留用户原话中的关键实体与限定词",
                },
                "top_k": {"type": "integer", "description": "返回条数，默认 5，上限 20"},
            },
            "required": ["query"],
        },
        "executor": search_knowledge_base,
    },
    {
        "name": "search_knowledge_graph",
        "description": "企业与标的物、采购单位、供应商之间的关系查询（知识图谱）",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": list(_GRAPH_QUERY_TYPES),
                    "description": (
                        "search_entity=模糊查找实体；items_by_entity=某单位/供应商关联的标的物；"
                        "top_traded=采购频次排行；entity_detail=某标的物的完整关系；"
                        "supplier_by_subject=某标的物的供应商与金额；graph_stats=图谱规模统计"
                    ),
                },
                "entity": {
                    "type": "string",
                    "description": "实体名称（标的物 / 采购单位 / 供应商 / 地区）",
                },
                "limit": {"type": "integer", "description": "返回条数，默认 20，上限 200"},
            },
            "required": ["query_type"],
        },
        "executor": search_knowledge_graph,
    },
    {
        "name": "query_database",
        "description": "结构化业务数据查询（中标金额、交易频次等统计）",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": list(_PG_QUERY_TYPES),
                    "description": (
                        "search_by_keyword=关键词模糊搜索；by_purchaser/by_supplier=按采购单位/供应商查；"
                        "amount_range=按中标金额区间；location_dist=地区分布；trend=时间趋势（group_by 取 month/year）；"
                        "purchaser_ranking/supplier_ranking=排行；avg_amount_by_subject=某标的物平均金额；"
                        "top_subject=标的物频次排行；stats_overview=总览统计"
                    ),
                },
                "keyword": {"type": "string", "description": "关键词，search_by_keyword 用"},
                "purchaser": {"type": "string", "description": "采购单位名称，by_purchaser 用"},
                "supplier": {"type": "string", "description": "供应商名称，by_supplier 用"},
                "subject": {"type": "string", "description": "标的物名称，avg_amount_by_subject 用"},
                "min_amount": {"type": "number", "description": "最小中标金额（元），amount_range 用"},
                "max_amount": {"type": "number", "description": "最大中标金额（元），amount_range 用"},
                "group_by": {
                    "type": "string",
                    "enum": ["month", "year"],
                    "description": "trend 的聚合粒度，默认 month",
                },
                "limit": {"type": "integer", "description": "返回条数，默认 20，上限 200"},
            },
            "required": ["query_type"],
        },
        "executor": query_database,
    },
    {
        "name": "search_web",
        "description": "联网关键词搜索（最新公告、时效性强的信息）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "联网搜索关键词，尽量具体并带上时间限定词（如「2025年 招标公告」）",
                },
                "max_results": {"type": "integer", "description": "抓取条数，默认 5，上限 10"},
                "depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "basic 快而省（默认）；仅在 basic 结果不足时用 advanced",
                },
            },
            "required": ["query"],
        },
        "executor": search_web,
    },
    {
        "name": "search_web_cn",
        "description": (
            "中文联网搜索（智谱）：覆盖省市政府采购网、公共资源交易中心等中文站点，"
            "每条结果带【发布时间】与来源媒体，适合查国内招标公告、政策通知与时效性信息"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "中文搜索关键词，尽量带上地区、年份或项目类型"
                        "（如「2025年 安徽 政府采购 招标公告」）"
                    ),
                },
                "max_results": {"type": "integer", "description": "返回条数，默认 5，上限 10"},
                "search_engine": {
                    "type": "string",
                    "enum": ["search_std", "search_pro", "search_pro_sogou", "search_pro_quark"],
                    "description": "搜索档位：search_std 快而省（默认）；结果不足时用 search_pro",
                },
            },
            "required": ["query"],
        },
        "executor": search_web_cn,
    },
    {
        "name": "search_exa",
        "description": "联网语义搜索（开放性、研究类问题）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "语义搜索描述，写完整句子而非关键词，适合开放性、研究类问题",
                },
                "num_results": {"type": "integer", "description": "返回条数，默认 5，上限 10"},
            },
            "required": ["query"],
        },
        "executor": search_exa,
    },
)


def _build_tool(spec: dict) -> BaseTool:
    """按元信息动态产出一个 BaseTool 实例。

    为什么不手写五个子类：五个工具的差别**只在元信息**，执行逻辑都在上面的函数里。
    动态构造让"加工具"变成在 `_TOOL_SPECS` 里加一条，"注册进 TOOL_EXECUTORS"与
    "出现在工具清单里"由同一份数据派生，不可能出现两者不一致。
    """
    executor = spec["executor"]

    class _Tool(BaseTool):
        """由 `_TOOL_SPECS` 派生出的工具定义（参数校验 + 异常兜底由基类提供）。"""

        def run(self, **kwargs) -> ToolResult:
            """调用对应执行器；question 由 ToolRunner 注入，可能不存在。"""
            question = kwargs.pop("question", "")
            text, sources = executor(kwargs, question)
            # 与二元组契约保持一致：失败也走 success=True + 错误文本，
            # 两条调用路径（直接执行 / 经 ToolRunner）行为必须一样
            return ToolResult(success=True, data={"text": text, "sources": sources})

    _Tool.__name__ = f"Tool_{spec['name']}"
    return _Tool(name=spec["name"], description=spec["description"], parameters=spec["parameters"])


# 工具定义清单（供 API 层列出工具、ToolRunner 做参数校验）。
#
# ⚠️ 位置**必须**在 `_all_specs()` 定义之后——模块级赋值是立即求值的，
# 放在前面会 `NameError: _all_specs is not defined`，且是**导入期**报错，
# 会让所有依赖本模块的测试在 collection 阶段就挂掉（实测 13 个测试文件）。
# 见文件末尾的 `TOOLS = ...`。

# 冻结契约：工具名 → 执行器。键名必须与 src/agent/constants.py 的
# BASE_TOOL_NAMES + WEB_TOOL_NAMES 完全一致（src/agent/core.py 与
# src/agent/react_loop.py 各持一份模块级绑定，只读 .get(name) / list(...)）。
# 五个键**无条件注册**：模块缺失时由执行器给出可读的降级文案，
# 不注册的话上层只会说"工具暂不可用（未注册或后端未就绪）"，分不清是哪一路没通。
def _all_specs() -> tuple:
    """全部工具 spec：检索类（本模块）+ 规则判定类（compliance_tools）。

    ★ 为什么要合并而不是各建一个注册表
    ---------------------------------
    
eact_loop._resolve_tool_schemas 只读一处 get_tool_schemas。
    若合规工具另建注册表，就得改 Agent 层的解析逻辑——
    而加工具应该是**只改工具层**的事。

    延迟导入：合规工具依赖 services 层，顶层 import 会让
    services 任何一处缺依赖都把**所有**工具一起打挂（最坏的失败模式，
    与 _TOOL_SPECS 上方注释是同一考虑）。
    """
    specs = list(_TOOL_SPECS)
    try:
        from src.tools.compliance_tools import TOOL_SPEC as _compliance_spec

        specs.append(_compliance_spec)
    except Exception as exc:  # pragma: no cover - 合规工具不可用不影响检索类工具
        import logging

        logging.getLogger(__name__).warning("合规工具未注册（services 层不可用）: %s", exc)
    return tuple(specs)


TOOL_EXECUTORS = {spec["name"]: spec["executor"] for spec in _all_specs()}

# 工具定义清单（供 API 层列出工具、ToolRunner 做参数校验）。
#
# ⚠️ 必须与 `TOOL_EXECUTORS` 用同一个来源（`_all_specs()`），且**位置在它之后**：
# 模块级赋值是立即求值的，放在 `_all_specs` 定义之前会导入期 NameError，
# 让所有依赖本模块的测试在 collection 阶段就挂掉（实测 13 个测试文件）。
TOOLS: list[BaseTool] = [_build_tool(spec) for spec in _all_specs()]


def get_tool_schemas(names: "tuple[str, ...] | list[str] | None" = None) -> list[dict]:
    """产出 OpenAI / DeepSeek function-calling 格式的工具定义。

    用途：Agent 的决策轮需要把「有哪些工具可用」告诉大模型，模型才会返回
    tool_calls；本函数把 `_all_specs()` 这份唯一事实来源翻译成 API 参数，
    避免再手写一份 schema 造成两处漂移。

    `names` 为 None 时返回全部工具；传入白名单时只返回其中的
    （前端"联网搜索"开关关闭时据此只暴露 BASE_TOOL_NAMES）。

    ⚠️ 必须遍历 `_all_specs()` 而不是 `_TOOL_SPECS`。
    实测踩过：注册表已含合规工具，但本函数仍只遍历检索类的 `_TOOL_SPECS`，
    于是**工具注册了、schema 却没暴露**——模型看不到它就永远不会调用；
    而 `TOOL_EXECUTORS` 里查得到，从注册表看一切正常。这类"注册了但没暴露"
    只能靠断言 schema 列表才能发现。
    """
    allowed = set(names) if names is not None else None
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in _all_specs()
        if allowed is None or spec["name"] in allowed
    ]


__all__ = [
    "TOOLS",
    "TOOL_EXECUTORS",
    "get_tool_schemas",
    "search_knowledge_base",
    "search_knowledge_graph",
    "query_database",
    "search_web",
    "search_web_cn",
    "search_exa",
]
