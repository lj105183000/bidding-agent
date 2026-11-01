"""轻量数据中台的资产注册表（唯一事实来源）。

为什么用声明式 Python，而不是启动时扫库
------------------------------------
1. 目录端点要求**三库全挂也能返回**。扫库会让“目录”依赖它描述的系统，
   数据库一宕，连“有哪些表”都查不到，排障时最需要它反而最不可用。
2. owner、刷新方式、重建风险、上下游这些信息本来就不在数据库里，扫不出来。
3. 代价是可能与真实 schema 漂移。这个代价用 `catalog.drift` 质量规则
   （P1）和 `catalog_snapshot`（P2a）兜底，而不是让目录端点实时查库。

命名约定
--------
    pg.<表名>              PostgreSQL 表
    qdrant.<集合名>        Qdrant 集合
    neo4j.<标签/关系名>    Neo4j 节点标签或关系类型
    neo4j.graph            图谱整体（血缘聚合用的伪节点）
    file.<路径语义>        原始/中间/产物文件
    code.<模块语义>        代码内嵌的数据源（如法规语料）
    runtime.<入口>         运行时写入/读取入口
    service.<模块>         业务服务
    tools.<工具>           Agent 工具
    scheduler.<任务>       调度任务
    object_store.<名称>    声明但未接入的对象存储
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetField:
    """一个字段/属性的元数据。"""

    name: str
    type: str
    description: str = ""
    nullable: bool = True
    primary_key: bool = False
    semantic: str = ""  # money / datetime / keyword / vector / sparse_vector
    pii: bool = False


@dataclass(frozen=True)
class DataAsset:
    """一个数据资产。

    `refresh_mode` 用 `+` 连接多个模式（如 `full_rebuild+runtime_write`），
    因为 `bidding_records` 同时存在“导入脚本整表重建”和“运行时抽取写入”两条路径。
    用单个字符串而不是枚举，是为了让 API 直接可读，测试用 `modes` 判断。
    """

    name: str
    kind: str
    layer: str
    description: str
    owner: str = "data-platform"
    fields: tuple[AssetField, ...] = ()
    refresh_mode: str = "manual_only"
    schedulable: bool = False
    recreate_risk: str = "none"  # none / drops_table / drops_collection / clears_graph
    importer: str | None = None
    upstream: tuple[str, ...] = ()
    downstream: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    status: str = "active"  # active / declared / planned
    notes: str = ""

    @property
    def modes(self) -> frozenset[str]:
        return frozenset(part for part in self.refresh_mode.split("+") if part)


#: 合法的资产类型 / 分层。测试用它防住“随手写了个新 kind 但没人处理”。
ASSET_KINDS = frozenset(
    {
        "postgres_table",
        "qdrant_collection",
        "neo4j_label",
        "neo4j_relationship",
        "neo4j_graph",
        "file",
        "code_module",
        "runtime_stream",
        "service",
        "tool",
        "scheduler",
        "object_store",
    }
)

ASSET_LAYERS = frozenset(
    {"source", "raw", "processed", "staging", "serving", "runtime", "ops", "code"}
)


def _f(
    name: str,
    type_: str,
    description: str = "",
    *,
    nullable: bool = True,
    pk: bool = False,
    semantic: str = "",
    pii: bool = False,
) -> AssetField:
    return AssetField(
        name=name,
        type=type_,
        description=description,
        nullable=nullable,
        primary_key=pk,
        semantic=semantic,
        pii=pii,
    )


def _a(
    name: str,
    kind: str,
    layer: str,
    description: str,
    **kwargs: Any,
) -> DataAsset:
    return DataAsset(name=name, kind=kind, layer=layer, description=description, **kwargs)


# ---------------------------------------------------------------------------
# 字段清单（集中定义，资产里引用，避免同一张表在两个资产里写出两套字段）
# ---------------------------------------------------------------------------

_BIDDING_FIELDS: tuple[AssetField, ...] = (
    _f("id", "bigint", "自增主键；每次导入会重建，不是稳定业务主键", nullable=False, pk=True),
    _f("title", "text", "公告标题"),
    _f("category", "text", "货物 / 服务 / 工程；实测语义混杂", semantic="keyword"),
    _f("source", "text", "数据来源（政府采购 / 企业采购 / 招标采购）", semantic="keyword"),
    _f("publish_time", "timestamptz", "发布时间", semantic="datetime"),
    _f("province", "text", "省份", semantic="keyword"),
    _f("location_city", "text", "市区；所在地维度查询使用", semantic="keyword"),
    _f("county", "text", "县城"),
    _f("project_id", "text", "项目编号/招标编号", semantic="keyword"),
    _f("project_name", "text", "项目名称"),
    _f("purchaser", "text", "采购人全称", semantic="keyword"),
    _f("agency", "text", "招标代理机构；空值率约 88%", semantic="keyword"),
    _f("budget", "numeric", "预算金额（元）；空值率约 99.5%", semantic="money"),
    _f("project_address", "text", "项目地址"),
    _f("duration", "text", "服务期/工期"),
    _f("supplier", "text", "中标供应商全称", semantic="keyword"),
    _f("amount", "numeric", "中标金额（元）", semantic="money"),
    _f("bid_time", "timestamptz", "中标时间", semantic="datetime"),
    _f("subject_matter", "text", "标的物；物料归一化的输入", semantic="keyword"),
    _f("entity_subject", "text", "抽取出的实体_主体"),
    _f("entity_location", "text", "抽取出的实体_地名"),
    _f("entity_category", "text", "抽取出的实体_品类"),
    _f("created_at", "timestamptz", "趋势查询的时间列；由发布时间/中标时间派生", semantic="datetime"),
    _f(
        "record_source",
        "text",
        "P2b 新增：xlsx_import / api_extract / unknown / manual；当前库可能还没有此列",
    ),
)

_MESSAGE_FIELDS: tuple[AssetField, ...] = (
    _f("id", "bigint", "消息主键", nullable=False, pk=True),
    _f("conversation_id", "bigint", "所属会话；外键 ON DELETE CASCADE", nullable=False),
    _f("role", "text", "user / assistant", nullable=False),
    _f("content", "text", "消息正文", nullable=False, pii=True),
    _f("sources", "jsonb", "参考来源（工具名、片段、分数）"),
    _f("tool_name", "text", "本轮最后执行完的工具名"),
    _f("trace_id", "text", "P0 可选列：Langfuse trace id"),
    _f("image_analyses", "jsonb", "可选列：图片解析文本"),
    _f("created_at", "timestamptz", "写入时间", nullable=False, semantic="datetime"),
)

_TENDER_FIELDS: tuple[AssetField, ...] = (
    _f("id", "bigint", "文件解析记录主键", nullable=False, pk=True),
    _f("filename", "text", "上传文件名", nullable=False),
    _f("kind", "text", "文件类型"),
    _f("project_name", "text", "项目名称"),
    _f("project_id", "text", "项目编号", semantic="keyword"),
    _f("purchaser", "text", "采购人"),
    _f("agency", "text", "代理机构"),
    _f("budget", "numeric", "预算", semantic="money"),
    _f("max_price", "numeric", "最高限价", semantic="money"),
    _f("deposit", "numeric", "保证金", semantic="money"),
    _f("bid_deadline", "timestamptz", "投标截止时间", semantic="datetime"),
    _f("open_time", "timestamptz", "开标时间", semantic="datetime"),
    _f("evaluation_method", "text", "评标办法"),
    _f("price_weight", "text", "价格权重"),
    _f("pages", "integer", "页数"),
    _f("total_chars", "integer", "总字符数"),
    _f("completeness", "real", "抽取完整度"),
    _f("provider", "text", "抽取使用的 LLM provider"),
    _f("model", "text", "抽取使用的模型"),
    _f("parsed", "jsonb", "完整解析结果", nullable=False),
    _f("warnings", "jsonb", "抽取告警"),
    _f("created_at", "timestamptz", "写入时间", nullable=False, semantic="datetime"),
)


# ---------------------------------------------------------------------------
# 资产清单
# ---------------------------------------------------------------------------

ASSETS: tuple[DataAsset, ...] = (
    # ---- PostgreSQL：服务层（bidding_records 是核心分析表） ----
    _a(
        "pg.bidding_records",
        "postgres_table",
        "serving",
        "招投标中标记录主表；分析、竞对、物料、Agent 数据库工具的共同数据源",
        owner="tools-data",
        fields=_BIDDING_FIELDS,
        refresh_mode="full_rebuild+runtime_write",
        schedulable=False,
        recreate_risk="drops_table",
        importer="src/database/to_postgresql.py",
        upstream=("file.data.processed.subjects_xlsx", "runtime.api.extract"),
        downstream=(
            "service.analysis",
            "service.competitor",
            "service.material",
            "tools.query_database",
        ),
        tags=("core", "analytics"),
        notes="导入脚本当前 DROP+CREATE；P2b 改为 staging 合并并加 record_source。",
    ),
    _a(
        "pg.conversations",
        "postgres_table",
        "serving",
        "会话表；前端会话列表与历史恢复的入口",
        owner="api-llm",
        fields=(
            _f("id", "integer", "会话主键", nullable=False, pk=True),
            _f("session_id", "text", "会话 ID", semantic="keyword"),
            _f("title", "text", "会话标题"),
            _f("created_at", "timestamptz", "创建时间", nullable=False, semantic="datetime"),
        ),
        refresh_mode="runtime_write",
        downstream=("runtime.api.conversations",),
        tags=("conversation",),
    ),
    _a(
        "pg.messages",
        "postgres_table",
        "serving",
        "消息表；历史恢复、对话搜索、记忆压缩的输入",
        owner="api-llm",
        fields=_MESSAGE_FIELDS,
        refresh_mode="runtime_write",
        upstream=("runtime.api.conversations",),
        tags=("conversation", "pii"),
        notes="trace_id / image_analyses 是可选列，老库可能不存在。",
    ),
    _a(
        "pg.feedback",
        "postgres_table",
        "serving",
        "点赞/点踩与反馈汇总的来源",
        owner="api-llm",
        fields=(
            _f("id", "integer", "反馈主键", nullable=False, pk=True),
            _f("conversation_id", "integer", "所属会话"),
            _f("message_id", "integer", "所属消息"),
            _f("rating", "text", "up / down"),
            _f("comment", "text", "用户评论", pii=True),
            _f("created_at", "timestamptz", "创建时间", nullable=False, semantic="datetime"),
        ),
        refresh_mode="runtime_write",
        downstream=("runtime.api.feedback",),
        tags=("feedback",),
    ),
    _a(
        "pg.tender_documents",
        "postgres_table",
        "serving",
        "招标文件解析结果；截止日期、变更追踪、报告生成的输入",
        owner="tools-data",
        fields=_TENDER_FIELDS,
        refresh_mode="runtime_write",
        upstream=("runtime.api.tender_save",),
        tags=("tender",),
    ),
    _a(
        "pg.company_profiles",
        "postgres_table",
        "serving",
        "企业资质档案；资格审查的输入",
        owner="tools-data",
        fields=(
            _f("id", "integer", "档案主键", nullable=False, pk=True),
            _f("name", "text", "企业名称", nullable=False, semantic="keyword"),
            _f("profile", "jsonb", "资质档案（JSON）", nullable=False, pii=True),
            _f("created_at", "timestamptz", "创建时间", nullable=False, semantic="datetime"),
            _f("updated_at", "timestamptz", "更新时间", nullable=False, semantic="datetime"),
        ),
        refresh_mode="runtime_write",
        upstream=("runtime.api.qualification_profiles",),
        tags=("qualification", "pii"),
    ),
    _a(
        "pg.session_traces",
        "postgres_table",
        "ops",
        "会话最近一次 Langfuse trace 的映射；用户反馈接 Score 用",
        owner="api-llm",
        fields=(
            _f("session_id", "text", "会话 ID", nullable=False, pk=True),
            _f("trace_id", "text", "Langfuse trace id", nullable=False),
            _f("updated_at", "timestamptz", "更新时间", nullable=False, semantic="datetime"),
        ),
        refresh_mode="runtime_write",
        upstream=("runtime.agent.trace",),
        tags=("observability",),
        notes="由 langfuse_feedback.remember_session_trace 顺路写入并清理 24h 前的行。",
    ),
    _a(
        "pg.data_quality_reports",
        "postgres_table",
        "ops",
        "数据质量扫描报告（P1 新增）；scheduler 写入，/api/data/quality/latest 读取",
        owner="data-platform",
        fields=(
            _f("id", "bigint", "报告主键", nullable=False, pk=True),
            _f("generated_at", "timestamptz", "生成时间", nullable=False, semantic="datetime"),
            _f("overall", "text", "ok / degraded / fail", nullable=False),
            _f("duration_ms", "integer", "扫描耗时"),
            _f("summary", "jsonb", "规则计数汇总"),
            _f("report", "jsonb", "完整报告", nullable=False),
            _f("source", "text", "manual / scheduler"),
        ),
        refresh_mode="runtime_write",
        upstream=("scheduler.quality_scan",),
        downstream=("runtime.api.data_quality",),
        status="planned",
        tags=("quality",),
    ),
    _a(
        "pg.data_catalog_snapshots",
        "postgres_table",
        "ops",
        "资产目录快照与漂移检测（P2a 新增）",
        owner="data-platform",
        fields=(
            _f("id", "bigint", "快照主键", nullable=False, pk=True),
            _f("generated_at", "timestamptz", "生成时间", nullable=False, semantic="datetime"),
            _f("asset_count", "integer", "资产数量"),
            _f("snapshot", "jsonb", "目录快照"),
            _f("drift_summary", "jsonb", "与实库的差异汇总"),
            _f("source", "text", "manual / scheduler"),
        ),
        refresh_mode="runtime_write",
        upstream=("scheduler.catalog_snapshot",),
        status="planned",
        tags=("catalog",),
    ),
    # ---- Qdrant ----
    _a(
        "qdrant.bid_qa",
        "qdrant_collection",
        "serving",
        "问答知识库；dense(512) + sparse(BM25) 命名向量，RAG 混合检索的索引",
        owner="rag",
        fields=(
            _f("question", "text", "问题原文", semantic="keyword"),
            _f("answer", "text", "答案原文"),
            _f("dense", "vector", "BGE 稠密向量", semantic="vector"),
            _f("sparse", "sparse_vector", "jieba BM25 稀疏向量", semantic="sparse_vector"),
        ),
        refresh_mode="full_rebuild",
        schedulable=False,
        recreate_risk="drops_collection",
        importer="src/rag/ingest.py",
        upstream=("file.data.qa_xlsx",),
        downstream=("rag.pipeline",),
        tags=("core", "rag"),
        notes="物理集合名由 QDRANT_COLLECTION 决定（本项目为 bid_qa）；换数据后必须重新 ingest。",
    ),
    _a(
        "qdrant.bid_regulations",
        "qdrant_collection",
        "serving",
        "法规条文库；UUID5 幂等增量写入，合规检查的检索索引",
        owner="rag",
        fields=(
            _f("law", "text", "法律名称", semantic="keyword"),
            _f("law_short", "text", "法律简称", semantic="keyword"),
            _f("article", "text", "条款号", semantic="keyword"),
            _f("text", "text", "条文正文"),
            _f("topics", "keyword", "主题标签列表"),
            _f("level", "text", "效力层级"),
            _f("effective", "text", "生效状态"),
            _f("dense", "vector", "稠密向量", semantic="vector"),
            _f("sparse", "sparse_vector", "稀疏向量", semantic="sparse_vector"),
        ),
        refresh_mode="idempotent_incremental",
        schedulable=True,
        importer="scripts/ingest_regulations.py",
        upstream=("code.src.constants.regulations", "scheduler.regulations_incremental"),
        downstream=("service.compliance",),
        tags=("rag", "compliance"),
    ),
    # ---- Neo4j ----
    _a(
        "neo4j.graph",
        "neo4j_graph",
        "serving",
        "知识图谱整体；星型 schema，标的物为中心",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        schedulable=False,
        recreate_risk="clears_graph",
        importer="src/database/to_neo4j.py",
        upstream=(
            "file.data.processed.graph_frequency_csv",
            "file.data.processed.graph_transactions_csv",
        ),
        downstream=("tools.search_knowledge_graph", "service.competitor"),
        tags=("core", "graph"),
        notes="main.py graph --clear 会清空图谱，必须人工执行。",
    ),
    _a(
        "neo4j.SubjectMatter",
        "neo4j_label",
        "serving",
        "标的物节点；name 唯一，挂 tradeFrequency 聚合频次",
        owner="tools-data",
        fields=(
            _f("name", "string", "标的物名称", nullable=False, semantic="keyword"),
            _f("tradeFrequency", "integer", "交易频次"),
        ),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.Transaction",
        "neo4j_label",
        "serving",
        "交易节点；id 为 CSV 行号，amount 为交易金额",
        owner="tools-data",
        fields=(
            _f("id", "integer", "交易 id（CSV 行号）", nullable=False, pk=True),
            _f("amount", "float", "交易金额", semantic="money"),
        ),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.Purchaser",
        "neo4j_label",
        "serving",
        "采购人节点",
        owner="tools-data",
        fields=(_f("name", "string", "采购人名称", nullable=False, semantic="keyword"),),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.Supplier",
        "neo4j_label",
        "serving",
        "供应商节点",
        owner="tools-data",
        fields=(_f("name", "string", "供应商名称", nullable=False, semantic="keyword"),),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.Agency",
        "neo4j_label",
        "serving",
        "代理机构节点",
        owner="tools-data",
        fields=(_f("name", "string", "代理机构名称", nullable=False, semantic="keyword"),),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.Location",
        "neo4j_label",
        "serving",
        "所在地节点",
        owner="tools-data",
        fields=(_f("name", "string", "所在地名称", nullable=False, semantic="keyword"),),
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.graph",),
        tags=("graph",),
    ),
    _a(
        "neo4j.HAS_TRANSACTION",
        "neo4j_relationship",
        "serving",
        "标的物 → 交易",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.SubjectMatter", "neo4j.Transaction"),
        tags=("graph",),
    ),
    _a(
        "neo4j.PURCHASED_BY",
        "neo4j_relationship",
        "serving",
        "标的物 → 采购人",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.SubjectMatter", "neo4j.Purchaser"),
        tags=("graph",),
    ),
    _a(
        "neo4j.AGENT_BY",
        "neo4j_relationship",
        "serving",
        "标的物 → 代理机构",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.SubjectMatter", "neo4j.Agency"),
        tags=("graph",),
    ),
    _a(
        "neo4j.SUPPLIED_BY",
        "neo4j_relationship",
        "serving",
        "标的物 → 供应商",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.SubjectMatter", "neo4j.Supplier"),
        tags=("graph",),
    ),
    _a(
        "neo4j.LOCATED_AT",
        "neo4j_relationship",
        "serving",
        "标的物 → 所在地",
        owner="tools-data",
        refresh_mode="idempotent_incremental",
        upstream=("neo4j.SubjectMatter", "neo4j.Location"),
        tags=("graph",),
    ),
    # ---- 文件：batch 五步管道的输入与产物 ----
    _a(
        "file.data.raw.training_xlsx",
        "file",
        "raw",
        "原始招投标训练数据 xlsx（data/raw/ 未入库，需人工放置）",
        owner="tools-data",
        refresh_mode="manual_only",
        downstream=("file.data.processed.dedup_xlsx",),
        tags=("batch",),
    ),
    _a(
        "file.data.processed.dedup_xlsx",
        "file",
        "processed",
        "去重后的 xlsx",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.raw.training_xlsx",),
        downstream=("file.data.batch.requests_jsonl",),
        tags=("batch",),
    ),
    _a(
        "file.data.batch.requests_jsonl",
        "file",
        "processed",
        "提交给智谱 Batch API 的请求 JSONL",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.processed.dedup_xlsx",),
        downstream=("file.data.batch.results_raw_jsonl",),
        tags=("batch",),
    ),
    _a(
        "file.data.batch.results_raw_jsonl",
        "file",
        "processed",
        "Batch API 原始结果 JSONL",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.batch.requests_jsonl",),
        downstream=("file.data.batch.results_processed_jsonl",),
        tags=("batch",),
    ),
    _a(
        "file.data.batch.results_processed_jsonl",
        "file",
        "processed",
        "按 custom_id 排序后的结果 JSONL",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.batch.results_raw_jsonl",),
        downstream=("file.data.processed.subjects_xlsx",),
        tags=("batch",),
    ),
    _a(
        "file.data.processed.subjects_xlsx",
        "file",
        "processed",
        "回填标的物后的 xlsx；PostgreSQL 导入的输入",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.batch.results_processed_jsonl",),
        downstream=(
            "file.data.processed.frequency_xlsx",
            "file.data.processed.graph_frequency_csv",
            "file.data.processed.graph_transactions_csv",
            "pg.bidding_records",
        ),
        tags=("batch",),
    ),
    _a(
        "file.data.processed.frequency_xlsx",
        "file",
        "processed",
        "标的物交易频次 xlsx",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.processed.subjects_xlsx",),
        tags=("batch",),
    ),
    _a(
        "file.data.processed.graph_frequency_csv",
        "file",
        "processed",
        "图谱频次 CSV",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.processed.subjects_xlsx",),
        downstream=("neo4j.graph",),
        tags=("batch", "graph"),
    ),
    _a(
        "file.data.processed.graph_transactions_csv",
        "file",
        "processed",
        "图谱交易 CSV",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("file.data.processed.subjects_xlsx",),
        downstream=("neo4j.graph",),
        tags=("batch", "graph"),
    ),
    _a(
        "file.data.qa_xlsx",
        "file",
        "raw",
        "问答知识库 xlsx（问/答两列）；Qdrant ingest 的输入",
        owner="rag",
        refresh_mode="manual_only",
        downstream=("qdrant.bid_qa", "file.data.vocab_json"),
        tags=("rag",),
    ),
    _a(
        "file.data.vocab_json",
        "file",
        "processed",
        "jieba BM25 词表；必须与向量数据同步，换数据后要重新 ingest",
        owner="rag",
        refresh_mode="manual_only",
        upstream=("file.data.qa_xlsx",),
        downstream=("rag.pipeline",),
        tags=("rag",),
    ),
    # ---- 代码内嵌数据源 ----
    _a(
        "code.src.constants.regulations",
        "code_module",
        "source",
        "法规条文语料（Python 常量）；法规库的权威源",
        owner="rag",
        refresh_mode="manual_only",
        downstream=("qdrant.bid_regulations",),
        tags=("compliance",),
    ),
    # ---- 业务服务 ----
    _a(
        "service.analysis",
        "service",
        "code",
        "集采分析服务；分位数/排行/报价/数据质量",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("pg.bidding_records",),
        downstream=("runtime.api.analysis",),
        tags=("analytics",),
    ),
    _a(
        "service.competitor",
        "service",
        "code",
        "竞争对手分析服务；PG 做统计，Neo4j 做关系补充",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("pg.bidding_records", "neo4j.graph"),
        downstream=("runtime.api.competitor",),
        tags=("analytics", "graph"),
    ),
    _a(
        "service.material",
        "service",
        "code",
        "物料识别服务；subject_matter 归一化与模糊匹配",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("pg.bidding_records",),
        downstream=("runtime.api.material",),
        tags=("analytics",),
    ),
    _a(
        "service.compliance",
        "service",
        "code",
        "合规检查服务；法规检索 + 规则判定",
        owner="tools-data",
        refresh_mode="manual_only",
        upstream=("qdrant.bid_regulations",),
        downstream=("tools.check_tender_compliance", "runtime.api.compliance"),
        tags=("compliance",),
    ),
    _a(
        "rag.pipeline",
        "code_module",
        "code",
        "RAG 混合检索流水线；分类 → Prefetch/RRF → CrossEncoder 精排",
        owner="rag",
        refresh_mode="manual_only",
        upstream=("qdrant.bid_qa", "file.data.vocab_json"),
        downstream=("tools.search_knowledge_base", "runtime.api.chat"),
        tags=("rag",),
    ),
    # ---- Agent 工具 ----
    _a(
        "tools.query_database",
        "tool",
        "code",
        "Agent 数据库查询工具",
        owner="agent-core",
        refresh_mode="manual_only",
        upstream=("pg.bidding_records",),
        tags=("agent",),
    ),
    _a(
        "tools.search_knowledge_graph",
        "tool",
        "code",
        "Agent 图谱查询工具",
        owner="agent-core",
        refresh_mode="manual_only",
        upstream=("neo4j.graph",),
        tags=("agent", "graph"),
    ),
    _a(
        "tools.search_knowledge_base",
        "tool",
        "code",
        "Agent 知识库检索工具",
        owner="agent-core",
        refresh_mode="manual_only",
        upstream=("rag.pipeline",),
        tags=("agent", "rag"),
    ),
    _a(
        "tools.check_tender_compliance",
        "tool",
        "code",
        "Agent 合规检查工具",
        owner="agent-core",
        refresh_mode="manual_only",
        upstream=("service.compliance",),
        tags=("agent", "compliance"),
    ),
    # ---- 运行时入口 ----
    _a(
        "runtime.api.extract",
        "runtime_stream",
        "runtime",
        "标讯抽取端点；save=true 时写 bidding_records",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.bidding_records",),
        tags=("api",),
    ),
    _a(
        "runtime.api.conversations",
        "runtime_stream",
        "runtime",
        "会话 CRUD 端点",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.conversations", "pg.messages"),
        tags=("api",),
    ),
    _a(
        "runtime.api.feedback",
        "runtime_stream",
        "runtime",
        "点赞/点踩端点",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.feedback",),
        tags=("api",),
    ),
    _a(
        "runtime.api.tender_save",
        "runtime_stream",
        "runtime",
        "招标文件保存端点",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.tender_documents",),
        tags=("api",),
    ),
    _a(
        "runtime.api.qualification_profiles",
        "runtime_stream",
        "runtime",
        "企业资质档案端点",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.company_profiles",),
        tags=("api",),
    ),
    _a(
        "runtime.api.analysis",
        "runtime_stream",
        "runtime",
        "/api/analysis/* 只读端点",
        owner="api-llm",
        refresh_mode="manual_only",
        upstream=("service.analysis",),
        tags=("api",),
    ),
    _a(
        "runtime.api.competitor",
        "runtime_stream",
        "runtime",
        "/api/competitor/* 只读端点",
        owner="api-llm",
        refresh_mode="manual_only",
        upstream=("service.competitor",),
        tags=("api",),
    ),
    _a(
        "runtime.api.material",
        "runtime_stream",
        "runtime",
        "/api/material/* 只读端点",
        owner="api-llm",
        refresh_mode="manual_only",
        upstream=("service.material",),
        tags=("api",),
    ),
    _a(
        "runtime.api.compliance",
        "runtime_stream",
        "runtime",
        "/api/compliance/* 只读端点",
        owner="api-llm",
        refresh_mode="manual_only",
        upstream=("service.compliance",),
        tags=("api",),
    ),
    _a(
        "runtime.api.chat",
        "runtime_stream",
        "runtime",
        "问答端点 /api/ask 与 /api/chat*",
        owner="api-llm",
        refresh_mode="manual_only",
        upstream=("rag.pipeline",),
        tags=("api",),
    ),
    _a(
        "runtime.api.data_quality",
        "runtime_stream",
        "runtime",
        "/api/data/quality/latest 只读端点（P1）",
        owner="data-platform",
        refresh_mode="manual_only",
        upstream=("pg.data_quality_reports",),
        status="planned",
        tags=("api", "quality"),
    ),
    _a(
        "runtime.agent.trace",
        "runtime_stream",
        "runtime",
        "Agent 每轮问答的 Langfuse trace 映射",
        owner="api-llm",
        refresh_mode="runtime_write",
        downstream=("pg.session_traces",),
        tags=("observability",),
    ),
    # ---- 调度任务 ----
    _a(
        "scheduler.quality_scan",
        "scheduler",
        "ops",
        "数据质量扫描任务（P1/P2a）",
        owner="data-platform",
        refresh_mode="idempotent_incremental",
        schedulable=True,
        downstream=("pg.data_quality_reports",),
        status="planned",
        tags=("scheduler", "quality"),
    ),
    _a(
        "scheduler.catalog_snapshot",
        "scheduler",
        "ops",
        "资产目录快照与漂移检测任务（P2a）",
        owner="data-platform",
        refresh_mode="idempotent_incremental",
        schedulable=True,
        downstream=("pg.data_catalog_snapshots",),
        status="planned",
        tags=("scheduler", "catalog"),
    ),
    _a(
        "scheduler.regulations_incremental",
        "scheduler",
        "ops",
        "法规增量入库任务（P2a）；UUID5 幂等",
        owner="data-platform",
        refresh_mode="idempotent_incremental",
        schedulable=True,
        downstream=("qdrant.bid_regulations",),
        status="planned",
        tags=("scheduler", "compliance"),
    ),
    # ---- 声明未接入 ----
    _a(
        "object_store.minio",
        "object_store",
        "raw",
        "MinIO 对象存储；docker-compose 已声明，应用代码尚未接入",
        owner="data-platform",
        refresh_mode="manual_only",
        status="declared",
        tags=("object-store",),
        notes="当前上传原图不落盘；接入前不要把它当作已生效的存储。",
    ),
)


# ---------------------------------------------------------------------------
# 索引与查询
# ---------------------------------------------------------------------------

ASSET_INDEX: dict[str, DataAsset] = {}
for _asset in ASSETS:
    if _asset.name in ASSET_INDEX:
        raise RuntimeError(f"数据资产重名：{_asset.name}")
    ASSET_INDEX[_asset.name] = _asset


_SHORT_PREFIXES = (
    "pg.",
    "qdrant.",
    "neo4j.",
    "file.",
    "code.",
    "runtime.",
    "service.",
    "tools.",
    "scheduler.",
    "object_store.",
)


def find_asset(name: str) -> DataAsset | None:
    """按全名或短名找资产。

    短名（`bidding_records`、`bid_qa`、`SubjectMatter`）让 API 使用者不必记前缀；
    完整名优先，避免 `neo4j.graph` 与 `pg.graph` 这类未来冲突被短名规则抢走。
    """
    key = (name or "").strip()
    if not key:
        return None
    exact = ASSET_INDEX.get(key)
    if exact is not None:
        return exact
    for prefix in _SHORT_PREFIXES:
        candidate = ASSET_INDEX.get(f"{prefix}{key}")
        if candidate is not None:
            return candidate
    # Neo4j 标签/关系习惯用大驼峰/大写；短名匹配时允许大小写不敏感。
    # 同时比较全名与短名：只比全名的话 `subjectmatter` 匹配不到
    # `neo4j.SubjectMatter`（前缀不同），而这正是最自然的输入。
    lowered = key.lower()
    for asset_name, asset in ASSET_INDEX.items():
        if asset_name.lower() == lowered:
            return asset
        short_name = asset_name.split(".", 1)[1] if "." in asset_name else asset_name
        if short_name.lower() == lowered:
            return asset
    return None


def list_assets(kind: str | None = None, layer: str | None = None) -> list[DataAsset]:
    """按类型/分层过滤资产；非法条件由调用方（service）先校验。"""
    return [
        asset
        for asset in ASSETS
        if (kind is None or asset.kind == kind) and (layer is None or asset.layer == layer)
    ]


def asset_to_dict(asset: DataAsset, *, include_fields: bool = True) -> dict[str, Any]:
    """把资产转成 API 响应体。

    列表视图不带字段明细：50 多个资产 × 每张表二十多个字段会让目录响应
    膨胀到几百 KB，而列表页只需要“有哪些资产”；点进详情再取字段。
    """
    payload: dict[str, Any] = {
        "name": asset.name,
        "kind": asset.kind,
        "layer": asset.layer,
        "description": asset.description,
        "owner": asset.owner,
        "refresh_mode": asset.refresh_mode,
        "schedulable": asset.schedulable,
        "recreate_risk": asset.recreate_risk,
        "importer": asset.importer,
        "upstream": list(asset.upstream),
        "downstream": list(asset.downstream),
        "tags": list(asset.tags),
        "status": asset.status,
        "notes": asset.notes,
        "field_count": len(asset.fields),
    }
    if include_fields:
        payload["fields"] = [asdict(field) for field in asset.fields]
    return payload
