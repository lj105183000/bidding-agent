# 评测三线落地设计（ragas_eval / agent_eval / dashboard）

> 日期：2026-09-30
> 背景：`分配说明.md` §8 阶段二验收三条中，端到端联调与测试已完成，仅剩"eval 三条评测线出报告"。文档（`开发文档.md` §10、`技术栈.md` §9、`组件工作机制.md` §20）已承诺三个脚本但均未落地，本文档为实现设计。

## 1. 目标与范围

| 交付物 | 说明 |
|---|---|
| `eval/ragas_eval.py` | RAG 回答质量评测，**基于 ragas 库**（用户指定）：Faithfulness / AnswerRelevancy / LLMContextRecall |
| `eval/agent_eval.py` | Agent 端到端：工具选择准确率 + 延迟 |
| `eval/dashboard.py` | 运行看板：feedback 汇总 + 会话统计 + 缓存统计（受限） |
| `eval/common.py` | 三脚本共享基建（约 100 行） |
| `eval/test_questions.json` | ragas_eval 用例：`[{question, reference}]` |
| `eval/test_cases.json` | agent_eval 用例：`[{question, expected_tool, reference}]` |
| `src/database/postgresql_client.py` | 新增 `feedback_summary(days)` 与 `conversation_stats()` 只读方法（dashboard 依赖） |
| 新增 pytest 测试 | `test_eval_common.py` / `test_ragas_eval.py` / `test_agent_eval.py` + `feedback_summary` 测试 |
| 依赖 | `uv add --dev ragas langchain-openai`，pin `ragas>=0.4,<0.5` |

用例为**手工精选 20~30 条**，覆盖知识库检索类、图谱查询类、数据库统计类、闲聊/拒答类（`expected_tool` 为空字符串）。从知识库问题拟初版，后续团队直接编辑 JSON 增补。

运行环境已确认可真实跑通（.env 有 LLM 密钥、Qdrant 已 ingest、PostgreSQL 有数据），脚本按真实运行设计。

## 2. 总体架构

三个脚本**独立可跑**（`python eval/xxx.py`），共享 `common.py`；`retrieval_eval.py` 保持原样不迁移。

```
eval/
├── common.py            # UTF-8 stdout、load_cases()、run_cases() 容错循环、平均分聚合、表格打印
├── test_questions.json  # ragas_eval 用例
├── test_cases.json      # agent_eval 用例
├── ragas_eval.py        # 评测线 1
├── agent_eval.py        # 评测线 2
├── dashboard.py         # 评测线 3（只读）
└── retrieval_eval.py    # 已有，不动
```

### common.py 提供

- `sys.stdout.reconfigure(encoding="utf-8")`（Windows GBK 控制台，文档 §10 约定）
- `load_cases(path, required_fields)`：JSON 加载 + 字段校验 + `--limit N` 支持
- `run_cases(fn)`：逐题 try/except 容错循环——单题失败计 `error` 且**计入分母**，不中断整轮
- 平均分/准确率/延迟聚合与 CLI 表格式打印

### 数据流（三线互不依赖）

1. **ragas_eval**：`test_questions.json` 逐题 → `rag_pipeline.chat(question)`（非流式，docstring 明示"评测脚本用"）拿 `answer` + `sources` → 组装 ragas 标准列名 `user_input / response / retrieved_contexts / reference` → `evaluate()` 出三指标
2. **agent_eval**：`test_cases.json` 逐题 → `bidding_agent.chat(question)`（非流式、默认参数：无联网/无深度思考，文档 §10 口径）→ `result["tool_name"]` 与 `expected_tool` **单字符串精确比较**（§6.4 口径：多轮后为最后执行完的工具）→ 准确率 + `elapsed_ms` 平均/P95
3. **dashboard**：`PostgresClient.feedback_summary()`（PG，实数据）+ conversations/messages 聚合 + 进程内 `rag_pipeline.cache_info()`（受限，见 §4）

## 3. ragas_eval.py 详细设计（ragas 库集成）

### 3.1 依赖与版本

- `uv add --dev ragas langchain-openai`（dev 组，不进生产依赖）
- pin `ragas>=0.4,<0.5`（v0.4 破坏性大版本，锁死防 API 漂移）
- `uv sync` 后验证依赖树与 FastAPI/pydantic 共存无冲突（真实执行）

### 3.2 裁判 LLM：ChatOpenAI → DeepSeek 兼容端点

- `LangchainLLMWrapper(langchain_openai.ChatOpenAI(model=…, base_url=DeepSeek, api_key=…))`
- **红线适配**：openai SDK 底层 httpx 默认 `trust_env=True` 会继承系统代理（WinError 10061 坑源）。给 `ChatOpenAI` 传 `http_client=httpx.Client(trust_env=False)`，与项目出站策略一致

### 3.3 嵌入：复用项目本地 BGE

- `AnswerRelevancy` 需要 embeddings；DeepSeek 无 embeddings API、OpenAI embeddings 需额外 key
- 写一个实现 langchain `Embeddings` 接口的适配类，内部调 `embedder.encode_query`，零新增外部依赖

### 3.4 指标映射

| 文档口径 | ragas v0.4 指标 | 所需字段 |
|---|---|---|
| Faithfulness 忠实度 | `Faithfulness()` | response + retrieved_contexts |
| Relevancy 相关性 | `AnswerRelevancy()` | user_input + response + embeddings |
| Recall 召回率 | `LLMContextRecall()` | reference + retrieved_contexts |

- 每题一次 `evaluate()`（内部对 Faithfulness 拆论断多次调 LLM）；20~30 题 × ~4 次调用，在 DeepSeek 限流范围内
- `sources` 为空的题跳过评测记 error（知识库未命中，测忠实度无意义）

## 4. dashboard.py 设计

- **反馈汇总（实数据）**：新增 `PostgresClient.feedback_summary(days=7)` 公开只读方法——总反馈数、赞/踩及占比、最近 N 天每日反馈数。参数化 SQL，与客户端现有风格一致
- **会话统计（实数据）**：新增 `PostgresClient.conversation_stats()` 只读方法——conversations 总数、messages 总数。以上两方法是对 `src/database/` 的全部改动
- **缓存统计（受限，如实标注）**：`rag_pipeline` 是进程内单例，缓存与命中账本只存在于本进程内存——dashboard 是独立进程，读不到 API 服务的账本（"空白新账本"问题）。打印当前进程 `cache_info()` 并标注："此为脚本本进程数据，仅作冒烟验证，真实命中率需在长驻 API 进程内观察"。不做跨进程上报（API 不加埋点）
- 输出 CLI 表格（用户已确认，不做 HTML/文件报告）
- PG 连不上：打印用户可读提示，exit 1，不泄露连接串/原始异常（安全红线）

## 5. 错误处理（延续"评测不中断"原则）

| 故障 | 处理 |
|---|---|
| 单题 LLM 调用超时/异常 | `run_cases` 逐题 try/except，计 error 入分母，继续 |
| ragas 内部单题失败 | 外层同上兜底 |
| `.env` 缺 LLM 密钥 | `client.available()` 为 False 启动即退出并提示（同 `retrieval_eval.py`） |
| 知识库未就绪 | `rag_pipeline.ready` 为 False 启动即退出，提示先 ingest |
| PG 连不上 | 可读提示 + exit 1，不泄内部细节 |
| 裁判输出异常（自写版遗留项） | ragas 库内置重试与解析，无需自写 |

脚本骨架沿用 `retrieval_eval.py`：`sys.path.insert` + `setup_logging()` + argparse + `main()` 返回 int。

## 6. 测试（延续文件内局部 fake 类约定，CI 可跑，不碰真实服务）

- `tests/test_eval_common.py`：`load_cases` 校验（缺字段/空文件）、`run_cases` 容错（第 2 题抛异常不影响第 3 题、error 计入分母）、聚合正确性
- `tests/test_ragas_eval.py`：`_FakePipeline` 脚本化 answer/sources；monkeypatch 掉模块内 `evaluate()`（fake 返回预设分数）——验证数据组装列名、sources 为空跳过、指标汇总
- `tests/test_agent_eval.py`：`_FakeAgent` 脚本化 `chat()` 返回不同 `tool_name`/`elapsed_ms`——验证单字符串精确比较、准确率/延迟统计
- `feedback_summary()`：fake connection 验证 SQL 参数化与字段解析（并入现有 PG 测试风格）
- 不测 dashboard 主流程（薄胶水层）

## 7. 文档同步义务（CLAUDE.md §对照表）

| 改动 | 同步位置 |
|---|---|
| eval/ 新增三脚本 + 两用例 JSON + common.py | `开发文档.md` §1 目录树、§10 评测指南（"待落地"改为实际说明）、`技术栈.md` §9 |
| `PostgresClient.feedback_summary()` / `conversation_stats()` | `开发文档.md` §5/§6、`组件工作机制.md` §20 |
| ragas 依赖与版本锁定 | `技术栈.md` §9（含 `trust_env=False` 适配注记） |
| 测试数量变化 | CLAUDE.md"当前状态"一节的 561 例数字 |

## 8. 明确不做（YAGNI）

- HTML 报告 / 图表 / 导出
- 跨进程缓存统计上报
- 迁移 `retrieval_eval.py` 到 common.py
- 自写裁判提示词（已决定改用 ragas 库）

## 9. 已确认的决策记录

| 决策点 | 结论 |
|---|---|
| 范围 | 三个脚本一次做完 |
| 运行环境 | 本地 LLM/Qdrant/PG 全部可真实跑通 |
| 用例 | 手工精选 20~30 条 |
| Judge 实现 | **ragas 库**（用户指定，推翻最初自写提示词方案） |
| dashboard 输出 | CLI 表格 |
| 整体形态 | 方案 A：轻量 common.py + 三脚本自包含 |
