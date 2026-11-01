# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ 当前状态：开发中（先读这条）

后端代码已落地于 `bidding-agent/`（`src/`、`tests/`、`conftest.py`、`main.py` 等）；前端位于**仓库根** `frontend/`（Vite 5 SPA，**已接入后端 API**：Chat 页走真实流式链路，网络层 `src/lib/api.ts` SSE 直连 :8001）。`bidding-agent/docs/` 下的文档中，前端相关章节描述的是旧目标架构（Next.js），以本文档为准。

因此：

- 文档里的文件路径、函数名、端点**不代表可以在磁盘上读到**；改代码前先 `ls` 确认，不要假设。
- 前端 Chat 全链路（SSE 流式、provider 切换、联网/深度思考开关、会话持久化与列表、点赞/点踩、停止）已接通后端；Dashboard/extract/price/material/analysis 四页仍为 mock。
- **CI 是绿的**：`uv run pytest -q`（integration 默认排除）全绿；前端 `npm run build`（tsc 无错）+ `vitest run` 通过。
- Eval Gate 的 CI 配置（Secrets / Postgres service / LLM_PROVIDER / AGENT_VERIFY_REPAIR / timeout）见 `bidding-agent/docs/开发文档.md` §10.1；本地开启核对修复用 `AGENT_VERIFY_REPAIR=1`。
- 评测已扩为四线（agent / retrieval / retrieval_offline / rag），统一入口 `python eval/run_all.py`；另有独立 HTTP 压测 `eval/load_eval.py`。离线检索线接 PR 门禁 `.github/workflows/retrieval-offline-gate.yml`。
- 你的任务如果是"实现某模块"，文档已经把契约、验收标准和文件清单写好了——照做即可，不要另起炉灶。

## 仓库布局

```
C:\Users\10518\bidding-agent\
├── CLAUDE.md                    # ← 本文件
├── bidding-agent/               # ★ 真正的项目（5 人协作，代码将在这里落地）
│   ├── README.md                # 面向使用者：功能概览、快速开始
│   └── docs/
│       ├── 开发文档.md           # 面向开发者：环境/数据管道/API 契约/测试/部署/排障（最全，优先查）
│       ├── 技术栈.md             # 选型与版本锁定、版本兼容性踩坑注记
│       ├── 组件工作机制.md        # 20 个组件的 Mermaid 机制图解（理解"怎么跑"看这份）
│       └── 增加Agent功能.md       # 二期功能清单
└── frontend/                    # 前端（React 18 + Vite 5）
```

> 已从仓库移除：`招投标采购Agent项目讲解/`（教学素材）与 `bidding-agent/分配说明.md`
> （团队分工文档）。它们不随代码分发，故不再出现在结构树里；
> 若手上有旧副本，注意其中的 `03_项目参考/` 是 `bidding-agent/docs/` 的历史快照，
> 改文档一律改 `bidding-agent/docs/`。

## 常用命令

所有后端命令在 `bidding-agent/` 目录下执行。依赖用 **uv**（Python 3.12），不要用系统 pip。

```bash
uv sync                                  # 安装后端依赖（含 dev 组 pytest）
python main.py ingest                    # 知识库导入：Excel 问/答 → Qdrant（全量重建）
python main.py api                       # 启动后端 http://localhost:8001（开发带 reload）
python main.py dev                       # 后台起后端 + 打印前端启动提示
python main.py pg --dry-run              # 预览 PostgreSQL 导入列计划（不连库）
python main.py pg --force                # 确认 DROP 重建 bidding_records
python -m src.datahub.quality            # 跑一次数据质量扫描（打印 JSON，不落库）
python main.py scheduler --list          # 数据中台任务清单（不连后端）
DATA_SCHEDULER_ENABLED=1 python main.py scheduler  # 常驻调度（独立进程）

cd frontend && npm install && npm run dev # 前端 http://localhost:5173（Vite）

uv run pytest -q                          # 全量测试（integration 默认排除）
uv run pytest tests/test_react_loop.py -v # 单文件
uv run pytest -k "sse" -v                 # 关键词筛选

python eval/ragas_eval.py                # RAG 质量评测（LLM-as-Judge）
python eval/agent_eval.py                # Agent 工具选择准确率 + 延迟
```

生产启动（**必须单进程**，限流器/缓存/MCP 子进程均为进程内单例）：

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8001
```

## 目标架构

Vite SPA 前端（:5173，仓库根 `frontend/`）→ FastAPI（:8001）→ `BiddingAgent` 编排**四路数据源**回答招投标问题。

```
src/agent/          Agent 编排核心 —— core.py 主体，拆成 Mixin：
                    react_loop.py（ReAct ≤4 轮）/ generation.py（最终生成）
                    tool_defense.py（工具文本防泄漏）/ skills.py / prompts.py
                    关键：_chat_events 产出结构化事件元组，chat_stream 序列化为 SSE、
                    chat 收集为 dict —— 两条路径共用同一流程，新增逻辑只改一处。
src/rag/            Qdrant Named Vectors（dense 512 + jieba BM25 sparse）
                    + Prefetch/RRF 混合检索 + BGE CrossEncoder 精排
src/tools/          BaseTool + ToolRunner（4 线程并行 / 30s 超时 / 异常隔离）
                    rag_tools.py 定义工具 + TOOL_EXECUTORS 注册表
src/clients/        LLM 客户端：deepseek / zhipu / openai_compatible / vision + llm_factory
src/database/       Neo4j 知识图谱（6 类查询模板）/ PostgreSQL（6 类查询 + 会话/反馈 CRUD）
src/datahub/        轻量数据中台 —— 资产目录 registry.py / 静态血缘 lineage.py /
                    只读门面 service.py / 契约 contract.py（P0）；P1 起承载统一数据访问
src/mcp/ + web_search.py   Exa MCP（自研 stdio JSON-RPC 客户端）+ Tavily
src/config.py       Settings dataclass，__post_init__ 校验必需项，缺失直接退出
```

**关键设计（改代码前必须理解）**：

- **单例 + 懒加载 + 双检锁**：`settings / embedder / reranker / vector_store / rag_pipeline / bidding_agent / neo4j_client / postgresql_client / web_search_client / exa_search_client / rate_limiter` 均为模块级单例；模型对象懒加载并加锁（并发首用只加载一份）。
- **工具永不抛异常**：数据库/图谱/联网工具统一返回 `{"success", "data"/"results", "error"}`，格式化器处理所有失败分支。Agent 绝不因后端宕机崩溃——这是硬性约定。
- **异常分级**：完整异常（服务地址、原始错误、SQL）**只进日志**；用户可见提示不泄露内部细节。
- **出站 `trust_env=False`**：所有出站 HTTP 不继承系统代理。这是为了规避"桌面代理软件关闭后连接被路由到失效端口（WinError 10061）"的坑，**不要回退成默认 httpx 客户端**。
- **数据访问收口**：目录/血缘由 `src/datahub` 静态声明；`analysis / competitor / material` 已迁移到 `src/datahub/pg.py` / `neo4j.py`，新增数据访问不要再写 `psycopg.connect` / `GraphDatabase.driver`。目录端点不连库，三库全挂也应返回 200。
- **调度器独立进程**：`src/datahub/scheduler.py` 不导入 `api.server`、不占 8001；只调度幂等任务（法规增量/质量扫描/目录快照），`ingest`/`pg --force`/`graph --clear` 永远人工执行。

## 不可破坏的契约（红线）

这些被前端徽标、eval 指标或测试依赖，改动必须同步文档并通知相关成员：

| 契约 | 要求 |
|---|---|
| SSE `done` 帧 | 字段 `sources / web_sources / tool_name / elapsed_ms / phase_times`；**`tool_name` 必须是单个字符串**（最后一个执行完的工具名），不是数组 |
| 工具结果 | 统一 `{"success", "data"/"results", "error"}`，失败不抛异常 |
| 公共导入路径 | `from src.agent import ...` / `from src.rag import ...` 的重导出不可破坏 |
| 前端落库 | `PersistedMessage` 形状：`role/content/sources/toolCalled/toolName/image/imageName`；`thinking` 与计时**不落库** |
| `.env` 键名 | 见 `bidding-agent/docs/开发文档.md` §2.3 全表，新增配置项须同步文档 |
| 安全 | 数据库查询参数化 + 字段/聚合白名单；LIKE 通配符转义 |
| `.env` 与密钥 | **永不入库**；密钥不进日志、不进用户可见错误 |

## 代码风格约定

- **注释与日志用中文**，风格是**解释「为什么」而非复述「做什么」**。
- 常量集中在 `src/agent/constants.py` 顶部并附语义说明；magic number 一律提为命名常量。
- 单例命名小写下划线（`rag_pipeline`），类名大驼峰（`RAGPipeline`）。
- 所有外部调用必须有超时（LLM 60s / 本地模型 300s / Qdrant 30s / Neo4j 30s / MCP 60s / 工具总 30s）。
- 共享可变状态必须加锁。
- 全部端点用**同步 `def`**（FastAPI 自动放线程池），慢的 LLM 生成不阻塞事件循环。

## 测试约定（mock 基建）

- **文件内局部 fake 类**（`_FakeLLM` / `ScriptedLLM` 风格），不建全局 fixture。fake 的 `chat_raw` 按队列返回响应或抛异常，可脚本化多轮调用。
- 全部为**同步测试**（生成器用 `list(...)` 拉平），无 asyncio 插件。
- 根 `conftest.py` 预置必需 env，**无 `.env` 也能跑测试**（CI 同理）。
- **monkeypatch 目标易错（拆包后的真实绑定位置）**：
  - `TOOL_EXECUTORS` 有**两处**模块级绑定：`src.agent.core`（runner 构造）与 `src.agent.react_loop`（文本兜底过滤）——**需同时替换**。
  - `_match_skills` 补丁目标为 `src.agent.core._match_skills`。
  - 联网重排的 reranker 补丁目标为 `src.rag.reranker`（函数内导入读包级属性）。
  - 嵌入模型的补丁目标为 `src.rag.embedder.SentenceTransformer / CrossEncoder`。
- fake `_build_context` 必须返回**含 system 消息的列表**（`_clean_for_final` 取 `messages[0]`，空列表越界）。

## Git 协作流程

5 人小组，`main` 是稳定分支。

- **每人一条功能分支**：`feat/agent-core`（A，兼组长/集成）、`feat/rag`（B）、`feat/tools-data`（C）、`feat/api-llm`（D）、`feat/frontend`（E）。
- **只有通过 PR 才能合入 main**，不在本地直接 push main。
- **提交信息**：`类型(模块): 描述`，类型取 `feat / fix / docs / test / chore`，**描述用中文**，说明"做了什么 + 为什么"。
- PR 要求：CI 全绿 + 至少 1 人 review；推荐 **Squash and merge**。
- 冲突处理：`git pull origin main` 本地解决 → commit → push；**绝不 `--force` 覆盖他人分支**。
- 分工明细与阶段排期文档已从仓库移除；如需补充，请放在 `bidding-agent/docs/` 下。

## 文档同步义务

改了代码就要同步文档（对照表见 `开发文档.md` §13.4）：

| 改动 | 同步位置 |
|---|---|
| Agent 工具/流程变化 | 本文件、`组件工作机制.md`、`开发文档.md` §6 |
| 新端点/字段/错误码 | `开发文档.md` §5、`README.md` API 表 |
| 新配置项 | `开发文档.md` §2.3、本文件 |
| 数据中台资产/端点/重建策略 | 本文件、`开发文档.md` §4.4/§5.5、README |
| 包结构/文件移动 | 本文件、`开发文档.md` §1、`组件工作机制.md` 路径引用 |

## 排障速查

完整表见 `bidding-agent/docs/开发文档.md` §12。最高频的几条：

| 症状 | 原因 / 处理 |
|---|---|
| 启动即退出「缺少必需配置」 | `.env` 缺 `QDRANT_URL`/`QDRANT_API_KEY`（或默认 provider 的 key） |
| `python main.py pg` 直接退出「拒绝执行」 | 表已存在且未加 `--force`；先 `--dry-run` 确认列计划，再 `--force` |
| 首次问答极慢 / 工具 30s 超时 | 嵌入/精排模型冷启动下载，等日志「模型预热完成」；国内设 `HF_ENDPOINT=https://hf-mirror.com` |
| 检索质量骤降 | `data/vocab.json` 与向量数据不一致 → **换数据后必须重新 ingest** |
| 「知识库未就绪」503 | `vocab.json` 缺失 → 先 `python main.py ingest` |
| 前端流式整段一次显示（接入 SSE 后） | 响应被代理缓冲。前端为 Vite、无 dev 代理，接入时直连 :8001；生产反代需 `proxy_buffering off` |
| `Connection refused (WinError 10061)` | 环境代理残留 → 确认出站客户端仍为 `trust_env=False` |
| `429` 请求过于频繁 | 30 次/60s 滑动窗口；检查是否误开了多 worker |
