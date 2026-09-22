# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ 当前状态：骨架阶段（先读这条）

本仓库目前**只有文档，没有代码**。`bidding-agent/docs/` 下的四份文档描述的是**目标架构**，`src/`、`tests/`、`frontend/`、`main.py`、`conftest.py` 均**尚不存在**。

因此：

- 文档里的文件路径、函数名、端点**不代表可以在磁盘上读到**；改代码前先 `ls` 确认，不要假设。
- 初始化提交（`5273154`）只包含：`pyproject.toml`、`.env.example`、`.gitignore`、`README.md`、`.github/workflows/ci.yml`、`docs/`、`分配说明.md`。
- **CI 目前是红的**：`uv run pytest -q` 在无测试文件时返回退出码 5（no tests ran）。要绿需要先有 `tests/` 与 `conftest.py`。
- 你的任务如果是"实现某模块"，文档已经把契约、验收标准和文件清单写好了——照做即可，不要另起炉灶。

## 仓库布局

```
C:\Users\10518\bidding-agent\
├── CLAUDE.md                    # ← 本文件
├── bidding-agent/               # ★ 真正的项目（5 人协作，代码将在这里落地）
│   ├── README.md                # 面向使用者：功能概览、快速开始
│   ├── 分配说明.md               # 面向团队：5 人分工、Git 流程、阶段排期（协作唯一入口）
│   └── docs/
│       ├── 开发文档.md           # 面向开发者：环境/数据管道/API 契约/测试/部署/排障（最全，优先查）
│       ├── 技术栈.md             # 选型与版本锁定、版本兼容性踩坑注记
│       ├── 组件工作机制.md        # 20 个组件的 Mermaid 机制图解（理解"怎么跑"看这份）
│       └── 增加Agent功能.md       # 二期功能清单
└── 招投标采购Agent项目讲解/        # 教学素材，不是项目代码
    ├── 01_环境管理/ 02_AI编程/     # PDF 教程
    ├── 03_项目参考/               # ⚠️ 与 bidding-agent/docs/ 同源的快照副本，两边会分叉
    └── 04_Agent模式/              # 独立的 Agent 模式教学示例（见下）
```

**两个易踩的坑**：

1. `招投标采购Agent项目讲解/03_项目参考/` 是 `bidding-agent/docs/` 的副本快照，内容相同但**会随时间分叉**。改文档改 `bidding-agent/docs/`，不要改副本。
2. `招投标采购Agent项目讲解/04_Agent模式/01_agent_inference/` 是一个**完全独立的**教学小仓库（8 个自包含的 Agent 模式单文件），有自己的 `CLAUDE.md`、`pyproject.toml`、`uv.lock`。它与招投标 Agent 项目**无任何 import 或依赖关系**，改动时不要跨目录复制代码。

## 常用命令

所有后端命令在 `bidding-agent/` 目录下执行。依赖用 **uv**（Python 3.12），不要用系统 pip。

```bash
uv sync                                  # 安装后端依赖（含 dev 组 pytest）
python main.py ingest                    # 知识库导入：Excel 问/答 → Qdrant（全量重建）
python main.py api                       # 启动后端 http://localhost:8001（开发带 reload）
python main.py dev                       # 后台起后端 + 打印前端启动提示

cd frontend && npm install && npm run dev # 前端 http://localhost:3000

uv run pytest -q                          # 全量测试（目标 131 例）
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

Next.js 14 前端（:3000）→ FastAPI（:8001）→ `BiddingAgent` 编排**四路数据源**回答招投标问题。

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
src/mcp/ + web_search.py   Exa MCP（自研 stdio JSON-RPC 客户端）+ Tavily
src/config.py       Settings dataclass，__post_init__ 校验必需项，缺失直接退出
```

**关键设计（改代码前必须理解）**：

- **单例 + 懒加载 + 双检锁**：`settings / embedder / reranker / vector_store / rag_pipeline / bidding_agent / neo4j_client / postgresql_client / web_search_client / exa_search_client / rate_limiter` 均为模块级单例；模型对象懒加载并加锁（并发首用只加载一份）。
- **工具永不抛异常**：数据库/图谱/联网工具统一返回 `{"success", "data"/"results", "error"}`，格式化器处理所有失败分支。Agent 绝不因后端宕机崩溃——这是硬性约定。
- **异常分级**：完整异常（服务地址、原始错误、SQL）**只进日志**；用户可见提示不泄露内部细节。
- **出站 `trust_env=False`**：所有出站 HTTP 不继承系统代理。这是为了规避"桌面代理软件关闭后连接被路由到失效端口（WinError 10061）"的坑，**不要回退成默认 httpx 客户端**。

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
- 分工明细与阶段排期见 `bidding-agent/分配说明.md`。

## 文档同步义务

改了代码就要同步文档（对照表见 `开发文档.md` §13.4）：

| 改动 | 同步位置 |
|---|---|
| Agent 工具/流程变化 | 本文件、`组件工作机制.md`、`开发文档.md` §6 |
| 新端点/字段/错误码 | `开发文档.md` §5、`README.md` API 表 |
| 新配置项 | `开发文档.md` §2.3、本文件 |
| 包结构/文件移动 | 本文件、`开发文档.md` §1、`组件工作机制.md` 路径引用 |

## 排障速查

完整表见 `bidding-agent/docs/开发文档.md` §12。最高频的几条：

| 症状 | 原因 / 处理 |
|---|---|
| 启动即退出「缺少必需配置」 | `.env` 缺 `QDRANT_URL`/`QDRANT_API_KEY`（或默认 provider 的 key） |
| 首次问答极慢 / 工具 30s 超时 | 嵌入/精排模型冷启动下载，等日志「模型预热完成」；国内设 `HF_ENDPOINT=https://hf-mirror.com` |
| 检索质量骤降 | `data/vocab.json` 与向量数据不一致 → **换数据后必须重新 ingest** |
| 「知识库未就绪」503 | `vocab.json` 缺失 → 先 `python main.py ingest` |
| 前端流式整段一次显示 | SSE 走了缓冲代理。开发时 `lib/api.ts` 绕过 Next 代理直连 :8001；生产用 `NEXT_PUBLIC_API_BASE`，反代需 `proxy_buffering off` |
| `Connection refused (WinError 10061)` | 环境代理残留 → 确认出站客户端仍为 `trust_env=False` |
| `429` 请求过于频繁 | 30 次/60s 滑动窗口；检查是否误开了多 worker |
