# 招投标采购 Agent（招投标智能问答系统）

面向招投标采购场景的智能问答系统：以 **RAG 混合检索（Qdrant）+ 知识图谱（Neo4j）+ 结构化数据库（PostgreSQL）+ 联网搜索（Tavily/Exa）** 四路数据源为底座，由 **ReAct 多轮工具循环**的 Agent 编排回答，前端 **React 18 + Vite 5** 流式渲染（SSE）。

## 功能概览

**Chat 智能问答页**（已接入后端全链路）：

- SSE 流式输出，逐 token 渲染，支持随时停止
- LLM 服务商切换（DeepSeek / 智谱 / OpenAI 兼容端点）
- 联网搜索、深度思考开关
- 会话持久化与历史会话列表（PostgreSQL）
- 回答来源标注（知识库引用 + 网络来源）、点赞/点踩反馈

其余页面（Dashboard / 招标信息提取 Extract / 价格对比 PriceCompare / 物资 Material / 分析 Analysis）暂为 mock 数据展示，属二期功能。

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.12 · FastAPI · uvicorn（依赖用 uv 管理） |
| Agent 编排 | 自研 ReAct 循环（≤4 轮工具调用）+ 技能匹配 + 工具防泄漏 |
| 检索 | Qdrant Named Vectors（dense 512 + jieba BM25 sparse）· Prefetch/RRF 混合检索 · BGE CrossEncoder 精排 |
| LLM | DeepSeek / 智谱 GLM / vLLM·Ollama（OpenAI 兼容）/ 视觉模型 |
| 数据 | Neo4j 知识图谱 · PostgreSQL 会话与反馈 · Qdrant 向量库 |
| 联网 | Tavily API · Exa MCP（自研 stdio JSON-RPC 客户端） |
| 前端 | React 18 · TypeScript · Vite 5 · Redux Toolkit |
| 测试 | pytest（561 例）· Vitest |

## 快速开始

### 准备

```bash
# 后端依赖（Python ≥ 3.12，需先安装 uv）
cd bidding-agent && uv sync

# 前端依赖
cd frontend && npm install
```

### 配置

复制 `bidding-agent/.env.example` 为 `bidding-agent/.env` 并填写。`.env` 永不入库。必需项：

```ini
QDRANT_URL=          # Qdrant 服务地址
QDRANT_API_KEY=      # Qdrant 密钥
DEEPSEEK_API_KEY=    # 默认 LLM 服务商密钥
```

Neo4j / PostgreSQL / Tavily / 智谱等按需填写，完整配置项见 `bidding-agent/docs/开发文档.md` §2.3。国内网络建议设置 `HF_ENDPOINT=https://hf-mirror.com`（嵌入/精排模型首次下载镜像加速）。

### 启动

```bash
# 1. 导入知识库（Excel 问/答 → Qdrant，换数据后必须重新执行）
cd bidding-agent
python main.py ingest

# 2. 启动后端（http://localhost:8001，接口文档 /docs）
python main.py api

# 3. 启动前端（另开终端，http://localhost:5173）
cd frontend
npm run dev
```

生产启动（必须单进程：限流器/缓存/MCP 子进程均为进程内单例）：

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8001
```

## 常用命令

| 命令 | 说明 |
|---|---|
| `python main.py ingest` | 知识库导入：Excel 问/答 → Qdrant（全量重建） |
| `python main.py api` | 启动后端 :8001（开发带 reload） |
| `python main.py dev` | 后台起后端 + 打印前端启动提示 |
| `uv run pytest -q` | 全量后端测试（在 `bidding-agent/` 下执行） |
| `npm run dev` / `npm run build` | 前端开发 / 构建（在 `frontend/` 下执行） |
| `python eval/ragas_eval.py` | RAG 质量评测（LLM-as-Judge） |
| `python eval/agent_eval.py` | Agent 工具选择准确率 + 延迟 |

## 仓库结构

```
├── frontend/                    # ★ 前端（React 18 + Vite 5，:5173）
│   └── src/
│       ├── pages/               # Chat（已接后端）/ Dashboard / Extract / PriceCompare / Material / Analysis
│       └── lib/api.ts           # SSE 网络层，直连后端 :8001
├── bidding-agent/               # ★ 后端（FastAPI，:8001）
│   ├── main.py                  # CLI 入口：ingest / api / dev
│   ├── api/server.py            # 全部端点 + 限流 + lifespan
│   ├── src/
│   │   ├── agent/               # Agent 编排（ReAct 循环 / 生成 / 工具防泄漏 / 技能 / 提示词）
│   │   ├── rag/                 # 混合检索 + 精排 + ingest
│   │   ├── clients/             # LLM 客户端（deepseek / zhipu / openai_compatible / vision）
│   │   ├── tools/               # 工具定义与并行执行（4 线程 / 30s 超时 / 异常隔离）
│   │   ├── database/            # Neo4j / PostgreSQL 客户端
│   │   └── mcp/                 # Exa MCP 客户端
│   ├── tests/                   # pytest 561 例
│   ├── eval/                    # RAG / Agent 评测
│   ├── batch/                   # 数据处理脚本
│   └── docs/                    # 开发文档 / 技术栈 / 组件工作机制
├── CLAUDE.md / AGENTS.md        # AI 编码助手工作指引
└── 招投标采购Agent项目讲解/       # 教学素材（非项目代码）
```

## 文档地图

| 文档 | 定位 |
|---|---|
| `bidding-agent/分配说明.md` | **团队协作入口**：5 人分工、Git 流程、接口契约、联调计划 |
| `bidding-agent/docs/开发文档.md` | 面向开发者：环境、数据管道、API 契约、测试、部署、排障（最全） |
| `bidding-agent/docs/技术栈.md` | 选型与版本锁定、版本兼容性踩坑注记 |
| `bidding-agent/docs/组件工作机制.md` | 20 个组件的 Mermaid 机制图解 |
| `bidding-agent/docs/增加Agent功能.md` | 二期可扩展功能清单 |
| `CLAUDE.md` | AI 编码助手工作指引（架构约定、契约红线、测试约定） |

## Git 协作

5 人小组，`main` 为稳定分支，**只有通过 PR 才能合入**（CI 全绿 + 至少 1 人 review）。每人一条功能分支：

| 分支 | 模块 |
|---|---|
| `feat/agent-core` | Agent 编排核心 |
| `feat/rag` | 混合检索 |
| `feat/tools-data` | 工具与数据 |
| `feat/api-llm` | API 服务与 LLM 客户端 |
| `feat/frontend` | 前端 |

提交信息格式：`类型(模块): 中文描述`，类型取 `feat / fix / docs / test / chore`。详细分工与排期见 `bidding-agent/分配说明.md`。
