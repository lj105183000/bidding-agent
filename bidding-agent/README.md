# 招投标采购 Agent

面向招投标采购场景的智能问答系统：以 **RAG 混合检索（Qdrant）+ 知识图谱（Neo4j）+ 结构化数据库（PostgreSQL）+ 联网搜索（Tavily/Exa）** 四路数据源为底座，由 **ReAct 多轮工具循环**的 Agent 编排回答，前端 **Next.js 14** 流式渲染（SSE）。

## 快速开始



```
\# 1. 安装后端依赖（Python ≥ 3.12，uv 管理）

uv sync

\# 2. 配置 .env（必需项：QDRANT\_URL / QDRANT\_API\_KEY / DEEPSEEK\_API\_KEY）

\#    首次运行嵌入/精排模型请设置 HF\_ENDPOINT=https://hf-mirror.com

\# 3. 导入知识库（Excel 问/答 → Qdrant）

python main.py ingest

\# 4. 启动后端（http://localhost:8001，接口文档 /docs）

python main.py api

\# 5. 启动前端（另开终端，http://localhost:3000）

cd frontend && npm install && npm run dev
```

## 文档地图



| 文档                  | 定位                                  |
| ------------------- | ----------------------------------- |
| `分配说明.md`           | **团队协作入口**：5 人任务分工、Git 流程、接口契约、联调计划 |
| `docs/开发文档.md`      | 面向开发者：环境、数据管道、API 契约、测试、部署、排障       |
| `docs/技术栈.md`       | 技术栈选型与版本锁定                          |
| `docs/组件工作机制.md`    | 20 个组件的 Mermaid 机制图解                |
| `docs/增加Agent功能.md` | 二期可扩展功能清单与实施路线                      |

## 仓库结构（骨架）



```
├── main.py                  # CLI 入口: ingest / api / dev

├── pyproject.toml           # 依赖声明（uv）

├── conftest.py              # pytest 根配置（预置必需 env）

├── .github/workflows/ci.yml # push/PR 自动 uv sync + pytest

├── api/server.py            # FastAPI 全部端点 + 限流 + lifespan

├── src/

│   ├── agent/               # Agent 编排（core/react\_loop/generation/tool\_defense/skills/prompts）

│   ├── rag/                 # 混合检索（pipeline/vector\_store/embedder/ingest）

│   ├── clients/             # LLM 客户端（deepseek/zhipu/openai\_compatible/vision）

│   ├── tools/               # 工具定义与执行（base/rag\_tools）

│   ├── database/            # Neo4j / PostgreSQL 客户端与导入

│   ├── mcp/                 # Exa MCP 客户端

│   └── ...                  # web\_search / config / rate\_limiter / http\_client 等

├── frontend/                # Next.js 14 前端（SSE 流式渲染）

├── tests/                   # pytest 131 例

├── eval/                    # RAG / Agent 评测

└── batch/                   # 数据处理脚本
```

> 当前为团队协作起点（origin）。各模块代码由 5 人按 
>
> `分配说明.md`
>
>  分工实现，通过 GitHub PR 合入 
>
> `main`
>
> 。