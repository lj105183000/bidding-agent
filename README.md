# 招投标采购 Agent（招投标智能问答系统）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

面向招投标采购场景的智能问答系统：以 **RAG 混合检索（Qdrant）+ 知识图谱（Neo4j）+ 结构化数据库（PostgreSQL）+ 联网搜索（Tavily/Exa）** 四路数据源为底座，由 **ReAct 多轮工具循环**或**可选 LangGraph 多 Agent 编排（Phase 1a/1b）**回答问题，前端 **React 18 + Vite 5** 流式渲染（SSE）。

## 功能概览

**Chat 智能问答**

- SSE 流式输出，逐 token 渲染，支持随时停止
- LLM 服务商切换（DeepSeek / 智谱 / OpenAI 兼容端点）
- 联网搜索、深度思考开关
- 会话持久化与历史会话列表（PostgreSQL）
- 回答来源标注（知识库引用 + 网络来源）、点赞/点踩反馈
- 回答核对：条款号/数字/法规名未在工具结果中出现时标注；可开启定向修复

**多 Agent 编排（Phase 1a/1b，可选）**

- `AGENT_MULTI_AGENT=off|auto|on`：默认 off；auto 只在路由出 **≥2 个 specialist** 时走 LangGraph，on 供评测/A-B。
- `AGENT_MULTI_ROUTER=off|auto|llm`：off=规则路由；auto=规则只给 1 个专家且问题复合时用一次 LLM 兜底；llm=每次先尝试 LLM。5s 超时/非法 JSON/空结果回落规则。
- 多 Agent 阶段强制执行 **50s 总 deadline、8 次共享 LLM 调用预算、LLM/工具各 4 并发槽**；超时/预算耗尽按 partial → fragment → legacy 分级降级。
- `done.multi_agent` 记录 mode/router/activated/decision/fallback/deadline/budget，可区分 off、auto 被拒、已走图。

**灰度可观测**

- `GET /api/metrics` 的 `multi_agent` 段：全量按 mode/decision/router/fallback 聚合，含激活率、LLM 路由率、deadline/budget 率、P50/P95、LLM 调用、tokens、核对/无来源；支持 `?format=prometheus`。
- `GET /api/eval/online` 的 `multi_agent` 段：采样 + 持久化聚合，供 `/eval` 看板展示路由/激活/降级/延迟/成本。

其余页面（Dashboard / 招标信息提取 Extract / 招标文件 / 比价 Compare / 资格 Qualification / 合规 Compliance / 截止 Deadline / 价格对比 PriceCompare / 物资 Material / 分析 Analysis / 图谱 / 评测）均已接入后端 API；仅部分静态示例/展示元数据保留在 `frontend/src/data/`。

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.12 · FastAPI · uvicorn（依赖用 uv 管理） |
| Agent 编排 | 自研 ReAct 循环（≤4 轮工具调用）+ 技能匹配 + 工具防泄漏；可选 LangGraph 0.3 多 Agent 编排（规则/LLM 路由、并行 specialist、deadline/预算/并发硬化） |
| 检索 | Qdrant Named Vectors（dense 512 + jieba BM25 sparse）· Prefetch/RRF 混合检索 · BGE CrossEncoder 精排 · 读操作瞬时错误重试 2 次 |
| LLM | DeepSeek / 智谱 GLM / vLLM·Ollama（OpenAI 兼容）/ 视觉模型 |
| 数据 | Neo4j 知识图谱 · PostgreSQL 会话/反馈/评测历史 · Qdrant 向量库 |
| 联网 | Tavily API · Exa MCP（自研 stdio JSON-RPC 客户端） |
| 前端 | React 18 · TypeScript · Vite 5 · Tailwind CSS · Recharts · React Router |
| 测试 | pytest（3497 passed / 3 skipped / 10 deselected）· Vitest（100 passed / 2 skipped） |

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

想试多 Agent（默认 off，不影响旧路径）：

```bash
AGENT_MULTI_AGENT=auto AGENT_MULTI_ROUTER=auto \
uv run uvicorn api.server:app --host 127.0.0.1 --port 8001
```

## 常用命令

| 命令 | 说明 |
|---|---|
| `python main.py ingest` | 知识库导入：Excel 问/答 → Qdrant（全量重建） |
| `python main.py api` | 启动后端 :8001（开发带 reload） |
| `python main.py dev` | 后台起后端 + 打印前端启动提示 |
| `python main.py pg --dry-run` / `--force` | 预览 / 执行 PostgreSQL 导入；表已存在时无 `--force` 拒绝执行 |
| `python main.py scheduler --list` | 数据中台任务清单（不连后端） |
| `DATA_SCHEDULER_ENABLED=1 python main.py scheduler` | 常驻调度（独立进程，只跑幂等任务） |
| `uv run pytest -q` | 全量后端测试（在 `bidding-agent/` 下执行） |
| `npm run dev` / `npm run build` / `npm run test` | 前端开发 / 构建 / Vitest（在 `frontend/` 下执行） |
| `AGENT_VERIFY_REPAIR=1 python eval/run_all.py --suites agent_multi --repeat 3 --save` | 多 Agent 线 3 次中位基线（web off） |
| `python eval/run_all.py --suites agent --gate --fail-on agent` | 旧 ReAct 线回归门禁（agent_multi 同理） |
| `python eval/run_all.py --suites retrieval_offline --save --save-db` | 评测结果同时写趋势文件与 PostgreSQL（看板历史） |
| `gh workflow run eval-gate.yml -f suites=agent_multi -f save=true` | CI 分线 Eval Gate（建立/更新基线） |
| `curl -s http://127.0.0.1:8001/api/metrics \| jq '.multi_agent.summary'` | 多 Agent 全量灰度指标 |
| `curl -s "http://127.0.0.1:8001/api/eval/online?limit=2000" \| jq '.multi_agent'` | 在线采样聚合 |

## 多 Agent 编排与灰度

### 路由与并行

- `route_question` 先做确定性规则路由（关键词 + 技能 + 复合判断），最多 3 个 specialist。
- `AGENT_MULTI_ROUTER=auto` 只在“规则单专家 + 问题复合”时调用一次 LLM 路由；规则已给 ≥2 个专家时零额外 LLM 成本。
- 图结构：`plan → Send × N（并行 specialist）→ merge → synthesize`；只有 synthesizer 产最终 token。
- 每个 specialist 复用 `_run_tool_loop`，独立 sub-meta/client；单专家 22s 墙钟超时，merge 按 `agent_index` 确定性排序。

### 硬化限额

| 限制 | 行为 |
|---|---|
| 50s 总 deadline | 含 LLM 路由；超时前无可见正文 → 回退旧 ReAct；已有正文 → 返回 partial，不重复输出 |
| 8 次 LLM 预算 | specialist + synthesis + repair/planner 共享；耗尽时 partial/fragment/legacy 降级 |
| LLM 并发 4 | 请求级信号量；max 3 specialist 下通常不成为瓶颈 |
| 工具并发 4 | 跨 specialist 共享；排队超时按工具超时处理 |
| 5s 路由超时 | LLM 路由失败/非法 JSON/空结果一律回落规则 |

### 观测入口

```bash
# 全量（本进程；经代理访问需 LOG_API_KEY + X-API-Key）
curl -s http://127.0.0.1:8001/api/metrics | jq '.multi_agent.summary, .multi_agent.by_mode.auto'
curl -s "http://127.0.0.1:8001/api/metrics?format=prometheus" | grep bidding_multi_agent

# 采样持久化（跨重启）
curl -s "http://127.0.0.1:8001/api/eval/online?limit=2000" | jq '.multi_agent'
```

重点看 `routing_llm_rate`、`deadline_rate`、`budget_rate`、激活率、P95 和 `verify_*`。`/eval` 页面有对应的多 Agent 灰度面板与按模式明细。

## 评测体系

五条线统一由 `bidding-agent/eval/run_all.py` 运行：

| 线 | 内容 |
|---|---|
| `agent` | 旧 ReAct 工具选择准确率 + 延迟 + token/成本 |
| `agent_multi` | 多 Agent 路由/专家工具/综合引用/降级；24 条用例（web off 实跑 22 条、20 条进入核对分母） |
| `retrieval` | 在线检索相关性对比（含精排策略） |
| `retrieval_offline` | 离线固定集（565 文档 / 40 查询 / 显式 qrels），接 PR 门禁 |
| `rag` | RAGAS 风格 LLM-as-Judge（faithfulness / relevancy / context recall） |

- `eval/trend.jsonl` 是门禁事实源；**只有 `--save` 写趋势**，`--gate` 只比较。
- 加 `--save-db` 写入 PostgreSQL `eval_runs`，`python eval/backfill_db.py` 可幂等回填；DB 不可用时看板回退趋势文件。
- `agent_multi` 首轮基线（3 次中位）：P95 14s、路由/工具/引用/核对 100%；观测阈值已标定（`calibrated_on_runs=3`）。
- Eval Gate 按 `ref+suite` 隔离缓存；没有本线缓存时以仓库 trend 为基线。`save=false` 且无基线会让 workflow 直接失败。
- 手动触发默认关闭（`EVAL_TRIGGER_ENABLED=1` 开启），仅支持 `retrieval_offline`；昂贵线仍走 GitHub Actions。

## 数据中台（只读）

`GET /api/data/catalog`、`GET /api/data/catalog/{asset}`、`GET /api/data/lineage` 提供资产目录与静态血缘，`GET /api/data/quality/latest` 返回最近一份数据质量报告。目录来自 `bidding-agent/src/datahub/registry.py`，**不连接数据库**，三库全挂也能返回；质量扫描可手工跑 `python -m src.datahub.quality`。

## 仓库结构

```
├── frontend/                        # ★ 前端（React 18 + Vite 5，:5173）
│   └── src/
│       ├── pages/                   # 14 个业务页面（Chat / Dashboard / KnowledgeGraph / GraphViz
│       │                            #   / Extract / TenderDoc / Compare / Qualification / Compliance
│       │                            #   / Deadline / PriceCompare / Material / Analysis / EvalDashboard）
│       ├── components/              # 通用组件（答案图表 / 力导向图 / 来源列表 / 微条形图 / 移动端导航）
│       ├── lib/                     # api.ts（SSE 网络层）/ evalApi.ts / multiAgentCanary.ts / chartStyles.ts / pdf.ts
│       ├── contexts/                # 全局状态（模型选择，localStorage 持久化）
│       └── hooks/                   # useMediaQuery / useStableCallback
├── bidding-agent/                   # ★ 后端（FastAPI，:8001）
│   ├── main.py                      # CLI 入口：ingest / api / dev / pg / scheduler
│   ├── api/server.py                # 全部端点 + 限流 + lifespan
│   ├── src/
│   │   ├── agent/                   # ReAct / 生成 / 工具防泄漏 / 质量判定 / 答案核对
│   │   │   ├── orchestration/      # 多 Agent：mode/router/llm_router/runtime/graph/specialists/merge
│   │   │   └── llm_budget.py       # 请求级 LLM 预算
│   │   ├── rag/                     # 混合检索 / 精排 / ingest / Qdrant 读重试
│   │   ├── clients/                 # LLM 客户端 + 厂商级降级与分档冷却
│   │   ├── tools/                   # 工具定义与并行执行
│   │   ├── services/                # 合规 / 资格 / 法规 / 比价 / 提取 / 报告等
│   │   ├── database/                # Neo4j / PostgreSQL 客户端
│   │   ├── datahub/                 # 资产目录 / 血缘 / 质量 / 调度器 / 只读门面
│   │   ├── eval_online/             # 在线评测：采样 + 聚合
│   │   ├── eval_offline/            # 离线评测快照与标签
│   │   ├── eval_runs/               # eval_runs / eval_jobs 持久化与对比
│   │   ├── monitoring_multi_agent.py# 多 Agent 灰度指标
│   │   ├── monitoring.py            # 请求/阶段/组件指标
│   │   ├── health.py                # live/ready/deep 健康检查
│   │   ├── log_redaction.py         # 日志出口脱敏
│   │   ├── api_access.py            # /api/logs、/api/metrics 访问控制
│   │   └── langfuse_tracing.py      # 链路追踪（可选）
│   ├── tests/                       # pytest（3497 passed / 3 skipped / 10 deselected）
│   ├── eval/                        # 五线评测 + 趋势/门禁/看板
│   ├── batch/                       # 数据处理脚本
│   └── docs/                        # 开发文档 / 技术栈 / 组件工作机制 / 并行协作
├── docs/superpowers/                # 设计与实施记录（plans / specs）
├── scripts/                         # 启动与运维脚本
├── .github/workflows/               # CI / Eval Gate / 离线检索门禁 / 安全检查
├── CLAUDE.md / AGENTS.md            # AI 编码助手工作指引
└── docker-compose.yml               # 本地编排（另有 docker-compose.cloud.yml）
```

> 说明：`bidding-agent/.github/` 是早期旧工作流副本，实际生效的是仓库根 `.github/workflows/`。

## 文档地图

| 文档 | 定位 |
|---|---|
| `bidding-agent/docs/开发文档.md` | **面向开发者**：环境、数据管道、API 契约、测试、部署、排障（最全，优先查） |
| `bidding-agent/docs/技术栈.md` | 选型与版本锁定、版本兼容性踩坑注记 |
| `bidding-agent/docs/组件工作机制.md` | 20+ 组件的 Mermaid 机制图解 |
| `bidding-agent/docs/并行开发协作.md` | 多 Agent 并行开发：所有权/契约/共享资源/合并流程 |
| `bidding-agent/docs/增加Agent功能.md` | 二期功能清单（历史参考；多 Agent 等已落地） |
| `CONTRIBUTING.md` | 贡献流程：环境、分支/PR、测试、代码风格、评测与安全约定 |
| `CLAUDE.md` / `AGENTS.md` | AI 编码助手工作指引（架构约定、契约红线、测试约定） |

## License

本项目使用 [MIT License](LICENSE)。

## Git 协作

`main` 为稳定分支，**只有通过 PR 才能合入**（CI 全绿 + 至少 1 人 review）。当前主要模块分支：

| 分支 | 模块 |
|---|---|
| `feat/agent-core` | Agent 编排核心 |
| `feat/rag` | 混合检索 |
| `feat/tools-data` | 工具与数据 |
| `feat/api-llm` | API 服务与 LLM 客户端 |
| `feat/frontend` | 前端 |

提交信息格式：`类型(模块): 中文描述`，类型取 `feat / fix / docs / test / chore`。提交前请本地跑通 `uv run pytest -q` 与 `npm run build`。
