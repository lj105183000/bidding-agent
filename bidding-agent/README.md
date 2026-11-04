# 招投标采购 Agent（后端）

面向招投标采购场景的智能问答系统：以 **RAG 混合检索（Qdrant）+ 知识图谱（Neo4j）+ 结构化数据库（PostgreSQL）+ 联网搜索（Tavily/Exa）** 四路数据源为底座，由 **ReAct 多轮工具循环**或**可选 LangGraph 多 Agent 编排（Phase 1a/1b）**回答问题；前端位于仓库根 `frontend/`，通过 SSE 流式渲染。

## 快速开始

> 完整说明见仓库根 `README.md`；本文件为后端子目录概览。

```bash
# 1. 安装后端依赖（Python ≥ 3.12，uv 管理）
uv sync

# 2. 配置 .env（必需项：QDRANT_URL / QDRANT_API_KEY / DEEPSEEK_API_KEY）
#    首次运行嵌入/精排模型请设置 HF_ENDPOINT=https://hf-mirror.com

# 3. 导入知识库（Excel 问/答 → Qdrant）
python main.py ingest

# 4. 启动后端（http://localhost:8001，接口文档 /docs）
python main.py api

# 5. 启动前端（另开终端，在仓库根 frontend/ 目录下）
cd ../frontend && npm install && npm run dev
```

## 多 Agent 编排（可选）

```bash
# 默认 off：旧 ReAct 行为完全不变
AGENT_MULTI_AGENT=auto AGENT_MULTI_ROUTER=auto \
  uv run uvicorn api.server:app --host 127.0.0.1 --port 8001
```

- `AGENT_MULTI_AGENT=off|auto|on`：auto 只在路由出 ≥2 个 specialist 时走图；on 供评测/A-B。
- `AGENT_MULTI_ROUTER=off|auto|llm`：auto 只在“规则单专家 + 复合题”时调用一次 LLM；llm 每次先尝试；5s 超时/非法输出回落规则。
- 多 Agent 阶段强制执行 50s 总 deadline、8 次共享 LLM 调用预算、LLM/工具各 4 并发槽；超时/预算耗尽按 partial → fragment → legacy 降级。
- `done.multi_agent` 记录 `mode/configured_mode/mode_reason/activated/decision/route/router/fallback/partial/deadline_exceeded` 等 additive 字段。

观测：

```bash
curl -s http://127.0.0.1:8001/api/metrics | jq '.multi_agent.summary'
curl -s "http://127.0.0.1:8001/api/eval/online?limit=2000" | jq '.multi_agent'
```

## 评测（五线）

统一入口 `python eval/run_all.py`：

| 线 | 说明 |
|---|---|
| `agent` | 旧 ReAct 工具选择准确率 + 延迟 + token/成本 |
| `agent_multi` | 多 Agent 路由/专家工具/综合引用/降级；24 条用例（web off 实跑 22、20 条进入核对分母） |
| `retrieval` | 在线检索相关性对比 |
| `retrieval_offline` | 离线固定集（565 文档 / 40 查询 / 显式 qrels），接 PR 门禁 |
| `rag` | RAGAS 风格 LLM-as-Judge |

```bash
# 多 Agent 3 次中位基线（web off；与 Eval Gate 同口径）
AGENT_VERIFY_REPAIR=1 python eval/run_all.py --suites agent_multi --repeat 3 --save

# 回归门禁：--gate 只比较，--save 才写趋势
python eval/run_all.py --suites agent --gate --fail-on agent

# 写 PostgreSQL 评测历史（看板对比）
python eval/run_all.py --suites retrieval_offline --save --save-db
python eval/backfill_db.py --dry-run    # 预览
python eval/backfill_db.py              # 幂等回填
```

- `eval/trend.jsonl` 是门禁事实源；`--gate` 不再隐式写趋势。
- Eval Gate 按 `ref+suite` 隔离缓存；`save=false` 且无基线时 workflow 直接失败。
- 手动触发默认关闭（`EVAL_TRIGGER_ENABLED=1` 开启），仅 `retrieval_offline`；昂贵线走 GitHub Actions。

## 数据中台（只读）

后端内置轻量数据中台：资产目录 + 静态血缘，目录端点**不连接数据库**，三库全挂也能返回。

```bash
python main.py api
curl http://127.0.0.1:8001/api/data/catalog
curl http://127.0.0.1:8001/api/data/catalog/pg.bidding_records
curl "http://127.0.0.1:8001/api/data/lineage?asset=pg.bidding_records&depth=2"
curl http://127.0.0.1:8001/api/data/quality/latest
```

质量扫描可手工跑（只打印 JSON，不落库）：`python -m src.datahub.quality`。

轻量调度（独立进程，只跑幂等任务）：

```bash
python main.py scheduler --list                  # 任务清单，不连后端
python main.py scheduler --run-once quality      # 手工跑一次质量扫描并落库
DATA_SCHEDULER_ENABLED=1 python main.py scheduler # 常驻调度
```

`bidding_records` 的批量导入默认拒绝覆盖已有表：先 `python main.py pg --dry-run` 预览列计划，确认后 `python main.py pg --force`（会 DROP 重建，删除运行时抽取写入的行）。

## 测试

```bash
uv run pytest -q          # 3497 passed / 3 skipped / 10 deselected
```

集成测试默认排除（`-m integration`）；CI 另跑前端 build + Vitest 与离线检索门禁。

## 目录骨架

```
├── main.py                  # CLI 入口: ingest / api / dev / pg / scheduler
├── pyproject.toml           # 依赖声明（uv）
├── conftest.py              # pytest 根配置（预置必需 env）
├── api/server.py            # FastAPI 全部端点 + 限流 + lifespan
├── src/
│   ├── agent/               # ReAct / 生成 / 工具防泄漏 / 核对 / skills
│   │   ├── orchestration/   # 多 Agent: mode/router/llm_router/runtime/graph/specialists/merge
│   │   └── llm_budget.py    # 请求级 LLM 预算
│   ├── rag/                 # 混合检索（pipeline/vector_store/embedder/ingest）
│   ├── clients/             # LLM 客户端（deepseek/zhipu/openai_compatible/vision）
│   ├── tools/               # 工具定义与并行执行
│   ├── services/            # 合规/资格/价格/提取/报告等业务服务
│   ├── database/            # Neo4j / PostgreSQL 客户端与导入
│   ├── datahub/             # 资产目录 / 血缘 / 质量 / 调度器 / 只读门面
│   ├── eval_online/         # 在线评测采样与聚合
│   ├── eval_offline/        # 离线评测快照与标签
│   ├── eval_runs/           # eval_runs / eval_jobs 持久化与对比
│   ├── monitoring*.py       # 请求/阶段/多 Agent 灰度指标
│   ├── mcp/                 # Exa MCP 客户端
│   └── ...                  # web_search / config / rate_limiter / http_client 等
├── tests/                   # pytest（3497 passed / 3 skipped / 10 deselected）
├── eval/                    # 五线评测 + 趋势/门禁/看板
└── batch/                   # 数据处理脚本
```

## 文档地图

| 文档 | 定位 |
|---|---|
| `docs/开发文档.md` | 面向开发者：环境、数据管道、API 契约、测试、部署、排障（最全） |
| `docs/技术栈.md` | 技术栈选型与版本锁定 |
| `docs/组件工作机制.md` | 20+ 组件的 Mermaid 机制图解 |
| `docs/并行开发协作.md` | 多 Agent 并行开发：所有权/契约/共享资源/合并流程 |
| `docs/增加Agent功能.md` | 二期功能清单（历史参考；多 Agent 等已落地） |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 贡献流程：环境、分支/PR、测试、代码风格与安全约定 |
