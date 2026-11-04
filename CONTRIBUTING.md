# 贡献指南

感谢参与招投标采购 Agent。提交代码前请先阅读：

- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)：社区行为准则
- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md)：架构约定、契约红线、测试约定
- [`bidding-agent/docs/并行开发协作.md`](bidding-agent/docs/并行开发协作.md)：多 Agent 并行开发的所有权/冻结契约/合并流程
- [`bidding-agent/docs/开发文档.md`](bidding-agent/docs/开发文档.md)：环境、API 契约、评测、部署、排障

## 环境与启动

```bash
# 后端（Python ≥ 3.12，uv 管理）
cd bidding-agent
uv sync
cp .env.example .env   # 填写 QDRANT_* / DEEPSEEK_API_KEY 等；.env 永不入库
python main.py ingest  # 首次导入知识库
python main.py api     # http://localhost:8001

# 前端（另开终端）
cd frontend
npm install
npm run dev            # http://localhost:5173
```

## 分支与 PR

- `main` 是稳定分支，**只有通过 PR 才能合入**；推荐 Squash and merge。
- 每人一条功能分支，例如 `feat/agent-core`、`feat/rag`、`feat/tools-data`、`feat/api-llm`、`feat/frontend`。
- 并行开发只在独立 worktree + `feat/parallel/<lane>` 分支中进行；`main` 只由 integrator 合并。
- PR 必须带 targeted 测试与「文档增量」；不要直接在 `main` 上 push。
- 冻结契约（SSE `done` 帧、`tool_name` 单字符串、工具结果形状、公共导入路径、`.env` 键名等）改动必须同步文档并通知相关成员。

## 提交信息

格式：`类型(模块): 中文描述`，类型取 `feat / fix / docs / test / chore`。描述说明“做了什么 + 为什么”。

## 测试与 CI

```bash
# 后端（在 bidding-agent/ 下）
uv run pytest -q                 # 全量；integration 默认排除
uv run pytest tests/test_xxx.py -v

# 前端（在 frontend/ 下）
npm run test
npm run build
```

CI 会跑：后端 pytest（含依赖约束）、前端 build + Vitest、gitleaks、PR 标题/提交信息规范、离线检索门禁（路径相关时）。不要在 PR 中提交 `.env`、密钥、真实业务数据或 `eval/trend.jsonl` 的无关改动。

## 代码风格

- 注释与日志用中文，解释「为什么」而不是复述「做什么」。
- 常量集中在 `src/agent/constants.py`；所有外部调用必须有超时；共享可变状态必须加锁。
- 工具永不抛异常，统一返回 `{"success", "data"/"results", "error"}`。
- 新增数据访问优先走 `src/datahub/` 的 `PGCursor` / `PGTransaction` / `neo4j` / `qdrant`，不要直接 `psycopg.connect` / `GraphDatabase.driver`。
- 出站 HTTP 保持 `trust_env=False`。

## 评测

五线评测统一入口：

```bash
cd bidding-agent
AGENT_VERIFY_REPAIR=1 python eval/run_all.py --suites agent_multi --repeat 3 --save
python eval/run_all.py --suites retrieval_offline --save --save-db
```

- `eval/trend.jsonl` 是门禁事实源；`--gate` 只比较，`--save` 才写趋势。
- 改评测用例/阈值/门禁行为时，同步更新 `bidding-agent/docs/开发文档.md` §10 与相关 README。

## 安全问题

不要在 issue/PR/日志中粘贴密钥、数据库连接串或用户隐私数据。发现安全问题请通过私下渠道联系维护者。
