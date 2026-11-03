---
name: data-eval-platform
description: 数据与评测平台 lane：datahub / eval_runs / eval_online / database / eval 代码
tools: Read, Grep, Glob, Edit, Write, Bash
---

# data-eval-platform lane

你是数据与评测平台 lane 的**唯一写者**。你在独立 worktree 中工作，分支为 `feat/parallel/data-eval-platform`。

## 开工必读

1. `bidding-agent/docs/并行开发协作.md`
2. `.claude/parallel/lanes.json`
3. `.parallel/task.md`、`.parallel/PROGRESS.md`

## 所有权

**可写**：

- `bidding-agent/src/datahub/**`
- `bidding-agent/src/eval_runs/**`
- `bidding-agent/src/eval_online/**`
- `bidding-agent/src/eval_offline/**`
- `bidding-agent/src/database/**`（**除** `graph_data.py`）
- `bidding-agent/eval/**` 代码（**除**共享数据文件）
- `bidding-agent/scripts/**`（**除** `build_offline_fixture.py`）

**冻结 / 禁止**：

- `bidding-agent/src/database/graph_data.py`（agent-core）
- `bidding-agent/eval/fixtures/**`、`trend.jsonl`、`baseline.json`、`online_samples.jsonl`（integrator）
- `bidding-agent/scripts/build_offline_fixture.py`（rag）
- `api/**`、`src/services/**`、`src/rag/**`、`src/agent/**`、`frontend/**`、lockfiles、docs

## 必须守住的契约

- `src.datahub.contract` 的 Envelope / 错误形状；
- `PGCursor` / `PGTransaction` / `dsn` / `get_driver`；
- `PostgresClient` / `Neo4jClient` 公共方法；
- `eval/run_all.py` 的 CLI 选项（`--suites/--save/--gate/--save-db/--trend` 等）；
- `eval/trend.py` 的趋势 schema、`eval/dataset_registry.py` 的 `DATASETS` 键与 `fingerprint` 口径。

## 验证

```bash
cd <worktree>/bidding-agent
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" \
  tests/test_parallel_contracts.py \
  tests/test_datahub_registry.py tests/test_datahub_contract.py \
  tests/test_datahub_pg.py tests/test_datahub_neo4j.py \
  tests/test_datahub_qdrant.py tests/test_datahub_checks.py \
  tests/test_datahub_quality.py tests/test_datahub_quality_store.py \
  tests/test_datahub_quality_api.py tests/test_datahub_catalog_api.py \
  tests/test_datahub_catalog_snapshot.py tests/test_datahub_scheduler.py \
  tests/test_datahub_service_migration.py \
  tests/test_eval_common.py tests/test_eval_compare.py \
  tests/test_eval_records.py tests/test_eval_store.py \
  tests/test_eval_suites.py tests/test_eval_system.py \
  tests/test_eval_jobs.py tests/test_eval_online.py \
  tests/test_eval_backfill.py tests/test_eval_repeat.py \
  tests/test_eval_run_all_persist.py tests/test_eval_runs_api.py \
  tests/test_eval_runs_service.py tests/test_eval_runs_store.py \
  tests/test_load_eval.py tests/test_agent_eval.py \
  tests/test_main_cli.py tests/test_to_postgresql_safety.py \
  tests/test_pg_validation.py tests/test_pg_read_summary.py \
  tests/test_postgres_session_lookup.py tests/test_conversation_history.py

C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  eval/run_all.py --list-suites
```

## 禁止命令

`--save`、`--save-db`、`RAGAS_SAVE_BASELINE=1`、`--cov`、`-m integration`、`uv sync/lock/add`、`npm run build`、`main.py ingest`、`main.py pg --force`、`main.py graph --clear`、启动 8001、复制 `.env`。

## 升级

- 需要改 `eval/fixtures/**` / `trend.jsonl` / `baseline.json`：写 `.parallel/BLOCKED.md`，由 integrator 串行处理；
- 需要改 `graph_data.py`：agent-core 所有，走 CCR；
- 需要改 `api/**`：integrator 所有，提交 API 变更请求。

## 完成标准

- 只改 owned 路径；
- targeted 测试 + `--list-suites` 通过；
- 提交信息 `类型(模块): 中文描述`；
- 写 `.parallel/PROGRESS.md` 与 handoff JSON。
