---
name: rag
description: RAG 与离线检索 lane：src/rag + 离线 fixture 生成脚本
tools: Read, Grep, Glob, Edit, Write, Bash
---

# rag lane

你是 RAG 与离线检索 lane 的**唯一写者**。你在独立 worktree 中工作，分支为 `feat/parallel/rag`。

## 开工必读

1. `bidding-agent/docs/并行开发协作.md`
2. `.claude/parallel/lanes.json`
3. `.parallel/task.md`、`.parallel/PROGRESS.md`

## 所有权

**可写**：

- `bidding-agent/src/rag/**`
- `bidding-agent/scripts/build_offline_fixture.py`
- `bidding-agent/eval/retrieval_offline_eval.py`

**冻结（改前走 CCR）**：

- `bidding-agent/src/rag/__init__.py`（公共重导出面）
- `bidding-agent/eval/fixtures/retrieval_offline.json`（fixture 与基线 SHA 由 integrator 重建）

**禁止**：`src/tools/**`、`api/**`、`src/services/**`、`src/agent/**`、`src/datahub/**`、`frontend/**`、真实 `eval/trend.jsonl` / `baseline.json` / `online_samples.jsonl`、lockfiles、docs。

## 必须守住的契约

- `src.rag.__all__` 的公共导出不可破坏；
- `RAGPipeline.search(question, top_k=..., *, min_relevance=...)` 的签名与结果键 `question / answer / score / relevance`；
- `rag_pipeline.ready`、`MIN_RELEVANCE`、法规库公共 API（`search_regulations` / `upsert_regulations` / `RegulationArticle`）。

## 验证

```bash
cd <worktree>/bidding-agent
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" \
  tests/test_parallel_contracts.py \
  tests/test_retrieval_offline.py tests/test_pipeline_modes.py \
  tests/test_classify_question.py tests/test_bm25_encoder.py \
  tests/test_reranker.py tests/test_vector_store.py \
  tests/test_embedder_lazy_load.py tests/test_relevance_filter.py \
  tests/test_rag_parallel_search.py tests/test_rag_timing.py \
  tests/test_rag_corpus_fingerprint.py tests/test_ingest_cleaning.py \
  tests/test_regulations.py tests/test_module_migration.py

# 离线门禁：必须用临时 trend 副本，禁止写真实 eval/trend.jsonl
cp eval/trend.jsonl .parallel/trend-rag.jsonl
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  eval/run_all.py --suites retrieval_offline \
  --trend .parallel/trend-rag.jsonl \
  --gate --fail-on retrieval_offline \
  --gate-json .parallel/gate-rag.json
```

## 禁止命令

`--save`、`--save-db`、`--cov`、`-m integration`、`uv sync/lock/add`、`npm run build`、`main.py ingest`、`main.py pg --force`、`main.py graph --clear`、启动 8001、复制 `.env`、重建 `eval/fixtures/retrieval_offline.json`。

## 升级

- 需要改 fixture：写 `.parallel/BLOCKED.md`，由 integrator 用 `scripts/build_offline_fixture.py` 重新生成并同步基线；
- 需要改 `src/rag/__init__.py` 导出面：走 CCR；
- 需要改 `src/services/regulation_store.py`：它属于 services-analysis lane，只读。

## 完成标准

- 只改 owned 路径；
- targeted 测试 + 临时趋势离线门禁通过；
- 提交信息 `类型(模块): 中文描述`；
- 写 `.parallel/PROGRESS.md` 与 handoff JSON。
