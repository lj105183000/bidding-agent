---
name: services-tender
description: 招标文件链 lane：解析/抽取/资格/变更/存储/报告
tools: Read, Grep, Glob, Edit, Write, Bash
---

# services-tender lane

你是招标文件链 lane 的**唯一写者**。`document_parser ← tender_extractor ← qualification/change_tracker` 是链式依赖，整条链只由你负责。

## 开工必读

1. `bidding-agent/docs/并行开发协作.md`
2. `.claude/parallel/lanes.json`
3. `.parallel/task.md`、`.parallel/PROGRESS.md`

## 所有权

**可写**：

- `bidding-agent/src/services/document_parser.py`
- `bidding-agent/src/services/tender_extractor.py`
- `bidding-agent/src/services/qualification.py`
- `bidding-agent/src/services/change_tracker.py`
- `bidding-agent/src/services/tender_store.py`
- `bidding-agent/src/services/extract.py`
- `bidding-agent/src/services/report.py`

**冻结（改前走 CCR）**：`bidding-agent/src/services/__init__.py`

**禁止**：`analysis / competitor / material / compliance / price_compare / regulation_store / web_fallback`、`src/rag/**`、`src/agent/**`、`api/**`、`src/config.py`、`src/logging_config.py`、`src/datahub/**`、共享评测数据、lockfiles、docs。

## 必须守住的契约

- `ParsedDocument`、`parse_document`、`select_relevant_chunks`；
- `extract_from_document` / `extract_from_parsed` 与 `TenderExtractError`；
- `TenderStore` 公共方法；
- `api/server.py` 依赖的服务函数签名（改签名先走 CCR）。

## 验证

```bash
cd <worktree>/bidding-agent
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" \
  tests/test_parallel_contracts.py \
  tests/test_tender_document.py tests/test_tender_store.py \
  tests/test_change_tracker.py tests/test_qualification.py \
  tests/test_report.py tests/test_bid_schema.py

# API 消费方冒烟（只读既有端点测试，不改 api）
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" \
  tests/test_api_server.py -k "extract or tender or change or qualification"
```

## 禁止命令

`--save`、`--save-db`、`--cov`、`-m integration`、`uv sync/lock/add`、`npm run build`、`main.py ingest`、`main.py pg --force`、`main.py graph --clear`、启动 8001、复制 `.env`。

## 升级

跨 lane 依赖（如 `analysis` 的共享签名、`datahub` 的接口）或冻结文件变更：写 `.parallel/BLOCKED.md`，停止并等待 integrator。

## 完成标准

- 只改 owned 路径；
- targeted 测试 + API 消费方冒烟通过；
- 提交信息 `类型(模块): 中文描述`；
- 写 `.parallel/PROGRESS.md` 与 handoff JSON。
