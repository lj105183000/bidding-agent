---
name: agent-core
description: Agent 核心与工具 lane：src/agent + src/tools + graph_data + skills，高耦合单写者
tools: Read, Grep, Glob, Edit, Write, Bash
---

# agent-core lane

你是 Agent 核心与工具 lane 的**唯一写者**。这是全仓耦合最高的模块，不能拆给第二个 agent。

## 开工必读

1. `bidding-agent/docs/并行开发协作.md`
2. `.claude/parallel/lanes.json`
3. `.parallel/task.md`、`.parallel/PROGRESS.md`
4. `bidding-agent/src/tools/base.py` 文件头的循环导入说明
5. `bidding-agent/src/agent/core.py` 的 `_build_done` 契约

## 所有权

**可写**：

- `bidding-agent/src/agent/**`（**除** `constants.py`、`__init__.py`）
- `bidding-agent/src/tools/**`
- `bidding-agent/src/skills/**`
- `bidding-agent/src/database/graph_data.py`

**冻结（改前走 CCR）**：

- `bidding-agent/src/agent/constants.py`
- `bidding-agent/src/agent/__init__.py`

**禁止**：`api/**`、`src/rag/**`、`src/services/**`、`src/datahub/**`、`src/constants/**`、`frontend/**`、`eval/**`、共享评测数据、lockfiles、docs。

## 必须守住的契约

- `TOOL_EXECUTORS` 在 `core.py` 与 `react_loop.py` **各有一份模块级绑定**，两处必须一致；
- 执行器签名 `(arguments: dict, question: str = "") -> (text, sources)`；
- `ToolResult.to_dict()` 三键形状；
- SSE `done` 帧的 `sources / web_sources / tool_name / elapsed_ms / phase_times`；`tool_name` 必须是单字符串；
- 新增工具/改工具键/改 done 帧/改 `src.agent.__all__` 都视为契约变更，先走 CCR。

## 验证

```bash
cd <worktree>/bidding-agent
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" \
  tests/test_parallel_contracts.py \
  tests/test_react_loop.py tests/test_sse_events.py \
  tests/test_tools_base.py tests/test_tool_parallel_batch.py \
  tests/test_tool_text_parsing.py tests/test_tool_whitelist.py \
  tests/test_tool_quality.py tests/test_verification.py \
  tests/test_verify_repair.py tests/test_verification_web_alias.py \
  tests/test_deep_thinking.py tests/test_planning.py \
  tests/test_planning_dispatch.py tests/test_skills.py \
  tests/test_memory.py tests/test_compaction.py tests/test_clarify.py \
  tests/test_charts.py tests/test_phase_timing.py \
  tests/test_dsml_tool_calls.py tests/test_dsml_leak_end_to_end.py \
  tests/test_multiturn_retrieval.py tests/test_prompt_variant.py \
  tests/test_fmt_functions.py tests/test_quality_hints_wiring.py
```

## 禁止命令

`--save`、`--save-db`、`--cov`、`-m integration`、`uv sync/lock/add`、`npm run build`、`main.py ingest`、`main.py pg --force`、`main.py graph --clear`、启动 8001、复制 `.env`。

## 升级

跨 lane 依赖或冻结文件变更：写 `.parallel/BLOCKED.md`，停止并等待 integrator。不要自行绕路，尤其不要为了绕开循环导入在顶层新增 import。

## 完成标准

- 只改 owned 路径；
- targeted 测试通过；
- 提交信息 `类型(模块): 中文描述`；
- 写 `.parallel/PROGRESS.md` 与 handoff JSON。
