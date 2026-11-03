## 变更说明

<!-- 做了什么 + 为什么。关联 issue/任务链接。 -->

## Lane 信息

- Lane：`integrator` / `contracts-platform` / `agent-core` / `rag` / `services-tender` / `services-analysis` / `data-eval-platform` / `frontend`
- 分支：`feat/parallel/<lane>`（或 `feat/contract/<name>`）
- Base tag：`parallel-base-vN`

## 契约检查

- [ ] 未修改 `bidding-agent/docs/并行开发协作.md` §4.1 的冻结契约
- [ ] 若修改了冻结契约：已走 CCR，链接：______
- [ ] 未修改 `api/server.py`、`src/agent/__init__.py`、`src/rag/__init__.py`、lockfiles（这些由 integrator 单写）

## 验证

<!-- 粘贴实际命令与结果，不要只写“通过”。 -->

```bash
# 示例
C:/Users/10518/bidding-agent/bidding-agent/.venv/Scripts/python.exe \
  -m pytest -q -m "not integration" tests/test_parallel_contracts.py tests/<lane-tests>.py
```

- [ ] targeted pytest 通过
- [ ] RAG lane：`retrieval_offline` 临时趋势门禁通过（`--trend .parallel/trend-rag.jsonl`）
- [ ] frontend lane：`npm run test` + `npx tsc -b` 通过

## 文档增量

<!-- 改了什么行为、需要同步到哪些文档。由 integrator 统一应用；lane 不直接改共享 docs。 -->

- [ ] 无文档变化
- [ ] 需要同步：______（Agent 流程 / 新端点 / 新配置 / 包结构 / 前端状态）

## 共享资源检查

- [ ] 未写真实 `bidding-agent/eval/trend.jsonl` / `baseline.json` / `fixtures/**` / `online_samples.jsonl`
- [ ] 未跑 `--save` / `--save-db` / `RAGAS_SAVE_BASELINE=1`
- [ ] 未跑 `main.py ingest` / `pg --force` / `graph --clear`
- [ ] 未启动 8001 / 未跑 `LIVE_API=1`
- [ ] 未跑 `--cov` / `npm run build`
- [ ] worktree 未复制 `.env`

## Handoff

```json
{
  "lane": "",
  "base_tag": "parallel-base-v1",
  "commits": [],
  "files": [],
  "tests": [],
  "contract_changes": [],
  "blocked": [],
  "next_steps": []
}
```

## 风险与回滚

<!-- 影响面、回滚方式。 -->
