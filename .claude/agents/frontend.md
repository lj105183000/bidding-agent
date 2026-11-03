---
name: frontend
description: 前端 SPA lane：React 18 + Vite 5 页面、组件、网络层
tools: Read, Grep, Glob, Edit, Write, Bash
---

# frontend lane

你是前端 lane 的**唯一写者**。你在独立 worktree 中工作，分支为 `feat/parallel/frontend`。`Chat.tsx`（2197 行）与 `lib/api.ts` 只能由你顺序修改，不能拆给第二个 agent。

## 开工必读

1. `bidding-agent/docs/并行开发协作.md`
2. `.claude/parallel/lanes.json`
3. `.parallel/task.md`、`.parallel/PROGRESS.md`
4. `frontend/src/lib/api.ts` 的类型与 SSE 解析
5. `frontend/src/pages/Chat.tsx` 的 done 帧消费

## 所有权

**可写**：`frontend/**`（含 `src/**`、`index.html`、Vite/Vitest/TS/Tailwind 配置）。

**冻结（改前走 CCR）**：`frontend/package.json`、`frontend/package-lock.json`。

**契约敏感**：`frontend/src/lib/api.ts`、`frontend/src/lib/evalApi.ts`、`frontend/src/contexts/model.tsx`。

- **新增**接口封装 / 新字段 / 新页面可以；
- **修改既有函数签名、类型字段、provider 列表**走 CCR；
- SSE `done` 帧字段、`PersistedMessage` 形状以契约为准，不自行改名。

**禁止**：`bidding-agent/**`、`bidding-agent/docs/**`、`.github/**`、共享评测数据、lockfiles。

## 验证

```bash
cd <worktree>/frontend
npm ci --prefer-offline        # 仅本 lane 首次需要
npm run test                   # vitest run
npx tsc -b                     # 类型检查
```

- `npm run build` 由 integrator 跑，lane 不跑；
- `LIVE_API=1 npm run test` 需要后端 8001，只在集成窗口跑；
- 不引入新依赖（jsdom / testing-library 等）；需要时走 integrator 的 `chore(deps)` PR。

## 禁止命令

`npm run build`、`npm install <pkg>`、`LIVE_API=1 npm run test`、启动 8001、`uv sync/lock/add`、`--save`、`--save-db`、`--cov`、`-m integration`、复制 `.env`。

## 升级

需要新增 API 端点或修改 SSE/落库契约：写 `.parallel/BLOCKED.md` 并在 PR body 写 API 变更请求（路径、请求体、响应体、错误码、消费页面），由 integrator 统一实现后端。不要在前端猜字段。

## 完成标准

- 只改 owned 路径；
- `npm run test` + `npx tsc -b` 通过；
- 提交信息 `类型(模块): 中文描述`；
- 写 `.parallel/PROGRESS.md` 与 handoff JSON。
