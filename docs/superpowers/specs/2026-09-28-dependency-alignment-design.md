# 依赖对齐设计：后端按文档锁版本 + 文档同步实际栈

日期：2026-09-28
状态：已确认（方案 A）

## 背景与目标

对照《技术栈.md》检查发现三类偏差：

1. **后端跨大版本漂移**：环境装了 `openai 3.15.0`、`sentence-transformers 6.0.1`，而文档 §12 明确"项目代码面向 OpenAI SDK v2 API"、§1 锁定 sentence-transformers 5.6.1。`>=` 约束无法阻止再次漂移。
2. **锁文件过期**：`bidding-agent/uv.lock` 与 pyproject 不同步（`uv lock --check` 失败）；`ipykernel 7.3.0` 文档有、声明无。
3. **前端文档失实**：`frontend/` 实为 Vite 5 + react-router-dom + recharts（dev 端口 5173，无代理配置），文档写的是 Next.js 14 + react-markdown/remark-gfm + 端口 3000。
4. **根目录杂项**：根 `pyproject.toml`（空依赖占位）、根 `uv.lock`、根空 `.venv` 为误生成，导致 `uv pip` 解析到错误环境。

**决策（用户确认）**：前端保持 Vite 不迁移 Next.js，文档反向改为描述实际栈；后端用边界锁版本（方案 A）；删除根目录杂项。

## 变更内容

### 1. `bidding-agent/pyproject.toml`

| 依赖 | 现状 | 改为 |
|---|---|---|
| openai | `>=2.48.0` | `>=2.48.0,<3.0.0` |
| sentence-transformers | `>=5.6.1` | `>=5.6.1,<6.0.0` |
| dev 组 | `pytest>=9.1.1` | 增加 `ipykernel>=7.3.0` |

其余依赖不动（已装版本均满足约束且无跨大版本问题）。

### 2. 重新锁定并同步环境

在 `bidding-agent/` 执行 `uv lock` + `uv sync`。预期：openai 回落 2.x 最新、sentence-transformers 回落 5.x 最新，torch/transformers 等传递依赖随 ST 5.x 重解析。完成后 `uv pip list` 核对 openai 2.x / sentence-transformers 5.x / ipykernel 在列，并记录实际解析版本供文档刷新引用。不动代码、不跑测试回归。

### 3. 删除根目录杂项

删除根 `pyproject.toml`、根 `uv.lock`、根 `.venv/`。影响：`uv pip` 不再误解析空环境；无代码引用这些文件。

### 4. 文档同步

- **`bidding-agent/docs/技术栈.md`**：
  - §1/§5：openai、sentence-transformers、torch、transformers 等版本号按步骤 2 实际锁文件结果刷新。
  - §8 整节改写：Vite 5 + React 18.3 + react-router-dom 6.x + recharts 3.x + Tailwind 3.4 + TypeScript 5.6 + lucide-react 1.x；dev 端口 5173，无框架级代理（前端直连后端 :8001）；删去 Next.js rewrites / react-markdown / remark-gfm / @types/node 的表述。
  - §9 补 ipykernel（dev 组，notebook 用）。
  - §12 保留 OpenAI SDK v2 注记，补充"pyproject 已加 `<3` 上界"。
- **`CLAUDE.md`**：常用命令中前端端口 3000 → 5173；`frontend/` 相关的 Next.js 表述改为 Vite 实况；仓库布局注明 `frontend/` 位于仓库根。

## 验收标准

- `uv lock --check` 在 `bidding-agent/` 通过。
- `uv pip list` 显示 openai 2.x、sentence-transformers 5.x、ipykernel 存在。
- 根目录无 `pyproject.toml` / `uv.lock` / `.venv`。
- 《技术栈.md》§8 与 `frontend/package.json` 逐项一致；《技术栈.md》§1 版本号与 `uv.lock` 一致；CLAUDE.md 无 Next.js/3000 残留表述。
