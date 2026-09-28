# 依赖对齐实施计划（后端锁版本 + 删根目录杂项 + 文档同步）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 后端依赖锁回文档要求的大版本（openai 2.x / sentence-transformers 5.x），删除根目录误生成的占位文件，并把《技术栈.md》《CLAUDE.md》同步为实际技术栈（Vite 前端）。

**Architecture:** 纯依赖/文档变更，不动任何代码。先删根目录杂项（消除 `uv pip` 解析到空 `.venv` 的歧义），再改 `pyproject.toml` 边界锁版本，`uv lock` + `uv sync` 重解析，最后按实际解析结果刷新文档。

**Tech Stack:** uv、pyproject.toml、Markdown 文档

**Spec:** `docs/superpowers/specs/2026-09-28-dependency-alignment-design.md`

**注意：**
- 全程在 `dev` 分支操作。仓库中已有他人未提交改动（`frontend/package-lock.json`），**不要**把它们加进提交。
- Bash 工作目录注意：执行前先 `cd` 到正确目录（本仓库根为 `D:\bidding-agent`，后端项目在 `D:\bidding-agent\bidding-agent`）。
- Task 3 中记录的版本号是 Task 5 文档刷新的输入，执行时**必须**记录下来。

---

### Task 1: 删除根目录误生成的杂项

**Files:**
- Delete: `D:\bidding-agent\pyproject.toml`（空依赖占位）
- Delete: `D:\bidding-agent\uv.lock`（该占位项目的锁文件，git status 中本就是未提交的修改）
- Delete: `D:\bidding-agent\.venv\`（空虚拟环境，未被 git 跟踪）

- [ ] **Step 1: 确认根 pyproject 确实是占位**

```bash
cat D:/bidding-agent/pyproject.toml
```

预期：`dependencies = []` 的三行占位。如果内容不是这样，**停止**并向用户确认。

- [ ] **Step 2: 删除三个杂项**

```bash
rm D:/bidding-agent/pyproject.toml D:/bidding-agent/uv.lock && rm -rf D:/bidding-agent/.venv
```

- [ ] **Step 3: 验证删除且 git 状态符合预期**

```bash
cd D:/bidding-agent && ls pyproject.toml uv.lock .venv 2>&1; git status --short
```

预期：三个路径均 "No such file"；git status 只剩 `frontend/package-lock.json` 的修改（他人的，不碰）。

---

### Task 2: 修订 `bidding-agent/pyproject.toml`（边界锁版本）

**Files:**
- Modify: `D:\bidding-agent\bidding-agent\pyproject.toml`

- [ ] **Step 1: 三处编辑**

编辑 1 — openai 加上界：

```toml
# 旧
    "openai>=2.48.0",
# 新
    "openai>=2.48.0,<3.0.0",
```

编辑 2 — sentence-transformers 加上界：

```toml
# 旧
    "sentence-transformers>=5.6.1",
# 新
    "sentence-transformers>=5.6.1,<6.0.0",
```

编辑 3 — dev 组加 ipykernel：

```toml
# 旧
[dependency-groups]
dev = ["pytest>=9.1.1"]
# 新
[dependency-groups]
dev = ["pytest>=9.1.1", "ipykernel>=7.3.0"]
```

- [ ] **Step 2: 提交（pyproject 先行，uv.lock 下个任务一起提交）**

```bash
cd D:/bidding-agent && git add bidding-agent/pyproject.toml && git commit -m "$(cat <<'EOF'
chore(deps): openai/sentence-transformers 加大版本上界，dev 组补 ipykernel

文档（技术栈.md §12）明确代码面向 OpenAI SDK v2 API；>= 裸约束导致环境漂移到
openai 3.x / sentence-transformers 6.x。加 <3 / <6 上界防再次漂移。
EOF
)"
```

---

### Task 3: 重新锁定并同步环境，记录实际版本

**Files:**
- Modify: `D:\bidding-agent\bidding-agent\uv.lock`

- [ ] **Step 1: 重新生成锁文件并同步环境**

```bash
cd D:/bidding-agent/bidding-agent && uv lock && uv sync
```

预期：uv 解析出 openai 2.x 最新、sentence-transformers 5.x 最新；`uv sync` 安装（含 dev 组 ipykernel）。

- [ ] **Step 2: 核对关键版本并记录**

```bash
cd D:/bidding-agent/bidding-agent && uv pip list | grep -Ei "^(openai|sentence-transformers|torch|transformers|ipykernel|pytest) "
```

预期：openai 2.x.x、sentence-transformers 5.x.x、ipykernel 7.x、pytest 9.x。

**把这条命令的完整输出记录下来**（torch/transformers 的版本号 Task 5 要用）。

- [ ] **Step 3: 环境自检 + 测试冒烟**

```bash
cd D:/bidding-agent/bidding-agent && uv lock --check && uv run pytest -q 2>&1 | tail -3
```

预期：`uv lock --check` 输出 lockfile 为最新；pytest 通过（若有个别失败，确认是既有失败、与本次依赖变更无关再继续，必要时向用户报告）。

- [ ] **Step 4: 提交锁文件**

```bash
cd D:/bidding-agent && git add bidding-agent/uv.lock && git commit -m "$(cat <<'EOF'
chore(deps): 重新生成 uv.lock，openai 回落 2.x / sentence-transformers 回落 5.x
EOF
)"
```

---

### Task 4: 《技术栈.md》§8 前端章节改写为 Vite 实况

**Files:**
- Modify: `D:\bidding-agent\bidding-agent\docs\技术栈.md`（§8 整节）

- [ ] **Step 1: 用下方内容整体替换 §8 的表格与正文**

将现有的 `## 8. 前端技术栈` 一节（从 `| 类别 | 依赖 | 版本 | 说明 |` 表头到该节末尾）替换为：

```markdown
| 类别 | 依赖 | 版本 | 说明 |
|---|---|---|---|
| 框架 | vite + @vitejs/plugin-react | ^5.4.10 / ^4.3.3 | SPA 开发服务器与打包；dev 端口 **5173**（`npm run dev`），`npm run build` 产出 dist/，vendor 手动拆包（react / charts） |
| UI | react / react-dom | ^18.3.1 | 函数组件 + Hooks |
| 路由 | react-router-dom | ^6.30.6 | SPA 客户端路由 |
| 图表 | recharts | ^3.10.1 | 数据可视化 |
| 语言 | typescript | ~5.6.2 | 严格类型；`tsc -b` 参与构建 |
| 样式 | tailwindcss + postcss + autoprefixer | ^3.4.19 / ^8.5.28 / ^10.6.1 | 暗色模式 |
| 图标 | lucide-react | ^1.47.0 | UI 图标 |
| 代码检查 | eslint + typescript-eslint + eslint-plugin-react-hooks + eslint-plugin-react-refresh | ^9.13.0 / ^8.11.0 / ^5.0.0 / ^0.4.14 | `npm run lint` |
| 类型 | @types/react / @types/react-dom | ^18.3.12 / ^18.3.1 | TS 类型支持 |

> 注：前端目前为纯前端 SPA 骨架，**尚未接入后端 API**（`src/` 中无 fetch/EventSource 调用）。
> 后续接入 SSE 时应直连后端 :8001（Vite 无 dev 代理，生产反代需 `proxy_buffering off` 防缓冲）。
```

- [ ] **Step 2: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/docs/技术栈.md && git commit -m "$(cat <<'EOF'
docs(技术栈): §8 前端章节改为实际 Vite 5 栈

frontend/ 实为 Vite 5 + react-router-dom + recharts（dev 端口 5173），
并非文档所写 Next.js 14 + react-markdown；按实际 package.json 逐项对齐。
EOF
)"
```

---

### Task 5: 《技术栈.md》其余章节按实际锁定版本刷新

**Files:**
- Modify: `D:\bidding-agent\bidding-agent\docs\技术栈.md`（§1 / §5 / §9 / §12）

以下 `<v>` 均填入 **Task 3 Step 2 记录的实际版本号**（以 `uv pip list` 输出为准，不是凭记忆猜）。

- [ ] **Step 1: §1 总览表三处替换**

行「LLM 接入」：

```markdown
# 旧
| LLM 接入 | OpenAI SDK / zai-sdk | 2.48.0 / 0.2.3 | DeepSeek 与本地兼容端点 / 智谱文本与视觉 |
# 新（<v> 填 openai 实际版本）
| LLM 接入 | OpenAI SDK / zai-sdk | <v> / 0.2.3 | DeepSeek 与本地兼容端点 / 智谱文本与视觉（openai 上界锁 <3.0.0） |
```

行「前端框架」（把 Next.js 行改为 Vite 行）：

```markdown
# 旧
| 前端框架 | Next.js（App Router） | ^14.2.0 | SSR 框架 + 开发代理 |
# 新
| 前端框架 | Vite（SPA） | ^5.4.10 | 开发服务器 + 打包，dev 端口 5173 |
```

行「UI 库」：

```markdown
# 旧
| UI 库 | React + TypeScript + Tailwind CSS | ^18.3.0 / ^5.5.0 / ^3.4.0 | 界面 |
# 新
| UI 库 | React + TypeScript + Tailwind CSS | ^18.3.1 / ~5.6.2 / ^3.4.19 | 界面（路由 react-router-dom ^6.30.6，图表 recharts ^3.10.1） |
```

行「Markdown 渲染」（前端无 markdown 渲染依赖，改为图标行并修正 lucide 版本）：

```markdown
# 旧
| Markdown 渲染 | react-markdown + remark-gfm | ^9.0.0 / ^4.0.1 | 回答渲染（含 GFM 表格） |
# 新
| 图标 | lucide-react | ^1.47.0 | UI 图标 |
```

行「图标 lucide-react ^0.400.0」（与上一条合并后**删除原行**，避免重复）：

```markdown
# 删除此行
| 图标 | lucide-react | ^0.400.0 | UI 图标 |
```

- [ ] **Step 2: §1 嵌入模型行 + §5 载体行刷新 torch/transformers/ST 版本**

§1 行「嵌入模型」：

```markdown
# 旧
| 嵌入模型 | BAAI/bge-small-zh-v1.5（sentence-transformers 5.6.1 / torch 2.13.0 / transformers 5.14.1） | — | Dense 语义编码（512 维） |
# 新（<v> 分别填 sentence-transformers / torch / transformers 实际版本）
| 嵌入模型 | BAAI/bge-small-zh-v1.5（sentence-transformers <v> / torch <v> / transformers <v>） | — | Dense 语义编码（512 维） |
```

§5 中「载体」一行同样替换版本号：

```markdown
# 旧
- 载体：**sentence-transformers 5.6.1** + **transformers 5.14.1** + **torch 2.13.0**（CUDA 可用时走 GPU）
# 新
- 载体：**sentence-transformers <v>** + **transformers <v>** + **torch <v>**（CUDA 可用时走 GPU；ST 上界锁 <6.0.0）
```

- [ ] **Step 3: §9 与 §12 补注**

§9 表格 `ipykernel` 行的「用途」列末尾追加 `（pyproject dev 组声明）`：

```markdown
# 旧
| ipykernel | 7.3.0 | `batch/批处理标的物.ipynb` 交互式 notebook |
# 新（7.3.0 若与实际版本不符也一并改为实际值）
| ipykernel | 7.3.0 | `batch/批处理标的物.ipynb` 交互式 notebook（pyproject dev 组声明） |
```

§12 表格「OpenAI SDK v2」行的「原因」列末尾追加 `；pyproject 已加 <3.0.0 上界`：

```markdown
# 旧
| OpenAI SDK v2 | 项目代码面向 SDK v2 API（`chat.completions.create` 与 `reasoning_content` 属性访问），升级 SDK 大版本需回归客户端测试 |
# 新
| OpenAI SDK v2 | 项目代码面向 SDK v2 API（`chat.completions.create` 与 `reasoning_content` 属性访问），升级 SDK 大版本需回归客户端测试；pyproject 已加 `<3.0.0` 上界 |
```

- [ ] **Step 4: 提交**

```bash
cd D:/bidding-agent && git add bidding-agent/docs/技术栈.md && git commit -m "$(cat <<'EOF'
docs(技术栈): §1/§5/§9/§12 版本号与注记按 uv.lock 实际结果刷新
EOF
)"
```

---

### Task 6: 《CLAUDE.md》同步实际形态

**Files:**
- Modify: `D:\bidding-agent\CLAUDE.md`

- [ ] **Step 1: 第 7 行区块——撤掉已失实的"骨架阶段"表述**

```markdown
# 旧（第 7-14 行 ⚠️ 区块的开头两行）
## ⚠️ 当前状态：骨架阶段（先读这条）

本仓库目前**只有文档，没有代码**。`bidding-agent/docs/` 下的四份文档描述的是**目标架构**，`src/`、`tests/`、`frontend/`、`main.py`、`conftest.py` 均**尚不存在**。
# 新
## ⚠️ 当前状态：开发中（先读这条）

后端代码已落地于 `bidding-agent/`（`src/`、`tests/`、`conftest.py`、`main.py` 等）；前端位于**仓库根** `frontend/`（Vite 5 SPA，尚未接入后端 API）。`bidding-agent/docs/` 下的文档中，前端相关章节描述的是旧目标架构（Next.js），以本文档为准。
```

- [ ] **Step 2: 第 50 行——前端命令端口**

```markdown
# 旧
cd frontend && npm install && npm run dev # 前端 http://localhost:3000
# 新
cd frontend && npm install && npm run dev # 前端 http://localhost:5173（Vite）
```

- [ ] **Step 3: 第 68 行——架构链路表述**

```markdown
# 旧
Next.js 14 前端（:3000）→ FastAPI（:8001）→ `BiddingAgent` 编排**四路数据源**回答招投标问题。
# 新
Vite SPA 前端（:5173，仓库根 `frontend/`）→ FastAPI（:8001）→ `BiddingAgent` 编排**四路数据源**回答招投标问题。
```

- [ ] **Step 4: 第 160 行——排障表中 Next.js 残留**

```markdown
# 旧
| 前端流式整段一次显示 | SSE 走了缓冲代理。开发时 `lib/api.ts` 绕过 Next 代理直连 :8001；生产用 `NEXT_PUBLIC_API_BASE`，反代需 `proxy_buffering off` |
# 新
| 前端流式整段一次显示（接入 SSE 后） | 响应被代理缓冲。前端为 Vite、无 dev 代理，接入时直连 :8001；生产反代需 `proxy_buffering off` |
```

- [ ] **Step 5: 全文残留检查**

```bash
cd D:/bidding-agent && grep -n "3000\|Next.js\|next dev\|NEXT_PUBLIC\|react-markdown" CLAUDE.md
```

预期：无匹配（`分配说明.md`、`docs/` 里的历史表述不在本任务范围）。

- [ ] **Step 6: 提交**

```bash
cd D:/bidding-agent && git add CLAUDE.md && git commit -m "$(cat <<'EOF'
docs(CLAUDE): 撤销失实的骨架阶段表述，前端描述对齐 Vite 实况（:5173）
EOF
)"
```

---

### Task 7: 终验（对照 spec 验收标准）

- [ ] **Step 1: 逐条验收**

```bash
cd D:/bidding-agent/bidding-agent && uv lock --check
cd D:/bidding-agent/bidding-agent && uv pip list | grep -Ei "^(openai|sentence-transformers|ipykernel) "
cd D:/bidding-agent && ls pyproject.toml uv.lock .venv 2>&1
cd D:/bidding-agent && grep -rn "Next.js" CLAUDE.md bidding-agent/docs/技术栈.md
```

预期依次：lockfile 最新；openai 2.x / sentence-transformers 5.x / ipykernel 在列；三个杂项不存在；无 Next.js 残留。

- [ ] **Step 2: 对照 `frontend/package.json` 目视核对 §8 表格**（依赖名与版本前缀逐项一致）。

- [ ] **Step 3: `git log --oneline -6` 确认提交序列完整，向用户汇报结果。**
