# 前后端接口联调设计：Chat 全链路（后端补齐 + 前端接入）

日期：2026-09-28
状态：已确认（方案 A：直连 :8001 + CORS；全链路范围）

## 背景与决策

- `dev` 分支没有 `api/server.py`；API 服务在 `origin/feat/api-llm`（队友 D），为骨架版：直连 `get_llm_client` 未接 `BiddingAgent`，SSE 帧用 `event` 键且 done 帧恒空，与 dev 分支 `tests/test_sse_events.py` 的 `type` 键口径冲突。
- 前端（Vite 5 SPA，仓库根 `frontend/`）完全 mock：无 fetch/EventSource/localStorage，消息类型与 PersistedMessage 契约不一致，provider 列表为假选项。
- **用户决策**：全链路联调（后端补齐 + 前端接入）；后端以 `git merge origin/feat/api-llm` 合入 dev 后补齐（保留队友历史）；回答渲染引入 react-markdown + remark-gfm；前端直连 `:8001`（CORS 加 `:5173`），不用 Vite 代理（与 CLAUDE.md/技术栈.md 既有指引一致，SSE 无缓冲风险）。
- **范围**：仅 Chat 页全链路（SSE 流式问答、provider 切换、联网/深度思考开关、会话持久化与列表、点赞/点踩反馈、健康状态）。Dashboard/extract/price/material/analysis 四页继续 mock 不动；`/api/vision` 不做。

## 1. 后端（dev 上补齐）

### 1.1 合入
`git merge origin/feat/api-llm`；冲突解决原则：**SSE 帧格式以 dev 分支测试口径为准**（`type` 键、`phase_times` 为 `[[名, ms], ...]`）。`src/config.py` 两版差异大，合并时保留 dev 的配置结构并纳入 feat 分支新增的 API/CORS 相关键。

### 1.2 `api/server.py` 补齐
- `POST /api/chat/stream` 与 `POST /api/chat` 改为调用 `bidding_agent` 单例（`chat_stream` / `chat`），不再直连 `get_llm_client`。
- done 帧填齐：`sources`（`[{question, answer, score}]`）、`web_sources`（含 `url`）、`tool_name`（**单个字符串**，最后执行完的工具名）、`elapsed_ms`、`phase_times`（`[[名, ms], ...]`）。
- SSE 帧统一 `{type: "status"|"token"|"thinking"|"reset"|"done"|"error", ...}`；保留首帧 2KB padding、尾帧 `[DONE]`。
- 全部端点保持同步 `def`；响应头含 `X-Accel-Buffering: no`。

### 1.3 CORS
- `src/config.py` 的 `cors_origin_list` 默认值加 `http://localhost:5173`（保留 `:3000`）。
- `.env.example` 增加相应键；同步《开发文档.md》§2.3 配置全表。

## 2. 前端网络层 `src/lib/api.ts`（唯一网络层）

- `API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8001'`。
- `askStream(question, opts, handlers)`：`fetch` POST `/api/chat/stream` + `ReadableStream` 手写 SSE 解析：
  - 处理 2KB padding、`data:` 前缀、`[DONE]` 尾帧、`reset` 清屏、坏帧 JSON 容错；
  - 回调 `onStatus / onToken / onThinking / onDone / onError`；
  - `AbortController` 中断；请求参数 `{question, history, web_search_enabled, provider, deep_thinking_enabled}`。
- 其余封装：`fetchHealth()`（GET /api/health）、会话列表/详情/删除（GET/POST/DELETE /api/conversations*）、`sendFeedback()`（POST /api/feedback，`rating: 'up'|'down'`）。

## 3. 前端 Chat 页改造（`src/pages/Chat.tsx` 及新增组件）

- 消息类型对齐：`{id, role: 'user'|'assistant', content, sources?, webSources?, toolCalled?, toolName?, thinking?, elapsedMs?, status}`；**thinking 与计时不落库**（PersistedMessage 契约：`role/content/sources/toolCalled/toolName/image/imageName`）。
- 删除 mock matcher 与 800ms 假延迟，改接 `askStream` 流式渲染。
- 新增依赖 `react-markdown ^9` + `remark-gfm ^4`：表格/列表/代码块渲染，链接强制 `target=_blank + rel=noopener`；流式未闭合标签容错。
- UI：thinking 折叠块、sources 来源徽标、toolName 工具徽标、elapsed 耗时；停止按钮（AbortController）。
- provider：假名单（GPT-4o 等）换成 `deepseek/zhipu/vllm/ollama`，localStorage `chat_llm_provider`，逐请求携带。
- 输入区新增"联网搜索"与"深度思考"开关（逐请求参数）。
- 会话：`session_id = crypto.randomUUID()` 存 localStorage；回合结束 `POST /api/conversations` 落库；侧栏会话列表（拉取/切换/删除）。
- 点赞/点踩接 `POST /api/feedback`（替换现有本地 toast 逻辑）。

## 4. 错误处理

- 非 200：429（限流）、503（知识库/会话存储未就绪）、网络失败——气泡内用户友好提示，不泄露内部细节。
- SSE 中途断流/解析异常：保留已收文本，追加错误标注。
- 不做前端自动重试（后端已保证工具永不抛异常）。

## 5. 验证

- 后端：`uv run pytest` 全绿，`test_sse_events.py` 口径不被破坏；新增 server→Agent 接线测试（文件内局部 fake agent，同步风格，不引入 asyncio 插件）。
- 前端：新增 **vitest**，仅为 SSE 解析器写最小单测（纯函数：padding/帧切分/[DONE]/reset/坏帧），不铺开测试框架。
- 手工联调（黄金路径）：后端 :8001 + 前端 :5173 起服务，真实提问，逐项核对：流式输出、GFM 表格、thinking 折叠、来源/工具徽标、耗时、会话落库与列表、反馈上报、停止按钮、provider 切换、联网开关、429/503 提示。

## 6. 文档同步

- 《技术栈.md》§1/§8 加回 react-markdown/remark-gfm 与 vitest；"尚未接入后端 API"注记改为已接入。
- 《开发文档.md》§2.3 补 CORS 配置键。
- CLAUDE.md 状态段（前端尚未接入后端 API 的表述）更新。

## 验收标准

1. `uv run pytest` 全绿（含 SSE 口径测试与新增接线测试）。
2. 手工联调黄金路径全部通过（§5 清单）。
3. 前端 `npm run build` 通过（tsc 无错）。
4. done 帧字段与 CLAUDE.md 红线一致（tool_name 单字符串、phase_times 数组对）。
5. 落库消息形状符合 PersistedMessage 契约（无 thinking/计时字段）。
