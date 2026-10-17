/**
 * 唯一网络层：所有对后端的请求都从这里走。
 *
 * - `API_BASE`：默认直连 :8001（Vite 无 dev 代理，SSE 无缓冲风险，与 CLAUDE.md 一致）。
 * - `askStream`：POST /api/chat/stream + ReadableStream 手写 SSE 解析
 *   （2KB padding、`data:` 前缀、`[DONE]` 尾帧、`reset` 清屏、坏帧 JSON 容错）。
 * - 其余封装：健康检查、会话列表/详情/删除、点赞/点踩反馈。
 */

export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8001'

/** done 帧里的知识库来源 */
export interface SourceItem {
  question?: string
  answer?: string
  score?: number
  /** 与查询的稠密余弦相似度（本地知识库来源才有） */
  relevance?: number
  /** 联网来源才有 */
  url?: string
}

/** 联网搜索来源（`/api/web/search` 与 Agent 的 web_sources 共用形状） */
export interface WebSourceItem {
  title: string
  url: string
  content: string
  /** 相关度 0~1（可能为空：两路都没给出分数时） */
  score?: number | null
  /** 发布时间（智谱提供，Tavily 不给） */
  published?: string
  /** 数据源："tavily" | "zhipu" */
  source?: string
  /** 分数口径："engine" 引擎自带 / "cosine" 本地补算 */
  score_kind?: string
}

/** 联网搜索模式：auto 本地不够用时自动联网 / on 强制 / off 只用本地 */
export type WebSearchMode = 'auto' | 'on' | 'off'

/** 自动兜底的判定结果（后端 done 帧的 web_fallback） */
export interface WebFallbackInfo {
  enough: boolean
  best_relevance: number | null
  threshold: number
  hits_count?: number
  reason: string
  /** 是否做了 LLM 判读；false 表示结论只基于相似度，可信度较低 */
  judged?: boolean
  judge_reason?: string
}

export interface WebStatus {
  providers: { tavily: boolean; zhipu: boolean; any: boolean }
  min_relevance: number
  default_top_k: number
  modes: Record<WebSearchMode, string>
}

export interface WebSearchResult {
  results: WebSourceItem[]
  answer: string
  providers_used: string[]
  errors: { provider: string; reason: string }[]
  /** 被 min_score 过滤掉的条数 */
  filtered: number
}

/** 联网数据源可用性与判定阈值 */
export async function fetchWebStatus(): Promise<WebStatus> {
  const resp = await fetch(`${API_BASE}/api/web/status`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as WebStatus
}

/** 直接联网搜索（不经 Agent），返回带相关性评分的结果 */
export async function searchWeb(
  query: string,
  opts: { maxResults?: number; minScore?: number; providers?: string[] } = {},
): Promise<WebSearchResult> {
  const resp = await fetch(`${API_BASE}/api/web/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      max_results: opts.maxResults ?? 5,
      min_score: opts.minScore ?? null,
      providers: opts.providers ?? null,
    }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as WebSearchResult
}

/** 判断本地知识库能否回答该问题（两层判定，含一次 LLM 判读，约 5~20s） */
export async function checkLocalCoverage(query: string): Promise<WebFallbackInfo> {
  const resp = await fetch(`${API_BASE}/api/web/coverage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as WebFallbackInfo
}

/** done 帧（后端契约，type 键见开发文档 §6.4） */
export interface DoneEvent {
  type: 'done'
  sources: SourceItem[]
  web_sources: SourceItem[]
  tool_called: boolean
  /** 单字符串：最后一个执行完成的工具名 */
  tool_name: string
  elapsed_ms: number
  /**
   * 阶段耗时：`[阶段名, 该阶段毫秒, 该阶段开始的相对毫秒]`。
   *
   * 第三项是**偏移**（相对本轮问答开始）。有了它前端才能画时间轴、
   * 也才能看出"两次 LLM 调用之间有 3s 空档"这类问题——
   * 只有耗时的话只能列出一串数字，看不出先后与间隔。
   *
   * 旧后端（或降级路径）可能只给前两项，故第三项标可选。
   */
  phase_times: [string, number, number?][]
  /**
   * 分阶段合计（毫秒）。
   *
   * 与 `elapsed_ms` 的差额 = 没被任何阶段覆盖的时间
   * （初始化、会话记忆检索、答案核对、SSE 收尾等）。
   * 显式给出而不是前端自己 reduce：避免两端对"合计"的口径不一致。
   */
  phase_total_ms?: number
  /**
   * 自动兜底的判定结果；`null` 表示未启用自动模式。
   * 前端据此说明"为什么这次联网了"。
   */
  web_fallback?: WebFallbackInfo | null
  /** LLM 调用明细与 token 用量（本轮问答的全部 LLM 调用）。 */
  llm_trace?: LLMTrace | null
  /** 每个工具的执行耗时（与 phase_times 的"整阶段合计"互补）。 */
  tool_timings?: ToolTiming[]
  /** RAG 检索耗时（含各步骤分解）。
   *
   *  ★ 与 phase_times 里的"检索"互补
   *  -----------------------------
   *  那是**工具层的笼统阶段**，这是**检索内部的真实构成**。
   *  实测一次检索 32.6s 里查询向量化占 28.6s —— 只给总数看不出该优化哪一步。 */
  rag_timing?: RagTimingSummary | null
  /** 图表数据：后端从**工具原始输出**提取的结构化数值。
   *
   *  ★ 前端不解析回答正文来画图
   *  ----------------------
   *  回答是模型写的自然语言，数字可能被改写或省略；只有后端见过工具原文。
   *  故这里只消费后端给的结构化数据。 */
  charts?: AnswerChart[]
  /** 本轮是否以「向用户追问」结束（此时没有正文，来源与工具均为空）。
   *
   *  有这个标志，前端才能区分"答完了但内容为空"（异常）与
   *  "本轮就是在问问题"（正常），不必靠猜。 */
  clarified?: boolean
}

/** LLM 调用追踪聚合（对应后端 done 帧的 `llm_trace`）。
 *
 *  ★ 所有 token 字段都可能是 `null`
 *  ----------------------------
 *  厂商不返回 `usage` 时为 null，**不是 0**——`0` 会被读成"没花 token"，
 *  而事实是"没拿到这个数据"。故展示层要区分"0"与"未知"。
 */
export interface LLMTrace {
  call_count: number
  failed_count: number
  /** 超出记录上限而丢弃的条数（>0 说明本轮调用异常多） */
  dropped_count: number
  total_ms: number
  max_call_ms: number
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
  /** prompt 缓存命中 token —— DeepSeek 对命中部分打折计费，是省钱信号 */
  cache_hit_tokens: number | null
  cache_miss_tokens: number | null
  reasoning_chars: number
  content_chars: number
  /** 按调用类型分组的次数与耗时（decision / stream / thinking） */
  by_kind: Record<string, { count: number; ms: number }>
  calls: LLMCallRecord[]
}

/** 单次 LLM 调用的观测记录。 */
export interface LLMCallRecord {
  kind: string
  provider: string
  model: string
  elapsed_ms: number
  ok: boolean
  error: string
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
  cache_hit_tokens: number | null
  cache_miss_tokens: number | null
  /** 思维链长度：用于区分"慢是因为在想"还是"慢是因为输出长" */
  reasoning_chars: number
  content_chars: number
  /** 该次调用涉及的工具名（决策轮解析出的） */
  tools: string[]
}

/** RAG 检索耗时汇总（后端 
ag/timing_summary.py 产出）。 */
export interface RagTimingSummary {
  /** 本轮做了几次检索（多轮 ReAct 可能多次） */
  count: number
  /** **各次检索耗时之和**（不是墙钟；缓存命中记 0，并发检索会重叠） */
  total_ms: number
  /** 单次最长，用于识别"某一次特别慢" */
  max_ms: number
  /** 按步骤聚合：sum 为总耗时、max 为单次最长。已按耗时降序 */
  steps: RagTimingStep[]
  /** 步骤合计与 total 的差额 = 未打点开销（分类、参数解析、结果组装等） */
  unaccounted_ms: number
  /** 命中缓存的检索次数 */
  cache_hits: number
  /** 是否全部命中缓存（此时 steps 为空、total 为 0） */
  all_cached: boolean
  /** 精排启用次数 */
  reranked: number
  /** 精排失败次数（失败会退回混合检索顺序，质量略降） */
  rerank_failed: number
  /** 召回条数（只统计未命中缓存的那次） */
  result_count: number
  /** 问题分类（决定检索参数：rrf_k、召回深度） */
  categories: string[]
}

/** RAG 检索的一个步骤耗时（供面板渲染）。 */
export interface RagTimingStep {
  step: string
  label: string
  /** 本轮该步骤总耗时（多次检索累加） */
  sum: number
  /** 单次最长 */
  max: number
}

/** 单张图表的数据（后端 charts.py 的 extract_charts 产出）。 */
export interface AnswerChart {
  /** 目前只有 "bar"（横向条形图） */
  kind: string
  /** 图表标题，取自工具输出里表格上方那行（如"业务数据库查询结果（共 6 条）"） */
  title: string
  /** 数值轴的名字（如"数量"、"数值"） */
  value_label: string
  /** 类别轴的名字（如"标的物"、"所在地"、"供应商"） */
  label_label: string
  /** 最多 12 条（后端截断，并在 note 里说明） */
  data: { label: string; value: number }[]
  /** 截断等说明；空串表示无 */
  note: string
  /** 产出该数据的工具名（便于排查"哪个工具的格式变了"） */
  source_tool?: string
}

/** 单个工具的执行耗时。 */
export interface ToolTiming {
  round: number
  name: string
  ms: number
  ok: boolean
  sources: number
  chars: number
}

export interface SseEvent {
  type: string
  content?: string
  [key: string]: unknown
}

export interface AskOptions {
  question: string
  history: { role: string; content: string }[]
  /** 旧字段，保留兼容（true 等价于 web_search_mode: 'on'） */
  web_search_enabled?: boolean
  /** 联网模式：auto 本地不够用时自动联网 / on 强制 / off 只用本地 */
  web_search_mode?: WebSearchMode
  provider: string
  deep_thinking_enabled: boolean
}

export interface AskHandlers {
  onStatus?: (text: string) => void
  onToken?: (text: string) => void
  onThinking?: (text: string) => void
  onReset?: () => void
  /** 首轮追问：模型判定信息不足，需要用户补充后才能作答。
   *
   *  ★ 收到本事件后**不要再等 `onDone`**
   *  ------------------------------
   *  追问是本轮的终止事件（后端确实还会补发一个 `done`，但那是为了让流式
   *  状态收尾，其中 `clarified: true`、没有正文）。界面应当立刻从"生成中"
   *  切到"等待用户回答"，否则用户会对着几个问题却无从下口。
   */
  onClarify?: (ev: ClarifyEvent) => void
  onDone?: (done: DoneEvent) => void
  onError?: (text: string) => void
}

/** 一条追问问题。 */
export interface ClarifyQuestion {
  question: string
  /** 候选项：非空时前端渲染成可点选按钮，比让用户手打更快更准 */
  options: string[]
}

/** 追问事件（对应后端 SSE 的 `type: "clarify"`）。 */
export interface ClarifyEvent {
  /** 最多 3 条（后端在 Agent 层夹取，前端不必再截断） */
  questions: ClarifyQuestion[]
  /** 一句话说明为什么必须问清楚，用于让追问不显得突兀 */
  reason: string
}

export interface ConversationSummary {
  id: number
  session_id: string
  title: string
  created_at: string
}

/** 会话列表项（带消息数与预览），供「跨会话恢复」使用 */
export interface ConversationDetail extends ConversationSummary {
  /** 该会话的消息条数；0 表示用户点了新建但没提问 */
  message_count: number
  /** 首条**用户**消息的预览（首条 assistant 通常是欢迎语，各会话都一样，无区分度） */
  preview: string
  /** 最后一条消息时间，用于「最近聊过」排序 */
  last_message_at: string | null
}

/** 对话历史搜索结果 */
export interface ConversationSearchHit extends ConversationDetail {
  /** 命中位置：标题 / 消息 / 两者都有 */
  matched_in: 'title' | 'message' | 'both'
  /** 命中处的上下文片段（以关键词为中心，两端省略号） */
  snippet: string
  /** pg_trgm 相似度，用于排序 */
  relevance: number | null
}

/** 一张图的视觉解析结果（**只存文本，不存原图**）。 */
export interface ImageAnalysis {
  /** 原始文件名（合并进问题文本时用于标注"哪张图"） */
  name: string
  /** 解析出的文本；空串表示这张图没解析出内容 */
  analysis: string
  /** 产出该结果的视觉模型名（便于回溯"当时用的哪个模型"） */
  model?: string
}

export interface PersistedMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[] | null
  toolName?: string | null
  /** 该条用户消息附带的图片解析文本；无图为 null/undefined */
  imageAnalyses?: ImageAnalysis[] | null
}

/** 非 2xx 抛出的错误，带后端 detail 与状态码 */
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function parseError(resp: Response): Promise<string> {
  try {
    const data = await resp.json()
    if (data && typeof data.detail === 'string') return data.detail
  } catch {
    /* 响应体不是 JSON，回退通用文案 */
  }
  return `请求失败（HTTP ${resp.status}）`
}

function assertOk(resp: Response): Promise<void> {
  if (!resp.ok) return Promise.reject(new ApiError(resp.status, '请求失败'))
  return Promise.resolve()
}

// ---------------------------------------------------------------------------
// SSE 解析（纯函数，供 vitest 最小单测）
// ---------------------------------------------------------------------------

/**
 * 把一段 SSE 文本解析为事件数组。
 * - 跳过注释行（`: ` 开头）、空行；
 * - 只认 `data: ` 前缀行；
 * - `[DONE]` 尾帧不产生事件；
 * - 坏帧 JSON 直接跳过（容错，不中断后续帧）。
 */
export function parseSse(text: string): SseEvent[] {
  const events: SseEvent[] = []
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trimEnd()
    if (!line.startsWith('data:')) continue
    const payload = line.slice('data:'.length).trim()
    if (!payload || payload === '[DONE]' || payload.startsWith(':')) continue
    try {
      const parsed = JSON.parse(payload) as SseEvent
      if (parsed && typeof parsed.type === 'string') events.push(parsed)
    } catch {
      // 坏帧容错：忽略
    }
  }
  return events
}

function dispatchEvent(ev: SseEvent, handlers: AskHandlers): void {
  switch (ev.type) {
    case 'status':
      handlers.onStatus?.(ev.content ?? '')
      break
    case 'token':
      handlers.onToken?.(ev.content ?? '')
      break
    case 'thinking':
      handlers.onThinking?.(ev.content ?? '')
      break
    case 'reset':
      handlers.onReset?.()
      break
    case 'clarify':
      handlers.onClarify?.(ev as unknown as ClarifyEvent)
      break
    case 'done':
      handlers.onDone?.(ev as unknown as DoneEvent)
      break
    case 'error':
      handlers.onError?.(ev.content ?? '回答生成失败，请稍后重试')
      break
    default:
      break
  }
}

// ---------------------------------------------------------------------------
// 流式问答
// ---------------------------------------------------------------------------

export async function askStream(
  opts: AskOptions,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: opts.question,
      history: opts.history,
      // 三态模式优先；旧字段仍带上以兼容只认旧字段的后端版本
      web_search_mode: opts.web_search_mode ?? 'auto',
      web_search_enabled: opts.web_search_enabled ?? false,
      provider: opts.provider,
      deep_thinking_enabled: opts.deep_thinking_enabled,
    }),
    signal,
  })

  if (!resp.ok) {
    const detail = await parseError(resp)
    throw new ApiError(resp.status, detail)
  }
  if (!resp.body) throw new ApiError(0, '浏览器不支持流式响应')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  // 分批消费：累积到完整帧（\n\n 分隔）再解析，尾部不完整帧留在 buffer
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const ev of parseSse(block)) dispatchEvent(ev, handlers)
    }
  }
  // 流结束：先让解码器吐出**最后一个不完整的多字节序列**，再解析残留 buffer
  //
  // ★ 为什么必须补这一行（中文回答会真丢字）
  // ------------------------------------
  // `decode(value, { stream: true })` 会把"末尾可能被切断的多字节字符"留在
  // 解码器内部，等下一次调用再补。流到此结束、不再有下一次 —— 若最后一个
  // 分块正好切在一个汉字的中间（UTF-8 里 3 字节），那半个字符就永远丢了，
  // 表现为**回答最后一个字消失**。不传 `{stream:true}` 的这次调用即"最终
  // flush"，会把内部残留字节吐出来。
  buffer += decoder.decode()
  if (buffer.trim()) {
    for (const ev of parseSse(buffer)) dispatchEvent(ev, handlers)
  }
}

// ---------------------------------------------------------------------------
// 健康检查 / 会话 / 反馈
// ---------------------------------------------------------------------------

/**
 * 健康检查响应。
 *
 * ⚠️ `points_count` 的 `-1` 表示**探测失败**，与"集合为空（0）"是两回事——
 * 后端早期版本把该字段硬编码为 0，会让监控误判向量库为空。
 */
export interface HealthStatus {
  ready: boolean
  agent_ready: boolean
  graph_ready: boolean
  pg_ready: boolean
  qdrant_ready: boolean
  points_count: number
  latencies: { neo4j_ms: number; postgres_ms: number; qdrant_check_ms: number }
  qdrant_error?: string
}

export async function fetchHealth(): Promise<HealthStatus> {
  const resp = await fetch(`${API_BASE}/api/health`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as HealthStatus
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const resp = await fetch(`${API_BASE}/api/conversations`)
  await assertOk(resp)
  const data = await resp.json()
  return (data?.conversations ?? []) as ConversationSummary[]
}

/**
 * 会话列表（带消息数与预览），按**最近一条消息时间**排序。
 *
 * 与 `listConversations` 的区别：多拿 `message_count` / `preview` /
 * `last_message_at`——恢复会话时用户靠预览认人（自动标题信息量太低），
 * 靠消息数识别空会话。
 */
export async function listConversationsDetailed(limit = 50): Promise<ConversationDetail[]> {
  const resp = await fetch(`${API_BASE}/api/conversations/detailed?limit=${limit}`)
  await assertOk(resp)
  const data = await resp.json()
  return (data?.items ?? []) as ConversationDetail[]
}

/**
 * 搜索对话历史（标题 + 消息内容），按相关性降序。
 *
 * 中文搜索走后端 `ILIKE` + pg_trgm，**不用** PostgreSQL 全文检索——
 * 后者按空格分词，对中文会把整句当成一个 token（搜「香港」匹配不到）。
 */
export async function searchConversations(
  keyword: string,
  limit = 30,
): Promise<ConversationSearchHit[]> {
  const q = keyword.trim()
  if (!q) return []
  const qs = new URLSearchParams({ q, limit: String(limit) })
  const resp = await fetch(`${API_BASE}/api/conversations/search?${qs}`)
  await assertOk(resp)
  const data = await resp.json()
  return (data?.items ?? []) as ConversationSearchHit[]
}

/** 重命名会话（用户手动改标题） */
export async function renameConversation(sessionId: string, title: string): Promise<void> {
  const resp = await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}/title`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    },
  )
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
}

export async function getConversation(
  sessionId: string,
): Promise<PersistedMessage[]> {
  const resp = await fetch(`${API_BASE}/api/conversations/${sessionId}`)
  await assertOk(resp)
  const data = await resp.json()
  return (data?.messages ?? []) as PersistedMessage[]
}

/** 保存本轮新增消息，返回每条消息 id（前端取 assistant id 用于反馈） */
export async function saveConversation(
  sessionId: string,
  title: string,
  messages: {
    role: string
    content: string
    sources?: unknown
    toolName?: unknown
    /** 图片解析文本（"只存解析文本"方案；原图不入库） */
    imageAnalyses?: unknown
  }[],
): Promise<number[]> {
  const resp = await fetch(`${API_BASE}/api/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, title, messages }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return (data?.ids ?? []) as number[]
}

export async function deleteConversation(sessionId: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/conversations/${sessionId}`, {
    method: 'DELETE',
  })
  await assertOk(resp)
}

export async function sendFeedback(
  sessionId: string,
  messageId: number,
  rating: 'up' | 'down',
): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message_id: messageId, rating }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
}

// ---------------------------------------------------------------------------
// 知识图谱直查（前端"知识图谱"页，不经过 LLM）
// ---------------------------------------------------------------------------

/** 图谱节点标签（与后端 neo4j_client.ALLOWED_LABELS 对应） */
export type GraphLabel =
  | 'SubjectMatter'
  | 'Purchaser'
  | 'Supplier'
  | 'Agency'
  | 'Location'
  | 'Transaction'

/** 图谱关系类型（与后端 neo4j_client.ALLOWED_RELS 对应） */
export type GraphRelation =
  | 'HAS_TRANSACTION'
  | 'PURCHASED_BY'
  | 'SUPPLIED_BY'
  | 'AGENT_BY'
  | 'LOCATED_AT'

/** 单条图谱查询结果行——字段随 query_type 变化，故用宽松索引签名 */
export interface GraphRow {
  [key: string]: unknown
}

/**
 * 查询结果载荷。
 * - `graph_stats` 与 `entity_detail` 返回对象
 * - 其余 4 类返回对象数组
 */
export type GraphPayload = GraphRow[] | Record<string, unknown>

export interface GraphQueryResult {
  query_type: string
  data: GraphPayload
  /** 条目数：dict 记键数，list 记元素数 */
  count: number
}

export interface GraphTypeInfo {
  query_type: string
  /** 该类型是否必须提供实体名（前端据此决定输入框是否必填） */
  requires_entity: boolean
}

export interface GraphTypesResponse {
  types: GraphTypeInfo[]
  /** 图谱当前是否可用（Neo4j 已配置且能构造客户端） */
  graph_ready: boolean
}

/** 拉取可用的图谱查询类型清单（不在前端硬编码，避免与后端漂移） */
export async function fetchGraphTypes(): Promise<GraphTypesResponse> {  const resp = await fetch(`${API_BASE}/api/graph/types`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    types: (data?.types ?? []) as GraphTypeInfo[],
    graph_ready: Boolean(data?.graph_ready),
  }
}

/**
 * 执行一次图谱查询。`entity` 仅在需要实体的查询类型下必填。
 *
 * 错误语义（与后端状态码对齐）：
 *   400 -> 查询类型/参数不合法（ApiError.status === 400，detail 含可用清单）
 *   429 -> 触发限流
 *   503 -> 图谱不可用（Neo4j 未启动或配置错误）
 */
export async function queryGraph(params: {
  query_type: string
  entity?: string
  limit?: number
}): Promise<GraphQueryResult> {
  const resp = await fetch(`${API_BASE}/api/graph/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query_type: params.query_type,
      entity: params.entity ?? '',
      limit: params.limit ?? 20,
    }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    query_type: String(data?.query_type ?? params.query_type),
    data: (data?.data ?? []) as GraphPayload,
    count: Number(data?.count ?? 0),
  }
}

// ---------------------------------------------------------------------------
// 图片解析（智谱视觉模型）
// ---------------------------------------------------------------------------

export interface VisionResult {
  /** 模型输出的 Markdown 文本（含表格/清单） */
  analysis: string
  /** 实际使用的模型名，例如 glm-5.3-flashx */
  model: string
}

/**
 * 上传图片交给智谱视觉模型解析。
 *
 * `imageBase64` 接受纯 base64 或完整 data URL（`lib/image.ts` 的 prepareImage
 * 产出的是后者）。后端会按字节判断真实格式，不依赖前端自报的 MIME。
 *
 * 错误语义：413 图片过大 ｜ 429 限流 ｜ 503 未配置 ZHIPU_API_KEY ｜ 500 解析失败
 */
export async function analyzeImage(
  imageBase64: string,
  prompt = '',
  signal?: AbortSignal,
): Promise<VisionResult> {
  const resp = await fetch(`${API_BASE}/api/vision`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64, prompt }),
    signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    analysis: String(data?.analysis ?? ''),
    model: String(data?.model ?? ''),
  }
}

// ---------------------------------------------------------------------------
// 图片比价（视觉提取商品 → 多平台报价 → 择优建议）
// ---------------------------------------------------------------------------

/** 从图片提取出的商品信息 */
export interface CompareProduct {
  product_name: string
  brand: string
  model: string
  spec: string
  category: string
  sku: string
  /** 模型自评的提取准确度 0-1 */
  confidence: number
  /** 图片全部文字（可选，请求时用 include_raw_text 控制） */
  raw_text?: string
}

/** 一条平台报价 */
export interface CompareQuote {
  platform: string
  platform_key: string
  /** 综合电商 / 工业品 */
  kind: string
  price: number | null
  title: string
  url: string
  snippet: string
  has_price: boolean
  /**
   * 可信度：
   *   normal          正常参与比价
   *   suspicious_high/low  价格偏离较大，需人工核实
   *   extreme         判定为不同商品，已排除
   */
  trust: 'normal' | 'suspicious_high' | 'suspicious_low' | 'extreme'
  note: string
}

/** 单个平台的检索结果 */
export interface ComparePlatformResult {
  platform: string
  platform_key: string
  kind: string
  ok: boolean
  error: string
  quotes: CompareQuote[]
  /** 因型号/规格不符被排除的条目（让用户知道为什么没有更便宜的选项） */
  rejected: { title: string; url: string; reason: string }[]
  rejected_count: number
}

export interface CompareRecommendation {
  winner: CompareQuote | null
  /** 参与比价的报价数（已排除 extreme） */
  priced_count: number
  /** 被排除的报价数 */
  excluded_count: number
  summary: string
  /** **必读**：能力边界与风险提示，前端必须展示 */
  caveats: string[]
}

export interface PriceCompareResult {
  product: CompareProduct
  /** 实际用于检索的关键词 */
  query: string
  platforms: ComparePlatformResult[]
  recommendation: CompareRecommendation
}

/**
 * 上传商品图片做多平台比价。
 *
 * 耗时约 5~20 秒（视觉解析 + 并行检索 7 个平台），故调用方要有加载态。
 * 错误语义：400 图片无法解析/未识别出商品 ｜ 413 图片过大 ｜
 * 429 限流 ｜ 503 比价服务未就绪 ｜ 500 服务异常。
 */
export async function comparePrice(
  imageBase64: string,
  hint = '',
  signal?: AbortSignal,
): Promise<PriceCompareResult> {
  const resp = await fetch(`${API_BASE}/api/price-compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64, hint }),
    signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    product: data?.product ?? ({} as CompareProduct),
    query: String(data?.query ?? ''),
    platforms: (data?.platforms ?? []) as ComparePlatformResult[],
    recommendation: data?.recommendation ?? {
      winner: null,
      priced_count: 0,
      excluded_count: 0,
      summary: '',
      caveats: [],
    },
  }
}

// ---------------------------------------------------------------------------
// 标讯智能提取（公告文本 / 截图 → 17 个结构化字段）
// ---------------------------------------------------------------------------

/** 一个标讯字段的定义（由后端下发，前端不硬编码） */
export interface BidFieldDef {
  key: string
  label: string
  type: 'text' | 'money' | 'datetime'
  hint: string
  required: boolean
}

export interface BidSchemaResponse {
  fields: BidFieldDef[]
  count: number
}

/** 抽取结果：键为字段 key，值为归一化后的数据（金额是数字、日期是字符串） */
export type BidRecord = Record<string, string | number | null>

export interface ExtractMeta {
  source: 'text' | 'image'
  provider: string
  model: string
  input_chars: number
  /** 输入是否被截断（超长公告只取前 N 字） */
  truncated: boolean
  /** 字段完整度 0-1 */
  completeness: number
  /** 被清洗的可疑值说明，需展示给用户 */
  warnings: string[]
}

export interface ExtractSaveResult {
  saved: boolean
  id: number | null
  reason?: string
  skipped_columns?: string[]
}

export interface ExtractResult {
  record: BidRecord
  meta: ExtractMeta
  /** 仅在请求时 save=true 才返回 */
  save?: ExtractSaveResult
}

/** 拉取字段清单（后端为唯一事实来源，前端不硬编码 17 个字段） */
export async function fetchExtractSchema(): Promise<BidSchemaResponse> {
  const resp = await fetch(`${API_BASE}/api/extract/schema`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    fields: (data?.fields ?? []) as BidFieldDef[],
    count: Number(data?.count ?? 0),
  }
}

/**
 * 从公告文本或截图提取结构化字段。
 *
 * 耗时：文本模式走云端 LLM 约 10 秒；图片模式走视觉模型约 5 秒。
 * 错误语义：400 输入不合法/模型输出无法解析 ｜ 413 图片过大 ｜
 * 429 限流 ｜ 503 服务未就绪 ｜ 500 模型调用失败。
 */
export async function extractNotice(params: {
  source: 'text' | 'image'
  text?: string
  imageBase64?: string
  hint?: string
  save?: boolean
  signal?: AbortSignal
}): Promise<ExtractResult> {
  const resp = await fetch(`${API_BASE}/api/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source: params.source,
      text: params.text ?? '',
      image_base64: params.imageBase64 ?? '',
      hint: params.hint ?? '',
      save: params.save ?? false,
    }),
    signal: params.signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return {
    record: (data?.record ?? {}) as BidRecord,
    meta: data?.meta as ExtractMeta,
    save: data?.save as ExtractSaveResult | undefined,
  }
}

// ---------------------------------------------------------------------------
// 集采分析（PostgreSQL 多维聚合）
// ---------------------------------------------------------------------------

/**
 * 金额分位数。
 *
 * ⚠️ **本数据集必须看 `median` 而不是 `mean`**：金额横跨 6 个数量级
 * （办公用品几千元 ↔ 基建工程数亿元），均值是中位数的 500 倍以上。
 * 故后端把它俩并排返回，前端也必须并排展示。
 */
export interface AmountStats {
  count: number
  min: number | null
  p25: number | null
  median: number | null
  p75: number | null
  p90: number | null
  max: number | null
  mean: number | null
  sum: number | null
}

export interface AnalysisOverview {
  total_rows: number
  amount_stats: AmountStats
  suspicious_count: number
  excluded_suspicious: boolean
  dimension_cardinality: Record<string, number>
  time_range: { min: string | null; max: string | null; months: number | null }
  /** **必读**：数据集局限（单月数据/均值不可用/category 语义混杂），前端必须展示 */
  data_caveats: string[]
}

export interface RankingItem {
  name: string
  count: number
  median_amount: number | null
  mean_amount: number | null
  total_amount: number | null
  min_amount: number | null
  max_amount: number | null
}

export interface RankingResult {
  dimension: string
  dimension_label: string
  order_by: string
  distinct_total: number
  returned: number
  items: RankingItem[]
}

export interface DistributionBucket {
  label: string
  count: number
  mean_amount: number | null
  percent: number
}

export interface DistributionResult {
  total: number
  buckets: DistributionBucket[]
  note: string
}

export interface PriceProfile {
  subject: string
  matched: boolean
  count: number
  message?: string
  percentiles?: { p10: number | null; p25: number | null; median: number | null; p75: number | null; p90: number | null }
  min?: number | null
  max?: number | null
  mean?: number | null
  suggested_range?: { low: number | null; high: number | null }
  top_records?: {
    project_name: string
    purchaser: string
    supplier: string
    amount: number | null
    bid_time: string | null
    location_city: string | null
  }[]
  top_suppliers?: { name: string; count: number; median_amount: number | null }[]
  advice?: string[]
}

export interface SuspiciousRecord {
  id: number
  subject_matter: string
  purchaser: string
  supplier: string
  amount: number | null
  bid_time: string | null
  category: string
  /** 被判可疑的具体原因（让用户能自行判断是否认同） */
  reason: string
}

export interface DataQuality {
  total_rows: number
  empty_rates: { column: string; empty_count: number; empty_percent: number }[]
  suspicious_records: SuspiciousRecord[]
  suspicious_count: number
  suspicious_rules: Record<string, unknown>
  category_values: { name: string; count: number }[]
  notes: string[]
}

export interface DimensionOption {
  key: string
  label: string
}

/** 总览：条数、分位数、维度基数、数据局限提示 */
export async function fetchAnalysisOverview(excludeSuspicious = true): Promise<AnalysisOverview> {
  const resp = await fetch(`${API_BASE}/api/analysis/overview?exclude_suspicious=${excludeSuspicious}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as AnalysisOverview
}

/** 可用分组维度（后端下发，前端不硬编码） */
export async function fetchAnalysisDimensions(): Promise<DimensionOption[]> {
  const resp = await fetch(`${API_BASE}/api/analysis/dimensions`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return (data?.dimensions ?? []) as DimensionOption[]
}

/** 分组排行 */
export async function fetchAnalysisRanking(params: {
  dimension: string
  limit?: number
  orderBy?: 'count' | 'amount'
  excludeSuspicious?: boolean
}): Promise<RankingResult> {
  const qs = new URLSearchParams({
    dimension: params.dimension,
    limit: String(params.limit ?? 20),
    order_by: params.orderBy ?? 'count',
    exclude_suspicious: String(params.excludeSuspicious ?? true),
  })
  const resp = await fetch(`${API_BASE}/api/analysis/ranking?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as RankingResult
}

/** 金额数量级分布 */
export async function fetchAnalysisDistribution(excludeSuspicious = true): Promise<DistributionResult> {
  const resp = await fetch(`${API_BASE}/api/analysis/distribution?exclude_suspicious=${excludeSuspicious}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as DistributionResult
}

/** 报价策略：某标的物的价格画像 + 建议区间 */
export async function fetchPriceProfile(params: {
  subject: string
  excludeSuspicious?: boolean
  topN?: number
  signal?: AbortSignal
}): Promise<PriceProfile> {
  const resp = await fetch(`${API_BASE}/api/analysis/price-profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      subject: params.subject,
      exclude_suspicious: params.excludeSuspicious ?? true,
      top_n: params.topN ?? 10,
    }),
    signal: params.signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as PriceProfile
}

/** 数据质量：空值率 + 可疑金额明细 */
export async function fetchDataQuality(): Promise<DataQuality> {
  const resp = await fetch(`${API_BASE}/api/analysis/data-quality`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as DataQuality
}

export interface RecentRecord {
  project_name: string
  purchaser: string
  supplier: string
  amount: number | null
  bid_time: string | null
  subject_matter: string
  province: string
  location_city: string
  source: string
}

export interface RecentRecordsResult {
  items: RecentRecord[]
  returned: number
  /** ⚠️ 数据覆盖范围说明：单月数据，"最近"不等于"实时" */
  coverage: { min: string | null; max: string | null; months: number | null; caveat: string }
}

/** 最近的中标记录（工作台动态） */
export async function fetchRecentRecords(limit = 8): Promise<RecentRecordsResult> {
  const resp = await fetch(`${API_BASE}/api/analysis/recent?limit=${limit}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as RecentRecordsResult
}

// ---------------------------------------------------------------------------
// 知识图谱关系可视化（子图抽取）
// ---------------------------------------------------------------------------

export interface GraphVizNode {
  id: string
  label: string
  label_cn: string
  name: string
  /** 在原图中的连接数（非子图内度数），用于决定节点半径 */
  degree: number
  is_center?: boolean
  /** 仅中心节点有：交易笔数 / 合计金额 / 金额中位数 */
  txn_count?: number
  amount_sum?: number
  amount_median?: number
}

export interface GraphVizEdge {
  source: string
  target: string
  rel_type: string
  rel_cn: string
}

export interface SubgraphResponse {
  center: GraphVizNode
  nodes: GraphVizNode[]
  edges: GraphVizEdge[]
  /** 是否因节点数上限被截断（前端必须提示，否则用户以为图就是全部） */
  truncated: boolean
  total_neighbors: number
  shown_nodes: number
  max_nodes: number
  legend: {
    node_labels: { key: string; cn: string }[]
    /** **必读**：说明哪些节点未画入、为什么边不含金额 */
    notes: string[]
  }
}

export interface GraphHubEntity {
  label: string
  label_cn: string
  name: string
  degree: number
}

export interface GraphOverview {
  total_nodes: number
  total_relationships: number
  by_label: { key: string; cn: string; count: number }[]
  hub_entities: GraphHubEntity[]
  allowed_labels: { key: string; cn: string }[]
}

/** 图谱规模总览 + 枢纽实体（前端"推荐探索"列表） */
export async function fetchGraphOverview(): Promise<GraphOverview> {
  const resp = await fetch(`${API_BASE}/api/graph/overview`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as GraphOverview
}

/** 实体搜索（自动补全） */
export async function searchGraphEntities(keyword: string, limit = 15): Promise<GraphHubEntity[]> {
  const qs = new URLSearchParams({ keyword, limit: String(limit) })
  const resp = await fetch(`${API_BASE}/api/graph/search?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  const data = await resp.json()
  return (data?.items ?? []) as GraphHubEntity[]
}

/** 抽子图（力导向图数据源） */
export async function fetchSubgraph(
  label: string,
  name: string,
  maxNodes = 40,
  signal?: AbortSignal,
): Promise<SubgraphResponse> {
  const qs = new URLSearchParams({ label, name, max_nodes: String(maxNodes) })
  const resp = await fetch(`${API_BASE}/api/graph/subgraph?${qs}`, { signal })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as SubgraphResponse
}

// ---------------------------------------------------------------------------
// 物料识别（物料名归一化与归并）
// ---------------------------------------------------------------------------

/** 同一标准物料的一个原始写法 */
export interface MaterialVariant {
  name: string
  count: number
}

export interface MaterialMatchItem {
  standard_name: string
  /** 归并到该标准名的全部原始写法及各自次数 */
  matched_variants: MaterialVariant[]
  total_count: number
  /** 金额中位数（**不是均值**——金额跨 6 个数量级，均值不具代表性） */
  median_amount: number | null
  score: number
  /** 匹配理由：精确匹配 / 归一化后匹配 / 包含匹配 / 字符重叠 */
  match_reason: string
}

export interface MaterialMatchResult {
  query: string
  normalized_query: string
  matched: boolean
  total_candidates: number
  returned: number
  items: MaterialMatchItem[]
  note?: string
}

export interface MaterialStatsResult {
  /** 归并后的标准物料数 */
  standard_total: number
  /** 归并前的原始写法数 */
  raw_total: number
  /** 压缩率（同一物料的写法数占比，**不是数据脏污率**） */
  compression_ratio: number
  /** 由 ≥2 种写法归并而成的标准物料数 */
  merged_groups: number
  merged_rows: number
  table_rows: number
  empty_subject_rows: number
  top_materials: {
    standard_name: string
    total_count: number
    variant_count: number
    median_amount: number | null
  }[]
  notes: string[]
}

export interface MaterialDetailResult {
  standard_name: string
  matched?: boolean
  message?: string
  total_count?: number
  variant_count?: number
  variants?: MaterialVariant[]
  percentiles?: {
    p10: number | null
    p25: number | null
    median: number | null
    p75: number | null
    p90: number | null
  }
  min_amount?: number | null
  max_amount?: number | null
  mean_amount?: number | null
  top_suppliers?: { name: string; count: number; median_amount: number | null }[]
  top_purchasers?: { name: string; count: number; median_amount: number | null }[]
  top_regions?: { name: string; count: number }[]
  advice?: string[]
}

export interface MaterialSuggestion {
  a: string
  b: string
  score: number
  count_a: number
  count_b: number
  reason: string
}

export interface MaterialSuggestionsResult {
  total_candidates: number
  returned: number
  items: MaterialSuggestion[]
  rules?: string
  note?: string
}

/** 物料库总览（标准物料数 / 归并压缩率） */
export async function fetchMaterialStats(topN = 20): Promise<MaterialStatsResult> {
  const resp = await fetch(`${API_BASE}/api/material/stats?top_n=${topN}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as MaterialStatsResult
}

/** 物料模糊匹配 */
export async function matchMaterials(query: string, limit = 10): Promise<MaterialMatchResult> {
  const qs = new URLSearchParams({ query, limit: String(limit) })
  const resp = await fetch(`${API_BASE}/api/material/match?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as MaterialMatchResult
}

/** 单个标准物料详情 */
export async function fetchMaterialDetail(name: string): Promise<MaterialDetailResult> {
  const qs = new URLSearchParams({ name })
  const resp = await fetch(`${API_BASE}/api/material/detail?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as MaterialDetailResult
}

/** 疑似应归并的物料对（人工确认用） */
export async function fetchMaterialSuggestions(limit = 50): Promise<MaterialSuggestionsResult> {
  const resp = await fetch(`${API_BASE}/api/material/suggestions?limit=${limit}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as MaterialSuggestionsResult
}

// ---------------------------------------------------------------------------
// 竞争对手分析
// ---------------------------------------------------------------------------

export interface CompetitorSupplier {
  name: string
  count: number
  win_count?: number
  median_amount?: number | null
  total_amount: number | null
  mean_amount?: number | null
  first_win?: string | null
  last_win?: string | null
  regions?: string[]
  /** 按次数计算的市场份额（%） */
  share_by_count_percent?: number
  /** 按金额计算的市场份额（%） */
  share_by_amount_percent?: number
}

export interface CompetitorMarket {
  count: number
  median_amount: number | null
  p25: number | null
  p75: number | null
  mean_amount: number | null
  total_amount: number | null
  /** 均值/中位数 比值，用于提示偏态程度 */
  mean_over_median?: number
}

export interface CompetitorSuppliersResult {
  subject: string
  matched: boolean
  message?: string
  excluded_suspicious: boolean
  suspicious_count: number
  market: CompetitorMarket | null
  distinct_suppliers: number
  returned: number
  suppliers: CompetitorSupplier[]
  time_range?: { min: string | null; max: string | null }
  graph?: unknown
  graph_note?: string
  data_caveats?: string[]
}

/** HHI 分档说明（后端随数据返回，含免责说明） */
export interface HhiThreshold {
  /** 区间文案，如 "< 1500" / "1500 ~ 2500" */
  range: string
  /** 档位名，如 "竞争充分" */
  level: string
  /** 该档位对采购/投标的含义 */
  meaning: string
}

/** 集中度的一种计算口径（按金额 / 按次数），两种口径的结论可能不同 */
export interface ConcentrationBasis {
  /** 该口径的说明，如"按中标金额计算份额（市场份额的通常含义）" */
  basis: string
  hhi: number
  /** 按阈值判定出的档位名 */
  hhi_level: string
  cr3_percent: number
  cr5_percent: number
  leaders: { name: string; share_percent: number }[]
}

export interface CompetitorConcentrationResult {
  subject: string
  matched: boolean
  message?: string
  distinct_suppliers: number
  total_count: number
  median_amount: number | null
  mean_amount?: number | null
  total_amount?: number | null
  /** 按金额口径（市场份额的通常含义） */
  by_amount?: ConcentrationBasis
  /** 按次数口径（衡量"活给得频不频繁"） */
  by_count?: ConcentrationBasis
  top_suppliers?: CompetitorSupplier[]
  hhi_explanation?: { formula: string; thresholds: HhiThreshold[]; caveat: string }
  graph?: unknown
  graph_note?: string
  data_caveats?: string[]
}

/** 某标的物的供应商竞争格局 */
export async function fetchCompetitorSuppliers(params: {
  subject: string
  limit?: number
  excludeSuspicious?: boolean
  signal?: AbortSignal
}): Promise<CompetitorSuppliersResult> {
  const qs = new URLSearchParams({
    subject: params.subject,
    limit: String(params.limit ?? 20),
    exclude_suspicious: String(params.excludeSuspicious ?? true),
  })
  const resp = await fetch(`${API_BASE}/api/competitor/suppliers?${qs}`, { signal: params.signal })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as CompetitorSuppliersResult
}

/** 单供应商画像 */
export async function fetchCompetitorProfile(name: string): Promise<Record<string, unknown>> {
  const qs = new URLSearchParams({ name })
  const resp = await fetch(`${API_BASE}/api/competitor/profile?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as Record<string, unknown>
}

/** 某采购人的合作供应商格局 */
export async function fetchCompetitorPurchaser(
  purchaser: string,
  limit = 20,
): Promise<Record<string, unknown>> {
  const qs = new URLSearchParams({ purchaser, limit: String(limit) })
  const resp = await fetch(`${API_BASE}/api/competitor/purchaser?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as Record<string, unknown>
}

/** 市场集中度（CR3 / CR5 / HHI） */
export async function fetchMarketConcentration(subject: string): Promise<CompetitorConcentrationResult> {
  const qs = new URLSearchParams({ subject })
  const resp = await fetch(`${API_BASE}/api/competitor/concentration?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as CompetitorConcentrationResult
}

// ---------------------------------------------------------------------------
// 招标文件解析与抽取（PDF / Word）
// ---------------------------------------------------------------------------

export interface TenderFieldDef {
  key: string
  label: string
  type: 'text' | 'money' | 'datetime'
  hint: string
  required: boolean
}

export interface TenderSectionDef {
  key: string
  label: string
  hint: string
  item_fields: string[]
}

export interface TenderSchemaResponse {
  fields: TenderFieldDef[]
  sections: TenderSectionDef[]
  field_count: number
  section_count: number
}

/** 只解析不抽取的预览结果（秒级） */
export interface TenderParseResult {
  filename: string
  kind: 'pdf' | 'docx'
  pages: number
  total_chars: number
  truncated: boolean
  chunk_count: number
  warnings: string[]
  headings: string[]
  chunks: {
    index: number
    heading: string
    chars: number
    categories: string[]
    preview: string
  }[]
}

/** 一个列表章节的条目（字段随章节而变，见 TenderSectionDef.item_fields） */
export type TenderItem = Record<string, string | number | boolean | null>

export interface TenderParsed {
  fields: Record<string, string | number | null>
  sections: Record<string, TenderItem[]>
}

export interface TenderExtractMeta {
  provider: string
  model: string
  filename: string
  kind: string
  pages: number
  total_chars: number
  truncated: boolean
  chunk_count: number
  batch_count: number
  succeeded_batches: number
  completeness: number
  section_item_counts: Record<string, number>
  warnings: string[]
}

export interface TenderExtractResult {
  parsed: TenderParsed
  meta: TenderExtractMeta
}

/** 招标文件字段与章节定义（后端下发，前端不硬编码） */
export async function fetchTenderSchema(): Promise<TenderSchemaResponse> {
  const resp = await fetch(`${API_BASE}/api/tender/schema`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as TenderSchemaResponse
}

/**
 * 只解析文件（秒级），用于上传后先确认「文件读对了」。
 *
 * 抽取要调多次 LLM（云端 1~2 分钟），先给一次秒级预览能避免白等。
 */
export async function parseTenderDocument(file: File): Promise<TenderParseResult> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${API_BASE}/api/tender/parse`, { method: 'POST', body: form })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as TenderParseResult
}

/** 解析 + 抽取（非流式）。界面优先用流式版本以显示进度 */
export async function extractTenderDocument(file: File): Promise<TenderExtractResult> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${API_BASE}/api/tender/extract`, { method: 'POST', body: form })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as TenderExtractResult
}

export interface TenderStreamHandlers {
  onParse?: (info: TenderParseResult) => void
  onProgress?: (done: number, total: number, message: string) => void
  onDone?: (result: TenderExtractResult) => void
  onError?: (message: string) => void
}

/**
 * SSE 流式抽取。
 *
 * 用 `fetch` + `ReadableStream` 而非 `EventSource`：EventSource 只支持 GET，
 * 无法上传文件。SSE 帧格式与 `askStream` 一致（`data: {...}` + 末尾 `data: [DONE]`）。
 */
export async function extractTenderStream(
  file: File,
  handlers: TenderStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${API_BASE}/api/tender/extract/stream`, {
    method: 'POST',
    body: form,
    signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  if (!resp.body) throw new ApiError(0, '响应没有可读流')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (payload: string) => {
    const t = payload.trim()
    if (!t || t === '[DONE]') return
    let ev: Record<string, unknown>
    try {
      ev = JSON.parse(t)
    } catch {
      return // padding 帧等非 JSON 内容
    }
    const type = String(ev.type ?? '')
    if (type === 'parse') handlers.onParse?.(ev as unknown as TenderParseResult)
    else if (type === 'progress')
      handlers.onProgress?.(
        Number(ev.done ?? 0),
        Number(ev.total ?? 1),
        String(ev.message ?? ''),
      )
    else if (type === 'done') handlers.onDone?.(ev as unknown as TenderExtractResult)
    else if (type === 'error') handlers.onError?.(String(ev.content ?? '抽取失败'))
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx = buffer.indexOf('\n\n')
    while (idx >= 0) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const line of block.split('\n')) {
        const s = line.trim()
        if (s.startsWith('data:')) dispatch(s.slice(5))
      }
      idx = buffer.indexOf('\n\n')
    }
  }
  // 收尾：先做解码器最终 flush（见 `askStream` 同一处的说明：否则末尾被
  // 切断的汉字会丢），再解析最后一块可能没有以空行结束的帧
  buffer += decoder.decode()
  for (const line of buffer.split('\n')) {
    const s = line.trim()
    if (s.startsWith('data:')) dispatch(s.slice(5))
  }
}

// ---------------------------------------------------------------------------
// 招标文件持久化与截止日期监控
// ---------------------------------------------------------------------------

export interface TenderSaveResult {
  saved: boolean
  id: number | null
  /** 同项目编号此前已保存的那份的 id（用于提示"是否做变更对比"） */
  duplicate_of?: number | null
  reason?: string
}

/** 保存抽取结果（存下来才能做截止监控与变更追踪） */
export async function saveTenderResult(
  parsed: TenderParsed,
  meta: Record<string, unknown>,
): Promise<TenderSaveResult> {
  const resp = await fetch(`${API_BASE}/api/tender/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ parsed, meta }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as TenderSaveResult
}

/** 紧急度分档：已过期 / 3天内 / 7天内 / 30天内 / 未知 */
export type Urgency = 'expired' | 'urgent' | 'soon' | 'normal' | 'unknown'

export interface TenderDocumentSummary {
  id: number
  filename: string
  kind: string | null
  project_name: string | null
  project_id: string | null
  purchaser: string | null
  budget: number | null
  max_price: number | null
  deposit: number | null
  bid_deadline: string | null
  open_time: string | null
  evaluation_method: string | null
  pages: number | null
  completeness: number | null
  created_at: string | null
  /** 距截止的剩余天数（负数表示已过期；null 表示未抽到截止时间） */
  days_left: number | null
  urgency: Urgency
}

export interface TenderDocumentListResult {
  items: TenderDocumentSummary[]
  count: number
}

/** 已解析的招标文件列表 */
export async function fetchTenderDocuments(limit = 50): Promise<TenderDocumentListResult> {
  const resp = await fetch(`${API_BASE}/api/tender/documents?limit=${limit}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as TenderDocumentListResult
}

/** 删除一份文件记录 */
export async function deleteTenderDocument(id: number): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/tender/documents/${id}`, { method: 'DELETE' })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
}

export interface DeadlineResult {
  items: TenderDocumentSummary[]
  expired: TenderDocumentSummary[]
  summary: {
    total_documents: number
    with_deadline: number
    window_days: number
    buckets: Record<string, number>
  }
  /** **必读**：本监控只覆盖已上传的文件，不是全网监控 */
  coverage: string
}

/** 未来 N 天内截止的项目（截止日期监控） */
export async function fetchDeadlines(days = 30, includeExpired = true): Promise<DeadlineResult> {
  const qs = new URLSearchParams({ days: String(days), include_expired: String(includeExpired) })
  const resp = await fetch(`${API_BASE}/api/tender/deadlines?${qs}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as DeadlineResult
}

// ---------------------------------------------------------------------------
// ⑧ 招标文件变更追踪
// ---------------------------------------------------------------------------

/** 变更严重度：critical 实质性（须重做标书）/ major 需评估 / minor 信息性 */
export type ChangeSeverity = 'critical' | 'major' | 'minor'

/** 变更类型 */
export type ChangeType = 'added' | 'removed' | 'modified'

export interface FieldChange {
  key: string
  label: string
  old: string | number | null
  new: string | number | null
  change_type: ChangeType
  severity: ChangeSeverity
}

/** 列表章节的一条条目（字段随章节而变） */
export type SectionItem = Record<string, string | number | boolean | null>

export interface SectionChange {
  added: SectionItem[]
  removed: SectionItem[]
  unchanged_count: number
}

export interface ChangeDiff {
  field_changes: FieldChange[]
  section_changes: Record<string, SectionChange>
  summary: {
    total: number
    critical: number
    major: number
    minor: number
  }
  /** 人类可读的影响提示（含金额涨跌方向、时间提前/延后） */
  impact_notes: string[]
}

export interface ChangeTrackResult {
  diff: ChangeDiff
  old: Record<string, unknown>
  new: Record<string, unknown>
  meta: {
    requires_rebid?: boolean
    new_treated_as_patch?: boolean
    warnings?: string[]
    [k: string]: unknown
  }
}

/**
 * 比对两份文件，输出结构化变更清单（非流式）。
 *
 * `newIsPatch` 留空时由后端自动判断：补遗式澄清公告"没提到"**不等于**"被删除"。
 */
export async function compareTenderDocuments(
  oldFile: File,
  newFile: File,
  newIsPatch?: boolean,
): Promise<ChangeTrackResult> {
  const form = new FormData()
  form.append('old_file', oldFile)
  form.append('new_file', newFile)
  if (newIsPatch !== undefined) form.append('new_is_patch', String(newIsPatch))
  const resp = await fetch(`${API_BASE}/api/tender/compare`, { method: 'POST', body: form })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as ChangeTrackResult
}

export interface CompareStreamHandlers {
  /** 阶段事件：parse_old / extract_old / parse_new / extract_new / diff */
  onPhase?: (phase: string, payload: Record<string, unknown>) => void
  onDone?: (result: ChangeTrackResult) => void
  onError?: (message: string) => void
}

/** SSE 流式变更比对（要抽两份文件，耗时较长，需要进度反馈） */
export async function compareTenderStream(
  oldFile: File,
  newFile: File,
  handlers: CompareStreamHandlers,
  newIsPatch?: boolean,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append('old_file', oldFile)
  form.append('new_file', newFile)
  if (newIsPatch !== undefined) form.append('new_is_patch', String(newIsPatch))
  const resp = await fetch(`${API_BASE}/api/tender/compare/stream`, {
    method: 'POST',
    body: form,
    signal,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  if (!resp.body) throw new ApiError(0, '响应没有可读流')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (payload: string) => {
    const t = payload.trim()
    if (!t || t === '[DONE]') return
    let ev: Record<string, unknown>
    try {
      ev = JSON.parse(t)
    } catch {
      return
    }
    const type = String(ev.type ?? '')
    if (type === 'done') handlers.onDone?.(ev as unknown as ChangeTrackResult)
    else if (type === 'error') handlers.onError?.(String(ev.content ?? '比对失败'))
    else if (type) handlers.onPhase?.(type, ev)
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx = buffer.indexOf('\n\n')
    while (idx >= 0) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const line of block.split('\n')) {
        const s = line.trim()
        if (s.startsWith('data:')) dispatch(s.slice(5))
      }
      idx = buffer.indexOf('\n\n')
    }
  }
  for (const line of buffer.split('\n')) {
    const s = line.trim()
    if (s.startsWith('data:')) dispatch(s.slice(5))
  }
}

// ---------------------------------------------------------------------------
// ⑤ 资格审查辅助
// ---------------------------------------------------------------------------

/** 单条要求的判定结果：满足 / 不满足 / **需人工确认** */
export type QualStatus = 'met' | 'unmet' | 'uncertain'

/** 总体结论（保守判定：有 uncertain 时不会报 qualified） */
export type QualOverall = 'qualified' | 'unqualified' | 'need_review'

export interface QualificationRequirement {
  index: number
  raw_text: string
  category: string
  is_mandatory: boolean
  kind: string
  kind_label: string
  constraints: Record<string, unknown>
  status: QualStatus
  reason: string
  evidence?: string
}

export interface QualificationResult {
  overall: QualOverall
  overall_label: string
  requirements: QualificationRequirement[]
  /** 只列**不满足的硬性要求**（加分项不满足不进 gaps） */
  gaps: QualificationRequirement[]
  summary: {
    total: number
    met: number
    unmet: number
    uncertain: number
    mandatory_total: number
    mandatory_unmet: number
    mandatory_uncertain: number
    optional_unmet: number
    by_kind: Record<string, { total: number; met: number; unmet: number; uncertain: number }>
    reference_date?: string
    conclusion: string
    caveats: string[]
  }
}

/** 企业资质档案 */
export interface CompanyProfile {
  company_name?: string
  qualifications?: { name: string; level?: string; cert_no?: string; valid_until?: string }[]
  certifications?: string[]
  performance?: { project: string; amount?: number; year?: number }[]
  financial?: Record<string, unknown>
  personnel?: { name?: string; cert?: string; level?: string }[]
  credit?: Record<string, unknown>
  [k: string]: unknown
}

export interface ProfileListItem {
  name: string
  company_name?: string | null
  created_at?: string | null
  updated_at?: string | null
}

/** 已保存的企业资质档案列表 */
export async function fetchQualificationProfiles(): Promise<{ items: ProfileListItem[]; count: number }> {
  const resp = await fetch(`${API_BASE}/api/qualification/profiles`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as { items: ProfileListItem[]; count: number }
}

/** 读取单个档案 */
export async function fetchQualificationProfile(name: string): Promise<CompanyProfile> {
  const resp = await fetch(`${API_BASE}/api/qualification/profiles/${encodeURIComponent(name)}`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as CompanyProfile
}

/** 新建或覆盖保存档案（同名覆盖） */
export async function saveQualificationProfile(
  name: string,
  profile: CompanyProfile,
): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_BASE}/api/qualification/profiles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, profile }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as Record<string, unknown>
}

/** 删除档案 */
export async function deleteQualificationProfile(name: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/qualification/profiles/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
}

/**
 * 对已抽取的招标文件做资格审查（不重传文件）。
 *
 * 传 `profileName` 用已保存的档案；或直接传 `profile` 对象。
 */
export async function checkQualification(
  parsed: TenderParsed,
  opts: { profile?: CompanyProfile; profileName?: string; referenceDate?: string } = {},
): Promise<QualificationResult> {
  const resp = await fetch(`${API_BASE}/api/qualification/check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parsed,
      profile: opts.profile ?? {},
      profile_name: opts.profileName ?? '',
      reference_date: opts.referenceDate ?? null,
    }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as QualificationResult
}

/** 上传招标文件 + 档案名，一步完成解析与资格审查 */
export async function checkQualificationFromFile(
  file: File,
  profileName = '',
  referenceDate?: string,
): Promise<QualificationResult> {
  const form = new FormData()
  form.append('file', file)
  const qs = new URLSearchParams()
  if (profileName) qs.set('profile_name', profileName)
  if (referenceDate) qs.set('reference_date', referenceDate)
  const resp = await fetch(
    `${API_BASE}/api/qualification/check/upload${qs.toString() ? `?${qs}` : ''}`,
    { method: 'POST', body: form },
  )
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as QualificationResult
}

// ---------------------------------------------------------------------------
// ⑪ 报告导出（Word）
// ---------------------------------------------------------------------------

/**
 * 下载投标决策报告（Word）。
 *
 * 用 blob 下载而非直接 `window.open`：后者无法带 POST 体
 * （抽取结果可能很大，放 URL 不现实），也拿不到失败状态。
 */
export async function downloadTenderReport(payload: {
  parsed: TenderParsed
  meta: Record<string, unknown>
  qualification?: QualificationResult | null
  priceProfile?: Record<string, unknown> | null
  competitor?: Record<string, unknown> | null
}): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/report/tender`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parsed: payload.parsed,
      meta: payload.meta,
      qualification: payload.qualification ?? null,
      price_profile: payload.priceProfile ?? null,
      competitor: payload.competitor ?? null,
    }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))

  // 文件名优先取 RFC 5987 的 filename*（中文名），退回 filename
  const cd = resp.headers.get('Content-Disposition') ?? ''
  let filename = '投标决策报告.docx'
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd)
  if (star) {
    try {
      filename = decodeURIComponent(star[1])
    } catch {
      /* 保持默认名 */
    }
  } else {
    const plain = /filename="([^"]+)"/i.exec(cd)
    if (plain) filename = plain[1]
  }

  const blob = await resp.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // 立即 revoke 会让部分浏览器来不及下载，延迟释放
  setTimeout(() => URL.revokeObjectURL(url), 4000)
}

// ---------------------------------------------------------------------------
// ⑨ 合规性检查
// ---------------------------------------------------------------------------

/** 严重度：high 常被认定违规 / medium 需结合背景判断 / low 表述不规范 */
export type ComplianceSeverity = 'high' | 'medium' | 'low'

export interface ComplianceRuleDef {
  key: string
  title: string
  severity: ComplianceSeverity
  /** 为什么这种表述可疑 */
  why: string
  /** 怎么核实 / 怎么处理 */
  how_to_verify: string
  topics: string[]
}

export interface ComplianceRulesCatalog {
  rules: ComplianceRuleDef[]
  count: number
  severity_levels: Record<ComplianceSeverity, string>
}

/** 法规依据（语料未入库时为空数组） */
export interface LawBasis {
  law: string
  article: string
  excerpt: string
  score?: number
}

export interface ComplianceFinding {
  rule: string
  title: string
  severity: ComplianceSeverity
  /** 命中的章节 key */
  section: string
  item_index: number
  /** 条款原文 */
  clause: string
  /** 原文中命中的片段（证据必须来自原文） */
  evidence: string
  why: string
  how_to_verify: string
  topics: string[]
  law_basis: LawBasis[]
}

export interface ComplianceResult {
  findings: ComplianceFinding[]
  summary: {
    checked_clauses: number
    flagged_clauses: number
    counts: Record<ComplianceSeverity, number>
    total: number
    by_rule: Record<string, number>
    rules_total: number
    conclusion: string
  }
  /** **必读**：本结果不构成法律意见，且"未命中不代表合规" */
  disclaimer: string
  extract_meta?: TenderExtractMeta
}

/** 规则清单（前端不硬编码） */
export async function fetchComplianceRules(): Promise<ComplianceRulesCatalog> {
  const resp = await fetch(`${API_BASE}/api/compliance/rules`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as ComplianceRulesCatalog
}

/** 对已抽取的招标文件做合规检查 */
export async function checkCompliance(
  parsed: TenderParsed,
  includeLawBasis = true,
): Promise<ComplianceResult> {
  const resp = await fetch(`${API_BASE}/api/compliance/check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ parsed, include_law_basis: includeLawBasis }),
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as ComplianceResult
}

/** 上传招标文件，一步完成解析、抽取与合规检查 */
export async function checkComplianceFromFile(
  file: File,
  includeLawBasis = true,
): Promise<ComplianceResult> {
  const form = new FormData()
  form.append('file', file)
  const qs = new URLSearchParams({ include_law_basis: String(includeLawBasis) })
  const resp = await fetch(`${API_BASE}/api/compliance/check/upload?${qs}`, {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as ComplianceResult
}

/** 法规语料库规模（用于说明检查的法规依据覆盖程度） */
export async function fetchRegulationStats(): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_BASE}/api/compliance/regulations/stats`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return (await resp.json()) as Record<string, unknown>
}
