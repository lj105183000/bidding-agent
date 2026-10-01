/**
 * 唯一网络层：所有对后端的请求都从这里走。
 *
 * - `API_BASE`：默认直连 :8001（Vite 无 dev 代理，SSE 无缓冲风险，与 CLAUDE.md 一致）。
 * - `askStream`：POST /api/chat/stream + ReadableStream 手写 SSE 解析
 *   （2KB padding、`data:` 前缀、`[DONE]` 尾帧、`reset` 清屏、坏帧 JSON 容错）。
 * - 其余封装：健康检查、会话列表/详情/删除、点赞/点踩反馈。
 */

const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8001'

/** done 帧里的知识库来源 */
export interface SourceItem {
  question?: string
  answer?: string
  score?: number
  /** 联网来源才有 */
  url?: string
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
  /** [名, ms] 数组 */
  phase_times: [string, number][]
}

export interface SseEvent {
  type: string
  content?: string
  [key: string]: unknown
}

export interface AskOptions {
  question: string
  history: { role: string; content: string }[]
  web_search_enabled: boolean
  provider: string
  deep_thinking_enabled: boolean
}

export interface AskHandlers {
  onStatus?: (text: string) => void
  onToken?: (text: string) => void
  onThinking?: (text: string) => void
  onReset?: () => void
  onDone?: (done: DoneEvent) => void
  onError?: (text: string) => void
}

export interface ConversationSummary {
  id: number
  session_id: string
  title: string
  created_at: string
}

export interface PersistedMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[] | null
  toolName?: string | null
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
      web_search_enabled: opts.web_search_enabled,
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
  // 流结束：把残留 buffer 也解析一遍（防最后一块没有尾随空行）
  if (buffer.trim()) {
    for (const ev of parseSse(buffer)) dispatchEvent(ev, handlers)
  }
}

// ---------------------------------------------------------------------------
// 健康检查 / 会话 / 反馈
// ---------------------------------------------------------------------------

export async function fetchHealth(): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_BASE}/api/health`)
  if (!resp.ok) throw new ApiError(resp.status, await parseError(resp))
  return resp.json()
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const resp = await fetch(`${API_BASE}/api/conversations`)
  await assertOk(resp)
  const data = await resp.json()
  return (data?.conversations ?? []) as ConversationSummary[]
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
  messages: { role: string; content: string; sources?: unknown; toolName?: unknown }[],
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
