import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Copy,
  Globe,
  MessageSquare,
  Send,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import Badge from '../components/Badge'
import { useToast } from '../components/Toast'
import { useModel } from '../contexts/model'
import {
  askStream,
  deleteConversation,
  getConversation,
  listConversations,
  saveConversation,
  sendFeedback,
  ApiError,
  type ConversationSummary,
  type DoneEvent,
  type SourceItem,
} from '../lib/api'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  status: 'pending' | 'streaming' | 'done' | 'error'
  sources?: SourceItem[]
  webSources?: SourceItem[]
  toolCalled?: boolean
  toolName?: string
  thinking?: string
  elapsedMs?: number
  /** 落库后返回的 assistant 消息 id，用于点赞/点踩 */
  messageId?: number
  errorText?: string
}

const SESSION_KEY = 'chat_session_id'
const WELCOME =
  '您好！我是招采智脑，可以为您解答招投标全流程问题。试试问我：单一来源采购公示被质疑怎么办？'

const PROVIDER_LABEL: Record<string, string> = {
  deepseek: 'DeepSeek',
  zhipu: '智谱',
  vllm: 'vLLM',
  ollama: 'Ollama',
}

function getSessionId(): string {
  try {
    const existing = localStorage.getItem(SESSION_KEY)
    if (existing) return existing
  } catch {
    /* ignore */
  }
  const fresh = crypto.randomUUID()
  try {
    localStorage.setItem(SESSION_KEY, fresh)
  } catch {
    /* ignore */
  }
  return fresh
}

let msgSeq = 0
function nextId(): string {
  msgSeq += 1
  return `${Date.now()}-${msgSeq}`
}

/** AI 占位：三点跳动动画 */
function Dots() {
  return (
    <span className="flex items-center gap-1 py-1.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-400"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </span>
  )
}

function markdownComponents() {
  return {
    a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
      <a target="_blank" rel="noopener noreferrer" className="text-brand-700 underline" {...props} />
    ),
    table: (props: React.TableHTMLAttributes<HTMLTableElement>) => (
      <table className="my-2 w-full border-collapse text-[13px]" {...props} />
    ),
    th: (props: React.ThHTMLAttributes<HTMLTableCellElement>) => (
      <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-medium" {...props} />
    ),
    td: (props: React.TdHTMLAttributes<HTMLTableCellElement>) => (
      <td className="border border-slate-200 px-2 py-1" {...props} />
    ),
    code: (props: React.HTMLAttributes<HTMLElement>) => (
      <code className="rounded bg-slate-100 px-1 py-0.5 text-[12px]" {...props} />
    ),
    pre: (props: React.HTMLAttributes<HTMLPreElement>) => (
      <pre className="my-2 overflow-x-auto rounded-md bg-slate-900 p-3 text-[12px] text-slate-100" {...props} />
    ),
  }
}

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="mt-2 overflow-hidden rounded-md border border-purple-100 bg-purple-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1 px-3 py-1.5 text-left text-[11px] font-medium text-purple-700"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        深度思考
      </button>
      {open && (
        <div className="border-t border-purple-100 px-3 py-2 text-xs leading-5 text-purple-900">
          {text}
        </div>
      )}
    </div>
  )
}

interface AiBubbleProps {
  m: ChatMessage
  sessionId: string
  onScroll: () => void
}

function AiBubble({ m, sessionId, onScroll }: AiBubbleProps) {
  const toast = useToast()
  const [liked, setLiked] = useState<boolean | null>(null)

  useEffect(() => {
    onScroll()
  }, [m.content, m.thinking, onScroll])

  if (m.status === 'pending') {
    return (
      <div className="flex gap-3">
        <Avatar />
        <div className="rounded-xl border border-slate-200 bg-white px-4 py-2">
          <Dots />
        </div>
      </div>
    )
  }

  const copy = () => {
    navigator.clipboard
      .writeText(m.content)
      .then(() => toast.show('已复制到剪贴板', 'success'))
      .catch(() => toast.show('复制失败', 'error'))
  }

  const feedback = async (rating: 'up' | 'down') => {
    setLiked(rating === 'up')
    if (!m.messageId) {
      toast.show('该回答尚未落库，无法反馈', 'error')
      return
    }
    try {
      await sendFeedback(sessionId, m.messageId, rating)
      toast.show('感谢反馈，已记录', 'success')
    } catch (err) {
      toast.show(
        err instanceof ApiError ? `反馈失败：${err.message}` : '反馈失败',
        'error',
      )
    }
  }

  const showMeta = m.status === 'done' && (m.toolName || (m.sources?.length ?? 0) > 0 || m.elapsedMs !== undefined)

  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="max-w-[85%] min-w-0 rounded-xl border border-slate-200 bg-white px-4 py-3">
        {/* 思考折叠块（不落库，仅展示） */}
        {m.thinking && <ThinkingBlock text={m.thinking} />}

        <div className="prose prose-sm max-w-none text-[14px] leading-6 text-ink-700">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents()}>
            {m.content || ''}
          </ReactMarkdown>
          {m.status === 'streaming' && (
            <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-brand-600 align-middle" />
          )}
        </div>

        {m.status === 'error' && (
          <div className="mt-2 rounded-md border-l-2 border-danger bg-red-50 px-3 py-2 text-xs leading-5 text-danger">
            {m.errorText || '回答生成失败'}
          </div>
        )}

        {showMeta && (
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2">
            {m.toolName && (
              <Badge color="brand">
                <Brain className="h-3 w-3" />
                工具：{m.toolName}
              </Badge>
            )}
            {(m.sources?.length ?? 0) > 0 && (
              <Badge color="ok">来源 {m.sources!.length}</Badge>
            )}
            {(m.webSources?.length ?? 0) > 0 && (
              <Badge color="warn">
                <Globe className="h-3 w-3" />
                联网 {m.webSources!.length}
              </Badge>
            )}
            {m.elapsedMs !== undefined && (
              <span className="text-[11px] text-ink-400">耗时 {(m.elapsedMs / 1000).toFixed(1)}s</span>
            )}
          </div>
        )}

        {m.status === 'done' && (
          <div className="mt-3 flex items-center gap-1 border-t border-slate-100 pt-2">
            <button
              onClick={copy}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-ink-400 transition-colors hover:bg-ink-50 hover:text-ink-700"
            >
              <Copy className="h-3 w-3" />
              复制
            </button>
            <button
              onClick={() => feedback('up')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors hover:bg-ink-50 ${
                liked === true ? 'text-ok' : 'text-ink-400 hover:text-ink-700'
              }`}
            >
              <ThumbsUp className="h-3 w-3" />
              有用
            </button>
            <button
              onClick={() => feedback('down')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors hover:bg-ink-50 ${
                liked === false ? 'text-danger' : 'text-ink-400 hover:text-ink-700'
              }`}
            >
              <ThumbsDown className="h-3 w-3" />
              没帮助
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function Avatar() {
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
      <MessageSquare className="h-3.5 w-3.5 text-white" />
    </div>
  )
}

export default function Chat() {
  const { model } = useModel()
  const toast = useToast()
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 'welcome', role: 'assistant', content: WELCOME, status: 'done' },
  ])
  const [streaming, setStreaming] = useState(false)
  const [webSearch, setWebSearch] = useState(false)
  const [deepThinking, setDeepThinking] = useState(false)
  const [sessions, setSessions] = useState<ConversationSummary[]>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [])

  const sessionIdRef = useRef<string>(getSessionId())

  const refreshSessions = useCallback(async () => {
    try {
      const list = await listConversations()
      setSessions(list)
    } catch {
      /* 列表拉取失败静默：侧栏为空，不影响主对话 */
    }
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  const updateMessage = useCallback(
    (
      id: string,
      patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>),
    ) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? { ...m, ...(typeof patch === 'function' ? patch(m) : patch) }
            : m,
        ),
      )
    },
    [],
  )

  const switchSession = useCallback(
    async (sessionId: string) => {
      if (streaming) return
      try {
        const rows = await getConversation(sessionId)
        const restored: ChatMessage[] = rows.map((r) => ({
          id: `r-${r.id}`,
          role: r.role,
          content: r.content,
          status: 'done',
          sources: r.sources ?? undefined,
          toolName: r.toolName ?? undefined,
          messageId: r.id,
        }))
        setMessages(
          restored.length
            ? restored
            : [{ id: 'welcome', role: 'assistant', content: WELCOME, status: 'done' }],
        )
        sessionIdRef.current = sessionId
        try {
          localStorage.setItem(SESSION_KEY, sessionId)
        } catch {
          /* ignore */
        }
        scrollToBottom()
      } catch (err) {
        toast.show(
          err instanceof ApiError ? `加载会话失败：${err.message}` : '加载会话失败',
          'error',
        )
      }
    },
    [streaming, toast, scrollToBottom],
  )

  const removeSession = useCallback(
    async (sessionId: string) => {
      try {
        await deleteConversation(sessionId)
        if (sessionId === sessionIdRef.current) {
          sessionIdRef.current = getSessionId()
          setMessages([{ id: 'welcome', role: 'assistant', content: WELCOME, status: 'done' }])
        }
        await refreshSessions()
      } catch (err) {
        toast.show(
          err instanceof ApiError ? `删除失败：${err.message}` : '删除失败',
          'error',
        )
      }
    },
    [toast, refreshSessions],
  )

  const persist = useCallback(
    async (text: string, answer: ChatMessage) => {
      try {
        const ids = await saveConversation(sessionIdRef.current, text, [
          { role: 'user', content: text },
          {
            role: 'assistant',
            content: answer.content,
            sources: answer.sources,
            toolName: answer.toolName,
          },
        ])
        // assistant 消息 id 是 ids 的最后一条（先 user 后 assistant）
        const assistantId = ids[ids.length - 1]
        if (assistantId !== undefined) {
          updateMessage(answer.id, { messageId: assistantId })
        }
        refreshSessions()
      } catch (err) {
        // 落库失败不阻断对话：仅提示
        toast.show(
          err instanceof ApiError ? `会话保存失败：${err.message}` : '会话保存失败',
          'error',
        )
      }
    },
    [refreshSessions, toast, updateMessage],
  )

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim()
      if (!text || streaming) return

      // 组装已完成回合的历史（不含当前问题）
      const history = messages
        .filter((m) => m.status === 'done' && (m.role === 'user' || m.role === 'assistant'))
        .map((m) => ({ role: m.role, content: m.content }))

      const uid = nextId()
      const aid = nextId()
      const userMsg: ChatMessage = { id: uid, role: 'user', content: text, status: 'done' }
      const aiMsg: ChatMessage = { id: aid, role: 'assistant', content: '', status: 'pending' }

      setMessages((prev) => [...prev, userMsg, aiMsg])
      setInput('')
      setStreaming(true)
      scrollToBottom()

      const controller = new AbortController()
      abortRef.current = controller

      const done = (payload: DoneEvent) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aid
              ? {
                  ...m,
                  content: m.content || '（未生成回答）',
                  sources: payload.sources ?? [],
                  webSources: payload.web_sources ?? [],
                  toolCalled: payload.tool_called,
                  toolName: payload.tool_name,
                  elapsedMs: payload.elapsed_ms,
                  status: 'done',
                }
              : m,
          ),
        )
      }

      try {
        await askStream(
          {
            question: text,
            history,
            web_search_enabled: webSearch,
            provider: model,
            deep_thinking_enabled: deepThinking,
          },
          {
            onStatus: () => updateMessage(aid, { status: 'streaming' }),
            onToken: (chunk) =>
              updateMessage(aid, (m) => ({ status: 'streaming', content: m.content + chunk })),
            onThinking: (chunk) =>
              updateMessage(aid, (m) => ({ thinking: (m.thinking ?? '') + chunk })),
            onReset: () => updateMessage(aid, { content: '' }),
            onDone: done,
            onError: (text2) =>
              updateMessage(aid, { status: 'error', errorText: text2 }),
          },
          controller.signal,
        )
        // 流结束后落库（onDone 已更新消息，用最新内容）
        setMessages((prev) => {
          const last = prev.find((m) => m.id === aid)
          if (last && last.status === 'done') persist(text, last)
          return prev
        })
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          updateMessage(aid, {
            status: 'error',
            errorText: '已停止生成（保留已生成内容）',
          })
        } else {
          const friendly =
            err instanceof ApiError
              ? err.status === 429
                ? '请求过于频繁，请稍后再试'
                : err.status === 503
                  ? '服务暂未就绪，请稍后再试'
                  : `请求失败：${err.message}`
              : '网络异常，请检查后端服务是否已启动'
          updateMessage(aid, { status: 'error', errorText: friendly })
        }
      } finally {
        setStreaming(false)
        abortRef.current = null
        scrollToBottom()
      }
    },
    [messages, model, webSearch, deepThinking, streaming, updateMessage, persist, scrollToBottom, toast],
  )

  const stop = () => {
    abortRef.current?.abort()
  }

  const handleSend = () => {
    void send(input)
  }

  return (
    <div>
      <PageHeader title="智能问答" description="基于企业知识库的招投标采购智能问答助手" />

      <div className="flex h-[calc(100vh-140px)] min-h-[520px] gap-4">
        {/* 左对话区 */}
        <div className="flex flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white">
          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-5">
            {messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[75%] rounded-xl bg-brand-600 px-4 py-2.5 text-[14px] leading-6 text-white">
                    {m.content}
                  </div>
                </div>
              ) : (
                <AiBubble
                  key={m.id}
                  m={m}
                  sessionId={sessionIdRef.current}
                  onScroll={scrollToBottom}
                />
              ),
            )}
          </div>

          {/* 输入区 */}
          <div className="border-t border-slate-200 bg-white p-4">
            <div className="mb-2 flex flex-wrap items-center gap-3">
              {/* 联网搜索开关 */}
              <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-ink-600">
                <input
                  type="checkbox"
                  checked={webSearch}
                  onChange={(e) => setWebSearch(e.target.checked)}
                  className="h-3.5 w-3.5 accent-brand-600"
                />
                <Globe className="h-3.5 w-3.5 text-ink-400" />
                联网搜索
              </label>
              {/* 深度思考开关 */}
              <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-ink-600">
                <input
                  type="checkbox"
                  checked={deepThinking}
                  onChange={(e) => setDeepThinking(e.target.checked)}
                  className="h-3.5 w-3.5 accent-purple-600"
                />
                <Brain className="h-3.5 w-3.5 text-purple-500" />
                深度思考
              </label>
              <span className="ml-auto text-[11px] text-ink-400">
                当前模型：{PROVIDER_LABEL[model] ?? model}
              </span>
            </div>

            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
                rows={2}
                placeholder="请输入您的问题，Enter 发送，Shift+Enter 换行"
                className="max-h-32 flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-[13px] leading-5 text-ink-900 outline-none transition-colors placeholder:text-ink-300 focus:border-brand-500"
              />
              {streaming ? (
                <Button variant="secondary" onClick={stop} icon={<Square className="h-3.5 w-3.5" />}>
                  停止
                </Button>
              ) : (
                <Button onClick={handleSend} icon={<Send className="h-3.5 w-3.5" />}>
                  发送
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* 右侧栏：会话 + 推荐问题 */}
        <aside className="flex w-[300px] shrink-0 flex-col gap-4 overflow-y-auto">
          <Card title="历史会话" subtitle="点击切换，垃圾桶删除">
            <div className="space-y-1">
              {sessions.length === 0 && (
                <div className="py-2 text-center text-xs text-ink-400">暂无会话</div>
              )}
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`group flex items-center gap-1 rounded-lg border px-2 py-1.5 transition-colors ${
                    s.session_id === sessionIdRef.current
                      ? 'border-brand-200 bg-brand-50'
                      : 'border-slate-200 hover:border-brand-200 hover:bg-brand-50'
                  }`}
                >
                  <button
                    onClick={() => void switchSession(s.session_id)}
                    className="min-w-0 flex-1 truncate text-left text-xs text-ink-700"
                    title={s.title}
                  >
                    {s.title || '（未命名会话）'}
                  </button>
                  <button
                    onClick={() => void removeSession(s.session_id)}
                    className="hidden shrink-0 rounded p-0.5 text-ink-400 hover:text-danger group-hover:block"
                    title="删除会话"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </Card>

          <Card title="推荐问题" subtitle="点击自动提问">
            <div className="space-y-2">
              {[
                '单一来源采购公示被质疑怎么办？',
                '政府采购中投标保证金如何计算？',
                '医疗设备采购有哪些注意事项？',
                '招标文件中常见的废标条款有哪些？',
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => void send(q)}
                  disabled={streaming}
                  className="block w-full rounded-lg border border-slate-200 px-3 py-2 text-left text-xs leading-5 text-ink-700 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700 disabled:opacity-50"
                >
                  {q}
                </button>
              ))}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  )
}
