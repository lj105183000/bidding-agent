import { useEffect, useRef, useState } from 'react'
import { Copy, Send, Sparkles, ThumbsDown, ThumbsUp } from 'lucide-react'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'
import { useTypewriter } from '../hooks/useTypewriter'
import { QA_DATA } from '../data/qa'
import { matchQA } from '../lib/matcher'

interface ChatMessage {
  id: number
  role: 'user' | 'ai'
  content: string
  status: 'pending' | 'typing' | 'done'
  basis: string
}

const WELCOME =
  '您好！我是招采智脑，可以为您解答招投标全流程问题。试试问我：单一来源采购公示被质疑怎么办？'

const FALLBACK =
  '抱歉，知识库中暂时没有找到与您问题匹配的内容。建议换个关键词试试（如“单一来源”“保证金”“中标”等），或联系人工客服获取帮助。'

const KB_COVERAGE = [
  { label: '招投标法律法规', value: 86 },
  { label: '政府采购流程', value: 78 },
  { label: '医疗设备采购', value: 64 },
  { label: '工业品物料', value: 72 },
]

/** AI 占位: 三点跳动动画 */
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

function AiBubble({ m, onUpdate }: { m: ChatMessage; onUpdate: () => void }) {
  const toast = useToast()
  const [liked, setLiked] = useState<boolean | null>(null)
  const { typed, done } = useTypewriter(m.status === 'typing' ? m.content : '', 30)
  // typing 状态显示逐字文本, done 状态直接显示全文
  const displayText = m.status === 'done' ? m.content : typed

  useEffect(() => {
    onUpdate()
  }, [typed, done, onUpdate])

  if (m.status === 'pending') {
    return (
      <div className="flex gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
          <Sparkles className="h-3.5 w-3.5 text-white" />
        </div>
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

  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
        <Sparkles className="h-3.5 w-3.5 text-white" />
      </div>
      <div className="max-w-[75%] rounded-xl border border-slate-200 bg-white px-4 py-3">
        <div className="whitespace-pre-wrap text-[14px] leading-6 text-ink-700">
          {displayText}
          {!done && (
            <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-brand-600 align-middle" />
          )}
        </div>

        {done && m.basis && (
          <div className="mt-3 rounded-md border-l-2 border-brand-500 bg-brand-50 px-3 py-2">
            <div className="text-[11px] font-medium text-brand-700">法规依据</div>
            <div className="mt-0.5 text-xs leading-5 text-ink-700">{m.basis}</div>
          </div>
        )}

        {done && (
          <div className="mt-3 flex items-center gap-1 border-t border-slate-100 pt-2">
            <button
              onClick={copy}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-ink-400 transition-colors hover:bg-ink-50 hover:text-ink-700"
            >
              <Copy className="h-3 w-3" />
              复制
            </button>
            <button
              onClick={() => {
                setLiked(true)
                toast.show('感谢反馈，已记录', 'info')
              }}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors hover:bg-ink-50 ${
                liked === true ? 'text-ok' : 'text-ink-400 hover:text-ink-700'
              }`}
            >
              <ThumbsUp className="h-3 w-3" />
              有用
            </button>
            <button
              onClick={() => {
                setLiked(false)
                toast.show('已记录，将持续优化答案', 'info')
              }}
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

export default function Chat() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 0, role: 'ai', content: WELCOME, status: 'done', basis: '' },
  ])
  const idRef = useRef(1)
  const scrollRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }

  const send = (raw: string) => {
    const text = raw.trim()
    if (!text) return
    const uid = idRef.current++
    const aid = idRef.current++
    setMessages((prev) => [
      ...prev,
      { id: uid, role: 'user', content: text, status: 'done', basis: '' },
      { id: aid, role: 'ai', content: '', status: 'pending', basis: '' },
    ])
    setInput('')

    // 800ms 后调用 matchQA
    setTimeout(() => {
      const hits = matchQA(text)
      if (hits.length > 0) {
        const top = hits[0].item
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aid
              ? { ...m, content: top.answer, basis: top.basis, status: 'typing' }
              : m,
          ),
        )
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aid ? { ...m, content: FALLBACK, status: 'done' } : m,
          ),
        )
      }
    }, 800)
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
                <AiBubble key={m.id} m={m} onUpdate={scrollToBottom} />
              ),
            )}
          </div>

          {/* 输入区 */}
          <div className="border-t border-slate-200 bg-white p-4">
            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send(input)
                  }
                }}
                rows={2}
                placeholder="请输入您的问题，Enter 发送，Shift+Enter 换行"
                className="max-h-32 flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-[13px] leading-5 text-ink-900 outline-none transition-colors placeholder:text-ink-300 focus:border-brand-500"
              />
              <Button onClick={() => send(input)} icon={<Send className="h-3.5 w-3.5" />}>
                发送
              </Button>
            </div>
          </div>
        </div>

        {/* 右侧推荐问题栏 */}
        <aside className="w-[300px] shrink-0 space-y-4 overflow-y-auto">
          <Card title="推荐问题" subtitle="点击自动提问">
            <div className="space-y-2">
              {QA_DATA.slice(0, 6).map((q) => (
                <button
                  key={q.id}
                  onClick={() => send(q.question)}
                  className="block w-full rounded-lg border border-slate-200 px-3 py-2 text-left text-xs leading-5 text-ink-700 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700"
                >
                  {q.question}
                </button>
              ))}
            </div>
          </Card>

          <Card title="知识库覆盖">
            {KB_COVERAGE.map((k) => (
              <div key={k.label} className="mb-3 last:mb-0">
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="text-ink-700">{k.label}</span>
                  <span className="tabular-nums text-ink-400">{k.value}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-ink-100">
                  <div
                    className="h-full rounded-full bg-brand-500"
                    style={{ width: `${k.value}%` }}
                  />
                </div>
              </div>
            ))}
          </Card>
        </aside>
      </div>
    </div>
  )
}
