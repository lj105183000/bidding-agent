import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { StackedBar, ColorDot } from '../components/MicroBars'
import { phaseColor } from '../lib/chartStyles'
import {
  AlertTriangle,
  Brain,
  ChevronDown,
  ChevronRight,
  Copy,
  Globe,
  ImageIcon,
  Loader2,
  MessageCircleQuestion,
  MessageSquare,
  Paperclip,
  Pencil,
  Search,
  Send,
  Square,
  ThumbsDown,
  ThumbsUp,
  Timer,
  Trash2,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import Badge from '../components/Badge'
import AnswerCharts from '../components/AnswerCharts'
import RagTimingPanel from '../components/RagTimingPanel'
import { useToast } from '../components/Toast'
import { useModel } from '../contexts/model'
import {
  analyzeImage,
  askStream,
  deleteConversation,
  getConversation,
  listConversationsDetailed,
  searchConversations,
  renameConversation,
  saveConversation,
  sendFeedback,
  ApiError,
  type ConversationDetail,
  type ConversationSearchHit,
  type AnswerChart,
  type RagTimingSummary,
  type ClarifyEvent,
  type DoneEvent,
  type LLMTrace,
  type SourceItem,
  type ImageAnalysis,
  type ToolTiming,
  type WebFallbackInfo,
  type WebSearchMode,
  type WebSourceItem,
} from '../lib/api'
import { fmtSec } from '../lib/format'
import { ImagePrepareError, formatBytes, prepareImage, type PreparedImage } from '../lib/image'
import {
  MAX_ATTACHMENTS,
  describeAdditions,
  mergeImageAnalyses,
  planAdditions,
  splitByKind,
} from '../lib/imageQueue'
import { MAX_PDF_PAGES, PdfConvertError, pdfToImages } from '../lib/pdf'
import SourceList from '../components/SourceList'
import useStableCallback from '../hooks/useStableCallback'
import { createStreamBuffer } from '../lib/streamBuffer'
import { isNearBottom } from '../lib/scroll'
interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** `clarify` = 本轮以「向用户追问」结束，等待用户补充信息。
   *
   *  ★ 为什么单列一个状态而不是复用 `done`
   *  -------------------------------
   *  追问后没有正文。若沿用 `done`，渲染层会走进"答完了但内容为空"的分支，
   *  显示一个空回复框（实测常见）+ 来源卡（空）。单列状态让"等待用户回答"
   *  这件事在类型层面就与"答完"区分开。
   */
  status: 'pending' | 'streaming' | 'done' | 'error' | 'clarify'
  sources?: SourceItem[]
  webSources?: WebSourceItem[]
  /** 首轮追问的内容（问题与候选项），仅 `status === 'clarify'` 时存在 */
  clarification?: ClarifyEvent
  /** 追问卡片是否已提交过答案（提交后禁用输入，防重复发问） */
  clarifiedAnswered?: boolean
  /** 自动兜底的判定结果（说明为什么联网） */
  webFallback?: WebFallbackInfo | null
  toolCalled?: boolean
  toolName?: string
  thinking?: string
  /** 后端进度文案（「正在深度思考...」/「正在补充检索（第1轮）...」等）。
   *
   *  ★ 此前被丢弃：`onStatus: () => updateMessage(...)` 没接参数，
   *  后端 11 种状态文案一个都没到界面上。而这是**用户唯一能看到的
   *  「系统在做什么」信号**——尤其深度思考时思维链可能十几秒不出内容，
   *  没有它界面看起来像卡住了。
   */
  statusText?: string
  elapsedMs?: number
  /** 阶段耗时 `[名称, 毫秒, 起始偏移毫秒?]`（后端在真实边界打点，前端只展示） */
  phaseTimes?: [string, number, number?][]
  /** 分阶段合计（毫秒）；与 elapsedMs 的差额 = 未被任何阶段覆盖的时间 */
  phaseTotalMs?: number
  /** LLM 调用明细与 token 用量 */
  llmTrace?: LLMTrace | null
  /** 每个工具的执行耗时 */
  toolTimings?: ToolTiming[]
  /** 图表数据（后端从工具原始输出提取，前端只负责画） */
  charts?: AnswerChart[]
  /** RAG 检索耗时（含步骤分解） */
  ragTiming?: RagTimingSummary | null
  /** 落库后返回的 assistant 消息 id，用于点赞/点踩 */
  messageId?: number
  errorText?: string
  /** 用户随消息上传的图片（data URL），用于气泡内缩略图 */
  imageUrl?: string
  /** 多图上传时的预览地址列表（>1 张才有；单张仍走 `imageUrl`） */
  imageUrls?: string[]
  /** 该图片由哪个视觉模型解析（气泡上标注来源） */
  imageModel?: string
  /** 图片解析文本（"只存解析文本"：原图不入库，故刷新页面后只剩这段文字） */
  imageAnalyses?: ImageAnalysis[]
}
const SESSION_KEY = 'chat_session_id'

/**
 * 把一条消息转成送给后端的 history 行。
 *
 * ★ 为什么带图片的消息要**还原成当轮那份文本**（而不是只给 `m.content`）
 * ------------------------------------------------------------------
 * 图片解析结果当轮是拼进问题里的（`questionForAgent`）。若 history 只带
 * 原始问题文本，后续轮次模型就"忘了"图里读到过什么 —— 表现为用户接着问
 * "那这个金额合规吗"，模型回答"您没有提供金额"。
 *
 * 用与当轮**同一个** `mergeImageAnalyses` 重建，格式逐字一致；
 * 无图片的消息原样返回（绝大多数路径不受影响）。
 */
function _historyRow(m: ChatMessage): { role: string; content: string } {
  return {
    role: m.role,
    content: m.imageAnalyses?.length
      ? `${mergeImageAnalyses(m.imageAnalyses)}\n\n【用户问题】\n${m.content}`
      : m.content,
  }
}const WELCOME =
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

/**
 * Markdown 渲染器映射表：**模块级常量**（原来是一个函数，每次渲染都新建对象）。
 *
 * ★ 为什么必须提出来
 * ----------------
 * 原来是 `components={markdownComponents()}` —— 每渲染一次就造一个**新对象**。
 * 流式回答时每条消息要渲染几十上百次，于是：① 白白产生大量临时对象；
 * ② `components` 引用每次都变，React 无法复用上一次的结果树，
 * 表格/代码块这些子树的 props 对比全部落空，只能整棵重建。
 *
 * 这个映射表**不依赖任何 props / state**（只是把标签换成带样式的标签），
 * 故提升到模块级既安全又必要 —— 全应用共用同一个对象。
 */
const MARKDOWN_COMPONENTS = markdownComponents()
/** 深度思考的思维链折叠面板。
 *
 *  ★ 交互约定（按需求确定）
 *  -------------------
 *      ① 可折叠面板，**默认展开**
 *      ② 思考过程中保持展开（用户能看到推理实时长出来）
 *      ③ 思考**完成后仍默认展开**，用户可手动收起
 *      ④ 用户的手动选择**不被覆盖** —— 不因为"思考完成"而自动收起
 *
 *  为什么完成时不自动收起：需求明确要"思考过程可见性强"，
 *  自动收起会把刚产生的思维链立刻藏起来，用户看不到它。
 *  代价是界面较长（需求也接受这一点）。
 *
 *  ★ 头部为什么要显示状态与进度
 *  ------------------------
 *  思维链是**逐步**下发的：服务端可能十几秒才吐第一个字（要等模型思考）。
 *  这段时间若面板里空着，用户会以为卡住了。
 *  故头部始终显示后端进度文案（「正在深度思考...」等）+ 已收到字数。
 *
 *  ★ 这个组件此前丢失了一个重要信号
 *  -------------------------
 *  后端有 11 种进度文案，但 `onStatus` 之前写成 `() => updateMessage(...)`
 *  没接参数，全部丢弃。而它是**用户唯一能看到的"系统在做什么"**，
 *  故现在接到 `statusText` 并显示在这里。
 */
function ThinkingBlock({
  text,
  streaming,
  statusText,
}: {
  text: string
  /** 该消息是否仍在流式（决定「思考中」态与忙碌指示） */
  streaming?: boolean
  /** 后端进度文案 */
  statusText?: string
}) {
  const [open, setOpen] = useState(true)
  const thinking = Boolean(text)
  // 仍在思考：本消息在流式、且思维链已开始产出
  const busy = Boolean(streaming) && thinking
  const chars = text.length
  // 头部摘要：思考中显示实时进度，完成后显示总量
  const summary = busy
    ? `思考中… 已 ${chars} 字`
    : `已完成${chars ? ` · ${chars} 字` : ''}`
  return (
    <div className="mt-2 overflow-hidden rounded-md border border-purple-100 bg-purple-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-[11px] font-medium text-purple-700 hover:bg-purple-100/60"
        aria-expanded={open}
        aria-label={open ? '收起思考过程' : '展开思考过程'}
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <Brain className="h-3 w-3 shrink-0 text-purple-500" />
        <span className="whitespace-nowrap">深度思考</span>
        {/* 思考中：转圈 + 后端进度文案；完成后：字数摘要 */}
        {busy ? (
          <>
            <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
            <span className="truncate font-normal text-purple-600/90">
              {statusText || '思考中…'}
            </span>
          </>
        ) : (
          <span className="truncate font-normal text-purple-600/70">{summary}</span>
        )}
        <span className="ml-auto shrink-0 font-normal text-purple-500/70">
          {open ? '收起' : '展开'}
        </span>
      </button>
      {open && (
        /* 长思维链要能看全但不能无限拉长页面：
           `max-h` + 内部滚动。滚动条用细样式，避免在浅紫面板里太突兀。 */
        <div className="max-h-[320px] overflow-y-auto border-t border-purple-100 px-3 py-2 text-xs leading-5 whitespace-pre-wrap text-purple-900">
          {text}
          {busy && (
            <span className="ml-0.5 inline-block h-3 w-0.5 animate-pulse bg-purple-500 align-middle" />
          )}
        </div>
      )}
    </div>
  )
}
interface AiBubbleProps {
  m: ChatMessage
  sessionId: string
  onScroll: () => void
  /** 是否正在生成（追问卡片据此禁用提交，避免并发发问） */
  busy: boolean
  /** 提交追问的回答（由页面层提供：它需要构造 history 并发起新一轮请求） */
  onSubmitClarification: (assistantId: string, answer: string) => void
}

/** LLM 调用类型的可读名。后端的 kind 是英文枚举（decision / stream / thinking），
 *  直接用会让中文界面里混出英文；这里映射成"这一步在做什么"。
 *  未登记的 kind 原样显示（新增类型时不会显示成空白）。 */
const KIND_LABEL: Record<string, string> = {
  decision: '决策轮',
  stream: '生成回答',
  thinking: '思考流',
}


/**
 * 首轮追问卡片。
 *
 * ★ 为什么要有这个组件（而不是让用户自己在输入框里补一句）
 * ------------------------------------------------
 * 追问的价值在于"降低用户表达成本"。若只显示三行问题、让用户自己组织语言，
 * 用户仍要猜"我该说多少才够"。故：
 *   · 有候选项的问题渲染成**可点选按钮**（一键选中，无需打字）
 *   · 每题一个输入框，支持"点选 + 补充"（如选了"广东"再补"深圳"）
 *   · 全部答完才能提交，避免发出半截信息
 *
 * ★ 为什么"全部答完"才允许提交
 * ------------------------
 * 允许部分提交的话，模型很可能针对缺失的那条**再次追问**，
 * 而 `clarify_allowed` 只在首轮为真——第二轮不会追问，于是它会
 * 基于残缺信息硬答。宁可让用户在这里补齐，也不要产生"看似有依据"的回答。
 */
function ClarifyCard({
  payload,
  disabled,
  onSubmit,
}: {
  payload: ClarifyEvent
  disabled?: boolean
  onSubmit: (answer: string) => void
}) {
  const [answers, setAnswers] = useState<string[]>(() =>
    payload.questions.map(() => ''),
  )

  const setAt = (i: number, v: string) =>
    setAnswers((prev) => prev.map((x, k) => (k === i ? v : x)))

  // 点选候选项：再点一次取消（用户点错了不用手动删文字）
  const toggleOption = (i: number, opt: string) =>
    setAt(i, answers[i] === opt ? '' : opt)

  const ready = answers.every((a) => a.trim().length > 0)

  const submit = () => {
    if (!ready || disabled) return
    const lines = payload.questions.map(
      (q, i) => `${i + 1}. ${q.question}\n   答：${answers[i].trim()}`,
    )
    onSubmit(lines.join('\n'))
  }

  return (
    <div className="mt-2 rounded-lg border border-brand-100 bg-brand-50/50 p-3">
      <div className="flex items-center gap-1.5 text-[13px] font-semibold text-brand-700">
        <MessageCircleQuestion className="h-3.5 w-3.5 shrink-0" />
        为了答准，我需要先确认几件事
      </div>
      {payload.reason && (
        <div className="mt-1 text-[11px] leading-5 text-ink-500">{payload.reason}</div>
      )}

      <div className="mt-2.5 space-y-2.5">
        {payload.questions.map((q, i) => (
          <div key={`${i}-${q.question}`}>
            <div className="text-[12px] leading-5 text-ink-700">
              <span className="mr-1 text-ink-400">{i + 1}.</span>
              {q.question}
            </div>
            {q.options.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1.5">
                {q.options.map((opt) => {
                  const active = answers[i] === opt
                  return (
                    <button
                      key={opt}
                      type="button"
                      onClick={() => toggleOption(i, opt)}
                      disabled={disabled}
                      aria-pressed={active}
                      /* touch-target 保证触屏下 ≥44px 高（见 index.css） */
                      className={`touch-target rounded-full border px-2.5 py-1 text-[12px] transition-colors disabled:opacity-50 ${
                        active
                          ? 'border-brand-500 bg-brand-500 text-white'
                          : 'border-slate-200 bg-white text-ink-700 hover:border-brand-500 hover:text-brand-600'
                      }`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>
            )}
            <input
              value={answers[i]}
              onChange={(e) => setAt(i, e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  submit()
                }
              }}
              disabled={disabled}
              placeholder={q.options.length ? '或直接输入…' : '请输入…'}
              aria-label={q.question}
              className="mt-1 w-full rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-[13px] text-ink-900 placeholder:text-ink-400 focus:border-brand-500 focus:outline-none disabled:bg-slate-50"
            />
          </div>
        ))}
      </div>

      <div className="mt-2.5 flex items-center gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={!ready || disabled}
          className="touch-target inline-flex items-center gap-1 rounded-md bg-brand-600 px-3 py-1.5 text-[12px] font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-ink-200 disabled:text-ink-400"
        >
          <Send className="h-3 w-3" />
          提交并继续
        </button>
        {!ready && (
          <span className="text-[11px] text-ink-400">填完全部问题后提交</span>
        )}
      </div>
    </div>
  )
}

/**
 * 阶段耗时面板。
 *
 * ★ 为什么要这个面板
 * --------------
 * 用户问"为什么这次这么慢"时，只有总耗时答不出来：要区分是**检索慢**、
 * **工具执行慢**还是**模型生成慢**——三者的应对完全不同
 * （检索慢要查向量库/网络，工具慢要查数据源，生成慢才是模型问题）。
 *
 * ★ 数据来源与口径
 * ------------
 * `phase_times` 由后端在真实边界处打点（`_mark_phase`），前端只做展示。
 * 显示"分阶段合计"与"端到端总耗时"**两个数**，差额即未被任何阶段覆盖的
 * 时间（初始化、会话记忆、答案核对、SSE 收尾）。不把差额藏起来是因为：
 * 若差额很大（如占了一半），那本身就是"哪里慢"的答案。
 *
 * ★ 交互
 * ----
 * 默认**收起**（一行摘要），点开看明细与占比条。
 * 与"思考过程"面板的默认展开不同——思考过程是回答的一部分（用户要看），
 * 耗时是元数据（想知道时才看）。两者定位不同，故默认态也不同。
 */
function TimingPanel({
  phaseTimes,
  totalMs,
  phaseTotalMs,
  llmTrace,
  toolTimings,
}: {
  phaseTimes: [string, number, number?][]
  totalMs?: number
  phaseTotalMs?: number
  llmTrace?: LLMTrace | null
  toolTimings?: ToolTiming[]
}) {
  const [open, setOpen] = useState(false)

  // 同名阶段可能重复出现（如"补充检索"多轮、"分析检索结果"多轮）——
  // 顺序追加，**不合并**，否则会抹掉"多轮检索各花多久"这个关键信息
  const sum = phaseTotalMs ?? phaseTimes.reduce((acc, p) => acc + (p[1] || 0), 0)
  const max = Math.max(1, ...phaseTimes.map((p) => p[1] || 0))
  // 未被任何阶段覆盖的时间（可能为负：阶段有重叠，如流式生成期间还在计时）
  const uncovered = totalMs !== undefined ? totalMs - sum : undefined

  const calls = llmTrace?.calls ?? []
  const tools = toolTimings ?? []
  // 缓存命中率：DeepSeek 对命中部分打折计费，故这是**省钱信号**。
  // 分母用"命中+未命中"而不是 prompt_tokens——后者可能把缓存计入也可能不计。
  const hit = llmTrace?.cache_hit_tokens ?? null
  const miss = llmTrace?.cache_miss_tokens ?? null
  const cacheRate =
    hit !== null && miss !== null && hit + miss > 0
      ? Math.round((hit / (hit + miss)) * 100)
      : null

  if (!phaseTimes.length && totalMs === undefined && !calls.length) return null

  // ★ 分阶段耗时堆叠条（手写 SVG/色块，零依赖）
  //
  // 为什么要它：原先是"每阶段一条独立进度条"，各自按 max 归一 ——
  // 于是**看不出阶段之间的比例关系**（4 条都接近满格时，谁是大头要靠读数字）。
  // 堆叠条把总时长当整条，每段宽 = 该阶段占比，一眼就能看出构成。
  //
  // 未覆盖的时间也画进去（`uncovered`）：否则各段之和小于整条，
  // 剩下的空白会让人以为"少画了一段"。
  const phaseSegments = (() => {
    const segs: { key: string; label: string; value: number; color: string }[] = []
    phaseTimes.forEach(([name, ms], i) => {
      const v = ms || 0
      if (v > 0) segs.push({ key: `${name}-${i}`, label: name, value: v, color: phaseColor(i) })
    })
    if (uncovered !== undefined && uncovered > 0) {
      segs.push({
        key: '__uncovered__',
        label: '未分阶段',
        value: uncovered,
        color: '#E2E8F0',
      })
    }
    return segs
  })()

  return (
    <div className="mt-2 overflow-hidden rounded-md border border-slate-200 bg-slate-50/70">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-[11px] font-medium text-ink-500 hover:bg-slate-100"
        aria-expanded={open}
        aria-label={open ? '收起耗时明细' : '展开耗时明细'}
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <Timer className="h-3 w-3 shrink-0 text-ink-400" />
        <span className="whitespace-nowrap">总用时 {fmtSec(totalMs)}</span>
        {/* 摘要行也带上 token：不展开就能看到成本 */}
        {llmTrace?.total_tokens != null && (
          <span className="truncate font-normal text-ink-400">
            · {llmTrace.total_tokens.toLocaleString()} tokens
          </span>
        )}
        {cacheRate !== null && (
          <span
            className="truncate font-normal text-emerald-600"
            title="prompt 缓存命中率。DeepSeek 对命中部分打折计费，命中率越高越省"
          >
            · 缓存 {cacheRate}%
          </span>
        )}
        <span className="ml-auto shrink-0 font-normal text-ink-400">
          {open ? '收起' : '明细'}
        </span>
      </button>
      {open && (
        <div className="space-y-1 border-t border-slate-200 px-3 py-2">
          {/* ★ 堆叠占比条 + 图例：先给"整体构成"，再列各阶段明细。
              放在最上面是因为它回答的是"这一轮时间花在哪"这个首要问题。 */}
          {phaseSegments.length > 0 && (
            <div className="space-y-1 pb-1">
              <StackedBar segments={phaseSegments} height={10} />
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-500">
                {phaseSegments.map((s) => (
                  <ColorDot key={s.key} color={s.color}>
                    {s.label}
                  </ColorDot>
                ))}
              </div>
            </div>
          )}
          {phaseTimes.map(([name, ms, offset], i) => (
            <div key={`${name}-${i}`} className="flex items-center gap-2 text-[11px]">
              {/* 阶段名固定宽度：中文名长短不一（"首轮分析" vs "补充检索（第2轮）"），
                  不对齐则秒数列会参差 */}
              <span className="w-[116px] shrink-0 truncate text-ink-600" title={name}>
                {name}
              </span>
              {/* 占比条：颜色与上面堆叠条**同一序号同一色**，
                  这样明细行与堆叠段能对上（否则要读文字才能对应）。 */}
              <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-200">
                <span
                  className="block h-full rounded-full"
                  style={{ width: `${Math.max(2, ((ms || 0) / max) * 100)}%`, background: phaseColor(i) }}
                />
              </span>
              <span className="w-[48px] shrink-0 text-right tabular-nums text-ink-700">
                {fmtSec(ms)}
              </span>
              {/* 起始偏移：能看出阶段之间的空档 */}
              <span
                className="w-[52px] shrink-0 text-right tabular-nums text-ink-300"
                title="该阶段开始于本轮问答的第几秒"
              >
                {offset !== undefined ? `@${fmtSec(offset)}` : ''}
              </span>
            </div>
          ))}
          <div className="mt-1.5 flex items-center justify-between border-t border-slate-200 pt-1.5 text-[11px]">
            <span className="text-ink-500">分阶段合计</span>
            <span className="tabular-nums text-ink-700">{fmtSec(sum)}</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-ink-500" title="端到端总耗时（含初始化、记忆检索、答案核对、SSE 收尾）">
              端到端总耗时
            </span>
            <span className="tabular-nums font-medium text-ink-700">{fmtSec(totalMs)}</span>
          </div>
          {uncovered !== undefined && uncovered > 0 && (
            <div className="flex items-center justify-between text-[11px]">
              <span
                className="text-ink-400"
                title="未被任何阶段覆盖：初始化、会话记忆检索、答案核对、SSE 收尾等"
              >
                其他（未分阶段）
              </span>
              <span className="tabular-nums text-ink-400">{fmtSec(uncovered)}</span>
            </div>
          )}

          {/* ---- LLM 调用与 token ----
              这一组回答"花了多少钱、慢在哪一类调用"：
              by_kind 能区分"慢在决策轮"与"慢在生成"，优化方向完全不同。 */}
          {llmTrace && calls.length > 0 && (
            <div className="mt-2 border-t border-slate-200 pt-1.5">
              <div className="mb-1 text-[11px] font-medium text-ink-500">
                LLM 调用（{llmTrace.call_count} 次
                {llmTrace.failed_count > 0 && (
                  <span className="text-danger"> · 失败 {llmTrace.failed_count}</span>
                )}
                ）
              </div>
              {Object.entries(llmTrace.by_kind).map(([kind, g]) => (
                <div key={kind} className="flex items-center justify-between text-[11px]">
                  <span className="text-ink-500">
                    {KIND_LABEL[kind] ?? kind} × {g.count}
                  </span>
                  <span className="tabular-nums text-ink-700">{fmtSec(g.ms)}</span>
                </div>
              ))}
              <div className="mt-1 flex items-center justify-between text-[11px]">
                <span className="text-ink-500">token（输入 / 输出）</span>
                <span className="tabular-nums text-ink-700">
                  {llmTrace.prompt_tokens?.toLocaleString() ?? '未知'} /{' '}
                  {llmTrace.completion_tokens?.toLocaleString() ?? '未知'}
                </span>
              </div>
              {hit !== null && miss !== null && (
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-ink-400" title="命中缓存的 prompt token 按折扣计费">
                    prompt 缓存命中
                  </span>
                  <span className="tabular-nums text-emerald-600">
                    {hit.toLocaleString()} / {(hit + miss).toLocaleString()}
                    {cacheRate !== null && ` (${cacheRate}%)`}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ---- 每个工具的执行耗时 ----
              与上面"检索与搜索"整阶段合计互补：一轮里可能调多个工具，
              这里能看出是哪个慢、有没有重复调同一个。 */}
          {tools.length > 0 && (
            <div className="mt-2 border-t border-slate-200 pt-1.5">
              <div className="mb-1 text-[11px] font-medium text-ink-500">
                工具执行（{tools.length} 次）
              </div>
              {tools.map((t, i) => (
                <div key={`${t.name}-${i}`} className="flex items-center gap-2 text-[11px]">
                  <span className="w-[24px] shrink-0 text-ink-300">第{t.round}轮</span>
                  <span className="min-w-0 flex-1 truncate text-ink-600" title={t.name}>
                    {t.name}
                  </span>
                  {!t.ok && <span className="shrink-0 text-danger">失败</span>}
                  <span className="w-[52px] shrink-0 text-right text-ink-300">
                    {t.sources > 0 ? `${t.sources} 条` : ''}
                  </span>
                  <span className="w-[48px] shrink-0 text-right tabular-nums text-ink-700">
                    {fmtSec(t.ms)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * 单条 AI 消息。
 *
 * ★ 为什么要 `memo`（实测：一次回答把历史消息重渲染了 ~200 次）
 * ----------------------------------------------------------
 * 每收到一个 token 就 `setMessages`，父组件重渲染 → 列表里**每条**消息
 * 都跟着重渲染，而每条重渲染都要把该消息的 Markdown 从头解析一遍。
 * 实测 240 个 token 累计 760 ms CPU（`logs/fe_markdown_bench.mjs`），
 * 且随会话变长线性放大（消息越多，每条都被重解析一次）。
 *
 * `memo` 能生效的前提是 props 引用稳定：
 *   · `m`：`updateMessage` 只给目标消息换新对象，其余**保持同一引用** ✅
 *   · `onScroll` / `onSubmitClarification`：经 `useStableCallback` 包装，
 *     外壳引用恒定（见 `hooks/useStableCallback.ts`）✅
 *   · `busy`（即 `streaming`）：一轮只变两次 ✅
 * 于是流式期间**只有正在生成的那一条**会重渲染，历史消息全部跳过。
 *
 * ⚠️ 内部状态（点赞/复制提示）不受影响：`memo` 只影响"父组件要不要重渲染我"，
 * 自身的 `useState` 变化照样触发更新。
 */
const AiBubble = memo(function AiBubble({
  m,
  sessionId,
  onScroll,
  busy,
  onSubmitClarification,
}: AiBubbleProps) {
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
      {/* ★ 外层也要 min-w-0
          
          只有内层气泡有 `min-w-0` 是不够的：flex 项的默认 `min-width: auto`
          会让**这一层**按内容撑开，而它的宽度上限是 `max-w-[85%]`——百分比
          相对的是已被子元素撑宽的容器，形成"内容越宽、容器越宽"的正反馈。
          表现就是整页横向溢出、右侧被截断（截图实测）。
          `break-words` 处理长 URL / 无空格长串：没有它，一个长链接
          会无视容器宽度直接顶出去。

          注意是 `items-start` 而非 `items-stretch`：气泡要**贴合内容宽度**，
          拉伸到整宽就不是气泡了（短回答会拖出一条长白条）。
          元数据行（复制/有用/没帮助）则靠内层 `flex items-center` 排版。 */}
      <div className="flex min-w-0 max-w-[85%] flex-col items-start gap-1.5">
        <div className="flex min-w-0 max-w-full flex-col rounded-xl border border-slate-200 bg-white px-4 py-3 break-words">
        {/* 思考折叠块（不落库，仅展示）。默认展开、思考完成后仍展开，
            用户可手动收起（见 ThinkingBlock 的交互约定说明）。 */}
        {m.thinking && (
          <ThinkingBlock
            text={m.thinking}
            streaming={m.status === 'streaming'}
            statusText={m.statusText}
          />
        )}
        <div className="prose prose-sm max-w-none text-[14px] leading-6 text-ink-700">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
            {m.content || ''}
          </ReactMarkdown>
          {m.status === 'streaming' && (
            <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-brand-600 align-middle" />
          )}
        </div>
        {/* ★ RAG 检索耗时：检索内部的步骤分解 */}
        <RagTimingPanel timing={m.ragTiming} />
        {/* ★ 数据图表：来自工具原始输出的结构化数值（后端提取）
            prose 用于去重：模型常在正文里自己画一张同样的表格，
            那种情况下再画一遍图就是重复（实测见过正文表格与图表数值完全一致）。 */}
        <AnswerCharts charts={m.charts} prose={m.content} />
        {/* ★ 首轮追问卡片：正文为空，取而代之是问题与候选项 */}
        {m.status === 'clarify' && m.clarification && (
          <ClarifyCard
            payload={m.clarification}
            disabled={busy || m.clarifiedAnswered}
            onSubmit={(answer) => onSubmitClarification(m.id, answer)}
          />
        )}
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
              {/* 自动兜底时说明为什么联网了——不解释用户会以为系统乱联网 */}
              {m.webFallback && !m.webFallback.enough && (
                <Badge color="warn">
                  <AlertTriangle className="h-3 w-3" />
                  本地无匹配，已自动联网
                </Badge>
              )}
            {/* 总耗时与各阶段明细都收进 TimingPanel（原先是这里一行纯文本，
                只能看到总数、答不出"哪一段慢"）。 */}
          </div>
        )}
        {/* 阶段耗时面板：默认收起（一行"总用时 X.Xs"），点开看各阶段与占比。
            放在元数据行之后、来源列表之前——它属于回答的元信息，
            而来源是回答的**内容依据**，用户的阅读优先级更高。 */}
        {m.status === 'done' && (m.phaseTimes?.length || m.elapsedMs !== undefined) && (
          <TimingPanel
            phaseTimes={m.phaseTimes ?? []}
            totalMs={m.elapsedMs}
            phaseTotalMs={m.phaseTotalMs}
            llmTrace={m.llmTrace}
            toolTimings={m.toolTimings}
          />
        )}
        {/*
          来源与相关性评分。改造前这里只有"来源 N""联网 N"两个数量徽标——
          用户看不到具体来源、也看不到每条有多相关，无法判断该不该采信。
        */}
        <SourceList
          localSources={m.sources}
          webSources={m.webSources}
          fallbackNote={m.webFallback && !m.webFallback.enough ? m.webFallback.reason : undefined}
          threshold={m.webFallback?.threshold}
        />
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
    </div>
  )
})
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
  // 联网模式三态：auto（本地不够用时自动联网，默认）/ on（强制）/ off（只用本地）
  const [webMode, setWebMode] = useState<WebSearchMode>('auto')
  const [deepThinking, setDeepThinking] = useState(false)
  const [sessions, setSessions] = useState<ConversationDetail[]>([])
  /** 历史搜索关键词（空 = 不搜索，展示普通列表） */
  const [searchKeyword, setSearchKeyword] = useState('')
  const [searchHits, setSearchHits] = useState<ConversationSearchHit[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  /** 正在内联重命名的会话 id（null = 没有） */
  const [renamingId, setRenamingId] = useState<number | null>(null)
  /** 待发送的附件（已压缩好的图片，可多张）。空数组 = 无附件。
   *
   *  ★ 为什么是数组而不是单张：来源变多了（选择文件 / 粘贴 / 拖拽），
   *  三条路径共用同一个"加入"逻辑，数组是它们的自然交汇点。
   *  上限与体积闸门在 `lib/imageQueue.ts`（纯函数，带单测）。 */
  const [attachments, setAttachments] = useState<PreparedImage[]>([])
  /** 图片正在压缩（大图重编码需要几百毫秒，给个转圈反馈） */
  const [preparing, setPreparing] = useState(false)
  /** 是否正有文件拖到输入区上方（用于高亮提示可放下） */
  const [dragging, setDragging] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  /** 是否跟随最新内容（用户手动上翻后为 false）。 */
  const followRef = useRef(true)
  /** 同一帧内合并多次滚动请求（流式时每个 token 都会请求一次）。 */
  const scrollRafRef = useRef<number | null>(null)
  /**
   * 滚动跟随：只在"用户本来就在底部"时跟随新内容。
   *
   * ★ 修掉了两个问题（原来是无条件 `scrollTop = scrollHeight`）
   * ------------------------------------------------------
   * ① **用户往上翻会被拽回底部**：流式生成时每个 token 都会触发一次滚动，
   *    用户想回看上文根本做不到 —— 刚滚上去就被拉到底。
   * ② **每个 token 一次强制同步布局**：`scrollTop = scrollHeight` 会强制
   *    浏览器立刻重排正在变长的 DOM，一秒几十次。现在合并到**每帧最多一次**
   *    （`requestAnimationFrame`），且不在底部时不写。
   *
   * `force=true` 用于"用户主动要求到底部"的场景（发消息、切会话）——
   * 这时即使用户之前翻在上面，也应该跳到底部看新内容。
   */
  const scrollToBottom = useCallback((force = false) => {
    const el = scrollRef.current
    if (!el) return
    if (force) followRef.current = true
    if (!followRef.current) return
    if (scrollRafRef.current !== null) return // 同一帧只写一次
    scrollRafRef.current = requestAnimationFrame(() => {
      scrollRafRef.current = null
      const node = scrollRef.current
      if (node && followRef.current) node.scrollTop = node.scrollHeight
    })
  }, [])

  /**
   * 判断用户是否还在底部附近（阈值 80px）。
   *
   * 为什么用阈值而不是 `===`：① 浏览器滚动是小数，精确相等几乎不可能；
   * ② 触控板/滚轮的"到底"往往差几个像素；③ 留一点余量后，
   * 用户轻微回滚不会立刻退出跟随模式（否则会感觉"跟随时好时坏"）。
   */
  const handleListScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    followRef.current = isNearBottom(el)
  }, [])
  const sessionIdRef = useRef<string>(getSessionId())
  const refreshSessions = useCallback(async () => {
    try {
      const list = await listConversationsDetailed()
      setSessions(list)
    } catch {
      /* 列表拉取失败静默：侧栏为空，不影响主对话 */
    }
  }, [])
  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  /**
   * 搜索防抖：350ms。
   *
   * 为什么要防抖：搜索走的是数据库 ILIKE 查询，逐字符触发会在用户
   * 输入"中国大陆和香港"这 7 个字时打 7 次请求。350ms 是实测手感与
   * 请求数的平衡点（中文输入法一次上屏通常在这个间隔内完成）。
   */
  useEffect(() => {
    const kw = searchKeyword.trim()
    if (!kw) {
      setSearchHits([])
      setSearchLoading(false)
      return
    }
    setSearchLoading(true)
    const timer = setTimeout(() => {
      searchConversations(kw)
        .then(setSearchHits)
        .catch(() => setSearchHits([]))
        .finally(() => setSearchLoading(false))
    }, 350)
    return () => clearTimeout(timer)
  }, [searchKeyword])

  const searching = searchKeyword.trim().length > 0

  /** 提交重命名：空标题直接放弃（后端也会拒绝，但不必白跑一次请求） */
  const commitRename = useCallback(
    async (sessionId: string, title: string) => {
      setRenamingId(null)
      const next = title.trim()
      if (!next) return
      try {
        await renameConversation(sessionId, next)
        await refreshSessions()
        toast.show('标题已更新', 'success')
      } catch {
        toast.show('重命名失败', 'error')
      }
    },
    [refreshSessions, toast],
  )
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
          // 老库 / 该列不可用时为 null —— 按"这条消息没有图片"处理
          imageAnalyses:
            Array.isArray(r.imageAnalyses) && r.imageAnalyses.length
              ? (r.imageAnalyses as ImageAnalysis[])
              : undefined,
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
    async (text: string, answer: ChatMessage, imageAnalyses?: ImageAnalysis[]) => {
      try {
        const ids = await saveConversation(sessionIdRef.current, text, [
          {
            role: 'user',
            content: text,
            // ★ 图片解析文本随消息落库（原图不入库，见后端 image_analyses 列说明）
            //
            // 直接存**原始解析文本**、而不是拼接后的 questionForAgent：
            // 拼接文本里混着"【用户问题】"等脚手架，存下来既不好展示，
            // 也没法在未来换一种合并格式时复用。需要时用 mergeImageAnalyses 重建。
            imageAnalyses: imageAnalyses?.length ? imageAnalyses : undefined,
          },
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
  /**
   * 统一的"加入图片"入口：选择文件 / 粘贴 / 拖拽三条路径共用。
   *
   * ★ 为什么必须共用一条路径
   * ----------------------
   * 三条来源的差异只是"文件从哪来"，而闸门（张数/体积/类型）、压缩、
   * 去重、提示文案这些逻辑完全一样。各写一份的话，最典型的后果是
   * **粘贴进来的图绕过体积闸门**（拖拽那份写了、粘贴那份忘了），
   * 直到某个用户粘了一张 30MB 的截图才发现。
   *
   * 逐张压缩而不是并行：`prepareImage` 内部走 canvas 重编码，
   * 并行几张会让浏览器同时持有多个大画布（内存尖峰），而这里总共也就几张图，
   * 串行的总耗时完全可以接受（每张几百毫秒）。
   */
  const addImages = useCallback(
    async (files: File[] | FileList | null | undefined) => {
      const list = Array.from(files ?? []).filter(Boolean)
      if (!list.length) return

      // ★ 先按类型分流：PDF 要**先转成图片**，再和图片一起过同一道闸门
      //
      // 顺序很关键：闸门（张数/体积/类型）只认图片。若先过闸门再转 PDF，
      // PDF 会被判成 `not_image` 直接丢掉（用户拖了 PDF 却"什么都没发生"）。
      const { images, pdfs, others } = splitByKind(list)
      if (others.length) {
        toast.show(`已跳过 ${others.length} 个非图片/PDF 文件`, 'warning')
      }

      const expanded: File[] = [...images]
      if (pdfs.length) {
        setPreparing(true)
        try {
          // PDF 解析组件是**动态加载**的（首屏不为它买单，见 `lib/pdf.ts`），
          // 故第一次拖 PDF 时会有一次额外的组件下载
          toast.show(
            pdfs.length === 1 ? '正在转换 PDF…' : `正在转换 ${pdfs.length} 份 PDF…`,
            'success',
          )
          for (const pdf of pdfs) {
            try {
              const result = await pdfToImages(pdf)
              expanded.push(...result.files)
              if (result.skipped || result.failed) {
                const parts = [`${pdf.name}：转换 ${result.converted}/${result.pageCount} 页`]
                if (result.skipped) parts.push(`超出 ${MAX_PDF_PAGES} 页的部分已跳过`)
                if (result.failed) parts.push(`${result.failed} 页失败`)
                toast.show(parts.join('；'), 'warning')
              }
            } catch (err) {
              toast.show(
                err instanceof PdfConvertError
                  ? `${pdf.name}：${err.message}`
                  : `${pdf.name}：PDF 转换失败`,
                'error',
              )
            }
          }
        } finally {
          setPreparing(false)
        }
      }

      if (!expanded.length) return

      const plan = planAdditions(attachments.length, expanded)
      if (!plan.accepted.length) {
        const { text, kind } = describeAdditions(plan, [])
        toast.show(text, kind)
        return
      }

      setPreparing(true)
      const prepared: PreparedImage[] = []
      try {
        for (const index of plan.accepted) {
          try {
            prepared.push(await prepareImage(expanded[index]))
          } catch (err) {
            // 单张失败不影响其余：把失败原因汇总到最后的提示里
            const msg = err instanceof ImagePrepareError ? err.message : '图片处理失败'
            toast.show(`${expanded[index]?.name ?? '图片'}：${msg}`, 'error')
          }
        }
        if (prepared.length) {
          setAttachments((prev) => [...prev, ...prepared].slice(0, MAX_ATTACHMENTS))
        }
        const { text, kind } = describeAdditions(plan, prepared)
        toast.show(text, kind)
      } finally {
        setPreparing(false)
        // 清空 input.value：否则再次选同一个文件不会触发 change 事件
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    },
    [attachments.length, toast],
  )

  /** 选择图片（回形针按钮；支持多选） */
  const pickImage = useCallback(
    async (files: FileList | null | undefined) => {
      await addImages(files)
    },
    [addImages],
  )

  /**
   * 粘贴板里的图片（截图后直接 Ctrl+V）。
   *
   * ★ 只在**确实有图片**时拦截默认行为
   * ---------------------------------
   * 剪贴板常常同时带文本与图片（从网页复制一段带图的内容）。
   * 无脑 `preventDefault()` 会让用户粘不进来文字 —— 而文字粘贴是高频操作。
   * 故只有检测到图片文件时才接管，其余情况完全交给浏览器默认行为。
   */
  const handlePaste = useCallback(
    (event: React.ClipboardEvent<HTMLElement>) => {
      const files = Array.from(event.clipboardData?.files ?? [])
      const images = files.filter((f) => f.type.startsWith('image/'))
      if (!images.length) return
      event.preventDefault()
      void addImages(images)
    },
    [addImages],
  )

  /** 拖拽：进入/离开只切高亮，真正落地在 drop（见下方输入区） */
  const handleDragOver = useCallback((event: React.DragEvent<HTMLElement>) => {
    // 必须 preventDefault：否则浏览器按默认行为**打开**这个文件
    // （整个页面跳走，用户以为程序崩了）
    if (Array.from(event.dataTransfer?.types ?? []).includes('Files')) {
      event.preventDefault()
      setDragging(true)
    }
  }, [])

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLElement>) => {
    // 只有真正离开容器才取消高亮：拖动时会在子元素之间反复触发 leave/enter，
    // 不判断的话高亮会闪。relatedTarget 仍在容器内就忽略。
    const next = event.relatedTarget as Node | null
    if (next && event.currentTarget.contains(next)) return
    setDragging(false)
  }, [])

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLElement>) => {
      event.preventDefault()
      setDragging(false)
      void addImages(event.dataTransfer?.files)
    },
    [addImages],
  )
  /**
   * 提交追问的回答。
   *
   * ★ 协议：追问的回答是**新一轮请求**，不是同一次流式里的续传
   * --------------------------------------------------
   * 后端的 `clarify` 是本轮终止事件（不检索、不生成）。用户答完后，
   * 前端把「原问题 + 补充信息」作为新问题发出去；此时会话已有历史，
   * 后端的 `clarify_allowed` 为假 → 不会再追问，直接检索作答。
   *
   * ★ 为什么必须在这里**显式构造 history**，不能复用 `send` 的取法
   * ------------------------------------------------------
   * `send` 里是 `messages.filter(m => m.status === 'done')`——它读的是**闭包里的
   * messages 快照**。而此刻我们刚从 `setMessages` 改了状态（把追问卡片标记为已答），
   * 快照是旧的：追问那轮的 user 消息仍是 'done'，会被算进 history，
   * 于是**原问题在 history 与当前问题里各出现一次**。
   * 后果是模型看到重复的问题，答非所问或重复叙述。
   *
   * 故这里把 history 构造得早于"追问那轮"，并显式排除它：
   * 取到追问对应的 user 消息下标，只保留它**之前**的已完成回合。
   *
   * ★ 为什么合并成一个问题（而不是拆成两条 user 消息）
   * -------------------------------------------
   * 后端按"最后一条 user 消息"理解当前问题。拆成两条会让
   * "原问题"退化成上下文，而模型未必会把它当作待答目标。
   * 合并则保持"一个问题 + 补充说明"的清晰结构。
   */
  /**
   * 实际发起一次流式问答。`send` 与 `submitClarification` 共用。
   *
   * 拆出这一层是为了让"追问续答"能**自带 history** 而不必迁就 `send`
   * 从 state 推导 history 的写法（见 `submitClarification` 的说明）。
   */
  const sendInternal = useCallback(
    async (question: string, history: { role: string; content: string }[],
           images?: PreparedImage[] | null) => {
      if (streaming) return
      const picked = images ?? []
      const uid = nextId()
      const aid = nextId()
      const userMsg: ChatMessage = {
        id: uid,
        role: 'user',
        content: question,
        status: 'done',
        // 单张时仍写 imageUrl（历史渲染路径不变）；多张写 imageUrls
        imageUrl: picked.length === 1 ? picked[0].previewUrl : undefined,
        imageUrls: picked.length > 1 ? picked.map((p) => p.previewUrl) : undefined,
        // 注：`imageAnalyses` 在**解析完成后**才回填（此刻还没跑视觉模型），
        // 见下面 `updateMessage(uid, { imageAnalyses })`
      }
      const aiMsg: ChatMessage = { id: aid, role: 'assistant', content: '', status: 'pending' }
      setMessages((prev) => [...prev, userMsg, aiMsg])
      setInput('')
      setAttachments([])
      setStreaming(true)
      scrollToBottom()
      const controller = new AbortController()
      abortRef.current = controller
      /* ★ token 合并：同一帧内的多个 chunk 只触发一次 setState
         --------------------------------------------------
         后端一个 token 一个 SSE 帧（实测一次回答约 240 帧），原来每帧都
         `setMessages` → 该条消息的 Markdown **整篇重解析**。
         合并逻辑在 `lib/streamBuffer.ts`（纯函数，带单测）：它同时处理了
         "末帧不丢"、"中止不覆盖状态"、"reset 后迟到帧不复活"这三个易错点。
         这里只负责把提交动作接到 React 状态上。 */
      const buffer = createStreamBuffer({
        onFlush: ({ content, thinking }) => {
          updateMessage(aid, (m) => ({
            status: 'streaming',
            content: m.content + content,
            ...(thinking ? { thinking: (m.thinking ?? '') + thinking } : {}),
          }))
        },
      })
      const flushPending = (opts?: { keepStatus?: boolean }) => {
        // 状态已被中止/报错改写时，只补内容，不把 status 改回 streaming
        if (!opts?.keepStatus) {
          buffer.flush()
          return
        }
        const { content, thinking } = buffer.peek()
        buffer.clear()
        if (!content && !thinking) return
        updateMessage(aid, (m) => ({
          content: m.content + content,
          ...(thinking ? { thinking: (m.thinking ?? '') + thinking } : {}),
        }))
      }
      const done = (payload: DoneEvent) => {
        // ★ 先把缓冲里的最后几个 token 落进 state，再用 state 拼最终结果：
        //   否则末尾 1 帧的内容会丢（`done` 常在最后一个 token 的同一帧到达）
        flushPending()
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aid
              ? {
                  ...m,
                  // ★ 追问轮**不能**覆写成「未生成回答」
                  //
                  // `clarified` 为真时本轮本来就没有正文（后端不检索不生成），
                  // 而 `content || '（未生成回答）'` 会把它写成"未生成回答"，
                  // 同时把 status 改成 'done' —— 于是刚渲染出来的追问卡片
                  // 立刻被一个"未生成回答"的空白气泡取代，用户**看不到问题**。
                  // 实测：`clarify` 事件与 `done` 事件相隔极短，肉眼只看到后者。
                  content: payload.clarified
                    ? m.content
                    : m.content || '（未生成回答）',
                  status: payload.clarified ? 'clarify' : 'done',
                  // 追问轮没有来源与工具，不要用空数组覆盖（保持 undefined）
                  sources: payload.clarified ? m.sources : (payload.sources ?? []),
                  webSources: payload.clarified
                    ? m.webSources
                    : ((payload.web_sources ?? []) as unknown as WebSourceItem[]),
                  webFallback: payload.web_fallback ?? null,
                  toolCalled: payload.tool_called,
                  toolName: payload.tool_name,
                  elapsedMs: payload.elapsed_ms,
                  phaseTimes: payload.phase_times ?? [],
                  phaseTotalMs: payload.phase_total_ms,
                  llmTrace: payload.llm_trace ?? null,
                  toolTimings: payload.tool_timings ?? [],
                  charts: payload.charts ?? [],
                  ragTiming: payload.rag_timing ?? null,
                }
              : m,
          ),
        )
      }
      try {
        // ── 阶段一：有附件先做视觉解析 ──────────────────────────────
        // 把解析结果**拼进问题文本**而不是新增一个后端参数：这样 Agent 的
        // RAG / 图谱 / 业务库工具仍能正常工作（用户可能边传公告截图边问"这个项目谁中标过"），
        // 也避免了为一次性的图片上下文改动 chat 请求契约。
        let questionForAgent = question
        // ★ 提到 if 外面：落库（persist）时要用它，而 persist 的调用点在
        //   try 末尾、`if (picked.length)` 作用域之外
        let imageAnalyses: ImageAnalysis[] = []
        if (picked.length) {
          // ★ 逐张解析（串行），每张都报进度
          //
          // 为什么串行而不是 Promise.all：① 视觉模型有并发限流，几张图同时打
          //   容易触发 429；② 串行才能给出"第 2/3 张"这种**确定的进度**；
          //   ③ 单张失败时能立刻继续下一张，而不是整体回滚。
          const analyses: ImageAnalysis[] = []
          let lastModel = ''
          for (let i = 0; i < picked.length; i += 1) {
            const item = picked[i]
            updateMessage(aid, {
              status: 'streaming',
              content:
                picked.length === 1
                  ? '正在解析图片…'
                  : `正在解析图片 ${i + 1}/${picked.length}…`,
            })
            try {
              const vision = await analyzeImage(item.dataUrl, '', controller.signal)
              lastModel = vision.model || lastModel
              analyses.push({
                name: item.name,
                analysis: (vision.analysis || '').trim(),
                model: vision.model || undefined,
              })
            } catch (err) {
              // 中止要**立刻冒泡**（用户点了停止，不该继续解析后面的图）
              if (err instanceof DOMException && err.name === 'AbortError') throw err
              // 单张失败：记一条空的，继续解析其余（最后统一提示）
              analyses.push({ name: item.name, analysis: '' })
              toast.show(
                `${item.name}：${err instanceof ApiError ? err.message : '图片解析失败'}`,
                'error',
              )
            }
          }
          imageAnalyses = analyses
          // 回填到用户消息上：
          //   ① 刷新后仍能看到"当时从图里读出了什么"（原图不入库）
          //   ② 后续轮次重建 history 时用它还原图片上下文
          updateMessage(uid, {
            imageAnalyses: analyses.length ? analyses : undefined,
            ...(lastModel ? { imageModel: lastModel } : {}),
          })
          const merged = mergeImageAnalyses(analyses)
          const okCount = analyses.filter((a) => a.analysis).length
          if (merged) {
            questionForAgent = `${merged}\n\n【用户问题】\n${question}`
            if (okCount < analyses.length) {
              // 部分成功：明确告知，避免用户以为"全都看过了"
              toast.show(
                `${analyses.length} 张图里解析成功 ${okCount} 张，按已完成的部分作答`,
                'warning',
              )
            }
          } else {
            updateMessage(aid, {
              status: 'error',
              errorText: '图片解析没有返回内容，请换一张图或直接用文字提问',
            })
            return
          }
          updateMessage(aid, { content: '' })
        }
        // ── 阶段二 + 三：交给 Agent（检索 → 生成）─────────────────
        await askStream(
          {
            question: questionForAgent,
            history,
            web_search_mode: webMode,
            provider: model,
            deep_thinking_enabled: deepThinking,
          },
          {
            onStatus: (text) =>
              updateMessage(aid, { status: 'streaming', statusText: text }),
            onToken: (chunk) => buffer.pushToken(chunk),
            onThinking: (chunk) => buffer.pushThinking(chunk),
            onReset: () => {
              // 后端要求丢弃已生成内容：缓冲也要一起清，否则迟到的帧会把
              // 刚清空的气泡又填回去
              buffer.clear()
              updateMessage(aid, { content: '' })
            },
            // ★ 首轮追问：把问题挂到消息上，界面切到"等待用户回答"
            //
            // 这里**不产正文**、也不结束这轮消息——消息保留在列表里，
            // 用户答完后点击提交，会把「原问题 + 补充信息」作为新问题发出去。
            // 故本消息的状态设为 'clarify'（既不是 streaming 也不是 done），
            // 由渲染层据此显示追问卡片。
            onClarify: (ev) =>
              updateMessage(aid, {
                status: 'clarify',
                statusText: '',
                clarification: ev,
              }),
            onDone: done,
            onError: (text2) =>
              updateMessage(aid, { status: 'error', errorText: text2 }),
          },
          controller.signal,
        )
        // 流结束后落库（onDone 已更新消息，用最新内容）
        setMessages((prev) => {
          const last = prev.find((m) => m.id === aid)
          if (last && last.status === 'done') persist(question, last, imageAnalyses)
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
              ? err.status === 413
                ? '图片过大，请压缩后重试'
                : err.status === 400
                  ? `图片数据无法解析：${err.message}`
                  : err.status === 429
                    ? '请求过于频繁，请稍后再试'
                    : err.status === 503
                      ? '图片解析服务未就绪（后端未配置 ZHIPU_API_KEY）'
                      : `请求失败：${err.message}`
              : '网络异常，请检查后端服务是否已启动'
          updateMessage(aid, { status: 'error', errorText: friendly })
        }
      } finally {
        // 收尾补提交：正常结束已由 `done` 提交过（这里 no-op）；
        // 中止/报错时把缓冲里最后 1 帧的内容补上（`keepStatus` 避免把
        // "已停止生成"的错误提示覆盖回 streaming）
        flushPending({ keepStatus: true })
        buffer.cancel() // 防止流结束后还有排期的提交打到已结束的消息上
        setStreaming(false)
        abortRef.current = null
        scrollToBottom()
      }
    },
    [
      messages,
      model,
      webMode,
      deepThinking,
      streaming,
      attachments,
      toast,
      updateMessage,
      persist,
      scrollToBottom,
    ],
  )

  /**
   * 提交追问的回答。
   *
   * ★ 协议：追问的回答是**新一轮请求**，不是同一次流式里的续传
   * --------------------------------------------------
   * 后端的 `clarify` 是本轮终止事件（不检索、不生成）。用户答完后，
   * 前端把「原问题 + 补充信息」作为新问题发出去；此时会话已有历史，
   * 后端的 `clarify_allowed` 为假 → 不会再追问，直接检索作答。
   *
   * ★ 为什么必须在这里**显式构造 history**，不能复用 `send` 的取法
   * ------------------------------------------------------
   * `send` 里是 `messages.filter(m => m.status === 'done')`——读的是**闭包里的
   * messages 快照**。而此刻我们刚改了 state（追问卡片标记已答），
   * 快照仍是旧的：追问那轮的 user 消息还是 'done'，会被算进 history，
   * 于是**原问题在 history 与当前问题里各出现一次**，模型看到重复的问题。
   *
   * 故这里把 history 截到"追问那轮之前"，显式排除它。
   *
   * ★ 为什么合并成一个问题（而不是拆成两条 user 消息）
   * -------------------------------------------
   * 后端按"最后一条 user 消息"理解当前问题。拆成两条会让"原问题"退化成
   * 上下文，模型未必把它当待答目标。合并则保持"一个问题 + 补充说明"的结构。
   *
   * 注意：定义位置必须在 `sendInternal` **之后**——`const` 有暂时性死区，
   * 放在前面会在调用时抛 "Cannot access before initialization"。
   */

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim()
      // 有附件时允许空文本（用户只想让模型看图）
      if ((!text && !attachments.length) || streaming) return
      const images = attachments
      const question =
        text ||
        (images.length > 1 ? '请解析这些图片的内容。' : '请解析这张图片的内容。')
      // 组装已完成回合的历史（不含当前问题）
      //
      // ★ 必须排除欢迎语（`id === 'welcome'`）
      // --------------------------------
      // 欢迎语的状态是 `done`（它确实不是"生成中"），于是被这个 filter 算作
      // **一轮已完成对话**。后果：首次提问时 history 非空 → 后端的
      // `clarify_allowed` 恒为假 → **首轮追问功能完全失效**。
      //
      // 实测确认：直接打 API 时"空历史"会追问 3 条、"带 2 条历史"则直接作答；
      // 而界面里发出去的首轮问题永远带着那条欢迎语。
      //
      // 单元测试没抓到这个，因为测试传的是 `history=None`——**前端真实的取法
      // 才是要覆盖的那条路径**。
      const history = messages
        .filter(
          (m) =>
            m.id !== 'welcome' &&
            m.status === 'done' &&
            (m.role === 'user' || m.role === 'assistant'),
        )
        .map(_historyRow)
      await sendInternal(question, history, images)
    },
    [attachments, messages, sendInternal, streaming],
  )

  const submitClarification = useCallback(
    async (assistantId: string, answer: string) => {
      if (streaming) return
      const idx = messages.findIndex((m) => m.id === assistantId)
      if (idx < 1) return
      // 追问卡片对应的用户原问题（紧邻其上一条 user 消息）
      let uidIdx = -1
      for (let i = idx - 1; i >= 0; i -= 1) {
        if (messages[i].role === 'user') {
          uidIdx = i
          break
        }
      }
      if (uidIdx < 0) return
      const combined =
        `${messages[uidIdx].content}\n\n【补充信息】\n${answer}\n\n请据此给出结论。`

      // 历史 = 原问题**之前**的已完成回合（不含原问题本身，也不含追问轮）
      const history = messages
        .slice(0, uidIdx)
        .filter((m) => m.status === 'done')
        .map(_historyRow)

      // 追问卡片标记已答，避免重复提交（输入框与按钮一并禁用）
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, clarifiedAnswered: true } : m)),
      )
      await sendInternal(combined, history)
    },
    [messages, streaming, sendInternal],
  )

  /**
   * 传给被 `memo` 的消息气泡的回调必须是**引用恒定**的。
   *
   * `submitClarification` 依赖 `messages`（要找出追问卡上方那条用户问题），
   * 流式期间 `messages` 每帧都变 → 它的引用每帧都变 → 所有气泡的 memo 失效，
   * 历史消息被反复重渲染。`useStableCallback` 给出恒定外壳 + 最新实现，
   * 既让 memo 生效，又不会让追问卡读到过期的消息列表。
   */
  const stableSubmitClarification = useStableCallback(submitClarification)

  const stop = () => {
    abortRef.current?.abort()
  }
  const handleSend = () => {
    // 校验放在 send 内部（它知道有没有附件）；这里只挡"图片还在压缩中"的情况，
    // 否则用户点了发送但附件尚未就绪，会静默漏掉图片
    if (preparing) {
      toast.show('图片还在处理中，请稍候', 'warning')
      return
    }
    void send(input)
  }
  return (
    <div>
      <PageHeader title="智能问答" description="基于企业知识库的招投标采购智能问答助手" />
      {/* ★ 移动端改为**纵向堆叠**（实测缺陷修复）
          
          原来是 `flex` + 右侧栏 `w-[300px] shrink-0`，即侧栏**永不收缩**。
          375px 屏上：300px 侧栏 + 16px gap = 316px 被占掉，对话区只剩约 59px
          （再扣掉 p-5 的内边距约 20px）——消息气泡里**每行只能放一个字**，
          整段回答变成一根竖条，完全不可读（截图实测确认）。

          修法：窄屏 `flex-col`（对话区占满整宽，侧栏降到下方），
          宽屏恢复 `lg:flex-row` 的左右布局。
          
          顺带修的两处：
            · 高度  h-[calc(100vh-140px)] 是按"只有顶栏"算的；移动端还有
              顶栏(≈52px) + 底部导航(≈56px)，故窄屏要减更多，否则
              **输入框会被底部导航盖住**（滚动也救不回来，因为外层是固定高）
            · gap-4 → gap-3：窄屏纵向堆叠时 16px 间距显得松散 */}
      {/* ★ 高度改由布局分配（原来用估的常量）
          
          原来是 `h-[calc(100vh-140px)]` —— 一个**猜的魔数**：它假设
          "视口减去 140px 就够"。实测这个假设在两个方向都不准：
              · 桌面：可用高度 = vh - 0(顶栏) - 64(main 内边距) → 应减 64
              · 移动：还要再减 53(顶栏) + 54(底部导航) + 80(内边距)
          于是移动端出现"输入框被底部导航盖住"，桌面端又多留了空白
          （`docs/43` 记过这个问题）。

          现在 Layout 已把正文区约束成一屏高（并正确处理了 dvh / 安全区 /
          内边距），这里只需 `lg:h-full` 填满父级即可 —— 不再需要猜任何数字。

          窄屏仍保留 `h-[calc(...)]`：Layout 在窄屏给的是「一屏高 + 页面级滚动」
          的语义，问答页需要一个确定的高度才能让内部消息区滚动、
          同时把输入框固定在底部（这是聊天界面的标准形态）。
          该值在 `docs/43` 已按实测校准（顶栏 53 + 底部导航 54 + 内边距 80）。 */}
      <div className="flex flex-col gap-3 lg:h-full lg:min-h-0 lg:flex-row lg:gap-4">
        {/* 左对话区 */}
        <div className="flex h-[calc(100vh-232px)] min-h-[320px] flex-col overflow-hidden rounded-xl border border-slate-200 bg-white lg:h-auto lg:min-h-0 lg:flex-1">
          <div
            ref={scrollRef}
            onScroll={handleListScroll}
            className="flex-1 space-y-4 overflow-y-auto p-3 sm:p-5"
          >
            {messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="flex justify-end">
                  <div className="flex max-w-[75%] flex-col items-end gap-1.5">
                    {m.imageUrls?.length ? (
                      /* 多图：一行缩略图，点击看原图 */
                      <div className="flex flex-wrap justify-end gap-1.5">
                        {m.imageUrls.map((url, index) => (
                          <a
                            key={index}
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            title={`点击查看第 ${index + 1} 张原图`}
                            className="block overflow-hidden rounded-lg border border-brand-200 transition-opacity hover:opacity-90"
                          >
                            <img
                              src={url}
                              alt={`上传的图片 ${index + 1}`}
                              className="h-20 w-20 object-cover"
                            />
                          </a>
                        ))}
                      </div>
                    ) : (
                      m.imageUrl && (
                        <a
                          href={m.imageUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          title="点击查看原图"
                          className="block overflow-hidden rounded-xl border border-brand-200 transition-opacity hover:opacity-90"
                        >
                          <img
                            src={m.imageUrl}
                            alt="上传的图片"
                            className="max-h-56 w-auto max-w-full object-contain"
                          />
                        </a>
                      )
                    )}
                    <div className="rounded-xl bg-brand-600 px-4 py-2.5 text-[14px] leading-6 text-white">
                      {m.content}
                    </div>
                    {m.imageModel && (
                      <span className="flex items-center gap-1 text-[10px] text-ink-400">
                        <ImageIcon className="h-2.5 w-2.5" />
                        图片由 {m.imageModel} 解析
                      </span>
                    )}
                    {/* 图片解析文本：**刷新页面后唯一还在的图片信息**
                        （原图不入库，见后端 image_analyses 列说明）。
                        默认折叠：它可能很长，展开会把对话挤散；
                        但必须能展开——用户要核对"系统到底从图里读出了什么"。 */}
                    {!!m.imageAnalyses?.length && (
                      <details className="w-full rounded-lg border border-slate-200 bg-white/70 px-2.5 py-1.5 text-left">
                        <summary className="cursor-pointer text-[11px] text-ink-500 marker:text-ink-300">
                          图片解析结果（{m.imageAnalyses.length} 张，原图未保存）
                        </summary>
                        <div className="mt-1.5 space-y-1.5">
                          {m.imageAnalyses.map((item, index) => (
                            <div key={`${item.name}-${index}`} className="text-[11px] leading-5">
                              <span className="text-ink-400">
                                〔图片 {index + 1}：{item.name || '未命名'}〕
                              </span>
                              <div className="whitespace-pre-wrap break-words text-ink-700">
                                {item.analysis || '（这张图没有解析出内容）'}
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ) : (
                <AiBubble
                  key={m.id}
                  m={m}
                  sessionId={sessionIdRef.current}
                  onScroll={scrollToBottom}
                  busy={streaming}
                  onSubmitClarification={stableSubmitClarification}
                />
              ),
            )}
          </div>
          {/* 输入区 */}
          <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
            {/* ★ 移动端精简这一行（实测溢出修复）
                
                原来固定一行放：联网搜索图标 + "联网搜索" + 三态按钮 + 深色思考开关
                + "当前模型：DeepSeek"，实测约 420px > 375px 屏宽 →
                **整行溢出、右侧内容被截断**（截图确认："当前…"只能露一半）。

                精简策略（移动端）：
                  · "联网搜索" 文字隐藏，只留图标（title 仍可长按看到说明）
                  · "当前模型：X" 整段隐藏 —— 顶栏已有模型下拉，重复显示没意义
                  · gap-3 → gap-2，控件 padding 收紧
                宽屏（sm 起）恢复完整文字。 */}
            <div className="mb-2 flex flex-wrap items-center gap-2 sm:gap-3">
              {/* 联网搜索：三态（自动 / 强制 / 关闭） */}
              <div className="flex min-w-0 items-center gap-1.5" title="本地知识库无匹配时自动联网查最新公告与法规更新">
                <Globe className="h-3.5 w-3.5 shrink-0 text-ink-400" />
                <span className="hidden text-[12px] text-ink-600 sm:inline">联网搜索</span>
                <div className="flex overflow-hidden rounded-lg border border-slate-200">
                  {(
                    [
                      ['auto', '自动'],
                      ['on', '强制'],
                      ['off', '关闭'],
                    ] as const
                  ).map(([mode, label]) => (
                    <button
                      key={mode}
                      onClick={() => setWebMode(mode)}
                      title={
                        mode === 'auto'
                          ? '本地知识库无匹配答案时，自动联网搜索最新招投标公告与法规更新'
                          : mode === 'on'
                            ? '每次提问都联网检索（适合明确要最新信息的场景）'
                            : '只用本地知识库，不联网'
                      }
                      className={`px-2 py-1 text-[11px] transition-colors ${
                        webMode === mode
                          ? 'bg-brand-600 font-medium text-white'
                          : 'bg-white text-ink-500 hover:bg-ink-50'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              {/* 深度思考开关 */}
              <label className="flex shrink-0 cursor-pointer items-center gap-1.5 text-[12px] text-ink-600">
                <input
                  type="checkbox"
                  checked={deepThinking}
                  onChange={(e) => setDeepThinking(e.target.checked)}
                  className="h-3.5 w-3.5 accent-purple-600"
                />
                <Brain className="h-3.5 w-3.5 shrink-0 text-purple-500" />
                <span className="whitespace-nowrap">深度思考</span>
              </label>
              {/* 当前模型名：移动端隐藏（顶栏已有模型下拉，重复显示只会挤爆这一行） */}
              <span className="ml-auto hidden text-[11px] text-ink-400 sm:inline">
                当前模型：{PROVIDER_LABEL[model] ?? model}
              </span>
            </div>
            {/* 附件缩略图条：挂了图才出现（支持多张，逐张可删） */}
            {(attachments.length > 0 || preparing) && (
              <div className="mb-2 flex flex-wrap items-center gap-2 rounded-lg border border-brand-100 bg-brand-50 px-3 py-2">
                {attachments.map((item, index) => (
                  <div
                    key={`${item.name}-${index}`}
                    className="group relative shrink-0"
                    title={`${item.name}（${
                      item.compressed
                        ? `已压缩 ${formatBytes(item.originalBytes)} → ${formatBytes(item.bytes)}`
                        : formatBytes(item.bytes)
                    }）`}
                  >
                    <img
                      src={item.previewUrl}
                      alt={`待发送图片 ${index + 1}`}
                      className="h-14 w-14 rounded-md border border-brand-200 object-cover"
                    />
                    <button
                      onClick={() =>
                        setAttachments((prev) => prev.filter((_x, i) => i !== index))
                      }
                      title="移除这张"
                      aria-label={`移除第 ${index + 1} 张图片`}
                      className="absolute -right-1.5 -top-1.5 rounded-full border border-brand-200 bg-white p-0.5 text-ink-400 shadow-sm transition-colors hover:text-danger"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
                {preparing && (
                  <span className="flex items-center gap-1.5 text-[12px] text-ink-500">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                    正在处理图片…
                  </span>
                )}
                {!preparing && (
                  <span className="ml-auto text-[11px] text-ink-400">
                    共 {attachments.length}/{MAX_ATTACHMENTS} 张｜将由智谱视觉模型逐张解析
                  </span>
                )}
              </div>
            )}
            <div
              className="flex items-end gap-3"
              onPaste={handlePaste}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {/* 隐藏的 file input：由回形针按钮触发（支持多选） */}
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/png,image/jpeg,image/webp,image/bmp,application/pdf"
                className="hidden"
                onChange={(e) => void pickImage(e.target.files)}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={streaming || preparing}
                title={`上传图片（招标公告截图、证照、报价单等），最多 ${MAX_ATTACHMENTS} 张；也支持直接粘贴或拖入`}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 text-ink-500 transition-colors hover:border-brand-300 hover:bg-brand-50 hover:text-brand-600 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {preparing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Paperclip className="h-4 w-4" />
                )}
              </button>
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
                placeholder={
                  attachments.length
                    ? '可补充你的问题（留空则直接解析图片），Enter 发送'
                    : '请输入您的问题，Enter 发送，Shift+Enter 换行；可粘贴/拖入图片或 PDF，也可点左侧回形针'
                }
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
            {/* 拖拽提示：文件悬停在输入区上方时出现 */}
            {dragging && (
              <div className="pointer-events-none mt-2 rounded-lg border border-dashed border-brand-400 bg-brand-50 px-3 py-2 text-center text-[12px] text-brand-700">
                松开即可添加图片或 PDF（最多 {MAX_ATTACHMENTS} 张，PDF 取前 {MAX_PDF_PAGES} 页）
              </div>
            )}
          </div>
        </div>
        {/* 右侧栏：会话 + 推荐问题

            ★ 移动端不再固定 300px
            --------------------
            窄屏改成"占满宽度、放在对话区下方"。`shrink-0` 只在宽屏保留——
            它在窄屏正是把对话区挤成一条的原因。 */}
        <aside className="flex w-full flex-col gap-3 lg:w-[300px] lg:shrink-0 lg:gap-4 lg:overflow-y-auto">
          <Card
            title="历史会话"
            subtitle={searching ? `「${searchKeyword.trim()}」命中 ${searchHits.length} 条` : '点击切换，可重命名或删除'}
          >
            {/* 搜索框：跨会话找到"那次问香港法规的对话" */}
            <div className="relative mb-2">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-300" />
              <input
                value={searchKeyword}
                onChange={(e) => setSearchKeyword(e.target.value)}
                placeholder="搜索历史对话（标题 + 内容）"
                className="h-8 w-full rounded-lg border border-slate-200 pl-7 pr-7 text-xs outline-none focus:border-brand-500"
              />
              {searchKeyword && (
                <button
                  onClick={() => setSearchKeyword('')}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-ink-400 hover:text-danger"
                  title="清空"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>

            <div className="space-y-1">
              {searching ? (
                <>
                  {searchLoading && (
                    <div className="flex items-center gap-1.5 py-2 text-xs text-ink-400">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      搜索中…
                    </div>
                  )}
                  {!searchLoading && searchHits.length === 0 && (
                    <div className="py-2 text-center text-xs text-ink-400">没有匹配的对话</div>
                  )}
                  {searchHits.map((s) => (
                    <div
                      key={s.id}
                      className={`group rounded-lg border px-2 py-1.5 transition-colors ${
                        s.session_id === sessionIdRef.current
                          ? 'border-brand-200 bg-brand-50'
                          : 'border-slate-200 hover:border-brand-200 hover:bg-brand-50'
                      }`}
                    >
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => void switchSession(s.session_id)}
                          className="min-w-0 flex-1 truncate text-left text-xs text-ink-700"
                          title={s.title}
                        >
                          {s.title || '（未命名会话）'}
                        </button>
                        <Badge
                          color={s.matched_in === 'title' ? 'brand' : 'neutral'}
                        >
                          {s.matched_in === 'title'
                            ? '标题命中'
                            : s.matched_in === 'both'
                              ? '标题+内容'
                              : '内容命中'}
                        </Badge>
                        <button
                          onClick={() => void removeSession(s.session_id)}
                          className="hidden shrink-0 rounded p-0.5 text-ink-400 hover:text-danger group-hover:block"
                          title="删除会话"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                      {/* 命中片段：不显示的话用户看不出"为什么这条被搜出来" */}
                      {s.snippet && (
                        <div className="mt-0.5 line-clamp-2 text-[10px] leading-4 text-ink-500">
                          {s.snippet}
                        </div>
                      )}
                      <div className="mt-0.5 flex items-center gap-2 text-[10px] text-ink-400">
                        <span>{s.message_count} 条消息</span>
                        {s.relevance !== null && <span>相关度 {s.relevance.toFixed(3)}</span>}
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                <>
                  {sessions.length === 0 && (
                    <div className="py-2 text-center text-xs text-ink-400">暂无会话</div>
                  )}
                  {sessions.map((s) => (
                    <div
                      key={s.id}
                      className={`group rounded-lg border px-2 py-1.5 transition-colors ${
                        s.session_id === sessionIdRef.current
                          ? 'border-brand-200 bg-brand-50'
                          : 'border-slate-200 hover:border-brand-200 hover:bg-brand-50'
                      }`}
                    >
                      {renamingId === s.id ? (
                        <input
                          autoFocus
                          defaultValue={s.title}
                          onBlur={(e) => void commitRename(s.session_id, e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter')
                              void commitRename(s.session_id, e.currentTarget.value)
                            if (e.key === 'Escape') setRenamingId(null)
                          }}
                          className="h-6 w-full rounded border border-brand-300 px-1 text-xs outline-none"
                        />
                      ) : (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => void switchSession(s.session_id)}
                            className="min-w-0 flex-1 truncate text-left text-xs text-ink-700"
                            title={s.title}
                          >
                            {s.title || '（未命名会话）'}
                          </button>
                          {/* 空会话（点了新建但没提问）标灰，避免列表里全是空壳 */}
                          {s.message_count === 0 && <Badge color="neutral">空</Badge>}
                          <button
                            onClick={() => setRenamingId(s.id)}
                            className="hidden shrink-0 rounded p-0.5 text-ink-400 hover:text-brand-600 group-hover:block"
                            title="重命名"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            onClick={() => void removeSession(s.session_id)}
                            className="hidden shrink-0 rounded p-0.5 text-ink-400 hover:text-danger group-hover:block"
                            title="删除会话"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      )}
                      {s.message_count > 0 && s.preview && (
                        <div className="mt-0.5 line-clamp-2 text-[10px] leading-4 text-ink-400">
                          {s.preview}
                        </div>
                      )}
                    </div>
                  ))}
                </>
              )}
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
