/**
 * 「参考来源」面板：本地知识库 + 联网搜索，**合并成一个列表**，每条带相关性评分与 URL。
 *
 * 为什么合成一个面板
 * ----------------
 * 改造前分成"本地知识库"与"联网搜索"两段，各排各的。问题在于：
 *
 *   1. **无法横向比较**：用户看到本地 0.71 与联网 0.65 时，不知道哪个更值得信
 *   2. **两段各自很短**：本地 3 条 + 联网 5 条本可以混排成 8 条按相关度排列，
 *      分段后需要来回对照
 *   3. **看不到全貌**：回答同时引用了本地与联网内容时，用户想知道
 *      "这句话的依据在哪"——混排并按分数降序才直观
 *
 * 故现在是一个统一列表，按相关度降序，每条明确标注**来源类型**。
 *
 * 为什么 URL 必须显示（且可点）
 * --------------------------
 * 联网结果的**唯一核对途径就是打开原网页**。只给标题等于让用户相信一个
 * 无法验证的结论——尤其招投标场景里法规与公告都要以原文为准。
 * 故 URL 用可点链接 + 域名单独展示（便于判断站点可信度）。
 *
 * ★ 分数口径必须标出来
 * ------------------
 *   relevance（本地知识库）  与查询的**稠密余弦相似度**，可与阈值比较
 *   score（联网搜索）        引擎自带分（Tavily）或本地补算的余弦（智谱）
 *
 * 外部分数没有绝对基准，**不可与本地 relevance 直接横比**——
 * "这条联网结果 0.65 比本地 0.71 差"是错误结论。
 */
import { useState } from 'react'
import { ThresholdBar } from './MicroBars'
import {
  RELEVANCE_NEUTRAL_COLOR,
  RELEVANCE_OK_COLOR,
  RELEVANCE_WARN_COLOR,
} from '../lib/chartStyles'

import {
  AlertTriangle,
  BookOpen,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Globe,
  Info,
  Link2,
} from 'lucide-react'
import Badge from './Badge'

/** 联网来源。两种字段形态都支持：
 *  - Agent 的 `web_sources`：`{question(标题), answer(正文), score, url}`
 *  - `/api/web/search`：`{title, content, score, published, source, score_kind}`
 */
export interface WebSourceFields {
  title?: string
  url?: string
  content?: string
  /** Agent 契约里承载**标题** */
  question?: string
  /** Agent 契约里承载**正文** */
  answer?: string
  /** 相关度 0~1（可能为空） */
  score?: number | null
  /** 发布时间（智谱提供） */
  published?: string
  /** 数据源 */
  source?: string
  /** 分数口径：engine 引擎自带 / cosine 本地补算 */
  score_kind?: string
}

/** 本地知识库来源：`{question, answer, relevance}`（`relevance` 为稠密余弦相似度） */
export interface LocalSource {
  question?: string
  answer?: string
  score?: number
  /** 与查询的稠密余弦相似度（后端 `attach_relevance` 计算） */
  relevance?: number
}

interface Props {
  localSources?: LocalSource[]
  webSources?: WebSourceFields[]
  /** 自动兜底的判定说明（本地为何不够用） */
  fallbackNote?: string
  /** 相关度阈值，用于标出"低于阈值"的来源 */
  threshold?: number
  defaultOpen?: boolean
}

/** 归一后的来源条目（混排用同一个形状） */
interface UnifiedSource {
  kind: 'local' | 'web'
  title: string
  body: string
  url: string
  /** 展示用分数（本地取 relevance，联网取 score） */
  score: number | null
  /** 分数口径说明 */
  scoreLabel: string
  /** 引擎分与余弦分口径不同，不可横比 */
  comparable: boolean
  published: string
  provider: string
}

/** 从 URL 提取域名（判断站点可信度的第一依据） */
function domainOf(url: string): string {
  try {
    return new URL(url).hostname
  } catch {
    return ''
  }
}

/** 分数 → 颜色。阈值以上的分级依据实测：0.6+ 稳定相关，0.5 附近勉强相关。 */
function scoreTone(score: number | null, threshold: number): 'ok' | 'warn' | 'neutral' {
  if (score === null) return 'neutral'
  if (score >= 0.65) return 'ok'
  if (score >= threshold) return 'warn'
  return 'neutral'
}

function ScoreBar({ value, tone, threshold }: {
  value: number
  tone: 'ok' | 'warn' | 'neutral'
  threshold?: number
}) {
  // ★ 从 div+width% 改为带**阈值标记**的条
  //
  // 三处变化与理由：
  //   ① 加宽（48 → 84px）：原宽度下 0.4 与 0.5 的差别只有 5px，读不出来
  //   ② 加**阈值竖线**：`min_relevance` 之下会被后端过滤掉，
  //      故"这条够不够格"是关键判断 —— 原来只能靠颜色猜，现在能直接看
  //   ③ 颜色改从 `chartStyles` 取：与其它图表的"绿=好/橙=警告"同一套值。
  //      原来这里写 Tailwind class、chartStyles 写十六进制，两处各改各的
  //      会出现"胶囊文字是绿的、条是灰的"这种矛盾显示。
  const color =
    tone === 'ok'
      ? RELEVANCE_OK_COLOR
      : tone === 'warn'
        ? RELEVANCE_WARN_COLOR
        : RELEVANCE_NEUTRAL_COLOR
  return (
    <ThresholdBar
      value={value}
      threshold={threshold ?? 0}
      width={84}
      height={6}
      color={color}
      title={
        threshold !== undefined
          ? `相关度 ${value.toFixed(3)}（可用线 ${threshold.toFixed(2)}）`
          : `相关度 ${value.toFixed(3)}`
      }
    />
  )
}

function normalize(
  localSources: LocalSource[],
  webSources: WebSourceFields[],
): UnifiedSource[] {
  const out: UnifiedSource[] = []

  for (const s of localSources) {
    out.push({
      kind: 'local',
      title: s.question || '(无标题)',
      body: s.answer || '',
      url: '',
      score: typeof s.relevance === 'number' ? s.relevance : null,
      scoreLabel: '本地匹配度（余弦相似度）',
      comparable: true,
      published: '',
      provider: '本地知识库',
    })
  }

  for (const s of webSources) {
    // 字段兼容：Agent 契约用 question/answer 承载标题/正文
    const provider =
      s.source === 'zhipu' ? '智谱' : s.source === 'tavily' ? 'Tavily' : s.source || '联网'
    out.push({
      kind: 'web',
      title: s.title || s.question || s.url || '(无标题)',
      body: s.content || s.answer || '',
      url: s.url || '',
      score: typeof s.score === 'number' ? s.score : null,
      scoreLabel:
        s.score_kind === 'engine' ? '外部引擎评分' : s.score_kind === 'cosine'
          ? '本地补算（余弦相似度）'
          : '相关度',
      // 只有本地补算的余弦分与本地知识库同量纲；引擎分口径不同
      comparable: s.score_kind !== 'engine',
      published: s.published || '',
      provider,
    })
  }

  // 按分数降序；无分数的排最后（不等于 0 分，排前面会误导）
  out.sort((a, b) => {
    if (a.score === null && b.score === null) return 0
    if (a.score === null) return 1
    if (b.score === null) return -1
    return b.score - a.score
  })
  return out
}

function SourceRow({ s, threshold }: { s: UnifiedSource; threshold: number }) {
  const [open, setOpen] = useState(false)
  const tone = scoreTone(s.score, threshold)
  const Icon = s.kind === 'local' ? BookOpen : Globe
  const iconColor = s.kind === 'local' ? 'text-emerald-600' : 'text-warn'
  const domain = s.url ? domainOf(s.url) : ''

  return (
    <div
      className={`rounded-lg border px-3 py-2 ${
        s.kind === 'local' ? 'border-slate-200' : 'border-amber-100 bg-amber-50/40'
      }`}
    >
      <div className="flex items-start gap-2">
        <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${iconColor}`} />

        <div className="min-w-0 flex-1">
          {/* 标题：联网结果可点开原文 */}
          {s.url ? (
            <a
              href={s.url}
              target="_blank"
              rel="noreferrer noopener"
              className="group inline-flex items-start gap-1 text-[12px] font-medium leading-5 text-brand-700 hover:underline"
              title={s.url}
            >
              <span className="min-w-0">{s.title}</span>
              <ExternalLink className="mt-0.5 h-3 w-3 shrink-0 opacity-60" />
            </a>
          ) : (
            <div className="text-[12px] font-medium leading-5 text-ink-800">{s.title}</div>
          )}

          {/* URL 单独一行：域名是判断站点可信度的第一依据 */}
          {s.url && (
            <div className="mt-0.5 flex items-start gap-1">
              <Link2 className="mt-0.5 h-2.5 w-2.5 shrink-0 text-ink-300" />
              <a
                href={s.url}
                target="_blank"
                rel="noreferrer noopener"
                className="min-w-0 break-all text-[10px] leading-4 text-ink-400 hover:text-brand-600 hover:underline"
                title={s.url}
              >
                {domain && <span className="font-medium text-ink-500">{domain}</span>}
                {domain && <span className="mx-1">·</span>}
                {s.url}
              </a>
            </div>
          )}

          {/* 来源类型 + 分数 */}
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <Badge color={s.kind === 'local' ? 'ok' : 'warn'}>
              {s.kind === 'local' ? '本地知识库' : `联网 · ${s.provider}`}
            </Badge>
            {s.score !== null ? (
              <>
                <Badge color={tone}>{s.score.toFixed(2)}</Badge>
                {/* 传 threshold：条上会画一条**可用线**，
                    让"这条够不够格"一眼可见，而不必靠颜色猜。
                    本地来源才有 relevance 语义；联网分数没有绝对基准
                    （见文件头注释），故那里的阈值线只是参考位置。 */}
                <ScoreBar value={s.score} tone={tone} threshold={threshold} />
                <span className="text-[10px] text-ink-400">{s.scoreLabel}</span>
              </>
            ) : (
              <Badge color="neutral">无分数</Badge>
            )}
            {s.published && (
              <span className="text-[10px] text-ink-500">· 发布 {s.published.slice(0, 10)}</span>
            )}
          </div>

          {/* 正文可折叠 */}
          {s.body && (
            <>
              <button
                onClick={() => setOpen((v) => !v)}
                className="mt-1.5 flex items-center gap-0.5 text-[11px] text-brand-600 hover:text-brand-700"
              >
                {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                {open ? '收起原文' : '查看原文'}
              </button>
              {open && (
                <div className="mt-1 whitespace-pre-wrap break-words rounded bg-white/70 px-2 py-1.5 text-[11px] leading-5 text-ink-600">
                  {s.body}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default function SourceList({
  localSources = [],
  webSources = [],
  fallbackNote,
  threshold = 0.5,
  defaultOpen = false,
}: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const items = normalize(localSources, webSources)
  if (items.length === 0 && !fallbackNote) return null

  const localCount = items.filter((i) => i.kind === 'local').length
  const webCount = items.length - localCount
  const withUrl = items.filter((i) => i.url).length

  return (
    <div className="mt-3 border-t border-slate-100 pt-2">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 text-left">
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-ink-400" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-400" />
        )}
        <span className="text-[11px] font-medium text-ink-600">
          {open ? '收起参考来源' : '查看参考来源'}
        </span>
        <Badge color="neutral">共 {items.length} 条</Badge>
        {localCount > 0 && <Badge color="ok">本地 {localCount}</Badge>}
        {webCount > 0 && (
          <Badge color="warn">
            <Globe className="h-3 w-3" />
            联网 {webCount}
          </Badge>
        )}
        {withUrl > 0 && <span className="text-[10px] text-ink-400">{withUrl} 条含链接</span>}
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {/* 自动兜底说明：解释"为什么这次联网了" */}
          {fallbackNote && (
            <div className="flex items-start gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-2">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
              <div className="text-[11px] leading-5 text-ink-700">
                <b>本次自动联网：</b>
                {fallbackNote}
              </div>
            </div>
          )}

          <div className="text-[10px] text-ink-400">
            本地知识库与联网结果**合并按相关度降序**排列；联网结果可点标题或链接打开原网页核对。
          </div>

          <div className="space-y-1.5">
            {items.map((s, i) => (
              <SourceRow key={`${s.kind}-${i}`} s={s} threshold={threshold} />
            ))}
          </div>

          {/* 口径说明：不写清用户会拿两种分数直接横比 */}
          <div className="flex items-start gap-1.5 rounded-lg bg-ink-50 px-2.5 py-1.5">
            <Info className="mt-0.5 h-3 w-3 shrink-0 text-ink-400" />
            <div className="text-[10px] leading-4 text-ink-500">
              本地「匹配度」是与问题的余弦相似度，阈值 {threshold.toFixed(2)}
              （低于它的本地结果采信度不足）。联网结果若标注
              <b>外部引擎评分</b>，其口径与余弦不同，<b>不可与本地匹配度直接横比</b>；
              标注为「本地补算」的则同量纲、可比。
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
