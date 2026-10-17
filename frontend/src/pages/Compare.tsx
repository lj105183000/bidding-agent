/**
 * 招标文件变更追踪页（⑧）。
 *
 * 业务价值（这是本页存在的理由）
 * ----------------------------
 * 招标项目常发澄清/更正公告。投标员最怕的**不是报价高低，而是漏看变更**——
 * 按旧文件做完标书，开标时才发现最高限价改了、截止时间提前了，
 * 结果是**废标**（直接出局，比报价失误严重得多）。
 *
 * 三个设计取向
 * ------------
 * 1. **实质性变更最显眼**：`critical` 变更单独置顶并给出"是否须重做标书"的结论。
 *    用户打开页面第一眼就该看到"要不要重做"。
 * 2. **按危害排序，不按字段顺序**：废标条款 > 资格要求 > 评分标准 > 技术要求。
 *    漏一条废标条款的代价远大于漏一条联系方式。
 * 3. **如实说明局限**：比对基于**抽取出的字段**，附件/图纸/格式文件不在范围内；
 *    补遗模式下"未提及"不等于"被删除"。这些必须写在界面上，不能只写在文档里。
 */
import { useCallback, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowRight,
  Ban,
  CheckCircle2,
  ClipboardList,
  Equal,
  FileText,
  Info,
  ListChecks,
  Loader2,
  Minus,
  Plus,
  RefreshCw,
  Target,
  Upload,
  X,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'
import {
  ApiError,
  compareTenderStream,
  type ChangeDiff,
  type ChangeSeverity,
  type ChangeTrackResult,
  type ChangeType,
  type FieldChange,
  type SectionItem,
} from '../lib/api'

const SEVERITY_META: Record<
  ChangeSeverity,
  { color: 'danger' | 'warn' | 'neutral'; label: string; hint: string }
> = {
  critical: {
    color: 'danger',
    label: '实质性',
    hint: '影响报价或资格，必须据此重做标书相应内容，否则有废标风险',
  },
  major: { color: 'warn', label: '重要', hint: '需评估影响，可能要调整方案或承诺条款' },
  minor: { color: 'neutral', label: '信息性', hint: '更新相关文件中的信息即可，无需重做标书' },
}

const CHANGE_TYPE_META: Record<ChangeType, { icon: typeof Plus; label: string; color: string }> = {
  added: { icon: Plus, label: '新增', color: 'text-emerald-600' },
  removed: { icon: Minus, label: '删除', color: 'text-danger' },
  modified: { icon: RefreshCw, label: '修改', color: 'text-brand-600' },
}

/** 章节的展示优先级：按"漏看代价"排序，而不是 schema 顺序 */
const SECTION_ORDER = [
  'disqualification',
  'qualifications',
  'scoring_items',
  'technical_requirements',
  'required_documents',
]
const SECTION_LABEL: Record<string, string> = {
  disqualification: '废标条款',
  qualifications: '资格要求',
  scoring_items: '评分标准',
  technical_requirements: '技术要求',
  required_documents: '需提交材料',
}

const PHASE_LABEL: Record<string, string> = {
  parse_old: '解析原文件',
  extract_old: '抽取原文件内容',
  parse_new: '解析新文件',
  extract_new: '抽取新文件内容',
  diff: '比对差异',
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 400) return err.message
    if (err.status === 413) return '文件过大，上限 50MB'
    if (err.status === 503) return '变更追踪服务未就绪（检查后端 LLM 配置）'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

function FilePicker({
  label,
  file,
  onPick,
  onClear,
  accent,
}: {
  label: string
  file: File | null
  onPick: (f: File) => void
  onClear: () => void
  accent: string
}) {
  const ref = useRef<HTMLInputElement>(null)
  return (
    <div>
      <input
        ref={ref}
        type="file"
        accept=".pdf,.docx"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onPick(f)
          if (ref.current) ref.current.value = ''
        }}
      />
      <div className="mb-1.5 text-[11px] font-medium text-ink-500">{label}</div>
      {file ? (
        <div className={`flex items-start gap-2 rounded-lg border p-2.5 ${accent}`}>
          <FileText className="mt-0.5 h-4 w-4 shrink-0" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-[12px] font-medium text-ink-900" title={file.name}>
              {file.name}
            </div>
            <div className="text-[10px] text-ink-400">{(file.size / 1024).toFixed(0)} KB</div>
          </div>
          <button onClick={onClear} className="shrink-0 rounded p-0.5 text-ink-400 hover:text-danger">
            <X className="h-3 w-3" />
          </button>
        </div>
      ) : (
        <button
          onClick={() => ref.current?.click()}
          className="flex h-20 w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed border-ink-200 transition-colors hover:border-brand-300 hover:bg-brand-50/40"
        >
          <Upload className="h-4 w-4 text-ink-300" />
          <span className="text-[11px] text-ink-500">选择 PDF / Word</span>
        </button>
      )}
    </div>
  )
}

function ValueChip({ value, tone }: { value: string | number | null; tone: 'old' | 'new' }) {
  const empty = value === null || value === undefined || value === ''
  const cls =
    tone === 'old'
      ? 'bg-red-50 text-red-700 line-through decoration-red-300'
      : 'bg-emerald-50 text-emerald-700'
  return (
    <span
      className={`inline-block max-w-[220px] truncate rounded px-1.5 py-0.5 align-middle text-[12px] ${
        empty ? 'text-ink-300' : cls
      }`}
      title={empty ? '(空)' : String(value)}
    >
      {empty ? '(空)' : String(value)}
    </span>
  )
}

function ItemText({ item }: { item: SectionItem }) {
  const text = String(item.text ?? '')
  const extra: string[] = []
  if (item.score !== undefined && item.score !== null && item.score !== '') extra.push(`${item.score} 分`)
  if (item.category) extra.push(String(item.category))
  if (item.is_mandatory === true) extra.push('硬性门槛')
  return (
    <span className="min-w-0 flex-1 text-[12px] leading-5 text-ink-800">
      {text}
      {extra.length > 0 && <span className="ml-1.5 text-[10px] text-ink-400">（{extra.join(' / ')}）</span>}
    </span>
  )
}

function DiffView({ diff }: { diff: ChangeDiff }) {
  const { summary } = diff
  const changedSections = SECTION_ORDER.filter((k) => {
    const s = diff.section_changes[k]
    return s && (s.added.length > 0 || s.removed.length > 0)
  })

  return (
    <>
      {/* 结论置顶：用户第一眼要知道"要不要重做" */}
      <div
        className={`mb-4 rounded-xl border-l-4 p-4 ${
          summary.critical > 0 ? 'border-danger bg-red-50' : 'border-emerald-400 bg-emerald-50'
        }`}
      >
        <div className="flex items-start gap-2.5">
          {summary.critical > 0 ? (
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
          ) : (
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
          )}
          <div className="min-w-0">
            <div className={`text-[15px] font-bold ${summary.critical > 0 ? 'text-danger' : 'text-emerald-700'}`}>
              {summary.critical > 0
                ? `检出 ${summary.critical} 项实质性变更，须重做标书相应内容`
                : summary.total > 0
                  ? `检出 ${summary.total} 项变更，无实质性变更`
                  : '未检出变更'}
            </div>
            <div className="mt-1 text-[12px] leading-5 text-ink-600">
              {summary.total === 0
                ? '两份文件在字段与章节上一致（附件、图纸、格式文件等非文本内容不在比对范围内）'
                : summary.critical > 0
                  ? SEVERITY_META.critical.hint
                  : '可只更新相关信息，无需重做标书'}
            </div>
          </div>
        </div>
      </div>

      {/* 分档统计 */}
      <div className="mb-4 grid grid-cols-4 gap-3">
        {(
          [
            ['total', '变更总数', summary.total, 'text-ink-900'],
            ['critical', '实质性', summary.critical, 'text-danger'],
            ['major', '重要', summary.major, 'text-warn'],
            ['minor', '信息性', summary.minor, 'text-ink-500'],
          ] as const
        ).map(([key, label, count, cls]) => (
          <div key={key} className="rounded-xl border border-slate-200 bg-white p-3">
            <div className="text-[11px] text-ink-500">{label}</div>
            <div className={`mt-0.5 text-xl font-bold tabular-nums ${cls}`}>{count}</div>
          </div>
        ))}
      </div>

      {/* 字段变更 */}
      {diff.field_changes.length > 0 && (
        <Card title={`字段变更（${diff.field_changes.length} 项）`} subtitle="项目信息层面的变化">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                <th className="w-16 py-2">程度</th>
                <th className="w-16 py-2">类型</th>
                <th className="w-28 py-2">字段</th>
                <th className="py-2">原值</th>
                <th className="w-6 py-2"></th>
                <th className="py-2">新值</th>
              </tr>
            </thead>
            <tbody>
              {diff.field_changes.map((c: FieldChange) => {
                const sev = SEVERITY_META[c.severity]
                const ct = CHANGE_TYPE_META[c.change_type]
                const Icon = ct.icon
                return (
                  <tr key={c.key} className="border-b border-slate-100 last:border-0">
                    <td className="py-1.5">
                      <Badge color={sev.color}>{sev.label}</Badge>
                    </td>
                    <td className="py-1.5">
                      <span className={`flex items-center gap-1 text-[11px] ${ct.color}`}>
                        <Icon className="h-3 w-3" />
                        {ct.label}
                      </span>
                    </td>
                    <td className="py-1.5 text-[12px] text-ink-600">{c.label}</td>
                    <td className="py-1.5">
                      <ValueChip value={c.old} tone="old" />
                    </td>
                    <td className="py-1.5 text-center text-ink-300">
                      <ArrowRight className="inline h-3 w-3" />
                    </td>
                    <td className="py-1.5">
                      <ValueChip value={c.new} tone="new" />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Card>
      )}

      {/* 章节变更：按危害排序 */}
      {changedSections.length > 0 && (
        <Card
          className="mt-4"
          title={`章节变更（${changedSections.length} 个章节有变化）`}
          subtitle="按漏看代价排序：废标条款 → 资格要求 → 评分标准 → 技术要求 → 需提交材料"
        >
          <div className="space-y-3">
            {changedSections.map((key) => {
              const s = diff.section_changes[key]
              const isDisq = key === 'disqualification'
              return (
                <div
                  key={key}
                  className={`rounded-lg border p-3 ${isDisq ? 'border-red-200 bg-red-50/50' : 'border-slate-200'}`}
                >
                  <div className="mb-2 flex items-center gap-2">
                    {isDisq ? (
                      <Ban className="h-3.5 w-3.5 text-danger" />
                    ) : (
                      <ListChecks className="h-3.5 w-3.5 text-ink-400" />
                    )}
                    <span className="text-[13px] font-semibold text-ink-900">
                      {SECTION_LABEL[key] ?? key}
                    </span>
                    {s.added.length > 0 && <Badge color="ok">新增 {s.added.length}</Badge>}
                    {s.removed.length > 0 && <Badge color="danger">删除 {s.removed.length}</Badge>}
                    {s.unchanged_count > 0 && (
                      <span className="flex items-center gap-1 text-[10px] text-ink-400">
                        <Equal className="h-2.5 w-2.5" />
                        未变 {s.unchanged_count}
                      </span>
                    )}
                  </div>
                  <ul className="space-y-1">
                    {s.added.map((it, i) => (
                      <li key={`a${i}`} className="flex items-start gap-1.5">
                        <Plus className="mt-1 h-3 w-3 shrink-0 text-emerald-600" />
                        <ItemText item={it} />
                      </li>
                    ))}
                    {s.removed.map((it, i) => (
                      <li key={`r${i}`} className="flex items-start gap-1.5">
                        <Minus className="mt-1 h-3 w-3 shrink-0 text-danger" />
                        <span className="line-through decoration-red-300">
                          <ItemText item={it} />
                        </span>
                      </li>
                    ))}
                  </ul>
                  {isDisq && s.added.length > 0 && (
                    <div className="mt-2 text-[11px] leading-5 text-ink-700">
                      新增废标条款意味着**新增了一道否决条件**，请确认标书是否满足。
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* 影响提示 */}
      {diff.impact_notes.length > 0 && (
        <Card className="mt-4" title="影响提示" subtitle="每条变更对投标工作的实际影响">
          <ul className="space-y-2">
            {diff.impact_notes.map((n, i) => (
              <li key={i} className="flex items-start gap-2">
                <Target className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-600" />
                <span className="text-[12px] leading-5 text-ink-700">{n}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </>
  )
}

export default function Compare() {
  const toast = useToast()
  const abortRef = useRef<AbortController | null>(null)

  const [oldFile, setOldFile] = useState<File | null>(null)
  const [newFile, setNewFile] = useState<File | null>(null)
  const [patchMode, setPatchMode] = useState<'auto' | 'full' | 'patch'>('auto')
  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState('')
  const [result, setResult] = useState<ChangeTrackResult | null>(null)
  const [error, setError] = useState('')

  const run = useCallback(async () => {
    if (!oldFile || !newFile || running) return
    setRunning(true)
    setResult(null)
    setError('')
    setPhase('准备中…')

    const controller = new AbortController()
    abortRef.current = controller
    const newIsPatch = patchMode === 'auto' ? undefined : patchMode === 'patch'

    try {
      await compareTenderStream(
        oldFile,
        newFile,
        {
          onPhase: (p, payload) => {
            const label = PHASE_LABEL[p] ?? p
            const extra =
              typeof payload.done === 'number' && typeof payload.total === 'number'
                ? `（${payload.done}/${payload.total}）`
                : ''
            setPhase(`${label}${extra}…`)
          },
          onDone: (r) => {
            setResult(r)
            const s = r.diff?.summary
            if (s) {
              toast.show(
                s.critical > 0
                  ? `检出 ${s.critical} 项实质性变更，须重做标书`
                  : `共 ${s.total} 项变更，无实质性变更`,
                s.critical > 0 ? 'warning' : 'success',
              )
            }
          },
          onError: (msg) => {
            setError(msg)
            toast.show(msg, 'error')
          },
        },
        newIsPatch,
        controller.signal,
      )
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        const msg = describeError(err)
        setError(msg)
        toast.show(msg, 'error')
      }
    } finally {
      setRunning(false)
      setPhase('')
      abortRef.current = null
    }
  }, [oldFile, newFile, running, patchMode, toast])

  return (
    <div>
      <PageHeader
        title="招标文件变更追踪"
        description="上传原招标文件与澄清/更正文件，比对出所有变更并标出实质性变更"
      />

      {/* ★ 移动端纵向堆叠（实测缺陷修复）
          原来 `flex items-start` + 上传卡 `w-[340px] shrink-0`：
          `shrink-0` 让它**永不收缩**，375px 屏上 340px 卡 + 16px 间距后
          只剩约 19px 给右侧结果区 —— 内容被推出屏外、用户看不到（审计实测）。
          `lg:shrink-0` 只在宽屏保留"上传卡固定宽度"的设计意图。 */}
      <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-start lg:gap-4">
        <Card
          className="w-full lg:w-[340px] lg:shrink-0"
          title="选择两份文件"
          subtitle="原文件 vs 澄清/更正后的文件"
        >
          <div className="space-y-3">
            <FilePicker
              label="① 原招标文件"
              file={oldFile}
              onPick={(f) => {
                setOldFile(f)
                setResult(null)
                setError('')
              }}
              onClear={() => setOldFile(null)}
              accent="border-slate-200 bg-ink-50/60 text-ink-500"
            />
            <FilePicker
              label="② 澄清 / 更正文件"
              file={newFile}
              onPick={(f) => {
                setNewFile(f)
                setResult(null)
                setError('')
              }}
              onClear={() => setNewFile(null)}
              accent="border-brand-200 bg-brand-50/60 text-brand-600"
            />

            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">比对模式</div>
              <select
                value={patchMode}
                onChange={(e) => setPatchMode(e.target.value as typeof patchMode)}
                className="h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              >
                <option value="auto">自动判断（推荐）</option>
                <option value="full">完整文件比对</option>
                <option value="patch">补遗公告（只列改动）</option>
              </select>
              <div className="mt-1 text-[10px] leading-4 text-ink-400">
                {patchMode === 'patch'
                  ? '补遗模式下「未提及」不等于「被删除」，不会把没变的条目报成删除'
                  : patchMode === 'full'
                    ? '严格逐字段比对，未出现即视为被删除'
                    : '按信息量自动判断是否为补遗公告，判断理由会在结果里说明'}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {running ? (
                <Button variant="secondary" onClick={() => abortRef.current?.abort()} icon={<X className="h-3.5 w-3.5" />}>
                  停止
                </Button>
              ) : (
                <Button
                  onClick={() => void run()}
                  disabled={!oldFile || !newFile}
                  icon={<RefreshCw className="h-3.5 w-3.5" />}
                >
                  开始比对
                </Button>
              )}
            </div>

            {running && (
              <div className="rounded-lg border border-brand-100 bg-brand-50 p-3">
                <div className="flex items-center gap-2 text-[12px] text-ink-700">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                  {phase || '处理中…'}
                </div>
                <div className="mt-1.5 text-[10px] leading-4 text-ink-500">
                  要抽取两份文件，通常需要 20~60 秒
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-lg border-l-2 border-danger bg-red-50 px-3 py-2">
                <div className="flex items-start gap-2 text-[12px] font-semibold text-danger">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span className="min-w-0 break-words">{error}</span>
                </div>
              </div>
            )}

            {/* 局限说明：必须写在界面上，不能只写在文档里 */}
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
              <div className="flex items-start gap-1.5">
                <Info className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                <div className="text-[10px] leading-4 text-ink-600">
                  比对基于**抽取出的字段与章节**。附件、图纸、格式文件等非文本内容不在范围内；
                  抽取本身也可能遗漏，<b>重要变更请对照原文复核</b>。
                </div>
              </div>
            </div>
          </div>
        </Card>

        <div className="min-w-0 flex-1">
          {!result && !running && (
            <Card>
              <EmptyState
                icon={<ClipboardList className="h-6 w-6" />}
                title="选择两份文件后开始比对"
                description="系统会分别抽取两份文件，逐字段与逐章节比对，并标出哪些是必须重做标书的实质性变更"
              />
            </Card>
          )}
          {running && !result && (
            <Card>
              <div className="flex flex-col items-center justify-center gap-3 py-12">
                <Loader2 className="h-7 w-7 animate-spin text-brand-600" />
                <div className="text-[13px] text-ink-600">{phase || '处理中…'}</div>
                <div className="text-[11px] text-ink-400">
                  正在解析与抽取两份文件，请勿关闭页面
                </div>
              </div>
            </Card>
          )}
          {result?.diff && <DiffView diff={result.diff} />}
          {result?.meta?.warnings && result.meta.warnings.length > 0 && (
            <Card className="mt-4" title="比对提示">
              <ul className="space-y-1">
                {result.meta.warnings.map((w, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-[12px] leading-5 text-ink-600">
                    <Info className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                    <span>{w}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
