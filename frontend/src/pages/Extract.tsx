/**
 * 标讯智能提取页。
 *
 * 数据来源：`POST /api/extract`（文本走 LLM 抽取，图片走视觉模型抽取，同一份 schema）。
 *
 * 与早期版本的差别：早期用 `data/extractSamples.ts` 的 expected 字段配合
 * `setTimeout` 假装"AI 提取"，结果是预先写死的。本版接真实链路，
 * **字段清单也从后端拉**（`/api/extract/schema`），所以后端加字段前端自动跟上。
 *
 * 三条设计取向：
 *   1. **如实展示缺失**：抽不到的字段显示"—"，不填占位值也不隐藏列；
 *   2. **展示可信度**：完整度、被清洗的可疑值、是否截断，都摆在结果上方；
 *   3. **入库是显式动作**：默认只看不入库，用户确认后才写 `bidding_records`。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Database,
  Download,
  FileSearch,
  Image as ImageIcon,
  Info,
  Loader2,
  Play,
  Save,
  Trash2,
  Upload,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import {
  ApiError,
  extractNotice,
  fetchExtractSchema,
  type BidFieldDef,
  type ExtractResult,
} from '../lib/api'
import { EXTRACT_SAMPLES } from '../data/extractSamples'
import { formatMoney, shortDate } from '../lib/format'
import { ImagePrepareError, formatBytes, prepareImage, type PreparedImage } from '../lib/image'

const TAB_NAMES = ['示例一', '示例二', '示例三']

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 413) return '图片过大，请压缩后重试'
    if (err.status === 400) return err.message
    if (err.status === 429) return '请求过于频繁，请稍后再试'
    if (err.status === 503) return '提取服务未就绪（检查后端 LLM / ZHIPU_API_KEY 配置）'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

function categoryColor(c: string): 'brand' | 'ok' | 'warn' | 'neutral' {
  if (c === '货物') return 'brand'
  if (c === '服务') return 'ok'
  if (c === '工程') return 'warn'
  return 'neutral'
}

function sourceColor(s: string): 'brand' | 'ok' | 'warn' | 'neutral' {
  if (s === '政府采购') return 'brand'
  if (s === '企业采购') return 'ok'
  if (s === '招标采购') return 'warn'
  return 'neutral'
}

/** 单字段渲染：按类型走不同展示（金额千分位、日期短格式、枚举徽章） */
function FieldValue({ field, value }: { field: BidFieldDef; value: string | number | null }) {
  if (value === null || value === undefined || value === '') {
    return <span className="text-ink-300">—</span>
  }
  if (field.key === 'category') return <Badge color={categoryColor(String(value))}>{value}</Badge>
  if (field.key === 'source') return <Badge color={sourceColor(String(value))}>{value}</Badge>
  if (field.type === 'money') return <span className="font-medium tabular-nums">{formatMoney(Number(value))}</span>
  if (field.type === 'datetime') return <span className="tabular-nums">{shortDate(String(value))}</span>
  return <span className="break-words">{String(value)}</span>
}

export default function Extract() {
  const toast = useToast()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [fields, setFields] = useState<BidFieldDef[]>([])
  const [schemaReady, setSchemaReady] = useState(false)

  const [mode, setMode] = useState<'text' | 'image'>('text')
  const [active, setActive] = useState(0)
  const [text, setText] = useState(EXTRACT_SAMPLES[0].rawText)
  const [image, setImage] = useState<PreparedImage | null>(null)
  const [hint, setHint] = useState('')

  const [status, setStatus] = useState<'idle' | 'extracting' | 'done'>('idle')
  const [result, setResult] = useState<ExtractResult | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  // 字段清单从后端拉：后端加字段前端自动跟上
  useEffect(() => {
    fetchExtractSchema()
      .then((res) => {
        setFields(res.fields)
        setSchemaReady(true)
      })
      .catch(() => {
        setSchemaReady(true)
        toast.show('字段清单拉取失败，请确认后端已启动', 'error')
      })
  }, [toast])

  const pickImage = useCallback(
    async (file: File | undefined) => {
      if (!file) return
      try {
        const prepared = await prepareImage(file)
        setImage(prepared)
        setMode('image')
        setResult(null)
        setError('')
        toast.show(
          prepared.compressed
            ? `已加载图片（${formatBytes(prepared.originalBytes)} → ${formatBytes(prepared.bytes)}）`
            : '已加载图片',
          'success',
        )
      } catch (err) {
        toast.show(err instanceof ImagePrepareError ? err.message : '图片处理失败', 'error')
      } finally {
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    },
    [toast],
  )

  const run = useCallback(
    async (save = false) => {
      if (status === 'extracting') return
      if (mode === 'text' && text.trim().length < 10) {
        toast.show('请输入至少 10 个字符的公告文本', 'warning')
        return
      }
      if (mode === 'image' && !image) {
        toast.show('请先选择公告截图', 'warning')
        return
      }

      setStatus('extracting')
      setError('')
      setResult(null)
      if (save) setSaving(true)

      try {
        const res = await extractNotice({
          source: mode,
          text: mode === 'text' ? text : '',
          imageBase64: mode === 'image' ? image!.dataUrl : '',
          hint: mode === 'image' ? hint : '',
          save,
        })
        setResult(res)
        setStatus('done')
        if (save && res.save) {
          if (res.save.saved) {
            toast.show(`已写入数据库（记录 id ${res.save.id}）`, 'success')
          } else {
            toast.show(`未入库：${res.save.reason ?? '未知原因'}`, 'warning')
          }
        } else {
          toast.show(
            `提取完成，字段完整度 ${(res.meta.completeness * 100).toFixed(0)}%`,
            res.meta.completeness >= 0.6 ? 'success' : 'warning',
          )
        }
      } catch (err) {
        const msg = describeError(err)
        setError(msg)
        setStatus('idle')
        toast.show(msg, 'error')
      } finally {
        setSaving(false)
      }
    },
    [mode, text, image, hint, status, toast],
  )

  const copyJson = () => {
    if (!result) return
    const payload = JSON.stringify(result.record, null, 2)
    navigator.clipboard
      .writeText(payload)
      .then(() => toast.show('JSON 已复制到剪贴板', 'success'))
      .catch(() => toast.show('复制失败', 'error'))
  }

  const downloadCsv = () => {
    if (!result) return
    const header = fields.map((f) => f.label).join(',')
    const row = fields
      .map((f) => {
        const v = result.record[f.key]
        const s = v === null || v === undefined ? '' : String(v)
        return `"${s.replace(/"/g, '""')}"`
      })
      .join(',')
    // 加 BOM，Excel 打开中文才不乱码
    const blob = new Blob(['\uFEFF' + header + '\n' + row], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `标讯提取_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
    toast.show('CSV 已下载', 'success')
  }

  const filledCount = result
    ? fields.filter((f) => result.record[f.key] !== null && result.record[f.key] !== '').length
    : 0

  return (
    <div>
      <PageHeader
        title="标讯智能提取"
        description="粘贴公告文本或上传公告截图，自动抽取 17 个结构化字段并可写入业务数据库"
        actions={
          schemaReady && fields.length > 0 ? (
            <Badge color="neutral">
              <FileSearch className="h-3 w-3" />
              {fields.length} 个字段
            </Badge>
          ) : undefined
        }
      />

      <div className="flex items-start gap-4">
        {/* ── 左侧：输入区 ─────────────────────────────────────── */}
        <Card
          className="flex-1"
          title="公告输入"
          subtitle="文本模式走大模型抽取，图片模式走视觉模型，字段完全一致"
          extra={
            <div className="flex rounded-lg border border-slate-200 p-0.5">
              {(['text', 'image'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    mode === m ? 'bg-brand-600 text-white' : 'text-ink-500 hover:bg-ink-50'
                  }`}
                >
                  {m === 'text' ? '粘贴文本' : '上传截图'}
                </button>
              ))}
            </div>
          }
        >
          {mode === 'text' ? (
            <>
              {/* 示例切换 */}
              <div className="mb-2 flex gap-1">
                {EXTRACT_SAMPLES.map((s, i) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setActive(i)
                      setText(s.rawText)
                      setResult(null)
                    }}
                    className={`rounded-md px-2 py-1 text-[11px] transition-colors ${
                      active === i
                        ? 'bg-brand-50 font-medium text-brand-700'
                        : 'text-ink-400 hover:bg-ink-50'
                    }`}
                  >
                    {TAB_NAMES[i] ?? s.id}
                  </button>
                ))}
                <button
                  onClick={() => {
                    setText('')
                    setResult(null)
                  }}
                  className="ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-ink-400 hover:bg-ink-50"
                >
                  <Trash2 className="h-3 w-3" />
                  清空
                </button>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="h-56 w-full resize-none rounded-lg border border-slate-200 p-3 text-[13px] leading-6 text-ink-900 outline-none transition-colors focus:border-brand-500"
                placeholder="粘贴招标公告、中标公告等文本，至少 10 个字符…"
              />
              <div className="mt-1 text-right text-[11px] text-ink-300">{text.length} 字符</div>
            </>
          ) : (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/bmp"
                className="hidden"
                onChange={(e) => void pickImage(e.target.files?.[0])}
              />
              {!image ? (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex h-56 w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-ink-200 transition-colors hover:border-brand-300 hover:bg-brand-50/40"
                >
                  <Upload className="h-7 w-7 text-ink-300" />
                  <div className="text-[13px] font-medium text-ink-700">点击选择公告截图</div>
                  <div className="text-[11px] text-ink-400">自动压缩后上传</div>
                </button>
              ) : (
                <div>
                  <img
                    src={image.previewUrl}
                    alt="公告截图"
                    className="max-h-56 w-full rounded-lg border border-slate-200 object-contain"
                  />
                  <div className="mt-2 flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate text-[11px] text-ink-400">
                      {image.name}（
                      {image.compressed
                        ? `${formatBytes(image.originalBytes)} → ${formatBytes(image.bytes)}`
                        : formatBytes(image.bytes)}
                      ）
                    </span>
                    <button
                      onClick={() => {
                        setImage(null)
                        setResult(null)
                      }}
                      className="text-[11px] text-ink-400 hover:text-danger"
                    >
                      换一张
                    </button>
                  </div>
                  <input
                    value={hint}
                    onChange={(e) => setHint(e.target.value)}
                    placeholder="补充说明（可选）"
                    className="mt-2 h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
                  />
                </div>
              )}
            </>
          )}

          <div className="mt-4 flex items-center gap-2">
            <Button
              onClick={() => void run(false)}
              loading={status === 'extracting' && !saving}
              icon={<Play className="h-3.5 w-3.5" />}
            >
              开始提取
            </Button>
            {result && (
              <Button
                variant="secondary"
                onClick={() => void run(true)}
                loading={saving}
                icon={<Save className="h-3.5 w-3.5" />}
              >
                提取并入库
              </Button>
            )}
          </div>

          {error && (
            <div className="mt-3 rounded-lg border-l-2 border-danger bg-red-50 px-3 py-2">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-danger">
                <AlertTriangle className="h-3.5 w-3.5" />
                {error}
              </div>
            </div>
          )}
        </Card>

        {/* ── 右侧：结果区 ─────────────────────────────────────── */}
        <Card
          className="flex-1"
          title="提取结果"
          subtitle={result ? `完整度 ${(result.meta.completeness * 100).toFixed(0)}%（${filledCount}/${fields.length} 字段有值）` : '17 个结构化字段'}
          extra={
            result && (
              <div className="flex items-center gap-1">
                <button
                  onClick={copyJson}
                  title="复制 JSON"
                  className="rounded-md p-1.5 text-ink-400 hover:bg-ink-50 hover:text-ink-700"
                >
                  <Copy className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={downloadCsv}
                  title="导出 CSV"
                  className="rounded-md p-1.5 text-ink-400 hover:bg-ink-50 hover:text-ink-700"
                >
                  <Download className="h-3.5 w-3.5" />
                </button>
              </div>
            )
          }
        >
          {status === 'extracting' && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-[12px] text-ink-500">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                {mode === 'image' ? '视觉模型正在读图…' : '大模型正在抽取字段…'}（约 5~15 秒）
              </div>
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-6 w-full" />
              ))}
            </div>
          )}

          {status !== 'extracting' && !result && (
            <EmptyState
              icon={<FileSearch className="h-6 w-6" />}
              title="等待提取"
              description={
                mode === 'text'
                  ? '粘贴公告文本后点「开始提取」，或直接点示例标签试用'
                  : '上传公告截图后点「开始提取」'
              }
            />
          )}

          {status !== 'extracting' && result && (
            <div>
              {/* 元信息条：让用户知道这次抽取有多可信 */}
              <div className="mb-3 flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-ink-50/60 px-3 py-2">
                <Badge color={result.meta.completeness >= 0.6 ? 'ok' : 'warn'}>
                  <CheckCircle2 className="h-3 w-3" />
                  完整度 {(result.meta.completeness * 100).toFixed(0)}%
                </Badge>
                <Badge color="neutral">
                  {result.meta.source === 'image' ? (
                    <>
                      <ImageIcon className="h-3 w-3" />
                      图片模式
                    </>
                  ) : (
                    '文本模式'
                  )}
                </Badge>
                <span className="text-[11px] text-ink-400">
                  {result.meta.provider} / {result.meta.model}
                </span>
                {result.meta.truncated && (
                  <Badge color="warn">
                    <AlertTriangle className="h-3 w-3" />
                    输入已截断
                  </Badge>
                )}
                {result.save?.saved && (
                  <Badge color="brand">
                    <Database className="h-3 w-3" />
                    已入库 id={result.save.id}
                  </Badge>
                )}
              </div>

              {result.meta.warnings.length > 0 && (
                <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                  <div className="flex items-center gap-1.5 text-[11px] font-semibold text-warn">
                    <Info className="h-3 w-3" />
                    模型给出了可疑值，已自动清洗
                  </div>
                  <ul className="mt-1 space-y-0.5">
                    {result.meta.warnings.map((w, i) => (
                      <li key={i} className="text-[11px] text-ink-600">
                        • {w}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 字段表 */}
              <table className="w-full text-[13px]">
                <tbody>
                  {fields.map((f) => {
                    const value = result.record[f.key]
                    const empty = value === null || value === undefined || value === ''
                    return (
                      <tr key={f.key} className="border-b border-slate-100 last:border-0">
                        <td className="w-24 py-2 pr-3 align-top text-xs text-ink-400">
                          {f.label}
                          {f.required && <span className="ml-0.5 text-danger">*</span>}
                        </td>
                        <td className={`py-2 ${empty ? '' : 'text-ink-900'}`}>
                          <FieldValue field={f} value={value} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>

              {result.save && !result.save.saved && (
                <div className="mt-3 rounded-lg border-l-2 border-warn bg-amber-50 px-3 py-2 text-[11px] text-ink-600">
                  未入库：{result.save.reason}
                  {result.save.skipped_columns && result.save.skipped_columns.length > 0 && (
                    <span>（字段 {result.save.skipped_columns.join('、')} 在表中无对应列）</span>
                  )}
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
