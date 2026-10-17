/**
 * 招标文件解析页。
 *
 * 数据来源：`/api/tender/*`（parse 预览 + extract 抽取 + extract/stream 流式）
 *
 * 与「标讯智能提取」页的分工
 * -------------------------
 *   /extract  面向**中标结果公告**——一页纸，17 个平铺字段，用途是入库统计
 *   /tender   面向**招标文件正文**——几十页 PDF/Word，
 *             含资格要求/评分标准/废标条款等**列表型章节**，用途是投标决策
 *
 * 三个设计取向
 * ------------
 * 1. **先预览再抽取**：上传后立刻调 `/parse`（秒级）显示页数与章节结构。
 *    抽取要调多次 LLM（云端 1~2 分钟），先确认"文件读对了"能避免白等。
 * 2. **流式显示进度**：分块抽取有多批，没有进度反馈用户会以为卡死。
 * 3. **列表章节是重点**：把「废标条款」与「资格要求」放在最显眼位置——
 *    漏看一条废标条款的代价远大于报价高低。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  ClipboardList,
  Copy,
  Database,
  FileDown,
  FileText,
  Info,
  ListChecks,
  Loader2,
  Save,
  Target,
  Upload,
  X,
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
  extractTenderStream,
  fetchTenderSchema,
  downloadTenderReport,
  parseTenderDocument,
  saveTenderResult,
  type TenderExtractResult,
  type TenderFieldDef,
  type TenderItem,
  type TenderParseResult,
  type TenderSectionDef,
} from '../lib/api'
import { formatMoney } from '../lib/format'

/** 章节的视觉强调级别：废标条款最需要被看见 */
const SECTION_ACCENT: Record<string, { color: 'danger' | 'brand' | 'warn' | 'ok' | 'neutral'; icon: typeof Ban }> = {
  disqualification: { color: 'danger', icon: Ban },
  qualifications: { color: 'brand', icon: CheckCircle2 },
  scoring_items: { color: 'warn', icon: Target },
  technical_requirements: { color: 'neutral', icon: ListChecks },
  required_documents: { color: 'neutral', icon: ClipboardList },
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 413) return err.message
    if (err.status === 400) return err.message
    if (err.status === 503) return '招标文件解析服务未就绪（检查后端 LLM 配置）'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

function FieldValue({ field, value }: { field: TenderFieldDef; value: string | number | null }) {
  if (value === null || value === undefined || value === '') {
    return <span className="text-ink-300">—</span>
  }
  if (field.type === 'money') {
    return <span className="font-medium tabular-nums">{formatMoney(Number(value))}</span>
  }
  if (field.type === 'datetime') {
    // 招标文件的时间精确到分钟，shortDate 会丢掉时分，故这里直接显示原文
    return <span className="tabular-nums">{String(value).slice(0, 16)}</span>
  }
  return <span className="break-words">{String(value)}</span>
}

/** 列表章节里一条条目的渲染：按 item_fields 决定显示哪些子字段 */
function ItemRow({ item, index, section }: { item: TenderItem; index: number; section: TenderSectionDef }) {
  const text = String(item.text ?? '')
  const category = item.category ? String(item.category) : ''
  const score = item.score !== undefined && item.score !== null && item.score !== '' ? Number(item.score) : null
  // is_mandatory 只在资格要求章节有意义（后端只给该章节抽这个子字段）
  const hasMandatory = section.item_fields.includes('is_mandatory')
  const mandatory = item.is_mandatory === true

  return (
    <li className="flex items-start gap-2 py-1.5">
      <span className="mt-0.5 w-6 shrink-0 text-right text-[11px] tabular-nums text-ink-300">
        {index + 1}
      </span>
      <span className="min-w-0 flex-1 text-[13px] leading-6 text-ink-900">{text}</span>
      <span className="flex shrink-0 items-center gap-1.5">
        {score !== null && <Badge color="warn">{score} 分</Badge>}
        {category && <Badge color="neutral">{category}</Badge>}
        {hasMandatory && (
          <Badge color={mandatory ? 'danger' : 'ok'}>
            {mandatory ? '硬性门槛' : '加分项'}
          </Badge>
        )}
      </span>
    </li>
  )
}

export default function TenderDoc() {
  const toast = useToast()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const [schema, setSchema] = useState<{ fields: TenderFieldDef[]; sections: TenderSectionDef[] }>({
    fields: [],
    sections: [],
  })

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<TenderParseResult | null>(null)
  const [parsing, setParsing] = useState(false)

  const [extracting, setExtracting] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number; message: string } | null>(null)
  const [result, setResult] = useState<TenderExtractResult | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [savedId, setSavedId] = useState<number | null>(null)
  const [exporting, setExporting] = useState(false)

  useEffect(() => {
    fetchTenderSchema()
      .then((r) => setSchema({ fields: r.fields, sections: r.sections }))
      .catch(() => toast.show('字段定义拉取失败，请确认后端已启动', 'error'))
  }, [toast])

  const pickFile = useCallback(
    async (f: File | undefined) => {
      if (!f) return
      setFile(f)
      setResult(null)
      setError('')
      setPreview(null)
      setProgress(null)
      if (fileInputRef.current) fileInputRef.current.value = ''

      // 上传后立刻解析预览：秒级反馈，让用户确认文件读对了再花时间抽取
      setParsing(true)
      try {
        const p = await parseTenderDocument(f)
        setPreview(p)
        toast.show(`已解析 ${p.pages} 页 / ${p.total_chars.toLocaleString('zh-CN')} 字`, 'success')
      } catch (err) {
        const msg = describeError(err)
        setError(msg)
        toast.show(msg, 'error')
      } finally {
        setParsing(false)
      }
    },
    [toast],
  )

  const runExtract = useCallback(async () => {
    if (!file || extracting) return
    setExtracting(true)
    setResult(null)
    setError('')
    setSavedId(null)
    setExporting(false)
    setProgress({ done: 0, total: 1, message: '正在解析文档…' })

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await extractTenderStream(
        file,
        {
          onParse: (info) => {
            setPreview(info)
            setProgress({ done: 0, total: 1, message: `已解析 ${info.chunk_count} 个章节块，准备抽取…` })
          },
          onProgress: (done, total, message) => setProgress({ done, total, message }),
          onDone: (r) => {
            setResult(r)
            toast.show(
              `抽取完成，完整度 ${(r.meta.completeness * 100).toFixed(0)}%`,
              r.meta.completeness >= 0.6 ? 'success' : 'warning',
            )
          },
          onError: (msg) => {
            setError(msg)
            toast.show(msg, 'error')
          },
        },
        controller.signal,
      )
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        const msg = describeError(err)
        setError(msg)
        toast.show(msg, 'error')
      }
    } finally {
      setExtracting(false)
      setProgress(null)
      abortRef.current = null
    }
  }, [file, extracting, toast])

  const stop = () => abortRef.current?.abort()

  /** 保存抽取结果：存下来才能做截止监控与变更追踪 */
  const saveResult = useCallback(async () => {
    if (!result || saving) return
    setSaving(true)
    try {
      const r = await saveTenderResult(result.parsed, result.meta as unknown as Record<string, unknown>)
      if (r.saved) {
        setSavedId(r.id)
        if (r.duplicate_of) {
          // 同项目编号已存在：不是错误，而是变更追踪的入口
          toast.show(
            `已保存（id ${r.id}）。该项目此前已解析过（id ${r.duplicate_of}），可用于变更对比`,
            'success',
          )
        } else {
          toast.show(`已保存到数据库（id ${r.id}）`, 'success')
        }
      } else {
        toast.show(`保存失败：${r.reason ?? '未知原因'}`, 'warning')
      }
    } catch (err) {
      toast.show(describeError(err), 'error')
    } finally {
      setSaving(false)
    }
  }, [result, saving, toast])

  /** 导出投标决策报告（Word） */
  const exportReport = useCallback(async () => {
    if (!result || exporting) return
    setExporting(true)
    try {
      await downloadTenderReport({
        parsed: result.parsed,
        meta: result.meta as unknown as Record<string, unknown>,
      })
      toast.show('报告已开始下载', 'success')
    } catch (err) {
      toast.show(describeError(err), 'error')
    } finally {
      setExporting(false)
    }
  }, [result, exporting, toast])

  const copyJson = () => {
    if (!result) return
    navigator.clipboard
      .writeText(JSON.stringify(result.parsed, null, 2))
      .then(() => toast.show('JSON 已复制到剪贴板', 'success'))
      .catch(() => toast.show('复制失败', 'error'))
  }

  const filledFields = result
    ? schema.fields.filter((f) => result.parsed.fields[f.key] !== null && result.parsed.fields[f.key] !== '')
    : []
  const totalItems = result
    ? Object.values(result.parsed.sections).reduce((n, arr) => n + arr.length, 0)
    : 0

  return (
    <div>
      <PageHeader
        title="招标文件解析"
        description="上传 PDF / Word 招标文件，抽取项目信息、资格要求、评分标准与废标条款"
        actions={
          schema.fields.length > 0 && (
            <Badge color="neutral">
              <FileText className="h-3 w-3" />
              {schema.fields.length} 字段 + {schema.sections.length} 章节
            </Badge>
          )
        }
      />

      {/* ★ 移动端纵向堆叠（实测缺陷修复）：原来是横向 flex，
            而子元素卡片固定宽度（如 340px）且 shrink-0 不收缩，
            375px 屏上会把右侧结果区推出屏外（审计实测确认）。 */}
      <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-start lg:gap-4">
        {/* ── 左侧：上传与预览 ─────────────────────────────────── */}
        <Card
          className="w-full lg:w-[380px] lg:shrink-0"
          title="上传文件"
          subtitle="支持 PDF 与 Word(.docx)；扫描件请改用截图上传"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            className="hidden"
            onChange={(e) => void pickFile(e.target.files?.[0])}
          />

          {!file ? (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex h-40 w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-ink-200 transition-colors hover:border-brand-300 hover:bg-brand-50/40"
            >
              <Upload className="h-7 w-7 text-ink-300" />
              <div className="text-[13px] font-medium text-ink-700">点击选择招标文件</div>
              <div className="text-[11px] text-ink-400">PDF / Word(.docx)，最大 50MB</div>
            </button>
          ) : (
            <div>
              <div className="flex items-start gap-3 rounded-lg border border-slate-200 p-3">
                <FileText className="mt-0.5 h-5 w-5 shrink-0 text-brand-600" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] font-medium text-ink-900" title={file.name}>
                    {file.name}
                  </div>
                  <div className="mt-0.5 text-[11px] text-ink-400">
                    {(file.size / 1024).toFixed(0)} KB
                  </div>
                </div>
                <button
                  onClick={() => {
                    setFile(null)
                    setPreview(null)
                    setResult(null)
                    setError('')
                  }}
                  className="shrink-0 rounded p-0.5 text-ink-400 hover:text-danger"
                  title="移除"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="mt-3 flex items-center gap-2">
                {extracting ? (
                  <Button variant="secondary" onClick={stop} icon={<X className="h-3.5 w-3.5" />}>
                    停止抽取
                  </Button>
                ) : (
                  <Button
                    onClick={() => void runExtract()}
                    loading={parsing}
                    icon={<Target className="h-3.5 w-3.5" />}
                  >
                    {result ? '重新抽取' : '开始抽取'}
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* 解析预览 */}
          {parsing && (
            <div className="mt-3 flex items-center gap-2 text-[12px] text-ink-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
              正在解析文档结构…
            </div>
          )}
          {preview && (
            <div className="mt-3 rounded-lg border border-slate-200 bg-ink-50/50 p-3">
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <Badge color="brand">{preview.kind.toUpperCase()}</Badge>
                <span className="text-ink-500">{preview.pages} 页</span>
                <span className="text-ink-400">·</span>
                <span className="text-ink-500">{preview.total_chars.toLocaleString('zh-CN')} 字</span>
                <span className="text-ink-400">·</span>
                <span className="text-ink-500">{preview.chunk_count} 章节块</span>
              </div>
              {preview.truncated && (
                <div className="mt-2 flex items-start gap-1.5 text-[11px] text-warn">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                  文档过长已截断，后文未参与抽取
                </div>
              )}
              {preview.warnings.map((w, i) => (
                <div key={i} className="mt-2 flex items-start gap-1.5 text-[11px] leading-4 text-warn">
                  <Info className="mt-0.5 h-3 w-3 shrink-0" />
                  <span>{w}</span>
                </div>
              ))}
              {preview.headings.length > 0 && (
                <div className="mt-2">
                  <div className="text-[11px] text-ink-400">识别到的章节：</div>
                  <ul className="mt-1 max-h-40 space-y-0.5 overflow-y-auto">
                    {preview.headings.map((h, i) => (
                      <li key={i} className="truncate text-[11px] text-ink-600" title={h}>
                        · {h}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* 抽取进度 */}
          {extracting && progress && (
            <div className="mt-3 rounded-lg border border-brand-100 bg-brand-50 p-3">
              <div className="flex items-center gap-2 text-[12px] text-ink-700">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                {progress.message}
              </div>
              {progress.total > 1 && (
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white">
                  <div
                    className="h-full bg-brand-500 transition-all"
                    style={{ width: `${Math.round((progress.done / progress.total) * 100)}%` }}
                  />
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mt-3 rounded-lg border-l-2 border-danger bg-red-50 px-3 py-2">
              <div className="flex items-start gap-2 text-[12px] font-semibold text-danger">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span className="min-w-0 break-words">{error}</span>
              </div>
            </div>
          )}
        </Card>

        {/* ── 右侧：抽取结果 ───────────────────────────────────── */}
        <div className="min-w-0 flex-1">
          {!result && !extracting && (
            <Card>
              <EmptyState
                icon={<FileText className="h-6 w-6" />}
                title="上传招标文件后开始抽取"
                description="支持 PDF 与 Word；系统会先解析章节结构，再分块抽取 16 个字段与 5 类章节"
              />
            </Card>
          )}

          {extracting && !result && (
            <Card>
              <div className="space-y-3">
                <div className="text-[12px] text-ink-500">正在抽取，通常需要 1~3 分钟…</div>
                {Array.from({ length: 10 }).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-full" />
                ))}
              </div>
            </Card>
          )}

          {result && (
            <>
              {/* 元信息 */}
              <Card
                title="抽取结果"
                subtitle={`${result.meta.filename}｜${result.meta.pages} 页｜${result.meta.batch_count} 批（成功 ${result.meta.succeeded_batches}）`}
                extra={
                  <div className="flex items-center gap-2">
                    <Badge color={result.meta.completeness >= 0.6 ? 'ok' : 'warn'}>
                      完整度 {(result.meta.completeness * 100).toFixed(0)}%
                    </Badge>
                    {savedId !== null ? (
                      <Badge color="brand">
                        <Database className="h-3 w-3" />
                        已入库 id={savedId}
                      </Badge>
                    ) : (
                      <Button
                        variant="secondary"
                        size="sm"
                        loading={saving}
                        onClick={() => void saveResult()}
                        icon={<Save className="h-3.5 w-3.5" />}
                      >
                        保存
                      </Button>
                    )}
                    <Button
                      variant="secondary"
                      size="sm"
                      loading={exporting}
                      onClick={() => void exportReport()}
                      icon={<FileDown className="h-3.5 w-3.5" />}
                    >
                      导出报告
                    </Button>
                    <button
                      onClick={copyJson}
                      title="复制 JSON"
                      className="rounded-md p-1.5 text-ink-400 hover:bg-ink-50 hover:text-ink-700"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>
                }
              >
                {result.meta.warnings.length > 0 && (
                  <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold text-warn">
                      <AlertTriangle className="h-3 w-3" />
                      抽取过程中的提示
                    </div>
                    <ul className="mt-1 space-y-0.5">
                      {result.meta.warnings.map((w, i) => (
                        <li key={i} className="text-[11px] leading-5 text-ink-600">
                          • {w}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 标量字段 */}
                <div className="mb-2 flex items-center gap-2 text-[12px] font-medium text-ink-600">
                  <FileText className="h-3.5 w-3.5" />
                  项目信息（{filledFields.length}/{schema.fields.length} 项有值）
                </div>
                <table className="w-full text-[13px]">
                  <tbody>
                    {schema.fields.map((f) => (
                      <tr key={f.key} className="border-b border-slate-100 last:border-0">
                        <td className="w-28 py-1.5 pr-3 align-top text-xs text-ink-400">
                          {f.label}
                          {f.required && <span className="ml-0.5 text-danger">*</span>}
                        </td>
                        <td className="py-1.5 text-ink-900">
                          <FieldValue field={f} value={result.parsed.fields[f.key]} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              {/* 列表章节 */}
              <div className="mt-4 flex items-center gap-2 text-[15px] font-bold text-ink-900">
                章节明细
                <Badge color="neutral">共 {totalItems} 条</Badge>
              </div>

              {schema.sections.map((s) => {
                const items = result.parsed.sections[s.key] ?? []
                const accent = SECTION_ACCENT[s.key] ?? { color: 'neutral' as const, icon: ListChecks }
                const Icon = accent.icon
                return (
                  <Card
                    key={s.key}
                    className="mt-3"
                    title={
                      <span className="flex items-center gap-2">
                        <Icon className="h-4 w-4" />
                        {s.label}
                        <Badge color={items.length ? accent.color : 'neutral'}>{items.length} 条</Badge>
                      </span>
                    }
                    subtitle={s.hint}
                  >
                    {items.length === 0 ? (
                      <div className="py-2 text-[12px] text-ink-400">
                        文档中未找到该章节内容（或该章节不存在于本文档）
                      </div>
                    ) : (
                      <ul className="divide-y divide-slate-100">
                        {items.map((item, i) => (
                          <ItemRow key={i} item={item} index={i} section={s} />
                        ))}
                      </ul>
                    )}
                    {s.key === 'disqualification' && items.length > 0 && (
                      <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2">
                        <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-danger" />
                        <span className="text-[11px] leading-5 text-ink-700">
                          以上任一条不满足都会导致<b>废标</b>，请逐条核对投标文件。
                          这些条款常分散在「投标人须知」「投标文件的编制」「评标办法」等多个章节，
                          模型可能漏抽——<b>建议人工复核原文</b>。
                        </span>
                      </div>
                    )}
                  </Card>
                )
              })}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
