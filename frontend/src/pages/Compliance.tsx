/**
 * 合规性检查页（⑨）。
 *
 * ⚠️ 本页最重要的设计约束：**如实表达"这是疑似，不是认定"**
 * ------------------------------------------------------
 * 合规判断是要担责任的。把"疑似"包装成"违规认定"会让用户拿去做错误决策，
 * 比不做检查更糟。故本页：
 *    - 标题与措辞统一用"疑似"
 *    - 每条都展示 `why`（为什么可疑）与 `how_to_verify`（怎么核实）
 *    - 免责说明**放在结果最上方**，不是藏在页面底部
 *    - 明确写出"未命中的条款不代表合规"
 *
 * 第二层设计：证据必须可追溯
 * ------------------------
 * 每条命中都展示**原文中命中的片段**（`evidence`），用户能立刻判断
 * 规则有没有误读——这比让用户信任一个黑盒结论重要得多。
 *
 * 第三层：法规依据是增强而非前提
 * --------------------------
 * 法规语料未入库时，规则检出仍完整工作（只是没有条文引用）。
 * 页面要如实说明当前处于哪种状态，而不是假装有依据。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  FileSearch,
  Info,
  Loader2,
  Scale,
  ShieldAlert,
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
  checkComplianceFromFile,
  fetchComplianceRules,
  fetchRegulationStats,
  type ComplianceFinding,
  type ComplianceResult,
  type ComplianceRuleDef,
  type ComplianceRulesCatalog,
  type ComplianceSeverity,
} from '../lib/api'

const SEVERITY_META: Record<
  ComplianceSeverity,
  { color: 'danger' | 'warn' | 'neutral'; label: string; icon: typeof ShieldAlert }
> = {
  high: { color: 'danger', label: '高风险', icon: ShieldAlert },
  medium: { color: 'warn', label: '需核实', icon: AlertTriangle },
  low: { color: 'neutral', label: '表述不规范', icon: Info },
}

/** 章节 key → 中文名（与招标文件 schema 对应） */
const SECTION_LABEL: Record<string, string> = {
  disqualification: '废标条款',
  qualifications: '资格要求',
  scoring_items: '评分标准',
  technical_requirements: '技术要求',
  required_documents: '需提交材料',
  payment_terms: '付款方式',
  budget: '预算金额',
  max_price: '最高限价',
  deposit: '投标保证金',
  duration: '服务期/工期',
  evaluation_method: '评标办法',
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 400) return err.message
    if (err.status === 413) return '文件过大，上限 50MB'
    if (err.status === 503) return '合规检查服务未就绪'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

function FindingCard({ finding }: { finding: ComplianceFinding }) {
  const meta = SEVERITY_META[finding.severity] ?? SEVERITY_META.low
  const Icon = meta.icon
  const section = SECTION_LABEL[finding.section] ?? finding.section

  return (
    <div
      className={`rounded-lg border p-3.5 ${
        finding.severity === 'high'
          ? 'border-red-200 bg-red-50/50'
          : finding.severity === 'medium'
            ? 'border-amber-200 bg-amber-50/40'
            : 'border-slate-200'
      }`}
    >
      <div className="flex items-start gap-2.5">
        <Icon
          className={`mt-0.5 h-4 w-4 shrink-0 ${
            finding.severity === 'high'
              ? 'text-danger'
              : finding.severity === 'medium'
                ? 'text-warn'
                : 'text-ink-400'
          }`}
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[13px] font-semibold text-ink-900">{finding.title}</span>
            <Badge color={meta.color}>{meta.label}</Badge>
            <Badge color="neutral">{section}</Badge>
          </div>

          {/* 原文 */}
          <div className="mt-2 text-[12px] leading-5 text-ink-700">
            <span className="text-ink-400">条款原文：</span>
            {finding.clause}
          </div>

          {/* 证据：必须高亮命中片段，让用户能判断规则有没有误读 */}
          <div className="mt-1 rounded bg-white/70 px-2 py-1 text-[11px] leading-5">
            <span className="text-ink-400">命中片段：</span>
            <span className="font-medium text-danger">{finding.evidence}</span>
          </div>

          <div className="mt-1.5 space-y-1 text-[11px] leading-5">
            <div className="text-ink-600">
              <span className="text-ink-400">为什么可疑：</span>
              {finding.why}
            </div>
            <div className="text-ink-600">
              <span className="text-ink-400">怎么核实：</span>
              {finding.how_to_verify}
            </div>
          </div>

          {/* 法规依据 */}
          {finding.law_basis.length > 0 ? (
            <div className="mt-2 space-y-1 rounded border border-brand-100 bg-brand-50/50 px-2 py-1.5">
              <div className="flex items-center gap-1 text-[10px] font-semibold text-brand-700">
                <BookOpen className="h-3 w-3" />
                法规依据
              </div>
              {finding.law_basis.map((b, i) => (
                <div key={i} className="text-[11px] leading-5 text-ink-600">
                  <span className="font-medium text-brand-700">
                    《{b.law}》{b.article}
                  </span>
                  <div className="text-ink-500">{b.excerpt}…</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-2 text-[10px] text-ink-400">
              未附法规条文（法规语料未入库）——本项仅由规则匹配得出
            </div>
          )}

          {finding.topics.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {finding.topics.map((t) => (
                <span key={t} className="rounded bg-ink-50 px-1.5 py-0.5 text-[10px] text-ink-500">
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Compliance() {
  const toast = useToast()
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const [catalog, setCatalog] = useState<ComplianceRulesCatalog | null>(null)
  const [regStats, setRegStats] = useState<Record<string, unknown> | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState('')
  const [result, setResult] = useState<ComplianceResult | null>(null)
  const [error, setError] = useState('')
  const [showRules, setShowRules] = useState(false)

  useEffect(() => {
    fetchComplianceRules()
      .then(setCatalog)
      .catch(() => toast.show('规则清单拉取失败，请确认后端已启动', 'error'))
    fetchRegulationStats()
      .then(setRegStats)
      .catch(() => undefined)
  }, [toast])

  const run = useCallback(async () => {
    if (!file || running) return
    setRunning(true)
    setResult(null)
    setError('')
    setPhase('正在解析、抽取并检查…')

    const controller = new AbortController()
    abortRef.current = controller

    try {
      // 上传版端点一步完成解析 + 抽取 + 检查；抽取要调 LLM，通常 1~3 分钟
      const r = await checkComplianceFromFile(file, true)
      setResult(r)
      const s = r.summary
      toast.show(
        s.total === 0
          ? '未检出规则库覆盖范围内的疑似条款'
          : `检出 ${s.total} 项疑似条款（高风险 ${s.counts.high}）`,
        s.counts.high > 0 ? 'warning' : s.total > 0 ? 'info' : 'success',
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
  }, [file, running, toast])

  const regAvailable = regStats?.available === true

  return (
    <div>
      <PageHeader
        title="合规性检查"
        description="检查招标文件中涉嫌以不合理条件限制竞争、差别待遇等违规情形的条款"
        actions={
          catalog && (
            <div className="flex items-center gap-2">
              <Badge color="neutral">
                <Scale className="h-3 w-3" />
                {catalog.count} 条检查规则
              </Badge>
              <Button variant="ghost" size="sm" onClick={() => setShowRules((v) => !v)}>
                {showRules ? '收起规则' : '查看规则'}
              </Button>
            </div>
          )
        }
      />

      {/* 规则清单（可折叠） */}
      {showRules && catalog && (
        <Card className="mb-4" title="检查规则清单" subtitle="我们检查以下方面；未列入的不代表合规">
          <div className="space-y-2">
            {catalog.rules.map((r: ComplianceRuleDef) => {
              const meta = SEVERITY_META[r.severity]
              return (
                <div key={r.key} className="rounded-lg border border-slate-200 p-2.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[12px] font-semibold text-ink-900">{r.title}</span>
                    <Badge color={meta.color}>{meta.label}</Badge>
                    {r.topics.map((t) => (
                      <span key={t} className="rounded bg-ink-50 px-1.5 py-0.5 text-[10px] text-ink-500">
                        {t}
                      </span>
                    ))}
                  </div>
                  <div className="mt-1 text-[11px] leading-5 text-ink-600">{r.why}</div>
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* ★ 移动端纵向堆叠（实测缺陷修复）：原来是横向 flex，
            而子元素卡片固定宽度（如 340px）且 shrink-0 不收缩，
            375px 屏上会把右侧结果区推出屏外（审计实测确认）。 */}
      <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-start lg:gap-4">
        {/* ── 左：上传 ───────────────────────────────────────── */}
        <Card className="w-full lg:w-[340px] lg:shrink-0" title="上传招标文件" subtitle="支持 PDF 与 Word(.docx)">
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) {
                setFile(f)
                setResult(null)
                setError('')
              }
              if (fileRef.current) fileRef.current.value = ''
            }}
          />

          {!file ? (
            <button
              onClick={() => fileRef.current?.click()}
              className="flex h-32 w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-ink-200 transition-colors hover:border-brand-300 hover:bg-brand-50/40"
            >
              <Upload className="h-6 w-6 text-ink-300" />
              <span className="text-[12px] font-medium text-ink-700">点击选择招标文件</span>
              <span className="text-[10px] text-ink-400">PDF / Word(.docx)，最大 50MB</span>
            </button>
          ) : (
            <div>
              <div className="flex items-start gap-2 rounded-lg border border-slate-200 p-2.5">
                <FileSearch className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[12px] font-medium text-ink-900" title={file.name}>
                    {file.name}
                  </div>
                  <div className="text-[10px] text-ink-400">{(file.size / 1024).toFixed(0)} KB</div>
                </div>
                <button
                  onClick={() => {
                    setFile(null)
                    setResult(null)
                    setError('')
                  }}
                  className="shrink-0 rounded p-0.5 text-ink-400 hover:text-danger"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <div className="mt-3">
                {running ? (
                  <Button variant="secondary" onClick={() => abortRef.current?.abort()} icon={<X className="h-3.5 w-3.5" />}>
                    停止
                  </Button>
                ) : (
                  <Button onClick={() => void run()} icon={<Scale className="h-3.5 w-3.5" />}>
                    {result ? '重新检查' : '开始检查'}
                  </Button>
                )}
              </div>
            </div>
          )}

          {running && (
            <div className="mt-3 rounded-lg border border-brand-100 bg-brand-50 p-3">
              <div className="flex items-center gap-2 text-[12px] text-ink-700">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                {phase}
              </div>
              <div className="mt-1 text-[10px] text-ink-500">需先抽取招标文件，通常 1~3 分钟</div>
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

          {/* 法规语料状态：如实说明当前有无条文引用 */}
          <div
            className={`mt-3 rounded-lg border px-3 py-2 ${
              regAvailable ? 'border-brand-200 bg-brand-50/50' : 'border-amber-200 bg-amber-50'
            }`}
          >
            <div className="flex items-start gap-1.5">
              <BookOpen className={`mt-0.5 h-3 w-3 shrink-0 ${regAvailable ? 'text-brand-600' : 'text-warn'}`} />
              <div className="text-[10px] leading-4 text-ink-600">
                {regAvailable ? (
                  <>
                    法规语料已入库
                    {typeof regStats?.points === 'number' && `（${regStats.points} 条条文`}
                    {typeof regStats?.laws === 'number' && ` / ${regStats.laws} 部法规）`}
                    ，检查结果会附上法规依据
                    {regStats?.in_sync === false && (
                      <span className="text-warn">
                        （⚠ 库内条数与语料不一致，建议重新运行入库脚本）
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    <b>法规语料未入库</b>
                    ：当前检查只用规则匹配，<b>不附法规条文引用</b>。
                    规则检出本身完整有效，但建议同时人工查阅法规原文。
                  </>
                )}
              </div>
            </div>
          </div>
        </Card>

        {/* ── 右：结果 ───────────────────────────────────────── */}
        <div className="min-w-0 flex-1">
          {!result && !running && (
            <Card>
              <EmptyState
                icon={<Scale className="h-6 w-6" />}
                title="上传招标文件后开始检查"
                description="系统会抽取招标文件内容，逐条比对 12 类已知的违规特征模式，并附上法规依据"
              />
            </Card>
          )}

          {running && !result && (
            <Card>
              <div className="flex flex-col items-center justify-center gap-3 py-12">
                <Loader2 className="h-7 w-7 animate-spin text-brand-600" />
                <div className="text-[13px] text-ink-600">{phase}</div>
                <div className="text-[11px] text-ink-400">正在抽取招标文件内容，请勿关闭页面</div>
              </div>
            </Card>
          )}

          {result && (
            <>
              {/* ⚠️ 免责说明放在**最上方**：不是藏在底部 */}
              <div className="mb-4 flex items-start gap-2.5 rounded-xl border-l-4 border-amber-400 bg-amber-50 p-4">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
                <div className="min-w-0 text-[11px] leading-5 text-ink-700">
                  <b>请先读这段：</b>
                  <span dangerouslySetInnerHTML={{ __html: result.disclaimer.replace(/\*\*/g, '') }} />
                </div>
              </div>

              {/* 结论 + 统计 */}
              <div
                className={`mb-4 rounded-xl border-l-4 p-4 ${
                  result.summary.counts.high > 0
                    ? 'border-danger bg-red-50'
                    : result.summary.total > 0
                      ? 'border-amber-400 bg-amber-50'
                      : 'border-emerald-400 bg-emerald-50'
                }`}
              >
                <div className="flex items-start gap-2.5">
                  {result.summary.total === 0 ? (
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
                  ) : (
                    <ShieldAlert
                      className={`mt-0.5 h-5 w-5 shrink-0 ${
                        result.summary.counts.high > 0 ? 'text-danger' : 'text-warn'
                      }`}
                    />
                  )}
                  <div className="min-w-0">
                    <div className="text-[15px] font-bold text-ink-900">
                      {result.summary.total === 0
                        ? '未检出规则库覆盖范围内的疑似条款'
                        : `检出 ${result.summary.total} 项疑似条款`}
                    </div>
                    <div className="mt-1 text-[12px] leading-5 text-ink-700">
                      {result.summary.conclusion}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mb-4 grid grid-cols-4 gap-3">
                {(
                  [
                    ['checked', '已检查条款', result.summary.checked_clauses, 'text-ink-900'],
                    ['high', '高风险', result.summary.counts.high, 'text-danger'],
                    ['medium', '需核实', result.summary.counts.medium, 'text-warn'],
                    ['low', '表述不规范', result.summary.counts.low, 'text-ink-500'],
                  ] as const
                ).map(([k, label, count, cls]) => (
                  <div key={k} className="rounded-xl border border-slate-200 bg-white p-3">
                    <div className="text-[11px] text-ink-500">{label}</div>
                    <div className={`mt-0.5 text-xl font-bold tabular-nums ${cls}`}>{count}</div>
                  </div>
                ))}
              </div>

              {/* 抽取质量：检查结论的可靠性取决于抽取质量 */}
              {result.extract_meta && (
                <div className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-ink-50/50 px-3 py-2 text-[11px] text-ink-600">
                  <span>
                    抽取自 <b>{result.extract_meta.filename}</b>
                  </span>
                  <span className="text-ink-400">·</span>
                  <span>{result.extract_meta.pages} 页</span>
                  <span className="text-ink-400">·</span>
                  <span>
                    完整度{' '}
                    <b>
                      {(result.extract_meta.completeness * 100).toFixed(0)}%
                    </b>
                  </span>
                  {result.extract_meta.truncated && (
                    <>
                      <span className="text-ink-400">·</span>
                      <span className="text-danger">文件过长已截断，后文未检查</span>
                    </>
                  )}
                  {(result.extract_meta.warnings ?? []).length > 0 && (
                    <>
                      <span className="text-ink-400">·</span>
                      <span className="text-warn">
                        {result.extract_meta.warnings.length} 条抽取提示
                      </span>
                    </>
                  )}
                </div>
              )}

              {/* 命中明细 */}
              {result.findings.length > 0 && (
                <Card
                  title={`疑似条款明细（${result.findings.length} 项）`}
                  subtitle="按严重度排序：高风险 → 需核实 → 表述不规范"
                >
                  <div className="space-y-2.5">
                    {result.findings.map((f, i) => (
                      <FindingCard key={`${f.rule}-${i}`} finding={f} />
                    ))}
                  </div>
                </Card>
              )}

              {result.findings.length === 0 && (
                <Card>
                  <EmptyState
                    icon={<CheckCircle2 className="h-6 w-6" />}
                    title="规则库覆盖范围内未发现问题"
                    description="这不代表文件完全合规——规则库只覆盖已知的违规特征模式，请仍对照法规原文自行核对"
                  />
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
