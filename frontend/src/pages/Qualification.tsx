/**
 * 资格审查辅助页（⑤）。
 *
 * 这个页面回答一个问题：**我能不能投这个标**。
 *
 * 核心设计：三值判定，不是二值
 * --------------------------
 * `met` 满足 / `unmet` 不满足 / `uncertain` **需人工确认**
 *
 * 第三档不能省，因为两种误判的代价都是真实的：
 *   - 把不确定判成「不满足」→ 用户白白放弃一个能投的标
 *   - 判成「满足」→ 用户投一个必废的标
 *
 * 故总体结论也保守：有硬性要求不满足 → `unqualified`；
 * 有需确认项 → `need_review`（**不会报 qualified**）；全满足才 `qualified`。
 *
 * 两栏结构
 * --------
 *   左：企业资质档案（录入一次，反复用于不同项目）
 *   右：审查结果（逐条判定 + 理由 + 证据）
 *
 * 「需人工确认」的条目必须给出**缺什么信息才能判定**，否则用户无从下手。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  BadgeCheck,
  Building2,
  CheckCircle2,
  HelpCircle,
  Loader2,
  Plus,
  Save,
  ShieldQuestion,
  Trash2,
  Upload,
  X,
  XCircle,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'
import {
  ApiError,
  checkQualification,
  deleteQualificationProfile,
  extractTenderStream,
  fetchQualificationProfile,
  fetchQualificationProfiles,
  saveQualificationProfile,
  type CompanyProfile,
  type QualificationRequirement,
  type QualificationResult,
  type QualOverall,
  type QualStatus,
  type ProfileListItem,
  type TenderParsed,
} from '../lib/api'

const OVERALL_META: Record<
  QualOverall,
  { color: 'ok' | 'danger' | 'warn'; icon: typeof CheckCircle2; label: string }
> = {
  qualified: { color: 'ok', icon: CheckCircle2, label: '满足全部资格要求' },
  unqualified: { color: 'danger', icon: XCircle, label: '存在不满足的硬性要求' },
  need_review: { color: 'warn', icon: ShieldQuestion, label: '需人工确认' },
}

const STATUS_META: Record<
  QualStatus,
  { color: 'ok' | 'danger' | 'warn'; icon: typeof CheckCircle2; label: string }
> = {
  met: { color: 'ok', icon: CheckCircle2, label: '满足' },
  unmet: { color: 'danger', icon: XCircle, label: '不满足' },
  uncertain: { color: 'warn', icon: HelpCircle, label: '需人工确认' },
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 400) return err.message
    if (err.status === 404) return err.message
    if (err.status === 503) return '资格审查服务未就绪（检查后端与 PostgreSQL）'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

/** 空档案模板：让用户一眼看懂能填什么 */
const EMPTY_PROFILE: CompanyProfile = {
  company_name: '',
  qualifications: [],
  certifications: [],
  performance: [],
  personnel: [],
  financial: {},
  credit: {},
}

function RequirementRow({ req }: { req: QualificationRequirement }) {
  const meta = STATUS_META[req.status] ?? STATUS_META.uncertain
  const Icon = meta.icon
  return (
    <div
      className={`rounded-lg border p-3 ${
        req.status === 'unmet'
          ? 'border-red-200 bg-red-50/50'
          : req.status === 'uncertain'
            ? 'border-amber-200 bg-amber-50/40'
            : 'border-slate-200'
      }`}
    >
      <div className="flex items-start gap-2">
        <Icon
          className={`mt-0.5 h-4 w-4 shrink-0 ${
            req.status === 'unmet' ? 'text-danger' : req.status === 'uncertain' ? 'text-warn' : 'text-emerald-600'
          }`}
        />
        <div className="min-w-0 flex-1">
          <div className="text-[13px] leading-6 text-ink-900">{req.raw_text}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <Badge color={meta.color}>{meta.label}</Badge>
            {req.kind_label && <Badge color="neutral">{req.kind_label}</Badge>}
            {req.is_mandatory ? (
              <Badge color="danger">硬性门槛</Badge>
            ) : (
              <Badge color="neutral">加分项</Badge>
            )}
          </div>
          <div className="mt-1.5 text-[11px] leading-5 text-ink-600">
            <span className="text-ink-400">理由：</span>
            {req.reason}
          </div>
          {req.evidence && (
            <div className="mt-0.5 text-[11px] leading-5 text-ink-500">
              <span className="text-ink-400">依据：</span>
              {req.evidence}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Qualification() {
  const toast = useToast()
  const abortRef = useRef<AbortController | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const [profiles, setProfiles] = useState<ProfileListItem[]>([])
  const [profileName, setProfileName] = useState('')
  const [profile, setProfile] = useState<CompanyProfile>(EMPTY_PROFILE)
  const [certText, setCertText] = useState('')
  const [saving, setSaving] = useState(false)

  const [file, setFile] = useState<File | null>(null)
  const [checking, setChecking] = useState(false)
  const [phase, setPhase] = useState('')
  const [result, setResult] = useState<QualificationResult | null>(null)
  const [error, setError] = useState('')
  const [refDate, setRefDate] = useState('')

  const loadProfiles = useCallback(async () => {
    try {
      const r = await fetchQualificationProfiles()
      setProfiles(r.items)
    } catch (err) {
      toast.show(describeError(err), 'error')
    }
  }, [toast])

  useEffect(() => {
    void loadProfiles()
  }, [loadProfiles])

  const pickProfile = async (name: string) => {
    if (!name) {
      setProfileName('')
      setProfile(EMPTY_PROFILE)
      setCertText('')
      return
    }
    try {
      const p = await fetchQualificationProfile(name)
      setProfileName(name)
      setProfile(p)
      setCertText((p.certifications ?? []).join(', '))
      toast.show(`已载入档案「${name}」`, 'success')
    } catch (err) {
      toast.show(describeError(err), 'error')
    }
  }

  const save = async () => {
    if (!profileName.trim()) {
      toast.show('请先填写档案名称', 'warning')
      return
    }
    setSaving(true)
    try {
      // 认证列表从逗号/顿号分隔的文本框解析
      const merged: CompanyProfile = {
        ...profile,
        certifications: certText
          .split(/[,，、\s]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      }
      await saveQualificationProfile(profileName.trim(), merged)
      setProfile(merged)
      toast.show(`档案「${profileName}」已保存`, 'success')
      void loadProfiles()
    } catch (err) {
      toast.show(describeError(err), 'error')
    } finally {
      setSaving(false)
    }
  }

  const remove = async (name: string) => {
    try {
      await deleteQualificationProfile(name)
      toast.show(`已删除档案「${name}」`, 'success')
      if (profileName === name) {
        setProfileName('')
        setProfile(EMPTY_PROFILE)
        setCertText('')
      }
      void loadProfiles()
    } catch (err) {
      toast.show(describeError(err), 'error')
    }
  }

  const runCheck = async () => {
    if (!file || checking) return
    setChecking(true)
    setResult(null)
    setError('')
    setPhase('正在解析并抽取招标文件…')
    const controller = new AbortController()
    abortRef.current = controller

    try {
      // 先抽取（要拿到资格要求原文），再比对
      let parsed: TenderParsed | null = null
      let extractErr = ''
      await extractTenderStream(
        file,
        {
          onProgress: (_d, _t, m) => setPhase(m),
          onDone: (r) => {
            parsed = r.parsed
          },
          onError: (m) => {
            extractErr = m
          },
        },
        controller.signal,
      )
      if (extractErr) throw new Error(extractErr)
      if (!parsed) throw new Error('未能从文件中抽取出内容')

      setPhase('正在逐条比对资格要求…')
      const r = await checkQualification(parsed, {
        profileName: profileName.trim() || undefined,
        profile,
        referenceDate: refDate || undefined,
      })
      setResult(r)
      const s = r.summary
      toast.show(
        `审查完成：满足 ${s.met} / 不满足 ${s.unmet} / 需确认 ${s.uncertain}`,
        r.overall === 'unqualified' ? 'warning' : 'success',
      )
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        const msg = err instanceof Error && !(err instanceof ApiError) ? err.message : describeError(err)
        setError(msg)
        toast.show(msg, 'error')
      }
    } finally {
      setChecking(false)
      setPhase('')
      abortRef.current = null
    }
  }

  const overall = result ? OVERALL_META[result.overall] : null
  const OverallIcon = overall?.icon

  return (
    <div>
      <PageHeader
        title="资格审查辅助"
        description="录入企业资质档案，逐条比对招标文件的资格要求，回答「我能不能投这个标」"
      />

      {/* ★ 移动端纵向堆叠（实测缺陷修复）：原来是横向 flex，
            而子元素卡片固定宽度（如 340px）且 shrink-0 不收缩，
            375px 屏上会把右侧结果区推出屏外（审计实测确认）。 */}
      <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-start lg:gap-4">
        {/* ── 左：企业资质档案 ─────────────────────────────── */}
        <Card
          className="w-full lg:w-[400px] lg:shrink-0"
          title={
            <span className="flex items-center gap-2">
              <Building2 className="h-4 w-4" />
              企业资质档案
            </span>
          }
          subtitle="录入一次，可反复用于不同项目"
        >
          <div className="space-y-3">
            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">载入已有档案</div>
              <select
                value={profileName}
                onChange={(e) => void pickProfile(e.target.value)}
                className="h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              >
                <option value="">（新建档案）</option>
                {profiles.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                    {p.company_name ? ` — ${p.company_name}` : ''}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">档案名称（唯一标识）</div>
              <input
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder="如：本公司 / 分公司A"
                className="h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              />
            </div>

            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">企业名称</div>
              <input
                value={profile.company_name ?? ''}
                onChange={(e) => setProfile({ ...profile, company_name: e.target.value })}
                placeholder="营业执照上的全称"
                className="h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              />
            </div>

            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">
                体系认证（逗号分隔）
              </div>
              <input
                value={certText}
                onChange={(e) => setCertText(e.target.value)}
                placeholder="如：ISO9001, ISO27001, CMMI3"
                className="h-8 w-full rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              />
              <div className="mt-0.5 text-[10px] text-ink-400">
                招标文件里「具有 ISO9001 认证」这类要求会与此处比对
              </div>
            </div>

            {/* 资质证书 */}
            <div>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[11px] font-medium text-ink-500">资质证书</span>
                <button
                  onClick={() =>
                    setProfile({
                      ...profile,
                      qualifications: [...(profile.qualifications ?? []), { name: '', level: '' }],
                    })
                  }
                  className="flex items-center gap-0.5 text-[11px] text-brand-600 hover:text-brand-700"
                >
                  <Plus className="h-3 w-3" />
                  添加
                </button>
              </div>
              <div className="space-y-1.5">
                {(profile.qualifications ?? []).map((q, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <input
                      value={q.name}
                      onChange={(e) => {
                        const list = [...(profile.qualifications ?? [])]
                        list[i] = { ...list[i], name: e.target.value }
                        setProfile({ ...profile, qualifications: list })
                      }}
                      placeholder="资质名称"
                      className="h-7 min-w-0 flex-1 rounded border border-slate-200 px-1.5 text-[11px] outline-none focus:border-brand-500"
                    />
                    <input
                      value={q.level ?? ''}
                      onChange={(e) => {
                        const list = [...(profile.qualifications ?? [])]
                        list[i] = { ...list[i], level: e.target.value }
                        setProfile({ ...profile, qualifications: list })
                      }}
                      placeholder="等级"
                      className="h-7 w-16 shrink-0 rounded border border-slate-200 px-1.5 text-[11px] outline-none focus:border-brand-500"
                    />
                    <button
                      onClick={() =>
                        setProfile({
                          ...profile,
                          qualifications: (profile.qualifications ?? []).filter((_, j) => j !== i),
                        })
                      }
                      className="shrink-0 rounded p-0.5 text-ink-300 hover:text-danger"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
                {(profile.qualifications ?? []).length === 0 && (
                  <div className="text-[10px] text-ink-400">如「电子与智能化工程专业承包 / 二级」</div>
                )}
              </div>
            </div>

            {/* 业绩 */}
            <div>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[11px] font-medium text-ink-500">类似业绩</span>
                <button
                  onClick={() =>
                    setProfile({
                      ...profile,
                      performance: [...(profile.performance ?? []), { project: '', amount: 0, year: 0 }],
                    })
                  }
                  className="flex items-center gap-0.5 text-[11px] text-brand-600 hover:text-brand-700"
                >
                  <Plus className="h-3 w-3" />
                  添加
                </button>
              </div>
              <div className="space-y-1.5">
                {(profile.performance ?? []).map((p, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <input
                      value={p.project}
                      onChange={(e) => {
                        const list = [...(profile.performance ?? [])]
                        list[i] = { ...list[i], project: e.target.value }
                        setProfile({ ...profile, performance: list })
                      }}
                      placeholder="项目名称"
                      className="h-7 min-w-0 flex-1 rounded border border-slate-200 px-1.5 text-[11px] outline-none focus:border-brand-500"
                    />
                    <input
                      type="number"
                      value={p.amount || ''}
                      onChange={(e) => {
                        const list = [...(profile.performance ?? [])]
                        list[i] = { ...list[i], amount: Number(e.target.value) || 0 }
                        setProfile({ ...profile, performance: list })
                      }}
                      placeholder="金额"
                      className="h-7 w-24 shrink-0 rounded border border-slate-200 px-1.5 text-[11px] outline-none focus:border-brand-500"
                    />
                    <input
                      type="number"
                      value={p.year || ''}
                      onChange={(e) => {
                        const list = [...(profile.performance ?? [])]
                        list[i] = { ...list[i], year: Number(e.target.value) || 0 }
                        setProfile({ ...profile, performance: list })
                      }}
                      placeholder="年份"
                      className="h-7 w-16 shrink-0 rounded border border-slate-200 px-1.5 text-[11px] outline-none focus:border-brand-500"
                    />
                    <button
                      onClick={() =>
                        setProfile({
                          ...profile,
                          performance: (profile.performance ?? []).filter((_, j) => j !== i),
                        })
                      }
                      className="shrink-0 rounded p-0.5 text-ink-300 hover:text-danger"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
                {(profile.performance ?? []).length === 0 && (
                  <div className="text-[10px] text-ink-400">
                    用于判定「近三年完成过不低于 X 万元的项目」这类要求
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <Button onClick={() => void save()} loading={saving} icon={<Save className="h-3.5 w-3.5" />}>
                保存档案
              </Button>
              {profileName && profiles.some((p) => p.name === profileName) && (
                <Button
                  variant="ghost"
                  onClick={() => void remove(profileName)}
                  icon={<Trash2 className="h-3.5 w-3.5" />}
                >
                  删除
                </Button>
              )}
            </div>
          </div>
        </Card>

        {/* ── 右：审查 ─────────────────────────────────────── */}
        <div className="min-w-0 flex-1">
          <Card title="开始审查" subtitle="上传招标文件，系统抽取资格要求后逐条比对当前档案">
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
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => fileRef.current?.click()}
                icon={<Upload className="h-3.5 w-3.5" />}
              >
                {file ? file.name : '选择招标文件'}
              </Button>
              <input
                type="date"
                value={refDate}
                onChange={(e) => setRefDate(e.target.value)}
                title="基准日期：用于计算「近三年」，留空取今天"
                className="h-8 rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              />
              {checking ? (
                <Button variant="secondary" onClick={() => abortRef.current?.abort()} icon={<X className="h-3.5 w-3.5" />}>
                  停止
                </Button>
              ) : (
                <Button
                  onClick={() => void runCheck()}
                  disabled={!file}
                  icon={<BadgeCheck className="h-3.5 w-3.5" />}
                >
                  开始审查
                </Button>
              )}
              {!profileName && (
                <span className="text-[11px] text-warn">
                  未选档案，将用当前编辑中的内容比对
                </span>
              )}
            </div>
            <div className="mt-1.5 text-[10px] text-ink-400">
              基准日期用于推算「近三年」等时间窗口；留空取今天
            </div>

            {checking && (
              <div className="mt-3 flex items-center gap-2 rounded-lg border border-brand-100 bg-brand-50 p-3 text-[12px] text-ink-700">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                {phase || '处理中…'}
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

          {!result && !checking && (
            <Card className="mt-4">
              <EmptyState
                icon={<ShieldQuestion className="h-6 w-6" />}
                title="填写档案并上传招标文件后开始审查"
                description="系统会抽取资格要求章节，逐条与档案比对，给出「满足 / 不满足 / 需人工确认」三值判定"
              />
            </Card>
          )}

          {result && overall && OverallIcon && (
            <>
              {/* 结论置顶 */}
              <div
                className={`mt-4 rounded-xl border-l-4 p-4 ${
                  result.overall === 'unqualified'
                    ? 'border-danger bg-red-50'
                    : result.overall === 'need_review'
                      ? 'border-amber-400 bg-amber-50'
                      : 'border-emerald-400 bg-emerald-50'
                }`}
              >
                <div className="flex items-start gap-2.5">
                  <OverallIcon
                    className={`mt-0.5 h-5 w-5 shrink-0 ${
                      result.overall === 'unqualified'
                        ? 'text-danger'
                        : result.overall === 'need_review'
                          ? 'text-warn'
                          : 'text-emerald-600'
                    }`}
                  />
                  <div className="min-w-0">
                    <div className="text-[15px] font-bold text-ink-900">{overall.label}</div>
                    <div className="mt-1 text-[12px] leading-5 text-ink-700">
                      {result.summary.conclusion}
                    </div>
                  </div>
                </div>
              </div>

              {/* 统计 */}
              <div className="mt-4 grid grid-cols-4 gap-3">
                {(
                  [
                    ['total', '要求总数', result.summary.total, 'text-ink-900'],
                    ['met', '满足', result.summary.met, 'text-emerald-600'],
                    ['unmet', '不满足', result.summary.unmet, 'text-danger'],
                    ['uncertain', '需人工确认', result.summary.uncertain, 'text-warn'],
                  ] as const
                ).map(([k, label, count, cls]) => (
                  <div key={k} className="rounded-xl border border-slate-200 bg-white p-3">
                    <div className="text-[11px] text-ink-500">{label}</div>
                    <div className={`mt-0.5 text-xl font-bold tabular-nums ${cls}`}>{count}</div>
                  </div>
                ))}
              </div>

              {/* 缺口 */}
              {result.gaps.length > 0 && (
                <Card
                  className="mt-4"
                  title={`硬性要求缺口（${result.gaps.length} 项）`}
                  subtitle="这些是必须解决的门槛——不满足即无法投标"
                >
                  <div className="space-y-2">
                    {result.gaps.map((g) => (
                      <RequirementRow key={g.index} req={g} />
                    ))}
                  </div>
                </Card>
              )}

              {/* 逐条判定 */}
              <Card
                className="mt-4"
                title={`逐条判定（${result.requirements.length} 条）`}
                subtitle="按判定结果分组：不满足 → 需人工确认 → 满足"
              >
                <div className="space-y-2">
                  {[...result.requirements]
                    .sort((a, b) => {
                      const order = { unmet: 0, uncertain: 1, met: 2 }
                      return (order[a.status] ?? 3) - (order[b.status] ?? 3)
                    })
                    .map((r) => (
                      <RequirementRow key={r.index} req={r} />
                    ))}
                </div>
              </Card>

              {/* 局限说明 */}
              <Card className="mt-4" title="结论说明" subtitle="这些限制会影响结论的适用范围">
                <ul className="space-y-1.5">
                  {result.summary.caveats.map((c, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-[11px] leading-5 text-ink-600">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
