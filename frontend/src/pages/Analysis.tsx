/**
 * 集采分析页。
 *
 * 数据来源：`/api/analysis/*`（PostgreSQL 真实聚合）。
 *
 * 与早期版本的差别：早期用 `data/analysis.ts` 的 mock 图表，还画了一条
 * **编造的"近 12 个月中标金额趋势"曲线**。本版接真实数据后，那条曲线被删掉了——
 * 因为实测数据只覆盖 **2023-10 一个月**（8746 条全部落在 10/13~10/20），
 * 画趋势图等于凭空造 11 个月的数据。
 *
 * 三个必须坚持的展示原则
 * ----------------------
 * 1. **中位数优先**：金额跨 6 个数量级（几千元 ↔ 数亿元），均值是中位数的 500 倍。
 *    页面把中位数做成主指标，均值只作对照并标注"不具代表性"。
 * 2. **展示数据局限**：`data_caveats`（单月数据 / 均值不可用 / category 语义混杂）
 *    和 `notes`（字段空值率）直接摆在页面上，不藏起来。
 * 3. **可疑记录可追溯**：被判可疑的金额记录逐条列出原因，用户能自行判断是否认同。
 */
import { useCallback, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Tooltip, XAxis, YAxis } from 'recharts'
import {
  AlertTriangle,
  Building2,
  Calculator,
  Database,
  FileSearch,
  Info,
  Layers,
  MapPin,
  Search,
  Swords,
  TrendingUp,
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
  fetchAnalysisDimensions,
  fetchAnalysisDistribution,
  fetchAnalysisOverview,
  fetchAnalysisRanking,
  fetchCompetitorSuppliers,
  fetchDataQuality,
  fetchMarketConcentration,
  fetchPriceProfile,
  type AnalysisOverview,
  type CompetitorConcentrationResult,
  type CompetitorSuppliersResult,
  type DataQuality,
  type DimensionOption,
  type DistributionResult,
  type PriceProfile,
  type RankingResult,
} from '../lib/api'
import { BAR_COLOR, LOW_COLOR, WARN_COLOR, tooltipStyle } from '../lib/chartStyles'



/** 金额紧凑展示：跨 6 个数量级，必须自适应单位 */
function money(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  const n = Math.abs(v)
  if (n >= 100_000_000) return `¥${(v / 100_000_000).toFixed(2)}亿`
  if (n >= 10_000) return `¥${(v / 10_000).toFixed(1)}万`
  return `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
}

function moneyFull(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 2 })}`
}

type Tab = 'overview' | 'ranking' | 'pricing' | 'competitor' | 'quality'

const TABS: { key: Tab; label: string; icon: typeof Layers }[] = [
  { key: 'overview', label: '总览', icon: Layers },
  { key: 'ranking', label: '排行分析', icon: TrendingUp },
  { key: 'pricing', label: '报价策略', icon: Calculator },
  { key: 'competitor', label: '竞争对手', icon: Swords },
  { key: 'quality', label: '数据质量', icon: Database },
]

export default function Analysis() {
  const toast = useToast()
  const [tab, setTab] = useState<Tab>('overview')
  const [excludeSuspicious, setExcludeSuspicious] = useState(true)
  const [loading, setLoading] = useState(true)

  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [dimensions, setDimensions] = useState<DimensionOption[]>([])
  const [distribution, setDistribution] = useState<DistributionResult | null>(null)
  const [quality, setQuality] = useState<DataQuality | null>(null)

  const [dimension, setDimension] = useState('subject_matter')
  const [orderBy, setOrderBy] = useState<'count' | 'amount'>('count')
  const [ranking, setRanking] = useState<RankingResult | null>(null)
  const [rankingLoading, setRankingLoading] = useState(false)

  const [subject, setSubject] = useState('空调')
  const [profile, setProfile] = useState<PriceProfile | null>(null)
  const [profileLoading, setProfileLoading] = useState(false)

  // 竞争对手分析（与报价策略共用 subject 输入，便于对照看同一标的物）
  const [suppliers, setSuppliers] = useState<CompetitorSuppliersResult | null>(null)
  const [concentration, setConcentration] = useState<CompetitorConcentrationResult | null>(null)
  const [competitorLoading, setCompetitorLoading] = useState(false)

  const describe = (err: unknown) =>
    err instanceof ApiError ? `请求失败：${err.message}` : '网络异常，请检查后端服务'

  const loadBase = useCallback(async () => {
    setLoading(true)
    try {
      const [ov, dims, dist, q] = await Promise.all([
        fetchAnalysisOverview(excludeSuspicious),
        fetchAnalysisDimensions(),
        fetchAnalysisDistribution(excludeSuspicious),
        fetchDataQuality(),
      ])
      setOverview(ov)
      setDimensions(dims)
      setDistribution(dist)
      setQuality(q)
      if (dims.length && !dims.some((d) => d.key === dimension)) {
        setDimension(dims[0].key)
      }
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [excludeSuspicious, toast])

  useEffect(() => {
    void loadBase()
  }, [loadBase])

  const loadRanking = useCallback(async () => {
    setRankingLoading(true)
    try {
      setRanking(await fetchAnalysisRanking({ dimension, limit: 20, orderBy, excludeSuspicious }))
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setRankingLoading(false)
    }
  }, [dimension, orderBy, excludeSuspicious, toast])

  useEffect(() => {
    if (tab === 'ranking') void loadRanking()
  }, [tab, loadRanking])

  const runProfile = useCallback(async () => {
    if (!subject.trim()) {
      toast.show('请输入标的物关键词', 'warning')
      return
    }
    setProfileLoading(true)
    setProfile(null)
    try {
      setProfile(await fetchPriceProfile({ subject: subject.trim(), excludeSuspicious, topN: 10 }))
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setProfileLoading(false)
    }
  }, [subject, excludeSuspicious, toast])

  const runCompetitor = useCallback(async () => {
    const s = subject.trim()
    if (!s) {
      toast.show('请输入标的物关键词', 'warning')
      return
    }
    setCompetitorLoading(true)
    setSuppliers(null)
    setConcentration(null)
    try {
      // 两个接口并行：它们查的是同一批数据，串行会让用户多等一倍
      const [sup, con] = await Promise.all([
        fetchCompetitorSuppliers({ subject: s, limit: 20, excludeSuspicious }),
        fetchMarketConcentration(s),
      ])
      setSuppliers(sup)
      setConcentration(con)
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setCompetitorLoading(false)
    }
  }, [subject, excludeSuspicious, toast])

  const kpis = overview
    ? [
        {
          label: '标讯记录',
          value: overview.total_rows.toLocaleString('zh-CN'),
          icon: FileSearch,
          cls: 'bg-brand-50 text-brand-600',
          note: '业务库总行数',
        },
        {
          label: '中标金额中位数',
          value: money(overview.amount_stats.median),
          icon: TrendingUp,
          cls: 'bg-emerald-50 text-ok',
          note: `均值 ${money(overview.amount_stats.mean)}（偏态，不具代表性）`,
        },
        {
          label: '标的物种类',
          value: (overview.dimension_cardinality.subject_matter ?? 0).toLocaleString('zh-CN'),
          icon: Layers,
          cls: 'bg-amber-50 text-warn',
          note: `采购人 ${overview.dimension_cardinality.purchaser ?? 0} / 供应商 ${overview.dimension_cardinality.supplier ?? 0}`,
        },
        {
          label: '覆盖地区',
          value: `${overview.dimension_cardinality.province ?? 0} 省`,
          icon: MapPin,
          cls: 'bg-red-50 text-danger',
          note: `${overview.dimension_cardinality.location_city ?? 0} 个市区`,
        },
      ]
    : []

  return (
    <div>
      <PageHeader
        title="集采分析"
        description="基于业务库真实中标记录的统计分析与报价策略"
        actions={
          <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-ink-600">
            <input
              type="checkbox"
              checked={excludeSuspicious}
              onChange={(e) => setExcludeSuspicious(e.target.checked)}
              className="h-3.5 w-3.5 accent-brand-600"
            />
            排除可疑金额记录
          </label>
        }
      />

      {/* 标签页 */}
      <div className="mb-4 flex gap-1 border-b border-slate-200">
        {TABS.map((t) => {
          const Icon = t.icon
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-[13px] font-medium transition-colors ${
                tab === t.key
                  ? 'border-brand-600 text-brand-700'
                  : 'border-transparent text-ink-400 hover:text-ink-700'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {t.label}
            </button>
          )
        })}
      </div>

      {loading && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
          <Skeleton className="h-72 w-full" />
        </div>
      )}

      {/* ── 总览 ─────────────────────────────────────────────── */}
      {!loading && tab === 'overview' && overview && (
        <>
          <div className="grid grid-cols-4 gap-4">
            {kpis.map((k) => {
              const Icon = k.icon
              return (
                <div key={k.label} className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-ink-500">{k.label}</span>
                    <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${k.cls}`}>
                      <Icon className="h-3.5 w-3.5" />
                    </span>
                  </div>
                  <div className="mt-1.5 text-xl font-bold tabular-nums text-ink-900">{k.value}</div>
                  <div className="mt-0.5 text-[11px] text-ink-400">{k.note}</div>
                </div>
              )
            })}
          </div>

          {/* 数据局限提示：必须在显眼位置 */}
          {overview.data_caveats.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
              <div className="flex items-center gap-2 text-[13px] font-semibold text-warn">
                <AlertTriangle className="h-4 w-4" />
                使用本页数据前请知悉
              </div>
              <ul className="mt-2 space-y-1">
                {overview.data_caveats.map((c, i) => (
                  <li key={i} className="flex gap-2 text-[12px] leading-5 text-ink-700">
                    <span className="text-warn">•</span>
                    <span dangerouslySetInnerHTML={{ __html: c.replace(/\*\*/g, '') }} />
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 金额分位数 */}
          <Card
            className="mt-4"
            title="中标金额分位数"
            subtitle="本数据集金额跨 6 个数量级，请以中位数判断典型水平"
          >
            <div className="grid grid-cols-7 gap-3">
              {[
                ['最小', overview.amount_stats.min],
                ['P25', overview.amount_stats.p25],
                ['中位数', overview.amount_stats.median],
                ['P75', overview.amount_stats.p75],
                ['P90', overview.amount_stats.p90],
                ['最大', overview.amount_stats.max],
                ['均值', overview.amount_stats.mean],
              ].map(([label, value]) => {
                const isMedian = label === '中位数'
                const isMean = label === '均值'
                return (
                  <div
                    key={label as string}
                    className={`rounded-lg border p-3 text-center ${
                      isMedian
                        ? 'border-emerald-200 bg-emerald-50'
                        : isMean
                          ? 'border-amber-200 bg-amber-50'
                          : 'border-slate-200 bg-white'
                    }`}
                  >
                    <div className="text-[11px] text-ink-500">
                      {label as string}
                      {isMedian && ' ★'}
                    </div>
                    <div
                      className={`mt-1 text-[15px] font-bold tabular-nums ${
                        isMedian ? 'text-ok' : isMean ? 'text-warn' : 'text-ink-900'
                      }`}
                      title={moneyFull(value as number | null)}
                    >
                      {money(value as number | null)}
                    </div>
                    {isMean && <div className="mt-0.5 text-[10px] text-warn">偏态参考</div>}
                    {isMedian && <div className="mt-0.5 text-[10px] text-ok">推荐使用</div>}
                  </div>
                )
              })}
            </div>
            <div className="mt-3 text-[11px] text-ink-400">
              参与统计 {overview.amount_stats.count.toLocaleString('zh-CN')} 条
              {overview.suspicious_count > 0 && (
                <span className="text-warn">
                  ｜已排除 {overview.suspicious_count} 条可疑金额记录（详见「数据质量」页）
                </span>
              )}
            </div>
          </Card>

          {/* 数量级分布 */}
          {distribution && (
            <Card
              className="mt-4"
              title="金额数量级分布"
              subtitle={distribution.note}
            >
              <BarChart width={980} height={280} data={distribution.buckets}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="label"
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tick={{ fontSize: 11, fill: '#64748B' }}
                />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11, fill: '#94A3B8' }}
                  width={50}
                />
                <Tooltip
                  formatter={(v) => [`${Number(v).toLocaleString('zh-CN')} 条`, '记录数']}
                  contentStyle={tooltipStyle}
                  cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} barSize={48} isAnimationActive={false}>
                  {distribution.buckets.map((b, i) => (
                    <Cell
                      key={b.label}
                      fill={b.label.includes('≥1亿') ? WARN_COLOR : i === 1 ? LOW_COLOR : BAR_COLOR}
                    />
                  ))}
                  <LabelList
                    dataKey="count"
                    position="top"
                    style={{ fontSize: 11, fill: '#334155' }}
                  />
                </Bar>
              </BarChart>
            </Card>
          )}
        </>
      )}

      {/* ── 排行分析 ─────────────────────────────────────────── */}
      {!loading && tab === 'ranking' && (
        <Card
          title="维度排行"
          subtitle="每组给出中位数（组内同样可能有超大额记录）"
          extra={
            <div className="flex items-center gap-2">
              <select
                value={dimension}
                onChange={(e) => setDimension(e.target.value)}
                className="h-8 rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              >
                {dimensions.map((d) => (
                  <option key={d.key} value={d.key}>
                    {d.label}
                  </option>
                ))}
              </select>
              <select
                value={orderBy}
                onChange={(e) => setOrderBy(e.target.value as 'count' | 'amount')}
                className="h-8 rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
              >
                <option value="count">按次数</option>
                <option value="amount">按合计金额</option>
              </select>
            </div>
          }
        >
          {rankingLoading && (
            <div className="space-y-2">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-7 w-full" />
              ))}
            </div>
          )}
          {!rankingLoading && ranking && (
            <>
              <div className="mb-2 text-[11px] text-ink-400">
                共 {ranking.distinct_total.toLocaleString('zh-CN')} 个不同的
                {ranking.dimension_label}，显示前 {ranking.returned} 个
              </div>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                    <th className="w-10 py-2">#</th>
                    <th className="py-2">{ranking.dimension_label}</th>
                    <th className="w-20 py-2 text-right">次数</th>
                    <th className="w-28 py-2 text-right">金额中位数</th>
                    <th className="w-28 py-2 text-right">合计金额</th>
                    <th className="w-24 py-2 text-right">最大单笔</th>
                  </tr>
                </thead>
                <tbody>
                  {ranking.items.map((it, i) => (
                    <tr key={it.name} className="border-b border-slate-100 last:border-0">
                      <td className="py-2 text-[11px] text-ink-300">{i + 1}</td>
                      <td className="py-2 text-ink-900">{it.name}</td>
                      <td className="py-2 text-right tabular-nums">{it.count}</td>
                      <td className="py-2 text-right font-medium tabular-nums text-ink-900">
                        {money(it.median_amount)}
                      </td>
                      <td className="py-2 text-right tabular-nums text-ink-600">
                        {money(it.total_amount)}
                      </td>
                      <td className="py-2 text-right tabular-nums text-ink-400">
                        {money(it.max_amount)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </Card>
      )}

      {/* ── 报价策略 ─────────────────────────────────────────── */}
      {!loading && tab === 'pricing' && (
        <>
          <Card
            title="报价策略分析"
            subtitle="输入标的物关键词，基于历史中标金额给出建议报价区间"
          >
            <div className="flex items-center gap-3">
              <input
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void runProfile()}
                placeholder="如：空调 / 台式整机 / 复印纸"
                className="h-9 flex-1 rounded-lg border border-slate-200 px-3 text-[13px] outline-none focus:border-brand-500"
              />
              <Button
                onClick={() => void runProfile()}
                loading={profileLoading}
                icon={<Search className="h-3.5 w-3.5" />}
              >
                分析
              </Button>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {['空调', '台式整机', '复印纸', '粉盒', '办公椅', '文件柜'].map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setSubject(s)
                  }}
                  className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] text-ink-500 hover:border-brand-300 hover:bg-brand-50"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>

          {profileLoading && <Skeleton className="mt-4 h-64 w-full" />}

          {!profileLoading && profile && !profile.matched && (
            <Card className="mt-4">
              <EmptyState
                icon={<Search className="h-6 w-6" />}
                title="未找到匹配记录"
                description={profile.message ?? '请换个关键词试试'}
              />
            </Card>
          )}

          {!profileLoading && profile && profile.matched && (
            <>
              <Card
                className="mt-4"
                title={`「${profile.subject}」价格画像`}
                subtitle={`匹配 ${profile.count} 条历史中标记录`}
                extra={
                  profile.suggested_range && (
                    <Badge color="ok">
                      建议区间 {money(profile.suggested_range.low)} ~ {money(profile.suggested_range.high)}
                    </Badge>
                  )
                }
              >
                {/* 分位数条 */}
                <div className="grid grid-cols-5 gap-3">
                  {[
                    ['P10', profile.percentiles?.p10],
                    ['P25', profile.percentiles?.p25],
                    ['中位数', profile.percentiles?.median],
                    ['P75', profile.percentiles?.p75],
                    ['P90', profile.percentiles?.p90],
                  ].map(([label, value]) => {
                    const inRange = label === 'P25' || label === '中位数' || label === 'P75'
                    return (
                      <div
                        key={label as string}
                        className={`rounded-lg border p-3 text-center ${
                          inRange ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200'
                        }`}
                      >
                        <div className="text-[11px] text-ink-500">{label as string}</div>
                        <div
                          className={`mt-1 text-[15px] font-bold tabular-nums ${
                            inRange ? 'text-ok' : 'text-ink-900'
                          }`}
                        >
                          {money(value as number | null)}
                        </div>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-2 text-[11px] text-ink-400">
                  绿底区间（P25~P75）为建议报价区间｜最小 {money(profile.min)}｜最大 {money(profile.max)}｜均值{' '}
                  <span className="text-warn">{money(profile.mean)}</span>
                </div>

                {profile.advice && profile.advice.length > 0 && (
                  <div className="mt-3 rounded-lg border border-brand-100 bg-brand-50 p-3">
                    <div className="text-[12px] font-semibold text-brand-700">报价建议</div>
                    <ul className="mt-1.5 space-y-1">
                      {profile.advice.map((a, i) => (
                        <li
                          key={i}
                          className="text-[12px] leading-5 text-ink-700"
                          dangerouslySetInnerHTML={{ __html: '• ' + a.replace(/\*\*/g, '') }}
                        />
                      ))}
                    </ul>
                  </div>
                )}
              </Card>

              <div className="mt-4 grid grid-cols-2 gap-4">
                {profile.top_suppliers && profile.top_suppliers.length > 0 && (
                  <Card title="主要竞争对手" subtitle="按该标的物中标次数">
                    <table className="w-full text-[13px]">
                      <tbody>
                        {profile.top_suppliers.map((s) => (
                          <tr key={s.name} className="border-b border-slate-100 last:border-0">
                            <td className="py-1.5 text-ink-900">{s.name}</td>
                            <td className="w-16 py-1.5 text-right tabular-nums text-ink-500">
                              {s.count} 次
                            </td>
                            <td className="w-24 py-1.5 text-right tabular-nums text-ink-600">
                              {money(s.median_amount)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Card>
                )}
                <Card title="金额最高的案例" subtitle="可用于了解大额项目的特征">
                  <div className="space-y-2">
                    {profile.top_records?.slice(0, 5).map((r, i) => (
                      <div key={i} className="border-b border-slate-100 pb-1.5 last:border-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="min-w-0 flex-1 truncate text-[12px] text-ink-900" title={r.project_name}>
                            {r.project_name}
                          </span>
                          <span className="shrink-0 text-[12px] font-medium tabular-nums text-ink-900">
                            {money(r.amount)}
                          </span>
                        </div>
                        <div className="mt-0.5 truncate text-[11px] text-ink-400">
                          {r.purchaser}
                          {r.supplier && ` ← ${r.supplier}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </>
          )}
        </>
      )}

      {/* ── 竞争对手 ─────────────────────────────────────────── */}
      {!loading && tab === 'competitor' && (
        <>
          <Card
            title="竞争对手分析"
            subtitle="某标的物的供应商格局与市场集中度（聚合统计来自业务库逐条记录）"
          >
            <div className="flex items-center gap-3">
              <input
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void runCompetitor()}
                placeholder="如：空调 / 台式整机 / 复印纸"
                className="h-9 flex-1 rounded-lg border border-slate-200 px-3 text-[13px] outline-none focus:border-brand-500"
              />
              <Button
                onClick={() => void runCompetitor()}
                loading={competitorLoading}
                icon={<Swords className="h-3.5 w-3.5" />}
              >
                分析
              </Button>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {['空调', '台式整机', '复印纸', '粉盒', '硒鼓'].map((s) => (
                <button
                  key={s}
                  onClick={() => setSubject(s)}
                  className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] text-ink-500 hover:border-brand-300 hover:bg-brand-50"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>

          {competitorLoading && <Skeleton className="mt-4 h-72 w-full" />}

          {!competitorLoading && suppliers && !suppliers.matched && (
            <Card className="mt-4">
              <EmptyState
                icon={<Swords className="h-6 w-6" />}
                title="未找到匹配记录"
                description={suppliers.message ?? '请换个关键词试试'}
              />
            </Card>
          )}

          {!competitorLoading && suppliers && suppliers.matched && (
            <>
              {/* 市场概况 + 集中度 */}
              <div className="mt-4 grid grid-cols-4 gap-4">
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="text-[11px] text-ink-500">供应商数量</div>
                  <div className="mt-1 text-xl font-bold tabular-nums text-ink-900">
                    {suppliers.distinct_suppliers}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-400">
                    共 {suppliers.market?.count ?? 0} 条中标记录
                  </div>
                </div>
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                  <div className="text-[11px] text-ink-500">金额中位数</div>
                  <div className="mt-1 text-xl font-bold tabular-nums text-ok">
                    {money(suppliers.market?.median_amount)}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-400">
                    P25 {money(suppliers.market?.p25)} ~ P75 {money(suppliers.market?.p75)}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="text-[11px] text-ink-500">金额合计</div>
                  <div className="mt-1 text-xl font-bold tabular-nums text-ink-900">
                    {money(suppliers.market?.total_amount)}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-400">全部中标记录汇总</div>
                </div>
                {concentration?.by_amount && (
                  <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
                    <div className="text-[11px] text-ink-500">市场集中度 HHI</div>
                    <div className="mt-1 text-xl font-bold tabular-nums text-brand-700">
                      {concentration.by_amount.hhi.toFixed(1)}
                    </div>
                    <div className="mt-0.5 text-[10px] text-ink-400">
                      {concentration.by_amount.hhi_level}｜CR3{' '}
                      {concentration.by_amount.cr3_percent.toFixed(1)}%｜CR5{' '}
                      {concentration.by_amount.cr5_percent.toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>

              {/* 两种口径对照 + HHI 解读 */}
              {concentration?.hhi_explanation && (
                <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                  <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-600">
                    <Info className="h-3.5 w-3.5" />
                    HHI 如何解读
                  </div>
                  <div className="mt-1.5 font-mono text-[11px] text-ink-500">
                    {concentration.hhi_explanation.formula}
                  </div>
                  <div className="mt-2 space-y-1">
                    {concentration.hhi_explanation.thresholds.map((t) => (
                      <div
                        key={t.level}
                        className="flex flex-wrap items-baseline gap-2 text-[11px] text-ink-600"
                      >
                        <span className="w-24 shrink-0 font-mono text-ink-500">{t.range}</span>
                        <span className="w-20 shrink-0 font-medium text-ink-900">{t.level}</span>
                        <span className="text-ink-500">{t.meaning}</span>
                      </div>
                    ))}
                  </div>

                  {/* 两种口径可能得出不同结论，必须并排展示 */}
                  {concentration.by_amount && concentration.by_count && (
                    <div className="mt-3 grid grid-cols-2 gap-3 border-t border-slate-100 pt-3">
                      {[concentration.by_amount, concentration.by_count].map((b) => (
                        <div key={b.basis} className="rounded-lg border border-slate-100 bg-slate-50 p-2.5">
                          <div className="text-[11px] leading-4 text-ink-500">{b.basis}</div>
                          <div className="mt-1 flex items-baseline gap-2">
                            <span className="text-[15px] font-bold tabular-nums text-ink-900">
                              {b.hhi.toFixed(1)}
                            </span>
                            <Badge color="neutral">{b.hhi_level}</Badge>
                          </div>
                          <div className="mt-0.5 text-[10px] tabular-nums text-ink-400">
                            CR3 {b.cr3_percent.toFixed(1)}%｜CR5 {b.cr5_percent.toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-2 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                    <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                    <span className="text-[11px] leading-5 text-ink-600">
                      {concentration.hhi_explanation.caveat}
                    </span>
                  </div>
                </div>
              )}

              {/* 供应商排行 */}
              <Card
                className="mt-4"
                title="供应商竞争格局"
                subtitle="按中标次数排序；价格为中位数，与均值并排以便识别偏态"
              >
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                      <th className="w-10 py-2">#</th>
                      <th className="py-2">供应商</th>
                      <th className="w-16 py-2 text-right">中标次数</th>
                      <th className="w-24 py-2 text-right">金额中位数</th>
                      <th className="w-24 py-2 text-right">合计金额</th>
                      <th className="w-28 py-2 text-right">首次 / 最近</th>
                      <th className="w-28 py-2">主要地区</th>
                    </tr>
                  </thead>
                  <tbody>
                    {suppliers.suppliers.map((s, i) => (
                      <tr key={s.name} className="border-b border-slate-100 last:border-0">
                        <td className="py-1.5 text-[11px] text-ink-300">{i + 1}</td>
                        <td className="py-1.5 text-ink-900">{s.name}</td>
                        <td className="py-1.5 text-right tabular-nums font-medium text-ink-900">
                          {s.count}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink-900">
                          {money(s.median_amount)}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink-600">
                          {money(s.total_amount)}
                        </td>
                        <td className="py-1.5 text-right text-[11px] text-ink-400">
                          {s.first_win?.slice(5, 10)} / {s.last_win?.slice(5, 10)}
                        </td>
                        <td className="py-1.5 text-[11px] text-ink-500">
                          {(s.regions ?? []).slice(0, 2).join('、') || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              {/* 覆盖范围与数据局限 */}
              {((suppliers.data_caveats ?? []).length > 0 || suppliers.graph_note) && (
                <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
                  <div className="flex items-center gap-1.5 text-[12px] font-semibold text-warn">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    使用前请知悉
                  </div>
                  <ul className="mt-2 space-y-1">
                    {(suppliers.data_caveats ?? []).map((c, i) => (
                      <li key={i} className="text-[11px] leading-5 text-ink-700">
                        • {c.replace(/\*\*/g, '')}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-2 border-t border-amber-200 pt-2 text-[11px] leading-5 text-ink-500">
                    <b>为什么聚合统计不走知识图谱：</b>图谱中{' '}
                    <code className="rounded bg-white/70 px-1">SUPPLIED_BY</code>{' '}
                    是<b>无权重边</b>，金额只挂在「标的物→交易」上，
                    而交易节点与供应商之间没有关系——所以图谱<b>无法</b>
                    把某笔金额归到某个供应商头上。统计因此来自业务库的逐条记录，
                    图谱只作补充（补充失败会自动降级，不影响统计结果）。
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── 数据质量 ─────────────────────────────────────────── */}
      {!loading && tab === 'quality' && quality && (
        <>
          <Card
            title="字段空值率"
            subtitle="空值率高的字段在做分析时不可依赖"
          >
            <table className="w-full text-[13px]">
              <tbody>
                {quality.empty_rates.slice(0, 12).map((e) => (
                  <tr key={e.column} className="border-b border-slate-100 last:border-0">
                    <td className="w-40 py-1.5 font-mono text-[12px] text-ink-700">{e.column}</td>
                    <td className="py-1.5">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-48 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className={`h-full ${e.empty_percent > 50 ? 'bg-danger' : 'bg-warn'}`}
                            style={{ width: `${Math.min(100, e.empty_percent)}%` }}
                          />
                        </div>
                        <span className="text-[11px] tabular-nums text-ink-500">
                          {e.empty_percent.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="w-24 py-1.5 text-right text-[11px] tabular-nums text-ink-400">
                      {e.empty_count.toLocaleString('zh-CN')} 条
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-3 rounded-lg border border-slate-200 bg-ink-50/60 p-3">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-500">
                <Info className="h-3 w-3" />
                说明
              </div>
              <ul className="mt-1 space-y-0.5">
                {quality.notes.map((n, i) => (
                  <li key={i} className="text-[11px] leading-5 text-ink-500">
                    • {n}
                  </li>
                ))}
              </ul>
            </div>
          </Card>

          <Card
            className="mt-4"
            title={`可疑金额记录（${quality.suspicious_count} 条）`}
            subtitle="采用「金额 + 品类」组合规则判定，逐条给出原因，可自行判断是否认同"
          >
            {quality.suspicious_records.length === 0 ? (
              <EmptyState
                icon={<Database className="h-6 w-6" />}
                title="未发现可疑记录"
                description="当前数据通过全部校验规则"
              />
            ) : (
              <div className="space-y-3">
                {quality.suspicious_records.map((r) => (
                  <div key={r.id} className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="text-[13px] font-medium text-ink-900">
                          {r.subject_matter}
                        </div>
                        <div className="mt-0.5 text-[11px] text-ink-500">
                          id={r.id}｜{r.purchaser}
                          {r.supplier && ` ← ${r.supplier}`}
                          {r.category && `｜类别：${r.category}`}
                        </div>
                      </div>
                      <div className="shrink-0 text-right">
                        <div className="text-[15px] font-bold tabular-nums text-warn">
                          {money(r.amount)}
                        </div>
                        <div className="text-[10px] text-ink-400">{moneyFull(r.amount)}</div>
                      </div>
                    </div>
                    <div className="mt-2 flex gap-1.5 text-[11px] leading-5 text-warn">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                      <span>{r.reason}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card className="mt-4" title="category 字段的实际取值" subtitle="该字段语义混杂，使用前需自行归类">
            <div className="flex flex-wrap gap-2">
              {quality.category_values.map((c) => (
                <span
                  key={c.name}
                  className="rounded-lg border border-slate-200 px-2.5 py-1 text-[12px] text-ink-600"
                >
                  {c.name}
                  <span className="ml-1.5 text-[11px] tabular-nums text-ink-400">{c.count}</span>
                </span>
              ))}
            </div>
            <div className="mt-3 flex items-center gap-2 text-[11px] text-ink-400">
              <Building2 className="h-3 w-3" />
              含「学校/政府/医院」（采购主体）与「货物/工程/服务」（采购品类）两类语义
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
