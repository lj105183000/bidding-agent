/**
 * 物料智能识别页。
 *
 * 数据来源：`/api/material/*`（PostgreSQL 真实物料库）
 *
 * 与早期版本的差别：早期用 `data/material.ts` 的 mock，演示的是
 * **「一物多码跨系统」**场景——同一个物料在 ERP/MES/SRM 三个系统里写法不同，
 * 还带 `stdCode`（物料编码）、`categoryPath`（分类路径）、`brand`（品牌）。
 *
 * 但实测真实数据**没有**这几个字段：`bidding_records` 只有 `subject_matter`
 * 一列文本，既无物料编码也无分类与品牌。真实场景是
 * **「同一数据库内同一物料的多种写法需要归并」**，例如：
 *
 *     打印/复印纸的网上超市采购 (160)  ┐
 *     打印/复印纸                (88)  ┴→ 标准物料「打印/复印纸」(250)
 *     空调 (175) / 空调5件 (2) / 空调的网上超市采购 (1) → 「空调」(179)
 *
 * 故本页按真实能力重写：**匹配 → 归并 → 标准化建议**，
 * 并新增「标准化建议」标签页——那是人工确认归并规则的工作台。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  GitMerge,
  Info,
  Layers,
  Loader2,
  Package,
  Search,
  Sparkles,
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
  fetchMaterialDetail,
  fetchMaterialStats,
  fetchMaterialSuggestions,
  matchMaterials,
  type MaterialDetailResult,
  type MaterialMatchItem,
  type MaterialStatsResult,
  type MaterialSuggestionsResult,
} from '../lib/api'

function money(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  const n = Math.abs(v)
  if (n >= 100_000_000) return `¥${(v / 100_000_000).toFixed(2)}亿`
  if (n >= 10_000) return `¥${(v / 10_000).toFixed(1)}万`
  return `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
}

/** 匹配得分 → 颜色档位 */
function scoreColor(score: number): 'ok' | 'brand' | 'warn' | 'neutral' {
  if (score >= 0.95) return 'ok'
  if (score >= 0.75) return 'brand'
  if (score >= 0.5) return 'warn'
  return 'neutral'
}

type Tab = 'match' | 'library' | 'suggestions'

const TABS: { key: Tab; label: string; icon: typeof Search }[] = [
  { key: 'match', label: '物料匹配', icon: Search },
  { key: 'library', label: '物料库概览', icon: Layers },
  { key: 'suggestions', label: '标准化建议', icon: GitMerge },
]

export default function Material() {
  const toast = useToast()
  const debounceRef = useRef<number>()

  const [tab, setTab] = useState<Tab>('match')
  const [query, setQuery] = useState('空调')
  const [matches, setMatches] = useState<MaterialMatchItem[]>([])
  const [matchMeta, setMatchMeta] = useState<{ normalized: string; total: number } | null>(null)
  const [matching, setMatching] = useState(false)

  const [detail, setDetail] = useState<MaterialDetailResult | null>(null)
  const [detailName, setDetailName] = useState('')

  const [stats, setStats] = useState<MaterialStatsResult | null>(null)
  const [suggestions, setSuggestions] = useState<MaterialSuggestionsResult | null>(null)
  const [loadingTab, setLoadingTab] = useState(false)

  const describe = (err: unknown) =>
    err instanceof ApiError
      ? err.status === 400
        ? err.message
        : err.status === 503
          ? '物料服务不可用（请检查 PostgreSQL 是否启动）'
          : `请求失败：${err.message}`
      : '网络异常，请检查后端服务'

  const doMatch = useCallback(
    async (q: string) => {
      const clean = q.trim()
      if (!clean) return
      setMatching(true)
      setDetail(null)
      try {
        const res = await matchMaterials(clean, 10)
        setMatches(res.items)
        setMatchMeta({ normalized: res.normalized_query, total: res.total_candidates })
        if (res.items.length === 0) {
          toast.show('未匹配到物料，试试更短的关键词', 'warning')
        }
      } catch (err) {
        toast.show(describe(err), 'error')
      } finally {
        setMatching(false)
      }
    },
    [toast],
  )

  // 初次进入自动跑一次
  useEffect(() => {
    void doMatch('空调')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 切到概览/建议页时才拉数据
  useEffect(() => {
    if (tab === 'library' && !stats) {
      setLoadingTab(true)
      fetchMaterialStats(20)
        .then(setStats)
        .catch((err) => toast.show(describe(err), 'error'))
        .finally(() => setLoadingTab(false))
    }
    if (tab === 'suggestions' && !suggestions) {
      setLoadingTab(true)
      fetchMaterialSuggestions(50)
        .then(setSuggestions)
        .catch((err) => toast.show(describe(err), 'error'))
        .finally(() => setLoadingTab(false))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, stats, suggestions])

  const openDetail = async (name: string) => {
    setDetailName(name)
    setDetail(null)
    try {
      setDetail(await fetchMaterialDetail(name))
    } catch (err) {
      toast.show(describe(err), 'error')
    }
  }

  const onQueryChange = (v: string) => {
    setQuery(v)
    clearTimeout(debounceRef.current)
    if (v.trim().length >= 1) {
      debounceRef.current = window.setTimeout(() => void doMatch(v), 400)
    }
  }

  return (
    <div>
      <PageHeader
        title="物料智能识别"
        description="把同一物料的多种写法归并到标准名，并查看它的价格水平与交易情况"
        actions={
          stats && (
            <div className="flex items-center gap-2">
              <Badge color="brand">
                <Package className="h-3 w-3" />
                {stats.standard_total.toLocaleString('zh-CN')} 个标准物料
              </Badge>
              <Badge color="neutral">
                归并前 {stats.raw_total.toLocaleString('zh-CN')} 种写法
              </Badge>
            </div>
          )
        }
      />

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

      {/* ── 物料匹配 ─────────────────────────────────────────── */}
      {tab === 'match' && (
        <>
          <Card
            title="输入任意写法"
            subtitle="支持模糊匹配，会同时给出匹配理由与归并到的标准名"
          >
            <div className="flex items-center gap-3">
              <input
                value={query}
                onChange={(e) => onQueryChange(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void doMatch(query)}
                placeholder="如：空调 / 复印纸 / 硒鼓的网上超市采购"
                className="h-9 flex-1 rounded-lg border border-slate-200 px-3 text-[13px] outline-none focus:border-brand-500"
              />
              <Button onClick={() => void doMatch(query)} loading={matching} icon={<Search className="h-3.5 w-3.5" />}>
                识别
              </Button>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {['空调', '复印纸', '硒鼓', '台式整机', '粉盒', '办公椅', '文件柜'].map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setQuery(s)
                    void doMatch(s)
                  }}
                  className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] text-ink-500 hover:border-brand-300 hover:bg-brand-50"
                >
                  {s}
                </button>
              ))}
            </div>
            {matchMeta && (
              <div className="mt-2 text-[11px] text-ink-400">
                归一化结果：<span className="font-mono text-ink-600">{matchMeta.normalized}</span>
                ｜候选 {matchMeta.total} 个
              </div>
            )}
          </Card>

          {matching && (
            <Card className="mt-4">
              <div className="flex items-center gap-2 text-[12px] text-ink-500">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
                正在匹配…
              </div>
              <Skeleton className="mt-3 h-32 w-full" />
            </Card>
          )}

          {!matching && matches.length > 0 && (
            <div className="mt-4 space-y-3">
              {matches.map((m) => (
                <div
                  key={m.standard_name}
                  className="rounded-xl border border-slate-200 bg-white p-4 transition-shadow hover:shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[15px] font-semibold text-ink-900">
                          {m.standard_name}
                        </span>
                        <Badge color={scoreColor(m.score)}>
                          匹配度 {(m.score * 100).toFixed(0)}%
                        </Badge>
                        <Badge color="neutral">{m.match_reason}</Badge>
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[12px] text-ink-500">
                        <span>
                          累计 <span className="font-semibold text-ink-900">{m.total_count}</span> 次交易
                        </span>
                        <span>
                          金额中位{' '}
                          <span className="font-semibold text-ink-900">{money(m.median_amount)}</span>
                        </span>
                        <span>
                          归并 <span className="font-semibold text-ink-900">{m.matched_variants.length}</span> 种写法
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => void openDetail(m.standard_name)}
                      className="shrink-0 rounded-lg border border-slate-200 px-2.5 py-1 text-[11px] text-ink-500 hover:border-brand-300 hover:bg-brand-50"
                    >
                      查看详情
                    </button>
                  </div>

                  {/* 归并明细：这是本页最有价值的信息 */}
                  {m.matched_variants.length > 0 && (
                    <div className="mt-3 rounded-lg border border-slate-100 bg-ink-50/50 p-2.5">
                      <div className="flex items-center gap-1.5 text-[11px] text-ink-400">
                        <GitMerge className="h-3 w-3" />
                        归并到「{m.standard_name}」的原始写法
                      </div>
                      <div className="mt-1.5 space-y-1">
                        {m.matched_variants.map((v) => (
                          <div key={v.name} className="flex items-center gap-2 text-[12px]">
                            <span className="w-10 shrink-0 text-right tabular-nums text-ink-400">
                              {v.count}
                            </span>
                            <span className="text-ink-300">次</span>
                            {v.name === m.standard_name ? (
                              <span className="font-medium text-ink-900">{v.name}</span>
                            ) : (
                              <>
                                <span className="text-ink-600">{v.name}</span>
                                <ArrowRight className="h-3 w-3 shrink-0 text-ink-300" />
                                <span className="text-[11px] text-ok">归并</span>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {!matching && matches.length === 0 && (
            <Card className="mt-4">
              <EmptyState
                icon={<Package className="h-6 w-6" />}
                title="未匹配到物料"
                description="试试更短的关键词，或换用物料的核心词（如「打印机」而不是「打印机耗材采购」）"
              />
            </Card>
          )}

          {/* 详情 */}
          {detail && (
            <Card className="mt-4" title={`「${detailName}」详情`} subtitle="价格分位数与主要交易对手">
              {detail.matched === false ? (
                <div className="text-[13px] text-ink-500">{detail.message}</div>
              ) : (
                <>
                  <div className="grid grid-cols-5 gap-3">
                    {[
                      ['P10', detail.percentiles?.p10],
                      ['P25', detail.percentiles?.p25],
                      ['中位数', detail.percentiles?.median],
                      ['P75', detail.percentiles?.p75],
                      ['P90', detail.percentiles?.p90],
                    ].map(([label, value]) => (
                      <div
                        key={label as string}
                        className={`rounded-lg border p-3 text-center ${
                          label === '中位数' ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200'
                        }`}
                      >
                        <div className="text-[11px] text-ink-500">{label as string}</div>
                        <div
                          className={`mt-1 text-[15px] font-bold tabular-nums ${
                            label === '中位数' ? 'text-ok' : 'text-ink-900'
                          }`}
                        >
                          {money(value as number | null)}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-2 text-[11px] text-ink-400">
                    累计 {detail.total_count} 次交易｜归并 {detail.variant_count} 种写法｜最小{' '}
                    {money(detail.min_amount)}｜最大 {money(detail.max_amount)}
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-4">
                    {detail.top_suppliers && detail.top_suppliers.length > 0 && (
                      <div>
                        <div className="mb-1.5 text-[12px] font-medium text-ink-600">主要供应商</div>
                        <table className="w-full text-[12px]">
                          <tbody>
                            {detail.top_suppliers.slice(0, 6).map((s) => (
                              <tr key={s.name} className="border-b border-slate-100 last:border-0">
                                <td className="py-1 text-ink-900">{s.name}</td>
                                <td className="w-14 py-1 text-right tabular-nums text-ink-500">
                                  {s.count} 次
                                </td>
                                <td className="w-20 py-1 text-right tabular-nums text-ink-600">
                                  {money(s.median_amount)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {detail.top_purchasers && detail.top_purchasers.length > 0 && (
                      <div>
                        <div className="mb-1.5 text-[12px] font-medium text-ink-600">主要采购人</div>
                        <table className="w-full text-[12px]">
                          <tbody>
                            {detail.top_purchasers.slice(0, 6).map((p) => (
                              <tr key={p.name} className="border-b border-slate-100 last:border-0">
                                <td className="py-1 text-ink-900">{p.name}</td>
                                <td className="w-14 py-1 text-right tabular-nums text-ink-500">
                                  {p.count} 次
                                </td>
                                <td className="w-20 py-1 text-right tabular-nums text-ink-600">
                                  {money(p.median_amount)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  {detail.advice && detail.advice.length > 0 && (
                    <div className="mt-3 rounded-lg border border-brand-100 bg-brand-50 p-3">
                      <div className="text-[12px] font-semibold text-brand-700">建议</div>
                      <ul className="mt-1 space-y-0.5">
                        {detail.advice.map((a, i) => (
                          <li key={i} className="text-[12px] leading-5 text-ink-700">
                            • {a.replace(/\*\*/g, '')}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </Card>
          )}
        </>
      )}

      {/* ── 物料库概览 ───────────────────────────────────────── */}
      {tab === 'library' && (
        <>
          {loadingTab && <Skeleton className="h-64 w-full" />}
          {!loadingTab && stats && (
            <>
              <div className="grid grid-cols-4 gap-4">
                {[
                  ['标准物料', stats.standard_total.toLocaleString('zh-CN'), '归并后的物料数'],
                  ['原始写法', stats.raw_total.toLocaleString('zh-CN'), '归并前的写法数'],
                  ['归并组数', stats.merged_groups.toLocaleString('zh-CN'), '由 ≥2 种写法合并而成'],
                  [
                    '写法压缩率',
                    `${(stats.compression_ratio * 100).toFixed(1)}%`,
                    '不是数据脏污率',
                  ],
                ].map(([label, value, note]) => (
                  <div key={label} className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-[11px] text-ink-500">{label}</div>
                    <div className="mt-1 text-xl font-bold tabular-nums text-ink-900">{value}</div>
                    <div className="mt-0.5 text-[10px] text-ink-400">{note}</div>
                  </div>
                ))}
              </div>

              <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-600">
                  <Info className="h-3.5 w-3.5" />
                  口径说明
                </div>
                <ul className="mt-2 space-y-1">
                  {stats.notes.map((n, i) => (
                    <li key={i} className="text-[11px] leading-5 text-ink-500">
                      • {n}
                    </li>
                  ))}
                </ul>
              </div>

              <Card className="mt-4" title="物料 TOP 20" subtitle="按交易次数排序，金额为中位数">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                      <th className="w-10 py-2">#</th>
                      <th className="py-2">标准物料名</th>
                      <th className="w-20 py-2 text-right">交易次数</th>
                      <th className="w-20 py-2 text-right">归并写法</th>
                      <th className="w-28 py-2 text-right">金额中位数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.top_materials.map((m, i) => (
                      <tr key={m.standard_name} className="border-b border-slate-100 last:border-0">
                        <td className="py-1.5 text-[11px] text-ink-300">{i + 1}</td>
                        <td className="py-1.5">
                          <button
                            onClick={() => {
                              setTab('match')
                              setQuery(m.standard_name)
                              void doMatch(m.standard_name)
                            }}
                            className="text-left text-brand-700 hover:underline"
                          >
                            {m.standard_name}
                          </button>
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink-900">{m.total_count}</td>
                        <td className="py-1.5 text-right">
                          {m.variant_count > 1 ? (
                            <Badge color="brand">{m.variant_count} 种</Badge>
                          ) : (
                            <span className="text-[11px] text-ink-300">1 种</span>
                          )}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink-600">
                          {money(m.median_amount)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}
        </>
      )}

      {/* ── 标准化建议 ───────────────────────────────────────── */}
      {tab === 'suggestions' && (
        <>
          {loadingTab && <Skeleton className="h-64 w-full" />}
          {!loadingTab && suggestions && (
            <Card
              title="疑似应归并的物料对"
              subtitle={suggestions.rules ?? '名字高度相似但当前规则未合并，供人工确认'}
              extra={
                <Badge color="warn">
                  <AlertTriangle className="h-3 w-3" />
                  {suggestions.total_candidates} 组候选
                </Badge>
              }
            >
              {suggestions.items.length === 0 ? (
                <EmptyState
                  icon={<CheckCircle2 className="h-6 w-6" />}
                  title="没有待确认的归并候选"
                  description="当前归一化规则已覆盖所有高度相似的物料名"
                />
              ) : (
                <>
                  <table className="w-full text-[13px]">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                        <th className="w-10 py-2">#</th>
                        <th className="py-2">物料 A</th>
                        <th className="w-8 py-2"></th>
                        <th className="py-2">物料 B</th>
                        <th className="w-20 py-2 text-right">相似度</th>
                        <th className="w-40 py-2">判定依据</th>
                      </tr>
                    </thead>
                    <tbody>
                      {suggestions.items.map((s, i) => (
                        <tr key={`${s.a}|${s.b}`} className="border-b border-slate-100 last:border-0">
                          <td className="py-1.5 text-[11px] text-ink-300">{i + 1}</td>
                          <td className="py-1.5">
                            <span className="text-ink-900">{s.a}</span>
                            <span className="ml-1.5 text-[11px] tabular-nums text-ink-400">
                              ({s.count_a})
                            </span>
                          </td>
                          <td className="py-1.5 text-center text-ink-300">
                            <GitMerge className="inline h-3 w-3" />
                          </td>
                          <td className="py-1.5">
                            <span className="text-ink-900">{s.b}</span>
                            <span className="ml-1.5 text-[11px] tabular-nums text-ink-400">
                              ({s.count_b})
                            </span>
                          </td>
                          <td className="py-1.5 text-right">
                            <Badge color={s.score >= 0.9 ? 'ok' : 'warn'}>
                              {(s.score * 100).toFixed(0)}%
                            </Badge>
                          </td>
                          <td className="py-1.5 text-[11px] text-ink-500">{s.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {suggestions.note && (
                    <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                      <Info className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
                      <span className="text-[11px] leading-5 text-ink-600">{suggestions.note}</span>
                    </div>
                  )}
                </>
              )}
            </Card>
          )}
        </>
      )}

      {/* 能力说明 */}
      <div className="mt-4 flex items-start gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-[11px] leading-5 text-ink-400">
        <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>
          真实数据里只有 <code className="rounded bg-slate-100 px-1">subject_matter</code> 一列文本，
          <b>没有物料编码、分类路径或品牌</b>，所以本页做的是「同一物料多种写法的归并」，
          而不是跨 ERP/MES/SRM 的一物多码对齐。归并规则可扩展——见
          <code className="mx-1 rounded bg-slate-100 px-1">src/services/material.py</code>
          的噪声后缀列表。
        </span>
      </div>
    </div>
  )
}
