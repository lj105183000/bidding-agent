import { useEffect, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Clock, FileText, Package, Search, Sparkles } from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import { PRICE_DATA, type PriceItem } from '../data/price'

const QUICK_TAGS = ['内六角扳手', '施耐德断路器', '3M口罩', '西门子接触器']
const COLLECT_TIME = '2026-09-22 09:42'
const LOW_COLOR = '#10B981' // 最低价绿
const OTHER_COLOR = '#3B82F6' // 其余品牌蓝

/** 关键词 → 商品映射, 保证快捷标签与搜索词能命中对应商品 */
const KEYWORDS: [string, string[]][] = [
  ['price-001', ['内六角扳手', 'BS423181', '波斯']],
  ['price-002', ['断路器', 'iC65N', '施耐德']],
  ['price-003', ['口罩', '9502', '3M']],
  ['price-004', ['接触器', '3TF30', '西门子']],
]

function matchItem(q: string): { item: PriceItem; matched: boolean } {
  const hit = KEYWORDS.find(([, kws]) => kws.some((k) => q.includes(k)))
  if (hit) return { item: PRICE_DATA.find((p) => p.id === hit[0])!, matched: true }
  const nameHit = PRICE_DATA.find((p) => p.name.includes(q) || p.spec.includes(q))
  if (nameHit) return { item: nameHit, matched: true }
  return { item: PRICE_DATA[0], matched: false }
}

const tooltipStyle = {
  borderRadius: 8,
  border: '1px solid #E2E8F0',
  fontSize: 12,
  boxShadow: '0 4px 12px rgba(15, 23, 42, 0.08)',
}

export default function PriceCompare() {
  const toast = useToast()
  const [term, setTerm] = useState('波斯 加长球头内六角扳手 BS423181')
  const [result, setResult] = useState<PriceItem | null>(null)
  const [loading, setLoading] = useState(false)
  const timerRef = useRef<number>()

  const runSearch = (t: string) => {
    const q = t.trim()
    if (!q) {
      toast.show('请输入商品关键词', 'warning')
      return
    }
    setLoading(true)
    setResult(null)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      const { item, matched } = matchItem(q)
      setResult(item)
      setLoading(false)
      if (!matched) toast.show('未精确匹配，已展示最接近商品', 'warning')
    }, 800)
  }

  // 初始自动执行一次默认搜索
  useEffect(() => {
    runSearch(term)
    return () => clearTimeout(timerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!result && !loading) {
    return (
      <div>
        <PageHeader title="智能商品比价" description="多平台商品价格对比与择优采购" />
        <Card>
          <div className="flex items-center gap-3">
            <input
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runSearch(term)}
              className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-[13px] text-ink-900 outline-none transition-colors focus:border-brand-500"
              placeholder="输入商品名称或型号…"
            />
            <Button icon={<Search className="h-3.5 w-3.5" />} onClick={() => runSearch(term)}>
              智能比价
            </Button>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {QUICK_TAGS.map((t) => (
              <button
                key={t}
                onClick={() => {
                  setTerm(t)
                  runSearch(t)
                }}
                className="rounded-lg border border-slate-200 px-3 py-1 text-xs text-ink-500 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700"
              >
                {t}
              </button>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const platforms = result ? [...result.platforms].sort((a, b) => a.price - b.price) : []
  const low = platforms[0]
  const high = platforms[platforms.length - 1]
  const diffPct = low && high ? (((high.price - low.price) / low.price) * 100).toFixed(1) : '0.0'
  const barData = result ? result.platforms.map((p) => ({ name: p.name, price: p.price })) : []
  const barMax = result ? Math.max(...result.platforms.map((p) => p.price)) + 4 : 0
  const lineData = result
    ? result.trend.map((t) => ({ ...t, m: `${Number(t.month.slice(5))}月` }))
    : []
  const lineMin = result ? Math.min(...result.trend.map((t) => t.price)) - 3 : 0
  const lineMax = result ? Math.max(...result.trend.map((t) => t.price)) + 3 : 0

  return (
    <div>
      <PageHeader title="智能商品比价" description="多平台商品价格对比与择优采购" />

      {/* 搜索卡片 */}
      <Card>
        <div className="flex items-center gap-3">
          <input
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runSearch(term)}
            className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-[13px] text-ink-900 outline-none transition-colors focus:border-brand-500"
            placeholder="输入商品名称或型号…"
          />
          <Button icon={<Search className="h-3.5 w-3.5" />} onClick={() => runSearch(term)}>
            智能比价
          </Button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {QUICK_TAGS.map((t) => (
            <button
              key={t}
              onClick={() => {
                setTerm(t)
                runSearch(t)
              }}
              className="rounded-lg border border-slate-200 px-3 py-1 text-xs text-ink-500 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700"
            >
              {t}
            </button>
          ))}
        </div>
      </Card>

      {/* 加载骨架: 3 个 Skeleton, 800ms 后显示结果 */}
      {loading && (
        <div className="mt-6 space-y-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}

      {!loading && result && (
        <>
          {/* 1. 商品信息卡 */}
          <Card className="mt-6">
            <div className="flex items-center gap-5">
              <div className="flex h-24 w-24 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-50 to-brand-100">
                <Package className="h-10 w-10 text-brand-500" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-base font-semibold text-ink-900">{result.name}</span>
                  <Badge color="brand">品牌：{result.brand ?? '—'}</Badge>
                </div>
                <div className="mt-1 text-xs text-ink-400">规格型号：{result.spec}</div>
                <div className="mt-2 flex items-center gap-1 text-[11px] text-ink-400">
                  <Clock className="h-3 w-3" />
                  采集时间：{COLLECT_TIME}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-ink-400">参考价（{result.unit}）</div>
                <div className="mt-1 text-[28px] font-bold leading-8 text-brand-600 tabular-nums">
                  ¥{result.refPrice.toFixed(2)}
                </div>
              </div>
            </div>
          </Card>

          {/* 2. 多平台报价 */}
          <div className="mt-4 grid grid-cols-2 gap-4 xl:grid-cols-4">
            {platforms.map((p, i) => (
              <div
                key={p.name}
                className={`relative rounded-xl border p-4 transition-shadow ${
                  i === 0
                    ? 'border-emerald-200 bg-emerald-50'
                    : 'border-slate-200 bg-white hover:shadow-md'
                }`}
              >
                {i === 0 && (
                  <span className="absolute -right-2 -top-2 rounded-md bg-ok px-1.5 py-0.5 text-[10px] font-medium text-white">
                    最优
                  </span>
                )}
                <div className="text-xs text-ink-400">{p.name}</div>
                <div
                  className={`mt-1.5 text-lg font-bold tabular-nums ${
                    i === 0 ? 'text-ok' : 'text-ink-900'
                  }`}
                >
                  ¥{p.price.toFixed(2)}
                </div>
                <div className="mt-1 text-[11px] tabular-nums text-ink-400">
                  {p.price >= result.refPrice ? '+' : ''}
                  {(((p.price - result.refPrice) / result.refPrice) * 100).toFixed(1)}% vs 参考价
                </div>
              </div>
            ))}
          </div>

          {/* 3. 图表区: 左柱状 右折线 */}
          <div className="mt-4 grid grid-cols-2 gap-4">
            <Card title="平台价格对比" subtitle="绿色为最低报价">
              <BarChart width={500} height={260} data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="name"
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tick={{ fontSize: 12, fill: '#64748B' }}
                />
                <YAxis
                  domain={[0, barMax]}
                  tickFormatter={(v: number) => `¥${v}`}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11, fill: '#94A3B8' }}
                  width={48}
                />
                <Tooltip
                  formatter={(value) => [`¥${Number(value).toFixed(2)}`, '报价']}
                  contentStyle={tooltipStyle}
                  cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
                />
                <Bar dataKey="price" radius={[4, 4, 0, 0]} barSize={40} isAnimationActive={false}>
                  {barData.map((d) => (
                    <Cell
                      key={d.name}
                      fill={d.name === low.name ? LOW_COLOR : OTHER_COLOR}
                    />
                  ))}
                  <LabelList
                    dataKey="price"
                    position="top"
                    formatter={(v: React.ReactNode) => `¥${Number(v).toFixed(2)}`}
                    style={{ fontSize: 11, fill: '#334155' }}
                  />
                </Bar>
              </BarChart>
            </Card>

            <Card title="近 6 个月价格走势" subtitle="单位：元">
              <LineChart width={500} height={260} data={lineData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="m"
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tick={{ fontSize: 12, fill: '#64748B' }}
                />
                <YAxis
                  domain={[lineMin, lineMax]}
                  tickFormatter={(v: number) => `¥${v}`}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11, fill: '#94A3B8' }}
                  width={48}
                />
                <Tooltip
                  formatter={(value) => [`¥${Number(value).toFixed(2)}`, '参考价']}
                  contentStyle={tooltipStyle}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#2563EB"
                  strokeWidth={2}
                  isAnimationActive={false}
                  dot={{ r: 3.5, fill: '#2563EB', strokeWidth: 0 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </Card>
          </div>

          {/* 4. AI 采购建议 */}
          <div className="mt-4 rounded-xl border border-brand-100 bg-brand-50 p-5">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-brand-600" />
              <span className="text-[15px] font-semibold text-ink-900">AI 采购建议</span>
            </div>
            <p className="mt-2 text-[13px] leading-6 text-ink-700">
              当前 {platforms.length} 个平台中，{low.name}报价
              <span className="font-semibold text-ink-900">¥{low.price.toFixed(2)}</span>
              为最低价
              {platforms.length > 1
                ? `，较第二名${platforms[1].name}低${(((platforms[1].price - low.price) / low.price) * 100).toFixed(1)}%`
                : ''}
              。建议优先从{low.name}采购。若考虑账期与售后，
              {high.name}虽价格高 {diffPct}%，但支持 60 天账期。
            </p>
          </div>

          {/* 底部操作 */}
          <div className="mt-4 flex justify-end">
            <Button
              variant="secondary"
              icon={<FileText className="h-3.5 w-3.5" />}
              onClick={() => toast.show('比价报告已生成，可在工作台查看', 'success')}
            >
              生成比价报告
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
