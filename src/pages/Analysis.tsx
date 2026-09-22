import { useEffect, useRef, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Pie,
  PieChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Building2, FileSearch, MapPin, Sparkles, TrendingDown } from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import {
  CATEGORY_DIST,
  KPI,
  MONTHLY_AMOUNT,
  RECOMMEND_SUPPLIERS,
  TOP_SUPPLIERS,
} from '../data/analysis'

const DONUT_COLORS = ['#2563EB', '#10B981', '#F59E0B']
const BLUE = '#3B82F6'

const KPIS = [
  { label: '累计解析标讯', value: KPI.totalBids.toLocaleString('zh-CN'), icon: FileSearch, cls: 'bg-brand-50 text-brand-600' },
  { label: '覆盖省份', value: KPI.provinces.toLocaleString('zh-CN'), icon: MapPin, cls: 'bg-emerald-50 text-ok' },
  { label: '合作供应商', value: KPI.suppliers.toLocaleString('zh-CN'), icon: Building2, cls: 'bg-amber-50 text-warn' },
  { label: '平均节资率', value: `${KPI.savingRate}%`, icon: TrendingDown, cls: 'bg-red-50 text-danger' },
]

const tooltipStyle = {
  borderRadius: 8,
  border: '1px solid #E2E8F0',
  fontSize: 12,
  boxShadow: '0 4px 12px rgba(15, 23, 42, 0.08)',
}

/** 万元 → 亿/万 紧凑展示 */
function fmtWan(v: number): string {
  if (v >= 10000) return `${(v / 10000).toFixed(1)}亿`
  return `${v}万`
}

function ringColor(score: number): string {
  if (score > 90) return '#10B981'
  if (score >= 80) return '#2563EB'
  return '#F59E0B'
}

function ScoreRing({ score }: { score: number }) {
  const color = ringColor(score)
  const r = 22
  const c = 2 * Math.PI * r
  return (
    <div className="relative h-14 w-14 shrink-0">
      <svg viewBox="0 0 56 56" className="h-full w-full -rotate-90">
        <circle cx="28" cy="28" r={r} fill="none" stroke="#E2E8F0" strokeWidth="5" />
        <circle
          cx="28"
          cy="28"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={`${(score / 100) * c} ${c}`}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-[11px] font-bold tabular-nums" style={{ color }}>
        {score}%
      </span>
    </div>
  )
}

export default function Analysis() {
  const toast = useToast()
  const [term, setTerm] = useState('智能化系统集成及监控设备采购')
  const [loading, setLoading] = useState(false)
  const [showList, setShowList] = useState(false)
  const timerRef = useRef<number>()

  const runRecommend = (t: string) => {
    const q = t.trim()
    if (!q) {
      toast.show('请输入采购需求描述', 'warning')
      return
    }
    setLoading(true)
    setShowList(false)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setShowList(true)
      setLoading(false)
    }, 800)
  }

  useEffect(() => {
    runRecommend(term)
    return () => clearTimeout(timerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 推荐列表按匹配度降序
  const suppliers = [...RECOMMEND_SUPPLIERS].sort((a, b) => b.matchScore - a.matchScore)

  const areaData = MONTHLY_AMOUNT.map((v, i) => ({ m: `${i + 1}月`, amount: v }))
  const totalPct = CATEGORY_DIST.reduce((s, d) => s + d.value, 0)

  return (
    <div>
      <PageHeader title="集采分析与推荐" description="集中采购数据分析与智能推荐" />

      {/* 第一部分: KPI 行 */}
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        {KPIS.map((k) => (
          <Card key={k.label}>
            <div className="flex items-center gap-3">
              <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${k.cls}`}>
                <k.icon className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="text-[22px] font-bold leading-7 text-ink-900 tabular-nums">{k.value}</div>
                <div className="mt-0.5 text-xs text-ink-400">{k.label}</div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* 第二部分: 图表 */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <Card title="采购类别分布" subtitle="按品类采购占比">
          <div className="relative">
            <PieChart width={500} height={260}>
              <Pie
                data={CATEGORY_DIST}
                dataKey="value"
                nameKey="name"
                cx={170}
                cy={130}
                innerRadius={62}
                outerRadius={92}
                paddingAngle={2}
                strokeWidth={2}
                stroke="#FFFFFF"
                isAnimationActive={false}
              >
                {CATEGORY_DIST.map((d, i) => (
                  <Cell key={d.name} fill={DONUT_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name) => [`${value}%`, name]}
                contentStyle={tooltipStyle}
              />
              <Legend
                verticalAlign="middle"
                align="right"
                layout="vertical"
                iconType="circle"
                iconSize={9}
                formatter={(value: string, entry: { payload?: { value?: number } }) =>
                  `${value} ${entry?.payload?.value ?? 0}%`
                }
                wrapperStyle={{ fontSize: 12, color: '#334155', lineHeight: '24px' }}
              />
            </PieChart>
            <div className="pointer-events-none absolute left-[128px] top-[118px] text-center">
              <div className="text-[11px] text-ink-400">品类合计</div>
              <div className="text-[18px] font-bold text-ink-900 tabular-nums">{totalPct}%</div>
            </div>
          </div>
        </Card>

        <Card title="月度中标金额趋势" subtitle="单位：万元">
          <AreaChart width={500} height={260} data={areaData}>
            <defs>
              <linearGradient id="amountGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={BLUE} stopOpacity={0.25} />
                <stop offset="100%" stopColor={BLUE} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
            <XAxis
              dataKey="m"
              tickLine={false}
              axisLine={{ stroke: '#E2E8F0' }}
              tick={{ fontSize: 11, fill: '#64748B' }}
            />
            <YAxis
              domain={[2000, 7000]}
              tickFormatter={(v: number) => `${v}`}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: '#94A3B8' }}
              width={44}
            />
            <Tooltip
              formatter={(value) => [`${Number(value).toLocaleString('zh-CN')} 万元`, '中标金额']}
              contentStyle={tooltipStyle}
            />
            <Area
              type="monotone"
              dataKey="amount"
              stroke={BLUE}
              strokeWidth={2}
              fill="url(#amountGrad)"
              isAnimationActive={false}
              activeDot={{ r: 5 }}
            />
          </AreaChart>
        </Card>
      </div>

      <Card
        className="mt-4"
        title="Top 10 中标供应商"
        subtitle="按累计中标金额排序（万元），数据来自真实标讯统计"
      >
        <BarChart width={1000} height={330} data={TOP_SUPPLIERS} layout="vertical" margin={{ left: 4, right: 28 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
          <XAxis
            type="number"
            tickFormatter={(v: number) => fmtWan(v)}
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 11, fill: '#94A3B8' }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={204}
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 11, fill: '#64748B' }}
          />
          <Tooltip
            formatter={(value) => [`${Number(value).toLocaleString('zh-CN')} 万元`, '累计中标金额']}
            contentStyle={tooltipStyle}
            cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
          />
          <Bar dataKey="amount" fill={BLUE} radius={[0, 4, 4, 0]} barSize={14} isAnimationActive={false}>
            <LabelList
              dataKey="amount"
              position="right"
              formatter={(v: React.ReactNode) => fmtWan(Number(v))}
              style={{ fontSize: 11, fill: '#334155' }}
            />
          </Bar>
        </BarChart>
      </Card>

      {/* 第三部分: 智能推荐 */}
      <Card className="mt-6" title="智能推荐" subtitle="按采购需求匹配优质供应商">
        <div className="flex items-center gap-3">
          <input
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runRecommend(term)}
            className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-[13px] text-ink-900 outline-none transition-colors focus:border-brand-500"
            placeholder="输入采购需求描述…"
          />
          <Button
            onClick={() => runRecommend(term)}
            loading={loading}
            icon={<Sparkles className="h-3.5 w-3.5" />}
          >
            智能推荐
          </Button>
        </div>

        {loading && (
          <div className="mt-4 space-y-3">
            {[0, 1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        )}

        {!loading && !showList && (
          <EmptyState
            icon={<Sparkles className="h-6 w-6" />}
            title="暂无推荐结果"
            description='输入采购需求后点击"智能推荐"'
          />
        )}

        {!loading && showList && (
          <div className="mt-4 space-y-3">
            {suppliers.map((s) => (
              <div
                key={s.id}
                className="flex items-center gap-4 rounded-xl border border-slate-200 p-4 transition-shadow hover:shadow-md"
              >
                {/* 44x44 供应商首字方块 */}
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 text-lg font-bold text-white">
                  {s.name.charAt(0)}
                </div>

                {/* 中: 名称 + tags + 分类地区 + 合作/评分/匹配度 */}
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] font-semibold text-ink-900">{s.name}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-1">
                    {s.tags.map((t) => (
                      <Badge key={t} color="neutral">
                        {t}
                      </Badge>
                    ))}
                    <span className="text-[11px] text-ink-400">
                      {s.category} · {s.region}
                    </span>
                  </div>
                  <div className="mt-1.5 text-[11px] text-ink-400">
                    历史合作 {s.cooperation} 次 · 评分 {s.rating.toFixed(1)} · 匹配度 {s.matchScore}%
                  </div>
                </div>

                {/* 右: 匹配度圆环 */}
                <ScoreRing score={s.matchScore} />
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* AI 洞察 */}
      <div className="mt-4 rounded-xl border border-brand-100 bg-brand-50 p-5">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-brand-600" />
          <span className="text-[15px] font-semibold text-ink-900">AI 洞察</span>
        </div>
        <p className="mt-2 text-[13px] leading-6 text-ink-700">
          本季度货物类采购占比 42%，同比上升 6 个百分点；工程类平均节资率 15.2%，高于服务类
          9.8%。建议下季度重点对工程类实施带量集采，预计可再降本 3-5%。
        </p>
      </div>
    </div>
  )
}
