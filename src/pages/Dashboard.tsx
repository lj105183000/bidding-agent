import { Link } from 'react-router-dom'
import {
  ArrowRight,
  Building2,
  FileSearch,
  MapPin,
  MoreHorizontal,
  TrendingDown,
} from 'lucide-react'
import { NAV_ITEMS } from '../config/nav'
import Card from '../components/Card'
import { KPI } from '../data/analysis'
import { BID_RECORDS } from '../data/bids'
import { formatMoney, shortDate } from '../lib/format'

const SCENARIOS = [
  { ...NAV_ITEMS[1], desc: '企业知识库问答，法规咨询秒级应答' },
  { ...NAV_ITEMS[2], desc: '公告文本一键解析，17 字段结构化标讯' },
  { ...NAV_ITEMS[3], desc: '多平台聚合比价，价比三家择优采购' },
  { ...NAV_ITEMS[4], desc: '一物多码自动聚类，物料标准化治理' },
  { ...NAV_ITEMS[5], desc: '集采数据洞察 + 供应商智能推荐' },
]

const KPIS = [
  { label: '累计解析标讯', value: KPI.totalBids.toLocaleString('zh-CN'), icon: FileSearch, cls: 'bg-brand-50 text-brand-600' },
  { label: '覆盖省份', value: KPI.provinces.toLocaleString('zh-CN'), icon: MapPin, cls: 'bg-emerald-50 text-ok' },
  { label: '合作供应商', value: KPI.suppliers.toLocaleString('zh-CN'), icon: Building2, cls: 'bg-amber-50 text-warn' },
  { label: '平均节资率', value: `${KPI.savingRate}%`, icon: TrendingDown, cls: 'bg-red-50 text-danger' },
]

export default function Dashboard() {
  return (
    <div>
      {/* Hero 区 */}
      <div className="rounded-xl bg-gradient-to-br from-brand-600 via-brand-700 to-indigo-800 p-8 text-white">
        <h1 className="text-2xl font-bold leading-tight">大模型驱动的数字智能招投标采购平台</h1>
        <p className="mt-2 text-sm text-white/80">
          覆盖标讯解析 · 智能问答 · 比价选品 · 物料治理 · 集采决策全链路
        </p>
        <div className="mt-6 border-t border-white/20 pt-4 text-xs text-white/60">
          当前模型：GPT-4o ｜ 知识库：招投标法规 12,846 篇 ｜ 响应延迟 &lt; 800ms
        </div>
      </div>

      {/* KPI 行 */}
      <div className="mt-6 grid grid-cols-2 gap-4 xl:grid-cols-4">
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

      {/* 核心能力入口 */}
      <h2 className="mb-4 mt-8 text-[15px] font-bold text-ink-900">核心能力</h2>
      <div className="grid grid-cols-3 gap-4">
        {SCENARIOS.map((s) => (
          <Link key={s.path} to={s.path} className="card group block p-5">
            <div className="flex items-start justify-between">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
                <s.icon className="h-5 w-5" />
              </div>
              <ArrowRight className="h-4 w-4 text-ink-300 transition-all group-hover:translate-x-0.5 group-hover:text-brand-600" />
            </div>
            <div className="mt-3 text-[15px] font-semibold text-ink-900">{s.label}</div>
            <div className="mt-1 text-xs leading-5 text-ink-400">{s.desc}</div>
          </Link>
        ))}
        <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-ink-200 p-5 text-center">
          <MoreHorizontal className="h-6 w-6 text-ink-300" />
          <div className="mt-2 text-[15px] font-semibold text-ink-400">更多能力</div>
          <div className="mt-1 text-xs text-ink-300">敬请期待</div>
        </div>
      </div>

      {/* 最近中标动态 */}
      <Card
        className="mt-8"
        title="最近中标动态"
        subtitle="来自 reference/bids.xlsx 真实标讯精选"
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-ink-400">
              <th className="pb-3 pr-4 font-medium">项目名称</th>
              <th className="pb-3 pr-4 font-medium">采购人</th>
              <th className="pb-3 pr-4 font-medium">中标人</th>
              <th className="pb-3 pr-4 text-right font-medium">中标金额</th>
              <th className="pb-3 font-medium">中标时间</th>
            </tr>
          </thead>
          <tbody>
            {BID_RECORDS.slice(0, 6).map((b) => (
              <tr key={b.id} className="border-b border-slate-100 last:border-0 hover:bg-ink-50">
                <td className="max-w-[280px] truncate py-3 pr-4 text-ink-700" title={b.projectName}>
                  {b.projectName}
                </td>
                <td className="py-3 pr-4 text-ink-500">{b.purchaser}</td>
                <td className="py-3 pr-4 text-ink-500">{b.winner}</td>
                <td className="py-3 pr-4 text-right font-medium text-ink-900 tabular-nums">
                  {formatMoney(b.winAmount)}
                </td>
                <td className="py-3 text-ink-400 tabular-nums">{shortDate(b.winTime)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </Card>
    </div>
  )
}
