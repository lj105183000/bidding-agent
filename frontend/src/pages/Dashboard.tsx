/**
 * 工作台（首页）。
 *
 * 数据来源：全部走真实接口
 *   - `/api/health`                 → 向量库点数与各依赖延迟
 *   - `/api/analysis/overview`      → 标讯条数 / 覆盖省份 / 供应商数
 *   - `/api/analysis/recent`        → 最近中标动态
 *
 * ★ 与早期版本的差别：早期这里有**三处编造的数据**，其中一处是明确的虚假声明：
 *
 *     「当前模型：GPT-4o ｜ 知识库：招投标法规 12,846 篇 ｜ 响应延迟 < 800ms」
 *
 *   本项目从未使用 GPT-4o（实际是 DeepSeek / 智谱 / 本地 Ollama 可切换），
 *   知识库也不是 12,846 篇（实际向量库 75 条问答 + 图谱 10606 节点），
 *   响应延迟更不是恒定值（实测 5~20 秒不等，取决于是否触发工具调用）。
 *   **首页是用户看到的第一个界面，写死虚假指标是最不该犯的错。**
 *
 *   本版全部改为实测：模型名取自用户当前选择，知识库规模取自向量库真实点数，
 *   延迟取自后端各依赖的真实探测耗时。
 */
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  Building2,
  Database,
  FileSearch,
  Layers,
  Lightbulb,
  Loader2,
  MapPin,
  MoreHorizontal,
  Network,
  Sparkles,
} from 'lucide-react'
import { NAV_ITEMS } from '../config/nav'
import Card from '../components/Card'
import { useModel } from '../contexts/model'
import { HOME_EXAMPLES, buildExamplePayload } from '../data/graph'
import {
  fetchAnalysisOverview,
  fetchHealth,
  fetchRecentRecords,
  type HealthStatus,
  type RecentRecordsResult,
} from '../lib/api'
import { formatMoney, shortDate } from '../lib/format'

const SCENARIOS = [
  { ...NAV_ITEMS[1], desc: '企业知识库问答，法规咨询秒级应答' },
  { ...NAV_ITEMS[3], desc: '公告文本一键解析，17 字段结构化标讯' },
  { ...NAV_ITEMS[5], desc: '多平台聚合比价，价比三家择优采购' },
  { ...NAV_ITEMS[6], desc: '一物多码自动聚类，物料标准化治理' },
  { ...NAV_ITEMS[7], desc: '集采数据洞察 + 竞争对手分析' },
]

/** 当前选择的 provider 的中文名（与 Chat 页的标签保持一致） */
const PROVIDER_LABEL: Record<string, string> = {
  deepseek: 'DeepSeek',
  zhipu: '智谱 GLM',
  ollama: '本机 Ollama',
  vllm: '本地 vLLM',
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { model } = useModel()

  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [overview, setOverview] = useState<{
    total_rows: number
    dimension_cardinality: Record<string, number>
    amount_stats: { median: number | null }
  } | null>(null)
  const [recent, setRecent] = useState<RecentRecordsResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState<string[]>([])

  useEffect(() => {
    // 三个接口并行；**任何一个失败都不该让整页空白**，故用 allSettled
    // 并记录失败项，在页面上如实标注哪块数据没取到
    Promise.allSettled([fetchHealth(), fetchAnalysisOverview(), fetchRecentRecords(6)]).then(
      ([h, o, r]) => {
        const bad: string[] = []
        if (h.status === 'fulfilled') setHealth(h.value)
        else bad.push('服务状态')
        if (o.status === 'fulfilled')
          setOverview(o.value as unknown as typeof overview)
        else bad.push('标讯统计')
        if (r.status === 'fulfilled') setRecent(r.value)
        else bad.push('中标动态')
        setFailed(bad)
        setLoading(false)
      },
    )
  }, [])

  const kpis = [
    {
      label: '标讯记录',
      value: overview ? overview.total_rows.toLocaleString('zh-CN') : '—',
      icon: FileSearch,
      cls: 'bg-brand-50 text-brand-600',
    },
    {
      label: '覆盖省份',
      value: overview ? String(overview.dimension_cardinality.province ?? '—') : '—',
      icon: MapPin,
      cls: 'bg-emerald-50 text-ok',
    },
    {
      label: '合作供应商',
      value: overview ? (overview.dimension_cardinality.supplier ?? 0).toLocaleString('zh-CN') : '—',
      icon: Building2,
      cls: 'bg-amber-50 text-warn',
    },
    {
      label: '知识库问答',
      value:
        health && health.points_count >= 0 ? health.points_count.toLocaleString('zh-CN') : '—',
      icon: Database,
      cls: 'bg-red-50 text-danger',
    },
  ]

  return (
    /* ★ 内容交给 Layout 提供的滚动容器，本页保持普通流式布局
       ----------------------------------------------------
       首页原本靠**整页滚动**：正文一长（或用户放大页面），下方内容被推到
       视口外，必须滚整页才能看到。

       Layout 现在给所有页面统一提供了一层「一屏高 + 内部滚动」的容器
       （与「智能问答」页同形态），故这里**不需要**自己再做滚动区——
       保持普通块级 div，内容自然流，由外层承接滚动条。

       ★ 为什么不做"Hero + 推荐问题常驻、其余滚动"
       ----------------------------------------
       试过：给常驻区加 `max-h` 上限，结果在 720×450（浏览器放大等效视口）
       下 Hero 先占满上限、把推荐问题卡片挤出可见区，又得回头调 dvh 断点——
       **附加设计带进来一串新问题**，而用户要的只是"正文有滚动条"。
       一个滚动条足够，不引入嵌套裁切。 */
    <div>
        {/* Hero 区

            ★ 内边距与副标题按断点收敛
            -------------------------
            原来固定 `p-8` 且副标题始终显示：在 720×450（浏览器放大等效视口）下，
            Hero 本身就占掉约 250px，超过了常驻区的 45dvh 上限（约 203px），
            于是**推荐问题卡片被完全切掉**（截图实测：只剩黄条与半张「标讯记录」）。
            而"推荐问题始终可见"正是这一版要保证的事。

            收紧后（`p-5`、窄屏隐藏副标题）Hero 约 180px，卡片能露出前两行。
            副标题只是宣传文案，窄屏让位给操作入口是划算的。 */}
        <div className="rounded-xl bg-gradient-to-br from-brand-600 via-brand-700 to-indigo-800 p-5 text-white sm:p-6 lg:p-8">
        <h1 className="text-xl font-bold leading-tight sm:text-2xl">大模型驱动的数字智能招投标采购平台</h1>
        {/* 副标题只在 sm 及以上显示：它在窄屏/放大状态下会占掉一整行，
            把常驻区的空间挤给"没有操作价值的宣传文案"不划算 */}
        <p className="mt-2 hidden text-sm text-white/80 sm:block">
          覆盖标讯解析 · 智能问答 · 比价选品 · 物料治理 · 集采决策全链路
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-white/20 pt-3 text-xs text-white/70 sm:mt-6 sm:gap-x-5 sm:pt-4">
          <span>
            当前模型：<span className="font-medium text-white">{PROVIDER_LABEL[model] ?? model}</span>
          </span>
          <span className="flex items-center gap-1">
            <Database className="h-3 w-3" />
            向量库：
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : health && health.points_count >= 0 ? (
              <span className="font-medium text-white">{health.points_count} 条</span>
            ) : (
              <span className="text-amber-300">未连接</span>
            )}
          </span>
          <span className="flex items-center gap-1">
            <Network className="h-3 w-3" />
            图谱：
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : health?.graph_ready ? (
              <span className="font-medium text-white">已连接</span>
            ) : (
              <span className="text-amber-300">未连接</span>
            )}
          </span>
          <span className="flex items-center gap-1">
            <Layers className="h-3 w-3" />
            向量检索延迟：
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : health && health.latencies.qdrant_check_ms >= 0 ? (
              <span className="font-medium text-white">
                {health.latencies.qdrant_check_ms.toFixed(0)} ms
              </span>
            ) : (
              <span className="text-amber-300">—</span>
            )}
          </span>
        </div>
        </div>

      {failed.length > 0 && (
        <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-[12px] text-ink-600">
          以下数据未能取到（后端可能未启动）：{failed.join('、')}
        </div>
      )}

      {/* 示例问题
          
          ★ 这一段必须留在常驻区（`shrink-0` 的 div）**内部**
          ------------------------------------------------
          它是本页唯一把用户导向"提问"的入口。滚到下面看数据时仍需要能直接点
          一个问题——跟着正文滚走等于每次都要先滚回顶部。
          初版把它落在了滚动区里，等于"常驻"根本没生效（实测截图确认）。 */}
      <Card
        className="mt-6"
        title={
          <span className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4 text-brand-600" />
            试试这样问
          </span>
        }
        subtitle="点击任一条，直接跳转到对应页面并自动执行查询"
        extra={
          <span className="flex items-center gap-1 text-[11px] text-ink-400">
            <Sparkles className="h-3 w-3" />
            四路数据源演示
          </span>
        }
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {HOME_EXAMPLES.map((ex) => (
            <button
              key={ex.question}
              onClick={() => navigate(ex.path, { state: buildExamplePayload(ex) })}
              className="group flex items-start justify-between gap-3 rounded-xl border border-slate-200 p-4 text-left transition-all hover:border-brand-300 hover:bg-brand-50/50"
            >
              <div className="min-w-0">
                <div className="text-[13px] font-medium leading-5 text-ink-900">{ex.question}</div>
                <div className="mt-1.5 text-[11px] text-ink-400">{ex.source}</div>
              </div>
              <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-ink-300 transition-all group-hover:translate-x-0.5 group-hover:text-brand-600" />
            </button>
          ))}
        </div>
      </Card>

      {/* KPI 行（全部来自真实接口） */}
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        {kpis.map((k) => (
          <Card key={k.label}>
            <div className="flex items-center gap-3">
              <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${k.cls}`}>
                <k.icon className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="text-[22px] font-bold leading-7 tabular-nums text-ink-900">
                  {loading ? <Loader2 className="h-5 w-5 animate-spin text-ink-300" /> : k.value}
                </div>
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

      {/* 最近中标动态（真实数据 + 覆盖范围说明） */}
      <Card
        className="mt-8"
        title="最近中标动态"
        subtitle={
          recent?.coverage
            ? `按中标时间倒序｜${recent.coverage.caveat}`
            : '按中标时间倒序'
        }
      >
        {loading ? (
          <div className="flex items-center gap-2 py-6 text-[12px] text-ink-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            加载中…
          </div>
        ) : !recent || recent.items.length === 0 ? (
          <div className="py-6 text-center text-[12px] text-ink-400">
            暂无数据（后端或数据库未就绪）
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] text-[13px]">
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
                  {recent.items.map((b, i) => (
                    <tr key={`${b.project_name}-${i}`} className="border-b border-slate-100 last:border-0 hover:bg-ink-50">
                      <td className="max-w-[280px] truncate py-3 pr-4 text-ink-700" title={b.project_name}>
                        {b.project_name}
                      </td>
                      <td className="py-3 pr-4 text-ink-500">{b.purchaser}</td>
                      <td className="py-3 pr-4 text-ink-500">{b.supplier}</td>
                      <td className="py-3 pr-4 text-right font-medium tabular-nums text-ink-900">
                        {b.amount !== null ? formatMoney(b.amount) : '—'}
                      </td>
                      <td className="py-3 tabular-nums text-ink-400">
                        {b.bid_time ? shortDate(b.bid_time) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 text-[11px] leading-5 text-ink-400">
              「最近」指<b>本数据集内</b>中标时间最晚的记录，不代表当前正在进行的采购。
              完整分析见 <Link to="/analysis" className="text-brand-700 underline">集采分析</Link>。
            </div>
          </>
        )}
      </Card>
    </div>
  )
}
