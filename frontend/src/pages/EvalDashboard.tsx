/**
 * 评估看板（`/eval`）。
 *
 * ★ 为什么单独一页而不是塞进系统监控抽屉
 * ------------------------------
 * 评估的内容量比系统监控大一个量级（指标卡 + 分组 + 趋势 + 裁判一致性 + 在线信号），
 * 而那个抽屉只有 416px 宽 —— 图表放进去必然挤成一团。
 *
 * ★ 数据来源与"做不到的事"
 * ------------------
 *   离线：`/api/eval/offline` 读 `eval/` 下的落盘产物
 *   在线：`/api/eval/online`  读真实请求采样的聚合
 *
 * **逐用例表格做不出来**：后端 `EvalReport` 只保存聚合后的 `MetricResult`，
 * 逐用例结果在 `run_all.py` 内部用完即弃。故页面**显式说明**这一点，
 * 而不是画一张空表让人以为加载失败。
 *
 * ★ 为什么比例与计数必须分开渲染
 * ------------------------
 * 指标里混着两类：`faithfulness = 0.82`（比例）与
 * `redundant_calls_mean = 0.82`（次数）。若一律 `Math.round`，
 * 前者会渲染成 "1"（看起来像满分）。后端用**白名单**给出 `is_ratio`，
 * 前端据此选 `formatValue` 的分支 —— **不用值域猜**。
 */

import { useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Info,
  Loader2,
  RefreshCw,
  XCircle,
} from 'lucide-react'

import { ProgressBar } from '../components/MicroBars'
import {
  formatValue,
  GRID_COLOR,
  LOW_COLOR,
  RELEVANCE_NEUTRAL_COLOR,
  sourceColor,
  TICK_FAINT_COLOR,
  WARN_COLOR,
} from '../lib/chartStyles'
import {
  fetchOfflineEval,
  fetchOnlineEval,
  type OfflineEvalSnapshot,
  type OfflineMetric,
  type OnlineEvalSnapshot,
} from '../lib/evalApi'

/** 指标状态 → 图标与色。后端给的是 `pass` / `fail` / `''`（未知）。 */
function statusStyle(status: string): { color: string; Icon: typeof CheckCircle2; text: string } {
  if (status === 'pass') return { color: LOW_COLOR, Icon: CheckCircle2, text: '达标' }
  if (status === 'fail') return { color: WARN_COLOR, Icon: XCircle, text: '未达标' }
  return { color: RELEVANCE_NEUTRAL_COLOR, Icon: Activity, text: '无门禁' }
}

/**
 * 一条指标卡。
 *
 * ★ 比例类指标额外画一条进度条
 * ----------------------
 * 数字 `0.92` 与 `0.24` 在文字上差别不大，但一眼看去"哪个好"并不直观；
 * 进度条把 0~1 的语义**直接表达出来**（满格 = 满分）。
 * 非比例指标（如延迟 P95 = 19405ms）**不画条** —— 它没有 0~1 的上界，
 * 画出来会是满格或空格的误导。
 */
function MetricCard({ m }: { m: OfflineMetric }) {
  const { color, Icon, text } = statusStyle(m.status)
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] leading-tight text-ink-500" title={m.metric}>
          {m.label}
        </span>
        <span className="flex shrink-0 items-center gap-1 text-[10px]" style={{ color }}>
          <Icon className="h-3 w-3" />
          {text}
        </span>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-lg font-medium tabular-nums text-ink-800">
          {formatValue(m.value, m.is_ratio ? 'ratio' : 'number')}
        </span>
        {m.is_ratio && (
          <ProgressBar
            value={m.value}
            width={56}
            height={5}
            color={color}
            title={`${(m.value * 100).toFixed(1)}%`}
          />
        )}
      </div>
    </div>
  )
}

/**
 * 指标趋势（手写 SVG 折线）。
 *
 * ★ 为什么手写而不是用 recharts
 * ------------------------
 * 这是一条无轴、无交互的迷你趋势线（sparkline），recharts 的
 * `<LineChart>` 要配 `ResponsiveContainer` + 轴 + Tooltip 才画得出来，
 * 代码量与包体积都不划算。评估页只需"看出涨跌方向"。
 */
function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) {
    return <span className="text-[10px] text-ink-300">仅 1 次运行，无法看趋势</span>
  }
  const W = 120
  const H = 24
  const min = Math.min(...values)
  const max = Math.max(...values)
  // 全平序列：画中线而不是除零（除零会得到 NaN，路径变成非法值、整条消失）
  const span = max - min || 1
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W
    const y = H - ((v - min) / span) * H
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* 最新点加重：一眼看到"现在在哪" */}
      <circle
        cx={W}
        cy={H - ((values[values.length - 1] - min) / span) * H}
        r={2}
        fill={color}
      />
    </svg>
  )
}

/** 在线评测的一个比例指标（可能为 null = 没有样本）。 */
function RateRow({ label, value, hint }: { label: string; value: number | null | undefined; hint?: string }) {
  const has = typeof value === 'number'
  return (
    <div className="flex items-center justify-between gap-2 text-[11px]">
      <span className="text-ink-500" title={hint}>
        {label}
      </span>
      {has ? (
        <span className="flex items-center gap-2">
          <ProgressBar value={value as number} width={54} height={5} color={sourceColor('eval')} />
          <span className="w-[42px] text-right tabular-nums text-ink-700">
            {formatValue(value, 'ratio')}
          </span>
        </span>
      ) : (
        // ★ 无样本时给"—"而不是 0%：0% 会被读成"成绩很好"，
        //   而事实是"还没有数据"。后端也是这个口径（返回 null）。
        <span className="text-ink-300" title="没有可用样本">
          —
        </span>
      )}
    </div>
  )
}

function Section({
  title,
  icon,
  children,
  right,
}: {
  title: string
  icon?: React.ReactNode
  children: React.ReactNode
  right?: React.ReactNode
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <h2 className="text-[13px] font-medium text-ink-700">{title}</h2>
        {right && <span className="ml-auto text-[11px] text-ink-400">{right}</span>}
      </div>
      {children}
    </section>
  )
}

export default function EvalDashboard() {
  const [offline, setOffline] = useState<OfflineEvalSnapshot | null>(null)
  const [online, setOnline] = useState<OnlineEvalSnapshot | null>(null)
  const [err, setErr] = useState<string>('')
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setErr('')
    // 两个端点互不依赖，**并行取**：串行会白等一倍时间
    Promise.allSettled([fetchOfflineEval(), fetchOnlineEval()])
      .then(([o, n]) => {
        if (o.status === 'fulfilled') setOffline(o.value)
        else setErr((prev) => `${prev}${prev ? '；' : ''}离线评测：${o.reason}`)
        if (n.status === 'fulfilled') setOnline(n.value)
        else setErr((prev) => `${prev}${prev ? '；' : ''}在线评测：${n.reason}`)
      })
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  // ★ 定时轮询 + 切回标签页时立即刷新
  //
  // 为什么必须加（实测反馈）：原先只在挂载时取一次，用户去「智能问答」提完问
  // 再切回本页，看到的还是挂载那一刻的数据 —— 表现为「提了问题但看板不动」。
  //
  // 间隔 15 秒：在线评测是**采样**写入（默认 20%），再快也大概率没有新样本；
  // 而 /api/eval/online 每次都要读 jsonl 尾部并聚合，太快会给后端添负担。
  //
  // 另监听 visibilitychange：从其他标签页切回来时立即刷一次，
  // 免得「刚切回来还要等最多 15 秒」。
  //
  // 静默刷新：不置 loading、不清 err —— 否则按钮每 15 秒转一圈、错误提示闪烁。
  useEffect(() => {
    const silent = () => {
      Promise.allSettled([fetchOfflineEval(), fetchOnlineEval()]).then(([o, n]) => {
        if (o.status === 'fulfilled') setOffline(o.value)
        if (n.status === 'fulfilled') setOnline(n.value)
      })
    }
    const timer = window.setInterval(silent, 15000)
    const onVisible = () => {
      if (document.visibilityState === 'visible') silent()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  // 按 suite 分组：指标多时平铺会看不出结构
  const grouped = new Map<string, OfflineMetric[]>()
  for (const m of offline?.metrics ?? []) {
    const arr = grouped.get(m.suite) ?? []
    arr.push(m)
    grouped.set(m.suite, arr)
  }
  const suiteLabel = (s: string) => offline?.suite_labels?.[s] ?? s

  return (
    <div className="mx-auto max-w-5xl space-y-3 p-4">
      <header className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-ink-600" />
        <h1 className="text-lg font-medium text-ink-800">评估看板</h1>
        <button
          onClick={load}
          disabled={loading}
          className="ml-auto flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-[11px] text-ink-600 hover:bg-slate-50 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
          刷新
        </button>
      </header>

      {err && (
        <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{err}</span>
        </div>
      )}

      {/* ---------- 离线评测 ---------- */}
      <Section
        title="离线评测"
        icon={<Activity className="h-4 w-4 text-ink-400" />}
        right={
          offline
            ? `${offline.run_count} 次运行 · 最新 ${offline.latest.run_id}`
            : undefined
        }
      >
        {!offline ? (
          <p className="text-[11px] text-ink-400">
            {loading ? '加载中…' : '没有离线评测数据'}
          </p>
        ) : (
          <div className="space-y-3">
            {/* 最新一次运行的结论 */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-500">
              <span>
                结论{' '}
                <b
                  className={
                    offline.latest.verdict === 'pass' ? 'text-emerald-600' : 'text-amber-600'
                  }
                >
                  {offline.latest.verdict || '未知'}
                </b>
              </span>
              <span>版本 {offline.latest.version || '—'}</span>
              {Object.entries(offline.latest.case_counts).map(([k, v]) => (
                <span key={k}>
                  {k} 用例 {v}
                </span>
              ))}
            </div>

            {/* 按 suite 分组的指标卡 */}
            {[...grouped.entries()].map(([suite, items]) => (
              <div key={suite}>
                <div className="mb-1 text-[11px] font-medium text-ink-500">
                  {suiteLabel(suite)}
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                  {items.map((m) => (
                    <MetricCard key={m.metric} m={m} />
                  ))}
                </div>
              </div>
            ))}

            {/* 趋势：每个比例指标一条 sparkline */}
            {offline.capabilities.trend && Object.keys(offline.series).length > 0 && (
              <div>
                <div className="mb-1 text-[11px] font-medium text-ink-500">
                  指标趋势（{offline.run_count} 次运行，从左到右）
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
                  {offline.metrics.map((m) => {
                    const pts = offline.series[m.metric] ?? []
                    return (
                      <div key={m.metric} className="flex items-center justify-between gap-2">
                        <span className="truncate text-[10px] text-ink-500" title={m.metric}>
                          {m.label}
                        </span>
                        <Sparkline
                          values={pts.map((p) => p.value)}
                          color={statusStyle(m.status).color}
                        />
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 逐用例表格：如实说明为什么没有 */}
            {!offline.capabilities.per_case && (
              <div className="flex items-start gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-ink-500">
                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-400" />
                <span>
                  <b className="text-ink-600">没有逐用例明细</b>
                  {offline.capabilities.per_case_reason
                    ? `：${offline.capabilities.per_case_reason}`
                    : '。'}
                </span>
              </div>
            )}
          </div>
        )}
      </Section>

      {/* ---------- 裁判一致性 ---------- */}
      {offline?.capabilities.judge_agreement && (
        <Section title="裁判一致性（test-retest）" icon={<Activity className="h-4 w-4 text-ink-400" />}>
          <div className="space-y-2 text-[11px]">
            <div className="text-ink-500">
              {offline.judge_agreement.cases ?? '—'} 个用例 ×{' '}
              {offline.judge_agreement.repeats ?? '—'} 次重复
            </div>
            <div className="overflow-hidden rounded border border-slate-200">
              <table className="w-full text-[11px]">
                <thead className="bg-slate-50 text-ink-500">
                  <tr>
                    <th className="px-2 py-1 text-left font-normal">裁判模型</th>
                    <th className="px-2 py-1 text-left font-normal">服务商</th>
                    <th className="px-2 py-1 text-right font-normal">失败次数</th>
                  </tr>
                </thead>
                <tbody>
                  {offline.judge_agreement.judges.map(([prov, model]) => {
                    const f = offline.judge_agreement.failures?.[model] ?? 0
                    return (
                      <tr key={`${prov}-${model}`} className="border-t border-slate-100">
                        <td className="px-2 py-1 text-ink-700">{model}</td>
                        <td className="px-2 py-1 text-ink-500">{prov}</td>
                        <td
                          className={`px-2 py-1 text-right tabular-nums ${
                            f > 0 ? 'text-amber-600' : 'text-ink-400'
                          }`}
                        >
                          {f}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-[10px] text-ink-400">
              失败次数 &gt; 0 说明该裁判有调用失败，其一致性结论的样本量少于其它裁判。
            </p>
          </div>
        </Section>
      )}

      {/* ---------- 在线评测 ---------- */}
      <Section
        title="在线评测（真实请求采样）"
        icon={<Activity className="h-4 w-4 text-ink-400" />}
        right={
          online
            ? `样本 ${online.evaluable_count ?? 0} / ${online.sample_count ?? 0}`
            : undefined
        }
      >
        {!online ? (
          <p className="text-[11px] text-ink-400">{loading ? '加载中…' : '没有在线评测数据'}</p>
        ) : online.error ? (
          <p className="text-[11px] text-amber-600">{online.error}</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
              <RateRow label="核对通过率" value={online.verify_ok_rate} />
              <RateRow label="无来源比例" value={online.no_source_rate} />
              <RateRow label="联网兜底比例" value={online.web_fallback_rate} />
              <RateRow label="检索缓存命中率" value={online.rag_cache_hit_rate} />
              <RateRow
                label="多轮比例"
                value={online.agent_multi_round_rate}
                hint="需要 >1 轮的比例 —— 打转的量化指标"
              />
              <RateRow label="未调工具比例" value={online.agent_no_tool_rate} />
              <RateRow label="重复调用比例" value={online.agent_repeat_rate} />
              <RateRow label="工具失败比例" value={online.agent_tool_failure_rate} />
              <RateRow
                label="提供商降级比例"
                value={online.provider_switch_rate}
                hint=">0 说明主模型不可用、答案来自备用模型"
              />
              <RateRow label="上下文压缩比例" value={online.compaction_rate} />
            </div>

            <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-ink-500">
              {online.agent_tool_calls && (
                <span>
                  平均工具调用{' '}
                  <b className="text-ink-700">{online.agent_tool_calls.mean.toFixed(2)}</b>
                </span>
              )}
              {online.agent_rounds && (
                <span>
                  平均轮次 <b className="text-ink-700">{online.agent_rounds.mean.toFixed(2)}</b>
                </span>
              )}
              {online.relevance_top1 && (
                <span>
                  相关度 top1 均值{' '}
                  <b className="text-ink-700">{online.relevance_top1.mean.toFixed(3)}</b>
                </span>
              )}
              {online.duration_ms && (
                <span>
                  耗时中位 <b className="text-ink-700">{(online.duration_ms.mean / 1000).toFixed(1)}s</b>
                </span>
              )}
            </div>

            {/* ★ 口径提醒必须显示：在线评测没有参考答案，所有数字都是代理信号 */}
            {(online.caveats?.length ?? 0) > 0 && (
              <div className="flex items-start gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] text-ink-500">
                <Info className="mt-0.5 h-3 w-3 shrink-0 text-ink-400" />
                <ul className="list-disc space-y-0.5 pl-3">
                  {online.caveats!.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </div>
            )}

            {(online.excluded_count ?? 0) > 0 && (
              <p className="text-[10px] text-ink-400">
                已排除 {online.excluded_count} 条：{online.excluded_note}
              </p>
            )}
          </div>
        )}
      </Section>

      <p className="pb-2 text-[10px] text-ink-300" style={{ color: TICK_FAINT_COLOR }}>
        <span style={{ color: GRID_COLOR }}>·</span> 数据来自 <code>eval/</code> 落盘产物与真实请求采样
      </p>
    </div>
  )
}
