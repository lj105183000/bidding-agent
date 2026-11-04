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
 *   历史：`/api/eval/runs|compare|jobs` 读 PostgreSQL（DB 不可用时后端回退趋势文件）
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
  Info,
  Loader2,
  RefreshCw,
} from 'lucide-react'

import { EvalRunsPanel } from '../components/EvalRunsPanel'
import { Section } from '../components/EvalSection'
import { ProgressBar } from '../components/MicroBars'
import {
  formatValue,
  GRID_COLOR,
  sourceColor,
  TICK_FAINT_COLOR,
} from '../lib/chartStyles'
import { statusStyle } from '../lib/evalFormat'
import {
  fetchOfflineEval,
  fetchOnlineEval,
  fetchEvalRuns,
  type EvalRunsResponse,
  type OfflineEvalSnapshot,
  type OfflineMetric,
  type OnlineEvalSnapshot,
  type OnlineMultiAgentSummary,
} from '../lib/evalApi'
import {
  DECISION_LABELS,
  FALLBACK_LABELS,
  MODE_LABELS,
  REASON_LABELS,
  ROUTER_LABELS,
  formatMs,
  formatRate,
  hasMultiAgentData,
  rowsFromCounts,
  type CountRow,
} from '../lib/multiAgentCanary'

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

/** 一组带中文标签的计数 chip；只显示 count>0 的项。 */
function CountChips({ title, rows }: { title: string; rows: CountRow[] }) {
  const visible = rows.filter((row) => row.count > 0)
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
      <span className="text-ink-500">{title}</span>
      {visible.length ? (
        visible.map((row) => (
          <span
            key={row.key}
            className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-ink-600"
            title={`${(row.share * 100).toFixed(1)}%`}
          >
            {row.label} <b className="tabular-nums">{row.count}</b>
          </span>
        ))
      ) : (
        <span className="text-ink-300">—</span>
      )}
    </div>
  )
}

/**
 * 多 Agent 灰度面板：只消费 `/api/eval/online.multi_agent`。
 *
 * ★ 为什么单独一块而不是混进上面的比例网格
 * ------------------------------------
 * 上面的比例是"全量/单轮行为"；这里回答的是"auto 到底激活了多少、
 * 被拒的原因是什么、激活样本的延迟/成本/核对如何"。混在一起会让
 * `activated_rate` 被读成"答案质量"。
 */
function MultiAgentCanaryPanel({ ma }: { ma: OnlineMultiAgentSummary }) {
  const modeRows = rowsFromCounts(ma.mode_counts, MODE_LABELS)
  const decisionRows = rowsFromCounts(ma.decision_counts, DECISION_LABELS)
  const fallbackRows = rowsFromCounts(ma.fallback_counts, FALLBACK_LABELS)
  const reasonRows = rowsFromCounts(ma.mode_reason_counts, REASON_LABELS)
  const routerRows = rowsFromCounts(ma.router_counts, ROUTER_LABELS)
  const warningReasons = ['langgraph_unavailable', 'invalid_env'].filter(
    (key) => (ma.mode_reason_counts?.[key] ?? 0) > 0,
  )
  const order = ['off', 'auto', 'on', 'unknown']
  const modes = Object.keys(ma.by_mode ?? {}).sort(
    (a, b) => order.indexOf(a) - order.indexOf(b),
  )

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-medium text-ink-600">
          多 Agent 灰度（auto）
        </span>
        <span className="text-[10px] text-ink-400">
          {ma.sample_window
            ? `带时间戳样本 ${ma.sample_window.with_ts}`
            : '暂无时间戳样本'}
        </span>
      </div>

      {warningReasons.length > 0 && (
        <div className="mb-2 flex items-start gap-1.5 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] text-amber-700">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>
            配置了 auto 但实际未生效：
            {warningReasons
              .map((key) => `${REASON_LABELS[key] ?? key} ${ma.mode_reason_counts?.[key] ?? 0} 次`)
              .join('、')}
            。
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
        <RateRow
          label="auto 激活率"
          value={ma.activated_rate}
          hint="分母仅 auto/on 且 activated 为 bool 的样本"
        />
        <RateRow
          label="partial 比例"
          value={ma.partial_rate}
          hint="分母仅 activated=true 的样本"
        />
        <RateRow
          label="LLM 路由率"
          value={ma.routing_llm_rate}
          hint="router != rules 的占比；含 llm_fallback 尝试"
        />
        <RateRow
          label="deadline 比例"
          value={ma.deadline_rate}
          hint="多 Agent 总 deadline 触发率；分母仅 activated 样本"
        />
        <RateRow
          label="预算耗尽比例"
          value={ma.budget_rate}
          hint="8 次 LLM 预算触顶率；分母仅 activated 样本"
        />
      </div>

      <div className="mt-2 space-y-1">
        <CountChips title="模式" rows={modeRows} />
        <CountChips title="决策" rows={decisionRows} />
        <CountChips title="路由" rows={routerRows} />
        <CountChips title="回退" rows={fallbackRows} />
        <CountChips title="生效原因" rows={reasonRows} />
      </div>

      {modes.length > 0 && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-[10px] text-ink-600">
            <thead>
              <tr className="text-ink-400">
                <th className="py-0.5 pr-3 font-normal">模式</th>
                <th className="py-0.5 pr-3 font-normal">n</th>
                <th className="py-0.5 pr-3 font-normal">激活率</th>
                <th className="py-0.5 pr-3 font-normal">P50</th>
                <th className="py-0.5 pr-3 font-normal">P95</th>
                <th className="py-0.5 pr-3 font-normal">LLM 均值</th>
                <th className="py-0.5 pr-3 font-normal">critical 核对</th>
                <th className="py-0.5 pr-3 font-normal">deadline</th>
                <th className="py-0.5 pr-3 font-normal">预算</th>
                <th className="py-0.5 font-normal">无来源</th>
              </tr>
            </thead>
            <tbody>
              {modes.map((mode) => {
                const stats = ma.by_mode?.[mode]
                return (
                  <tr key={mode} className="border-t border-slate-200/70">
                    <td className="py-0.5 pr-3">{MODE_LABELS[mode] ?? mode}</td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {stats?.n ?? '—'}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatRate(stats?.activated_rate, stats?.activated_samples)}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatMs(stats?.duration_ms?.median)}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatMs(stats?.duration_ms?.p95)}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {stats?.llm_calls ? stats.llm_calls.mean.toFixed(2) : '—'}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatRate(
                        stats?.verify_critical_ok_rate,
                        stats?.verify_checked,
                      )}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatRate(stats?.deadline_rate, stats?.deadline_samples)}
                    </td>
                    <td className="py-0.5 pr-3 tabular-nums">
                      {formatRate(stats?.budget_rate, stats?.budget_samples)}
                    </td>
                    <td className="py-0.5 tabular-nums">
                      {formatRate(stats?.no_source_rate)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function EvalDashboard() {
  const [offline, setOffline] = useState<OfflineEvalSnapshot | null>(null)
  const [online, setOnline] = useState<OnlineEvalSnapshot | null>(null)
  const [runs, setRuns] = useState<EvalRunsResponse | null>(null)
  const [err, setErr] = useState<string>('')
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setErr('')
    // 三个端点互不依赖，**并行取**：串行会白等一倍时间
    Promise.allSettled([fetchOfflineEval(), fetchOnlineEval(), fetchEvalRuns()])
      .then(([o, n, r]) => {
        if (o.status === 'fulfilled') setOffline(o.value)
        else setErr((prev) => `${prev}${prev ? '；' : ''}离线评测：${o.reason}`)
        if (n.status === 'fulfilled') setOnline(n.value)
        else setErr((prev) => `${prev}${prev ? '；' : ''}在线评测：${n.reason}`)
        if (r.status === 'fulfilled') setRuns(r.value)
        else setErr((prev) => `${prev}${prev ? '；' : ''}运行历史：${r.reason}`)
      })
      .finally(() => setLoading(false))
  }

  /** 只刷新运行历史（触发任务终态后由子面板调用）。 */
  const loadRuns = () => {
    fetchEvalRuns()
      .then(setRuns)
      .catch((error: unknown) =>
        setErr(
          (prev) =>
            `${prev}${prev ? '；' : ''}运行历史：${
              error instanceof Error ? error.message : String(error)
            }`,
        ),
      )
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
      Promise.allSettled([fetchOfflineEval(), fetchOnlineEval(), fetchEvalRuns()]).then(
        ([o, n, r]) => {
          if (o.status === 'fulfilled') setOffline(o.value)
          if (n.status === 'fulfilled') setOnline(n.value)
          if (r.status === 'fulfilled') setRuns(r.value)
        },
      )
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

      {/* ---------- 运行历史与对比 ---------- */}
      <EvalRunsPanel runs={runs} loading={loading} onRefreshRuns={loadRuns} />

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

            {online.multi_agent && hasMultiAgentData(online.multi_agent) && (
              <MultiAgentCanaryPanel ma={online.multi_agent} />
            )}

            <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-ink-500">
              {/* ---------- 逐轮 LLM 裁判（RAGAS 风格）---------- */}
            {/*
              与上面那组比例指标的区别：
                上面 `核对通过率 / 无来源比例 …` 是**纯计算**得出的（每轮都有）；
                这里两条是**LLM 裁判**打分（只有被裁判的轮次才有）。
              两者口径不同，故单独成块并注明样本数，避免被误读成"整体质量"。
            */}
            <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-[11px] font-medium text-ink-600">
                  逐轮 LLM 裁判（RAGAS 风格）
                </span>
                <span className="text-[10px] text-ink-400">
                  {online.judge_count
                    ? `已评 ${online.judge_count} 轮${online.judge_failed_count ? ` · 判分失败 ${online.judge_failed_count}` : ''}`
                    : '暂无已评轮次'}
                </span>
              </div>
              {online.judge_count ? (
                <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
                  <RateRow
                    label="忠实度 faithfulness"
                    value={online.faithfulness?.mean ?? null}
                    hint="回答中的事实性内容有多少能被检索材料支持（LLM 裁判，0~1）"
                  />
                  <RateRow
                    label="相关性 relevancy"
                    value={online.relevancy?.mean ?? null}
                    hint="回答是否真正回应用户问题（LLM 裁判，0~1）"
                  />
                </div>
              ) : (
                <p className="text-[11px] text-ink-400">
                  暂无 —— 在线采样率为 20%、判分率 100%，多提几个问题后这里会出现分数。
                </p>
              )}
            </div>

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
        <span style={{ color: GRID_COLOR }}>·</span> 数据来自 <code>eval/</code> 落盘产物、PostgreSQL
        评测历史与真实请求采样
      </p>
    </div>
  )
}
