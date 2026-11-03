/**
 * 运行历史、两次对比与手动触发。
 *
 * ★ 为什么数据和轮询放在 `EvalDashboard`，本组件只做展开
 * ------------------------------------------------
 * 看板已有 15 秒轮询；运行历史跟着它刷新即可，不必再起一个独立轮询。
 * 本组件只负责“选哪两次、怎么显示差异、触发任务”。
 *
 * ★ 对比必须显示 caveats
 * ------------------
 * 套件集合/样本量/配置/数据集不同时，分数差异不能读成退化。后端把原因
 * 放在 `caveats` 里，这里**原样展示**，而不是只画一张漂亮的表。
 */

import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  BarChart3,
  Info,
  Loader2,
  Play,
} from 'lucide-react'

import { Section } from './EvalSection'
import { formatValue } from '../lib/chartStyles'
import {
  buildCompareGroups,
  changeLabel,
  changeTone,
  filterChangedRows,
  formatDelta,
  pickDefaultPair,
  statusChangeLabel,
  storageLabel,
} from '../lib/evalCompare'
import { statusStyle } from '../lib/evalFormat'
import {
  fetchEvalCompare,
  fetchEvalJob,
  fetchEvalJobs,
  triggerEvalJob,
  type EvalCompareResponse,
  type EvalCompareRow,
  type EvalJob,
  type EvalRunsResponse,
} from '../lib/evalApi'

interface Props {
  runs: EvalRunsResponse | null
  loading: boolean
  onRefreshRuns: () => void
}

function jobTone(status: string): string {
  if (status === 'succeeded') return 'text-emerald-600'
  if (status === 'failed' || status === 'lost') return 'text-amber-600'
  if (status === 'running') return 'text-sky-600'
  return 'text-ink-400'
}

function CompareRow({ row }: { row: EvalCompareRow }) {
  const baseStyle = statusStyle(row.base.status)
  const headStyle = statusStyle(row.head.status)
  const tone = changeTone(row.change)
  const deltaColor =
    tone === 'good'
      ? 'text-emerald-600'
      : tone === 'bad'
        ? 'text-amber-600'
        : 'text-ink-500'
  const changeColor =
    row.status_change === 'pass->fail'
      ? 'text-amber-600'
      : row.status_change === 'fail->pass'
        ? 'text-emerald-600'
        : 'text-ink-400'

  return (
    <tr className="border-t border-slate-100 align-top">
      <td className="px-2 py-1">
        <div className="text-ink-700">{row.label}</div>
        <div className="text-[10px] text-ink-400">{row.metric}</div>
      </td>
      <td className="px-2 py-1 text-right tabular-nums text-ink-600">
        {formatValue(row.base.value, row.is_ratio ? 'ratio' : 'number')}
        <div className="text-[10px]" style={{ color: baseStyle.color }}>
          {baseStyle.text}
        </div>
      </td>
      <td className="px-2 py-1 text-right tabular-nums text-ink-600">
        {formatValue(row.head.value, row.is_ratio ? 'ratio' : 'number')}
        <div className="text-[10px]" style={{ color: headStyle.color }}>
          {headStyle.text}
        </div>
      </td>
      <td className={`px-2 py-1 text-right tabular-nums ${deltaColor}`}>
        {formatDelta(row.delta, row.is_ratio)}
        <div className="text-[10px]">{changeLabel(row.change)}</div>
      </td>
      <td className="px-2 py-1 text-[10px]">
        {row.status_changed ? (
          <span className={changeColor}>{statusChangeLabel(row)}</span>
        ) : (
          <span className="text-ink-300">—</span>
        )}
        {row.note && <div className="text-ink-400">{row.note}</div>}
      </td>
    </tr>
  )
}

export function EvalRunsPanel({ runs, loading, onRefreshRuns }: Props) {
  const [base, setBase] = useState('')
  const [head, setHead] = useState('')
  const [compare, setCompare] = useState<EvalCompareResponse | null>(null)
  const [compareError, setCompareError] = useState('')
  const [compareLoading, setCompareLoading] = useState(false)
  const [changedOnly, setChangedOnly] = useState(false)
  const [jobs, setJobs] = useState<EvalJob[]>([])
  const [triggerError, setTriggerError] = useState('')
  const [triggering, setTriggering] = useState(false)

  // 父组件的回调身份可能每次渲染都变；轮询里用 ref 调用，避免 effect 反复重建。
  const refreshRef = useRef(onRefreshRuns)
  useEffect(() => {
    refreshRef.current = onRefreshRuns
  }, [onRefreshRuns])
  const refreshedJobs = useRef<Set<string>>(new Set())

  useEffect(() => {
    const list = runs?.runs ?? []
    if (list.length === 0) {
      setBase('')
      setHead('')
      return
    }
    const pair = pickDefaultPair(list)
    setBase((prev) =>
      prev && list.some((run) => run.run_id === prev)
        ? prev
        : (pair?.base ?? list[0].run_id),
    )
    setHead((prev) =>
      prev && list.some((run) => run.run_id === prev)
        ? prev
        : (pair?.head ?? list[0].run_id),
    )
  }, [runs])

  useEffect(() => {
    if (!base || !head || base === head) {
      setCompare(null)
      setCompareError('')
      return
    }
    let cancelled = false
    setCompareLoading(true)
    setCompareError('')
    fetchEvalCompare(base, head)
      .then((value) => {
        if (!cancelled) setCompare(value)
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setCompareError(error instanceof Error ? error.message : String(error))
        }
      })
      .finally(() => {
        if (!cancelled) setCompareLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [base, head])

  useEffect(() => {
    fetchEvalJobs({ limit: 5 })
      .then((payload) => setJobs(payload.jobs))
      .catch(() => {
        // 任务列表不可用不影响历史/对比；触发按钮会显示具体原因。
      })
  }, [])

  useEffect(() => {
    const active = jobs.find(
      (job) => job.status === 'queued' || job.status === 'running',
    )
    if (!active) return
    const timer = window.setInterval(() => {
      fetchEvalJob(active.job_id)
        .then((payload) => {
          setJobs((prev) =>
            [
              payload.job,
              ...prev.filter((job) => job.job_id !== payload.job.job_id),
            ].slice(0, 5),
          )
          const terminal =
            payload.job.status === 'succeeded' ||
            payload.job.status === 'failed' ||
            payload.job.status === 'lost'
          if (terminal && !refreshedJobs.current.has(payload.job.job_id)) {
            refreshedJobs.current.add(payload.job.job_id)
            refreshRef.current()
          }
        })
        .catch(() => {
          // 轮询失败保留上一状态；下一轮再试
        })
    }, 3000)
    return () => window.clearInterval(timer)
  }, [jobs])

  const trigger = runs?.capabilities.trigger
  const activeJob = jobs.find(
    (job) => job.status === 'queued' || job.status === 'running',
  )

  const onTrigger = async () => {
    if (!trigger?.enabled || triggering || activeJob) return
    setTriggering(true)
    setTriggerError('')
    try {
      const payload = await triggerEvalJob({
        suites: trigger.suites,
        limit: 0,
        note: '看板手动触发',
      })
      setJobs((prev) =>
        [payload.job, ...prev.filter((job) => job.job_id !== payload.job.job_id)].slice(
          0,
          5,
        ),
      )
      // 任务终态后会由轮询刷新运行列表；这里先刷新一次，让“进行中”立即可见。
      refreshRef.current()
    } catch (error: unknown) {
      setTriggerError(error instanceof Error ? error.message : String(error))
    } finally {
      setTriggering(false)
    }
  }

  const visibleRows =
    compare && changedOnly ? filterChangedRows(compare.metrics) : (compare?.metrics ?? [])
  const groups = buildCompareGroups(visibleRows)

  return (
    <Section
      title="运行历史与对比"
      icon={<BarChart3 className="h-4 w-4 text-ink-400" />}
      right={
        runs ? `${runs.total} 次运行 · ${storageLabel(runs.storage)}` : undefined
      }
    >
      {!runs ? (
        <p className="text-[11px] text-ink-400">
          {loading ? '加载中…' : '没有运行记录'}
        </p>
      ) : (
        <div className="space-y-3">
          {runs.storage.backend !== 'db' && (
            <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[10px] text-amber-800">
              <Info className="mt-0.5 h-3 w-3 shrink-0" />
              <span>
                数据来源：{storageLabel(runs.storage)}
                {runs.storage.reason === 'db_empty'
                  ? '。运行 `python eval/backfill_db.py` 可把趋势文件回填到数据库。'
                  : '（数据库不可用，展示的是趋势文件中的历史）。'}
              </span>
            </div>
          )}

          {runs.runs.length === 0 ? (
            <p className="text-[11px] text-ink-400">
              还没有运行记录。用 CLI 加 `--save-db`，或用下方按钮触发一次。
            </p>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[11px] text-ink-500">
                <label className="flex items-center gap-1">
                  基线
                  <select
                    value={base}
                    onChange={(event) => setBase(event.target.value)}
                    className="max-w-[240px] rounded border border-slate-200 bg-white px-1.5 py-1 text-[11px] text-ink-700"
                  >
                    {runs.runs.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {run.run_id} · {run.verdict} · {run.suites.join('/') || '—'}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-1">
                  对比
                  <select
                    value={head}
                    onChange={(event) => setHead(event.target.value)}
                    className="max-w-[240px] rounded border border-slate-200 bg-white px-1.5 py-1 text-[11px] text-ink-700"
                  >
                    {runs.runs.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {run.run_id} · {run.verdict} · {run.suites.join('/') || '—'}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={changedOnly}
                    onChange={(event) => setChangedOnly(event.target.checked)}
                  />
                  只看有变化
                </label>
                {base === head && (
                  <span className="text-ink-400">请选择两次不同的运行进行对比</span>
                )}
              </div>

              {compareError && (
                <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{compareError}</span>
                </div>
              )}

              {compareLoading && (
                <p className="flex items-center gap-1 text-[11px] text-ink-400">
                  <Loader2 className="h-3 w-3 animate-spin" /> 对比加载中…
                </p>
              )}

              {compare && !compare.comparable && (
                <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>两次运行的 schema_version 不一致，指标口径可能不可比。</span>
                </div>
              )}

              {compare && compare.caveats.length > 0 && (
                <div className="flex items-start gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] text-ink-500">
                  <Info className="mt-0.5 h-3 w-3 shrink-0 text-ink-400" />
                  <ul className="list-disc space-y-0.5 pl-3">
                    {compare.caveats.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {compare && groups.length > 0 && (
                <div className="overflow-x-auto rounded border border-slate-200">
                  <table className="w-full min-w-[620px] text-[11px]">
                    <thead className="bg-slate-50 text-ink-500">
                      <tr>
                        <th className="px-2 py-1 text-left font-normal">指标</th>
                        <th className="px-2 py-1 text-right font-normal">基线</th>
                        <th className="px-2 py-1 text-right font-normal">对比</th>
                        <th className="px-2 py-1 text-right font-normal">Δ</th>
                        <th className="px-2 py-1 text-left font-normal">判定</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groups.map(([suite]) => (
                        <tr key={suite} className="border-t border-slate-100 bg-slate-50/60">
                          <td colSpan={5} className="px-2 py-1 text-[10px] font-medium text-ink-500">
                            {suite}
                          </td>
                        </tr>
                      )).flatMap((header, index) => {
                        const suite = groups[index][0]
                        return [header, ...groups[index][1].map((row) => (
                          <CompareRow key={`${suite}-${row.metric}`} row={row} />
                        ))]
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {compare && groups.length === 0 && (
                <p className="text-[11px] text-ink-400">
                  {changedOnly ? '没有变化的指标' : '两次运行都没有指标'}
                </p>
              )}

              {compare &&
                (compare.config_diff.length > 0 || compare.dataset_diff.length > 0) && (
                  <details className="text-[10px] text-ink-500">
                    <summary className="cursor-pointer">
                      配置/数据集差异（{compare.config_diff.length + compare.dataset_diff.length} 项）
                    </summary>
                    <ul className="mt-1 space-y-0.5 pl-4">
                      {[...compare.config_diff, ...compare.dataset_diff].map((row) => (
                        <li key={`${row.key}`}>
                          <code>{row.key}</code>：{JSON.stringify(row.base)} →{' '}
                          {JSON.stringify(row.head)}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
            </>
          )}

          <div className="rounded-md border border-slate-200 bg-slate-50/60 px-3 py-2">
            <div className="flex flex-wrap items-center gap-2">
              <Play className="h-3.5 w-3.5 text-ink-400" />
              <span className="text-[11px] font-medium text-ink-600">手动触发</span>
              <button
                type="button"
                onClick={onTrigger}
                disabled={!trigger?.enabled || triggering || Boolean(activeJob)}
                className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-ink-600 hover:bg-slate-50 disabled:opacity-50"
              >
                {triggering ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Play className="h-3 w-3" />
                )}
                运行 {trigger?.suites.join(', ') || 'retrieval_offline'}
              </button>
              {activeJob && (
                <span className="text-[10px] text-sky-600">
                  任务 {activeJob.job_id}：{activeJob.status}
                </span>
              )}
            </div>
            {trigger && !trigger.enabled && (
              <p className="mt-1 text-[10px] text-ink-400">{trigger.reason}</p>
            )}
            {triggerError && (
              <p className="mt-1 text-[10px] text-amber-600">{triggerError}</p>
            )}
          </div>

          {jobs.length > 0 && (
            <div>
              <div className="mb-1 text-[11px] font-medium text-ink-500">最近任务</div>
              <div className="space-y-1">
                {jobs.map((job) => (
                  <div
                    key={job.job_id}
                    className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-ink-500"
                  >
                    <span className="font-mono">{job.job_id}</span>
                    <span className={jobTone(job.status)}>{job.status}</span>
                    <span>{job.suites.join('/')}</span>
                    {job.exit_code !== null && <span>exit {job.exit_code}</span>}
                    {job.error && <span className="text-amber-600">{job.error}</span>}
                    <span className="text-ink-300">{job.log_path}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Section>
  )
}
