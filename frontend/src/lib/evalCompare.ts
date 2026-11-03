/**
 * 运行历史/对比的纯展示逻辑。
 *
 * ★ 为什么单独成模块
 * ---------------
 * 这些函数决定“默认比哪两次”“变化怎么显示”。放在组件里就只能靠渲染测试，
 * 而项目没有 jsdom 依赖；抽出来后用 vitest 直接测边界（null、0、方向、状态变化）。
 */

import type { EvalCompareRow, EvalRunSummary, EvalStorage } from './evalApi'

/** 后端按 `created_at DESC` 返回；默认 head=最新，base=上一次。 */
export function pickDefaultPair(
  runs: EvalRunSummary[],
): { base: string; head: string } | null {
  if (runs.length < 2) return null
  return { base: runs[1].run_id, head: runs[0].run_id }
}

/**
 * Δ 的显示。
 *
 * 比例指标用「百分点（pp）」而不是相对百分比：0.90 → 0.85 是 -5 个百分点，
 * 说成“-5.6%”会让读者以为是相对变化。非比例指标沿用原单位。
 */
export function formatDelta(delta: number | null, isRatio: boolean): string {
  if (delta === null || !Number.isFinite(delta)) return '—'
  if (delta === 0) return isRatio ? '0.0 pp' : '0'
  const magnitude = isRatio
    ? `${(Math.abs(delta) * 100).toFixed(1)} pp`
    : Math.abs(delta).toLocaleString('zh-CN', { maximumFractionDigits: 3 })
  return delta > 0 ? `+${magnitude}` : `-${magnitude}`
}

function statusLabel(status: string): string {
  if (status === 'pass') return '达标'
  if (status === 'fail') return '未达标'
  if (status === 'not_run') return '未测'
  return status || '未知'
}

/** 只有状态真的变化时才有文案；颜色由调用方按方向决定。 */
export function statusChangeLabel(row: EvalCompareRow): string {
  if (!row.status_changed) return ''
  return `${statusLabel(row.base.status)} → ${statusLabel(row.head.status)}`
}

export function changeLabel(change: string): string {
  switch (change) {
    case 'improved':
      return '进步'
    case 'regressed':
      return '退化'
    case 'unchanged':
      return '持平'
    case 'observed':
      return '仅观测'
    case 'unknown':
      return '未知指标'
    case 'not_comparable':
      return '不可比'
    default:
      return change || '—'
  }
}

/** 用于选颜色：只有 improved/regressed 有明确好坏。 */
export function changeTone(change: string): 'good' | 'bad' | 'neutral' {
  if (change === 'improved') return 'good'
  if (change === 'regressed') return 'bad'
  return 'neutral'
}

/** 按 suite 分组，保留后端返回的指标顺序。 */
export function buildCompareGroups(
  rows: EvalCompareRow[],
): Array<[string, EvalCompareRow[]]> {
  const groups = new Map<string, EvalCompareRow[]>()
  for (const row of rows) {
    const items = groups.get(row.suite) ?? []
    items.push(row)
    groups.set(row.suite, items)
  }
  return [...groups.entries()]
}

/** “只看有变化”：状态变了，或数值确实变了。 */
export function filterChangedRows(rows: EvalCompareRow[]): EvalCompareRow[] {
  return rows.filter(
    (row) => row.status_changed || (row.delta !== null && row.delta !== 0),
  )
}

/** 来源标记的中文说明。回退时必须让用户知道数字不是来自数据库。 */
export function storageLabel(storage: EvalStorage | null | undefined): string {
  if (!storage) return '未知来源'
  if (storage.backend === 'db') return '数据库'
  if (storage.backend === 'trend_file') {
    return storage.reason === 'db_empty'
      ? '趋势文件（库为空，待回填）'
      : '趋势文件回退'
  }
  return '无数据'
}
