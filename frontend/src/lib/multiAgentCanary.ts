/**
 * 多 Agent 灰度看板的纯格式化工具。
 *
 * 放在 lib 而不是直接写在 EvalDashboard 里，是为了让计数/比例/时间格式化
 * 可以脱离 DOM 单测——这类"看起来只是排版"的代码最容易出现
 * "无样本显示 0%"的静默错误。
 */

import type { OnlineMultiAgentSummary } from './evalApi'

export interface CountRow {
  key: string
  label: string
  count: number
  share: number
}

export const MODE_LABELS: Record<string, string> = {
  off: 'off',
  auto: 'auto',
  on: 'on',
  unknown: '未知',
}

export const DECISION_LABELS: Record<string, string> = {
  off: 'off',
  activated: '已激活',
  declined_short: '短问题回退',
  declined_single_expert: '单专家回退',
  declined_no_tasks: '无路由回退',
  declined_error: '预路由异常',
  unknown: '未知',
}

export const FALLBACK_LABELS: Record<string, string> = {
  none: '正常',
  legacy: '回退旧 ReAct',
  fragment: '专家片段',
  unknown: '未知',
}

export const ROUTER_LABELS: Record<string, string> = {
  off: 'off',
  rules: '规则',
  llm: 'LLM',
  llm_fallback: 'LLM 回落',
  unknown: '未知',
}

export const REASON_LABELS: Record<string, string> = {
  active: '生效',
  off: 'off',
  invalid_env: '环境值非法',
  langgraph_unavailable: 'LangGraph 缺失',
  unknown: '未知',
}

/** 把 `{key: count}` 转成带中文标签与占比的行；无数据返回空数组。 */
export function rowsFromCounts(
  counts: Record<string, number> | undefined,
  labels: Record<string, string>,
): CountRow[] {
  if (!counts) return []
  const entries = Object.entries(counts)
  const total = entries.reduce(
    (sum, [, value]) => sum + (Number.isFinite(value) ? value : 0),
    0,
  )
  return entries.map(([key, count]) => ({
    key,
    label: labels[key] ?? key,
    count,
    share: total > 0 ? count / total : 0,
  }))
}

/** `null/undefined` 显示 "—"（不是 0%）；可选展示样本数。 */
export function formatRate(
  value: number | null | undefined,
  n?: number,
): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  const pct = `${(value * 100).toFixed(1)}%`
  return typeof n === 'number' ? `${pct}（n=${n}）` : pct
}

/** 毫秒转秒；缺失显示 "—"。 */
export function formatMs(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return `${(value / 1000).toFixed(1)}s`
}

/** 只有确实有样本时才渲染面板，避免全 0 的空块占位。 */
export function hasMultiAgentData(
  ma: OnlineMultiAgentSummary | undefined,
): boolean {
  if (!ma) return false
  const counts = { ...(ma.mode_counts ?? {}), ...(ma.decision_counts ?? {}) }
  return Object.values(counts).some(
    (value) => typeof value === 'number' && value > 0,
  )
}
