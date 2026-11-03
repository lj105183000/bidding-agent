/**
 * 评估页共享的状态样式。
 *
 * ★ 为什么从 `EvalDashboard.tsx` 抽出来
 * --------------------------------
 * 运行历史/对比面板也要显示 `pass/fail/not_run` 的图标与颜色。若各写一份，
 * 两处对“未测”的着色迟早会不一致——而那种不一致很难被主动发现。
 */

import { Activity, CheckCircle2, XCircle } from 'lucide-react'

import { LOW_COLOR, RELEVANCE_NEUTRAL_COLOR, WARN_COLOR } from './chartStyles'

export interface StatusStyle {
  color: string
  Icon: typeof CheckCircle2
  text: string
}

/** 指标状态 → 图标与色。后端给的是 `pass` / `fail` / `not_run` / `''`。 */
export function statusStyle(status: string): StatusStyle {
  if (status === 'pass') return { color: LOW_COLOR, Icon: CheckCircle2, text: '达标' }
  if (status === 'fail') return { color: WARN_COLOR, Icon: XCircle, text: '未达标' }
  if (status === 'not_run') {
    return { color: RELEVANCE_NEUTRAL_COLOR, Icon: Activity, text: '未测' }
  }
  return { color: RELEVANCE_NEUTRAL_COLOR, Icon: Activity, text: '无门禁' }
}
