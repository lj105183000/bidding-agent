/**
 * 评估页的 API 类型与取数。
 *
 * ★ 为什么类型写在这里而不是 `lib/api.ts`
 * ---------------------------------
 * `lib/api.ts` 已 900+ 行，它是**聊天链路**的契约（SSE 事件、done 帧字段）。
 * 评估页只读两个 GET 端点，独立成一个模块更清楚，
 * 也避免"改聊天契约时误动评估类型"。
 */

import { API_BASE } from './api'

/** 单个离线指标（最新一次运行的值）。 */
export interface OfflineMetric {
  metric: string
  label: string
  suite: string
  value: number
  status: string
  /** ★ 是否是 0~1 的比例。后端用**白名单**判定，不能用值域猜 ——
   *  `redundant_calls_mean` 的 0.82 是"0.82 次"不是 82%。 */
  is_ratio: boolean
}

/** 一个指标随运行的序列（画趋势线用）。 */
export interface MetricPoint {
  run_id: string
  value: number
}

/** 裁判一致性（test-retest）结果。 */
export interface JudgeAgreement {
  judges: [string, string][]
  cases: number | null
  repeats: number | null
  /** 裁判模型 → 失败次数 */
  failures: Record<string, number>
  /** 裁判模型 → 轮次 → 用例 → 各指标分数。形状较深，前端只做浅展示 */
  table: Record<string, Record<string, Record<string, Record<string, number>>>>
}

export interface OfflineEvalSnapshot {
  schema_version: number
  run_count: number
  latest: {
    run_id: string
    version: string
    verdict: string
    config: Record<string, unknown>
    case_counts: Record<string, number>
  }
  metrics: OfflineMetric[]
  series: Record<string, MetricPoint[]>
  suite_labels: Record<string, string>
  judge_agreement: JudgeAgreement
  /**
   * ★ 能力声明。前端**据此隐藏**做不到的东西，而不是画一张空表。
   *
   * `per_case: false` 表示后端没有逐用例明细 —— 原因是
   * `EvalReport` 只保存聚合后的 `MetricResult`，逐用例在 `run_all.py`
   * 内部用完即弃。要展示得先让它把逐用例也落盘。
   */
  capabilities: {
    per_case: boolean
    per_case_reason?: string
    trend: boolean
    judge_agreement: boolean
  }
  sources: Record<string, string>
}

/** 在线评测（真实请求采样）的聚合视图。字段较多，按需取用。 */
export interface OnlineEvalSnapshot {
  sample_count?: number
  evaluable_count?: number
  excluded_count?: number
  excluded_note?: string
  /** 比例类字段在无样本时是 `null` 而不是 0 */
  no_source_rate?: number | null
  verify_ok_rate?: number | null
  web_fallback_rate?: number | null
  rag_cache_hit_rate?: number | null
  relevance_top1?: { mean: number; n: number } | null
  duration_ms?: { mean: number; p95: number; n: number } | null
  rag_steps_mean?: Record<string, number>
  // ---- 逐轮 LLM 裁判（RAGAS 风格）----
  // 只统计**被裁判过**的轮次：采样率 0.2 × 判分率 1.0，故数量远少于样本总数。
  // 与 `relevance_top1`（余弦相似度，纯计算）口径不同，不可互相替代。
  judge_count?: number
  judge_failed_count?: number
  faithfulness?: { mean: number; p95?: number; n: number } | null
  relevancy?: { mean: number; p95?: number; n: number } | null
  agent_tool_calls?: { mean: number; n: number } | null
  agent_rounds?: { mean: number; n: number } | null
  agent_multi_round_rate?: number | null
  agent_no_tool_rate?: number | null
  agent_repeat_rate?: number | null
  agent_tool_failure_rate?: number | null
  agent_empty_result_rate?: number | null
  provider_switch_rate?: number | null
  compaction_rate?: number | null
  recorder?: Record<string, unknown>
  caveats?: string[]
  error?: string
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    throw new Error(`${path} 返回 HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

/** 离线评测快照。 */
export function fetchOfflineEval(): Promise<OfflineEvalSnapshot> {
  return getJson<OfflineEvalSnapshot>('/api/eval/offline')
}

/** 在线评测聚合。 */
export function fetchOnlineEval(limit = 2000): Promise<OnlineEvalSnapshot> {
  return getJson<OnlineEvalSnapshot>(`/api/eval/online?limit=${limit}`)
}
