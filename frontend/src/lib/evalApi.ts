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

/** 通用统计块（后端 `_stat` 的形状）。 */
export interface StatBlock {
  mean: number
  median?: number
  p95?: number
  min?: number
  max?: number
  n: number
}

/** 单个有效模式的灰度聚合。 */
export interface MultiAgentModeSummary {
  n?: number
  decision_counts?: Record<string, number>
  router_counts?: Record<string, number>
  routing_llm_rate?: number | null
  deadline_count?: number
  deadline_samples?: number
  deadline_rate?: number | null
  budget_count?: number
  budget_samples?: number
  budget_rate?: number | null
  activated_rate?: number | null
  activated_samples?: number
  fallback_counts?: Record<string, number>
  partial_rate?: number | null
  partial_samples?: number
  duration_ms?: StatBlock | null
  llm_ms?: StatBlock | null
  llm_calls?: StatBlock | null
  tokens?: StatBlock | null
  verify_checked?: number
  verify_ok_rate?: number | null
  verify_critical_ok_rate?: number | null
  no_source_rate?: number | null
}

/** `/api/eval/online` 的多 Agent 灰度聚合。旧后端没有该字段。 */
export interface OnlineMultiAgentSummary {
  mode_counts?: Record<string, number>
  configured_mode_counts?: Record<string, number>
  mode_reason_counts?: Record<string, number>
  decision_counts?: Record<string, number>
  router_counts?: Record<string, number>
  routing_llm_rate?: number | null
  deadline_count?: number
  deadline_samples?: number
  deadline_rate?: number | null
  budget_count?: number
  budget_samples?: number
  budget_rate?: number | null
  activated_rate?: number | null
  activated_samples?: number
  partial_rate?: number | null
  partial_samples?: number
  fallback_counts?: Record<string, number>
  route_size?: StatBlock | null
  by_mode?: Record<string, MultiAgentModeSummary>
  sample_window?: {
    with_ts: number
    oldest: number | null
    latest: number | null
  }
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
  llm_calls?: StatBlock | null
  llm_calls_total?: number
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
  /** 多 Agent 灰度聚合；旧后端没有该字段时前端不渲染面板。 */
  multi_agent?: OnlineMultiAgentSummary
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

/* -------------------------------------------------------------------------
 * 轻量评测中台：运行历史 / 对比 / 任务
 *
 * ★ 为什么追加而不是改写上面的类型
 * --------------------------
 * `/api/eval/offline` 是既有看板契约（指标转置 + 裁判一致性）；运行历史是
 * 另一组端点，形状不同。混在一组类型里会让“改历史接口时误动离线快照”。
 * ---------------------------------------------------------------------- */

export type EvalStorageBackend = 'db' | 'trend_file' | 'none'

/** 数据来源标记：前端据此如实展示“数字来自数据库还是旧趋势文件”。 */
export interface EvalStorage {
  backend: EvalStorageBackend
  degraded: boolean
  reason: string
  db_available: boolean
  file_available: boolean
}

export interface EvalTriggerCapability {
  enabled: boolean
  suites: string[]
  reason: string
}

export interface EvalCapabilities {
  per_case: boolean
  per_case_reason?: string
  compare?: boolean
  trigger?: EvalTriggerCapability
}

/** 运行列表里的一条（聚合级，不含逐指标）。 */
export interface EvalRunSummary {
  run_id: string
  created_at: string | null
  version: string
  schema_version: number
  source: string
  job_id: string | null
  suites: string[]
  suites_source: string
  verdict: string
  case_counts: Record<string, number>
  suite_verdicts: Record<string, string>
  metric_count: number
}

/** 单个指标的详情。 */
export interface EvalMetricView {
  metric: string
  label: string
  suite: string
  value: number | null
  status: string
  threshold: number | null
  unit: string
  direction: string
  is_ratio: boolean
  note: string
}

export interface EvalRunDetail extends EvalRunSummary {
  config: Record<string, unknown>
  suite_notes: Record<string, string>
  datasets: Record<string, unknown>
  metrics: EvalMetricView[]
}

export interface EvalRunResponse {
  run: EvalRunDetail
  storage: EvalStorage
}

export interface EvalRunsResponse {
  runs: EvalRunSummary[]
  total: number
  limit: number
  offset: number
  storage: EvalStorage
  capabilities: EvalCapabilities
}

export interface EvalCompareValue {
  value: number | null
  status: string
  threshold: number | null
}

export interface EvalCompareRow {
  metric: string
  label: string
  suite: string
  unit: string
  direction: string
  is_ratio: boolean
  base: EvalCompareValue
  head: EvalCompareValue
  delta: number | null
  pct_delta: number | null
  status_changed: boolean
  status_change: string
  /** improved / regressed / unchanged / observed / unknown / not_comparable */
  change: string
  track_regression: boolean
  note: string
}

export interface EvalCompareSide {
  run_id: string
  version: string
  verdict: string
  created_at: string | null
  suites: string[]
  case_counts: Record<string, number>
}

export interface EvalConfigDiffRow {
  key: string
  base: unknown
  head: unknown
}

export interface EvalCompareResponse {
  base: EvalCompareSide
  head: EvalCompareSide
  comparable: boolean
  caveats: string[]
  metrics: EvalCompareRow[]
  suite_verdicts: {
    base: Record<string, string>
    head: Record<string, string>
    changed: Record<string, { base: string | null; head: string | null }>
  }
  config_diff: EvalConfigDiffRow[]
  dataset_diff: EvalConfigDiffRow[]
  new_metrics: string[]
  missing_metrics: string[]
  tolerance: number
  storage: { base: EvalStorage; head: EvalStorage }
}

export interface EvalJob {
  job_id: string
  status: string
  suites: string[]
  note: string
  requested_at: string | null
  started_at: string | null
  finished_at: string | null
  run_id: string | null
  exit_code: number | null
  error: string | null
  log_path: string
}

export interface EvalJobsResponse {
  jobs: EvalJob[]
  total: number
  limit: number
  offset: number
  storage: EvalStorage
  capabilities: EvalCapabilities
}

export interface EvalTriggerRequest {
  suites: string[]
  limit?: number
  note?: string
}

export interface EvalTriggerResponse {
  job: EvalJob
  storage: EvalStorage
}

export interface EvalJobResponse {
  job: EvalJob
  storage: EvalStorage
  capabilities: EvalCapabilities
  /** 任务产出的运行摘要（若已写入 eval_runs）。 */
  run?: EvalRunSummary
}

function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    // 触发失败的原因（未开启 / 冷却 / 存储不可用）必须展示给用户，
    // 否则按钮点了没反应，看起来像前端坏了。
    let detail = ''
    try {
      const payload = (await res.json()) as { detail?: unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
    } catch {
      // 响应不是 JSON：忽略，用状态码兜底
    }
    throw new Error(`${path} 返回 HTTP ${res.status}${detail ? `：${detail}` : ''}`)
  }
  return (await res.json()) as T
}

/** 运行历史（DB 优先，后端会自动回退趋势文件）。 */
export function fetchEvalRuns(
  params: {
    limit?: number
    offset?: number
    suite?: string
    verdict?: string
    source?: string
  } = {},
): Promise<EvalRunsResponse> {
  return getJson<EvalRunsResponse>(`/api/eval/runs${queryString(params)}`)
}

/** 单次运行详情。 */
export function fetchEvalRun(runId: string): Promise<EvalRunResponse> {
  return getJson<EvalRunResponse>(`/api/eval/runs/${encodeURIComponent(runId)}`)
}

/** 两次运行对比。 */
export function fetchEvalCompare(
  base: string,
  head: string,
  tolerance = 0.02,
): Promise<EvalCompareResponse> {
  return getJson<EvalCompareResponse>(
    `/api/eval/compare${queryString({ base, head, tolerance })}`,
  )
}

/** 评测任务列表。 */
export function fetchEvalJobs(
  params: { limit?: number; offset?: number; status?: string } = {},
): Promise<EvalJobsResponse> {
  return getJson<EvalJobsResponse>(`/api/eval/jobs${queryString(params)}`)
}

/** 单个任务状态（轮询用）。 */
export function fetchEvalJob(jobId: string): Promise<EvalJobResponse> {
  return getJson<EvalJobResponse>(`/api/eval/jobs/${encodeURIComponent(jobId)}`)
}

/** 手动触发一次评测（默认关闭；后端只允许 retrieval_offline）。 */
export function triggerEvalJob(body: EvalTriggerRequest): Promise<EvalTriggerResponse> {
  return postJson<EvalTriggerResponse>('/api/eval/trigger', body)
}
