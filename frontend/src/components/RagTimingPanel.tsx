/**
 * RAG 检索耗时面板。
 *
 * ★ 与「耗时明细」面板的分工
 * ----------------------
 * 已有的耗时面板回答"这一轮**整体**慢在哪"（首轮分析 / 检索与搜索 / 生成回答）。
 * 本面板回答"**检索内部**慢在哪一步"：
 *
 *     查询向量化（嵌入模型）
 *     稀疏编码（BM25）
 *     混合检索（向量库往返 + RRF 融合）
 *     精排（CrossEncoder）
 *     相关性过滤
 *
 * 实测一次检索 32.6s 里**查询向量化占 28.6s** —— 只看整体耗时是看不出这一点的，
 * 而这五步的优化方向完全不同（换模型 / 加索引 / 关精排 / 改阈值）。
 *
 * ★ 为什么显示"缓存命中"而不是把 0ms 当正常
 * ------------------------------------
 * 命中与未命中的延迟差**两个数量级**（实测 未命中 32.6s / 命中 0ms）。
 * 不标注的话，用户看到两次问答的检索耗时差几百倍会以为是抖动或故障。
 */

import { useState } from 'react'
import { ChevronDown, ChevronRight, Database, Zap } from 'lucide-react'

import type { RagTimingSummary, RagTimingStep } from '../lib/api'
import { fmtSec } from '../lib/format'

/** 步骤条的颜色：占比最大的用品牌蓝，其余用浅蓝，避免整面板一个色看不出重点。 */
const BAR_MAIN = '#3B82F6'
const BAR_REST = '#93C5FD'

export default function RagTimingPanel({ timing }: { timing?: RagTimingSummary | null }) {
  const [open, setOpen] = useState(false)
  if (!timing) return null

  const steps = timing.steps ?? []
  const biggest = steps.length > 0 ? steps[0] : null
  // 占比分母用 total_ms 而非步骤合计：未打点开销也该算进去，
  // 否则占比之和会 >100%（看起来像 bug）
  const denom = Math.max(timing.total_ms, 1)

  return (
    <div className="mt-2 overflow-hidden rounded-md border border-slate-200 bg-slate-50/70">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-[11px] font-medium text-ink-500 hover:bg-slate-100"
        aria-expanded={open}
        aria-label={open ? '收起检索耗时明细' : '展开检索耗时明细'}
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <Database className="h-3 w-3 shrink-0 text-ink-400" />
        <span className="whitespace-nowrap">
          RAG 检索 {timing.all_cached ? '（缓存命中）' : fmtSec(timing.total_ms)}
        </span>
        {timing.count > 1 && (
          <span className="shrink-0 font-normal text-ink-400">×{timing.count} 次</span>
        )}
        {/* 一眼看出大头是谁，不必展开 */}
        {biggest && timing.total_ms > 0 && !timing.all_cached && (
          <span className="truncate font-normal text-ink-400">
            · {biggest.label} {fmtSec(biggest.sum)}
          </span>
        )}
        <span className="ml-auto shrink-0 font-normal text-ink-400">
          {open ? '收起' : '明细'}
        </span>
      </button>

      {open && (
        <div className="space-y-1 border-t border-slate-200 px-3 py-2">
          {timing.all_cached ? (
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-600">
              <Zap className="h-3 w-3" />
              全部命中检索缓存，未做实际检索（0ms）
            </div>
          ) : (
            <>
              {steps.map((s: RagTimingStep, i: number) => (
                <div key={s.step} className="flex items-center gap-2 text-[11px]">
                  <span className="w-[104px] shrink-0 truncate text-ink-600" title={s.label}>
                    {s.label}
                  </span>
                  <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-200">
                    <span
                      className="block h-full rounded-full"
                      style={{
                        width: `${Math.max(2, (s.sum / denom) * 100)}%`,
                        background: i === 0 ? BAR_MAIN : BAR_REST,
                      }}
                    />
                  </span>
                  <span className="w-[52px] shrink-0 text-right tabular-nums text-ink-700">
                    {fmtSec(s.sum)}
                  </span>
                  {/* 多次检索时给出"单次最长"，否则 sum 与 max 相同（无信息量） */}
                  {timing.count > 1 && (
                    <span
                      className="w-[62px] shrink-0 text-right tabular-nums text-ink-300"
                      title="单次最长"
                    >
                      单次 {fmtSec(s.max)}
                    </span>
                  )}
                </div>
              ))}
              <div className="mt-1.5 flex items-center justify-between border-t border-slate-200 pt-1.5 text-[11px]">
                <span className="text-ink-500">检索合计</span>
                <span className="tabular-nums font-medium text-ink-700">
                  {fmtSec(timing.total_ms)}
                </span>
              </div>
              {/* ★ 条件从 `> 0` 改成 `!== 0`（自查发现的问题）
                  --------------------------------
                  `unaccounted_ms` = `total_ms - 步骤合计`，**可能为负** ——
                  `total` 与各步骤的计时窗口不完全嵌套，加上毫秒取整，
                  步骤合计可能比 total 大几毫秒。

                  负值是**有意义的信号**（说明计时窗口有重叠、某段耗时被记了两次），
                  而原条件 `> 0` 会把它静默隐藏 —— 恰恰藏起了该被看到的情况。
                  故改为只隐藏"正好为 0"（步骤已完全覆盖，无信息量），
                  并用警示色标出负值。 */}
              {timing.unaccounted_ms !== 0 && (
                <div className="flex items-center justify-between text-[11px]">
                  <span
                    className="text-ink-400"
                    title="未打点的开销（分类、参数解析、结果组装等）。为负说明步骤合计超过了总耗时，通常是计时窗口有重叠。"
                  >
                    其他（未分步）
                  </span>
                  <span
                    className={`tabular-nums ${
                      timing.unaccounted_ms < 0 ? 'text-warn' : 'text-ink-400'
                    }`}
                  >
                    {fmtSec(timing.unaccounted_ms)}
                  </span>
                </div>
              )}
            </>
          )}

          {/* 上下文：这些数字是"几次、什么分类、有没有精排"的结果 */}
          <div className="mt-1.5 space-y-0.5 border-t border-slate-200 pt-1.5 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="text-ink-400">检索次数</span>
              <span className="tabular-nums text-ink-600">{timing.count}</span>
            </div>
            {timing.cache_hits > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-ink-400" title="命中缓存的检索几乎不耗时">
                  缓存命中
                </span>
                <span className="tabular-nums text-emerald-600">
                  {timing.cache_hits} / {timing.count}
                </span>
              </div>
            )}
            {timing.result_count > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-ink-400">召回条数</span>
                <span className="tabular-nums text-ink-600">{timing.result_count}</span>
              </div>
            )}
            {timing.categories.length > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-ink-400" title="问题分类决定检索参数（rrf_k、召回深度）">
                  问题分类
                </span>
                <span className="text-ink-600">{timing.categories.join(' / ')}</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-ink-400" title="精排是可选增益项，失败会退回混合检索顺序">
                精排
              </span>
              <span className={timing.rerank_failed > 0 ? 'text-warn' : 'text-ink-600'}>
                {timing.rerank_failed > 0
                  ? `失败 ${timing.rerank_failed} 次（已退回混合检索）`
                  : timing.reranked > 0
                    ? '已启用'
                    : '未启用'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
