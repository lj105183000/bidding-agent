/**
 * 手写 SVG 微型图表。
 *
 * ★ 为什么不引入图表库
 * ----------------
 * 这两个需求（一条堆叠占比条、一条进度条）用 SVG 各自不到 40 行即可完成。
 * 引库的代价是**包体积 + 一套新的 API 心智**，而收益（交互、缩放、图例）
 * 在这里一个都用不上。项目里 `SystemMonitor` 已经建立了同一范式
 * （连进度条都是手写的），这里沿用它。
 *
 * ★ 为什么用 SVG 而不是 div + width%
 * ------------------------------
 * 两者都能画。选 SVG 的理由是**多段堆叠**要处理边界：
 *   · div 方案每个色块要自己算百分比，浮点误差会让相邻块之间出现 1px 缝隙
 *   · SVG 用一个 `viewBox` + `width` 数值，由渲染器处理像素取整，
 *     并且能用 `<title>` 给每一段挂原生 tooltip（无需 JS）
 *
 * ★ 无状态、无动画
 * ------------
 * 这些图是**信息展示**不是装饰，动画会干扰读数。故不加过渡。
 */

import type { ReactNode } from 'react'

/** 一段占比条的输入。`key` 用于 React 列表与 tooltip。 */
export interface Segment {
  key: string
  label: string
  value: number
  color: string
}

/**
 * 一条堆叠占比条。
 *
 * ★ 全部为 0 时的处理
 * --------------
 * 不能除零（会得到 NaN，SVG 属性变成非法值、整条消失）。
 * 此时画一条空槽，并让调用方通过 `emptyHint` 说明原因 ——
 * 静默画成 0 宽会让人以为"加载失败"。
 */
export function StackedBar({
  segments,
  height = 10,
  emptyHint = '无数据',
}: {
  segments: Segment[]
  height?: number
  emptyHint?: string
}) {
  const total = segments.reduce((acc, s) => acc + (s.value > 0 ? s.value : 0), 0)
  if (total <= 0) {
    return (
      <div
        className="flex items-center justify-center rounded-sm bg-slate-100 text-[10px] text-ink-300"
        style={{ height }}
      >
        {emptyHint}
      </div>
    )
  }

  // 用百分比宽度：SVG 需要固定 viewBox 才能表达比例，而百分比宽度
  // 让它自适应容器（面板宽度不定）。故这里仍用绝对定位的 div 承载，
  // 但**每一段都带 SVG `<title>`** 以获得原生 tooltip。
  let offset = 0
  return (
    <div
      className="relative overflow-hidden rounded-sm bg-slate-100"
      style={{ height }}
      role="img"
      aria-label={segments.map((s) => `${s.label} ${Math.round((s.value / total) * 100)}%`).join('，')}
    >
      {segments.map((s) => {
        const pct = (Math.max(0, s.value) / total) * 100
        const left = offset
        offset += pct
        if (pct <= 0) return null
        return (
          <span
            key={s.key}
            className="absolute top-0 h-full"
            style={{ left: `${left}%`, width: `${pct}%`, background: s.color }}
            // 原生 tooltip：不需要 JS 状态，也不会被父容器的 overflow 裁掉
            title={`${s.label}：${Math.round(pct)}%`}
          />
        )
      })}
    </div>
  )
}

/**
 * 一条进度条（单一数值 0~1）。
 *
 * ★ 为什么要 `minPct`
 * ---------------
 * 极小的值（如 0.4%）在 48px 宽的条上只有 0.2px，**看不见** ——
 * 于是"几乎为 0"与"完全没有"在视觉上无法区分。给一个最小可见宽度
 * （默认 2%），让"有但很少"仍能看出来。
 */
export function ProgressBar({
  value,
  width = 48,
  height = 6,
  color = '#94A3B8',
  minPct = 2,
  title,
}: {
  value: number
  width?: number
  height?: number
  color?: string
  minPct?: number
  title?: string
}) {
  const clamped = Math.max(0, Math.min(1, value))
  const pct = Math.max(clamped > 0 ? minPct : 0, clamped * 100)
  return (
    <span
      className="inline-block overflow-hidden rounded-full bg-slate-200 align-middle"
      style={{ width, height }}
      title={title}
    >
      <span
        className="block h-full rounded-full"
        style={{ width: `${pct}%`, background: color }}
      />
    </span>
  )
}

/**
 * 带刻度的进度条：一条细槽 + 一个阈值标记。
 *
 * ★ 为什么单独做而不是给 `ProgressBar` 加参数
 * ------------------------------------
 * "阈值线"是**相关性判断**特有的需求（`min_relevance` 之上才算可用）。
 * 把它塞进通用组件会让 `ProgressBar` 承担它不该知道的语义。
 */
export function ThresholdBar({
  value,
  threshold,
  width = 84,
  height = 6,
  color = '#94A3B8',
  thresholdColor = '#CBD5E1',
  title,
}: {
  value: number
  threshold: number
  width?: number
  height?: number
  color?: string
  thresholdColor?: string
  title?: string
}) {
  const clamped = Math.max(0, Math.min(1, value))
  const thr = Math.max(0, Math.min(1, threshold))
  return (
    <span
      className="relative inline-block overflow-hidden rounded-full bg-slate-200 align-middle"
      style={{ width, height }}
      title={title}
    >
      <span
        className="absolute left-0 top-0 h-full rounded-full"
        style={{ width: `${clamped * 100}%`, background: color }}
      />
      {/* 阈值标记：1px 竖线。用 absolute 而非额外元素，避免影响条本身的宽度计算 */}
      <span
        className="absolute top-0 h-full w-px"
        style={{ left: `${thr * 100}%`, background: thresholdColor }}
      />
    </span>
  )
}

/** 供图例使用的小色块。 */
export function ColorDot({ color, children }: { color: string; children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="inline-block h-2 w-2 rounded-sm" style={{ background: color }} />
      {children}
    </span>
  )
}
