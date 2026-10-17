/**
 * 图表共享样式常量。
 *
 * ★ 为什么抽这个模块
 * --------------
 * `Analysis.tsx` 与 `PriceCompare.tsx` **各自复制了一份** `tooltipStyle` 与颜色常量
 * （实测确认）。再加一个"回答里画图"的组件就是第三份 —— 三处各改各的，
 * 同一个蓝色在不同页面深浅不一，而这种不一致**很难被主动发现**
 * （没人会同时打开三个页面对比色号）。
 *
 * 故把已在用两份的值抽到这里，三个消费方共用。值本身**没有改动**，
 * 只是去掉重复。
 */

/** 主色（品牌蓝，对应 tailwind brand-500） */
export const BAR_COLOR = '#3B82F6'
/** 次级/低位色（对应 ok 绿） */
export const LOW_COLOR = '#10B981'
/** 警示色（对应 warn 橙） */
export const WARN_COLOR = '#F59E0B'
/** 其他/中性色（品牌蓝同色，用于"非重点项"） */
export const OTHER_COLOR = '#3B82F6'
/** 可疑数据（橙，语义同 WARN） */
export const SUSPECT_COLOR = '#F59E0B'

/** 网格线 / 轴线颜色（对应 ink-200） */
export const GRID_COLOR = '#E2E8F0'
/** 刻度文字颜色（对应 ink-500） */
export const TICK_COLOR = '#64748B'
/** 次要刻度文字颜色（对应 ink-400） */
export const TICK_FAINT_COLOR = '#94A3B8'

/** recharts `<Tooltip>` 的统一样式（白底、细边框、圆角、小字号） */
export const tooltipStyle = {
  fontSize: 12,
  borderRadius: 8,
  border: '1px solid #E2E8F0',
  boxShadow: '0 4px 12px rgba(15, 23, 42, 0.08)',
} as const

/**
 * 千分位格式化数值。
 *
 * 图表轴与标签共用，避免同一张图里"轴上是 12345、提示里是 12,345"。
 */
export function fmtNumber(v: unknown): string {
  const n = Number(v)
  if (!Number.isFinite(n)) return '—'
  return n.toLocaleString('zh-CN')
}

/**
 * 阶段耗时堆叠条用的调色板。
 *
 * ★ 为什么是这条**固定序列**而不是按数值取色
 * --------------------------------
 * 堆叠条里相邻两段必须能区分，而"按数值大小取色"会让两个耗时相近的阶段
 * 拿到同一个颜色 —— 边界看不出来。故按**位置**循环取色。
 *
 * 序列的选取：首段用品牌蓝（`BAR_COLOR`，与其它图表的"主项"一致），
 * 其余在蓝/绿/橙/紫之间穿插，保证相邻两段色相差足够。
 * 长度 6 足够（实测阶段数最多 5 个），超出则回绕。
 */
export const PHASE_COLORS = [
  BAR_COLOR,      // #3B82F6 蓝
  LOW_COLOR,      // #10B981 绿
  WARN_COLOR,     // #F59E0B 橙
  '#8B5CF6',      // 紫
  '#EC4899',      // 粉
  '#14B8A6',      // 青
] as const

/** 按序号取阶段色（回绕）。 */
export function phaseColor(index: number): string {
  return PHASE_COLORS[index % PHASE_COLORS.length]
}

/**
 * 相关度分档色。
 *
 * ★ 与 `SourceList.scoreTone` 的分档必须一致
 * --------------------------------------
 * 那里用 Tailwind class（`bg-emerald-500` 等），这里用十六进制给 SVG 条用。
 * 两处若不一致，会出现"胶囊文字是绿的、条是灰的"这种矛盾显示。
 * 阈值同 `scoreTone`：`>= 0.65` 绿、`>= threshold` 橙、其余灰。
 */
export const RELEVANCE_OK_COLOR = LOW_COLOR        // >= 0.65
export const RELEVANCE_WARN_COLOR = WARN_COLOR     // >= threshold
export const RELEVANCE_NEUTRAL_COLOR = '#CBD5E1'   // 其余（slate-300）

/**
 * 图表「来源」→ 色相。
 *
 * ★ 为什么需要这一层（而不是各页面自己挑颜色）
 * -------------------------------------
 * 同一个概念在不同页面必须同色，否则用户会把"评估页的紫"与"聊天页的蓝"
 * 当成两种东西。集中成一张表，并让 TS 的 `Record<ChartSource, string>` **强制补全** ——
 * 新增来源时忘记配色会在**编译期**报错，而不是运行时渲染成 undefined（透明）。
 */
export type ChartSource = 'answer' | 'eval' | 'analysis' | 'compare'

/** 来源 → 主色。`Record<...>` 保证新增来源必须在此登记。 */
export const SOURCE_COLORS: Record<ChartSource, string> = {
  /** 回答里的图表（`AnswerCharts`） */
  answer: BAR_COLOR,
  /** 评估页（`/eval`）—— 用紫色与"回答"区分开 */
  eval: '#8B5CF6',
  /** 集采分析页 */
  analysis: '#14B8A6',
  /** 比价页 */
  compare: WARN_COLOR,
}

/** 取来源色（带兜底，避免表里漏登记时渲染成透明）。 */
export function sourceColor(source: ChartSource): string {
  return SOURCE_COLORS[source] ?? BAR_COLOR
}

/**
 * 数值格式。
 *
 * ★ 为什么必须有 `ratio` 分支
 * ----------------------
 * 评估指标里大量是 0~1 的比例（忠实度 0.82、精确率 0.92）。
 * 若一律用 `Math.round(value)`，**0.82 会渲染成 "1"** ——
 * 一个看起来完全合理、但错得离谱的数字（"忠实度 1"意味着满分）。
 *
 * `ratio` 分支渲染成百分比（0.82 → "82%"），并保留一位小数以区分 0.82 与 0.83。
 */
export type ChartFormat = 'number' | 'ratio'

export function formatValue(value: unknown, format: ChartFormat = 'number'): string {
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  if (format === 'ratio') return `${(n * 100).toFixed(1)}%`
  return fmtNumber(n)
}
