/**
 * 回答里的数据图表。
 *
 * ★ 数据从哪来（为什么不让前端解析回答）
 * --------------------------------
 * 后端从**工具原始输出**里提取结构化数值，随 `done` 帧的 `charts` 字段下发。
 * 前端只负责画 —— 不解析回答正文。理由：
 *   · 回答是模型写的自然语言，数字可能被改写、省略甚至编造
 *   · 只有后端见过工具的原始输出，那才是唯一事实来源
 * 这与项目里"来源卡片只显示后端给的结构化 sources"是同一原则。
 *
 * ★ 为什么默认展开而耗时面板默认收起
 * ----------------------------
 * 两者定位不同：耗时是元数据（想知道时才看）；图表是**回答的一部分**
 * （用户问"各地分布"就是想看图形）。故默认展开，与"思考过程"面板一致。
 *
 * ★ 为什么横向条形图
 * --------------
 * 标签是中文短语（"打印/复印纸的网上超市采购"），纵向柱状图的 X 轴放不下，
 * 只能斜着写或截断 —— 两者都难读。横向条形的标签在左侧有整行空间。
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { AnswerChart } from '../lib/api'
import {
  BAR_COLOR,
  GRID_COLOR,
  TICK_COLOR,
  fmtNumber,
  tooltipStyle,
} from '../lib/chartStyles'

/** 单张图的高度按条数算：条目多则高一些，避免条挤在一起。
 *
 *  上下各留 24px，每条 26px，最小 120px。
 *  `ResponsiveContainer` 需要一个确定高度的父容器（这是 recharts 的要求，
 *  用百分比高度在 flex 布局里会塌成 0）。
 */
function chartHeight(bars: number): number {
  return Math.max(120, bars * 26 + 48)
}

function AnswerChartCard({ chart }: { chart: AnswerChart }) {
  const data = chart.data
  const max = Math.max(...data.map((d) => d.value), 1)

  return (
    <figure className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
      <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="text-[12px] font-medium text-ink-700">
          {chart.title || '数据图表'}
        </span>
        {chart.note && (
          <span className="text-[10px] text-ink-400">{chart.note}</span>
        )}
      </figcaption>

      {/* ★ 无障碍：图形对屏幕阅读器不可读，故同时提供**表格化的等价文本**。
          用 sr-only 而不是省略 —— 数据本身是有意义的，不能只给看得见的人。 */}
      <table className="sr-only">
        <caption>{chart.title || '数据图表'}</caption>
        <thead>
          <tr>
            <th scope="col">{chart.label_label || '类别'}</th>
            <th scope="col">{chart.value_label || '数值'}</th>
          </tr>
        </thead>
        <tbody>
          {data.map((d) => (
            <tr key={d.label}>
              <th scope="row">{d.label}</th>
              <td>{fmtNumber(d.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div style={{ height: chartHeight(data.length) }} aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 44, bottom: 4, left: 4 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke={GRID_COLOR}
            />
            <XAxis
              type="number"
              tickLine={false}
              axisLine={{ stroke: GRID_COLOR }}
              tick={{ fontSize: 10, fill: TICK_COLOR }}
              tickFormatter={(v) => fmtNumber(v)}
              /* 留出空间给右侧数值标签，否则最长的条会盖住标签 */
              domain={[0, Math.ceil(max * 1.15)]}
            />
            <YAxis
              type="category"
              dataKey="label"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: TICK_COLOR }}
              /* ★ 宽度取 160 而不是 132
                 -------------------------
                 实测：中文标签较长（"打印/复印纸的网上超市采购" 12 字），
                 132px 时左侧第一个字会被裁掉（截图里显示成
                 "印/复印纸的网上超市采购"），**看起来像数据错了**。
                 160px 约容纳 13 个汉字（11px 字号），覆盖实测数据里最长的标签。 */
              width={160}
            />
            <Tooltip
              formatter={(v) => [fmtNumber(v), chart.value_label || '数值']}
              contentStyle={tooltipStyle}
              cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
            />
            <Bar
              dataKey="value"
              radius={[0, 4, 4, 0]}
              barSize={14}
              /* 关动画：图是"数据展示"而非"入场表演"，且关掉后
                 截图/打印/无障碍阅读都更稳定 */
              isAnimationActive={false}
            >
              {data.map((d, i) => (
                <Cell key={d.label} fill={i === 0 ? BAR_COLOR : BAR_COLOR} />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v) => fmtNumber(v)}
                style={{ fontSize: 10, fill: TICK_COLOR }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  )
}

/**
 * 判断回答正文里是否已经有一张"同样数据"的表格。
 *
 * ★ 为什么需要这个（实测发现的真问题）
 * ------------------------------
 * 模型非常喜欢在回答里自己画 Markdown 表格。实测一次回答里：
 *     正文：| 排名 | 标的物 | 采购数量 |  →  5 行
 *     图表：同样的 5 条数据，同样的数值
 * **完全重复**，用户看到两遍，反而更乱。
 *
 * 判据是"命中数字够多"而非"有没有表格"：正文可能有一张**别的**表格
 * （如"法律依据对照表"），那种情况下图表仍有价值，不该被误伤。
 *
 * 用数字字符串匹配而不解析表格结构：Markdown 表格的解析已经在
 * react-markdown 里做过一遍，这里只需判断"这些数值是否已出现"。
 * 命中阈值取 **60%**：容忍正文明细不全（模型常省略几行），
 * 但避免"仅命中一两个常见数字"（如年份、条号）就误判为重复。
 */
function duplicatesInProse(chart: AnswerChart, prose: string): boolean {
  if (!prose || chart.data.length === 0) return false
  const hits = chart.data.filter((d) => {
    // 同时匹配带千分位与不带的写法（正文常写 1,234 或 1234）
    const plain = String(d.value)
    const grouped = d.value.toLocaleString('zh-CN')
    return prose.includes(plain) || prose.includes(grouped)
  }).length
  return hits / chart.data.length >= 0.6
}

export default function AnswerCharts({
  charts,
  prose,
}: {
  charts?: AnswerChart[]
  /** 回答正文，用于判断是否已含同数据的表格 */
  prose?: string
}) {
  if (!charts || charts.length === 0) return null
  const useful = charts.filter((c) => !duplicatesInProse(c, prose || ''))
  if (useful.length === 0) return null
  return (
    <div>
      {useful.map((c, i) => (
        <AnswerChartCard key={`${c.title}-${i}`} chart={c} />
      ))}
    </div>
  )
}
