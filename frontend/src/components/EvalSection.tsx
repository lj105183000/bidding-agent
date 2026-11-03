/**
 * 评估页的区块容器。
 *
 * ★ 为什么抽成组件
 * ------------
 * 看板主体与运行历史/对比面板都要用同一套边框、标题与右侧说明。
 * 之前 `Section` 是 `EvalDashboard.tsx` 的局部函数，新面板无法复用；
 * 复制一份会让两处内边距/字号慢慢分叉。
 */

import type { ReactNode } from 'react'

export function Section({
  title,
  icon,
  children,
  right,
}: {
  title: string
  icon?: ReactNode
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <h2 className="text-[13px] font-medium text-ink-700">{title}</h2>
        {right && <span className="ml-auto text-[11px] text-ink-400">{right}</span>}
      </div>
      {children}
    </section>
  )
}
