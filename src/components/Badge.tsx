import type { ReactNode } from 'react'

type BadgeColor = 'brand' | 'ok' | 'warn' | 'danger' | 'neutral'

interface BadgeProps {
  color?: BadgeColor
  children: ReactNode
}

const COLORS: Record<BadgeColor, string> = {
  brand: 'border-brand-100 bg-brand-50 text-brand-700',
  ok: 'border-emerald-100 bg-emerald-50 text-ok',
  warn: 'border-amber-100 bg-amber-50 text-warn',
  danger: 'border-red-100 bg-red-50 text-danger',
  neutral: 'border-ink-200 bg-ink-50 text-ink-500',
}

export default function Badge({ color = 'neutral', children }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4 ${COLORS[color]}`}
    >
      {children}
    </span>
  )
}
