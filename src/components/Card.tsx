import type { ReactNode } from 'react'

interface CardProps {
  title?: ReactNode
  subtitle?: ReactNode
  extra?: ReactNode
  className?: string
  children?: ReactNode
}

export default function Card({ title, subtitle, extra, className = '', children }: CardProps) {
  return (
    <div className={`card ${className}`}>
      {(title || subtitle || extra) && (
        <div className="flex items-start justify-between gap-3 border-b border-slate-200 px-5 py-4">
          <div className="min-w-0">
            {title && <div className="text-[15px] font-semibold text-ink-900">{title}</div>}
            {subtitle && <div className="mt-0.5 text-xs text-ink-400">{subtitle}</div>}
          </div>
          {extra && <div className="flex shrink-0 items-center gap-2">{extra}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  )
}
