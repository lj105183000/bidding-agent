import type { ReactNode } from 'react'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
}

export default function EmptyState({ icon, title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {icon && (
        <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-ink-50 text-ink-400">
          {icon}
        </div>
      )}
      <div className="text-sm font-medium text-ink-700">{title}</div>
      {description && <div className="mt-1 text-xs text-ink-400">{description}</div>}
    </div>
  )
}
