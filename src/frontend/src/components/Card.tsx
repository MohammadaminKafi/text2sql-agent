import { ReactNode } from 'react'
import { clsx } from 'clsx'

export function Card({
  title,
  actions,
  children,
  className,
  isLoading,
  ariaBusyLabel,
}: {
  title: string
  actions?: ReactNode
  children: ReactNode
  className?: string
  isLoading?: boolean
  ariaBusyLabel?: string
}) {
  return (
    <section
      className={clsx(
        'rounded-2xl border border-border bg-card shadow-soft focus-ring transition-colors',
        className,
      )}
      aria-busy={isLoading || undefined}
      aria-live="polite"
    >
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold tracking-wide text-foreground/90">{title}</h2>
        <div className="flex items-center gap-2">{actions}</div>
      </header>
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 z-10 bg-card/60 backdrop-blur-[1px]" aria-label={ariaBusyLabel} />
        )}
        <div className="p-4">{children}</div>
      </div>
    </section>
  )
}