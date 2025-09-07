import { clsx } from 'clsx'

export function TextSkeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className={clsx('h-3 animate-pulse rounded bg-muted', i === lines - 1 && 'w-2/3')} />
      ))}
    </div>
  )
}

export function BlockSkeleton({ className }: { className?: string }) {
  return <div className={clsx('h-32 animate-pulse rounded-lg bg-muted', className)} />
}