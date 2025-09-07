import { Card } from './Card'
import { BlockSkeleton } from './Skeleton'
import { useState } from 'react'

function PlotImg({ b64, caption, width, height }: { b64: string; caption?: string; width?: number; height?: number }) {
  const [ok, setOk] = useState(true)
  if (!ok) return null
  return (
    <figure className="overflow-hidden rounded-xl border border-border bg-background">
      <img
        src={`data:image/png;base64,${b64}`}
        alt={caption || 'Plot'}
        width={width}
        height={height}
        onError={() => setOk(false)}
        className="block h-auto w-full"
        loading="lazy"
      />
      {caption && <figcaption className="border-t border-border p-2 text-center text-xs text-foreground/70">{caption}</figcaption>}
    </figure>
  )
}

export function PlotsCard({ plots, isLoading }: { plots?: { b64: string; caption?: string; width?: number; height?: number }[]; isLoading: boolean }) {
  return (
    <Card title="Plots" isLoading={isLoading}>
      {!plots || plots.length === 0 ? (
        <BlockSkeleton className="h-56" />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {plots.slice(0, 4).map((p, i) => (
            <PlotImg key={i} b64={p.b64} caption={p.caption} width={p.width} height={p.height} />
          ))}
        </div>
      )}
    </Card>
  )
}