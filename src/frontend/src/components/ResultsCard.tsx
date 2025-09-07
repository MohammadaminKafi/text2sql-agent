import { Card } from './Card'
import { BlockSkeleton } from './Skeleton'

function clampTo<T>(arr: T[], n: number) { return arr.slice(0, n) }

export function ResultsCard({
  columns,
  rows,
  rowCount,
  isLoading,
}: {
  columns?: string[]
  rows?: any[][]
  rowCount?: number
  isLoading: boolean
}) {
  const limitedRows = rows ? clampTo(rows, 200) : undefined
  const showing = limitedRows?.length ?? 0

  return (
    <Card title="Results" isLoading={isLoading}>
      {!columns || !limitedRows ? (
        <BlockSkeleton className="h-56" />
      ) : (
        <div>
          <div className="mb-2 text-xs text-foreground/60">Showing {showing.toLocaleString()} of {rowCount?.toLocaleString() ?? showing} rows</div>
          <div className="max-h-[50vh] overflow-auto rounded-xl border border-border">
            <table className="table-sticky w-full border-collapse text-left text-sm">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="border-b border-border px-3 py-2 font-medium text-foreground/80">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {limitedRows.map((r, i) => (
                  <tr key={i} className="odd:bg-transparent even:bg-muted/30">
                    {r.map((cell, j) => (
                      <td key={j} className="border-b border-border/60 px-3 py-2 align-top">
                        <span className="whitespace-pre-wrap text-foreground/90">{String(cell)}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  )
}