import { Card } from './Card'

export function WarningsCard({ warnings, isLoading }: { warnings?: string[]; isLoading: boolean }) {
  if (!warnings || warnings.length === 0) return null
  return (
    <Card title="Warnings" isLoading={isLoading}>
      <ul className="list-inside list-disc space-y-1 text-sm text-warning">
        {warnings.map((w, i) => (<li key={i}>{w}</li>))}
      </ul>
    </Card>
  )
}
