import { Card } from './Card'
import { TextSkeleton } from './Skeleton'

export function SummaryCard({ summary, isLoading }: { summary?: string; isLoading: boolean }) {
  return (
    <Card title="Summary" isLoading={isLoading}>
      {!summary ? <TextSkeleton lines={6} /> : (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">{summary}</p>
      )}
    </Card>
  )
}