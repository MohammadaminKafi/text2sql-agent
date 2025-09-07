import { Card } from './Card'
import { CodeBlock } from './CodeBlock'
import { TextSkeleton } from './Skeleton'
import { Clipboard } from 'lucide-react'

export function SQLCard({ sql, isLoading }: { sql?: string; isLoading: boolean }) {
  return (
    <Card
      title="SQL"
      isLoading={isLoading}
      actions={
        <button
          className="rounded-lg border border-border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
          disabled={!sql}
          onClick={() => sql && navigator.clipboard.writeText(sql)}
          aria-label="Copy SQL to clipboard"
        >
          <div className="flex items-center gap-2"><Clipboard size={14} /> Copy</div>
        </button>
      }
    >
      {!sql ? <TextSkeleton lines={5} /> : <CodeBlock code={sql} />}
    </Card>
  )
}