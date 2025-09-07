import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Card } from './Card'
import { buildCsvUrl } from '@/lib/api'
import type { ReportRequest } from '@/lib/types'
import { Download, Play, RotateCcw } from 'lucide-react'

export function InputCard({
  onRun,
  isLoading,
  lastRequest,
  error,
}: {
  onRun: (req: ReportRequest) => void
  isLoading: boolean
  lastRequest?: ReportRequest
  error?: string | null
}) {
  const [prompt, setPrompt] = useState(lastRequest?.prompt ?? '')
  const [schemaId, setSchemaId] = useState<string>(lastRequest?.schema_id ?? '')
  const [includePlots, setIncludePlots] = useState<boolean>(lastRequest?.include_plots ?? true)
  const [limit, setLimit] = useState<number>(200)

  const taRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    if (lastRequest) {
      setPrompt(lastRequest.prompt)
      setSchemaId(lastRequest.schema_id ?? '')
      setIncludePlots(!!lastRequest.include_plots)
    }
  }, [lastRequest])

  const run = useCallback(() => {
    if (!prompt.trim()) return
    onRun({ prompt: prompt.trim(), schema_id: schemaId.trim() || undefined, include_plots: includePlots })
  }, [prompt, schemaId, includePlots, onRun])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().includes('MAC')
      if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 'enter') {
        e.preventDefault()
        run()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [run])

  const csvHref = useMemo(() => buildCsvUrl({ prompt: prompt || lastRequest?.prompt || '', schema_id: schemaId || lastRequest?.schema_id, limit }), [prompt, schemaId, lastRequest, limit])

  return (
    <Card
      title="Prompt"
      isLoading={isLoading}
      ariaBusyLabel="Running query…"
      actions={
        <div className="flex items-center gap-2">
          <button
            onClick={run}
            className="rounded-lg border border-border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
            disabled={!prompt.trim() || isLoading}
            aria-label="Run (Ctrl/Cmd+Enter)"
          >
            <div className="flex items-center gap-2"><Play size={14} /> Run</div>
          </button>
          <button
            onClick={() => { setPrompt(''); setSchemaId(''); setIncludePlots(true); taRef.current?.focus() }}
            className="rounded-lg border border-border px-3 py-1.5 text-xs hover:bg-muted"
            aria-label="Reset"
          >
            <div className="flex items-center gap-2"><RotateCcw size={14} /> Reset</div>
          </button>
          <a
            href={csvHref}
            className="rounded-lg border border-border px-3 py-1.5 text-xs hover:bg-muted aria-disabled:opacity-50"
            aria-disabled={!prompt.trim() && !lastRequest?.prompt}
            onClick={(e) => { if (!prompt.trim() && !lastRequest?.prompt) e.preventDefault() }}
          >
            <div className="flex items-center gap-2"><Download size={14} /> Download CSV</div>
          </a>
        </div>
      }
    >
      {error && (
        <div role="alert" className="mb-3 rounded-lg border border-danger/40 bg-danger/10 p-3 text-sm text-danger">
          {error}
        </div>
      )}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="sm:col-span-2">
          <label className="mb-1 block text-xs text-foreground/70" htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            ref={taRef}
            rows={5}
            className="focus-ring w-full resize-y rounded-xl border border-border bg-background p-3 text-sm placeholder:text-foreground/50"
            placeholder="e.g., Show total sales by month for 2024 for the retail schema"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="mt-1 text-xs text-foreground/60">Tip: Press <kbd className="rounded bg-muted px-1">Ctrl/Cmd</kbd> + <kbd className="rounded bg-muted px-1">Enter</kbd> to run</div>
        </div>
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-foreground/70" htmlFor="schema">Schema ID (optional)</label>
            <input id="schema" value={schemaId} onChange={(e) => setSchemaId(e.target.value)} className="focus-ring w-full rounded-xl border border-border bg-background p-2.5 text-sm" />
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border bg-background p-2.5 text-sm">
            <label htmlFor="plots" className="text-foreground/80">Include plots</label>
            <input id="plots" type="checkbox" checked={includePlots} onChange={(e) => setIncludePlots(e.target.checked)} />
          </div>
          <div>
            <label className="mb-1 block text-xs text-foreground/70" htmlFor="limit">CSV row limit</label>
            <input id="limit" type="number" min={1} max={2000} value={limit} onChange={(e) => setLimit(parseInt(e.target.value || '0') || 200)} className="focus-ring w-full rounded-xl border border-border bg-background p-2.5 text-sm" />
          </div>
        </div>
      </div>
    </Card>
  )
}