import type { ReportRequest, ReportResponse } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
let currentController: AbortController | null = null

function url(path: string) {
  // If API_BASE is '', use same-origin.
  return `${API_BASE}${path}`
}

export async function runReport(req: ReportRequest): Promise<ReportResponse> {
  // Abort any in-flight request when kicked off.
  if (currentController) currentController.abort()
  currentController = new AbortController()
  try {
    const res = await fetch(url('/api/v1/report'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: currentController.signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(text || `Request failed with ${res.status}`)
    }
    const json = (await res.json()) as ReportResponse
    return json
  } finally {
    currentController = null
  }
}

export function buildCsvUrl(params: { prompt: string; schema_id?: string | null; limit?: number }) {
  const q = new URLSearchParams()
  q.set('prompt', params.prompt)
  if (params.schema_id) q.set('schema_id', params.schema_id)
  if (params.limit != null) q.set('limit', String(params.limit))
  return url(`/api/v1/report.csv?${q.toString()}`)
}

export function abortReport() {
  if (currentController) currentController.abort()
}