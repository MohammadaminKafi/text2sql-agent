import { useState } from 'react'
import { TopBar } from '@/components/TopBar'
import { Sidebar } from '@/components/Sidebar'
import { InputCard } from '@/components/InputCard'
import { SummaryCard } from '@/components/SummaryCard'
import { SQLCard } from '@/components/SQLCard'
import { ResultsCard } from '@/components/ResultsCard'
import { PlotsCard } from '@/components/PlotsCard'
import { WarningsCard } from '@/components/WarningsCard'
import type { ReportRequest, ReportResponse } from '@/lib/types'
import { runReport } from '@/lib/api'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [state, setState] = useState<{ request?: ReportRequest; response?: ReportResponse; error?: string | null; isLoading: boolean }>({ isLoading: false })

  async function handleRun(req: ReportRequest) {
    setState((s) => ({ ...s, request: req, isLoading: true, error: null }))
    try {
      const res = await runReport(req)
      setState((s) => ({ ...s, response: res, isLoading: false }))
    } catch (e: any) {
      if (e?.name === 'AbortError') return // silently ignore aborts
      setState((s) => ({ ...s, isLoading: false, error: e?.message || 'Request failed' }))
    }
  }

  const r = state.response

  return (
    <div className="min-h-dvh">
      <TopBar onToggleSidebar={() => setSidebarOpen((v) => !v)} />
      <div className="flex">
        <Sidebar open={sidebarOpen} />
        <main className="mx-auto w-full max-w-7xl p-4">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div className="lg:col-span-12">
              <InputCard onRun={handleRun} isLoading={state.isLoading} lastRequest={state.request} error={state.error} />
            </div>
            <div className="lg:col-span-6">
              <SummaryCard summary={r?.summary} isLoading={state.isLoading} />
            </div>
            <div className="lg:col-span-6">
              <SQLCard sql={r?.sql} isLoading={state.isLoading} />
            </div>
            <div className="lg:col-span-12">
              <ResultsCard columns={r?.data.columns} rows={r?.data.rows} rowCount={r?.data.rowCount} isLoading={state.isLoading} />
            </div>
            <div className="lg:col-span-12">
              <PlotsCard plots={r?.plots} isLoading={state.isLoading} />
            </div>
            <div className="lg:col-span-12">
              <WarningsCard warnings={r?.warnings} isLoading={state.isLoading} />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}