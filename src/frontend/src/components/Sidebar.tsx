import { useState, useEffect } from 'react'

interface LogEntry {
  timestamp: string
  level: string
  module: string
  message: string
  thread_id: string
}

interface LogsResponse {
  logs: LogEntry[]
  total_entries: number
  has_more: boolean
}

export function Sidebar({ open }: { open: boolean }) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedThread, setSelectedThread] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Fetch logs
  const fetchLogs = async (threadId?: string) => {
    try {
      setIsLoading(true)
      const url = threadId 
        ? `/api/v1/logs/recent?thread_id=${threadId}&limit=100`
        : '/api/v1/logs/recent?limit=50'
      
      const response = await fetch(url)
      const data: LogsResponse = await response.json()
      setLogs(data.logs)
    } catch (error) {
      console.error('Error fetching logs:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Auto-refresh logs
  useEffect(() => {
    if (!autoRefresh || !open) return

    fetchLogs(selectedThread || undefined)
    const interval = setInterval(() => {
      fetchLogs(selectedThread || undefined)
    }, 3000) // Refresh every 3 seconds

    return () => clearInterval(interval)
  }, [autoRefresh, open, selectedThread])

  // Initial load
  useEffect(() => {
    if (open) {
      fetchLogs()
    }
  }, [open])

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'text-red-600'
      case 'WARNING': return 'text-yellow-600'
      case 'INFO': return 'text-blue-600'
      case 'DEBUG': return 'text-gray-500'
      case 'SYSTEM_DEBUG': return 'text-purple-600'
      case 'SYSTEM_INFO': return 'text-green-600'
      case 'FLOW_DEBUG': return 'text-cyan-600'
      default: return 'text-gray-700'
    }
  }

  const formatMessage = (message: string) => {
    // Truncate long messages
    return message.length > 80 ? message.substring(0, 80) + '...' : message
  }

  return (
    <aside
      className="h-[calc(100dvh-56px)] border-r border-border bg-card transition-[width] duration-200"
      style={{ width: open ? 350 : 0, overflow: 'hidden' }}
      aria-hidden={!open}
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-foreground">Live Logs</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`text-xs px-2 py-1 rounded ${
                  autoRefresh 
                    ? 'bg-green-100 text-green-700' 
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                {autoRefresh ? 'Auto' : 'Manual'}
              </button>
              <button
                onClick={() => fetchLogs(selectedThread || undefined)}
                disabled={isLoading}
                className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded disabled:opacity-50"
              >
                {isLoading ? '...' : '🔄'}
              </button>
            </div>
          </div>
          
          {/* Thread filter */}
          <div className="text-xs">
            <button
              onClick={() => setSelectedThread(null)}
              className={`mr-2 px-2 py-1 rounded ${
                !selectedThread ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
              }`}
            >
              All
            </button>
            {logs.length > 0 && (
              <select
                value={selectedThread || ''}
                onChange={(e) => setSelectedThread(e.target.value || null)}
                className="text-xs border rounded px-1 py-1"
              >
                <option value="">All threads</option>
                {Array.from(new Set(logs.map(log => log.thread_id))).slice(0, 5).map(threadId => (
                  <option key={threadId} value={threadId}>
                    {threadId.replace('system-run-', '').substring(0, 8)}...
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {/* Logs content */}
        <div className="flex-1 overflow-y-auto p-2">
          {isLoading && logs.length === 0 ? (
            <div className="text-xs text-gray-500 p-2">Loading logs...</div>
          ) : logs.length === 0 ? (
            <div className="text-xs text-gray-500 p-2">No logs available</div>
          ) : (
            <div className="space-y-1">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className="text-xs border-l-2 border-gray-200 pl-2 py-1 hover:bg-gray-50"
                  style={{ borderLeftColor: getLevelColor(log.level).includes('red') ? '#dc2626' : 
                                           getLevelColor(log.level).includes('yellow') ? '#d97706' :
                                           getLevelColor(log.level).includes('blue') ? '#2563eb' :
                                           getLevelColor(log.level).includes('green') ? '#16a34a' :
                                           getLevelColor(log.level).includes('purple') ? '#9333ea' :
                                           getLevelColor(log.level).includes('cyan') ? '#0891b2' : '#6b7280' }}
                >
                  <div className="flex items-start gap-1">
                    <span className="text-gray-400 font-mono text-[10px] min-w-[45px]">
                      {log.timestamp}
                    </span>
                    <span className={`font-medium text-[10px] min-w-[60px] ${getLevelColor(log.level)}`}>
                      {log.level}
                    </span>
                  </div>
                  <div className="mt-0.5">
                    <div className="text-gray-600 text-[10px] truncate">
                      {log.module}
                    </div>
                    <div className="text-gray-800 text-[11px] leading-tight">
                      {formatMessage(log.message)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-2 border-t border-border text-[10px] text-gray-500">
          {logs.length > 0 && (
            <div>
              Showing {logs.length} entries
              {selectedThread && (
                <div>Thread: {selectedThread.replace('system-run-', '').substring(0, 8)}...</div>
              )}
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}