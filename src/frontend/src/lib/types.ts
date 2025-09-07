export type ReportRequest = {
  prompt: string
  schema_id?: string | null
  include_plots?: boolean
}

export type Plot = {
  format: 'png'
  b64: string
  width?: number
  height?: number
  caption?: string
}

export type ReportResponse = {
  summary: string
  sql: string
  data: { columns: string[]; rows: any[][]; rowCount: number }
  plots: Plot[]
  warnings: string[]
}

export type ReportState = {
  request?: ReportRequest
  response?: ReportResponse
  error?: string | null
  isLoading: boolean
}