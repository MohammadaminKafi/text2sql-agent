import { Moon, Sun, PanelLeft } from 'lucide-react'
import { useTheme, Theme } from '@/hooks/useTheme'

export function TopBar({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  const { theme, setTheme } = useTheme()

  const next: Record<Theme, Theme> = { system: 'dark', dark: 'light', light: 'system' }
  const icon = theme === 'dark' ? <Moon size={18} /> : theme === 'light' ? <Sun size={18} /> : <Sun size={18} />

  return (
    <div className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-background/80 px-4 py-3 backdrop-blur">
      <div className="flex items-center gap-3">
        <button
          className="rounded-lg border border-border p-2 hover:bg-muted"
          aria-label="Toggle chat history sidebar"
          onClick={onToggleSidebar}
        >
          <PanelLeft size={18} />
        </button>
        <h1 className="text-base font-semibold tracking-tight">Text2SQL</h1>
        <span className="text-xs text-foreground/60">small‑now • scalable‑later</span>
      </div>
      <div className="flex items-center gap-2">
        <button
          className="rounded-lg border border-border px-3 py-1.5 text-xs hover:bg-muted"
          onClick={() => setTheme(next[theme])}
          aria-label={`Theme: ${theme}. Click to switch.`}
          title={`Theme: ${theme}`}
        >
          <div className="flex items-center gap-2">{icon}<span className="capitalize">{theme}</span></div>
        </button>
      </div>
    </div>
  )
}