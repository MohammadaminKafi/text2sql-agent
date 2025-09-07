export function Sidebar({ open }: { open: boolean }) {
  return (
    <aside
      className="h-[calc(100dvh-56px)] border-r border-border bg-card transition-[width] duration-200"
      style={{ width: open ? 280 : 0, overflow: 'hidden' }}
      aria-hidden={!open}
    >
      <div className="p-4 text-sm text-foreground/80">
        <div className="mb-3 text-xs uppercase tracking-wider text-foreground/60">Chat History (stub)</div>
        <ul className="space-y-2">
          <li className="rounded-lg border border-border p-3">Coming soon…</li>
        </ul>
      </div>
    </aside>
  )
}