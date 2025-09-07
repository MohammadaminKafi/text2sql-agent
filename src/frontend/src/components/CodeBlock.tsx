import { useEffect, useRef } from 'react'
import hljs from 'highlight.js/lib/core'
import sql from 'highlight.js/lib/languages/sql'
import 'highlight.js/styles/atom-one-dark.css'

hljs.registerLanguage('sql', sql)

export function CodeBlock({ code, language = 'sql' }: { code: string; language?: string }) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    if (ref.current) hljs.highlightElement(ref.current)
  }, [code])

  return (
    <pre className="rounded-xl border border-border bg-background p-3 text-xs leading-relaxed">
      <code ref={ref} className={`language-${language}`}>{code}</code>
    </pre>
  )
}