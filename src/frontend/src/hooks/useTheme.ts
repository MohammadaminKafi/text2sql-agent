import { useEffect } from 'react'
import { useLocalStorage } from './useLocalStorage'

export type Theme = 'dark' | 'light' | 'system'

export function useTheme() {
  const [theme, setTheme] = useLocalStorage<Theme>('theme', 'system')

  useEffect(() => {
    const root = document.documentElement
    const apply = () => {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      const forceDark = theme === 'dark' || (theme === 'system' && prefersDark)
      root.classList.toggle('dark', forceDark)
    }
    apply()
    if (theme === 'system') {
      const mql = window.matchMedia('(prefers-color-scheme: dark)')
      const listener = () => apply()
      mql.addEventListener('change', listener)
      return () => mql.removeEventListener('change', listener)
    }
  }, [theme])

  return { theme, setTheme }
}