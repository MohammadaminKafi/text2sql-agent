import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: 'hsl(222, 22%, 9%)',
        },
        foreground: {
          DEFAULT: 'hsl(210, 20%, 96%)',
        },
        card: {
          DEFAULT: 'hsl(223, 16%, 13%)',
        },
        muted: {
          DEFAULT: 'hsl(220, 13%, 18%)'
        },
        border: 'hsl(220, 13%, 22%)',
        accent: 'hsl(208, 100%, 50%)',
        positive: 'hsl(141, 53%, 53%)',
        warning: 'hsl(38, 92%, 50%)',
        danger: 'hsl(0, 84%, 60%)'
      },
      boxShadow: {
        soft: '0 2px 24px rgba(0,0,0,0.18)'
      }
    }
  },
  plugins: []
} satisfies Config