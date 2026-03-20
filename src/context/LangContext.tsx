import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'

type Lang = 'en' | 'zh'

interface LangContextValue {
  lang: Lang
  toggle: () => void
  t: (en: string, zh: string) => string
}

const LangContext = createContext<LangContextValue>({
  lang: 'en',
  toggle: () => {},
  t: (en) => en,
})

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>('en')
  const toggle = () => setLang(l => l === 'en' ? 'zh' : 'en')
  const t = (en: string, zh: string) => lang === 'zh' ? zh : en
  return (
    <LangContext.Provider value={{ lang, toggle, t }}>
      {children}
    </LangContext.Provider>
  )
}

export const useLang = () => useContext(LangContext)
