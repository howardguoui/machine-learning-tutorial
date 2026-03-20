import { Link, useParams } from 'react-router-dom'
import { curriculum } from '../../content/curriculum'
import { useLang } from '../../context/LangContext'
import { useState } from 'react'

const BADGE: Record<string, { label: string; color: string }> = {
  'article':      { label: 'Article',  color: 'bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-300' },
  'visual':       { label: 'Visual',   color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300' },
  'code':         { label: 'Code',     color: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300' },
  'coming-soon':  { label: 'Soon',     color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-400' },
}

interface Props { isOpen: boolean }

export function Sidebar({ isOpen }: Props) {
  const { topicId } = useParams<{ topicId: string }>()
  const { lang } = useLang()
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [search, setSearch] = useState('')

  if (!isOpen) return null

  const toggle = (id: string) => setCollapsed(prev => ({ ...prev, [id]: !prev[id] }))

  return (
    <aside className="w-64 shrink-0 bg-white dark:bg-slate-900 border-r border-gray-200 dark:border-slate-800 flex flex-col overflow-hidden">
      {/* Search */}
      <div className="p-3 border-b border-gray-200 dark:border-slate-800">
        <div className="flex items-center gap-2 bg-gray-100 dark:bg-slate-800 rounded-lg px-3 py-2 text-sm focus-within:ring-2 focus-within:ring-emerald-500">
          <svg viewBox="0 0 24 24" className="w-4 h-4 fill-gray-400 shrink-0">
            <path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
          <input
            type="text"
            placeholder={lang === 'zh' ? '搜索主题...' : 'Search topics...'}
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-transparent border-none outline-none w-full text-sm text-gray-900 dark:text-slate-200 placeholder-gray-400 dark:placeholder-slate-500"
          />
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-2">
        {curriculum.map(chapter => {
          const q = search.toLowerCase()
          const filtered = chapter.topics.filter(t =>
            !q || t.title.en.toLowerCase().includes(q) || t.title.zh.includes(q)
          )
          if (filtered.length === 0 && q) return null

          const isCollapsed = collapsed[chapter.id] && !q
          const hasActive = chapter.topics.some(t => t.id === topicId)

          return (
            <div key={chapter.id} className="mb-1">
              <button
                onClick={() => toggle(chapter.id)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm font-semibold transition-colors hover:bg-gray-100 dark:hover:bg-slate-800 ${
                  hasActive ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-700 dark:text-slate-300'
                }`}
              >
                <span className="text-base">{chapter.icon}</span>
                <span className="flex-1 text-left">
                  {lang === 'zh' ? chapter.title.zh : chapter.title.en}
                </span>
                <svg viewBox="0 0 24 24" className={`w-4 h-4 fill-current text-gray-400 transition-transform ${isCollapsed ? '-rotate-90' : ''}`}>
                  <path d="M7 10l5 5 5-5z"/>
                </svg>
              </button>

              {!isCollapsed && (
                <div className="pl-2">
                  {filtered.map(topic => {
                    const isActive = topic.id === topicId
                    const badge = BADGE[topic.contentType] ?? BADGE['article']
                    return (
                      <Link
                        key={topic.id}
                        to={`/learn/${topic.id}`}
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg mx-1 mb-0.5 text-sm no-underline transition-colors ${
                          isActive
                            ? 'bg-emerald-600 text-white'
                            : 'text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800 hover:text-gray-900 dark:hover:text-slate-200'
                        }`}
                      >
                        <span className="text-base leading-none">{topic.emoji}</span>
                        <span className="flex-1 leading-snug text-xs">
                          {lang === 'zh' ? topic.title.zh : topic.title.en}
                        </span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${
                          isActive ? 'bg-emerald-500 text-emerald-100' : badge.color
                        }`}>
                          {badge.label}
                        </span>
                      </Link>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200 dark:border-slate-800 text-xs text-gray-400 dark:text-slate-600 text-center">
        {curriculum.flatMap(c => c.topics).length} {lang === 'zh' ? '个主题' : 'topics'}
        {' · '}
        {lang === 'zh' ? '持续更新中' : 'continuously updated'}
      </div>
    </aside>
  )
}
