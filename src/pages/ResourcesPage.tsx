import { useState } from 'react'
import { useLang } from '../context/LangContext'
import { resources, type ResourceCategory } from '../content/resources'
import type { Tier } from '../content/types'

const categoryIcons: Record<ResourceCategory, string> = {
  course: '📚',
  youtube: '▶️',
  github: '🐙',
  paper: '📄',
}

const categoryLabels: Record<ResourceCategory, { en: string; zh: string }> = {
  course: { en: 'Course', zh: '课程' },
  youtube: { en: 'YouTube', zh: 'YouTube' },
  github: { en: 'GitHub', zh: 'GitHub' },
  paper: { en: 'Paper', zh: '论文' },
}

const tierLabels: Record<Tier, { en: string; zh: string }> = {
  junior: { en: 'Junior', zh: '初级' },
  mid: { en: 'Mid', zh: '中级' },
  senior: { en: 'Senior', zh: '高级' },
}

const tierBadgeColor: Record<Tier, string> = {
  junior: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  mid: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
  senior: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
}

export function ResourcesPage() {
  const { t, lang } = useLang()
  const [tierFilter, setTierFilter] = useState<'all' | Tier>('all')
  const [categoryFilter, setCategoryFilter] = useState<'all' | ResourceCategory>('all')
  const [freeOnly, setFreeOnly] = useState(false)

  const filtered = resources.filter(r => {
    if (tierFilter !== 'all' && r.tier !== tierFilter) return false
    if (categoryFilter !== 'all' && r.category !== categoryFilter) return false
    if (freeOnly && !r.free) return false
    return true
  })

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      {/* Hero */}
      <div className="text-center mb-12">
        <div className="text-6xl mb-4">📚</div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-3">
          {t('Resources', '学习资源')}
        </h1>
        <p className="text-gray-500 dark:text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
          {t(
            'Curated courses, papers, and tools to deepen your machine learning knowledge.',
            '精选的课程、论文和工具来加深您的机器学习知识。'
          )}
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Tier Filter */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-slate-300 mb-3">
              {t('Level', '难度级别')}
            </label>
            <div className="flex flex-wrap gap-2">
              {['all', 'junior', 'mid', 'senior'].map(tier => (
                <button
                  key={tier}
                  onClick={() => setTierFilter(tier as 'all' | Tier)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    tierFilter === tier
                      ? 'bg-emerald-600 text-white'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                  }`}
                >
                  {tier === 'all'
                    ? t('All', '全部')
                    : tierLabels[tier as Tier][lang]}
                </button>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-slate-300 mb-3">
              {t('Category', '分类')}
            </label>
            <div className="flex flex-wrap gap-2">
              {['all', 'course', 'youtube', 'github', 'paper'].map(cat => (
                <button
                  key={cat}
                  onClick={() => setCategoryFilter(cat as 'all' | ResourceCategory)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    categoryFilter === cat
                      ? 'bg-emerald-600 text-white'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                  }`}
                >
                  {cat === 'all'
                    ? t('All', '全部')
                    : categoryIcons[cat as ResourceCategory] + ' ' + categoryLabels[cat as ResourceCategory][lang]}
                </button>
              ))}
            </div>
          </div>

          {/* Free Toggle */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-slate-300 mb-3">
              {t('Pricing', '价格')}
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setFreeOnly(false)}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  !freeOnly
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                }`}
              >
                {t('All', '全部')}
              </button>
              <button
                onClick={() => setFreeOnly(true)}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  freeOnly
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                }`}
              >
                {t('Free Only', '仅免费')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Results Count */}
      <div className="mb-6 text-sm text-gray-600 dark:text-slate-400">
        {t('Showing', '显示')} <span className="font-semibold text-emerald-600 dark:text-emerald-400">{filtered.length}</span> {t('resources', '个资源')}
      </div>

      {/* Resource Grid */}
      {filtered.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(resource => (
            <a
              key={resource.id}
              href={resource.url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 hover:border-emerald-300 dark:hover:border-emerald-700 hover:shadow-md transition-all group no-underline h-full flex flex-col"
            >
              {/* Header */}
              <div className="flex items-start gap-3 mb-3">
                <span className="text-3xl flex-shrink-0">{categoryIcons[resource.category]}</span>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 dark:text-white text-sm group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors line-clamp-2">
                    {lang === 'zh' ? resource.title.zh : resource.title.en}
                  </h3>
                </div>
              </div>

              {/* Description */}
              <p className="text-xs text-gray-600 dark:text-slate-400 mb-4 line-clamp-3 flex-1">
                {lang === 'zh' ? resource.description.zh : resource.description.en}
              </p>

              {/* Author */}
              {resource.author && (
                <div className="text-xs text-gray-500 dark:text-slate-500 mb-3">
                  <span className="font-medium">{resource.author}</span>
                </div>
              )}

              {/* Badges */}
              <div className="flex items-center gap-2 flex-wrap mt-auto pt-3 border-t border-gray-200 dark:border-slate-800">
                {/* Tier Badge */}
                <span className={`text-[10px] px-2 py-1 rounded font-medium ${tierBadgeColor[resource.tier]}`}>
                  {tierLabels[resource.tier][lang]}
                </span>

                {/* Category Badge */}
                <span className="text-[10px] px-2 py-1 rounded font-medium bg-gray-100 text-gray-700 dark:bg-slate-800 dark:text-slate-300">
                  {categoryLabels[resource.category][lang]}
                </span>

                {/* Free Badge */}
                {resource.free && (
                  <span className="text-[10px] px-2 py-1 rounded font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                    {t('Free', '免费')}
                  </span>
                )}

                {/* External Link Arrow */}
                <svg viewBox="0 0 24 24" className="w-3 h-3 fill-gray-400 group-hover:fill-emerald-600 dark:group-hover:fill-emerald-400 transition-colors ml-auto flex-shrink-0">
                  <path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83l1.41 1.41L19 6.41V10h2V3h-7z"/>
                </svg>
              </div>
            </a>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-slate-400 text-sm">
            {t('No resources match your filters.', '没有符合您筛选条件的资源。')}
          </p>
        </div>
      )}
    </div>
  )
}
