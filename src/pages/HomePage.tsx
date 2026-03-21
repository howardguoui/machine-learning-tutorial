import { Link } from 'react-router-dom'
import { curriculum, totalTopics } from '../content/curriculum'
import { useLang } from '../context/LangContext'

export function HomePage() {
  const { t, lang } = useLang()

  const firstTopic = curriculum[0]?.topics[0]
  const readyCount = curriculum.flatMap(c => c.topics).filter(t => t.contentType !== 'coming-soon').length

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Hero */}
      <div className="text-center mb-14">
        <div className="text-6xl mb-4">🧠</div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-3">
          {t('Machine Learning Tutorial', '机器学习教程')}
        </h1>
        <p className="text-gray-500 dark:text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
          {t(
            'A production-ready guide to Machine Learning with Python — from mathematics fundamentals to deploying models at scale.',
            '使用Python学习机器学习的生产就绪指南——从数学基础到大规模部署模型。'
          )}
        </p>
        <div className="flex items-center justify-center gap-6 mt-6 text-sm text-gray-400 dark:text-slate-500 flex-wrap">
          <span>📚 {totalTopics} {t('Topics', '个主题')}</span>
          <span>✅ {readyCount} {t('Ready', '已完成')}</span>
          <span>🐍 {t('Python Code', 'Python代码')}</span>
          <span>🌍 {t('EN / 中文', '中文 / EN')}</span>
        </div>
      </div>

      {/* Feature badges */}
      <div className="flex flex-wrap gap-2 justify-center mb-12">
        {[
          { icon: '📐', label: t('Math Fundamentals', '数学基础') },
          { icon: '🧹', label: t('Data Prep', '数据准备') },
          { icon: '🌳', label: t('Classical ML', '经典算法') },
          { icon: '🧠', label: t('Deep Learning', '深度学习') },
          { icon: '📊', label: t('Evaluation', '模型评估') },
          { icon: '🏭', label: t('Production', '生产部署') },
          { icon: '🌉', label: t('LLM Bridge', 'LLM桥梁') },
        ].map(({ icon, label }) => (
          <span key={label} className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400 rounded-full text-xs font-medium border border-emerald-200 dark:border-emerald-800">
            <span>{icon}</span>
            <span>{label}</span>
          </span>
        ))}
      </div>

      {/* CTA */}
      <div className="flex gap-3 justify-center mb-14 flex-wrap">
        {firstTopic && (
          <Link
            to={`/learn/${firstTopic.id}`}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-medium text-sm transition-colors no-underline shadow-sm"
          >
            {t('Start Learning →', '开始学习 →')}
          </Link>
        )}
        <Link
          to="/playground"
          className="px-6 py-3 bg-gray-200 dark:bg-slate-800 hover:bg-gray-300 dark:hover:bg-slate-700 text-gray-800 dark:text-slate-200 rounded-xl font-medium text-sm transition-colors no-underline"
        >
          {t('⚡ Python Playground', '⚡ Python 练习场')}
        </Link>
      </div>

      {/* Learning Path */}
      <div className="mb-8">
        <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <span>🗺️</span>
          <span>{t('Learning Path', '学习路线')}</span>
        </h2>
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 flex-wrap text-sm text-gray-600 dark:text-slate-400">
          {curriculum.map((chapter, idx) => (
            <div key={chapter.id} className="flex items-center gap-2">
              <Link
                to={`/learn/${chapter.topics[0].id}`}
                className="flex items-center gap-1 px-2 py-1 rounded-lg bg-gray-100 dark:bg-slate-800 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 hover:text-emerald-700 dark:hover:text-emerald-400 transition-colors no-underline"
              >
                <span>{chapter.icon}</span>
                <span className="text-xs">{lang === 'zh' ? chapter.title.zh : chapter.title.en}</span>
              </Link>
              {idx < curriculum.length - 1 && (
                <span className="text-gray-300 dark:text-slate-700 hidden sm:block">→</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Chapter grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {curriculum.map(chapter => {
          const ready = chapter.topics.filter(t => t.contentType !== 'coming-soon').length
          const firstAvailable = chapter.topics.find(t => t.contentType !== 'coming-soon') ?? chapter.topics[0]
          return (
            <Link
              key={chapter.id}
              to={`/learn/${firstAvailable.id}`}
              className="p-5 bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 hover:border-emerald-300 dark:hover:border-emerald-700 hover:shadow-md transition-all no-underline group"
            >
              <div className="flex items-start gap-3">
                <span className="text-3xl">{chapter.icon}</span>
                <div className="flex-1 min-w-0">
                  <h2 className="font-semibold text-gray-900 dark:text-white text-base group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                    {lang === 'zh' ? chapter.title.zh : chapter.title.en}
                  </h2>
                  <p className="text-xs text-gray-500 dark:text-slate-500 mt-0.5">
                    {chapter.topics.length} {t('topics', '个主题')}
                    {ready > 0 && ` · ${ready} ${t('ready', '已完成')}`}
                  </p>
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      {/* Footer note */}
      <div className="mt-12 text-center text-xs text-gray-400 dark:text-slate-600">
        {t(
          'All Python examples are production-ready and tested. Prerequisites: Python basics, NumPy familiarity.',
          '所有Python示例均经过测试，可直接用于生产。前提条件：Python基础，熟悉NumPy。'
        )}
      </div>
    </div>
  )
}
