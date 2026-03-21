import { useParams, Link } from 'react-router-dom'
import { curriculum } from '../content/curriculum'
import { useLang } from '../context/LangContext'
import { MarkdownRenderer } from '../components/Content/MarkdownRenderer'

export function TopicPage() {
  const { topicId } = useParams<{ topicId: string }>()
  const { lang, t } = useLang()

  const allTopics = curriculum.flatMap(c => c.topics)
  const topic = allTopics.find(t => t.id === topicId)
  const chapter = curriculum.find(c => c.topics.some(t => t.id === topicId))

  const topicIndex = allTopics.findIndex(t => t.id === topicId)
  const prevTopic = topicIndex > 0 ? allTopics[topicIndex - 1] : null
  const nextTopic = topicIndex < allTopics.length - 1 ? allTopics[topicIndex + 1] : null

  if (!topic) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl mb-4">🔍</div>
          <p className="text-gray-500 dark:text-slate-400">{t('Topic not found', '主题不存在')}</p>
          <Link to="/" className="text-emerald-600 text-sm mt-2 inline-block">{t('← Back to Home', '← 返回首页')}</Link>
        </div>
      </div>
    )
  }

  const content = lang === 'zh' ? topic.content.zh : topic.content.en
  const isComingSoon = topic.contentType === 'coming-soon' || !content

  const contentTypeMeta: Record<string, { label: string; color: string }> = {
    'visual':      { label: t('🎬 Interactive', '🎬 互动'),  color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300' },
    'code':        { label: t('💻 Code', '💻 代码'),          color: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300' },
    'coming-soon': { label: t('⏳ Coming Soon', '⏳ 即将上线'), color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-400' },
    'article':     { label: t('📄 Article', '📄 文章'),       color: 'bg-gray-100 text-gray-700 dark:bg-slate-800 dark:text-slate-300' },
  }

  const meta = contentTypeMeta[topic.contentType] ?? contentTypeMeta['article']

  return (
    <div className="max-w-3xl mx-auto px-6 py-8">
      {/* Breadcrumb */}
      {chapter && (
        <div className="flex items-center gap-2 text-sm text-gray-400 dark:text-slate-500 mb-6">
          <Link to="/" className="hover:text-emerald-600 transition-colors no-underline">{t('Home', '首页')}</Link>
          <span>›</span>
          <span>{lang === 'zh' ? chapter.title.zh : chapter.title.en}</span>
          <span>›</span>
          <span className="text-gray-600 dark:text-slate-300">{lang === 'zh' ? topic.title.zh : topic.title.en}</span>
        </div>
      )}

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {lang === 'zh' ? topic.title.zh : topic.title.en}
          </h1>
        </div>
        <div className="flex gap-2 mt-3 flex-wrap">
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${meta.color}`}>
            {meta.label}
          </span>
          {chapter && (
            <span className="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-400">
              {chapter.icon} {lang === 'zh' ? chapter.title.zh : chapter.title.en}
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      {isComingSoon ? (
        <div className="py-16 text-center">
          <div className="text-6xl mb-4">🚧</div>
          <h2 className="text-xl font-semibold text-gray-700 dark:text-slate-300 mb-2">
            {t('Content Coming Soon', '内容即将上线')}
          </h2>
          <p className="text-gray-400 dark:text-slate-500 text-sm">
            {t('This topic is being prepared. Check back soon!', '该主题内容正在整理中，敬请期待！')}
          </p>
        </div>
      ) : (
        <MarkdownRenderer content={content} />
      )}

      {/* Prev / Next */}
      <div className="flex gap-4 mt-12 pt-6 border-t border-gray-200 dark:border-slate-800">
        {prevTopic ? (
          <Link
            to={`/learn/${prevTopic.id}`}
            className="flex-1 p-4 bg-white dark:bg-slate-900 rounded-xl border border-gray-200 dark:border-slate-800 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors no-underline group"
          >
            <div className="text-xs text-gray-400 dark:text-slate-500 mb-1">{t('← Previous', '← 上一篇')}</div>
            <div className="text-sm font-medium text-gray-700 dark:text-slate-300 group-hover:text-emerald-600 dark:group-hover:text-emerald-400">
              {lang === 'zh' ? prevTopic.title.zh : prevTopic.title.en}
            </div>
          </Link>
        ) : <div className="flex-1" />}

        {nextTopic ? (
          <Link
            to={`/learn/${nextTopic.id}`}
            className="flex-1 p-4 bg-white dark:bg-slate-900 rounded-xl border border-gray-200 dark:border-slate-800 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors no-underline group text-right"
          >
            <div className="text-xs text-gray-400 dark:text-slate-500 mb-1">{t('Next →', '下一篇 →')}</div>
            <div className="text-sm font-medium text-gray-700 dark:text-slate-300 group-hover:text-emerald-600 dark:group-hover:text-emerald-400">
              {lang === 'zh' ? nextTopic.title.zh : nextTopic.title.en}
            </div>
          </Link>
        ) : <div className="flex-1" />}
      </div>
    </div>
  )
}
