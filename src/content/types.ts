export interface TopicContent {
  id: string
  title: { en: string; zh: string }
  content: { en: string; zh: string }
  /** 'article' = text only | 'visual' = interactive diagram | 'code' = code-heavy | 'coming-soon' = placeholder */
  contentType: 'article' | 'visual' | 'code' | 'coming-soon'
}

export interface Chapter {
  id: string
  title: { en: string; zh: string }
  icon: string
  topics: TopicContent[]
}
