import type { Chapter } from './types'

import { whatIsML, typesOfML } from './topics/fundamentals'
import { linearAlgebra, calcGradients, optimization, tensors, probability } from './topics/mathematics'
import { dataPrep, featureEngineering } from './topics/data'
import { linearRegression, decisionTrees, svm } from './topics/classical'
import { neuralNetworks, cnnRnn } from './topics/neural'
import { metrics, biasVariance } from './topics/evaluation'
import { mlPipelines, hyperparamTuning } from './topics/production'
import { mlToLLM } from './topics/llm-bridge'

export const curriculum: Chapter[] = [
  {
    id: 'fundamentals',
    title: { en: 'ML Fundamentals', zh: 'ML基础' },
    icon: '🤖',
    topics: [whatIsML, typesOfML],
  },
  {
    id: 'mathematics',
    title: { en: 'Mathematics of ML', zh: 'ML数学基础' },
    icon: '📐',
    topics: [linearAlgebra, calcGradients, optimization, tensors, probability],
  },
  {
    id: 'data',
    title: { en: 'Data Preparation', zh: '数据准备' },
    icon: '🧹',
    topics: [dataPrep, featureEngineering],
  },
  {
    id: 'classical',
    title: { en: 'Classical Algorithms', zh: '经典算法' },
    icon: '🌳',
    topics: [linearRegression, decisionTrees, svm],
  },
  {
    id: 'neural',
    title: { en: 'Neural Networks', zh: '神经网络' },
    icon: '🧠',
    topics: [neuralNetworks, cnnRnn],
  },
  {
    id: 'evaluation',
    title: { en: 'Model Evaluation', zh: '模型评估' },
    icon: '📊',
    topics: [metrics, biasVariance],
  },
  {
    id: 'production',
    title: { en: 'Production ML', zh: '生产ML' },
    icon: '🏭',
    topics: [mlPipelines, hyperparamTuning],
  },
  {
    id: 'llm-bridge',
    title: { en: 'ML → LLM Bridge', zh: 'ML到LLM的桥梁' },
    icon: '🌉',
    topics: [mlToLLM],
  },
]

export const totalTopics = curriculum.flatMap(c => c.topics).length
