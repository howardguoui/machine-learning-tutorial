import type { Chapter } from './types'

import { whatIsML, typesOfML } from './topics/fundamentals'
import { linearAlgebra, calcGradients, optimization, tensors, probability } from './topics/mathematics'
import { dataPrep, featureEngineering } from './topics/data'
import { linearRegression, decisionTrees, svm } from './topics/classical'
import { neuralNetworks, cnnRnn } from './topics/neural'
import { metrics, biasVariance } from './topics/evaluation'
import { mlPipelines, hyperparamTuning } from './topics/production'
import {
  interviewMathFoundations, interviewStatsProbability, interviewMLConcepts,
  interviewDataHandling, interviewLinearModels,
  interviewDecisionTrees, interviewModelValidation
} from './topics/interview-basics'
import {
  interviewFeatureEngineering, interviewModelEvaluation,
  interviewClassicalAlgorithms, interviewOptimization,
  interviewNeuralNetworks, interviewUnsupervised,
  interviewEnsembleAdvanced
} from './topics/interview'
import { mlToLLM } from './topics/llm-bridge'
import { lcelBasics, promptTemplates, simpleRAG } from './topics/phase1-lcel'
import { ragRetriever } from './topics/phase1-extended'
import { reactAgent, persistentMemory, toolRegistry } from './topics/phase2-agents'
import { agentExecutor, mcpFilesystem, mcpSqlite } from './topics/phase2-mcp'
import { hybridRAG, langGraphMultiAgent } from './topics/phase3-enterprise'
import { bm25Retriever, ensembleRetriever, rerankerDetails } from './topics/phase3-retrievers'
import { researcherNode, supervisorNode, writerNode, runGraph } from './topics/phase3-nodes'
import { githubMCP, slackMCP } from './topics/phase3-enterprise-mcp'
import { fastAPIProduction, selfHealingChains } from './topics/phase4-production'
import { ragEvaluation, codeSelfHeal } from './topics/phase4-evaluation'
import { sharedConfig, llmFactory, vectorStore } from './topics/shared-utilities'
import { gradioDemo } from './topics/demo-app'
import {
  llmInterviewArchitecture,
  llmInterviewEdgeCases,
  llmInterviewSystemDesign,
  llmInterviewCodePatterns,
} from './topics/llm-interview'
import { transformerArchitecture, tokenizationPromptEng } from './topics/llm-transformer'
import { peftDeepDive, scalingQuantization } from './topics/llm-advanced'

import { corpusPreprocessing, bpeTokenizer, dataPipeline, ddpSetup, lossMonitoring, checkpointRecovery, speculativeDecoding } from './topics/pretraining'
import { dataVsModelParallelism, ddpHandsOn, fsdpStrategies, pipelineParallelism, debuggingDistributed } from './topics/distributed-training'
import { kvCache, torchCompile, flashAttention, quantizationInference, vllmPatterns, latencyProfiling } from './topics/inference-optimization'
import { whyAlignment, rewardModelTraining, ppoLoop, dpoDirect, rlaif, preferenceDatasets, alignmentEval } from './topics/alignment'
import { gpuMemoryHierarchy, torchCompileInternals, tritonKernelBasics, flashAttentionKernel, pytorchProfiler, costPerformance } from './topics/gpu-optimization'
import { openaiTechnical, googleTechnical, bloombergTechnical, bloombergAlgorithmic, universalAIQuestions } from './topics/company-interview'

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
    id: 'ml-interview',
    title: { en: 'ML Interview Prep', zh: '机器学习面试准备' },
    icon: '🎯',
    topics: [
      // Basics first — math → stats → ML concepts → data → algorithms → validation
      interviewMathFoundations,
      interviewStatsProbability,
      interviewMLConcepts,
      interviewDataHandling,
      interviewLinearModels,
      interviewDecisionTrees,
      interviewModelValidation,
      // Intermediate — from original PDF analysis
      interviewFeatureEngineering,
      interviewModelEvaluation,
      interviewClassicalAlgorithms,
      interviewOptimization,
      interviewNeuralNetworks,
      interviewUnsupervised,
      interviewEnsembleAdvanced,
    ],
  },
  {
    id: 'llm-bridge',
    title: { en: 'ML → LLM Bridge', zh: 'ML到LLM的桥梁' },
    icon: '🌉',
    topics: [mlToLLM],
  },
  {
    id: 'phase1-lcel',
    title: { en: 'Phase 1: LCEL & RAG Foundations', zh: '阶段1：LCEL与RAG基础' },
    icon: '🔗',
    topics: [lcelBasics, promptTemplates, simpleRAG, ragRetriever],
  },
  {
    id: 'phase2-agents',
    title: { en: 'Phase 2: Agents & Memory', zh: '阶段2：代理与记忆' },
    icon: '🤖',
    topics: [reactAgent, persistentMemory, toolRegistry, agentExecutor, mcpFilesystem, mcpSqlite],
  },
  {
    id: 'phase3-enterprise',
    title: { en: 'Phase 3: Enterprise RAG & LangGraph', zh: '阶段3：企业RAG与LangGraph' },
    icon: '🏢',
    topics: [hybridRAG, langGraphMultiAgent, bm25Retriever, ensembleRetriever, rerankerDetails, researcherNode, supervisorNode, writerNode, runGraph],
  },
  {
    id: 'phase3-enterprise-mcp',
    title: { en: 'Phase 3: Enterprise MCP Integrations', zh: '阶段3：企业MCP集成' },
    icon: '🔌',
    topics: [githubMCP, slackMCP],
  },
  {
    id: 'phase4-production',
    title: { en: 'Phase 4: Production & Self-Healing', zh: '阶段4：生产部署与自愈' },
    icon: '⚙️',
    topics: [fastAPIProduction, selfHealingChains, ragEvaluation, codeSelfHeal],
  },
  {
    id: 'shared-utilities',
    title: { en: 'Shared Utilities & Config', zh: '共享工具与配置' },
    icon: '🔧',
    topics: [sharedConfig, llmFactory, vectorStore],
  },
  {
    id: 'demo-app',
    title: { en: 'Interactive Demo App', zh: '交互式演示应用' },
    icon: '🎮',
    topics: [gradioDemo],
  },
  {
    id: 'llm-foundations',
    title: { en: 'LLM Foundations', zh: 'LLM基础知识' },
    icon: '🔬',
    topics: [transformerArchitecture, tokenizationPromptEng],
  },
  {
    id: 'llm-optimization',
    title: { en: 'LLM Optimization & Scaling', zh: 'LLM优化与规模化' },
    icon: '⚡',
    topics: [peftDeepDive, scalingQuantization],
  },
  {
    id: 'llm-interview',
    title: { en: 'LLM Interview Prep', zh: 'LLM面试准备' },
    icon: '🧠',
    topics: [
      llmInterviewArchitecture,
      llmInterviewEdgeCases,
      llmInterviewSystemDesign,
      llmInterviewCodePatterns,
    ],
  },
  {
    id: 'pretraining',
    title: { en: 'Pretraining from Scratch', zh: '从零开始预训练' },
    icon: '🏗️',
    tier: 'senior',
    topics: [corpusPreprocessing, bpeTokenizer, dataPipeline, ddpSetup, lossMonitoring, checkpointRecovery, speculativeDecoding],
  },
  {
    id: 'distributed-training',
    title: { en: 'Distributed Training: DDP & FSDP', zh: '分布式训练：DDP与FSDP' },
    icon: '🖧',
    tier: 'mid',
    topics: [dataVsModelParallelism, ddpHandsOn, fsdpStrategies, pipelineParallelism, debuggingDistributed],
  },
  {
    id: 'inference-optimization',
    title: { en: 'Inference Optimization & Serving', zh: '推理优化与服务部署' },
    icon: '⚡',
    tier: 'mid',
    topics: [kvCache, torchCompile, flashAttention, quantizationInference, vllmPatterns, latencyProfiling],
  },
  {
    id: 'alignment',
    title: { en: 'Alignment: RLHF, DPO & RLAIF', zh: '对齐：RLHF、DPO与RLAIF' },
    icon: '🎯',
    tier: 'senior',
    topics: [whyAlignment, rewardModelTraining, ppoLoop, dpoDirect, rlaif, preferenceDatasets, alignmentEval],
  },
  {
    id: 'company-interview',
    title: { en: 'Company Interview Prep: OpenAI · Google · Bloomberg', zh: '公司面试准备：OpenAI · Google · Bloomberg' },
    icon: '🏢',
    tier: 'senior',
    topics: [openaiTechnical, googleTechnical, bloombergTechnical, bloombergAlgorithmic, universalAIQuestions],
  },
  {
    id: 'gpu-optimization',
    title: { en: 'GPU Optimization & CUDA Kernels', zh: 'GPU优化与CUDA核函数' },
    icon: '🔥',
    tier: 'senior',
    topics: [gpuMemoryHierarchy, torchCompileInternals, tritonKernelBasics, flashAttentionKernel, pytorchProfiler, costPerformance],
  },
]

export const totalTopics = curriculum.flatMap(c => c.topics).length
