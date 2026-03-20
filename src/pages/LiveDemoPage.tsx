import { useState, useRef, useEffect } from 'react'
import { useLang } from '../context/LangContext'

// ── Config ─────────────────────────────────────────────────────────────────
const OLLAMA_BASE = 'http://localhost:11434'
const DEFAULT_MODEL = 'qwen3:8b'

// ── Utilities ──────────────────────────────────────────────────────────────
function stripThinking(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
}

function chunkText(text: string, size = 500, overlap = 50): string[] {
  const chunks: string[] = []
  let i = 0
  while (i < text.length) {
    chunks.push(text.slice(i, i + size))
    i += Math.max(1, size - overlap)
  }
  return chunks.filter(c => c.trim().length > 20)
}

function retrieveTopChunks(chunks: string[], query: string, k = 3): string[] {
  const words = query.toLowerCase().split(/\W+/).filter(w => w.length > 2)
  const scored = chunks.map(c => ({
    c,
    score: words.reduce((s, w) => s + (c.toLowerCase().split(w).length - 1), 0),
  }))
  return scored.sort((a, b) => b.score - a.score).slice(0, k).map(x => x.c)
}

// ── Ollama streaming ───────────────────────────────────────────────────────
interface OllamaMsg {
  role: 'system' | 'user' | 'assistant'
  content: string
}

async function streamChat(
  messages: OllamaMsg[],
  onChunk: (partial: string) => void,
  signal: AbortSignal,
): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: DEFAULT_MODEL, messages, stream: true }),
    signal,
  })
  if (!res.ok) {
    throw new Error(`Ollama ${res.status} — is Ollama running? Run: ollama serve`)
  }
  if (!res.body) throw new Error('No response body')

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let full = ''
  let buf = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const obj = JSON.parse(line)
        if (obj.message?.content) {
          full += obj.message.content
          onChunk(full)
        }
      } catch { /* skip malformed lines */ }
    }
  }
  return full
}

// ── Types ──────────────────────────────────────────────────────────────────
interface ChatMsg { role: 'user' | 'assistant'; content: string }

// ── useChat hook — shared streaming state per tab ─────────────────────────
function useChat() {
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [streaming, setStreaming] = useState<string | null>(null)
  const [loading, setLoading]   = useState(false)
  const [err, setErr]           = useState<string | null>(null)
  const abortRef                = useRef<AbortController | null>(null)

  const send = async (text: string, buildOllamaMessages: (history: ChatMsg[]) => OllamaMsg[]) => {
    const updated: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages(updated)
    setLoading(true)
    setStreaming('')
    setErr(null)
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    try {
      const full = await streamChat(
        buildOllamaMessages(updated),
        t => setStreaming(t),
        abortRef.current.signal,
      )
      setMessages(prev => [...prev, { role: 'assistant', content: stripThinking(full) }])
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setErr((e as Error).message)
    } finally {
      setStreaming(null)
      setLoading(false)
    }
  }

  return { messages, setMessages, streaming, loading, err, send }
}

// ── Shared UI bits ─────────────────────────────────────────────────────────
function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="mb-3 p-3 bg-red-50 dark:bg-red-950/20 rounded-xl border border-red-200 dark:border-red-900 text-xs text-red-600 dark:text-red-400">
      ⚠️ {msg}
    </div>
  )
}

function InfoBanner({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-900 text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
      {children}
    </div>
  )
}

// ── ChatWindow ─────────────────────────────────────────────────────────────
interface ChatWindowProps {
  messages: ChatMsg[]
  streamingText: string | null
  onSend: (text: string) => void
  loading: boolean
  placeholder?: string
  disabled?: boolean
  disabledMsg?: string
}

function ChatWindow({
  messages, streamingText, onSend, loading,
  placeholder, disabled = false, disabledMsg,
}: ChatWindowProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  const handleSubmit = () => {
    const text = input.trim()
    if (!text || loading || disabled) return
    setInput('')
    onSend(text)
  }

  return (
    <div className="flex flex-col h-[400px]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50 dark:bg-slate-950 rounded-xl border border-gray-200 dark:border-slate-800">
        {messages.length === 0 && streamingText === null && (
          <div className="flex items-center justify-center h-full text-gray-400 dark:text-slate-600 text-sm">
            {disabled && disabledMsg ? disabledMsg : 'Send a message to start…'}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words ${
              msg.role === 'user'
                ? 'bg-emerald-600 text-white rounded-br-sm'
                : 'bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 border border-gray-200 dark:border-slate-700 rounded-bl-sm'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}

        {streamingText !== null && (
          <div className="flex justify-start">
            <div className="max-w-[80%] px-4 py-2.5 rounded-2xl rounded-bl-sm text-sm leading-relaxed whitespace-pre-wrap break-words bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 border border-gray-200 dark:border-slate-700">
              {stripThinking(streamingText) || (
                <span className="text-gray-400 dark:text-slate-600 italic text-xs">Thinking…</span>
              )}
              <span className="inline-block w-1.5 h-4 bg-emerald-500 animate-pulse ml-0.5 rounded-sm align-middle" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div className="flex gap-2 mt-3">
        <input
          className="flex-1 px-4 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:opacity-50"
          placeholder={placeholder ?? 'Type a message…'}
          value={input}
          disabled={loading || disabled}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
        />
        <button
          onClick={handleSubmit}
          disabled={loading || disabled || !input.trim()}
          className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-sm font-medium transition-colors shrink-0"
        >
          {loading ? '⏳' : 'Send →'}
        </button>
      </div>
    </div>
  )
}

// ── Tab 1: LCEL Chain ──────────────────────────────────────────────────────
function LCELTab() {
  const { messages, streaming, loading, err, send } = useChat()

  return (
    <div>
      <InfoBanner>
        <strong>How it works:</strong> Each turn builds an LCEL chain:{' '}
        <code className="bg-blue-100 dark:bg-blue-900 px-1 rounded">prompt | llm | StrOutputParser()</code>.
        Full history is included in every call so the model can follow the conversation.
      </InfoBanner>
      {err && <ErrorBanner msg={err} />}
      <ChatWindow
        messages={messages}
        streamingText={streaming}
        loading={loading}
        placeholder="What is LangChain in one sentence?"
        onSend={text =>
          send(text, history => [
            { role: 'system', content: 'You are a helpful assistant. Give clear, concise answers.' },
            ...history.map(m => ({ role: m.role, content: m.content })),
          ])
        }
      />
    </div>
  )
}

// ── Tab 2: Prompt Lab ──────────────────────────────────────────────────────
const PROMPT_PRESETS = [
  { label: '🏴‍☠️ Pirate',       prompt: "You are a friendly pirate. Use 'Arrr' frequently. Short, fun answers." },
  { label: '😤 Grumpy Prof',   prompt: 'You are a grumpy professor. Use mild sarcasm. Be brief but accurate.' },
  { label: '🧘 Zen Master',    prompt: 'You are a Zen master. Answer every question with a short poem or koan.' },
  { label: '🇨🇳 Chinese Only', prompt: 'Respond ONLY in Traditional Chinese (繁體中文). Be concise and clear.' },
]

function PromptLabTab() {
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.')
  const { messages, setMessages, streaming, loading, err, send } = useChat()

  const applyPreset = (prompt: string) => {
    setSystemPrompt(prompt)
    setMessages([])
  }

  return (
    <div>
      <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-xs font-semibold text-gray-600 dark:text-slate-400 mb-1.5 block">
            System Prompt — edit to change AI personality
          </label>
          <textarea
            className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-none"
            rows={5}
            value={systemPrompt}
            onChange={e => { setSystemPrompt(e.target.value); setMessages([]) }}
          />
          <p className="text-xs text-gray-400 dark:text-slate-600 mt-1">
            Changing the prompt resets chat history.
          </p>
        </div>
        <div>
          <label className="text-xs font-semibold text-gray-600 dark:text-slate-400 mb-1.5 block">
            Quick Presets
          </label>
          <div className="grid grid-cols-2 gap-2">
            {PROMPT_PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => applyPreset(p.prompt)}
                className="px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-emerald-400 dark:hover:border-emerald-600 text-xs text-left transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {err && <ErrorBanner msg={err} />}
      <ChatWindow
        messages={messages}
        streamingText={streaming}
        loading={loading}
        placeholder="What is machine learning?"
        onSend={text =>
          send(text, history => [
            { role: 'system', content: systemPrompt },
            ...history.map(m => ({ role: m.role, content: m.content })),
          ])
        }
      />
    </div>
  )
}

// ── Tab 3: RAG Chat ────────────────────────────────────────────────────────
const SAMPLE_DOC = `LangChain Expression Language (LCEL) is a declarative way to compose chains using the pipe operator |. Every component in LangChain implements the Runnable interface with methods: invoke, stream, batch, and async variants.

A basic LCEL chain: chain = prompt | llm | output_parser. The pipe operator connects components left to right, passing output of one as input to the next. This makes chains composable, readable, and easy to inspect.

RAG (Retrieval-Augmented Generation) retrieves relevant documents from a knowledge base first, then passes them as context to an LLM to generate a grounded answer. This prevents hallucination by anchoring the model to factual content.

The retrieval step uses vector embeddings and cosine similarity to find semantically similar documents. LangChain provides many retriever types: VectorStoreRetriever for semantic search, BM25Retriever for keyword matching, and EnsembleRetriever for combining both with Reciprocal Rank Fusion.

ChromaDB is a popular open-source vector database for local development. Create it with Chroma.from_documents() and retrieve with .as_retriever().

Agents use the ReAct framework: Thought → Action → Observation loop. The agent reasons about what tool to call, calls it, observes the result, and repeats until done. LangGraph extends this with explicit state graphs for complex multi-agent workflows.`

function RAGTab() {
  const [docText, setDocText]   = useState(SAMPLE_DOC)
  const [chunks, setChunks]     = useState<string[]>([])
  const [indexed, setIndexed]   = useState(false)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [streaming, setStreaming] = useState<string | null>(null)
  const [loading, setLoading]   = useState(false)
  const [err, setErr]           = useState<string | null>(null)
  const abortRef                = useRef<AbortController | null>(null)

  const indexDocument = () => {
    setChunks(chunkText(docText))
    setIndexed(true)
    setMessages([])
    setErr(null)
  }

  const send = async (text: string) => {
    const topChunks  = retrieveTopChunks(chunks, text, 3)
    const context    = topChunks.map((c, i) => `[Chunk ${i + 1}]\n${c}`).join('\n\n')
    const sysPrompt  = `You are a helpful assistant. Answer ONLY using the context below.\nIf the answer is not in the context, say "That information is not in the document."\n\nContext:\n${context}`

    const updated: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages(updated)
    setLoading(true)
    setStreaming('')
    setErr(null)
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    try {
      const ollamaMsgs: OllamaMsg[] = [
        { role: 'system', content: sysPrompt },
        { role: 'user',   content: text },
      ]
      const full = await streamChat(ollamaMsgs, t => setStreaming(t), abortRef.current.signal)
      setMessages(prev => [...prev, { role: 'assistant', content: stripThinking(full) }])
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setErr((e as Error).message)
    } finally {
      setStreaming(null)
      setLoading(false)
    }
  }

  const chunkCount = chunkText(docText).length

  return (
    <div>
      <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-xs font-semibold text-gray-600 dark:text-slate-400 mb-1.5 block">
            Document — paste any text, then index it
          </label>
          <textarea
            className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-gray-900 dark:text-white text-xs font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-none leading-relaxed"
            rows={7}
            value={docText}
            onChange={e => { setDocText(e.target.value); setIndexed(false); setMessages([]) }}
          />
        </div>
        <div className="flex flex-col gap-3 justify-between">
          <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-900 text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
            <strong>How it works:</strong> Text splits into ~500-char chunks.
            For each query the top 3 chunks are retrieved by keyword overlap and injected
            into the system prompt. Pure JS — no embeddings or vector DB!
          </div>
          <div>
            <button
              onClick={indexDocument}
              className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors"
            >
              ⚡ Index Document ({chunkCount} chunk{chunkCount !== 1 ? 's' : ''})
            </button>
            {indexed && (
              <p className="text-xs text-emerald-600 dark:text-emerald-400 text-center mt-1.5">
                ✓ {chunks.length} chunks indexed — ready to chat
              </p>
            )}
          </div>
        </div>
      </div>
      {err && <ErrorBanner msg={err} />}
      <ChatWindow
        messages={messages}
        streamingText={streaming}
        loading={loading}
        disabled={!indexed}
        disabledMsg="⚡ Click Index Document first"
        placeholder="What is LCEL? How does RAG work?"
        onSend={send}
      />
    </div>
  )
}

// ── Tab 4: Memory Chat ─────────────────────────────────────────────────────
function MemoryTab() {
  const { messages, setMessages, streaming, loading, err, send } = useChat()

  const tokenEst = Math.round(
    messages.reduce((s, m) => s + m.content.length, 0) / 4,
  )

  return (
    <div>
      <div className="mb-4 flex items-start gap-3">
        <InfoBanner>
          <strong>How it works:</strong> The <em>full</em> chat history is passed with every
          Ollama request as a message array. The model "remembers" by seeing the whole thread —
          identical to LangChain's{' '}
          <code className="bg-blue-100 dark:bg-blue-900 px-1 rounded">RunnableWithMessageHistory</code>.
        </InfoBanner>
        <div className="shrink-0 flex flex-col items-end gap-1 pt-1">
          <button
            onClick={() => setMessages([])}
            disabled={messages.length === 0}
            className="px-3 py-2 rounded-lg border border-gray-200 dark:border-slate-700 text-xs text-gray-500 hover:text-red-500 hover:border-red-300 disabled:opacity-30 transition-colors whitespace-nowrap"
          >
            🗑️ Clear
          </button>
          {messages.length > 0 && (
            <span className="text-xs text-gray-400 dark:text-slate-600">
              {messages.length} msgs · ~{tokenEst} tokens
            </span>
          )}
        </div>
      </div>
      {err && <ErrorBanner msg={err} />}
      <ChatWindow
        messages={messages}
        streamingText={streaming}
        loading={loading}
        placeholder="Hi! My name is Alice. Remember that."
        onSend={text =>
          send(text, history => [
            {
              role: 'system',
              content: 'You are a helpful, friendly assistant. You remember everything from our conversation.',
            },
            ...history.map(m => ({ role: m.role, content: m.content })),
          ])
        }
      />
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
const TABS = [
  { icon: '🔗', en: 'LCEL Chain',  zh: 'LCEL 链' },
  { icon: '🎭', en: 'Prompt Lab',  zh: 'Prompt 实验室' },
  { icon: '📚', en: 'RAG Chat',    zh: 'RAG 聊天' },
  { icon: '💬', en: 'Memory Chat', zh: '记忆聊天' },
]

export function LiveDemoPage() {
  const { t } = useLang()
  const [tab, setTab]           = useState(0)
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null)

  useEffect(() => {
    const ctrl = new AbortController()
    fetch(`${OLLAMA_BASE}/api/tags`, { signal: ctrl.signal })
      .then(r => setOllamaOk(r.ok))
      .catch(() => setOllamaOk(false))
    return () => ctrl.abort()
  }, [])

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-3xl">🤖</span>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t('Live LLM Demo', '实时 LLM 演示')}
          </h1>
        </div>
        <p className="text-sm text-gray-500 dark:text-slate-400">
          {t(
            '4 interactive chatbots running 100% locally via Ollama — no cloud API, no subscription.',
            '4 个通过 Ollama 100% 本地运行的交互式聊天机器人——无需云端 API，无需订阅。',
          )}
        </p>
      </div>

      {/* Ollama status */}
      {ollamaOk === false && (
        <div className="mb-6 p-4 bg-amber-50 dark:bg-amber-950/20 rounded-xl border border-amber-200 dark:border-amber-800">
          <p className="text-sm font-semibold text-amber-700 dark:text-amber-400 mb-2">
            ⚠️ {t('Ollama not detected at localhost:11434', '未检测到 Ollama（localhost:11434）')}
          </p>
          <pre className="text-xs text-amber-800 dark:text-amber-300 font-mono leading-relaxed whitespace-pre-wrap">{`# 1. Install Ollama:  https://ollama.com
# 2. Pull the model:
ollama pull qwen3:8b
# 3. Start server (usually auto-starts):
ollama serve
# 4. If serving from a remote host, allow browser origin:
OLLAMA_ORIGINS=* ollama serve`}</pre>
        </div>
      )}

      {ollamaOk === true && (
        <div className="mb-5 inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          {t('Ollama connected', 'Ollama 已连接')} · {DEFAULT_MODEL}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 p-1 bg-gray-100 dark:bg-slate-800 rounded-xl">
        {TABS.map((tabDef, i) => (
          <button
            key={i}
            onClick={() => setTab(i)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-sm font-medium transition-all ${
              tab === i
                ? 'bg-white dark:bg-slate-900 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300'
            }`}
          >
            <span>{tabDef.icon}</span>
            <span className="hidden sm:block">{t(tabDef.en, tabDef.zh)}</span>
          </button>
        ))}
      </div>

      {/* Tab panels — all mounted, hidden via CSS so state is preserved on tab switch */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 p-6">
        <div className={tab !== 0 ? 'hidden' : ''}><LCELTab /></div>
        <div className={tab !== 1 ? 'hidden' : ''}><PromptLabTab /></div>
        <div className={tab !== 2 ? 'hidden' : ''}><RAGTab /></div>
        <div className={tab !== 3 ? 'hidden' : ''}><MemoryTab /></div>
      </div>

      {/* Footer */}
      <p className="mt-4 text-xs text-gray-400 dark:text-slate-600 text-center">
        {t(
          `Runs 100% locally via Ollama REST API · No data sent to cloud · Model: ${DEFAULT_MODEL}`,
          `通过 Ollama REST API 100% 本地运行 · 数据不发送至云端 · 模型：${DEFAULT_MODEL}`,
        )}
      </p>
    </div>
  )
}
