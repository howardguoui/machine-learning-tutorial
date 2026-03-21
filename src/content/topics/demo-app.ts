import type { TopicContent } from '../types'

export const gradioDemo: TopicContent = {
  id: 'gradio-demo',
  title: { en: 'Interactive Demo App (Gradio)', zh: '交互式演示应用（Gradio）' },
  contentType: 'code',
  content: {
    en: `## Interactive Demo App (Gradio)

\`demo/app.py\` is a **Gradio web application** with 4 interactive tabs, each teaching one core LangChain concept with a live chat demo running 100% locally via Ollama.

### Running the Demo

\`\`\`bash
# Install dependencies
uv pip install -e ".[phase1]"
ollama pull qwen3:8b
ollama pull nomic-embed-text

# Start the app
python demo/app.py

# Open in browser: http://localhost:7860
\`\`\`

### App Structure

\`\`\`
Tab 1: 🔗 LCEL Chain      — basic prompt | llm | parser chain
Tab 2: 🎭 Prompt Lab      — edit system prompt, see AI personality change
Tab 3: 📚 RAG Chat        — chat with a document (ChromaDB, local embeddings)
Tab 4: 💬 Memory Chat     — persistent conversation memory across turns
\`\`\`

### Chain Builders

\`\`\`python
# demo/app.py

def _make_basic_chain(system_prompt: str):
    """Build a simple LCEL chain with the given system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    return prompt | get_llm(temperature=0.7) | StrOutputParser()


def _make_memory_chain():
    """Build a chain that accepts a list of LangChain messages as history."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, friendly assistant. "
                   "You remember everything from our conversation."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    return prompt | get_llm(temperature=0.7) | StrOutputParser()


def _build_rag_chain(text: str):
    """Build an in-memory RAG chain from raw text (no file upload needed)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    # In-memory Chroma (no persist_directory) — isolated per session
    vectorstore = Chroma.from_documents(docs, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer ONLY using the context below.\\n"
            "If the answer is not in the context, say 'That information is not in the document.'\\n\\n"
            "Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
        ),
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\\n\\n".join(f"[Chunk {i+1}] {d.page_content}" for i, d in enumerate(docs))

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
\`\`\`

### Streaming Respond Functions

\`\`\`python
def respond_lcel(message: str, history: list[dict]):
    """Stream a response using the basic LCEL chain."""
    chain = _make_basic_chain(
        "You are a helpful assistant. Give clear, concise answers."
    )
    history = history + [{"role": "user", "content": message}]
    yield history   # show user message immediately

    partial = ""
    for chunk in chain.stream({"question": message}):
        partial += chunk
        yield history + [{"role": "assistant", "content": strip_thinking(partial)}]


def respond_memory(message: str, history: list[dict]):
    """Stream a response using the memory-aware chain."""
    chain = _make_memory_chain()
    # Convert Gradio history format → LangChain message objects
    lc_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]

    history = history + [{"role": "user", "content": message}]
    yield history

    partial = ""
    for chunk in chain.stream({"chat_history": lc_history, "question": message}):
        partial += chunk
        yield history + [{"role": "assistant", "content": strip_thinking(partial)}]
\`\`\`

### Gradio UI Layout

\`\`\`python
with gr.Blocks(title="learn-llm Demo", theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.Markdown("# 🤖 learn-llm — Interactive Chatbot Demo")

    with gr.Tabs():
        with gr.TabItem("🔗 LCEL Chain"):
            with gr.Row():
                with gr.Column(scale=3):
                    lcel_bot = gr.Chatbot(height=480)
                    lcel_input = gr.Textbox(placeholder="Type your message…")
                    gr.Examples(examples=[
                        "What is LangChain in one sentence?",
                        "Explain the pipe | operator in Python",
                    ], inputs=lcel_input)
                with gr.Column(scale=2):
                    gr.Markdown(LCEL_EXPLAIN)   # explanation panel
                    with gr.Accordion("📋 View the code"):
                        gr.Markdown(LCEL_CODE)

        with gr.TabItem("🎭 Prompt Lab"):
            system_prompt_box = gr.Textbox(value="You are a helpful assistant.")
            # ... similar layout with editable system prompt

        with gr.TabItem("📚 RAG Chat"):
            rag_doc_box = gr.Textbox(value=SAMPLE_DOC, lines=8)
            rag_index_btn = gr.Button("⚡ Index Document")
            # ... chat interface

        with gr.TabItem("💬 Memory Chat"):
            # ... memory-aware chat

    app.launch(server_port=7860, inbrowser=True)
\`\`\`

### Prompt Lab Presets

\`\`\`python
gr.Examples(
    examples=[
        ["You are a friendly pirate. Use 'Arrr' a lot.", "What is machine learning?"],
        ["You are a grumpy professor. Use mild sarcasm.", "What is an API?"],
        ["You are a Zen master. Answer with a short poem.", "What is Python?"],
        ["Respond only in Traditional Chinese. Be concise.", "What is RAG?"],
    ],
    inputs=[system_prompt_box, prompt_input],
)
\`\`\`

### Key Implementation Details

| Detail | Implementation |
|---|---|
| Streaming | \`chain.stream()\` + Gradio generator functions |
| Thinking removal | \`strip_thinking()\` removes \`<think>...</think>\` blocks from Qwen3 |
| RAG caching | \`_rag_chain_cache\` dict — builds chain once per document |
| Pre-indexing | Sample doc indexed at startup so RAG is ready immediately |
| History format | Converts Gradio \`list[dict]\` ↔ LangChain \`BaseMessage\` objects |`,

    zh: `## 交互式演示应用（Gradio）

\`demo/app.py\` 是一个 **Gradio Web 应用程序**，有 4 个交互式标签页，每个标签页通过 Ollama 在本地 100% 运行的实时聊天演示教授一个核心 LangChain 概念。

### 运行演示

\`\`\`bash
uv pip install -e ".[phase1]"
ollama pull qwen3:8b
ollama pull nomic-embed-text

python demo/app.py
# 在浏览器中打开：http://localhost:7860
\`\`\`

### 应用结构

\`\`\`
标签1：🔗 LCEL 链      — 基本 prompt | llm | parser 链
标签2：🎭 Prompt 实验室 — 编辑系统提示，看 AI 个性变化
标签3：📚 RAG 聊天      — 与文档聊天（ChromaDB，本地嵌入）
标签4：💬 记忆聊天      — 跨轮次的持久对话记忆
\`\`\`

### 流式响应函数

\`\`\`python
def respond_lcel(message: str, history: list[dict]):
    chain = _make_basic_chain("你是一个有帮助的助手。")
    history = history + [{"role": "user", "content": message}]
    yield history  # 立即显示用户消息

    partial = ""
    for chunk in chain.stream({"question": message}):
        partial += chunk
        yield history + [{"role": "assistant", "content": strip_thinking(partial)}]
\`\`\`

### 关键实现细节

| 细节 | 实现 |
|---|---|
| 流式传输 | \`chain.stream()\` + Gradio 生成器函数 |
| 思考移除 | \`strip_thinking()\` 从 Qwen3 移除 \`<think>...\` 块 |
| RAG 缓存 | \`_rag_chain_cache\` 字典 — 每个文档构建一次链 |
| 历史格式 | 转换 Gradio \`list[dict]\` ↔ LangChain \`BaseMessage\` 对象 |`,
  },
}
