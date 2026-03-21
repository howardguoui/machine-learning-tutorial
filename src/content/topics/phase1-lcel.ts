import type { TopicContent } from '../types'

export const lcelBasics: TopicContent = {
  id: 'lcel-basics',
  title: { en: 'LCEL — LangChain Expression Language', zh: 'LCEL — LangChain表达式语言' },
  contentType: 'code',
  content: {
    en: `LCEL (LangChain Expression Language) is the **pipe-based composition system** at the heart of modern LangChain. Every component is a \`Runnable\` — they can be chained with \`|\`, streamed, parallelized, and batched uniformly.

## The Core Idea: Everything is a Runnable

\`\`\`python
# phase1_foundation/01_lcel_basics.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from shared.llm_factory import get_llm

# 1. Basic chain: prompt | llm | parser
def build_basic_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant. Answer in 1-2 sentences."),
        ("human", "{question}"),
    ])
    llm = get_llm(temperature=0.0)
    parser = StrOutputParser()

    # LCEL: each | connects Runnables — output of left becomes input of right
    chain = prompt | llm | parser
    return chain

chain = build_basic_chain()
answer = chain.invoke({"question": "What is LangChain?"})
print(f"Answer: {answer}")
\`\`\`

## Streaming — Tokens Arrive One by One

\`\`\`python
# 2. Streaming chain — watch tokens arrive in real time
def build_streaming_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    llm = get_llm(temperature=0.7, streaming=True)
    return prompt | llm | StrOutputParser()

stream_chain = build_streaming_chain()

# .stream() yields text chunks as they arrive from the LLM
print("Streaming: ", end="", flush=True)
for chunk in stream_chain.stream({"question": "Explain LCEL in one sentence."}):
    print(chunk, end="", flush=True)
print()
\`\`\`

## RunnableParallel — Run Two Chains Simultaneously

\`\`\`python
# 3. Parallel execution — both branches run concurrently
def build_parallel_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert teacher."),
        ("human", "{question}"),
    ])
    llm = get_llm(temperature=0.0)
    answer_chain = prompt | llm | StrOutputParser()

    followup_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert teacher."),
        ("human", "Give one follow-up question about: {question}"),
    ])
    followup_chain = followup_prompt | llm | StrOutputParser()

    # RunnableParallel runs both branches simultaneously and merges results
    return RunnableParallel(
        answer=answer_chain,
        followup=followup_chain,
    )

parallel_chain = build_parallel_chain()
result = parallel_chain.invoke({"question": "What are vector embeddings?"})
print(f"Answer:   {result['answer']}")
print(f"Follow-up: {result['followup']}")
# Output: {"answer": "...", "followup": "..."}
\`\`\`

## Multi-Step Chain — Compose Two LLM Calls

\`\`\`python
# 4. Two-step chain: answer in English → translate to Chinese
def build_translate_chain():
    llm = get_llm(temperature=0.0)

    # Step 1: answer the question
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer briefly."),
        ("human", "{question}"),
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()

    # Step 2: translate the answer
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following text to Chinese (Traditional)."),
        ("human", "{text}"),
    ])
    translate_chain = translate_prompt | llm | StrOutputParser()

    # Connect: step 1's output feeds into step 2's {text} variable
    full_chain = (
        {"text": answer_chain, "question": RunnablePassthrough()}
        | translate_chain
    )
    return full_chain

translate_chain = build_translate_chain()
translated = translate_chain.invoke({"question": "What is RAG?"})
print(f"Chinese translation: {translated}")
\`\`\`

## Setup

\`\`\`bash
# Install dependencies
pip install langchain langchain-core langchain-openai python-dotenv

# .env file
OPENAI_API_KEY=sk-...

# Run
python phase1_foundation/01_lcel_basics.py
\`\`\`

## LCEL Runnable Interface Summary

| Method | Description |
|---|---|
| \`.invoke(input)\` | Single synchronous call |
| \`.stream(input)\` | Generator yielding chunks |
| \`.batch([inputs])\` | Multiple inputs concurrently |
| \`.ainvoke(input)\` | Async single call |
| \`.astream(input)\` | Async generator |
| \`\|\` operator | Compose runnables into a chain |

> **Key insight**: Because every component (prompts, LLMs, parsers, retrievers) implements the same \`Runnable\` interface, they can be composed with \`|\` uniformly. This makes LCEL chains easy to test, swap, and extend without rewriting orchestration code.`,

    zh: `LCEL（LangChain表达式语言）是现代LangChain核心的**基于管道的组合系统**。每个组件都是一个\`Runnable\`——它们可以用\`|\`链接、流式传输、并行化和批量处理。

## 核心思想：一切皆为Runnable

\`\`\`python
# 1. 基础链：prompt | llm | parser
def build_basic_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的助手。用1-2句话回答。"),
        ("human", "{question}"),
    ])
    llm = get_llm(temperature=0.0)
    parser = StrOutputParser()

    # LCEL：每个 | 连接Runnables——左边的输出成为右边的输入
    chain = prompt | llm | parser
    return chain
\`\`\`

## 流式输出 — 逐个接收Token

\`\`\`python
# .stream() 在LLM生成时逐块产生文本
for chunk in stream_chain.stream({"question": "用一句话解释LCEL。"}):
    print(chunk, end="", flush=True)
\`\`\`

## RunnableParallel — 同时运行两个链

\`\`\`python
# RunnableParallel同时运行两个分支并合并结果
return RunnableParallel(
    answer=answer_chain,
    followup=followup_chain,
)
\`\`\`

## LCEL Runnable接口汇总

| 方法 | 描述 |
|---|---|
| \`.invoke(input)\` | 单个同步调用 |
| \`.stream(input)\` | 逐块产生的生成器 |
| \`.batch([inputs])\` | 多个输入并发 |
| \`.ainvoke(input)\` | 异步单次调用 |
| \`\|\` 运算符 | 将runnables组合成链 |

> **关键洞察**：因为每个组件（提示、LLM、解析器、检索器）都实现相同的\`Runnable\`接口，它们可以用\`|\`统一组合。这使LCEL链易于测试、替换和扩展，无需重写编排代码。`,
  },
}

export const promptTemplates: TopicContent = {
  id: 'prompt-templates',
  title: { en: 'Prompt Templates & Few-Shot', zh: '提示模板与少样本提示' },
  contentType: 'code',
  content: {
    en: `Prompt templates are **parameterized message factories** — they separate the structure of a prompt from its runtime values, enabling reuse, versioning, and YAML-based storage.

## PromptTemplate vs ChatPromptTemplate

\`\`\`python
# phase1_foundation/02_prompt_templates.py
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from shared.llm_factory import get_llm

# --- PromptTemplate: legacy string-based ---
template = PromptTemplate(
    template="Summarise the following in {style} style:\\n\\n{text}",
    input_variables=["style", "text"],
)
# Useful for simple string inputs; still composes with | operator
chain = template | get_llm(temperature=0.0) | StrOutputParser()
result = chain.invoke({"style": "bullet points", "text": "Python is versatile."})

# --- ChatPromptTemplate: preferred for chat models ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in {language}."),
    ("human", "{question}"),
])
# Dynamic system and human messages — all slots filled at invoke time
chain = chat_prompt | get_llm(temperature=0.0) | StrOutputParser()
result = chain.invoke({
    "role": "Python expert",
    "language": "English",
    "question": "What is a list comprehension?",
})
\`\`\`

## Few-Shot Prompting — Teach by Example

\`\`\`python
# Few-shot: give the model examples to learn from
examples = [
    {"input": "happy",   "output": "sad"},
    {"input": "fast",    "output": "slow"},
    {"input": "hot",     "output": "cold"},
]

# Template for each example
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])

# Compose examples into a reusable few-shot block
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Full prompt: system instruction + examples + new input
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You complete analogies. Answer with ONLY the antonym word."),
    few_shot_prompt,   # inject all examples here
    ("human", "{word}"),
])

chain = final_prompt | get_llm(temperature=0.0) | StrOutputParser()
for word in ["big", "beautiful", "easy"]:
    result = chain.invoke({"word": word})
    print(f"  {word} → {result.strip()}")
# big → small, beautiful → ugly, easy → hard
\`\`\`

## Loading Prompts from YAML

\`\`\`python
# shared/prompts/rag_qa.yaml
# input_variables: [context, question]
# template: |
#   You are a helpful assistant. Answer ONLY using the context below.
#   If the context lacks the answer, say: 'I don't have enough context.'
#
#   Context:
#   {context}
#
#   Question: {question}
#
#   Answer:

import yaml
from pathlib import Path

yaml_path = Path("shared/prompts/rag_qa.yaml")
with open(yaml_path) as f:
    config = yaml.safe_load(f)

prompt = PromptTemplate(
    template=config["template"],
    input_variables=config["input_variables"],
)
# Prompts stored in YAML → versionable, reviewable by non-engineers
\`\`\`

## Partial Binding — Pre-fill Variables

\`\`\`python
# Partial binding: fix some variables, leave others dynamic
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {language} expert."),
    ("human", "{question}"),
])

# Pre-bind language = "Python" — remaining variable is just {question}
python_expert_prompt = base_prompt.partial(language="Python")

chain = python_expert_prompt | get_llm(temperature=0.0) | StrOutputParser()
result = chain.invoke({"question": "What are decorators?"})
# No need to pass "language" — it's already bound

# Use case: create specialized chains from a single base template
rust_expert_chain = base_prompt.partial(language="Rust") | get_llm() | StrOutputParser()
go_expert_chain   = base_prompt.partial(language="Go")   | get_llm() | StrOutputParser()
\`\`\`

## Prompt Engineering Best Practices

| Pattern | When to Use |
|---|---|
| **System + Human** | Always — sets model persona and task clearly |
| **Few-shot examples** | When the output format needs to be precise |
| **YAML/file-based** | Production — enables versioning and review |
| **Partial binding** | When building domain-specific chain variants |
| **MessagesPlaceholder** | Conversations — inject dynamic history into prompt |`,

    zh: `提示模板是**参数化的消息工厂**——它们将提示的结构与运行时值分离，实现复用、版本控制和基于YAML的存储。

## PromptTemplate vs ChatPromptTemplate

\`\`\`python
# ChatPromptTemplate：聊天模型的首选
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。用{language}回答。"),
    ("human", "{question}"),
])
\`\`\`

## 少样本提示 — 通过示例教学

\`\`\`python
# 给模型提供示例以学习
examples = [
    {"input": "开心", "output": "悲伤"},
    {"input": "快",   "output": "慢"},
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
\`\`\`

## 提示工程最佳实践

| 模式 | 使用场景 |
|---|---|
| **系统+人类消息** | 始终使用——清晰设置模型角色和任务 |
| **少样本示例** | 需要精确输出格式时 |
| **基于YAML文件** | 生产环境——支持版本控制和审查 |
| **部分绑定** | 构建领域特定链变体时 |
| **MessagesPlaceholder** | 对话——将动态历史注入提示 |`,
  },
}

export const simpleRAG: TopicContent = {
  id: 'simple-rag',
  title: { en: 'Simple RAG — Retrieval-Augmented Generation', zh: '简单RAG — 检索增强生成' },
  contentType: 'code',
  content: {
    en: `RAG (Retrieval-Augmented Generation) solves the LLM knowledge gap by **retrieving relevant documents at query time** and injecting them into the prompt context.

## The RAG Pipeline

\`\`\`
Documents → [Load] → [Split] → [Embed] → [VectorDB]
Query     →                              [VectorDB] → [Top-k docs] → [Prompt] → [LLM] → Answer
\`\`\`

## Step 1: Ingestion — Load, Split, Embed, Persist

\`\`\`python
# phase1_foundation/03_simple_rag/ingest.py
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from shared.llm_factory import get_embeddings
from shared.config import settings


def load_documents(source: Path) -> list:
    """Load PDF or TXT files from a path or directory."""
    docs = []
    files = ([source] if source.is_file()
             else list(source.glob("*.pdf")) + list(source.glob("*.txt")))

    for file in files:
        loader = PyPDFLoader(str(file)) if file.suffix == ".pdf" else TextLoader(str(file))
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} pages from {len(files)} file(s)")
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Tries these separators in order, falls back to character splits
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def ingest(source, collection_name="learn_llm", chunk_size=1000, chunk_overlap=200):
    """Full pipeline: load → split → embed → persist to ChromaDB."""
    source = Path(source)
    docs = load_documents(source)
    chunks = split_documents(docs, chunk_size, chunk_overlap)

    # Embed all chunks and persist to local ChromaDB
    embeddings = get_embeddings()  # text-embedding-3-small by default
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,  # "indexes/chroma"
    )
    print(f"✅ Persisted {len(chunks)} chunks to '{collection_name}'")
    return vectorstore

# Usage:
# ingest("data/document.pdf")
# ingest("data/")   # all PDFs in directory
\`\`\`

## Step 2: Retrieval — Find Relevant Chunks

\`\`\`python
# phase1_foundation/03_simple_rag/retriever.py
from langchain_chroma import Chroma
from shared.llm_factory import get_embeddings
from shared.config import settings


def get_retriever(k: int = 4, collection_name: str = "learn_llm"):
    """Create a similarity-search retriever from ChromaDB."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    # search_type: "similarity" (default), "mmr" (diverse), "similarity_score_threshold"
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def format_docs(docs) -> str:
    """Format retrieved docs into a single context string."""
    return "\\n\\n---\\n\\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\\n{doc.page_content}"
        for doc in docs
    )
\`\`\`

## Step 3: QA Chain — Retrieve + Generate

\`\`\`python
# phase1_foundation/03_simple_rag/qa_chain.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from .retriever import format_docs, get_retriever
from shared.llm_factory import get_llm

RAG_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant. Answer ONLY using the context below.\\n"
        "If the context lacks the answer, say: 'I don't have enough context.'\\n\\n"
        "Context:\\n{context}\\n\\n"
        "Question: {question}\\n\\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


def build_rag_chain(k: int = 4, collection_name: str = "learn_llm"):
    """Build the complete RAG QA chain using LCEL.

    Pipeline:
        {"context": retriever | format_docs, "question": passthrough}
        → RAG prompt → LLM → parser
    """
    retriever = get_retriever(k=k, collection_name=collection_name)
    llm = get_llm(temperature=0.0)

    rag_chain = (
        {
            "context": retriever | format_docs,   # retrieve & format docs
            "question": RunnablePassthrough(),     # pass question through unchanged
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ask(question: str, k: int = 4) -> str:
    """Convenience: ask one question, get an answer."""
    chain = build_rag_chain(k=k)
    return chain.invoke({"question": question})


# Interactive streaming chatbot
def interactive_chat():
    print("🤖 RAG Chatbot — type 'exit' to quit")
    chain = build_rag_chain()

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print("Bot: ", end="", flush=True)
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
        print()
\`\`\`

## Setup & Running

\`\`\`bash
pip install langchain langchain-chroma langchain-community pypdf

# Step 1: Ingest your documents
python phase1_foundation/03_simple_rag/ingest.py --file data/my_doc.pdf

# Step 2: Ask questions
python phase1_foundation/03_simple_rag/qa_chain.py
# or single query:
python phase1_foundation/03_simple_rag/qa_chain.py --query "What is the main topic?"
\`\`\`

## Chunking Strategy Guide

| Chunk Size | Overlap | Best For |
|---|---|---|
| 500–800 chars | 100 | Q&A, fact retrieval |
| 1000–1500 chars | 200 | Summaries, longer context |
| 2000+ chars | 400 | Legal docs, dense technical text |

> **Rule of thumb**: Overlap should be ~20% of chunk size to preserve sentence context across chunk boundaries.`,

    zh: `RAG（检索增强生成）通过**在查询时检索相关文档**并将其注入提示上下文来解决LLM知识盲区问题。

## RAG流水线

\`\`\`
文档 → [加载] → [分割] → [嵌入] → [向量数据库]
查询 →                              [向量数据库] → [Top-k文档] → [提示] → [LLM] → 答案
\`\`\`

## 第1步：摄入 — 加载、分割、嵌入、持久化

\`\`\`python
def ingest(source, collection_name="learn_llm"):
    """完整流水线：加载 → 分割 → 嵌入 → 持久化到ChromaDB"""
    docs = load_documents(Path(source))
    chunks = split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    return vectorstore
\`\`\`

## 第3步：QA链 — 检索+生成

\`\`\`python
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索并格式化文档
        "question": RunnablePassthrough(),     # 原样传递问题
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
\`\`\`

| 块大小 | 重叠 | 最适合 |
|---|---|---|
| 500–800字符 | 100 | 问答、事实检索 |
| 1000–1500字符 | 200 | 摘要、较长上下文 |
| 2000+字符 | 400 | 法律文档、密集技术文本 |`,
  },
}
