import type { TopicContent } from '../types'

export const ragRetriever: TopicContent = {
  id: 'rag-retriever',
  emoji: '🔎',
  title: { en: 'RAG Retriever & Document Formatting', zh: 'RAG 检索器与文档格式化' },
  contentType: 'code',
  content: {
    en: `## RAG Retriever & Document Formatting

The retriever is the bridge between a user query and the ChromaDB vector store. This module wraps it with clean helpers for formatting retrieved documents into prompt-ready context strings.

### The Retriever Wrapper

\`\`\`python
# phase1_foundation/03_simple_rag/retriever.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from shared.config import settings
from shared.llm_factory import get_embeddings


def get_retriever(k: int = 4, collection_name: str = "learn_llm"):
    """Return a ChromaDB retriever for the persisted index.

    Args:
        k:               Number of documents to retrieve per query.
        collection_name: ChromaDB collection to query.

    Returns:
        A LangChain VectorStoreRetriever ready for LCEL composition.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
\`\`\`

### Document Formatter

\`\`\`python
def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string.

    Each document is annotated with source and page metadata.

    Example output:
        [Document 1 | data/langchain_intro.pdf, page 3]
        LangChain is a framework for building LLM applications...

        [Document 2 | data/rag_paper.pdf, page 7]
        Retrieval-Augmented Generation combines...
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = f"[Document {i} | {source}" + (f", page {page}]" if page else "]")
        parts.append(f"{header}\\n{doc.page_content}")
    return "\\n\\n".join(parts)


def similarity_search(
    query: str,
    k: int = 4,
    collection_name: str = "learn_llm",
) -> list[Document]:
    """Direct similarity search — useful for debugging retrieval quality."""
    retriever = get_retriever(k=k, collection_name=collection_name)
    return retriever.invoke(query)
\`\`\`

### Usage in a RAG Chain

\`\`\`python
from phase1_foundation._03_simple_rag_compat import get_retriever, format_docs
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = get_retriever(k=4)

prompt = PromptTemplate(
    template=(
        "Answer based ONLY on the context below.\\n\\n"
        "Context:\\n{context}\\n\\n"
        "Question: {question}\\n\\nAnswer:"
    ),
    input_variables=["context", "question"],
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | get_llm(temperature=0.0)
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LCEL?")
print(answer)
\`\`\`

### Debugging Retrieval Quality

\`\`\`python
# Check what documents are being retrieved before chaining
docs = similarity_search("What is LCEL?", k=4)

for doc in docs:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Score proxy: {doc.metadata.get('distance', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}\\n")

# If wrong docs are retrieved, try:
# - Increase k to retrieve more candidates
# - Re-ingest with smaller chunk_size for finer granularity
# - Switch to hybrid retrieval (Phase 3)
\`\`\`

### Key Concepts

| Concept | Description |
|---|---|
| \`search_type="similarity"\` | Cosine similarity in embedding space |
| \`k\` | Number of chunks returned (start with 3–5) |
| \`format_docs\` | Annotates each chunk with source + page |
| \`similarity_search\` | Direct call for offline debugging |`,

    zh: `## RAG 检索器与文档格式化

检索器是用户查询和 ChromaDB 向量存储之间的桥梁。该模块用干净的辅助函数封装它，将检索到的文档格式化为可注入提示的上下文字符串。

### 检索器包装器

\`\`\`python
def get_retriever(k: int = 4, collection_name: str = "learn_llm"):
    """返回持久化索引的 ChromaDB 检索器。"""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
\`\`\`

### 文档格式化器

\`\`\`python
def format_docs(docs: list[Document]) -> str:
    """将检索到的文档格式化为单个上下文字符串。"""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = f"[文档 {i} | {source}" + (f", 第 {page} 页]" if page else "]")
        parts.append(f"{header}\\n{doc.page_content}")
    return "\\n\\n".join(parts)
\`\`\`

### 在 RAG 链中使用

\`\`\`python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | get_llm(temperature=0.0)
    | StrOutputParser()
)

answer = rag_chain.invoke("什么是 LCEL？")
\`\`\`

### 调试检索质量

\`\`\`python
# 在链接之前检查正在检索哪些文档
docs = similarity_search("什么是 LCEL？", k=4)
for doc in docs:
    print(f"来源：{doc.metadata.get('source')}")
    print(f"内容：{doc.page_content[:200]}\\n")
\`\`\`

### 关键概念

| 概念 | 描述 |
|---|---|
| \`search_type="similarity"\` | 嵌入空间中的余弦相似度 |
| \`k\` | 返回的块数（从 3–5 开始） |
| \`format_docs\` | 用来源 + 页码注释每个块 |
| \`similarity_search\` | 用于离线调试的直接调用 |`,
  },
}
