import type { TopicContent } from '../types'

export const bm25Retriever: TopicContent = {
  id: 'bm25-retriever',
  title: { en: 'BM25 Keyword Retriever', zh: 'BM25 关键词检索器' },
  contentType: 'code',
  content: {
    en: `## BM25 Keyword Retriever

**BM25** (Best Match 25) is a classical information retrieval algorithm that scores documents based on **term frequency** and **inverse document frequency**. Unlike vector search, it's exact — it finds documents containing the actual query words.

### When BM25 Beats Vector Search

| Query type | BM25 | Vector |
|---|---|---|
| "LangGraph \`add_conditional_edges\`" | ✅ Exact match | ❌ May miss exact name |
| "product SKU ABC-123" | ✅ Exact keyword | ❌ Poor at IDs/codes |
| "What is graph-based orchestration?" | ❌ Misses paraphrases | ✅ Semantic match |

### Building the BM25 Retriever

\`\`\`python
# phase3_enterprise/hybrid_rag/bm25_retriever.py
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def build_bm25_retriever(
    documents: list[Document],
    k: int = 5,
) -> BM25Retriever:
    """Build a BM25 retriever from a list of documents.

    BM25 needs the full corpus in memory (no external DB).
    Uses token overlap for scoring — no API calls needed.

    Args:
        documents: Pre-chunked Document objects.
        k:         Number of documents to return per query.
    """
    return BM25Retriever.from_documents(documents, k=k)
\`\`\`

### Loading Documents from ChromaDB for BM25

\`\`\`python
def load_docs_from_chroma(
    collection_name: str = "learn_llm",
    persist_directory: str = "./indexes/chroma",
) -> list[Document]:
    """Load all documents from ChromaDB for BM25 indexing.

    BM25 needs the entire corpus — this extracts it from ChromaDB.
    """
    from langchain_chroma import Chroma
    from shared.llm_factory import get_embeddings

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
    raw = vs.get()   # returns {"documents": [...], "metadatas": [...]}

    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
        if content   # skip empty docs
    ]
\`\`\`

### Usage

\`\`\`python
# Load corpus from ChromaDB
docs = load_docs_from_chroma()
print(f"Loaded {len(docs)} documents for BM25 indexing")

# Build BM25 retriever
bm25 = build_bm25_retriever(docs, k=5)

# Query
results = bm25.invoke("LangChain LCEL pipe syntax")
for doc in results:
    print(doc.page_content[:200])
\`\`\`

### BM25 Scoring Formula

\`\`\`
score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

where:
  qi    = query term i
  f(qi,D) = frequency of qi in document D
  |D|   = document length
  avgdl = average document length in corpus
  k1, b = tuning parameters (BM25: k1=1.5, b=0.75)
\`\`\``,

    zh: `## BM25 关键词检索器

**BM25**（最佳匹配 25）是一种经典信息检索算法，基于**词频**和**逆文档频率**对文档评分。与向量搜索不同，它是精确的 — 它找到包含实际查询词的文档。

### 构建 BM25 检索器

\`\`\`python
def build_bm25_retriever(
    documents: list[Document],
    k: int = 5,
) -> BM25Retriever:
    """从文档列表构建 BM25 检索器。
    BM25 需要内存中的完整语料库（无外部数据库）。
    """
    return BM25Retriever.from_documents(documents, k=k)
\`\`\`

### 从 ChromaDB 加载文档

\`\`\`python
def load_docs_from_chroma(collection_name: str = "learn_llm") -> list[Document]:
    """从 ChromaDB 加载所有文档用于 BM25 索引。"""
    vs = Chroma(collection_name=collection_name, embedding_function=get_embeddings(),
                persist_directory="./indexes/chroma")
    raw = vs.get()
    return [Document(page_content=c, metadata=m)
            for c, m in zip(raw["documents"], raw["metadatas"]) if c]
\`\`\`

### BM25 vs 向量搜索

| 查询类型 | BM25 | 向量 |
|---|---|---|
| "LangGraph \`add_conditional_edges\`" | ✅ 精确匹配 | ❌ 可能遗漏精确名称 |
| "产品 SKU ABC-123" | ✅ 精确关键词 | ❌ ID/代码效果差 |
| "什么是基于图的编排？" | ❌ 遗漏释义 | ✅ 语义匹配 |`,
  },
}

export const ensembleRetriever: TopicContent = {
  id: 'ensemble-retriever',
  title: { en: 'Ensemble Retriever (RRF)', zh: '集成检索器（RRF）' },
  contentType: 'code',
  content: {
    en: `## Ensemble Retriever (Reciprocal Rank Fusion)

The \`EnsembleRetriever\` combines results from BM25 and vector retrieval using **Reciprocal Rank Fusion (RRF)** — a score fusion algorithm that rewards documents appearing in the top results of multiple retrievers.

### RRF Formula

\`\`\`
RRF_score(d) = Σ 1 / (k + rank_i(d))

where:
  rank_i(d) = rank of document d in retriever i's results
  k = 60 (smoothing constant)
\`\`\`

A document ranked #1 by both retrievers scores ~0.033 from each = 0.066 total.
A document ranked #1 by only one retriever scores only ~0.033.

### Building the Ensemble

\`\`\`python
# phase3_enterprise/hybrid_rag/ensemble_retriever.py
from langchain.retrievers import EnsembleRetriever
from phase3_enterprise.hybrid_rag.bm25_retriever import build_bm25_retriever, load_docs_from_chroma
from shared.vector_store import get_retriever
from shared.config import settings


def build_ensemble_retriever(
    k: int = 5,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    collection_name: str = "learn_llm",
) -> EnsembleRetriever:
    """Build BM25 + Vector ensemble with configurable weights.

    Weights must sum to 1.0.
    Recommended: 0.4/0.6 — favour vector for most conceptual corpora.
    Adjust to 0.6/0.4 for keyword-heavy content (legal, medical codes).
    """
    if abs(bm25_weight + vector_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0")

    # BM25 needs the full corpus in memory
    all_docs = load_docs_from_chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    if not all_docs:
        raise ValueError(
            "No documents in ChromaDB. "
            "Run: python phase1_foundation/03_simple_rag/ingest.py --dir data/"
        )

    bm25 = build_bm25_retriever(all_docs, k=k)
    vector = get_retriever(backend="chroma", k=k, collection_name=collection_name)

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[bm25_weight, vector_weight],
    )
\`\`\`

### Weight Tuning Guide

\`\`\`python
# For conceptual documentation (APIs, tutorials)
retriever = build_ensemble_retriever(bm25_weight=0.3, vector_weight=0.7)

# For technical reference (code snippets, function names)
retriever = build_ensemble_retriever(bm25_weight=0.6, vector_weight=0.4)

# Balanced default
retriever = build_ensemble_retriever(bm25_weight=0.4, vector_weight=0.6)
\`\`\`

### Usage

\`\`\`python
retriever = build_ensemble_retriever(k=5, bm25_weight=0.4, vector_weight=0.6)

docs = retriever.invoke("how does LCEL pipe syntax work?")
for doc in docs:
    print(doc.page_content[:200])
\`\`\``,

    zh: `## 集成检索器（倒数排名融合）

\`EnsembleRetriever\` 使用**倒数排名融合（RRF）**合并 BM25 和向量检索的结果 — 这是一种分数融合算法，奖励出现在多个检索器顶部结果中的文档。

### 构建集成检索器

\`\`\`python
def build_ensemble_retriever(
    k: int = 5,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    collection_name: str = "learn_llm",
) -> EnsembleRetriever:
    all_docs = load_docs_from_chroma(collection_name=collection_name)
    bm25 = build_bm25_retriever(all_docs, k=k)
    vector = get_retriever(backend="chroma", k=k, collection_name=collection_name)

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[bm25_weight, vector_weight],
    )
\`\`\`

### 权重调整指南

| 内容类型 | BM25 权重 | 向量权重 |
|---|---|---|
| 概念文档（API、教程） | 0.3 | 0.7 |
| 技术参考（代码、函数名） | 0.6 | 0.4 |
| 平衡默认值 | 0.4 | 0.6 |`,
  },
}

export const rerankerDetails: TopicContent = {
  id: 'reranker-details',
  title: { en: 'Cross-Encoder Reranker', zh: '交叉编码器重排序器' },
  contentType: 'code',
  content: {
    en: `## Cross-Encoder Reranker

After retrieval, cosine similarity gives a rough ranking. A **cross-encoder reranker** applies a dedicated model that jointly encodes the query + each document together for much more accurate relevance scoring.

### Cohere Rerank (API-based, highest quality)

\`\`\`python
# phase3_enterprise/hybrid_rag/reranker.py
import cohere
from langchain_core.documents import Document
from shared.config import settings


def rerank_with_cohere(
    docs: list[Document],
    query: str,
    top_n: int = 3,
) -> list[Document]:
    """Rerank using Cohere Rerank v3 API (best quality).

    Requires COHERE_API_KEY in .env.
    """
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY not set. Get one at https://cohere.com")

    client = cohere.Client(api_key=settings.cohere_api_key)
    texts = [doc.page_content for doc in docs]

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        doc = docs[result.index]
        doc.metadata["rerank_score"] = result.relevance_score
        reranked.append(doc)

    return reranked
\`\`\`

### CrossEncoder (Local, no API key needed)

\`\`\`python
from sentence_transformers import CrossEncoder


def rerank_with_crossencoder(
    docs: list[Document],
    query: str,
    top_n: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[Document]:
    """Rerank using a local cross-encoder model (no API, runs offline)."""
    model = CrossEncoder(model_name)

    # Score each (query, document) pair together
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    # Sort descending and take top_n
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = []
    for score, doc in scored[:top_n]:
        doc.metadata["rerank_score"] = float(score)
        result.append(doc)

    return result
\`\`\`

### Auto-Fallback Strategy

\`\`\`python
def rerank_documents(
    docs: list[Document],
    query: str,
    top_n: int = 3,
    method: str = "auto",
) -> list[Document]:
    """Rerank with the best available method.

    method: "auto" → try Cohere first, fall back to CrossEncoder
            "cohere" → force Cohere (raises if no API key)
            "crossencoder" → force local model
    """
    if method == "cohere" or (method == "auto" and settings.cohere_api_key):
        return rerank_with_cohere(docs, query, top_n)
    return rerank_with_crossencoder(docs, query, top_n)
\`\`\`

### Bi-encoder vs Cross-encoder

\`\`\`
Bi-encoder (used in vector search):
  query → embed independently → [0.2, -0.5, ...]
  doc   → embed independently → [0.21, -0.49, ...]
  similarity = cosine(query_vec, doc_vec)
  ✅ Fast (pre-computed vectors)
  ❌ Less accurate (query and doc don't interact)

Cross-encoder (reranker):
  [query + doc] → joint encoding → relevance score
  ✅ Much more accurate (query/doc interact directly)
  ❌ Slow (must re-run for every query+doc pair)
  → Use AFTER retrieval to rerank top-k candidates
\`\`\`

### Performance Comparison

| Method | Speed | Quality | API Cost |
|---|---|---|---|
| Vector only | Fast | Good | Free (local) |
| BM25 only | Fast | Good for keywords | Free |
| Ensemble | Moderate | Better | Free |
| **Ensemble + Reranker** | Moderate | **Best** | Free (CrossEncoder) or ~$0.001/call (Cohere) |`,

    zh: `## 交叉编码器重排序器

检索后，余弦相似度给出粗略排名。**交叉编码器重排序器**应用一个专用模型，将查询 + 每个文档联合编码在一起，以获得更准确的相关性评分。

### Cohere Rerank（基于 API，最高质量）

\`\`\`python
def rerank_with_cohere(docs, query, top_n=3):
    client = cohere.Client(api_key=settings.cohere_api_key)
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n=top_n,
    )
    for result in response.results:
        docs[result.index].metadata["rerank_score"] = result.relevance_score
    return [docs[r.index] for r in response.results]
\`\`\`

### CrossEncoder（本地，无需 API 密钥）

\`\`\`python
def rerank_with_crossencoder(docs, query, top_n=3):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]
\`\`\`

### 双编码器 vs 交叉编码器

\`\`\`
双编码器（向量搜索中使用）：
  独立嵌入查询和文档 → 余弦相似度
  ✅ 快速（预计算向量）
  ❌ 精度较低（查询和文档不交互）

交叉编码器（重排序器）：
  [查询 + 文档] → 联合编码 → 相关性分数
  ✅ 精度更高（查询/文档直接交互）
  ❌ 慢（每个查询+文档对必须重新运行）
\`\`\``,
  },
}
