import type { TopicContent } from '../types'

export const hybridRAG: TopicContent = {
  id: 'hybrid-rag',
  emoji: '🔍',
  title: { en: 'Hybrid RAG with Reranking', zh: '混合 RAG 与重排序' },
  contentType: 'code',
  content: {
    en: `## Hybrid RAG with Reranking

Simple vector search misses keyword matches. Simple BM25 misses semantic similarity. **Hybrid RAG** combines both via an EnsembleRetriever, then applies a cross-encoder reranker for final precision.

### Architecture

\`\`\`
Query
  │
  ├─── BM25 Retriever  (sparse — keyword exact match)    ─┐
  │                                                        ├── EnsembleRetriever (top-k docs)
  └─── Vector Retriever (dense — semantic similarity)    ─┘
                │
                ▼
         Reranker (cross-encoder — reorder by relevance score)
                │
                ▼
         Top-N docs with [Rank, score, source] annotation
                │
                ▼
         RAG Prompt → LLM → Answer
\`\`\`

### Building the Ensemble Retriever

\`\`\`python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def build_ensemble_retriever(k: int = 10) -> EnsembleRetriever:
    # Load all documents for BM25
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader("data/docs/", glob="**/*.txt")
    docs = loader.load()

    # BM25: sparse retriever using TF-IDF-like scoring
    bm25 = BM25Retriever.from_documents(docs, k=k)

    # Vector: dense semantic retriever
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="data/chroma_db",
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Ensemble: merge results (0.5 BM25 + 0.5 vector by default)
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5],
    )
\`\`\`

### Reranker

\`\`\`python
from sentence_transformers import CrossEncoder

_reranker = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank_documents(docs, query: str, top_n: int = 3):
    """Score all docs against the query and return top_n by score."""
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)

    # Sort descending and return top_n
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
\`\`\`

### Full Hybrid Chain

\`\`\`python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

HYBRID_RAG_PROMPT = PromptTemplate(
    template=(
        "You are an expert assistant using high-precision hybrid retrieval.\\n"
        "Answer the question based ONLY on the context below.\\n"
        "Context is ranked by relevance — prioritize documents at the top.\\n\\n"
        "Context:\\n{context}\\n\\n"
        "Question: {question}\\n\\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


def _format_reranked_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get("rerank_score", 0)
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Rank {i} | score={score:.3f} | {source}]\\n{doc.page_content}")
    return "\\n\\n".join(parts)


def build_hybrid_rag_chain(k: int = 10, top_n: int = 3):
    ensemble = build_ensemble_retriever(k=k)
    llm = get_llm(temperature=0.0)

    def retrieve_and_rerank(query: str) -> str:
        docs = ensemble.invoke(query)
        reranked = rerank_documents(docs, query=query, top_n=top_n)
        return _format_reranked_docs(reranked)

    return (
        {
            "context": RunnableLambda(retrieve_and_rerank),
            "question": RunnablePassthrough(),
        }
        | HYBRID_RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# Usage
chain = build_hybrid_rag_chain(k=10, top_n=3)
answer = chain.invoke({"question": "Explain BM25 vs vector search."})
print(answer)
\`\`\`

### When to Use Each Retriever

| Retriever | Best for |
|---|---|
| BM25 only | Keyword-heavy queries, exact product names, IDs |
| Vector only | Semantic/conceptual questions |
| **Hybrid + Reranker** | Production: highest accuracy, handles both cases |

### Performance Tips

- Tune \`weights=[0.4, 0.6]\` — favour dense if docs are long and conceptual
- \`top_n=3\` balances context window size vs coverage
- Cache the \`CrossEncoder\` instance (global singleton pattern above)`,

    zh: `## 混合 RAG 与重排序

简单向量搜索会错过关键词匹配。简单 BM25 会错过语义相似性。**混合 RAG** 通过 EnsembleRetriever 结合两者，然后应用交叉编码器重排序器实现最终精度。

### 架构

\`\`\`
查询
  │
  ├─── BM25 检索器  (稀疏 — 关键词精确匹配)    ─┐
  │                                              ├── EnsembleRetriever (前 k 个文档)
  └─── 向量检索器  (密集 — 语义相似性)         ─┘
                │
                ▼
         重排序器（交叉编码器 — 按相关性分数重新排序）
                │
                ▼
         带有 [排名, 分数, 来源] 注释的前 N 个文档
                │
                ▼
         RAG 提示 → LLM → 答案
\`\`\`

### 构建集成检索器

\`\`\`python
def build_ensemble_retriever(k: int = 10) -> EnsembleRetriever:
    # 为 BM25 加载所有文档
    docs = DirectoryLoader("data/docs/").load()

    # BM25：稀疏检索器
    bm25 = BM25Retriever.from_documents(docs, k=k)

    # 向量：密集语义检索器
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="data/chroma_db",
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 集成：合并结果（默认 0.5 BM25 + 0.5 向量）
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5],
    )
\`\`\`

### 重排序器

\`\`\`python
from sentence_transformers import CrossEncoder

def rerank_documents(docs, query: str, top_n: int = 3):
    """对所有文档针对查询打分，返回得分最高的 top_n 个。"""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
\`\`\`

### 何时使用各检索器

| 检索器 | 最适合 |
|---|---|
| 仅 BM25 | 关键词繁重的查询、精确产品名称、ID |
| 仅向量 | 语义/概念性问题 |
| **混合 + 重排序** | 生产环境：最高精度，处理两种情况 |`,
  },
}

export const langGraphMultiAgent: TopicContent = {
  id: 'langgraph-multi-agent',
  emoji: '🕸️',
  title: { en: 'LangGraph Multi-Agent System', zh: 'LangGraph 多智能体系统' },
  contentType: 'code',
  content: {
    en: `## LangGraph Multi-Agent System

LangGraph models agent orchestration as a **directed graph** where nodes are agents and edges carry routing decisions. This enables complex multi-agent pipelines with loops, conditional branching, and shared state.

### Shared State Schema

\`\`\`python
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph.

    Attributes:
        messages:        Full conversation history (append-only).
        task:            The original user task.
        research_output: Raw results from the researcher agent.
        final_report:    Polished report from the writer agent.
        next_step:       Routing decision from supervisor ("researcher"|"writer"|"end").
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    research_output: str
    final_report: str
    next_step: str
\`\`\`

### Graph Topology

\`\`\`
START → supervisor ──┬─→ researcher → supervisor (loop back)
                     ├─→ writer → END
                     └─→ END
\`\`\`

### Building the Graph

\`\`\`python
def route_to_agent(state: AgentState) -> str:
    """Conditional edge: read supervisor's routing decision."""
    next_step = state.get("next_step", "end").lower()
    if next_step == "researcher":
        return "researcher"
    if next_step == "writer":
        return "writer"
    return END


def build_graph(supervisor_node, researcher_node, writer_node) -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            END: END,
        },
    )

    # After research: return to supervisor for next decision
    builder.add_edge("researcher", "supervisor")

    # After writing: job done
    builder.add_edge("writer", END)

    return builder.compile()
\`\`\`

### Agent Node Implementations

\`\`\`python
# Supervisor: decides who acts next
def supervisor_node(state: AgentState) -> dict:
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a project supervisor.
Given the task and current state, decide what to do next.
Respond with ONLY one of: researcher, writer, end"""),
        ("human", "Task: {task}\\nResearch done: {has_research}\\nDecide next step:"),
    ])

    llm = get_llm(temperature=0.0)
    chain = supervisor_prompt | llm | StrOutputParser()

    decision = chain.invoke({
        "task": state["task"],
        "has_research": bool(state.get("research_output")),
    }).strip().lower()

    return {"next_step": decision}


# Researcher: gathers information
def researcher_node(state: AgentState) -> dict:
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research specialist. Gather key facts about the topic."),
        ("human", "Research this thoroughly: {task}"),
    ])
    llm = get_llm(temperature=0.2)
    chain = research_prompt | llm | StrOutputParser()
    research = chain.invoke({"task": state["task"]})
    return {"research_output": research}


# Writer: synthesises into a report
def writer_node(state: AgentState) -> dict:
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Create a clear, structured report."),
        ("human", "Task: {task}\\n\\nResearch:\\n{research}\\n\\nWrite report:"),
    ])
    llm = get_llm(temperature=0.3)
    chain = writer_prompt | llm | StrOutputParser()
    report = chain.invoke({
        "task": state["task"],
        "research": state.get("research_output", ""),
    })
    return {"final_report": report}
\`\`\`

### Running the Graph

\`\`\`python
graph = build_graph(supervisor_node, researcher_node, writer_node)

result = graph.invoke({
    "task": "Summarise the main LangGraph features and use cases",
    "messages": [],
    "research_output": "",
    "final_report": "",
    "next_step": "",
})

print(result["final_report"])
\`\`\`

### Key LangGraph Concepts

| Concept | Description |
|---|---|
| \`StateGraph\` | Graph where nodes share typed state |
| \`add_messages\` | Reducer that appends instead of replacing |
| \`add_conditional_edges\` | Route to different nodes based on state |
| \`builder.compile()\` | Returns executable graph |
| \`add_edge(A, B)\` | Deterministic transition from A to B |`,

    zh: `## LangGraph 多智能体系统

LangGraph 将代理编排建模为**有向图**，其中节点是代理，边携带路由决策。这支持具有循环、条件分支和共享状态的复杂多代理流水线。

### 共享状态模式

\`\`\`python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    research_output: str
    final_report: str
    next_step: str  # "researcher" | "writer" | "end"
\`\`\`

### 图拓扑

\`\`\`
START → supervisor ──┬─→ researcher → supervisor（循环）
                     ├─→ writer → END
                     └─→ END
\`\`\`

### 构建图

\`\`\`python
def build_graph(supervisor_node, researcher_node, writer_node) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {"researcher": "researcher", "writer": "writer", END: END},
    )

    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", END)

    return builder.compile()
\`\`\`

### 运行图

\`\`\`python
graph = build_graph(supervisor_node, researcher_node, writer_node)

result = graph.invoke({
    "task": "总结 LangGraph 的主要特性和用例",
    "messages": [],
    "research_output": "",
    "final_report": "",
    "next_step": "",
})

print(result["final_report"])
\`\`\`

### 关键 LangGraph 概念

| 概念 | 描述 |
|---|---|
| \`StateGraph\` | 节点共享类型状态的图 |
| \`add_messages\` | 追加而非替换的 reducer |
| \`add_conditional_edges\` | 根据状态路由到不同节点 |
| \`builder.compile()\` | 返回可执行图 |`,
  },
}
