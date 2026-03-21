import type { TopicContent } from '../types'

export const sharedConfig: TopicContent = {
  id: 'shared-config',
  title: { en: 'Centralised Settings (pydantic-settings)', zh: '集中配置（pydantic-settings）' },
  contentType: 'code',
  content: {
    en: `## Centralised Settings with pydantic-settings

\`shared/config.py\` is the **single source of truth** for all environment-driven configuration. Never call \`os.getenv()\` directly — import \`settings\` instead.

### The Settings Class

\`\`\`python
# shared/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-driven config for the learn-llm project.

    Values are loaded from .env automatically.
    Override any value by setting the env var at runtime.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # ignore unknown env vars
    )

    # ── Backend selector ──────────────────────────────────────────────────────
    llm_backend: str = "ollama"           # "ollama" or "openai"
    embedding_backend: str = "ollama"

    # ── Ollama (local, zero cost) ─────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── OpenAI (cloud) ────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── LangSmith observability ───────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "learn-llm-dev"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ── Vector databases ──────────────────────────────────────────────────────
    chroma_persist_dir: str = "./indexes/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "learn_llm"

    # ── MCP servers (Phase 2) ─────────────────────────────────────────────────
    mcp_filesystem_root: str = "./data"
    mcp_sqlite_path: str = "./data/app.db"

    # ── Enterprise MCP (Phase 3) ──────────────────────────────────────────────
    github_token: str = ""
    slack_bot_token: str = ""
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""

    # ── External services ─────────────────────────────────────────────────────
    tavily_api_key: str = ""    # web search
    cohere_api_key: str = ""    # reranker

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"


# Singleton — import this object everywhere, not the class
settings = Settings()
\`\`\`

### The \`.env\` File

\`\`\`bash
# .env.example — copy to .env and fill in your values

# Backend — uncomment one
LLM_BACKEND=ollama
# LLM_BACKEND=openai

# Ollama (default — zero cost, fully local)
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# OpenAI (optional)
# OPENAI_API_KEY=sk-...

# LangSmith (optional — enables tracing)
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=ls__...
# LANGSMITH_PROJECT=my-project

# Enterprise MCP (optional)
# GITHUB_TOKEN=ghp_...
# SLACK_BOT_TOKEN=xoxb-...
# COHERE_API_KEY=...
# TAVILY_API_KEY=...
\`\`\`

### Usage Pattern

\`\`\`python
# In any file — just import settings
from shared.config import settings

# Access any config value
backend = settings.llm_backend        # "ollama"
model   = settings.ollama_model       # "qwen3:8b"
db_path = settings.mcp_sqlite_path   # "./data/app.db"

# Guard against missing required keys
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in .env")
\`\`\`

### Why Not \`os.getenv\`?

| Approach | Problem |
|---|---|
| \`os.getenv("MODEL")\` | No type validation, no defaults, no \`.env\` loading |
| \`settings.ollama_model\` | Typed, validated, documented, IDE-autocomplete |`,

    zh: `## 使用 pydantic-settings 集中配置

\`shared/config.py\` 是所有环境驱动配置的**单一真相来源**。永远不要直接调用 \`os.getenv()\` — 改为导入 \`settings\`。

### 设置类

\`\`\`python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: str = "ollama"
    ollama_model: str = "qwen3:8b"
    openai_api_key: str = ""
    langsmith_tracing: bool = False
    chroma_persist_dir: str = "./indexes/chroma"
    github_token: str = ""
    # ... 等等

settings = Settings()  # 单例 — 到处导入这个对象
\`\`\`

### 使用模式

\`\`\`python
from shared.config import settings

backend = settings.llm_backend      # "ollama"
model   = settings.ollama_model     # "qwen3:8b"

if not settings.openai_api_key:
    raise ValueError("未在 .env 中设置 OPENAI_API_KEY")
\`\`\`

### 为什么不用 \`os.getenv\`？

| 方法 | 问题 |
|---|---|
| \`os.getenv("MODEL")\` | 无类型验证，无默认值，无 \`.env\` 加载 |
| \`settings.ollama_model\` | 已类型化、已验证、有文档、IDE 自动完成 |`,
  },
}

export const llmFactory: TopicContent = {
  id: 'llm-factory',
  title: { en: 'LLM Factory (Backend-Agnostic)', zh: 'LLM 工厂（后端无关）' },
  contentType: 'code',
  content: {
    en: `## LLM Factory (Backend-Agnostic)

\`shared/llm_factory.py\` provides a single \`get_llm()\` function that returns the correct chat model based on \`LLM_BACKEND\` in your \`.env\`. Swap between Ollama (local) and OpenAI (cloud) by changing one line.

### The Factory

\`\`\`python
# shared/llm_factory.py
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from shared.config import settings


def get_llm(
    temperature: float = 0.0,
    streaming: bool = False,
    model: str | None = None,
) -> BaseChatModel:
    """Central LLM factory — routes to Ollama or OpenAI based on LLM_BACKEND.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
        streaming:   Enable token streaming for real-time output.
        model:       Override model name per-call.
    """
    backend = settings.llm_backend.lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError("LLM_BACKEND=openai but OPENAI_API_KEY not set in .env")
        return ChatOpenAI(
            model=model or settings.openai_model,
            temperature=temperature,
            streaming=streaming,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown LLM_BACKEND: {settings.llm_backend!r}. "
        "Set LLM_BACKEND=ollama or LLM_BACKEND=openai in .env"
    )
\`\`\`

### Embeddings Factory

\`\`\`python
def get_embeddings(model: str | None = None) -> Embeddings:
    """Central embeddings factory.

    Embedding dimensions by model:
        nomic-embed-text (Ollama) → 768 dims  (free, local)
        mxbai-embed-large (Ollama) → 1024 dims
        text-embedding-3-small (OpenAI) → 1536 dims ($0.02/M tokens)
    """
    backend = settings.embedding_backend.lower()

    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=model or settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_BACKEND=openai but OPENAI_API_KEY not set")
        return OpenAIEmbeddings(
            model=model or settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend!r}")
\`\`\`

### Model Discovery (Ollama)

\`\`\`python
def list_ollama_models() -> list[str]:
    """Return names of all locally available Ollama models."""
    import urllib.request, json
    try:
        url = f"{settings.ollama_base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# Usage
models = list_ollama_models()
print(models)  # ['qwen3:8b', 'qwen2.5:14b', 'nomic-embed-text', ...]
\`\`\`

### Usage Patterns

\`\`\`python
from shared.llm_factory import get_llm, get_embeddings

# Default LLM (from LLM_BACKEND in .env)
llm = get_llm()
llm = get_llm(temperature=0.7)              # creative writing
llm = get_llm(model="qwen3-coder:30b")      # code specialist model

# Embeddings
embeddings = get_embeddings()
vec = embeddings.embed_query("What is RAG?")
print(len(vec))   # 768 (Ollama nomic-embed-text)

# In LCEL chains
from langchain_core.output_parsers import StrOutputParser
chain = prompt | get_llm() | StrOutputParser()
\`\`\`

### Backend Comparison

| Feature | Ollama (local) | OpenAI (cloud) |
|---|---|---|
| Cost | Free | Pay-per-token |
| Privacy | 100% local | Data sent to OpenAI |
| Speed | Depends on GPU | Fast (optimised infra) |
| Model choice | Many open-source | GPT-4o, GPT-4o-mini |
| Setup | \`ollama pull qwen3:8b\` | Set \`OPENAI_API_KEY\` |`,

    zh: `## LLM 工厂（后端无关）

\`shared/llm_factory.py\` 提供一个 \`get_llm()\` 函数，根据 \`.env\` 中的 \`LLM_BACKEND\` 返回正确的聊天模型。通过更改一行在 Ollama（本地）和 OpenAI（云）之间切换。

### 工厂函数

\`\`\`python
def get_llm(temperature=0.0, model=None) -> BaseChatModel:
    backend = settings.llm_backend.lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or settings.ollama_model, temperature=temperature)

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or settings.openai_model, temperature=temperature,
                          api_key=settings.openai_api_key)
\`\`\`

### 后端比较

| 特性 | Ollama（本地） | OpenAI（云） |
|---|---|---|
| 成本 | 免费 | 按 token 付费 |
| 隐私 | 100% 本地 | 数据发送给 OpenAI |
| 模型选择 | 许多开源模型 | GPT-4o、GPT-4o-mini |
| 设置 | \`ollama pull qwen3:8b\` | 设置 \`OPENAI_API_KEY\` |`,
  },
}

export const vectorStore: TopicContent = {
  id: 'vector-store',
  title: { en: 'Vector Store Abstraction (Chroma/Qdrant)', zh: '向量存储抽象（Chroma/Qdrant）' },
  contentType: 'code',
  content: {
    en: `## Vector Store Abstraction (Chroma/Qdrant)

\`shared/vector_store.py\` provides a unified \`get_vector_store()\` API that works with both ChromaDB (local development) and Qdrant (production). Switch backends without changing any chain code.

### The Abstraction

\`\`\`python
# shared/vector_store.py
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from shared.config import settings
from shared.llm_factory import get_embeddings


def get_vector_store(
    backend: str = "chroma",
    collection_name: str | None = None,
) -> VectorStore:
    """Return a configured vector store for the specified backend.

    Args:
        backend:         "chroma" (local, Phases 1–2) or "qdrant" (Phases 3–4).
        collection_name: Override collection name.
    """
    embeddings = get_embeddings()

    # ── Chroma: local, zero infra, file-based ──────────────────────────────
    if backend == "chroma":
        return Chroma(
            collection_name=collection_name or "learn_llm",
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    # ── Qdrant: enterprise, supports filtering, clustering ─────────────────
    if backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        client = QdrantClient(url=settings.qdrant_url)
        col = collection_name or settings.qdrant_collection

        # Infer embedding dimension from backend
        embed_dim = 768 if settings.embedding_backend == "ollama" else 1536

        # Auto-create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if col not in existing:
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )

        return QdrantVectorStore(
            client=client,
            collection_name=col,
            embedding=embeddings,
        )

    raise ValueError(f"Unsupported backend: {backend!r}. Use 'chroma' or 'qdrant'.")


def get_retriever(
    backend: str = "chroma",
    k: int = 4,
    collection_name: str | None = None,
):
    """Convenience wrapper: get a retriever from the vector store."""
    vs = get_vector_store(backend, collection_name)
    return vs.as_retriever(search_kwargs={"k": k})
\`\`\`

### Usage Examples

\`\`\`python
from shared.vector_store import get_vector_store, get_retriever

# Phase 1–2: local development with ChromaDB
vs = get_vector_store("chroma")
results = vs.similarity_search("What is LCEL?", k=3)

# Phase 3–4: production with Qdrant (Docker)
vs = get_vector_store("qdrant")
results = vs.similarity_search("LangGraph state management", k=5)

# Get a retriever for LCEL chains
retriever = get_retriever(backend="chroma", k=4)
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | ...
\`\`\`

### Chroma vs Qdrant

| Feature | ChromaDB | Qdrant |
|---|---|---|
| Setup | Zero — file-based | Docker: \`docker-compose up qdrant\` |
| Persistence | Local file (\`./indexes/chroma\`) | HTTP server (\`localhost:6333\`) |
| Filtering | Basic metadata filtering | Advanced payload filtering |
| Scalability | Local use, small datasets | Production, millions of vectors |
| Best for | Phase 1–2 development | Phase 3–4 enterprise |

### Backend Switch

\`\`\`bash
# To switch from Chroma to Qdrant:
# 1. Start Qdrant:
docker-compose up qdrant

# 2. Re-ingest documents (or migrate existing vectors)
python phase1_foundation/03_simple_rag/ingest.py --dir data/ --backend qdrant

# 3. Update .env — no code changes needed:
# VECTOR_BACKEND=qdrant  (if you add this setting)
# All code that calls get_vector_store() works unchanged
\`\`\``,

    zh: `## 向量存储抽象（Chroma/Qdrant）

\`shared/vector_store.py\` 提供统一的 \`get_vector_store()\` API，适用于 ChromaDB（本地开发）和 Qdrant（生产）。无需更改任何链代码即可切换后端。

### 抽象

\`\`\`python
def get_vector_store(backend="chroma", collection_name=None) -> VectorStore:
    embeddings = get_embeddings()

    if backend == "chroma":
        return Chroma(
            collection_name=collection_name or "learn_llm",
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    if backend == "qdrant":
        # 自动创建集合（如果不存在）
        client = QdrantClient(url=settings.qdrant_url)
        return QdrantVectorStore(client=client, collection_name=..., embedding=embeddings)

def get_retriever(backend="chroma", k=4, collection_name=None):
    return get_vector_store(backend, collection_name).as_retriever(search_kwargs={"k": k})
\`\`\`

### Chroma vs Qdrant

| 特性 | ChromaDB | Qdrant |
|---|---|---|
| 设置 | 零 — 基于文件 | Docker：\`docker-compose up qdrant\` |
| 可扩展性 | 本地使用，小数据集 | 生产，数百万向量 |
| 最适合 | 阶段 1–2 开发 | 阶段 3–4 企业 |`,
  },
}
