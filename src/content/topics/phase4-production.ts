import type { TopicContent } from '../types'

export const fastAPIProduction: TopicContent = {
  id: 'fastapi-production',
  title: { en: 'FastAPI Production API', zh: 'FastAPI 生产级 API' },
  contentType: 'code',
  content: {
    en: `## FastAPI Production API

A production LLM API needs health checks, background ingestion, RAG-powered chat, and evaluation endpoints — all properly structured with dependency injection.

### App Entry Point

\`\`\`python
# phase4_production/api/main.py
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks."""
    print("🚀 Starting learn-llm API server...")
    configure_langsmith()   # enable tracing if LANGSMITH_API_KEY is set
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title="learn-llm API",
    description="Production-grade RAG + agent API powered by LangChain",
    version="0.1.0",
    lifespan=lifespan,
)

add_middleware(app)   # CORS, logging, rate limiting
\`\`\`

### Pydantic Schemas

\`\`\`python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    use_rag: bool = True

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class IngestRequest(BaseModel):
    file_path: str
    collection_name: str = "knowledge_base"
    chunk_size: int = 512
    chunk_overlap: int = 64

class IngestResponse(BaseModel):
    status: str
    collection: str
    message: str

class EvalRequest(BaseModel):
    question: str
    context: str
    answer: str

class EvalResponse(BaseModel):
    context_precision: float
    answer_faithfulness: float
    answer_relevance: float
    overall: float
    explanation: str
\`\`\`

### Health Check

\`\`\`python
@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
\`\`\`

### Chat Endpoint (RAG-powered)

\`\`\`python
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    rag_chain=Depends(get_rag_chain),
    plain_chain=Depends(get_plain_chain),
) -> ChatResponse:
    """Answer using RAG or plain LLM, with optional LangSmith tracing."""
    chain = rag_chain if request.use_rag else plain_chain

    if settings.langsmith_tracing:
        from langsmith import traceable

        @traceable(project_name=settings.langsmith_project)
        def _invoke(q):
            return chain.invoke({"question": q})
        answer = _invoke(request.message)
    else:
        answer = chain.invoke({"question": request.message})

    return ChatResponse(answer=answer, session_id=request.session_id)
\`\`\`

### Background Ingest Endpoint

\`\`\`python
def _run_ingest(file_path: str, collection: str, chunk_size: int, chunk_overlap: int):
    """Background task: ingest document into ChromaDB."""
    from pathlib import Path
    from phase1_foundation.simple_rag_ingest import ingest
    ingest(str(Path("data") / file_path), collection, chunk_size, chunk_overlap)
    print(f"✅ Ingest complete: {file_path} → {collection}")


@app.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Accept document for ingestion (runs asynchronously in background)."""
    background_tasks.add_task(
        _run_ingest,
        request.file_path,
        request.collection_name,
        request.chunk_size,
        request.chunk_overlap,
    )
    return IngestResponse(
        status="accepted",
        collection=request.collection_name,
        message=f"Ingestion of '{request.file_path}' started in background.",
    )
\`\`\`

### Evaluate Endpoint (LLM-as-Judge)

\`\`\`python
@app.post("/evaluate", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate(request: EvalRequest) -> EvalResponse:
    """Score a RAG response: context precision, faithfulness, relevance."""
    score = evaluate_rag_response(
        question=request.question,
        context=request.context,
        answer=request.answer,
    )
    return EvalResponse(
        context_precision=score.context_precision,
        answer_faithfulness=score.answer_faithfulness,
        answer_relevance=score.answer_relevance,
        overall=score.overall,
        explanation=score.explanation,
    )
\`\`\`

### Running the Server

\`\`\`bash
# Development
uvicorn phase4_production.api.main:app --reload --port 8000

# Docker
docker-compose up api

# Test the API
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is LCEL?", "session_id": "test", "use_rag": true}'
\`\`\`

### API Design Principles

| Principle | Implementation |
|---|---|
| Background tasks | \`BackgroundTasks\` for slow ingestion |
| Dependency injection | \`Depends(get_rag_chain)\` for testability |
| Lifecycle hooks | \`lifespan\` for startup/shutdown |
| LangSmith tracing | Conditional \`@traceable\` wrapper |
| Schema validation | Pydantic \`BaseModel\` for all I/O |`,

    zh: `## FastAPI 生产级 API

生产级 LLM API 需要健康检查、后台摄取、RAG 驱动的聊天和评估端点 — 所有端点都通过依赖注入适当结构化。

### 应用入口点

\`\`\`python
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 启动 learn-llm API 服务器...")
    configure_langsmith()
    yield
    print("🛑 关闭中。")

app = FastAPI(
    title="learn-llm API",
    version="0.1.0",
    lifespan=lifespan,
)
add_middleware(app)
\`\`\`

### 聊天端点（RAG 驱动）

\`\`\`python
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_chain=Depends(get_rag_chain),
    plain_chain=Depends(get_plain_chain),
) -> ChatResponse:
    chain = rag_chain if request.use_rag else plain_chain
    answer = chain.invoke({"question": request.message})
    return ChatResponse(answer=answer, session_id=request.session_id)
\`\`\`

### 后台摄取端点

\`\`\`python
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    background_tasks.add_task(
        _run_ingest, request.file_path, request.collection_name,
        request.chunk_size, request.chunk_overlap,
    )
    return IngestResponse(status="accepted", collection=request.collection_name,
                          message=f"摄取 '{request.file_path}' 已在后台启动。")
\`\`\`

### 评估端点（LLM 作为评判者）

\`\`\`python
@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest) -> EvalResponse:
    score = evaluate_rag_response(
        question=request.question,
        context=request.context,
        answer=request.answer,
    )
    return EvalResponse(
        context_precision=score.context_precision,
        answer_faithfulness=score.answer_faithfulness,
        answer_relevance=score.answer_relevance,
        overall=score.overall,
        explanation=score.explanation,
    )
\`\`\`

### API 设计原则

| 原则 | 实现 |
|---|---|
| 后台任务 | 慢速摄取使用 \`BackgroundTasks\` |
| 依赖注入 | \`Depends(get_rag_chain)\` 用于可测试性 |
| 生命周期钩子 | \`lifespan\` 用于启动/关闭 |
| LangSmith 追踪 | 条件 \`@traceable\` 包装器 |
| 模式验证 | Pydantic \`BaseModel\` 用于所有 I/O |`,
  },
}

export const selfHealingChains: TopicContent = {
  id: 'self-healing-chains',
  title: { en: 'Self-Healing Chains', zh: '自愈链' },
  contentType: 'code',
  content: {
    en: `## Self-Healing Chains

LLMs occasionally generate invalid code or SQL. Self-healing wraps the generate → execute → correct loop so errors automatically feed back into the LLM for repair.

### The Pattern

\`\`\`
generate(request)
    │
    ▼
execute(output) ──── success ──→ return result
    │
    failure
    │
    ▼
correct(output + error_message)
    │
    ▼
execute(corrected) ──── repeat up to max_retries
\`\`\`

### Generic Retry Chain

\`\`\`python
from typing import Any, Callable
from langchain_core.runnables import Runnable


def build_retry_chain(
    generator_chain: Runnable,
    executor_fn: Callable[[str], Any],
    corrector_chain: Runnable,
    max_retries: int = 3,
):
    """Build a self-healing chain that retries on execution failure.

    Args:
        generator_chain: Produces initial code/SQL (e.g., SQL generator prompt | LLM)
        executor_fn:     Executes the output and raises on failure
        corrector_chain: Takes {failed_code, error_message, ...} and returns fixed code
        max_retries:     Maximum correction attempts
    """

    def run_with_retry(state: dict[str, Any]) -> dict[str, Any]:
        request = state["request"]
        current_output = generator_chain.invoke(state)

        for attempt in range(1, max_retries + 1):
            try:
                result = executor_fn(current_output)
                return {
                    "result": result,
                    "error": None,
                    "attempts": attempt,
                    "final_code": current_output,
                }
            except Exception as e:
                error_msg = str(e)
                print(f"  [Retry {attempt}/{max_retries}] Error: {error_msg[:100]}")

                if attempt == max_retries:
                    break

                # Ask the LLM to fix its own mistake
                current_output = corrector_chain.invoke({
                    "original_request": request,
                    "failed_code": current_output,
                    "error_message": error_msg,
                    "schema": state.get("schema", ""),
                })

        return {
            "result": None,
            "error": f"Max retries ({max_retries}) exceeded",
            "attempts": max_retries,
            "final_code": current_output,
        }

    return run_with_retry
\`\`\`

### SQL Self-Healing Pipeline

\`\`\`python
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SQL_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a SQL expert. Generate ONLY the SQL query — no markdown fences.\\n"
     "The database is SQLite. Use only valid SQLite syntax."),
    ("human", "Schema:\\n{schema}\\n\\nRequest: {request}\\n\\nSQL:"),
])

SQL_CORRECTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a SQL debugging expert. Fix the SQL error.\\n"
     "Return ONLY the corrected SQL — no markdown fences."),
    ("human",
     "Schema:\\n{schema}\\n\\n"
     "Original request: {original_request}\\n\\n"
     "Failed SQL:\\n{failed_code}\\n\\n"
     "Error:\\n{error_message}\\n\\n"
     "Corrected SQL:"),
])


def build_sql_heal_chain(db_path: str, max_retries: int = 3):
    llm = get_llm(temperature=0.0)

    generator_chain = SQL_GENERATOR_PROMPT | llm | StrOutputParser()
    corrector_chain = SQL_CORRECTOR_PROMPT | llm | StrOutputParser()

    def execute_sql(sql: str) -> list[dict]:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql.strip()).fetchall()
        return [dict(row) for row in rows]

    schema = _get_schema(db_path)
    retry_chain = build_retry_chain(
        generator_chain=generator_chain,
        executor_fn=execute_sql,
        corrector_chain=corrector_chain,
        max_retries=max_retries,
    )

    def chain_with_schema(state: dict) -> dict:
        return retry_chain({**state, "schema": state.get("schema") or schema})

    return chain_with_schema
\`\`\`

### Demo Run

\`\`\`python
chain = build_sql_heal_chain(db_path="data/app.db")

result = chain({"request": "Get all users created after 2024"})

if result["error"]:
    print(f"❌ Failed after {result['attempts']} attempts: {result['error']}")
else:
    print(f"✅ Success after {result['attempts']} attempt(s):")
    for row in result["result"]:
        print(f"  {row}")
\`\`\`

### Example Self-Healing Trace

\`\`\`
[Retry 1/3] Error: no such column: created_at
  LLM corrects: users.created_at → users.created_date

[Retry 2/3] Error: near ">": syntax error
  LLM corrects: WHERE created_date > "2024" → WHERE created_date > '2024'

✅ Success after 3 attempt(s): [{'id': 2, 'name': 'Bob', 'created_date': '2025-03-20'}]
\`\`\`

### LLM Evaluation (LLM-as-Judge)

\`\`\`python
from langchain_core.prompts import ChatPromptTemplate

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a RAG evaluation expert. Score the answer on 3 dimensions.
Return JSON: {{"context_precision": 0-1, "answer_faithfulness": 0-1, "answer_relevance": 0-1, "explanation": "..."}}"""),
    ("human",
     "Question: {question}\\n\\nContext:\\n{context}\\n\\nAnswer: {answer}\\n\\nScore:"),
])

def evaluate_rag_response(question: str, context: str, answer: str):
    import json
    llm = get_llm(temperature=0.0)
    chain = EVAL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "context": context, "answer": answer})
    scores = json.loads(raw)
    overall = (
        scores["context_precision"] * 0.3
        + scores["answer_faithfulness"] * 0.4
        + scores["answer_relevance"] * 0.3
    )
    scores["overall"] = round(overall, 3)
    return scores
\`\`\`

### Production Checklist

| Item | Implementation |
|---|---|
| Self-healing SQL | \`build_sql_heal_chain\` with \`max_retries=3\` |
| Generic retry | \`build_retry_chain\` works for any code generation |
| LLM evaluation | LLM-as-judge scores context precision + faithfulness + relevance |
| LangSmith tracing | \`@traceable\` wrapper on every chain invocation |
| Docker deploy | \`uvicorn\` behind nginx with health check endpoint |`,

    zh: `## 自愈链

LLM 偶尔会生成无效代码或 SQL。自愈封装了生成 → 执行 → 纠正循环，使错误自动反馈给 LLM 进行修复。

### 模式

\`\`\`
generate(request)
    │
    ▼
execute(output) ──── 成功 ──→ 返回结果
    │
    失败
    │
    ▼
correct(output + error_message)
    │
    ▼
execute(corrected) ──── 重复最多 max_retries 次
\`\`\`

### 通用重试链

\`\`\`python
def build_retry_chain(
    generator_chain: Runnable,
    executor_fn: Callable[[str], Any],
    corrector_chain: Runnable,
    max_retries: int = 3,
):
    def run_with_retry(state: dict[str, Any]) -> dict[str, Any]:
        current_output = generator_chain.invoke(state)

        for attempt in range(1, max_retries + 1):
            try:
                result = executor_fn(current_output)
                return {"result": result, "error": None, "attempts": attempt}
            except Exception as e:
                print(f"  [重试 {attempt}/{max_retries}] 错误：{str(e)[:100]}")
                if attempt < max_retries:
                    current_output = corrector_chain.invoke({
                        "original_request": state["request"],
                        "failed_code": current_output,
                        "error_message": str(e),
                        "schema": state.get("schema", ""),
                    })

        return {"result": None, "error": f"超过最大重试次数 ({max_retries})"}

    return run_with_retry
\`\`\`

### SQL 自愈流水线

\`\`\`python
def build_sql_heal_chain(db_path: str, max_retries: int = 3):
    llm = get_llm(temperature=0.0)
    generator_chain = SQL_GENERATOR_PROMPT | llm | StrOutputParser()
    corrector_chain = SQL_CORRECTOR_PROMPT | llm | StrOutputParser()

    def execute_sql(sql: str) -> list[dict]:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql.strip()).fetchall()]

    retry_chain = build_retry_chain(generator_chain, execute_sql, corrector_chain)
    return lambda state: retry_chain({**state, "schema": _get_schema(db_path)})
\`\`\`

### 示例自愈跟踪

\`\`\`
[重试 1/3] 错误：没有这样的列：created_at
  LLM 纠正：users.created_at → users.created_date

[重试 2/3] 错误：在 ">" 附近：语法错误
  LLM 纠正：WHERE created_date > "2024" → WHERE created_date > '2024'

✅ 3 次尝试后成功：[{'id': 2, 'name': 'Bob', 'created_date': '2025-03-20'}]
\`\`\`

### 生产检查清单

| 项目 | 实现 |
|---|---|
| 自愈 SQL | \`build_sql_heal_chain\` 配合 \`max_retries=3\` |
| 通用重试 | \`build_retry_chain\` 适用于任何代码生成 |
| LLM 评估 | LLM 作为评判者评分上下文精度 + 忠实度 + 相关性 |
| LangSmith 追踪 | 每次链调用上的 \`@traceable\` 包装器 |`,
  },
}
