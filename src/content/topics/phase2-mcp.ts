import type { TopicContent } from '../types'

export const agentExecutor: TopicContent = {
  id: 'agent-executor',
  emoji: '⚙️',
  title: { en: 'Production Agent Executor', zh: '生产级代理执行器' },
  contentType: 'code',
  content: {
    en: `## Production Agent Executor

The \`agent_executor.py\` module wraps the ReAct agent with automatic memory injection, graceful error handling, and history persistence — the pattern you use in a real application.

### The \`run_agent\` Function

\`\`\`python
# phase2_agency/agent/agent_executor.py
from phase2_agency.agent.react_agent import build_react_agent_executor
from phase2_agency.memory.chat_history import get_session_history


def run_agent(
    question: str,
    session_id: str = "default",
    verbose: bool = False,
) -> dict:
    """Run the ReAct agent with automatic memory injection.

    Args:
        question:   The user's question.
        session_id: Session ID for chat history lookup.
        verbose:    If True, print intermediate agent steps.

    Returns:
        Dict with keys: "answer", "steps", "session_id"
    """
    executor = build_react_agent_executor(session_id=session_id)

    # Load recent history for context injection
    history = get_session_history(session_id)
    history_text = "\\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {m.content}"
        for m in history.messages[-8:]   # last 8 messages = 4 turns
    ) or "No previous conversation."

    try:
        result = executor.invoke({
            "input": question,
            "chat_history": history_text,
        })

        answer = result.get("output", "No answer generated.")
        steps = result.get("intermediate_steps", [])

        # Persist this turn to history
        history.add_user_message(question)
        history.add_ai_message(answer)

        if verbose:
            print(f"\\n[{len(steps)} steps taken]")
            for i, (action, observation) in enumerate(steps, 1):
                print(f"  Step {i}: {action.tool}({action.tool_input!r})")
                print(f"         → {str(observation)[:200]}")

        return {"answer": answer, "steps": steps, "session_id": session_id}

    except Exception as e:
        error_msg = f"Agent error: {e}"
        # Still persist the failed turn so context is maintained
        history.add_user_message(question)
        history.add_ai_message(error_msg)
        return {"answer": error_msg, "steps": [], "session_id": session_id}
\`\`\`

### Usage

\`\`\`python
from phase2_agency.agent.agent_executor import run_agent

# Simple call
result = run_agent("What is the latest news about LangGraph?", session_id="alice")
print(result["answer"])

# Verbose mode shows all tool calls
result = run_agent(
    "Search the knowledge base for LCEL documentation.",
    session_id="alice",
    verbose=True,
)
# Output:
# [2 steps taken]
#   Step 1: private_rag_search('LCEL documentation')
#          → [Document 1 | data/langchain_docs.pdf, page 3]...
#   Step 2: private_rag_search('LCEL pipe operator examples')
#          → [Document 2 | ...
\`\`\`

### Design Patterns

| Pattern | Purpose |
|---|---|
| History window (last 8 messages) | Prevent token overflow from long sessions |
| Persist on error | Maintain conversation continuity even when agent fails |
| \`intermediate_steps\` | Expose tool calls for debugging and audit trails |
| Single \`run_agent\` entry point | Agent files never need to touch raw executors |`,

    zh: `## 生产级代理执行器

\`agent_executor.py\` 模块用自动记忆注入、优雅错误处理和历史持久化封装 ReAct 代理 — 这是在真实应用程序中使用的模式。

### \`run_agent\` 函数

\`\`\`python
def run_agent(
    question: str,
    session_id: str = "default",
    verbose: bool = False,
) -> dict:
    executor = build_react_agent_executor(session_id=session_id)

    # 为上下文注入加载最近历史
    history = get_session_history(session_id)
    history_text = "\\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {m.content}"
        for m in history.messages[-8:]
    ) or "没有之前的对话。"

    try:
        result = executor.invoke({
            "input": question,
            "chat_history": history_text,
        })
        answer = result.get("output", "未生成答案。")
        steps = result.get("intermediate_steps", [])

        # 将此轮次持久化到历史记录
        history.add_user_message(question)
        history.add_ai_message(answer)

        return {"answer": answer, "steps": steps, "session_id": session_id}

    except Exception as e:
        error_msg = f"代理错误：{e}"
        history.add_user_message(question)
        history.add_ai_message(error_msg)
        return {"answer": error_msg, "steps": [], "session_id": session_id}
\`\`\`

### 设计模式

| 模式 | 目的 |
|---|---|
| 历史窗口（最后 8 条消息） | 防止长会话的 token 溢出 |
| 错误时持久化 | 即使代理失败也保持对话连续性 |
| \`intermediate_steps\` | 暴露工具调用用于调试和审计 |`,
  },
}

export const mcpFilesystem: TopicContent = {
  id: 'mcp-filesystem',
  emoji: '📁',
  title: { en: 'MCP Filesystem Server', zh: 'MCP 文件系统服务器' },
  contentType: 'code',
  content: {
    en: `## MCP Filesystem Server

The MCP (Model Context Protocol) filesystem server exposes your local \`data/\` directory as tools that agents can call. It uses \`FastMCP\` to define tools with simple \`@mcp.tool()\` decorators.

### Server Setup

\`\`\`python
# phase2_agency/mcp_servers/filesystem_server.py
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from shared.config import settings

# All operations are restricted to this path for security
ALLOWED_ROOT = Path(settings.mcp_filesystem_root).resolve()

mcp = FastMCP("filesystem-server")


def _safe_path(relative_path: str) -> Path:
    """Validate path stays inside ALLOWED_ROOT (prevents directory traversal)."""
    resolved = (ALLOWED_ROOT / relative_path).resolve()
    if not str(resolved).startswith(str(ALLOWED_ROOT)):
        raise PermissionError(f"Path escape attempt blocked: {relative_path!r}")
    return resolved
\`\`\`

### Tool Definitions

\`\`\`python
@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List files and directories inside the allowed root."""
    target = _safe_path(directory)
    if not target.exists():
        return f"Directory not found: {directory}"

    entries = sorted(target.iterdir())
    lines = []
    for entry in entries:
        prefix = "[DIR] " if entry.is_dir() else "[FILE]"
        lines.append(f"{prefix} {entry.name}")
    return "\\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    target = _safe_path(filepath)
    if not target.exists():
        return f"File not found: {filepath}"
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[Binary file — cannot display as text: {filepath}]"


@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """Write content to a file (creates or overwrites)."""
    target = _safe_path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"✅ Written {len(content)} characters to {filepath}"


if __name__ == "__main__":
    print(f"Starting Filesystem MCP Server (root: {ALLOWED_ROOT})")
    mcp.run()
\`\`\`

### Connecting from LangChain Agent

\`\`\`python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def run_with_filesystem_tools(query: str):
    server_params = StdioServerParameters(
        command="python",
        args=["phase2_agency/mcp_servers/filesystem_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # tools = [list_files, read_file, write_file]

            executor = build_react_agent_executor_with_tools(tools)
            result = await executor.ainvoke({
                "input": query,
                "chat_history": "",
            })
            print(result["output"])

asyncio.run(run_with_filesystem_tools("List all files in the data directory"))
\`\`\`

### Security: Path Validation

\`\`\`python
# ✅ Safe: stays inside ALLOWED_ROOT
_safe_path("docs/report.txt")     # → /data/docs/report.txt ✅

# ❌ Blocked: directory traversal attack
_safe_path("../../etc/passwd")    # → PermissionError ❌
_safe_path("/abs/path")           # → PermissionError ❌
\`\`\`

### Key Points

- **\`FastMCP\`**: Minimal boilerplate — just decorate functions with \`@mcp.tool()\`
- **Stdio transport**: Agent spawns the server as a subprocess, communicates over stdin/stdout
- **Path validation**: Always resolve + verify paths to prevent directory traversal
- **\`settings.mcp_filesystem_root\`**: Configured in \`.env\` (default: \`./data\`)`,

    zh: `## MCP 文件系统服务器

MCP（模型上下文协议）文件系统服务器将本地 \`data/\` 目录作为代理可以调用的工具公开。它使用 \`FastMCP\` 通过简单的 \`@mcp.tool()\` 装饰器定义工具。

### 服务器设置

\`\`\`python
ALLOWED_ROOT = Path(settings.mcp_filesystem_root).resolve()
mcp = FastMCP("filesystem-server")

def _safe_path(relative_path: str) -> Path:
    """验证路径保持在 ALLOWED_ROOT 内（防止目录遍历）。"""
    resolved = (ALLOWED_ROOT / relative_path).resolve()
    if not str(resolved).startswith(str(ALLOWED_ROOT)):
        raise PermissionError(f"路径逃逸尝试被阻止：{relative_path!r}")
    return resolved
\`\`\`

### 工具定义

\`\`\`python
@mcp.tool()
def list_files(directory: str = ".") -> str:
    """列出允许根目录中的文件和目录。"""
    target = _safe_path(directory)
    entries = sorted(target.iterdir())
    lines = [f"{'[目录]' if e.is_dir() else '[文件]'} {e.name}" for e in entries]
    return "\\n".join(lines)

@mcp.tool()
def read_file(filepath: str) -> str:
    """读取文件内容。"""
    return _safe_path(filepath).read_text(encoding="utf-8")

@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """写入内容到文件（创建或覆盖）。"""
    target = _safe_path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"✅ 已写入 {len(content)} 个字符到 {filepath}"
\`\`\`

### 关键点

- **\`FastMCP\`**：最小样板 — 只需用 \`@mcp.tool()\` 装饰函数
- **Stdio 传输**：代理将服务器作为子进程启动，通过 stdin/stdout 通信
- **路径验证**：始终解析 + 验证路径以防止目录遍历
- **\`settings.mcp_filesystem_root\`**：在 \`.env\` 中配置（默认：\`./data\`）`,
  },
}

export const mcpSqlite: TopicContent = {
  id: 'mcp-sqlite',
  emoji: '🗄️',
  title: { en: 'MCP SQLite Server', zh: 'MCP SQLite 服务器' },
  contentType: 'code',
  content: {
    en: `## MCP SQLite Server

The SQLite MCP server exposes a database as four tools: read-only queries, schema inspection, and write operations. Agents can query structured data without writing SQL directly.

### Server Definition

\`\`\`python
# phase2_agency/mcp_servers/sqlite_server.py
import json
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from shared.config import settings

DB_PATH = Path(settings.mcp_sqlite_path)
mcp = FastMCP("sqlite-server")


def _get_connection() -> sqlite3.Connection:
    """Open SQLite with row factory for dict-like row access."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn
\`\`\`

### Schema Tools

\`\`\`python
@mcp.tool()
def list_tables() -> str:
    """List all user-defined tables in the database."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    tables = [row["name"] for row in rows]
    return json.dumps(tables, indent=2)
    # → ["orders", "products", "users"]


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get column schema for a specific table."""
    with _get_connection() as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name!r})").fetchall()

    columns = [
        {
            "name":     row["name"],
            "type":     row["type"],
            "nullable": not row["notnull"],
            "default":  row["dflt_value"],
            "pk":       bool(row["pk"]),
        }
        for row in rows
    ]
    return json.dumps(columns, indent=2)
\`\`\`

### Query Tools

\`\`\`python
@mcp.tool()
def query_db(sql: str) -> str:
    """Execute a SELECT query and return results as JSON.

    Only SELECT statements are allowed — use execute_sql for writes.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "❌ Only SELECT statements allowed via query_db."

    try:
        with _get_connection() as conn:
            rows = conn.execute(sql).fetchall()
        return json.dumps([dict(row) for row in rows], indent=2, default=str)
    except sqlite3.Error as e:
        return f"❌ SQL Error: {e}"


@mcp.tool()
def execute_sql(sql: str) -> str:
    """Execute a write SQL statement (INSERT, UPDATE, DELETE, CREATE)."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute(sql)
            conn.commit()
        return f"✅ Executed successfully. Rows affected: {cursor.rowcount}"
    except sqlite3.Error as e:
        return f"❌ SQL Error: {e}"
\`\`\`

### Example Agent Interaction

\`\`\`
User: "How many users were created in 2025?"

Agent Thought: I need to query the database. Let me check what tables exist.
Agent Action: list_tables()
Observation: ["orders", "users"]

Agent Thought: I'll describe the users table to understand the schema.
Agent Action: describe_table("users")
Observation: [{"name": "id", "type": "INTEGER", "pk": true},
              {"name": "name", "type": "TEXT"},
              {"name": "created_at", "type": "TEXT"}]

Agent Thought: Now I can write the query.
Agent Action: query_db("SELECT COUNT(*) as total FROM users WHERE created_at LIKE '2025%'")
Observation: [{"total": 42}]

Final Answer: 42 users were created in 2025.
\`\`\`

### Security Notes

| Concern | Mitigation |
|---|---|
| SQL injection | Tool accepts any SQL — use only with trusted agents |
| Write access | Separate \`query_db\` (read) from \`execute_sql\` (write) |
| DB path | Configured via \`settings.mcp_sqlite_path\` in \`.env\` |`,

    zh: `## MCP SQLite 服务器

SQLite MCP 服务器将数据库公开为四个工具：只读查询、模式检查和写操作。代理可以查询结构化数据而无需直接编写 SQL。

### 模式工具

\`\`\`python
@mcp.tool()
def list_tables() -> str:
    """列出数据库中所有用户定义的表。"""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return json.dumps([row["name"] for row in rows], indent=2)


@mcp.tool()
def describe_table(table_name: str) -> str:
    """获取特定表的列模式。"""
    with _get_connection() as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name!r})").fetchall()
    return json.dumps([
        {"name": r["name"], "type": r["type"], "pk": bool(r["pk"])}
        for r in rows
    ], indent=2)
\`\`\`

### 查询工具

\`\`\`python
@mcp.tool()
def query_db(sql: str) -> str:
    """执行 SELECT 查询并以 JSON 形式返回结果。"""
    if not sql.strip().upper().startswith("SELECT"):
        return "❌ 通过 query_db 只允许 SELECT 语句。"
    with _get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return json.dumps([dict(row) for row in rows], indent=2, default=str)

@mcp.tool()
def execute_sql(sql: str) -> str:
    """执行写 SQL 语句（INSERT、UPDATE、DELETE、CREATE）。"""
    with _get_connection() as conn:
        cursor = conn.execute(sql)
        conn.commit()
    return f"✅ 执行成功。受影响的行数：{cursor.rowcount}"
\`\`\`

### 安全说明

| 关注点 | 缓解 |
|---|---|
| SQL 注入 | 工具接受任何 SQL — 仅与受信任的代理一起使用 |
| 写访问 | 将 \`query_db\`（读）与 \`execute_sql\`（写）分开 |
| 数据库路径 | 通过 \`.env\` 中的 \`settings.mcp_sqlite_path\` 配置 |`,
  },
}
