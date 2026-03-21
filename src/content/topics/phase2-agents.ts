import type { TopicContent } from '../types'

export const reactAgent: TopicContent = {
  id: 'react-agent',
  title: { en: 'ReAct Agent with Tools', zh: 'ReAct 代理与工具集' },
  contentType: 'code',
  content: {
    en: `## ReAct Agent with Tools

The **ReAct** (Reason + Act) pattern lets the LLM alternate between thinking and calling tools until it has enough information to answer.

### The ReAct Loop

\`\`\`
Thought:  What do I need to find out?
Action:   web_search("LangGraph features 2025")
Observation: <tool result>
Thought:  Now I know enough.
Final Answer: ...
\`\`\`

### REACT Prompt Template

\`\`\`python
REACT_TEMPLATE = """You are a helpful research assistant.

You have access to the following tools:
{tools}

Use the following format:
Thought: Consider what information you need and which tool would help.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation history:
{chat_history}

Question: {input}
{agent_scratchpad}"""
\`\`\`

### Building the AgentExecutor

\`\`\`python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from phase2_agency.tools.tool_registry import get_all_tools
from shared.llm_factory import get_llm

def build_react_agent_executor(session_id: str = "default") -> AgentExecutor:
    tools = get_all_tools()
    llm = get_llm(temperature=0.0)

    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    )

    # Create the ReAct agent (prompt → LLM → action parser loop)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,              # show Thought/Action/Observation traces
        max_iterations=6,          # prevent infinite loops
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
\`\`\`

### Running the Agent

\`\`\`python
executor = build_react_agent_executor(session_id="alice")

result = executor.invoke({
    "input": "What documents are in the knowledge base? Search for 'LangChain'.",
    "chat_history": "No previous conversation.",
})

print(result["output"])
# result["intermediate_steps"] shows all Thought/Action/Observation traces
\`\`\`

### MCP Dynamic Tool Loading

\`\`\`python
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def load_mcp_tools_and_run(query: str) -> None:
    # Connect to a running MCP server via stdio transport
    server_params = StdioServerParameters(
        command="python",
        args=["phase2_agency/mcp_servers/sqlite_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Tools are discovered at runtime from the server
            mcp_tools = await load_mcp_tools(session)
            print(f"Loaded {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")

            agent = create_react_agent(llm=get_llm(), tools=mcp_tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=mcp_tools, max_iterations=5)
            result = await executor.ainvoke({"input": query, "chat_history": ""})
            print(f"Answer: {result['output']}")

asyncio.run(load_mcp_tools_and_run("List all rows in the users table"))
\`\`\`

### Key Concepts

| Concept | Description |
|---|---|
| \`create_react_agent\` | Builds the ReAct agent from LLM + tools + prompt |
| \`AgentExecutor\` | Runs the Thought→Action→Observation loop |
| \`max_iterations\` | Guards against infinite tool-calling loops |
| \`handle_parsing_errors\` | Gracefully recovers when LLM outputs malformed actions |
| \`return_intermediate_steps\` | Exposes full chain-of-thought for debugging |
| MCP tools | Dynamically loaded from external server via stdio transport |`,

    zh: `## ReAct 代理与工具集

**ReAct**（推理 + 行动）模式让 LLM 在思考和调用工具之间交替，直到有足够信息回答问题。

### ReAct 循环

\`\`\`
Thought:  我需要找什么？
Action:   web_search("LangGraph features 2025")
Observation: <工具结果>
Thought:  现在我知道足够了。
Final Answer: ...
\`\`\`

### ReAct 提示模板

\`\`\`python
REACT_TEMPLATE = """你是一个有帮助的研究助理。

你可以使用以下工具：
{tools}

使用以下格式：
Thought: 考虑需要什么信息以及哪个工具有帮助。
Action: 要采取的行动，应该是 [{tool_names}] 之一
Action Input: 行动的输入
Observation: 行动的结果
... (Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

对话历史：
{chat_history}

问题：{input}
{agent_scratchpad}"""
\`\`\`

### 构建 AgentExecutor

\`\`\`python
from langchain.agents import AgentExecutor, create_react_agent

def build_react_agent_executor(session_id: str = "default") -> AgentExecutor:
    tools = get_all_tools()
    llm = get_llm(temperature=0.0)

    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,              # 显示推理轨迹
        max_iterations=6,          # 防止无限循环
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
\`\`\`

### 运行代理

\`\`\`python
executor = build_react_agent_executor(session_id="alice")

result = executor.invoke({
    "input": "知识库中有什么文档？搜索 'LangChain'。",
    "chat_history": "没有之前的对话。",
})

print(result["output"])
\`\`\`

### MCP 动态工具加载

\`\`\`python
async def load_mcp_tools_and_run(query: str) -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["phase2_agency/mcp_servers/sqlite_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            print(f"加载了 {len(mcp_tools)} 个工具")

            executor = AgentExecutor(agent=agent, tools=mcp_tools)
            result = await executor.ainvoke({"input": query, "chat_history": ""})
            print(f"答案：{result['output']}")
\`\`\`

### 关键概念

| 概念 | 描述 |
|---|---|
| \`create_react_agent\` | 从 LLM + 工具 + 提示构建 ReAct 代理 |
| \`AgentExecutor\` | 运行思考→行动→观察循环 |
| \`max_iterations\` | 防止无限工具调用循环 |
| \`handle_parsing_errors\` | 当 LLM 输出格式错误时优雅恢复 |
| MCP 工具 | 通过 stdio 传输从外部服务器动态加载 |`,
  },
}

export const persistentMemory: TopicContent = {
  id: 'persistent-memory',
  title: { en: 'Persistent Conversation Memory', zh: '持久对话记忆' },
  contentType: 'code',
  content: {
    en: `## Persistent Conversation Memory

\`RunnableWithMessageHistory\` wraps any chain and automatically injects past conversation turns. History is persisted in SQLite and survives process restarts.

### Building the Memory Chain

\`\`\`python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

def build_conversation_chain() -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory of past conversations."),
        MessagesPlaceholder(variable_name="chat_history"),  # injected automatically
        ("human", "{question}"),
    ])

    llm = get_llm(temperature=0.0)
    base_chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,          # SQLite-backed history factory
        input_messages_key="question",
        history_messages_key="chat_history",
    )
\`\`\`

### SQLite Session History

\`\`\`python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """Load (or create) message history for a given session from SQLite."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///data/chat_history.db",
    )
\`\`\`

### Multi-Turn Conversation Demo

\`\`\`python
chain = build_conversation_chain()
session_config = {"configurable": {"session_id": "demo-session"}}

turns = [
    "My name is Alice and I'm learning LangChain.",
    "What framework am I learning?",   # should answer: LangChain
    "What is my name?",                # should answer: Alice
    "Give me a one-line summary of our conversation.",
]

for question in turns:
    print(f"User: {question}")
    answer = chain.invoke({"question": question}, config=session_config)
    print(f"Bot:  {answer}\\n")
\`\`\`

### Expected Output

\`\`\`
User: My name is Alice and I'm learning LangChain.
Bot:  Nice to meet you, Alice! LangChain is a great framework for building LLM apps.

User: What framework am I learning?
Bot:  You mentioned you're learning LangChain.

User: What is my name?
Bot:  Your name is Alice.
\`\`\`

### Memory Architecture

\`\`\`
invoke(question, config={session_id: "abc"})
    │
    ▼
RunnableWithMessageHistory
    │  load history from SQLite by session_id
    ▼
ChatPromptTemplate
  system: "You are helpful..."
  history: [HumanMessage("Alice"), AIMessage("Nice...")]  ← injected
  human: "What is my name?"
    │
    ▼
LLM → "Your name is Alice."
    │
    ▼
Append HumanMessage + AIMessage back to SQLite
\`\`\`

### Key Points

- **Session isolation**: each \`session_id\` gets its own history table row
- **Persistence**: messages survive server restarts (SQLite file)
- **Automatic injection**: \`MessagesPlaceholder\` populates from history
- **Truncation**: in production, limit history size to avoid token overflow`,

    zh: `## 持久对话记忆

\`RunnableWithMessageHistory\` 包装任何链并自动注入过去的对话轮次。历史记录持久化在 SQLite 中，进程重启后仍然存在。

### 构建记忆链

\`\`\`python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

def build_conversation_chain() -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有记忆的有帮助的助手。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 自动注入
        ("human", "{question}"),
    ])

    base_chain = prompt | get_llm(temperature=0.0) | StrOutputParser()

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,          # SQLite 历史工厂
        input_messages_key="question",
        history_messages_key="chat_history",
    )
\`\`\`

### SQLite 会话历史

\`\`\`python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///data/chat_history.db",
    )
\`\`\`

### 多轮对话演示

\`\`\`python
chain = build_conversation_chain()
session_config = {"configurable": {"session_id": "demo-session"}}

turns = [
    "我叫 Alice，正在学习 LangChain。",
    "我在学什么框架？",   # 应该回答：LangChain
    "我叫什么名字？",     # 应该回答：Alice
]

for question in turns:
    print(f"用户：{question}")
    answer = chain.invoke({"question": question}, config=session_config)
    print(f"机器人：{answer}\\n")
\`\`\`

### 记忆架构

\`\`\`
invoke(question, config={session_id: "abc"})
    │
    ▼
RunnableWithMessageHistory
    │  从 SQLite 按 session_id 加载历史
    ▼
ChatPromptTemplate
  system: "你是有帮助的..."
  history: [HumanMessage("Alice"), AIMessage("你好...")]  ← 注入
  human: "我叫什么名字？"
    │
    ▼
LLM → "你叫 Alice。"
    │
    ▼
将 HumanMessage + AIMessage 追加回 SQLite
\`\`\`

### 关键点

- **会话隔离**：每个 \`session_id\` 有自己的历史记录行
- **持久性**：消息在服务器重启后仍然存在（SQLite 文件）
- **自动注入**：\`MessagesPlaceholder\` 从历史记录中填充
- **截断**：在生产中，限制历史记录大小以避免 token 溢出`,
  },
}

export const toolRegistry: TopicContent = {
  id: 'tool-registry',
  title: { en: 'Tool Registry Pattern', zh: '工具注册表模式' },
  contentType: 'code',
  content: {
    en: `## Tool Registry Pattern

A central tool registry is the **single source of truth** for tool configuration. Agent files import \`get_all_tools()\` — never instantiate tools directly.

### The Registry

\`\`\`python
# phase2_agency/tools/tool_registry.py
from langchain_core.tools import Tool
from phase2_agency.tools.private_rag_tool import get_private_rag_tool
from phase2_agency.tools.web_search import get_web_search_tool
from shared.config import settings


def get_all_tools() -> list[Tool]:
    """Return all available tools, skipping those with missing credentials."""
    tools: list[Tool] = []

    # Always include private RAG retrieval
    tools.append(get_private_rag_tool(k=4))

    # Web search — only if Tavily API key is configured
    if settings.tavily_api_key:
        tools.append(get_web_search_tool(max_results=5))
    else:
        print("⚠️  TAVILY_API_KEY not set — web_search tool disabled")

    return tools


def get_tools_by_name(names: list[str]) -> list[Tool]:
    """Return only the tools matching the given names."""
    all_tools = {t.name: t for t in get_all_tools()}
    result = []
    for name in names:
        if name not in all_tools:
            raise ValueError(
                f"Tool {name!r} not found. Available: {list(all_tools.keys())}"
            )
        result.append(all_tools[name])
    return result
\`\`\`

### Private RAG Tool

\`\`\`python
# phase2_agency/tools/private_rag_tool.py
from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_private_rag_tool(k: int = 4) -> Tool:
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="data/chroma_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def search_knowledge_base(query: str) -> str:
        docs = retriever.invoke(query)
        return "\\n\\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\\n{d.page_content}"
            for d in docs
        )

    return Tool(
        name="private_rag_search",
        description=(
            "Search the private knowledge base for relevant documents. "
            "Use for questions about internal documentation."
        ),
        func=search_knowledge_base,
    )
\`\`\`

### Web Search Tool

\`\`\`python
# phase2_agency/tools/web_search.py
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool

def get_web_search_tool(max_results: int = 5) -> Tool:
    search = TavilySearchResults(max_results=max_results)

    def web_search(query: str) -> str:
        results = search.invoke(query)
        return "\\n\\n".join(
            f"[{r['url']}]\\n{r['content']}" for r in results
        )

    return Tool(
        name="web_search",
        description=(
            "Search the internet for current information. "
            "Use for recent news, live data, or topics not in the knowledge base."
        ),
        func=web_search,
    )
\`\`\`

### Usage in Agents

\`\`\`python
# In any agent file
from phase2_agency.tools.tool_registry import get_all_tools, get_tools_by_name

# Get all available tools
tools = get_all_tools()

# Get specific tools
rag_only = get_tools_by_name(["private_rag_search"])
\`\`\`

### Benefits

| Pattern | Benefit |
|---|---|
| Single registry | One place to add/remove/configure tools |
| Graceful degradation | Tools requiring missing API keys are skipped |
| \`get_tools_by_name\` | Agents can request only the tools they need |
| Separation of concerns | Tool logic isolated from agent orchestration |`,

    zh: `## 工具注册表模式

中央工具注册表是工具配置的**单一真相来源**。代理文件导入 \`get_all_tools()\`，而不是直接实例化工具。

### 注册表

\`\`\`python
# phase2_agency/tools/tool_registry.py
from langchain_core.tools import Tool

def get_all_tools() -> list[Tool]:
    """返回所有可用工具，跳过缺少凭据的工具。"""
    tools: list[Tool] = []

    # 始终包含私有 RAG 检索
    tools.append(get_private_rag_tool(k=4))

    # 网络搜索 — 仅当配置了 Tavily API 密钥时
    if settings.tavily_api_key:
        tools.append(get_web_search_tool(max_results=5))
    else:
        print("⚠️  未设置 TAVILY_API_KEY — 已禁用 web_search 工具")

    return tools


def get_tools_by_name(names: list[str]) -> list[Tool]:
    """返回匹配给定名称的工具。"""
    all_tools = {t.name: t for t in get_all_tools()}
    for name in names:
        if name not in all_tools:
            raise ValueError(f"未找到工具 {name!r}")
    return [all_tools[name] for name in names]
\`\`\`

### 私有 RAG 工具

\`\`\`python
def get_private_rag_tool(k: int = 4) -> Tool:
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="data/chroma_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def search_knowledge_base(query: str) -> str:
        docs = retriever.invoke(query)
        return "\\n\\n".join(
            f"[来源：{d.metadata.get('source', 'unknown')}]\\n{d.page_content}"
            for d in docs
        )

    return Tool(
        name="private_rag_search",
        description="搜索私有知识库中的相关文档。",
        func=search_knowledge_base,
    )
\`\`\`

### 在代理中使用

\`\`\`python
from phase2_agency.tools.tool_registry import get_all_tools, get_tools_by_name

# 获取所有可用工具
tools = get_all_tools()

# 获取特定工具
rag_only = get_tools_by_name(["private_rag_search"])
\`\`\`

### 优势

| 模式 | 优势 |
|---|---|
| 单一注册表 | 一个地方添加/删除/配置工具 |
| 优雅降级 | 跳过需要缺失 API 密钥的工具 |
| \`get_tools_by_name\` | 代理可以只请求它们需要的工具 |
| 关注点分离 | 工具逻辑与代理编排隔离 |`,
  },
}
