import type { TopicContent } from '../types'

export const researcherNode: TopicContent = {
  id: 'researcher-node',
  emoji: '🔬',
  title: { en: 'Researcher Node Implementation', zh: '研究员节点实现' },
  contentType: 'code',
  content: {
    en: `## Researcher Node Implementation

The researcher is a LangGraph node that gathers information by querying the RAG knowledge base and synthesising structured research notes.

### The Researcher Prompt

\`\`\`python
# phase3_enterprise/multi_agent/researcher_agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher. Your task is to gather comprehensive, accurate information.\\n"
     "Use the provided context from the knowledge base. Be thorough and cite sources.\\n"
     "Format your output as structured research notes."),
    ("human",
     "Task: {task}\\n\\n"
     "Retrieved context:\\n{context}\\n\\n"
     "Provide detailed research notes:"),
])
\`\`\`

### Context Retrieval Helper

\`\`\`python
def _retrieve_context(task: str, k: int = 6) -> str:
    """Retrieve relevant documents from the vector store.

    Falls back gracefully if the knowledge base is unavailable.
    """
    try:
        retriever = get_retriever(backend="chroma", k=k)
        docs = retriever.invoke(task)
        if not docs:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\\n{doc.page_content}")
        return "\\n\\n".join(parts)

    except Exception as e:
        return f"Knowledge base unavailable: {e}"
\`\`\`

### The Node Function

\`\`\`python
def researcher_node(state: AgentState) -> AgentState:
    """LangGraph node: gather information for the given task.

    Returns updated state with research_output populated.
    """
    task = state["task"]
    llm = get_llm(temperature=0.2)   # slight creativity for synthesis

    # Step 1: retrieve context from vector store
    context = _retrieve_context(task)

    # Step 2: synthesise into structured research notes
    chain = RESEARCHER_PROMPT | llm
    result = chain.invoke({"task": task, "context": context})
    research_output = result.content

    # Step 3: update state with research + append message
    return {
        **state,
        "research_output": research_output,
        "messages": state["messages"] + [
            AIMessage(content=f"[Researcher] {research_output[:200]}...")
        ],
    }
\`\`\`

### State Flow

\`\`\`
supervisor sets next_step = "researcher"
    │
    ▼
researcher_node(state)
    │  retrieve context (k=6 docs)
    │  synthesise research notes (LLM)
    ▼
state["research_output"] = "## Research Notes\\n1. LCEL uses the | operator..."
state["messages"]        += [AIMessage("[Researcher] Research notes generated")]
    │
    ▼
supervisor runs again (sees research_output → routes to writer)
\`\`\``,

    zh: `## 研究员节点实现

研究员是一个 LangGraph 节点，通过查询 RAG 知识库并综合结构化研究笔记来收集信息。

### 研究员提示

\`\`\`python
RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一名专业研究员。你的任务是收集全面、准确的信息。\\n"
     "使用知识库提供的上下文。要全面并引用来源。\\n"
     "将输出格式化为结构化研究笔记。"),
    ("human",
     "任务：{task}\\n\\n"
     "检索到的上下文：\\n{context}\\n\\n"
     "提供详细的研究笔记："),
])
\`\`\`

### 节点函数

\`\`\`python
def researcher_node(state: AgentState) -> AgentState:
    """LangGraph 节点：收集给定任务的信息。"""
    task = state["task"]
    context = _retrieve_context(task)   # 从向量存储检索

    chain = RESEARCHER_PROMPT | get_llm(temperature=0.2)
    result = chain.invoke({"task": task, "context": context})

    return {
        **state,
        "research_output": result.content,
        "messages": state["messages"] + [
            AIMessage(content=f"[研究员] {result.content[:200]}...")
        ],
    }
\`\`\``,
  },
}

export const supervisorNode: TopicContent = {
  id: 'supervisor-node',
  emoji: '🎯',
  title: { en: 'Supervisor Node: Task Routing', zh: '监督节点：任务路由' },
  contentType: 'code',
  content: {
    en: `## Supervisor Node: Task Routing

The supervisor is the orchestrator — it reads the current state and decides which agent should act next. It uses rule-based logic first for speed, then optionally confirms with an LLM for ambiguous cases.

### Routing Logic

\`\`\`python
# phase3_enterprise/multi_agent/supervisor.py
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supervisor orchestrating a research team.\\n"
     "Decide the next action based on the current state.\\n"
     "Respond with EXACTLY ONE of: 'researcher', 'writer', or 'end'\\n"
     "Rules:\\n"
     "  - If research_output is empty → respond 'researcher'\\n"
     "  - If research_output exists but final_report is empty → respond 'writer'\\n"
     "  - If final_report exists and is complete → respond 'end'\\n"
     "Do not include any other text."),
    ("human",
     "Task: {task}\\n"
     "Research done: {has_research}\\n"
     "Report done: {has_report}\\n\\n"
     "Next action:"),
])


def supervisor_node(state: AgentState) -> AgentState:
    """LangGraph node: determine which agent should run next.

    Uses rule-based routing for clear cases; LLM confirmation for ambiguous ones.
    """
    has_research = bool(state.get("research_output", "").strip())
    has_report = bool(state.get("final_report", "").strip())

    # ── Rule-based routing (fast, no LLM call for clear cases) ──────────────
    if not has_research:
        next_step = "researcher"
    elif has_research and not has_report:
        next_step = "writer"
    else:
        next_step = "end"

    # ── LLM confirmation for edge cases (only when report exists) ────────────
    if has_research and has_report:
        llm = get_llm(temperature=0.0)
        chain = SUPERVISOR_PROMPT | llm | StrOutputParser()
        llm_decision = chain.invoke({
            "task": state["task"],
            "has_research": str(has_research),
            "has_report": str(has_report),
        }).strip().lower()

        # Only accept known values; ignore malformed LLM output
        if llm_decision in {"researcher", "writer", "end"}:
            next_step = llm_decision

    return {
        **state,
        "next_step": next_step,
        "messages": state["messages"] + [
            AIMessage(content=f"[Supervisor] Routing to: {next_step}")
        ],
    }
\`\`\`

### Decision Tree

\`\`\`
START
  │
  ▼
Supervisor
  │
  ├── research_output empty? ──────── YES → "researcher"
  │
  ├── research_output exists,
  │   final_report empty? ─────────── YES → "writer"
  │
  └── both exist? ──────────────────── YES → LLM confirm → "end"
\`\`\`

### Why Hybrid Routing?

| Approach | Pros | Cons |
|---|---|---|
| Pure rules | Fast, predictable, no LLM cost | Rigid — can't handle edge cases |
| Pure LLM | Flexible, handles nuance | Slow, expensive, can hallucinate |
| **Hybrid** | Fast for common cases + smart for edge cases | Slightly more complex |`,

    zh: `## 监督节点：任务路由

监督节点是编排器 — 它读取当前状态并决定下一个应该行动的代理。它首先使用基于规则的逻辑以提高速度，然后可选地使用 LLM 确认模糊情况。

### 路由逻辑

\`\`\`python
def supervisor_node(state: AgentState) -> AgentState:
    has_research = bool(state.get("research_output", "").strip())
    has_report = bool(state.get("final_report", "").strip())

    # 基于规则的路由（快速，明确情况无需 LLM 调用）
    if not has_research:
        next_step = "researcher"
    elif has_research and not has_report:
        next_step = "writer"
    else:
        next_step = "end"

    # 边缘情况的 LLM 确认（仅当报告存在时）
    if has_research and has_report:
        chain = SUPERVISOR_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
        llm_decision = chain.invoke({...}).strip().lower()
        if llm_decision in {"researcher", "writer", "end"}:
            next_step = llm_decision

    return {**state, "next_step": next_step}
\`\`\`

### 为什么使用混合路由？

| 方法 | 优点 | 缺点 |
|---|---|---|
| 纯规则 | 快速、可预测、无 LLM 成本 | 僵硬 — 无法处理边缘情况 |
| 纯 LLM | 灵活，处理细微差别 | 慢、昂贵、可能产生幻觉 |
| **混合** | 常见情况快速 + 边缘情况智能 | 稍微复杂 |`,
  },
}

export const writerNode: TopicContent = {
  id: 'writer-node',
  emoji: '✍️',
  title: { en: 'Writer Node: Report Synthesis', zh: '写作节点：报告综合' },
  contentType: 'code',
  content: {
    en: `## Writer Node: Report Synthesis

The writer node takes the researcher's raw notes and transforms them into a polished, structured markdown report. It's the final stage before the graph exits.

### Implementation

\`\`\`python
# phase3_enterprise/multi_agent/writer_agent.py
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional technical writer.\\n"
     "Transform research notes into a clear, well-structured report.\\n"
     "Use markdown formatting: headers, bullet points, code blocks where appropriate.\\n"
     "Be concise but comprehensive. Cite sources when available."),
    ("human",
     "Task: {task}\\n\\n"
     "Research notes:\\n{research}\\n\\n"
     "Write the final report:"),
])


def writer_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesise research into a polished report."""
    llm = get_llm(temperature=0.3)   # slight creativity for better writing
    chain = WRITER_PROMPT | llm | StrOutputParser()

    report = chain.invoke({
        "task": state["task"],
        "research": state.get("research_output", ""),
    })

    return {
        **state,
        "final_report": report,
        "messages": state["messages"] + [
            AIMessage(content=f"[Writer] Report generated ({len(report)} chars)")
        ],
    }
\`\`\`

### Example Output

\`\`\`markdown
# LangGraph: State Management Architecture

## Overview
LangGraph improves upon standard agent loops by modeling execution as a
directed state machine rather than a simple while-loop.

## Key Features

### 1. Typed State (TypedDict)
All agents share a single, strongly-typed state object...

### 2. Checkpointing
LangGraph supports saving state between steps using...

### 3. Human-in-the-Loop
The graph can pause at any node and wait for...

## Sources
- [Source 1: langchain_docs.pdf, page 12]
- [Source 2: langgraph_paper.pdf, page 3]
\`\`\``,

    zh: `## 写作节点：报告综合

写作节点获取研究员的原始笔记并将其转换为精心制作的结构化 Markdown 报告。这是图形退出之前的最后阶段。

### 实现

\`\`\`python
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一名专业技术写作人员。\\n"
     "将研究笔记转换为清晰、结构良好的报告。\\n"
     "使用 Markdown 格式：标题、项目符号、适当的代码块。\\n"
     "简洁而全面。有来源时引用来源。"),
    ("human",
     "任务：{task}\\n\\n"
     "研究笔记：\\n{research}\\n\\n"
     "撰写最终报告："),
])

def writer_node(state: AgentState) -> AgentState:
    chain = WRITER_PROMPT | get_llm(temperature=0.3) | StrOutputParser()
    report = chain.invoke({
        "task": state["task"],
        "research": state.get("research_output", ""),
    })
    return {**state, "final_report": report}
\`\`\``,
  },
}

export const runGraph: TopicContent = {
  id: 'run-graph',
  emoji: '▶️',
  title: { en: 'Running the Multi-Agent Graph', zh: '运行多智能体图' },
  contentType: 'code',
  content: {
    en: `## Running the Multi-Agent Graph

\`run_graph.py\` is the entry point that wires all nodes together, compiles the graph, and streams execution events to the console.

### Compiling the Graph

\`\`\`python
# phase3_enterprise/multi_agent/run_graph.py
from phase3_enterprise.multi_agent.graph_definition import AgentState, build_graph
from phase3_enterprise.multi_agent.researcher_agent import researcher_node
from phase3_enterprise.multi_agent.supervisor import supervisor_node
from phase3_enterprise.multi_agent.writer_agent import writer_node


def get_compiled_graph():
    """Compile the multi-agent graph with all nodes registered."""
    return build_graph(
        supervisor_node=supervisor_node,
        researcher_node=researcher_node,
        writer_node=writer_node,
    )
\`\`\`

### Streaming Execution

\`\`\`python
def run(task: str, verbose: bool = True) -> AgentState:
    """Execute the multi-agent pipeline and stream progress to console."""
    graph = get_compiled_graph()

    initial_state: AgentState = {
        "messages": [],
        "task": task,
        "research_output": "",
        "final_report": "",
        "next_step": "",
    }

    print(f"MULTI-AGENT GRAPH — Task: {task}")
    final_state = initial_state

    # graph.stream() yields state snapshots after each node runs
    for event in graph.stream(initial_state):
        for node_name, node_state in event.items():
            if verbose:
                step = node_state.get("next_step", "") or "producing output"
                print(f"\\n[{node_name.upper()}] → {step}")

                if node_name == "researcher" and node_state.get("research_output"):
                    print(f"  Research preview: {node_state['research_output'][:300]}...")

                if node_name == "writer" and node_state.get("final_report"):
                    print(f"  Report preview: {node_state['final_report'][:300]}...")

        final_state = {**final_state, **node_state}

    return final_state
\`\`\`

### Running It

\`\`\`python
# From Python
result = run(
    task="Explain how LangGraph improves upon standard agent loops, "
         "including state management, checkpointing, and human-in-the-loop."
)
print(result["final_report"])

# From command line
# python phase3_enterprise/multi_agent/run_graph.py
\`\`\`

### Streaming Output Example

\`\`\`
MULTI-AGENT GRAPH — Task: Explain LangGraph vs standard agent loops

[SUPERVISOR] → researcher

[RESEARCHER] → producing output
  Research preview: ## Research Notes
  1. LangGraph models execution as a directed graph...
  2. Standard AgentExecutor uses a simple while-loop...

[SUPERVISOR] → writer

[WRITER] → producing output
  Report preview: # LangGraph vs Standard Agent Loops

  ## Overview
  LangGraph introduces stateful, graph-based orchestration...

[SUPERVISOR] → end
\`\`\`

### \`graph.stream\` vs \`graph.invoke\`

| Method | Use when |
|---|---|
| \`graph.invoke(state)\` | Just need the final result |
| \`graph.stream(state)\` | Want real-time progress + intermediate states |`,

    zh: `## 运行多智能体图

\`run_graph.py\` 是将所有节点连接在一起、编译图并将执行事件流式传输到控制台的入口点。

### 编译图

\`\`\`python
def get_compiled_graph():
    """编译具有所有注册节点的多智能体图。"""
    return build_graph(
        supervisor_node=supervisor_node,
        researcher_node=researcher_node,
        writer_node=writer_node,
    )
\`\`\`

### 流式执行

\`\`\`python
def run(task: str, verbose: bool = True) -> AgentState:
    graph = get_compiled_graph()
    initial_state: AgentState = {
        "messages": [], "task": task,
        "research_output": "", "final_report": "", "next_step": "",
    }

    for event in graph.stream(initial_state):
        for node_name, node_state in event.items():
            if verbose:
                print(f"[{node_name.upper()}] → {node_state.get('next_step', '生成输出')}")

    return final_state
\`\`\`

### \`graph.stream\` 与 \`graph.invoke\`

| 方法 | 使用场景 |
|---|---|
| \`graph.invoke(state)\` | 只需要最终结果 |
| \`graph.stream(state)\` | 需要实时进度 + 中间状态 |`,
  },
}
