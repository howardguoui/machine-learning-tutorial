import type { TopicContent } from '../types'

export const ragEvaluation: TopicContent = {
  id: 'rag-evaluation',
  emoji: '📏',
  title: { en: 'RAG Evaluation (LLM-as-Judge)', zh: 'RAG 评估（LLM 作为评判者）' },
  contentType: 'code',
  content: {
    en: `## RAG Evaluation (LLM-as-Judge)

Measuring RAG quality requires three independent metrics, each scored 0–1 by an LLM acting as an objective judge.

### The Three Metrics

| Metric | Question it answers | Target |
|---|---|---|
| Context Precision | Is the retrieved context relevant to the question? | > 0.8 |
| Answer Faithfulness | Does the answer stay within the retrieved context? | > 0.9 |
| Answer Relevance | Does the answer address the original question? | > 0.8 |

### Score Dataclass

\`\`\`python
# phase4_production/evaluation/eval_rag.py
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class RAGEvalScore:
    context_precision: float    # 0–1: how relevant is the retrieved context?
    answer_faithfulness: float  # 0–1: does answer stay within context?
    answer_relevance: float     # 0–1: does answer address the question?
    overall: float              # average of all three
    explanation: str            # human-readable summary
\`\`\`

### Evaluation Prompts

\`\`\`python
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an objective evaluator. Score the answer's faithfulness to the context on 0.0–1.0.\\n"
     "1.0 = answer uses ONLY information from the context\\n"
     "0.5 = answer partially uses context but adds outside info\\n"
     "0.0 = answer ignores context or contradicts it\\n"
     "Respond with ONLY a number between 0.0 and 1.0."),
    ("human", "Context: {context}\\n\\nAnswer: {answer}\\n\\nFaithfulness score:"),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Score how well the answer addresses the question on 0.0–1.0.\\n"
     "1.0 = answer directly and completely answers the question\\n"
     "0.0 = answer is off-topic or doesn't answer the question\\n"
     "Respond with ONLY a number between 0.0 and 1.0."),
    ("human", "Question: {question}\\n\\nAnswer: {answer}\\n\\nRelevance score:"),
])

PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Score how relevant the context is to answering the question on 0.0–1.0.\\n"
     "1.0 = context directly contains info needed to answer the question\\n"
     "0.0 = context is irrelevant to the question\\n"
     "Respond with ONLY a number between 0.0 and 1.0."),
    ("human", "Question: {question}\\n\\nContext: {context}\\n\\nContext precision score:"),
])
\`\`\`

### Score Parser + Evaluator

\`\`\`python
import re

def _parse_score(raw: str) -> float:
    """Extract a float score from LLM output. Defaults to 0.5 on failure."""
    matches = re.findall(r"0\\.\\d+|1\\.0|[01]", raw.strip())
    if matches:
        return max(0.0, min(1.0, float(matches[0])))
    return 0.5


def evaluate_rag_response(
    question: str,
    context: str,
    answer: str,
) -> RAGEvalScore:
    """Evaluate a RAG response using three independent LLM-as-judge calls."""
    llm = get_llm(temperature=0.0)

    faithfulness_chain = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    relevance_chain    = RELEVANCE_PROMPT    | llm | StrOutputParser()
    precision_chain    = PRECISION_PROMPT    | llm | StrOutputParser()

    faith = _parse_score(faithfulness_chain.invoke({"context": context, "answer": answer}))
    rel   = _parse_score(relevance_chain.invoke({"question": question, "answer": answer}))
    prec  = _parse_score(precision_chain.invoke({"question": question, "context": context}))

    overall = (faith + rel + prec) / 3
    explanation = (
        f"Context Precision: {prec:.2f} | "
        f"Answer Faithfulness: {faith:.2f} | "
        f"Answer Relevance: {rel:.2f}"
    )

    return RAGEvalScore(
        context_precision=prec,
        answer_faithfulness=faith,
        answer_relevance=rel,
        overall=overall,
        explanation=explanation,
    )
\`\`\`

### Example Evaluation Run

\`\`\`python
score = evaluate_rag_response(
    question="What is LCEL?",
    context="LCEL (LangChain Expression Language) uses the | pipe operator to compose chains...",
    answer="LCEL stands for LangChain Expression Language. It uses the | operator to connect components.",
)

print(f"Context Precision:   {score.context_precision:.2f}")   # 0.95
print(f"Answer Faithfulness: {score.answer_faithfulness:.2f}") # 0.98
print(f"Answer Relevance:    {score.answer_relevance:.2f}")    # 0.97
print(f"Overall:             {score.overall:.2f}")             # 0.97
\`\`\``,

    zh: `## RAG 评估（LLM 作为评判者）

衡量 RAG 质量需要三个独立指标，每个指标由充当客观评判者的 LLM 评分 0–1。

### 三个指标

| 指标 | 它回答的问题 | 目标 |
|---|---|---|
| 上下文精度 | 检索到的上下文与问题相关吗？ | > 0.8 |
| 答案忠实度 | 答案是否保持在检索到的上下文内？ | > 0.9 |
| 答案相关性 | 答案是否回答了原始问题？ | > 0.8 |

### 评估函数

\`\`\`python
def evaluate_rag_response(question, context, answer) -> RAGEvalScore:
    llm = get_llm(temperature=0.0)

    faith = _parse_score((FAITHFULNESS_PROMPT | llm | StrOutputParser())
                         .invoke({"context": context, "answer": answer}))
    rel   = _parse_score((RELEVANCE_PROMPT | llm | StrOutputParser())
                         .invoke({"question": question, "answer": answer}))
    prec  = _parse_score((PRECISION_PROMPT | llm | StrOutputParser())
                         .invoke({"question": question, "context": context}))

    return RAGEvalScore(
        context_precision=prec,
        answer_faithfulness=faith,
        answer_relevance=rel,
        overall=(faith + rel + prec) / 3,
        explanation=f"精度：{prec:.2f} | 忠实度：{faith:.2f} | 相关性：{rel:.2f}",
    )
\`\`\``,
  },
}

export const codeSelfHeal: TopicContent = {
  id: 'code-self-heal',
  emoji: '🩹',
  title: { en: 'Python Code Self-Healing', zh: 'Python 代码自愈' },
  contentType: 'code',
  content: {
    en: `## Python Code Self-Healing

Like the SQL self-healer, this pattern generates Python code from a natural language request, executes it in a sandbox, and automatically corrects errors using the traceback as feedback.

### Prompts

\`\`\`python
# phase4_production/self_healing/code_self_heal.py
from langchain_core.prompts import ChatPromptTemplate

CODE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Python expert. Write clean, working Python code.\\n"
     "RULES:\\n"
     "- Return ONLY raw Python code — no markdown fences, no explanations\\n"
     "- Use only Python standard library (no external imports)\\n"
     "- Store the final answer in a variable called \`result\`"),
    ("human", "Task: {request}\\n\\nPython code:"),
])

CODE_CORRECTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Python debugging expert. Fix the code error.\\n"
     "Return ONLY raw Python code — no fences, no explanations.\\n"
     "Store the final answer in a variable called \`result\`"),
    ("human",
     "Original request: {original_request}\\n\\n"
     "Failed code:\\n{failed_code}\\n\\n"
     "Error:\\n{error_message}\\n\\n"
     "Corrected code:"),
])
\`\`\`

### Sandboxed Python Executor

\`\`\`python
import io
import contextlib
from typing import Any


def _execute_python(code: str) -> Any:
    """Execute Python code in a restricted namespace.

    The code must set a variable called \`result\`.
    stdout is captured to avoid terminal output pollution.

    Returns:
        The value of \`result\` after execution.

    Raises:
        SyntaxError, NameError, KeyError, or any runtime exception.
    """
    stdout_buf = io.StringIO()
    namespace: dict[str, Any] = {}

    with contextlib.redirect_stdout(stdout_buf):
        exec(code, namespace)   # isolated namespace — no access to globals

    if "result" not in namespace:
        # Fall back to stdout output if no \`result\` variable set
        stdout_output = stdout_buf.getvalue()
        if stdout_output:
            return stdout_output.strip()
        raise KeyError("Code did not set a \`result\` variable and produced no output")

    return namespace["result"]
\`\`\`

### Building the Healing Chain

\`\`\`python
def build_code_heal_chain(max_retries: int = 3):
    """Build a self-healing Python code generation + execution chain."""
    llm = get_llm(temperature=0.1)   # slight randomness helps generate diverse fixes

    generator_chain = CODE_GENERATOR_PROMPT | llm | StrOutputParser()
    corrector_chain = CODE_CORRECTOR_PROMPT | llm | StrOutputParser()

    return build_retry_chain(
        generator_chain=generator_chain,
        executor_fn=_execute_python,
        corrector_chain=corrector_chain,
        max_retries=max_retries,
    )
\`\`\`

### Demo Run

\`\`\`python
chain = build_code_heal_chain()

tasks = [
    "Reverse a string 'hello world' and store it in \`result\`",
    "Find the sum of squares of numbers 1 to 10 and store in \`result\`",
    "Create a dict with keys 'a','b','c' mapped to 1,2,3 and store in \`result\`",
]

for task in tasks:
    output = chain({"request": task})
    if output["error"]:
        print(f"  ❌ Failed: {output['error']}")
    else:
        print(f"  ✅ {output['result']} (in {output['attempts']} attempt(s))")

# Output:
# ✅ dlrow olleh (in 1 attempt(s))
# ✅ 385 (in 1 attempt(s))
# ✅ {'a': 1, 'b': 2, 'c': 3} (in 1 attempt(s))
\`\`\`

### Self-Healing in Action

\`\`\`
Task: "Sort a list of dicts by the 'score' key descending, store in \`result\`"

Attempt 1 — Generated code:
  data = [{"name": "A", "score": 3}, {"name": "B", "score": 1}]
  result = sorted(data, key=lambda x: x["score"], descending=True)  # ← wrong kwarg

Error: TypeError: 'descending' is an invalid keyword argument for sorted()

Attempt 2 — Corrected code:
  data = [{"name": "A", "score": 3}, {"name": "B", "score": 1}]
  result = sorted(data, key=lambda x: x["score"], reverse=True)  # ← fixed!

✅ Success after 2 attempts: [{'name': 'A', 'score': 3}, {'name': 'B', 'score': 1}]
\`\`\`

### SQL vs Python Self-Healing Comparison

| Feature | SQL Self-Heal | Python Self-Heal |
|---|---|---|
| Generator | SQL generator prompt | Python code generator |
| Executor | SQLite connection | Sandboxed \`exec()\` |
| Error type | \`sqlite3.Error\` | Any Python exception |
| Schema injection | Yes (from SQLite) | No |
| Security concern | SQL injection | Arbitrary code exec (sandbox it!) |`,

    zh: `## Python 代码自愈

与 SQL 自愈器类似，该模式从自然语言请求生成 Python 代码，在沙箱中执行它，并使用回溯作为反馈自动纠正错误。

### 沙箱执行器

\`\`\`python
def _execute_python(code: str) -> Any:
    """在受限命名空间中执行 Python 代码。"""
    stdout_buf = io.StringIO()
    namespace: dict[str, Any] = {}

    with contextlib.redirect_stdout(stdout_buf):
        exec(code, namespace)   # 隔离命名空间

    if "result" not in namespace:
        if stdout_output := stdout_buf.getvalue():
            return stdout_output.strip()
        raise KeyError("代码未设置 \`result\` 变量")
    return namespace["result"]
\`\`\`

### 自愈演示

\`\`\`python
chain = build_code_heal_chain()
output = chain({"request": "将字符串 'hello world' 反转并存储在 \`result\` 中"})
print(output["result"])  # → 'dlrow olleh'
\`\`\`

### SQL vs Python 自愈比较

| 特性 | SQL 自愈 | Python 自愈 |
|---|---|---|
| 生成器 | SQL 生成器提示 | Python 代码生成器 |
| 执行器 | SQLite 连接 | 沙箱 \`exec()\` |
| 错误类型 | \`sqlite3.Error\` | 任何 Python 异常 |
| 安全关注 | SQL 注入 | 任意代码执行（沙箱化！） |`,
  },
}
