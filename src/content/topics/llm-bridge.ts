import type { TopicContent } from '../types'

export const mlToLLM: TopicContent = {
  id: 'ml-to-llm',
  title: { en: 'From Classical ML to LLMs', zh: '从传统ML到大语言模型' },
  contentType: 'article',
  content: {
    en: `Understanding classical ML gives you the foundation to understand **why LLMs work** and when to use them versus traditional approaches.

## The Evolution: ML → Deep Learning → LLMs

| Era | Representation | Key Techniques | Limitation |
|---|---|---|---|
| **Classical ML** | Hand-crafted features | SVM, Trees, Logistic Reg | Feature engineering bottleneck |
| **Deep Learning** | Learned features | CNNs, RNNs, Autoencoders | Task-specific, data-hungry |
| **Pre-trained Models** | Transferable embeddings | Word2Vec, BERT, GPT | Context window limits |
| **LLMs** | In-context learning | GPT-4, Claude, LLaMA | Cost, hallucination, reasoning |

## When Classical ML Still Wins

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# SCENARIO: Tabular data with engineered features — GBT often beats LLMs

# Credit risk: structured, tabular, requires fast inference, auditability
credit_features = ['credit_score', 'debt_to_income', 'employment_years',
                   'num_credit_lines', 'payment_history', 'account_age']

# GBT: deterministic, fast, interpretable, no API cost
gbt = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ))
])

# Rule of thumb: for tabular data < 1M rows with known features,
# start with GBT before considering LLMs.
print("Classical ML strengths for structured data:")
print("  ✅ Fast inference (microseconds vs seconds)")
print("  ✅ Deterministic predictions")
print("  ✅ Interpretable (SHAP values)")
print("  ✅ No API cost, no privacy concerns")
print("  ✅ Works well with < 10k samples")
\`\`\`

## Embeddings: The Bridge Between ML and LLMs

\`\`\`python
# Embeddings convert text/items into dense vectors — enabling ML techniques on text

# Classical: TF-IDF (sparse, no semantics)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "machine learning is fascinating",
    "deep learning is a subset of machine learning",
    "cats and dogs are popular pets",
]
labels = [0, 0, 1, 1, 0]  # 0=animals, 1=ML

tfidf = TfidfVectorizer(max_features=50)
X_tfidf = tfidf.fit_transform(texts)
print(f"TF-IDF shape: {X_tfidf.shape}")  # (5, 50) — sparse vectors

# Modern: sentence-transformers (dense, semantic)
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    print(f"Semantic embeddings shape: {embeddings.shape}")  # (5, 384)
    # These embeddings capture meaning — "cat" and "dog" are close
    # Can now use ANY sklearn algorithm on text!
except ImportError:
    print("Install: pip install sentence-transformers")
    # OpenAI alternative:
    print("Or use OpenAI API: text-embedding-3-small → 1536-dim vectors")
\`\`\`

## RAG: Retrieval-Augmented Generation

\`\`\`python
# RAG combines classical ML (vector search) with LLMs (generation)
# Architecture: Query → Embed → Vector DB Search → Top-k Docs → LLM → Answer

class SimpleRAG:
    """
    Minimal RAG implementation using sklearn for retrieval.
    Production: replace with Chroma, Qdrant, or Pinecone.
    """
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def index(self, documents: list[str]):
        """Embed and store documents."""
        self.documents = documents
        # In production: use sentence-transformers or OpenAI embeddings
        # Here: TF-IDF as a proxy for demonstration
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(documents)
        print(f"Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Find top-k most similar documents."""
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

    def answer(self, query: str) -> str:
        """Retrieve context and (mock) generate answer."""
        relevant_docs = self.retrieve(query, top_k=2)
        context = "\\n".join([doc for doc, _ in relevant_docs])
        # In production: pass context to Claude/GPT API
        return f"Based on retrieved context:\\n{context}\\n\\n[LLM would answer here]"

# Usage
rag = SimpleRAG()
rag.index([
    "Python was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence.",
    "PyTorch is a popular deep learning framework.",
    "NumPy provides numerical computing tools for Python.",
    "The transformer architecture was introduced in 2017.",
])
answer = rag.answer("When was Python created?")
print(answer)
\`\`\`

## Choosing Between Classical ML and LLMs

\`\`\`
Decision Tree for ML vs LLM:

Is the data structured/tabular?
├── YES → Use GBT (XGBoost/LightGBM) first
│         Only add LLM for text features or if GBT fails
└── NO → Is the task NLP/vision/audio?
         ├── NLP → Is it classification/extraction?
         │         ├── Small dataset + known schema → Fine-tune BERT
         │         └── Complex, open-ended → LLM API (Claude/GPT)
         ├── Vision → Pre-trained CNN (ResNet/ViT) + fine-tune
         └── Time series → Temporal fusion transformer or classical

Is real-time inference required (< 10ms)?
├── YES → Classical ML (tree-based or small NN)
└── NO → LLM may be acceptable

Is interpretability legally required?
├── YES → Linear model + SHAP / Logistic Regression
└── NO → Any model
\`\`\`

> **Key insight**: LLMs are best viewed as **universal zero-shot learners** for unstructured data, while classical ML remains the gold standard for structured, tabular data. The most powerful production systems combine both.`,

    zh: `理解传统ML为您奠定了理解**LLM为何有效**的基础，以及何时使用它们而不是传统方法。

## 演变：ML → 深度学习 → LLM

| 时代 | 表示方式 | 关键技术 | 局限 |
|---|---|---|---|
| **传统ML** | 手工特征 | SVM、树、逻辑回归 | 特征工程瓶颈 |
| **深度学习** | 学习特征 | CNN、RNN | 任务特定，数据饥渴 |
| **预训练模型** | 可迁移嵌入 | Word2Vec、BERT | 上下文窗口限制 |
| **LLM** | 上下文学习 | GPT-4、Claude | 成本、幻觉 |

## 何时传统ML仍然胜出

- 结构化/表格数据：GBT通常优于LLM
- 需要确定性预测和可审计性
- 快速推理需求（微秒级而非秒级）
- 小数据集（< 10k样本）

## RAG：连接ML与LLM的桥梁

RAG结合了传统ML（向量搜索）和LLM（生成）：
**查询 → 嵌入 → 向量数据库检索 → Top-k文档 → LLM → 答案**

> **关键洞察**：LLM最好被视为非结构化数据的**通用零样本学习器**，而传统ML仍然是结构化、表格数据的金标准。最强大的生产系统结合了两者。`,
  },
}
