import type { TopicContent } from '../types'

export const whatIsML: TopicContent = {
  id: 'what-is-ml',
  title: { en: 'What is Machine Learning?', zh: '什么是机器学习' },
  contentType: 'article',
  content: {
    en: `Machine Learning (ML) is a branch of artificial intelligence that enables systems to **learn from data** and improve their performance without being explicitly programmed for each task.

## The Core Idea

Instead of writing rules manually:

\`\`\`
# Traditional programming
if temperature > 30: recommend("shorts")
elif temperature > 20: recommend("t-shirt")
else: recommend("jacket")
\`\`\`

Machine learning **discovers these rules automatically** from examples:

\`\`\`python
# Machine learning approach
from sklearn.tree import DecisionTreeClassifier

# Historical data: [temperature, humidity] → clothing recommendation
X = [[32, 60], [25, 55], [18, 70], [10, 80]]
y = ["shorts", "t-shirt", "jacket", "coat"]

model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for new data
print(model.predict([[28, 65]]))  # ['t-shirt']
\`\`\`

## Why Machine Learning Now?

Three forces converged to make ML practical:

| Factor | Old State | Now |
|---|---|---|
| **Data** | Scarce, expensive | Abundant, cheap |
| **Compute** | CPU-limited | GPU/TPU clusters |
| **Algorithms** | Shallow models | Deep neural networks |

## The ML Workflow

Every ML project follows the same fundamental pipeline:

1. **Problem Definition** — What are we predicting? What's the cost of errors?
2. **Data Collection** — Gather labeled examples
3. **Data Preparation** — Clean, transform, feature engineering
4. **Model Selection** — Choose algorithm family
5. **Training** — Fit model parameters to data
6. **Evaluation** — Measure generalization on held-out data
7. **Deployment** — Serve predictions in production
8. **Monitoring** — Track drift and performance over time

## What ML Can and Cannot Do

**ML excels at:**
- Pattern recognition in high-dimensional data (images, text, audio)
- Forecasting from historical sequences
- Anomaly detection in complex systems
- Personalization at scale

**ML struggles with:**
- Tasks requiring strict logical reasoning
- Out-of-distribution generalization
- Very small datasets (< ~100 examples)
- Explaining *why* it made a decision

> The goal of ML is not to memorize training data, but to **generalize** to new, unseen examples. This tension between fitting the training data well and generalizing is the central challenge of machine learning.`,

    zh: `机器学习（ML）是人工智能的一个分支，它使系统能够**从数据中学习**，并在不被显式编程的情况下提升性能。

## 核心思想

与其手动写规则：

\`\`\`
# 传统编程
if temperature > 30: recommend("shorts")
elif temperature > 20: recommend("t-shirt")
else: recommend("jacket")
\`\`\`

机器学习从示例中**自动发现这些规则**：

\`\`\`python
# 机器学习方法
from sklearn.tree import DecisionTreeClassifier

# 历史数据：[温度, 湿度] → 服装推荐
X = [[32, 60], [25, 55], [18, 70], [10, 80]]
y = ["shorts", "t-shirt", "jacket", "coat"]

model = DecisionTreeClassifier()
model.fit(X, y)

# 对新数据进行预测
print(model.predict([[28, 65]]))  # ['t-shirt']
\`\`\`

## 为什么现在是机器学习的时代？

三股力量汇聚，使ML变得实用：

| 因素 | 过去 | 现在 |
|---|---|---|
| **数据** | 稀缺、昂贵 | 丰富、廉价 |
| **计算力** | 受CPU限制 | GPU/TPU集群 |
| **算法** | 浅层模型 | 深度神经网络 |

## ML工作流程

每个ML项目都遵循相同的基本流水线：

1. **问题定义** — 我们要预测什么？错误的代价是什么？
2. **数据收集** — 收集标注样本
3. **数据准备** — 清洗、转换、特征工程
4. **模型选择** — 选择算法族
5. **训练** — 将模型参数拟合到数据
6. **评估** — 在保留数据上衡量泛化能力
7. **部署** — 在生产环境中提供预测
8. **监控** — 追踪分布漂移和性能

> ML的目标不是记住训练数据，而是**泛化**到新的、未见过的样本。在训练数据上拟合良好与泛化之间的张力，是机器学习的核心挑战。`,
  },
}

export const typesOfML: TopicContent = {
  id: 'types-of-ml',
  title: { en: 'Types of Machine Learning', zh: '机器学习的类型' },
  contentType: 'article',
  content: {
    en: `Machine learning is broadly divided into three paradigms based on the **nature of the training signal** available.

## Supervised Learning

The model learns from **labeled examples** — each input has a corresponding target output.

\`\`\`python
import numpy as np
from sklearn.linear_model import LinearRegression

# House price prediction
# Features: [sqft, bedrooms, age]
X_train = np.array([
    [1200, 2, 10],
    [1800, 3, 5],
    [2400, 4, 2],
    [900,  1, 20],
])
y_train = np.array([250000, 380000, 520000, 180000])  # prices

model = LinearRegression()
model.fit(X_train, y_train)

# Predict new house
new_house = np.array([[1500, 3, 8]])
print(f"Predicted price: \${model.predict(new_house)[0]:,.0f}")
\`\`\`

**Subtypes:**
- **Regression** — predict a continuous value (house prices, temperature)
- **Classification** — predict a discrete label (spam/not-spam, cat/dog)

## Unsupervised Learning

No labels provided — the model discovers **hidden structure** in data.

\`\`\`python
from sklearn.cluster import KMeans
import numpy as np

# Customer segmentation — no labels, just purchase behavior
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Cluster assignments:", kmeans.labels_)
# Customers grouped by behavior without predefined categories
\`\`\`

**Common tasks:**
- **Clustering** — group similar data points (K-Means, DBSCAN)
- **Dimensionality Reduction** — compress features (PCA, t-SNE, UMAP)
- **Anomaly Detection** — find outliers (Isolation Forest)
- **Generative Modeling** — learn data distribution (VAE, GAN)

## Reinforcement Learning

An **agent** learns by interacting with an **environment**, receiving **rewards** or **penalties**.

\`\`\`python
# Conceptual RL loop — not sklearn, but illustrates the paradigm
class SimpleAgent:
    def __init__(self, n_actions):
        self.q_table = {}  # state → action values
        self.learning_rate = 0.1
        self.gamma = 0.99  # discount factor

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
        return max(range(self.n_actions),
                   key=lambda a: self.q_table[state][a])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table.get(next_state, [0.0]))
        # Bellman equation
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
\`\`\`

## Semi-Supervised & Self-Supervised

Modern approaches that leverage **unlabeled data** with limited labeled examples:

| Paradigm | Labeled Data | Unlabeled Data | Examples |
|---|---|---|---|
| Supervised | Large | None | ResNet, BERT fine-tuning |
| Unsupervised | None | Large | K-Means, PCA |
| Semi-Supervised | Small | Large | Label Propagation |
| Self-Supervised | None | Large | BERT pre-training, SimCLR |
| Reinforcement | Rewards only | Environment | AlphaGo, ChatGPT (RLHF) |

> **Key insight**: Self-supervised learning (contrastive learning, masked prediction) has become the dominant pre-training strategy for large models because it can leverage virtually unlimited unlabeled data.`,

    zh: `机器学习根据**训练信号的性质**大致分为三种范式。

## 监督学习

模型从**标记样本**中学习——每个输入都有对应的目标输出。

\`\`\`python
import numpy as np
from sklearn.linear_model import LinearRegression

# 房价预测
# 特征：[面积, 卧室数, 房龄]
X_train = np.array([
    [1200, 2, 10],
    [1800, 3, 5],
    [2400, 4, 2],
    [900,  1, 20],
])
y_train = np.array([250000, 380000, 520000, 180000])

model = LinearRegression()
model.fit(X_train, y_train)

new_house = np.array([[1500, 3, 8]])
print(f"预测价格: \${model.predict(new_house)[0]:,.0f}")
\`\`\`

**子类型：**
- **回归** — 预测连续值（房价、温度）
- **分类** — 预测离散标签（垃圾邮件/正常邮件，猫/狗）

## 无监督学习

不提供标签——模型发现数据中的**隐藏结构**。

\`\`\`python
from sklearn.cluster import KMeans
import numpy as np

# 客户细分——没有标签，只有购买行为
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("聚类分配:", kmeans.labels_)
\`\`\`

## 强化学习

**智能体**通过与**环境**交互来学习，接收**奖励**或**惩罚**。

| 范式 | 标记数据 | 未标记数据 | 示例 |
|---|---|---|---|
| 监督学习 | 大量 | 无 | ResNet, BERT微调 |
| 无监督学习 | 无 | 大量 | K-Means, PCA |
| 半监督学习 | 少量 | 大量 | 标签传播 |
| 自监督学习 | 无 | 大量 | BERT预训练, SimCLR |
| 强化学习 | 仅奖励 | 环境 | AlphaGo, ChatGPT (RLHF) |

> **关键洞察**：自监督学习（对比学习、掩码预测）已成为大模型的主流预训练策略，因为它可以利用几乎无限的未标记数据。`,
  },
}
