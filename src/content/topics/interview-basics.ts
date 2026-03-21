import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview BASICS — 7 beginner-first topics
// Math → Stats → ML Concepts → Data → Linear Models → Trees → Validation
// ─────────────────────────────────────────────────────────────────────────────

// ─── 1. Math Foundations ─────────────────────────────────────────────────────
export const interviewMathFoundations: TopicContent = {
  id: 'interview-math-foundations',
  title: { en: 'Interview: Math Foundations for ML', zh: '面试：ML数学基础' },
  contentType: 'article',
  content: {
    en: `## Math Foundations for ML — Interview Questions

> Start here before anything else. ML is just applied math. Once you see the intuition, the algorithms make sense.

---

### Q1 ★☆☆☆☆ — What is a vector? Why does ML use them?

**Simple answer:** A vector is just a list of numbers.

\`\`\`
A person: [age=25, height=175, weight=70]  ← this is a feature vector
An image: [pixel1=0.2, pixel2=0.8, ...]    ← 784 numbers for 28×28 image
A word:   [0.3, -0.1, 0.9, ...]            ← Word2Vec embedding
\`\`\`

**Why it matters:** Every data point in ML is a vector. The model learns by doing math on these vectors.

\`\`\`python
import numpy as np

person = np.array([25, 175, 70])   # age, height, weight
print(person.shape)                 # (3,) → 3-dimensional vector
\`\`\`

---

### Q2 ★☆☆☆☆ — What is a matrix? How is matrix multiplication used in ML?

**Simple answer:** A matrix is a table of numbers (rows × columns).

\`\`\`
Dataset with 3 people, 2 features:
X = [[25, 175],   ← person 1
     [30, 180],   ← person 2
     [22, 165]]   ← person 3
Shape: (3, 2)  →  3 rows, 2 columns
\`\`\`

**Matrix multiplication in a neural net:**
- Input X has shape (batch=100, features=784)
- Weight W has shape (features=784, hidden=256)
- Output = X @ W → shape (100, 256) — each of 100 samples gets a 256-dim output

\`\`\`python
X = np.random.randn(100, 784)   # 100 images
W = np.random.randn(784, 256)   # learned weights
output = X @ W                  # matrix multiply → (100, 256)
\`\`\`

**Rule:** (m×n) @ (n×p) = (m×p). The inner dimension must match.

---

### Q3 ★★☆☆☆ — What is a derivative and why does ML need it?

**Simple answer:** A derivative tells you the slope — how fast a function changes.

\`\`\`
f(x) = x²
f'(x) = 2x     ← derivative

At x=3: slope = 2×3 = 6  (going uphill steeply)
At x=0: slope = 0        (at the bottom — minimum!)
\`\`\`

**Why ML needs it:** Training = finding parameters that minimize loss.
- Loss is high → model is wrong
- Derivative tells us which direction to move the parameters to reduce loss

| Derivative | Meaning | Action |
|-----------|---------|--------|
| Positive (> 0) | Loss increases as param increases | Decrease param |
| Negative (< 0) | Loss decreases as param increases | Increase param |
| Zero (= 0) | At a minimum | Stop! |

---

### Q4 ★★☆☆☆ — What is a gradient? How is it different from a derivative?

**Simple answer:**
- **Derivative**: slope of a function with ONE input variable
- **Gradient**: slope in ALL directions when you have MANY variables

\`\`\`
Loss function with 2 weights (w1, w2):
L(w1, w2) = w1² + w2²

Gradient = [∂L/∂w1, ∂L/∂w2] = [2w1, 2w2]
           ↑ partial derivative with respect to each weight
\`\`\`

**Intuition:** Imagine you're on a mountain in the fog. The gradient is the direction of steepest ascent. To reach the valley (minimum loss), walk opposite to the gradient.

\`\`\`python
# Simple gradient computation with PyTorch
import torch

w = torch.tensor([3.0, 4.0], requires_grad=True)
loss = (w ** 2).sum()   # L = w1² + w2²
loss.backward()          # compute gradients

print(w.grad)            # tensor([6., 8.]) = [2*3, 2*4]
\`\`\`

---

### Q5 ★★☆☆☆ — What is a dot product and why does it measure similarity?

**Simple answer:** Dot product = multiply element-by-element, then sum.

\`\`\`
a = [1, 2, 3]
b = [4, 5, 6]
a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
\`\`\`

**Why it measures similarity:**
- Same direction → large positive dot product (similar)
- Perpendicular → dot product = 0 (unrelated)
- Opposite direction → large negative dot product (opposite)

**Used everywhere in ML:**
- Attention mechanism: similarity between query and key vectors
- Cosine similarity: dot product divided by lengths
- Linear layer: output = X · W (weighted sum of inputs)

\`\`\`python
import numpy as np

# Cosine similarity (dot product normalized)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 0.0])   # pointing right
b = np.array([1.0, 0.0])   # same direction
c = np.array([0.0, 1.0])   # perpendicular

print(cosine_sim(a, b))   # 1.0 (identical)
print(cosine_sim(a, c))   # 0.0 (no similarity)
\`\`\``,

    zh: `## ML数学基础面试题

> 一切从这里开始。ML本质上是应用数学。理解直觉后，所有算法都变得简单。

---

### Q1 ★☆☆☆☆ — 什么是向量？ML为什么要使用它？

**通俗解释：** 向量就是一组数字。

\`\`\`
一个人的特征：[年龄=25, 身高=175, 体重=70]  ← 特征向量
一张图片：    [像素1=0.2, 像素2=0.8, ...]    ← 784个数字（28×28图片）
一个词：      [0.3, -0.1, 0.9, ...]          ← Word2Vec词向量
\`\`\`

**为什么重要：** ML中每个数据点都是一个向量，模型通过对这些向量做数学运算来学习。

\`\`\`python
import numpy as np

person = np.array([25, 175, 70])   # 年龄、身高、体重
print(person.shape)                 # (3,) → 3维向量
\`\`\`

---

### Q2 ★☆☆☆☆ — 什么是矩阵？矩阵乘法在ML中有什么用？

**通俗解释：** 矩阵就是一张数字表格（行×列）。

\`\`\`
3个人，2个特征的数据集：
X = [[25, 175],   ← 第1个人
     [30, 180],   ← 第2个人
     [22, 165]]   ← 第3个人
形状：(3, 2)  →  3行2列
\`\`\`

**神经网络中的矩阵乘法：**
- 输入X形状：(batch=100, features=784)
- 权重W形状：(features=784, hidden=256)
- 输出 = X @ W → 形状(100, 256)

\`\`\`python
X = np.random.randn(100, 784)   # 100张图片
W = np.random.randn(784, 256)   # 学习到的权重
output = X @ W                  # 矩阵乘法 → (100, 256)
\`\`\`

**规则：** (m×n) @ (n×p) = (m×p)，内维度必须一致。

---

### Q3 ★★☆☆☆ — 什么是导数？ML为什么需要它？

**通俗解释：** 导数告诉你斜率——函数变化有多快。

\`\`\`
f(x) = x²
f'(x) = 2x     ← 导数

在x=3处：斜率 = 6（陡坡上升）
在x=0处：斜率 = 0（在最低点——极小值！）
\`\`\`

**ML为什么需要它：** 训练 = 找到能最小化损失的参数。
- 损失高 → 模型预测错误
- 导数告诉我们朝哪个方向调整参数才能减小损失

| 导数值 | 含义 | 操作 |
|--------|------|------|
| 正值 (>0) | 参数增大，损失增大 | 减小参数 |
| 负值 (<0) | 参数增大，损失减小 | 增大参数 |
| 零 (=0) | 在极小值处 | 停止！ |

---

### Q4 ★★☆☆☆ — 什么是梯度？它与导数有什么区别？

**通俗解释：**
- **导数**：只有一个输入变量时的斜率
- **梯度**：多个变量时，每个方向上的斜率合集

\`\`\`
有两个权重w1、w2的损失函数：
L(w1, w2) = w1² + w2²

梯度 = [∂L/∂w1, ∂L/∂w2] = [2w1, 2w2]
       ↑ 对每个权重的偏导数
\`\`\`

**直觉：** 想象你在雾中的山上。梯度指向最陡峭的上坡方向。要到达山谷（最小损失），朝梯度的反方向走。

\`\`\`python
import torch

w = torch.tensor([3.0, 4.0], requires_grad=True)
loss = (w ** 2).sum()   # L = w1² + w2²
loss.backward()          # 计算梯度

print(w.grad)            # tensor([6., 8.]) = [2*3, 2*4]
\`\`\`

---

### Q5 ★★☆☆☆ — 什么是点积？为什么它能衡量相似性？

**通俗解释：** 点积 = 逐元素相乘，然后求和。

\`\`\`
a = [1, 2, 3]
b = [4, 5, 6]
a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
\`\`\`

**为什么能衡量相似性：**
- 方向相同 → 点积大（相似）
- 垂直 → 点积=0（无关）
- 方向相反 → 点积为负（相反）

\`\`\`python
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 0.0])   # 向右
b = np.array([1.0, 0.0])   # 同方向
c = np.array([0.0, 1.0])   # 垂直

print(cosine_sim(a, b))   # 1.0（完全相同）
print(cosine_sim(a, c))   # 0.0（无相似性）
\`\`\``
  }
}

// ─── 2. Stats & Probability ──────────────────────────────────────────────────
export const interviewStatsProbability: TopicContent = {
  id: 'interview-stats-probability',
  title: { en: 'Interview: Statistics & Probability', zh: '面试：统计与概率' },
  contentType: 'article',
  content: {
    en: `## Statistics & Probability — Interview Questions

> Statistics is the language of data. Before you can build ML models, you need to understand the data you're working with.

---

### Q1 ★☆☆☆☆ — What are mean, median, variance, and standard deviation?

**Mean (average):** Sum of all values divided by count.
\`\`\`python
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = sum(data) / len(data)   # = 5.0
\`\`\`

**Median:** Middle value when sorted.
\`\`\`
Sorted: [2, 4, 4, 4, 5, 5, 7, 9]
Median = (4+5)/2 = 4.5   (even count → average middle two)
\`\`\`

**When to use which?**
- Use **mean** when data has no extreme outliers
- Use **median** when data has outliers (e.g., salaries, house prices)
- Example: In [1, 2, 3, 4, 100], mean=22 but median=3. Median is more representative.

**Variance (σ²):** Average of squared distances from the mean.
\`\`\`
Variance = mean of (each value - mean)²
         = [(2-5)² + (4-5)² + ... + (9-5)²] / 8 = 4.0
\`\`\`

**Standard Deviation (σ):** Square root of variance. Same units as data.
\`\`\`python
import numpy as np
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(np.mean(data), np.std(data))   # 5.0  2.0
\`\`\`

---

### Q2 ★☆☆☆☆ — What is a normal distribution? Why is it so common in ML?

**The bell curve:** Most values cluster around the mean, with fewer values at the extremes.

\`\`\`
μ=0, σ=1 (standard normal):

         ████
       ████████
     ████████████
   ████████████████
 -3  -2  -1   0   1   2   3
\`\`\`

**The 68-95-99.7 rule:**
- 68% of data falls within ±1σ
- 95% of data falls within ±2σ
- 99.7% of data falls within ±3σ

**Why it appears everywhere:**
- Central Limit Theorem: averages of any distribution become normal as sample size grows
- Many natural processes (height, weight, errors) follow it
- Many ML assumptions rely on it (e.g., Gaussian Naive Bayes, linear regression residuals)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Generate normal data
data = np.random.normal(loc=0, scale=1, size=10000)
print(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")
\`\`\`

---

### Q3 ★★☆☆☆ — What is conditional probability? State Bayes' Theorem.

**Conditional probability P(A|B):** Probability of A given that B has already happened.

\`\`\`
Example: Email spam detection
P(spam) = 0.2               ← 20% of all emails are spam
P(contains "free" | spam) = 0.8    ← 80% of spam contains "free"
P(contains "free" | not spam) = 0.1

Q: Given email contains "free", what's P(spam | "free")?
\`\`\`

**Bayes' Theorem:**
\`\`\`
                P(B|A) × P(A)
P(A|B) =    ─────────────────
                   P(B)

Plugging in:
P(spam | "free") = P("free" | spam) × P(spam) / P("free")
                 = 0.8 × 0.2 / (0.8×0.2 + 0.1×0.8)
                 = 0.16 / (0.16 + 0.08) = 0.67 (67%)
\`\`\`

**Where it's used in ML:**
- Naive Bayes classifier (direct application)
- Bayesian optimization for hyperparameter tuning
- Probabilistic graphical models

---

### Q4 ★★☆☆☆ — What is the Central Limit Theorem (CLT)?

**Statement:** If you take enough random samples from ANY distribution and compute their means, those means will be approximately normally distributed — regardless of the original distribution's shape.

\`\`\`
Dice rolls: discrete uniform distribution [1,2,3,4,5,6]
Average of 1 roll:   roughly uniform
Average of 2 rolls:  starts looking like a triangle
Average of 30 rolls: looks like a bell curve!
\`\`\`

**Why it matters for ML:**
- Justifies using t-tests and z-tests for model comparison
- Explains why model ensembles (averaging many models) are more stable
- Foundation of statistical hypothesis testing

\`\`\`python
import numpy as np

# Simulate CLT: roll a die many times, take averages
n_samples = 10000
n_rolls = 30

means = [np.random.randint(1, 7, n_rolls).mean() for _ in range(n_samples)]
print(f"Mean of means: {np.mean(means):.2f}")   # ~3.5
print(f"Std of means: {np.std(means):.2f}")     # ~0.54 (very small!)
\`\`\`

---

### Q5 ★★★☆☆ — What is correlation vs. causation? Why does it matter?

**Correlation:** Two variables move together statistically.
**Causation:** One variable directly causes the other.

\`\`\`
Ice cream sales ↑ and drowning deaths ↑ → correlated!
But: eating ice cream does NOT cause drowning.
True cause: hot weather causes both.
(Confounding variable!)
\`\`\`

**Pearson Correlation Coefficient r:**
- r = +1: perfect positive correlation
- r = 0: no linear relationship
- r = -1: perfect negative correlation

\`\`\`python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
r = np.corrcoef(x, y)[0, 1]
print(f"r = {r:.3f}")   # ~0.897 (strong positive)
\`\`\`

**Why it matters in ML:**
- Highly correlated features can cause multicollinearity in linear regression
- Feature selection based on correlation with target
- Never deploy a model based on spurious correlations — it will fail in production`,

    zh: `## 统计与概率面试题

> 统计学是数据的语言。在构建ML模型之前，你必须先理解你正在处理的数据。

---

### Q1 ★☆☆☆☆ — 什么是均值、中位数、方差和标准差？

**均值（平均数）：** 所有值的总和除以数量。
\`\`\`python
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = sum(data) / len(data)   # = 5.0
\`\`\`

**中位数：** 排序后位于中间的值。
\`\`\`
排序后：[2, 4, 4, 4, 5, 5, 7, 9]
中位数 = (4+5)/2 = 4.5   （偶数个数时取中间两个的平均）
\`\`\`

**何时用哪个？**
- 无极端异常值时用**均值**
- 有异常值时用**中位数**（如工资、房价）
- 示例：[1, 2, 3, 4, 100]中，均值=22，中位数=3，中位数更具代表性。

**方差(σ²)：** 每个值与均值之差的平方的平均。

**标准差(σ)：** 方差的平方根，与数据单位相同。
\`\`\`python
import numpy as np
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(np.mean(data), np.std(data))   # 5.0  2.0
\`\`\`

---

### Q2 ★☆☆☆☆ — 什么是正态分布？为什么它在ML中如此常见？

**钟形曲线：** 大多数值聚集在均值附近，极端值较少。

**68-95-99.7法则：**
- 68%的数据落在均值±1σ内
- 95%的数据落在均值±2σ内
- 99.7%的数据落在均值±3σ内

**为什么它无处不在：**
- 中心极限定理：随着样本量增大，任何分布的样本均值都趋向正态分布
- 许多自然现象（身高、体重、误差）都符合它
- 许多ML算法的假设依赖于它（如高斯朴素贝叶斯）

---

### Q3 ★★☆☆☆ — 什么是条件概率？贝叶斯定理是什么？

**条件概率 P(A|B)：** 已知B发生的前提下，A发生的概率。

\`\`\`
示例：邮件垃圾检测
P(垃圾邮件) = 0.2
P(含"免费" | 垃圾邮件) = 0.8
P(含"免费" | 非垃圾邮件) = 0.1

问：邮件含"免费"，P(垃圾邮件 | 含"免费") = ?
\`\`\`

**贝叶斯定理：**
\`\`\`
P(A|B) = P(B|A) × P(A) / P(B)

代入得：
P(垃圾 | "免费") = 0.8 × 0.2 / (0.8×0.2 + 0.1×0.8) = 67%
\`\`\`

**在ML中的应用：**
- 朴素贝叶斯分类器（直接应用）
- 贝叶斯优化（超参数调优）

---

### Q4 ★★☆☆☆ — 什么是中心极限定理（CLT）？

**定理内容：** 从任意分布中抽取足够多的随机样本并计算均值，这些均值将近似服从正态分布——无论原始分布是什么形状。

**为什么对ML重要：**
- 为模型比较中使用t检验和z检验提供依据
- 解释为什么模型集成（平均多个模型）更稳定
- 统计假设检验的基础

---

### Q5 ★★★☆☆ — 什么是相关性与因果关系？为什么重要？

**相关性：** 两个变量在统计上一起变动。
**因果关系：** 一个变量直接导致另一个变量。

\`\`\`
冰淇淋销量↑ 和溺水死亡↑ → 相关！
但：吃冰淇淋并不导致溺水。
真正原因：炎热天气同时导致两者。（混淆变量！）
\`\`\`

**皮尔逊相关系数 r：**
- r = +1：完全正相关
- r = 0：无线性关系
- r = -1：完全负相关

**在ML中的重要性：**
- 高度相关的特征会在线性回归中导致多重共线性
- 不要基于虚假相关部署模型——它会在生产中失败`
  }
}

// ─── 3. What is ML ───────────────────────────────────────────────────────────
export const interviewMLConcepts: TopicContent = {
  id: 'interview-ml-concepts',
  title: { en: 'Interview: Core ML Concepts', zh: '面试：ML核心概念' },
  contentType: 'article',
  content: {
    en: `## Core ML Concepts — Interview Questions

> The foundational vocabulary. Every ML interview starts with these. Know them cold.

---

### Q1 ★☆☆☆☆ — What is Machine Learning? How is it different from regular programming?

**Traditional programming:**
\`\`\`
Rules + Data → Output
You write IF email contains "free" THEN spam
\`\`\`

**Machine Learning:**
\`\`\`
Data + Answers → Rules (the model learns the rules!)
You give 10,000 labeled emails → model discovers its own spam rules
\`\`\`

**Formal definition:** ML is a subset of AI where systems learn from data to improve performance on a task without being explicitly programmed.

**Real example:** A spam filter that learns from examples instead of hand-coded rules can handle new spam variations it's never seen before.

---

### Q2 ★☆☆☆☆ — What are the 3 main types of Machine Learning?

| Type | Has Labels? | Goal | Example |
|------|------------|------|---------|
| **Supervised** | ✅ Yes | Predict output for new inputs | Email = spam/not spam |
| **Unsupervised** | ❌ No | Find hidden structure | Group customers by behavior |
| **Reinforcement** | 🎮 Rewards | Learn actions to maximize reward | AlphaGo, game-playing AI |

**Supervised subtypes:**
- **Classification:** Predict a category (spam vs. not spam, cat vs. dog)
- **Regression:** Predict a number (house price, temperature tomorrow)

\`\`\`python
# Supervised classification
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)   # y_train has labels: 0 or 1
predictions = model.predict(X_test)

# Supervised regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)   # y_train has numbers: house prices
predictions = model.predict(X_test)
\`\`\`

---

### Q3 ★★☆☆☆ — What is the difference between training, validation, and test sets?

**The golden rule: never evaluate on data used for training.**

\`\`\`
All your data:  [──────────────── 10,000 samples ────────────────]
                [── Train 70% ──][── Validation 15% ──][─ Test 15% ─]
\`\`\`

| Set | Used For | When |
|-----|---------|------|
| **Training** | Learning model weights | During training |
| **Validation** | Tuning hyperparameters, early stopping | During development |
| **Test** | Final unbiased evaluation | ONE time, at the end |

**Analogy:**
- Training = studying from textbooks
- Validation = practice exams
- Test = the real exam you've never seen

**The danger of test set leakage:** If you tune hyperparameters on the test set, your model is essentially memorizing the test answers. It won't generalize.

---

### Q4 ★★☆☆☆ — What is overfitting and underfitting?

**Underfitting (high bias):**
- Model is too simple to capture patterns
- High error on both training AND test data
- Like a student who didn't study enough

**Overfitting (high variance):**
- Model memorizes training data including noise
- Low error on training data, HIGH error on test data
- Like a student who memorized exact textbook problems but can't solve new ones

\`\`\`
Training accuracy vs Test accuracy:

Underfit:  Train=60%, Test=58%   ← both bad
Just right: Train=92%, Test=90%  ← small gap, both good
Overfit:   Train=99%, Test=72%   ← huge gap, test is bad!
\`\`\`

**How to fix overfitting:**
- Get more training data (best fix)
- Use regularization (L1, L2, Dropout)
- Reduce model complexity
- Use cross-validation

**How to fix underfitting:**
- Use a more complex model
- Add more features
- Train longer

---

### Q5 ★★☆☆☆ — What is the difference between a parameter and a hyperparameter?

| | Parameter | Hyperparameter |
|--|-----------|---------------|
| **What is it?** | Learned from data | Set by you before training |
| **Example (Linear Reg)** | Weights w, bias b | Learning rate, regularization strength |
| **Example (Decision Tree)** | Which feature to split on | Max depth, min samples per leaf |
| **How set?** | Gradient descent (automatic) | Grid search / manual (your job!) |

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

# Hyperparameters (YOU set these):
model = RandomForestClassifier(
    n_estimators=100,   # hyperparameter: number of trees
    max_depth=5,        # hyperparameter: tree depth
    random_state=42
)

# Parameters (model learns these from data):
model.fit(X_train, y_train)
# Now model.estimators_ contains the learned trees
# Each split threshold was learned from data
\`\`\``,

    zh: `## ML核心概念面试题

> 最基础的词汇表。每次ML面试都从这里开始，必须烂熟于心。

---

### Q1 ★☆☆☆☆ — 什么是机器学习？它与普通编程有什么不同？

**传统编程：**
\`\`\`
规则 + 数据 → 输出
你手动写：如果邮件含"免费" → 垃圾邮件
\`\`\`

**机器学习：**
\`\`\`
数据 + 答案 → 规则（模型自己学习规则！）
你给1万封标注好的邮件 → 模型自己发现垃圾邮件规律
\`\`\`

**正式定义：** 机器学习是AI的子集，系统从数据中学习，无需明确编程就能提升在特定任务上的表现。

---

### Q2 ★☆☆☆☆ — 机器学习有哪3种主要类型？

| 类型 | 有标签吗？ | 目标 | 示例 |
|------|-----------|------|------|
| **监督学习** | ✅ 有 | 预测新输入的输出 | 邮件=垃圾/正常 |
| **无监督学习** | ❌ 没有 | 发现隐藏结构 | 按行为对客户分组 |
| **强化学习** | 🎮 奖励 | 学习最大化奖励的动作 | AlphaGo |

**监督学习子类型：**
- **分类：** 预测类别（垃圾邮件vs正常，猫vs狗）
- **回归：** 预测数值（房价、明日气温）

---

### Q3 ★★☆☆☆ — 训练集、验证集和测试集有什么区别？

**黄金法则：绝不在训练过的数据上评估。**

| 集合 | 用途 | 时机 |
|------|------|------|
| **训练集** | 学习模型权重 | 训练过程中 |
| **验证集** | 调整超参数、早停 | 开发过程中 |
| **测试集** | 最终无偏评估 | 只用一次，最后用 |

**类比：**
- 训练集 = 课本
- 验证集 = 模拟考试
- 测试集 = 从未见过的真实考试

**测试集泄漏的危险：** 如果你在测试集上调整超参数，等于在背答案，模型无法泛化。

---

### Q4 ★★☆☆☆ — 什么是过拟合和欠拟合？

**欠拟合（高偏差）：**
- 模型太简单，无法捕捉规律
- 训练和测试误差都高
- 比喻：没认真学习的学生

**过拟合（高方差）：**
- 模型记住了训练数据包括噪声
- 训练误差低，测试误差高
- 比喻：只会背原题，不会举一反三的学生

\`\`\`
欠拟合：  训练=60%, 测试=58%   ← 都差
恰到好处：训练=92%, 测试=90%  ← 差距小，都好
过拟合：  训练=99%, 测试=72%   ← 差距大，测试很差！
\`\`\`

**解决过拟合：** 获取更多数据、正则化(L1/L2/Dropout)、降低模型复杂度

**解决欠拟合：** 使用更复杂的模型、增加特征、训练更长时间

---

### Q5 ★★☆☆☆ — 参数和超参数有什么区别？

| | 参数 | 超参数 |
|--|------|--------|
| **定义** | 从数据中学习 | 训练前由你设置 |
| **线性回归示例** | 权重w、偏置b | 学习率、正则化强度 |
| **决策树示例** | 分裂特征和阈值 | 最大深度、最少样本数 |
| **如何确定** | 梯度下降（自动） | 网格搜索/手动（你的任务！） |

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

# 超参数（你来设置）：
model = RandomForestClassifier(
    n_estimators=100,   # 超参数：树的数量
    max_depth=5,        # 超参数：树的深度
)

# 参数（模型从数据中学习）：
model.fit(X_train, y_train)
# 每棵树的分裂条件是从数据中学习得到的
\`\`\``
  }
}

// ─── 4. Data Handling ────────────────────────────────────────────────────────
export const interviewDataHandling: TopicContent = {
  id: 'interview-data-handling',
  title: { en: 'Interview: Data Cleaning & Preprocessing', zh: '面试：数据清洗与预处理' },
  contentType: 'article',
  content: {
    en: `## Data Cleaning & Preprocessing — Interview Questions

> "Garbage in, garbage out." 80% of real ML work is data preparation.

---

### Q1 ★☆☆☆☆ — How do you handle missing values?

**Step 1: Understand why values are missing**

| Type | Meaning | Example |
|------|---------|---------|
| **MCAR** (Missing Completely At Random) | No pattern | Random survey non-response |
| **MAR** (Missing At Random) | Pattern depends on other columns | Young users skip income field |
| **MNAR** (Missing Not At Random) | Pattern depends on the missing value itself | Very sick patients missing health score |

**Step 2: Choose a strategy**

\`\`\`python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({'age': [25, None, 30, None, 45],
                   'salary': [50000, 60000, None, 55000, 80000]})

# Option 1: Drop rows with missing values (only if <5% missing)
df.dropna()

# Option 2: Fill with mean/median/mode
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Option 3: Sklearn imputer (use median for robustness)
imputer = SimpleImputer(strategy='median')
df_filled = imputer.fit_transform(df)

# Option 4: Add indicator column (tells model "this was missing")
df['age_was_missing'] = df['age'].isna().astype(int)
\`\`\`

**Rule of thumb:**
- < 5% missing: drop rows or mean impute
- 5–30% missing: median impute or add indicator column
- > 30% missing: consider dropping the entire feature

---

### Q2 ★★☆☆☆ — How do you detect and handle outliers?

**Detection methods:**

\`\`\`python
import numpy as np

data = np.array([10, 12, 11, 14, 13, 100, 12, 11])  # 100 is an outlier

# Method 1: IQR (Interquartile Range) — robust method
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = data[(data < lower) | (data > upper)]
print(f"Outliers: {outliers}")   # [100]

# Method 2: Z-score — assumes normal distribution
from scipy import stats
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]   # beyond 3 standard deviations
\`\`\`

**What to do with outliers:**

| Action | When |
|--------|------|
| **Remove** | Clearly erroneous (sensor error, data entry mistake) |
| **Cap (Winsorize)** | Extreme but valid values, don't want to lose data |
| **Transform (log)** | Right-skewed data like income, prices |
| **Keep** | Outlier IS the signal (fraud detection!) |

\`\`\`python
# Log transform for skewed data (e.g., income)
import numpy as np
df['log_income'] = np.log1p(df['income'])  # log1p = log(1+x), handles 0
\`\`\`

---

### Q3 ★★☆☆☆ — What is an imbalanced dataset? How do you handle it?

**Problem:** If 99% of transactions are normal and 1% are fraud, a model that always predicts "normal" gets 99% accuracy but is useless!

\`\`\`python
from sklearn.datasets import make_classification
import numpy as np

# Check class balance
print(pd.Series(y_train).value_counts())
# Output: 0 → 9900 (normal)
#         1 → 100  (fraud) ← very imbalanced!
\`\`\`

**Solutions:**

\`\`\`python
# 1. Oversample minority class (duplicate fraud cases)
from imblearn.over_sampling import RandomOverSampler, SMOTE

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# SMOTE: creates synthetic minority samples (better than duplication)
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# 2. Undersample majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

# 3. Use class weights (tell model to penalize minority errors more)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# 4. Use the right metric! (NOT accuracy)
# → Use: Precision, Recall, F1, AUC-ROC instead
\`\`\`

---

### Q4 ★★☆☆☆ — What is the difference between normalization and standardization?

| | Normalization (Min-Max) | Standardization (Z-score) |
|--|------------------------|--------------------------|
| **Formula** | x' = (x - min) / (max - min) | x' = (x - μ) / σ |
| **Range** | [0, 1] | mean=0, std=1 (no fixed range) |
| **Use when** | Bounded range, neural networks, KNN | Unknown range, linear models, SVM |
| **Sensitive to outliers?** | YES (outlier becomes 1.0, compresses rest) | Less so (outlier still extreme but doesn't change 0-1 range) |

\`\`\`python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalization
mm = MinMaxScaler()
X_norm = mm.fit_transform(X_train)    # fit on train, apply to test!

# Standardization
ss = StandardScaler()
X_std = ss.fit_transform(X_train)

# CRITICAL: fit only on train data, then transform test
X_test_norm = mm.transform(X_test)    # use train's min/max
X_test_std  = ss.transform(X_test)    # use train's mean/std
\`\`\`

**Why must you fit only on training data?** Using test statistics "leaks" test information into the model, making evaluation invalid.

---

### Q5 ★★☆☆☆ — What is exploratory data analysis (EDA)? What do you look for?

**EDA checklist before any ML project:**

\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

# 1. Shape and types
print(df.shape)         # (rows, columns)
print(df.dtypes)        # check for wrong types (e.g., numeric stored as object)
print(df.head())

# 2. Missing values
print(df.isnull().sum())
print(df.isnull().mean() * 100)   # % missing per column

# 3. Basic statistics
print(df.describe())   # count, mean, std, min, 25%, 50%, 75%, max

# 4. Target distribution
df['target'].value_counts(normalize=True).plot(kind='bar')

# 5. Correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# 6. Outliers
df.boxplot()
\`\`\`

**Key things to look for:**
- Class imbalance in target variable
- Highly correlated features (multicollinearity)
- Features with very high % missing
- Unexpected distributions (bimodal, extreme skew)
- Features that "leak" the answer (recorded after the event you're predicting)`,

    zh: `## 数据清洗与预处理面试题

> "垃圾进，垃圾出。" 真实ML工作中80%都是数据准备。

---

### Q1 ★☆☆☆☆ — 如何处理缺失值？

**第一步：理解为什么会有缺失值**

| 类型 | 含义 | 示例 |
|------|------|------|
| **MCAR**（完全随机缺失） | 无规律 | 随机调查无响应 |
| **MAR**（随机缺失） | 规律取决于其他列 | 年轻用户不填收入 |
| **MNAR**（非随机缺失） | 规律取决于缺失值本身 | 病重患者缺少健康评分 |

**第二步：选择处理策略**

\`\`\`python
# 方案1：删除有缺失值的行（只适用于<5%缺失时）
df.dropna()

# 方案2：用均值/中位数/众数填充
df['age'].fillna(df['age'].median(), inplace=True)

# 方案3：添加指示列（告诉模型"这个值缺失了"）
df['age_was_missing'] = df['age'].isna().astype(int)
\`\`\`

**经验规则：**
- 缺失<5%：删除行或用均值填充
- 缺失5-30%：用中位数填充或添加指示列
- 缺失>30%：考虑删除整个特征

---

### Q2 ★★☆☆☆ — 如何检测和处理异常值？

\`\`\`python
# 方法1：IQR（四分位距）——鲁棒方法
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = data[(data < lower) | (data > upper)]

# 方法2：Z分数——假设正态分布
from scipy import stats
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]
\`\`\`

**如何处理异常值：**
- **删除**：明显错误的数据（传感器故障、录入错误）
- **截断**：极端但有效的值
- **对数变换**：偏斜数据（如收入、价格）
- **保留**：异常值本身就是信号（欺诈检测！）

---

### Q3 ★★☆☆☆ — 什么是不平衡数据集？如何处理？

**问题：** 如果99%交易是正常的，1%是欺诈，一个总是预测"正常"的模型准确率99%，但完全没用！

**解决方案：**
- **过采样少数类**：SMOTE生成合成少数样本（比简单复制更好）
- **欠采样多数类**：减少多数类样本数量
- **使用类别权重**：\`class_weight='balanced'\`，让模型对少数类错误惩罚更重
- **使用正确的评估指标**：不要用准确率！用精确率、召回率、F1、AUC-ROC

---

### Q4 ★★☆☆☆ — 归一化和标准化有什么区别？

| | 归一化（Min-Max） | 标准化（Z分数） |
|--|-----------------|----------------|
| **公式** | x' = (x-min)/(max-min) | x' = (x-μ)/σ |
| **范围** | [0, 1] | 均值=0，标准差=1 |
| **适用** | 有界范围、神经网络、KNN | 未知范围、线性模型、SVM |
| **对异常值敏感？** | 是（异常值变成1.0） | 较不敏感 |

**关键：** 只在训练数据上fit，然后用训练数据的统计量变换测试数据！

---

### Q5 ★★☆☆☆ — 什么是探索性数据分析（EDA）？你会关注哪些方面？

**开始任何ML项目前的EDA清单：**
1. 数据形状和类型（\`.shape\`, \`.dtypes\`）
2. 缺失值统计（\`.isnull().sum()\`）
3. 基本统计描述（\`.describe()\`）
4. 目标变量分布（类别是否平衡？）
5. 特征相关性（热力图）
6. 异常值检测（箱线图）

**重点关注：**
- 目标变量的类别不平衡
- 高度相关的特征（多重共线性）
- 缺失比例很高的特征
- 数据泄漏（在你预测的事件之后记录的特征）`
  }
}

// ─── 5. Linear Models ────────────────────────────────────────────────────────
export const interviewLinearModels: TopicContent = {
  id: 'interview-linear-models',
  title: { en: 'Interview: Linear & Logistic Regression', zh: '面试：线性与逻辑回归' },
  contentType: 'article',
  content: {
    en: `## Linear & Logistic Regression — Interview Questions

> The building blocks. If you understand these deeply, you understand 80% of ML.

---

### Q1 ★☆☆☆☆ — Explain linear regression in simple terms.

**Idea:** Fit a straight line through data points that minimizes prediction errors.

\`\`\`
House price = w1 × size + w2 × bedrooms + b

        Price
    $500k |          /
    $400k |        /   •  •
    $300k |      / •
    $200k |    /•
    $100k |  /
          |______________ Size
           500  1000  1500  2000 sqft
\`\`\`

**The math:**
\`\`\`
y_pred = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
       = w·x + b   (dot product notation)
\`\`\`

\`\`\`python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1000], [1500], [2000], [2500]])  # house sizes
y = np.array([200000, 300000, 400000, 500000])  # prices

model = LinearRegression()
model.fit(X, y)

print(f"Slope (w): {model.coef_[0]:.0f}")    # 200 ($/sqft)
print(f"Intercept (b): {model.intercept_:.0f}")  # ~0

print(model.predict([[1800]]))   # ~360,000
\`\`\`

---

### Q2 ★★☆☆☆ — What is the cost function? How do we minimize it?

**Cost function (MSE — Mean Squared Error):**
\`\`\`
L = (1/n) × Σ (y_true - y_pred)²

Why square? → Penalizes large errors more, makes math easier (differentiable)
Why not absolute value? → |error| has a kink at 0, hard to differentiate
\`\`\`

**Gradient descent to minimize L:**
\`\`\`
Start with random weights → compute loss → compute gradient →
update weights → repeat until loss stops decreasing

w := w - α × ∂L/∂w    (α = learning rate, e.g. 0.01)
\`\`\`

\`\`\`python
# Manual gradient descent for linear regression
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)  # y = 2x

w, b = 0.0, 0.0    # start with zeros
lr = 0.01

for epoch in range(1000):
    y_pred = w * X + b
    loss = np.mean((y - y_pred) ** 2)

    # Gradients
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # Update
    w -= lr * dw
    b -= lr * db

print(f"w={w:.3f}, b={b:.3f}")  # w≈2.0, b≈0.0
\`\`\`

---

### Q3 ★★☆☆☆ — How does logistic regression differ from linear regression?

**Problem with linear regression for classification:**
\`\`\`
Linear: y = wx + b   → can output any number: -∞ to +∞
For spam (0 or 1), we need: output between 0 and 1
\`\`\`

**Solution: Sigmoid function** squashes any number to (0, 1):
\`\`\`
σ(z) = 1 / (1 + e^(-z))

z=-10 → σ ≈ 0.00005  (almost 0)
z=0   → σ = 0.5      (50% probability)
z=+10 → σ ≈ 0.99995  (almost 1)
\`\`\`

**Logistic Regression:**
\`\`\`
z = w·x + b            (linear combination, like linear regression)
P(y=1) = σ(z)          (probability via sigmoid)
predict = 1 if P(y=1) ≥ 0.5 else 0
\`\`\`

\`\`\`python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)   # [[0.3, 0.7], ...] = P(class0), P(class1)
preds = model.predict(X_test)         # [0, 1, 1, 0, ...]
\`\`\`

---

### Q4 ★★☆☆☆ — What is regularization? Explain L1 vs L2.

**Why regularization?** Without it, models overfit — they make weights very large to perfectly fit training noise.

**Intuition:** Add a penalty for complexity. Force the model to be simpler.

\`\`\`
L1 (Lasso):   Loss = MSE + λ × Σ|wᵢ|     ← penalty on absolute values
L2 (Ridge):   Loss = MSE + λ × Σwᵢ²      ← penalty on squared values
\`\`\`

| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| **Effect on weights** | Drives many to exactly 0 (sparse) | Shrinks all weights toward 0 (small but not zero) |
| **Feature selection** | Yes (zero weights = removed features) | No (keeps all features) |
| **Use when** | Many irrelevant features | All features somewhat relevant |

\`\`\`python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)    # alpha = λ, larger = more regularization
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Lasso may have many zero coefficients (sparse model)
print(lasso.coef_)
\`\`\`

---

### Q5 ★★★☆☆ — What assumptions does linear regression make?

1. **Linearity:** Relationship between features and target is linear
2. **Independence:** Observations are independent (no time series autocorrelation)
3. **Homoscedasticity:** Residuals have constant variance (don't fan out)
4. **Normality:** Residuals are normally distributed (for inference, not prediction)
5. **No multicollinearity:** Features are not highly correlated with each other

\`\`\`python
import pandas as pd
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
residuals = y_train - model.predict(X_train)

# Check normality of residuals
import matplotlib.pyplot as plt
plt.hist(residuals, bins=30)   # Should look like a bell curve

# Check homoscedasticity
plt.scatter(model.predict(X_train), residuals)  # Should be random (no pattern)
\`\`\``,

    zh: `## 线性与逻辑回归面试题

> 最基础的积木。深刻理解这些，你就理解了80%的ML。

---

### Q1 ★☆☆☆☆ — 用简单的话解释线性回归。

**思想：** 通过数据点拟合一条直线，最小化预测误差。

\`\`\`
房价 = w1 × 面积 + w2 × 卧室数 + b
\`\`\`

**数学：**
\`\`\`
y_pred = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w·x + b
\`\`\`

\`\`\`python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1000], [1500], [2000], [2500]])  # 房屋面积
y = np.array([200000, 300000, 400000, 500000])  # 价格

model = LinearRegression()
model.fit(X, y)
print(model.predict([[1800]]))   # ~360,000
\`\`\`

---

### Q2 ★★☆☆☆ — 什么是损失函数？如何最小化它？

**均方误差（MSE）：**
\`\`\`
L = (1/n) × Σ (y真实 - y预测)²

为什么平方？→ 对大误差惩罚更重，数学上更容易求导
\`\`\`

**梯度下降最小化损失：**
\`\`\`
从随机权重开始 → 计算损失 → 计算梯度 → 更新权重 → 重复直到损失不再下降

w := w - α × ∂L/∂w    （α=学习率，如0.01）
\`\`\`

---

### Q3 ★★☆☆☆ — 逻辑回归与线性回归有什么不同？

**线性回归用于分类的问题：**
- 输出可以是任意数字（-∞到+∞）
- 但分类需要输出0到1之间的概率

**解决方案：Sigmoid函数** 将任意数压缩到(0,1)：
\`\`\`
σ(z) = 1 / (1 + e^(-z))
z=0 → σ=0.5（50%概率）
\`\`\`

**逻辑回归：**
\`\`\`
z = w·x + b         （线性组合）
P(y=1) = σ(z)       （通过sigmoid得到概率）
预测 = 1 如果P(y=1) ≥ 0.5 否则 0
\`\`\`

---

### Q4 ★★☆☆☆ — 什么是正则化？L1 vs L2有什么区别？

**为什么需要正则化？** 没有正则化，模型会过拟合——权重变得非常大以完美拟合训练噪声。

\`\`\`
L1 (Lasso):  损失 = MSE + λ × Σ|wᵢ|    ← 惩罚绝对值
L2 (Ridge):  损失 = MSE + λ × Σwᵢ²     ← 惩罚平方值
\`\`\`

| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| **对权重的影响** | 使许多权重精确为0（稀疏） | 将所有权重缩小趋向0（但不为零） |
| **特征选择** | 是（零权重=删除特征） | 否（保留所有特征） |
| **适用场景** | 有大量无关特征时 | 所有特征都有一定相关性时 |

---

### Q5 ★★★☆☆ — 线性回归有哪些假设？

1. **线性性：** 特征与目标之间是线性关系
2. **独立性：** 观测值相互独立（无时序自相关）
3. **同方差性：** 残差方差恒定
4. **正态性：** 残差服从正态分布
5. **无多重共线性：** 特征之间不高度相关`
  }
}

// ─── 6. Decision Trees & Forests ────────────────────────────────────────────
export const interviewDecisionTrees: TopicContent = {
  id: 'interview-decision-trees',
  title: { en: 'Interview: Decision Trees & Random Forest', zh: '面试：决策树与随机森林' },
  contentType: 'article',
  content: {
    en: `## Decision Trees & Random Forest — Interview Questions

> Trees are the most intuitive ML models. Forests are powerful ensembles of trees.

---

### Q1 ★☆☆☆☆ — How does a decision tree work? Explain with an example.

**Idea:** Ask yes/no questions about features to split data into groups.

\`\`\`
Should I play tennis today?

           Outlook?
          /    |    \\
      Sunny  Cloudy  Rainy
        |       |       |
    Humidity?  YES  Wind?
    /      \\        /    \\
 High    Normal  Strong  Weak
  |         |      |      |
  NO        YES    NO    YES
\`\`\`

Each internal node = a question about a feature
Each leaf node = a prediction (Yes/No, or a number for regression)
Each branch = an answer to the question

\`\`\`python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# See the rules the tree learned
print(export_text(model, feature_names=feature_names))
\`\`\`

---

### Q2 ★★☆☆☆ — How does a tree choose which feature to split on? What is entropy?

**Goal:** Each split should make the resulting groups as "pure" as possible (one class dominates).

**Entropy (measure of impurity):**
\`\`\`
H = -Σ p_i × log₂(p_i)

Perfect node (all same class): H = 0    (no uncertainty)
Worst node (50-50 split):      H = 1    (maximum uncertainty)
\`\`\`

**Information Gain = entropy before split - weighted entropy after split**
\`\`\`
Parent: [5 cats, 5 dogs]  → H = 1.0 (maximum uncertainty)

Split on "has whiskers?":
  Yes: [5 cats, 0 dogs] → H = 0.0 (perfect purity!)
  No:  [0 cats, 5 dogs] → H = 0.0 (perfect purity!)

Information Gain = 1.0 - (0.5×0.0 + 0.5×0.0) = 1.0  ← best possible!
\`\`\`

**Gini impurity** (alternative, used by sklearn default):
\`\`\`
Gini = 1 - Σ p_i²

50-50 split: 1 - (0.5² + 0.5²) = 0.5    (worst)
All same class: 1 - (1² + 0²) = 0.0      (best)
\`\`\`

---

### Q3 ★★☆☆☆ — How do you prevent a decision tree from overfitting?

**The problem:** A tree with no limits will grow until every leaf has exactly one sample — 100% training accuracy, terrible generalization.

**Pre-pruning (limit during growth):**

\`\`\`python
from sklearn.tree import DecisionTreeClassifier

# Limit depth
model = DecisionTreeClassifier(max_depth=5)

# Minimum samples needed to make a split
model = DecisionTreeClassifier(min_samples_split=20)

# Minimum samples in a leaf
model = DecisionTreeClassifier(min_samples_leaf=10)

# Best combination to try:
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=5,
    random_state=42
)
\`\`\`

**Post-pruning:** Grow full tree, then remove branches that don't improve validation performance.

**Visually recognizing overfitting:**
\`\`\`
max_depth=1:   Train=60%, Test=59%   ← underfitting
max_depth=5:   Train=88%, Test=86%   ← just right
max_depth=20:  Train=99%, Test=72%   ← overfitting
\`\`\`

---

### Q4 ★★☆☆☆ — What is Random Forest and why is it better than one tree?

**Problem with a single tree:** Small change in data → completely different tree. Very high variance.

**Random Forest = Many trees, each trained differently:**

\`\`\`
Step 1 (Bootstrap): For each tree, sample n rows WITH replacement
         Tree 1 sees: [rows 1,3,3,7,12,...]
         Tree 2 sees: [rows 2,5,8,8,11,...]
         Tree 3 sees: [rows 4,6,9,10,10,...]

Step 2 (Feature Subspace): Each split considers only √d random features
         (prevents all trees from splitting on same strong feature)

Step 3 (Majority Vote): Final prediction = most common prediction
         Tree 1: cat, Tree 2: dog, Tree 3: cat → Final: cat
\`\`\`

**Why it works:** Each tree makes different errors. When we average/vote, errors cancel out!

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # 100 trees
    max_features='sqrt', # √(num_features) per split
    bootstrap=True,      # sample with replacement
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance (which features matter most?)
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh')
\`\`\`

---

### Q5 ★★★☆☆ — What is gradient boosting? How is it different from Random Forest?

| | Random Forest | Gradient Boosting |
|--|--------------|------------------|
| **Build style** | Trees in **parallel** (independent) | Trees in **sequence** (each fixes previous errors) |
| **Combination** | Vote / average | Weighted sum |
| **Training speed** | Fast | Slow (sequential) |
| **Prone to overfitting** | Less | More (needs careful tuning) |
| **Libraries** | sklearn RF | XGBoost, LightGBM, CatBoost |

**Gradient Boosting intuition:**
\`\`\`
Tree 1: predicts [100, 200, 150]  (actual: [120, 180, 160])
Residuals:       [ +20,  -20,  +10]  ← errors

Tree 2: trained to predict RESIDUALS → [-20, +20, -10]... etc.
Final prediction = Tree1 + Tree2 + ... (sum of all corrections)
\`\`\`

\`\`\`python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,   # how much each tree contributes
    max_depth=3,          # shallow trees are better in boosting
    subsample=0.8,        # use 80% of rows per tree
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          early_stopping_rounds=10, verbose=False)
\`\`\``,

    zh: `## 决策树与随机森林面试题

> 树是最直观的ML模型，随机森林是树的强大集成。

---

### Q1 ★☆☆☆☆ — 决策树是如何工作的？举例说明。

**思想：** 通过对特征提问yes/no来将数据分成不同组。

\`\`\`
今天该打网球吗？

           天气如何？
          /    |    \\
      晴天   多云   雨天
        |       |       |
    湿度如何？  是   风大吗？
    /      \\        /    \\
   高     正常   强风   微风
   |         |      |      |
  否        是    否    是
\`\`\`

每个内部节点 = 关于特征的问题
每个叶子节点 = 预测结果
每条分支 = 问题的回答

---

### Q2 ★★☆☆☆ — 决策树如何选择分裂特征？什么是熵？

**目标：** 每次分裂后，子节点应该尽可能"纯净"（一个类别占主导）。

**熵（不纯度的度量）：**
\`\`\`
H = -Σ p_i × log₂(p_i)

完美节点（全是同一类）：H = 0（无不确定性）
最差节点（50-50分布）：H = 1（最大不确定性）
\`\`\`

**信息增益 = 分裂前熵 - 分裂后加权熵**
\`\`\`
父节点：[5只猫, 5只狗] → H = 1.0
按"有胡须？"分裂：
  是：[5猫, 0狗] → H = 0.0 ← 完全纯净！
  否：[0猫, 5狗] → H = 0.0 ← 完全纯净！
信息增益 = 1.0 ← 最好的可能！
\`\`\`

---

### Q3 ★★☆☆☆ — 如何防止决策树过拟合？

**问题：** 不加限制的树会一直生长，直到每个叶子只有一个样本——训练准确率100%，泛化能力极差。

\`\`\`python
# 预剪枝：在生长过程中限制
model = DecisionTreeClassifier(
    max_depth=5,            # 限制深度
    min_samples_split=20,   # 分裂需要的最少样本数
    min_samples_leaf=5,     # 叶节点最少样本数
)
\`\`\`

---

### Q4 ★★☆☆☆ — 什么是随机森林？为什么它比单棵树更好？

**随机森林 = 多棵不同的树：**

\`\`\`
步骤1（自助采样）：每棵树对行有放回地随机采样
步骤2（特征随机）：每次分裂只考虑√d个随机特征
步骤3（多数投票）：最终预测 = 多棵树的最多数预测
\`\`\`

**为什么有效：** 每棵树犯不同的错误，平均/投票后错误相互抵消！

---

### Q5 ★★★☆☆ — 什么是梯度提升？它与随机森林有什么不同？

| | 随机森林 | 梯度提升 |
|--|---------|---------|
| **构建方式** | **并行**（独立） | **串行**（每棵修正前一棵的错误） |
| **组合方式** | 投票/平均 | 加权求和 |
| **训练速度** | 快 | 慢（串行） |
| **代表库** | sklearn RF | XGBoost, LightGBM |

**梯度提升的直觉：**
\`\`\`
树1：预测 [100, 200, 150]（实际：[120, 180, 160]）
残差：     [ +20,  -20,  +10] ← 错误

树2：训练来预测残差
最终预测 = 树1 + 树2 + ...（所有修正的累积）
\`\`\``
  }
}

// ─── 7. Model Validation ─────────────────────────────────────────────────────
export const interviewModelValidation: TopicContent = {
  id: 'interview-model-validation',
  title: { en: 'Interview: Model Validation & Evaluation', zh: '面试：模型验证与评估' },
  contentType: 'article',
  content: {
    en: `## Model Validation & Evaluation — Interview Questions

> Knowing how to correctly measure model performance is as important as building the model.

---

### Q1 ★☆☆☆☆ — What is a confusion matrix? Walk through it.

\`\`\`
Actual \\ Predicted    Positive    Negative
   Positive           TP          FN
   Negative           FP          TN

TP (True Positive):  Model says YES, reality is YES  ✅
TN (True Negative):  Model says NO,  reality is NO   ✅
FP (False Positive): Model says YES, reality is NO   ❌ (Type I error)
FN (False Negative): Model says NO,  reality is YES  ❌ (Type II error)
\`\`\`

**Example — Disease detection:**
\`\`\`
                    Predicted Sick  Predicted Healthy
Actually Sick            90              10        ← 10 missed (FN = dangerous!)
Actually Healthy          5             895
\`\`\`

\`\`\`python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
\`\`\`

---

### Q2 ★★☆☆☆ — What is precision vs recall? When do you prioritize each?

\`\`\`
Precision = TP / (TP + FP)  ← "When I say positive, how often am I right?"
Recall    = TP / (TP + FN)  ← "Of all actual positives, how many did I catch?"

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)  ← harmonic mean
\`\`\`

**When to prioritize:**

| Scenario | Priority | Reason |
|---------|---------|--------|
| Cancer detection | **High Recall** | Missing a sick patient (FN) is catastrophic |
| Email spam filter | **High Precision** | Blocking real email (FP) is very annoying |
| Fraud detection | **F1** or **Recall** | Miss a fraud (FN) = money lost |
| Recommendation system | **Precision** | Bad recommendations (FP) hurt user experience |

\`\`\`python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(classification_report(y_test, y_pred))
# Output shows precision, recall, F1 per class
\`\`\`

---

### Q3 ★★☆☆☆ — What is an ROC curve and AUC? How do you interpret it?

**ROC curve:** Plot True Positive Rate (Recall) vs False Positive Rate at every possible threshold.

\`\`\`
TPR (y-axis) = TP / (TP + FN) = Recall
FPR (x-axis) = FP / (FP + TN) = 1 - Specificity

    TPR
1.0 |___________
    |          /|
0.8 |         / |
0.6 |        /  |
0.4 |       /   |
0.2 |      /    |  AUC = area under curve
0.0 |─────/─────|
    0   0.5    1.0  FPR
        (random)
\`\`\`

**AUC interpretation:**
- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random guessing (diagonal line)
- AUC = 0.7–0.8 → good
- AUC = 0.8–0.9 → very good
- AUC > 0.9 → excellent

**Key advantage of AUC:** Works well for imbalanced datasets and doesn't depend on threshold choice.

\`\`\`python
from sklearn.metrics import roc_auc_score, roc_curve

probs = model.predict_proba(X_test)[:, 1]  # probability of positive class
auc = roc_auc_score(y_test, probs)
print(f"AUC: {auc:.3f}")

fpr, tpr, thresholds = roc_curve(y_test, probs)
\`\`\`

---

### Q4 ★★☆☆☆ — What is k-fold cross-validation and why use it?

**Problem with a single train/test split:** If you happen to get a lucky (or unlucky) split, your evaluation is misleading.

**K-Fold solution:** Repeat the evaluation k times with different splits:
\`\`\`
k=5 folds:
Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]

Final score = mean of 5 scores (+ standard deviation = reliability)
\`\`\`

\`\`\`python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
# Mean: 0.867 ± 0.023  (low std = consistent performance = trustworthy)
\`\`\`

**Stratified K-Fold:** For imbalanced datasets, ensure each fold has the same class ratio.

---

### Q5 ★★★☆☆ — How do you tune hyperparameters? Compare grid search vs random search.

**Grid Search:** Try every combination of specified values.

\`\`\`python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
# Total: 3×3×3 = 27 combinations × 5 folds = 135 model fits

gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_)
\`\`\`

**Random Search:** Randomly sample combinations.

\`\`\`python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),   # any integer in range
    'max_depth': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

rs = RandomizedSearchCV(RandomForestClassifier(), param_dist,
                        n_iter=50, cv=5, n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
\`\`\`

| | Grid Search | Random Search |
|--|------------|--------------|
| **Coverage** | Exhaustive (all combos) | Stochastic (random sample) |
| **Speed** | Slow (exponential with params) | Fast (fixed n_iter) |
| **Best for** | Few parameters, small ranges | Many parameters, wide ranges |
| **Find optimal?** | Guaranteed (within grid) | Probabilistic — often close enough |`,

    zh: `## 模型验证与评估面试题

> 正确衡量模型性能与构建模型同等重要。

---

### Q1 ★☆☆☆☆ — 什么是混淆矩阵？请逐一讲解。

\`\`\`
         预测正类  预测负类
实际正类    TP       FN
实际负类    FP       TN

TP：预测是，实际是  ✅
TN：预测否，实际否  ✅
FP：预测是，实际否  ❌（I类错误）
FN：预测否，实际是  ❌（II类错误）
\`\`\`

---

### Q2 ★★☆☆☆ — 什么是精确率与召回率？什么时候优先考虑哪个？

\`\`\`
精确率 = TP / (TP + FP)  ← "我说的阳性，有多少是对的？"
召回率 = TP / (TP + FN)  ← "所有实际阳性，我抓住了多少？"
F1分数 = 2 × (精确率 × 召回率) / (精确率 + 召回率)
\`\`\`

**何时优先考虑哪个：**

| 场景 | 优先 | 原因 |
|------|------|------|
| 癌症检测 | **高召回率** | 漏诊（FN）是灾难性的 |
| 垃圾邮件过滤 | **高精确率** | 误拦正常邮件（FP）很烦人 |
| 欺诈检测 | **F1或召回率** | 漏检欺诈（FN）=损失金钱 |

---

### Q3 ★★☆☆☆ — 什么是ROC曲线和AUC？如何解读？

**ROC曲线：** 在所有可能阈值下，绘制真正率（召回率）vs假正率的图。

**AUC解读：**
- AUC = 1.0 → 完美分类器
- AUC = 0.5 → 随机猜测（对角线）
- AUC = 0.7-0.8 → 良好
- AUC > 0.9 → 优秀

**关键优势：** 对不平衡数据集效果好，不依赖阈值选择。

---

### Q4 ★★☆☆☆ — 什么是k折交叉验证？为什么要用它？

**问题：** 单次训练/测试分割可能因为运气好坏而产生误导性评估。

**k折解决方案：** 重复评估k次，每次使用不同的分割：
\`\`\`
最终得分 = k次得分的均值（+标准差=可靠性）
\`\`\`

\`\`\`python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"均值：{scores.mean():.3f} ± {scores.std():.3f}")
\`\`\`

**分层k折：** 对不平衡数据集，确保每折的类别比例相同。

---

### Q5 ★★★☆☆ — 如何调优超参数？网格搜索vs随机搜索的比较。

| | 网格搜索 | 随机搜索 |
|--|---------|---------|
| **覆盖** | 穷举（所有组合） | 随机采样 |
| **速度** | 慢（参数指数级增长） | 快（固定迭代次数） |
| **适用** | 参数少、范围小 | 参数多、范围大 |

\`\`\`python
from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(model, param_dist,
                        n_iter=50, cv=5, n_jobs=-1)
rs.fit(X_train, y_train)
print(rs.best_params_)
\`\`\``
  }
}
