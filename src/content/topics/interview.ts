import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Feature Engineering
// Source: 百面机器学习 Ch.1 + 机器学习面试题 Ch.6
// ─────────────────────────────────────────────────────────────────────────────
export const interviewFeatureEngineering: TopicContent = {
  id: 'interview-feature-engineering',
  title: { en: 'Interview: Feature Engineering', zh: '面试：特征工程' },
  contentType: 'article',
  content: {
    en: `## Feature Engineering Interview Questions

> Source: *百面机器学习* Ch.1 + *机器学习面试题* Ch.6
> Covers the most common feature engineering questions asked in ML engineering interviews.

---

### Q1 ★☆☆☆☆ — Why do we normalize numerical features?

**Answer:** Normalization brings all features onto the same scale so that no single feature dominates due to its magnitude. Two main methods:

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Min-Max Scaling** | x' = (x − x_min) / (x_max − x_min) → [0, 1] | Known bounded range |
| **Z-Score (Standardization)** | x' = (x − μ) / σ → mean 0, std 1 | Unknown / unbounded range |

**Key insight:** Gradient-descent-based models (linear regression, logistic regression, SVM, neural nets) converge much faster when features are on the same scale. The loss surface becomes more circular instead of elongated.

**Decision trees do NOT need normalization** — they split on thresholds and are invariant to monotonic transformations.

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # use train stats only!

# Min-max normalization
mm = MinMaxScaler()
X_mm = mm.fit_transform(X_train)
\`\`\`

---

### Q2 ★★☆☆☆ — How do you handle categorical features?

**Answer:** Three standard encodings:

| Encoding | Use Case | Output Size |
|----------|---------|------------|
| **Ordinal Encoding** | Ordered categories (Low < Med < High) | 1 integer column |
| **One-Hot Encoding** | Unordered categories with few values | k binary columns |
| **Binary Encoding** | Unordered with many values | ⌈log₂k⌉ binary columns |

**One-hot pitfalls for high-cardinality features:**
- High dimensionality → curse of dimensionality in KNN, overfitting in LR
- Sparse vectors → use sparse matrix representation
- Solution: combine with **feature selection** or use **embedding layers**

\`\`\`python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# One-hot (drops first to avoid multicollinearity)
ohe = OneHotEncoder(drop='first', sparse_output=True)
X_enc = ohe.fit_transform(X[['blood_type']])

# Pandas shortcut
pd.get_dummies(df, columns=['blood_type'], drop_first=True)
\`\`\`

---

### Q3 ★★☆☆☆ — What are combinatorial features and why use them?

**Answer:** Combinatorial (interaction) features are cross-products of two or more raw features that capture joint effects invisible to first-order models.

**Example — Ad Click Prediction:**

| Language | Type | Click |
|----------|------|-------|
| Chinese | Movie | 0 |
| English | Movie | 1 |
| Chinese | Drama | 1 |
| English | Drama | 0 |

A linear model sees Language and Type independently. But the *combination* (Language × Type) is the real signal. The 2nd-order feature "English × Movie" → click = 1.

**Challenge with high-dimensional combinations:** If feature A has m values and B has n values, A×B has m×n dimensions — explodes quickly.

**Solutions:**
- **Factorization Machines (FM):** model pairwise interactions with low-rank embeddings in O(kd) time
- **Gradient Boosted Trees:** discover interactions automatically through splits
- **Deep Learning:** embedding layers learn compact joint representations

---

### Q4 ★★☆☆☆ — How do you represent text? Compare bag-of-words vs. Word2Vec vs. BERT.

**Answer:**

| Method | Representation | Captures Semantics? | Context-Aware? |
|--------|---------------|-------------------|----------------|
| **Bag of Words (BoW)** | Sparse count vector | No | No |
| **TF-IDF** | Weighted sparse vector | Partly (IDF down-weights common words) | No |
| **Word2Vec (CBOW/Skip-gram)** | Dense 100-300d embedding | Yes | No (static) |
| **GloVe** | Dense embedding from co-occurrence | Yes | No (static) |
| **BERT** | Contextual embedding (768d per token) | Yes | Yes |

**Word2Vec key insight:** Train a shallow neural net to predict neighbors (Skip-gram) or predict center word from context (CBOW). Learned weights capture analogies: *king − man + woman ≈ queen*.

**LDA (Latent Dirichlet Allocation):** Unlike Word2Vec, LDA models **documents as mixtures of topics** — useful for topic modeling and document clustering, not sequence understanding.

---

### Q5 ★★☆☆☆ — How do you handle insufficient image training data?

**Answer — Data Augmentation Techniques:**

| Technique | Description |
|-----------|-------------|
| **Geometric** | Flip, rotate, crop, scale, translate |
| **Color jitter** | Adjust brightness, contrast, saturation, hue |
| **Cutout / Erasing** | Randomly mask rectangular patches |
| **Mixup** | Blend two images and interpolate labels |
| **CutMix** | Cut a patch from one image and paste into another |

**Transfer Learning:** Fine-tune a pre-trained model (ResNet, EfficientNet, ViT) on your small dataset. The pre-trained backbone already knows low-level edges and textures.

\`\`\`python
import torchvision.transforms as T

train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
\`\`\`

---

### Q6 ★★★☆☆ — What is feature selection and how does it differ from feature extraction?

**Feature Selection:** Selects a *subset* of existing features (preserves interpretability).
- Filter: correlation, mutual information, chi-square
- Wrapper: RFE (Recursive Feature Elimination) with cross-validation
- Embedded: L1 regularization (Lasso), tree feature importance

**Feature Extraction:** Transforms features into a *new lower-dimensional* space (loses interpretability).
- PCA: maximizes variance, orthogonal components
- LDA: maximizes class separability
- Autoencoders: non-linear compression

\`\`\`python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

# Lasso embedded feature selection
lasso = Lasso(alpha=0.01).fit(X_train, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X_train)  # keeps non-zero Lasso features
\`\`\``,

    zh: `## 特征工程面试题

> 来源：《百面机器学习》第1章 + 《机器学习面试题》第6章
> 涵盖ML工程师面试中最常见的特征工程问题。

---

### Q1 ★☆☆☆☆ — 为什么需要对数值类型的特征做归一化？

**解答：** 归一化使所有特征处于相同的数值量级，避免某一特征因数值较大而主导模型。两种最常用的方法：

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| **线性归一化（Min-Max）** | x' = (x − x_min) / (x_max − x_min) → [0, 1] | 特征范围已知 |
| **零均值归一化（Z-Score）** | x' = (x − μ) / σ → 均值0，标准差1 | 特征范围未知/无界 |

**核心要点：** 使用梯度下降的模型（线性回归、逻辑回归、SVM、神经网络）在特征同量级时收敛更快——损失曲面从细长椭圆变为接近圆形。

**决策树不需要归一化** —— 它依赖信息增益分裂，对单调变换不变。

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # 只用训练集的统计量！
\`\`\`

---

### Q2 ★★☆☆☆ — 怎样处理类别型特征？

**解答：** 三种标准编码方式：

| 编码方式 | 适用场景 | 输出维度 |
|---------|---------|---------|
| **序号编码** | 有大小关系的类别（低<中<高）| 1个整数列 |
| **独热编码** | 无序类别，取值较少 | k个二进制列 |
| **二进制编码** | 无序类别，取值较多 | ⌈log₂k⌉个二进制列 |

**独热编码在高基数特征时的问题：**
- 维度爆炸 → KNN的维度灾难，逻辑回归易过拟合
- 稀疏向量 → 用稀疏矩阵存储
- 解决方案：结合**特征选择**或使用**嵌入层**

---

### Q3 ★★☆☆☆ — 什么是组合特征？如何处理高维组合特征？

**解答：** 组合特征是两个或多个原始特征的笛卡尔积，能捕捉一阶特征无法表达的联合效应。

**挑战：** 若特征A有m个取值、特征B有n个取值，A×B有m×n维 —— 规模快速爆炸。

**解决方案：**
- **因子分解机（FM）：** 用低秩嵌入建模特征交互，O(kd) 时间复杂度
- **GBDT：** 通过分裂自动发现交互
- **深度学习：** 嵌入层学习紧凑的联合表示

---

### Q4 ★★☆☆☆ — 有哪些文本表示模型？各有什么优缺点？

**解答：**

| 方法 | 表示形式 | 语义信息 | 上下文感知 |
|------|---------|---------|---------|
| **词袋模型（BoW）** | 稀疏计数向量 | 无 | 无 |
| **TF-IDF** | 加权稀疏向量 | 部分 | 无 |
| **Word2Vec** | 稠密100-300维嵌入 | 有 | 无（静态）|
| **BERT** | 上下文嵌入（768维/token）| 有 | 有 |

**Word2Vec核心：** 训练浅层神经网络预测邻居词（Skip-gram）或从上下文预测中心词（CBOW）。学到的权重捕捉类比关系：国王 − 男人 + 女人 ≈ 女王。

**LDA（隐狄利克雷分配）：** 将文档建模为话题的混合，适合话题建模，不适合序列理解。

---

### Q5 ★★☆☆☆ — 如何缓解图像分类任务中训练数据不足的问题？

**解答 —— 数据增强技术：**

| 技术 | 描述 |
|------|------|
| **几何变换** | 翻转、旋转、裁剪、缩放、平移 |
| **颜色抖动** | 调整亮度、对比度、饱和度、色调 |
| **Cutout / 随机擦除** | 随机遮挡矩形区域 |
| **Mixup** | 混合两张图像并插值标签 |
| **CutMix** | 将一张图的区域粘贴到另一张图上 |

**迁移学习：** 在预训练模型（ResNet、EfficientNet、ViT）上微调，预训练骨干网络已学会底层边缘和纹理特征。

---

### Q6 ★★★☆☆ — 特征选择与特征提取有何区别？

**特征选择：** 从现有特征中选取子集（保留可解释性）
- 过滤式：相关系数、互信息、卡方检验
- 包裹式：递归特征消除（RFE）
- 嵌入式：L1正则化（Lasso）、树模型特征重要性

**特征提取：** 将特征变换到新的低维空间（丢失可解释性）
- PCA：最大化方差，正交主成分
- LDA：最大化类间可分性
- 自编码器：非线性压缩`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Model Evaluation
// Source: 百面机器学习 Ch.2
// ─────────────────────────────────────────────────────────────────────────────
export const interviewModelEvaluation: TopicContent = {
  id: 'interview-model-evaluation',
  title: { en: 'Interview: Model Evaluation', zh: '面试：模型评估' },
  contentType: 'article',
  content: {
    en: `## Model Evaluation Interview Questions

> Source: *百面机器学习* Ch.2 + *机器学习面试题* Ch.5
> Critical evaluation concepts that appear in virtually every ML interview.

---

### Q1 ★☆☆☆☆ — What are the limitations of accuracy as a metric?

**Answer:** Accuracy = (TP + TN) / Total. It is **misleading on imbalanced datasets**.

**Classic trap:** A cancer screening model that always predicts "no cancer" achieves 99% accuracy on a dataset where only 1% of patients have cancer — yet it is completely useless.

**Better alternatives for imbalanced data:**
- **Precision & Recall** — focus on the positive class
- **F1 Score** — harmonic mean of precision and recall
- **ROC-AUC** — threshold-independent ranking quality
- **PR-AUC** — better for severe class imbalance

---

### Q2 ★☆☆☆☆ — Explain Precision, Recall, and the F1 Score.

**Answer:**

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actually Positive** | TP | FN |
| **Actually Negative** | FP | TN |

$$\\text{Precision} = \\frac{TP}{TP + FP} \\quad \\text{(of predicted positives, how many are real?)}$$

$$\\text{Recall} = \\frac{TP}{TP + FN} \\quad \\text{(of all real positives, how many did we catch?)}$$

$$\\text{F1} = \\frac{2 \\times P \\times R}{P + R} \\quad \\text{(harmonic mean)}$$

**Precision–Recall tradeoff:** Raising the classification threshold increases precision but decreases recall (and vice versa). Choose based on the cost of each error type:
- Medical diagnosis: prefer high recall (don't miss sick patients)
- Spam detection: prefer high precision (don't flag important emails)

---

### Q3 ★★☆☆☆ — What is the ROC curve and AUC? When would you prefer PR-AUC?

**Answer:**

**ROC (Receiver Operating Characteristic):** Plots TPR (Recall) vs. FPR at all thresholds.
- **AUC** = area under ROC = probability that the model ranks a random positive higher than a random negative
- AUC = 0.5 → random classifier; AUC = 1.0 → perfect

**How to draw ROC:** Sort predictions descending. For each threshold, compute (FPR, TPR) — plot the curve. Efficient O(n log n) algorithm using sort + cumulative TP/FP counts.

**When to prefer PR-AUC:**
- Severely imbalanced datasets (e.g., fraud detection: 1 in 10,000)
- ROC is optimistic under imbalance because TN is huge, keeping FPR low even for poor models
- PR curve focuses only on positive class performance

\`\`\`python
from sklearn.metrics import roc_auc_score, average_precision_score

auc     = roc_auc_score(y_true, y_scores)      # ROC-AUC
pr_auc  = average_precision_score(y_true, y_scores)  # PR-AUC
\`\`\`

---

### Q4 ★☆☆☆☆ — What is the "surprise" with RMSE?

**Answer:** RMSE = √(MSE). Its surprising property: **it penalizes large errors disproportionately** due to squaring.

Two models can have the same MAE but very different RMSE. Model A with a few large errors looks much worse under RMSE than Model B with many small errors. This is desirable when large errors are catastrophic (autonomous driving, medical dosing) but misleading for robust evaluation.

**Alternatives:**
- **MAE** — linear penalty, robust to outliers
- **Huber Loss** — quadratic for small errors, linear for large errors (best of both)

---

### Q5 ★☆☆☆☆ — What is cross-validation and why use it?

**Answer:** Cross-validation estimates generalization performance by training and evaluating on different data splits, reducing variance compared to a single train/test split.

| Method | Description | When to Use |
|--------|-------------|------------|
| **k-Fold CV** | Split into k folds; train on k-1, test on 1 | General purpose |
| **Stratified k-Fold** | Preserves class ratios in each fold | Classification |
| **Leave-One-Out (LOO)** | k = n; every sample is a test point once | Very small datasets |
| **Time-Series Split** | Always test on future data | Time-ordered data |

\`\`\`python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
\`\`\`

---

### Q6 ★★☆☆☆ — What is overfitting vs. underfitting? How do you diagnose and fix each?

**Answer:**

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Underfitting** (high bias) | High train error AND high val error | More complex model, more features, less regularization |
| **Overfitting** (high variance) | Low train error, high val error | More data, regularization (L1/L2/dropout), simpler model, early stopping |

**Learning curves** are the diagnostic tool: plot train and validation error vs. training set size.
- Underfitting: both curves plateau high and close together
- Overfitting: train curve drops, validation curve stays high — large gap

**Bias-variance decomposition:**
$$\\text{Expected Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$

---

### Q7 ★☆☆☆☆ — What are the pitfalls of A/B testing?

**Answer:**

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Novelty effect** | Users engage with new feature just because it's new | Run test long enough (weeks) |
| **Sample size too small** | Underpowered test, high false-negative rate | Pre-calculate required n via power analysis |
| **Multiple comparisons** | Testing 20 variants → 1 will be significant by chance | Bonferroni correction; limit number of variants |
| **Simpson's Paradox** | Aggregated result reverses when stratified | Stratify analysis by user segment |
| **Network effects** | Control group affected by treatment group | Use cluster-based randomization |

**Statistical significance ≠ practical significance.** Always check effect size (Cohen's d, relative lift) alongside p-value.`,

    zh: `## 模型评估面试题

> 来源：《百面机器学习》第2章 + 《机器学习面试题》第5章
> 几乎每次ML面试都会涉及的核心评估概念。

---

### Q1 ★☆☆☆☆ — 准确率有哪些局限性？

**解答：** 准确率 = (TP + TN) / 总数，在**类别不平衡数据集上具有误导性**。

**经典陷阱：** 癌症筛查模型，若全预测"无癌症"，在1%患病率的数据集上准确率达99%——但完全没用。

**不平衡数据的更好替代指标：**
- **精确率 & 召回率** —— 关注正类
- **F1分数** —— 精确率和召回率的调和平均
- **ROC-AUC** —— 与阈值无关的排序质量
- **PR-AUC** —— 严重类别不平衡时更好

---

### Q2 ★☆☆☆☆ — 解释精确率、召回率和F1分数。

**解答：**

|  | 预测正例 | 预测负例 |
|--|---------|---------|
| **实际正例** | TP | FN |
| **实际负例** | FP | TN |

- 精确率 = TP / (TP + FP)：预测为正的样本中，真正为正的比例
- 召回率 = TP / (TP + FN)：所有真正为正的样本中，被正确预测的比例
- F1 = 2 × P × R / (P + R)：调和平均

**精确率-召回率权衡：** 提高分类阈值 → 精确率上升，召回率下降。根据错误代价选择：
- 医学诊断：偏向高召回率（不漏掉病人）
- 垃圾邮件检测：偏向高精确率（不误判重要邮件）

---

### Q3 ★★☆☆☆ — 什么是ROC曲线和AUC？何时应使用PR-AUC？

**解答：**

**ROC曲线：** 在所有阈值下，绘制真正率（召回率）与假正率的关系。
- **AUC** = ROC曲线面积 = 模型将随机正例排在随机负例之前的概率
- AUC = 0.5 → 随机分类器；AUC = 1.0 → 完美

**绘制方法：** 按预测分数降序排列，逐阈值计算 (FPR, TPR)，依次绘点。

**何时用PR-AUC：**
- 严重类别不平衡（如欺诈检测：1/10000）
- 类别不平衡时ROC偏于乐观（TN基数大，FPR始终偏低）
- PR曲线仅关注正类性能

---

### Q4 ★☆☆☆☆ — 均方根误差的"意外"是什么？

**解答：** RMSE的特性：**因为平方运算，会不成比例地惩罚大误差**。

两个模型MAE相同，RMSE可能差异很大。模型A少数大误差 vs. 模型B较多小误差，RMSE下A看起来差很多。这在大误差代价高昂时是好事，但用于鲁棒性评估可能具有误导性。

**替代方案：**
- **MAE** —— 线性惩罚，对异常值鲁棒
- **Huber Loss** —— 小误差用平方，大误差用线性（兼顾两者）

---

### Q5 ★☆☆☆☆ — 什么是交叉验证？为什么使用它？

**解答：** 交叉验证通过在不同数据划分上训练和评估模型，估计泛化性能，减少单次划分的方差。

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| **k折交叉验证** | 分k份；k-1份训练，1份测试 | 通用 |
| **分层k折** | 每折保持类别比例 | 分类任务 |
| **留一法（LOO）** | k=n；每个样本依次作为测试点 | 极小数据集 |
| **时间序列划分** | 始终用未来数据测试 | 时序数据 |

---

### Q6 ★★☆☆☆ — 什么是过拟合和欠拟合？如何诊断和解决？

**解答：**

| 问题 | 症状 | 解决方法 |
|------|------|---------|
| **欠拟合**（高偏差）| 训练误差和验证误差都高 | 更复杂模型、更多特征、减少正则化 |
| **过拟合**（高方差）| 训练误差低，验证误差高 | 更多数据、正则化（L1/L2/Dropout）、简化模型、早停 |

**学习曲线**是诊断工具：画训练/验证误差 vs. 训练集大小。
- 欠拟合：两条曲线都在高处趋于平稳
- 过拟合：训练曲线下降，验证曲线居高，间距大

**偏差-方差分解：**
期望误差 = 偏差² + 方差 + 不可约噪声

---

### Q7 ★☆☆☆☆ — A/B测试有哪些陷阱？

**解答：**

| 陷阱 | 描述 | 解决方案 |
|------|------|---------|
| **新奇效应** | 用户仅因新功能而参与 | 测试时间足够长（数周）|
| **样本量不足** | 低功效测试，假阴性率高 | 通过功效分析预先计算所需n |
| **多重比较** | 测试20个变体，随机有1个显著 | Bonferroni校正 |
| **辛普森悖论** | 汇总结果与分层结果相反 | 按用户分组分层分析 |
| **网络效应** | 对照组受实验组影响 | 基于集群的随机化 |

**统计显著≠实际显著。** 始终同时检查效应大小（Cohen's d、相对提升）和p值。`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Classical Algorithms
// Source: 百面机器学习 Ch.3 + 机器学习面试题 Ch.9 (SVM), Ch.2 (LR)
// ─────────────────────────────────────────────────────────────────────────────
export const interviewClassicalAlgorithms: TopicContent = {
  id: 'interview-classical-algorithms',
  title: { en: 'Interview: Classical Algorithms', zh: '面试：经典算法' },
  contentType: 'article',
  content: {
    en: `## Classical Algorithms Interview Questions

> Source: *百面机器学习* Ch.3 + *机器学习面试题* Ch.2, 9
> SVM, Logistic Regression, Decision Trees — the backbone of ML interviews.

---

### Q1 ★★☆☆☆ — Explain SVM. What is the margin and why maximize it?

**Answer:** SVM finds the **maximum-margin hyperplane** that separates classes. The margin is the perpendicular distance from the hyperplane to the nearest data points (support vectors) of each class.

**Why maximize margin?**
- By VC-dimension theory, larger margin → smaller generalization error bound
- More robust to test points near the boundary

**Primal problem (hard margin):**
$$\\min_{w,b} \\frac{1}{2}\\|w\\|^2 \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\geq 1 \\; \\forall i$$

**Soft margin (C-SVM):** Allows misclassification via slack variables ξᵢ, controlled by C:
- Large C → small margin, fewer misclassifications (risk overfitting)
- Small C → large margin, more misclassifications (risk underfitting)

**Kernel trick:** Map data to higher-dimensional space using a kernel function K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ) — compute the inner product without explicitly computing φ.

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | xᵢᵀxⱼ | Linearly separable or high-dim text |
| RBF / Gaussian | exp(−γ‖xᵢ − xⱼ‖²) | General purpose |
| Polynomial | (xᵢᵀxⱼ + c)ᵈ | Image features |

---

### Q2 ★★☆☆☆ — What is Logistic Regression? Why use sigmoid? Why cross-entropy loss?

**Answer:** LR is a **discriminative linear classifier** that models P(y=1|x) using the sigmoid function:

$$P(y=1|x) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-w^T x - b}}$$

**Why sigmoid?** It maps (−∞, +∞) → (0, 1), yielding a valid probability. It arises naturally from the exponential family / maximum entropy framework.

**Why cross-entropy loss?** Derived from Maximum Likelihood Estimation (MLE):

$$\\mathcal{L} = -\\sum_i [y_i \\log \\hat{p}_i + (1-y_i) \\log(1-\\hat{p}_i)]$$

Using MSE for probability estimation is problematic because the gradient saturates (sigmoid is flat at extremes), causing vanishing gradients.

**LR vs. SVM:**
| | Logistic Regression | SVM |
|--|--|--|
| Output | Calibrated probability | Class label (no natural probability) |
| Loss | Log-loss (all points) | Hinge (only support vectors matter) |
| Outliers | Sensitive | Robust (only support vectors matter) |
| Kernel | Possible but uncommon | Natural with kernel trick |

---

### Q3 ★★☆☆☆ — How does a decision tree split nodes? Compare ID3, C4.5, and CART.

**Answer:** All three algorithms greedily choose the split that maximizes information gain (or a variant):

| Algorithm | Split Criterion | Output | Key Advantage |
|-----------|----------------|--------|---------------|
| **ID3** | Information Gain | Multi-way split, classification only | Simple |
| **C4.5** | Information Gain Ratio (normalizes by feature entropy) | Multi-way split | Handles continuous features, biased toward many-valued features corrected |
| **CART** | Gini impurity (classification) or MSE (regression) | Binary split only | Works for both tasks, basis for Random Forest |

**Gini Impurity:**
$$G(D) = 1 - \\sum_k p_k^2$$

**Information Gain:**
$$IG(D, A) = H(D) - \\sum_{v} \\frac{|D_v|}{|D|} H(D_v)$$

**Preventing overfitting in decision trees:**
- **Pre-pruning:** Stop splitting when information gain < threshold, min samples per leaf
- **Post-pruning:** Grow full tree, then prune branches using validation set (Reduced Error Pruning, Cost Complexity Pruning)

---

### Q4 ★★★☆☆ — What is the kernel trick and why doesn't it require computing φ(x) explicitly?

**Answer:** The kernel trick exploits the fact that SVM optimization only requires inner products xᵢᵀxⱼ (via the dual formulation). If we can compute K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ) directly, we never need to materialize the (possibly infinite-dimensional) feature map φ.

**Example — Polynomial kernel:**
$$K(x, z) = (x \\cdot z + 1)^2 = \\phi(x) \\cdot \\phi(z)$$

where φ(x) = [x₁², x₂², √2·x₁x₂, √2·x₁, √2·x₂, 1] — a 6-dim mapping computed in O(d) time via the kernel function rather than O(d²).

**Mercer's theorem:** A function K is a valid kernel if and only if the Gram matrix [K(xᵢ, xⱼ)] is positive semi-definite for any set of inputs.

---

### Q5 ★☆☆☆☆ — Logistic Regression vs. SVM: when to use each?

| Criterion | Use LR | Use SVM |
|-----------|--------|---------|
| Need probability output | ✓ | ✗ |
| Well-separated classes | Either | ✓ (hard margin) |
| Overlapping classes | ✓ | ✓ (soft margin) |
| High-dimensional sparse (text) | ✓ | ✓ (linear kernel) |
| Small dataset | Either | ✓ |
| Large dataset | ✓ (scales better) | ✗ (kernel SVM O(n²-n³)) |`,

    zh: `## 经典算法面试题

> 来源：《百面机器学习》第3章 + 《机器学习面试题》第2章、第9章
> SVM、逻辑回归、决策树 —— ML面试的核心。

---

### Q1 ★★☆☆☆ — 解释SVM。什么是间隔？为什么要最大化间隔？

**解答：** SVM寻找将类别分开的**最大间隔超平面**。间隔是超平面到每类最近数据点（支持向量）的垂直距离。

**为什么最大化间隔？**
- 由VC维理论，间隔越大 → 泛化误差上界越小
- 对测试集中边界附近的点更鲁棒

**软间隔（C-SVM）：** 通过松弛变量ξᵢ允许误分类，由C控制：
- 大C → 小间隔，允许少量误分类（有过拟合风险）
- 小C → 大间隔，允许更多误分类（有欠拟合风险）

**核技巧：** 通过核函数K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ) 将数据映射到高维空间，无需显式计算φ。

| 核函数 | 公式 | 适用场景 |
|--------|------|---------|
| 线性核 | xᵢᵀxⱼ | 线性可分或高维文本 |
| RBF/高斯核 | exp(−γ‖xᵢ − xⱼ‖²) | 通用 |
| 多项式核 | (xᵢᵀxⱼ + c)ᵈ | 图像特征 |

---

### Q2 ★★☆☆☆ — 什么是逻辑回归？为什么用Sigmoid？为什么用交叉熵损失？

**解答：** 逻辑回归是**判别式线性分类器**，用Sigmoid函数建模P(y=1|x)：

P(y=1|x) = 1 / (1 + e^{-w^T x - b})

**为什么用Sigmoid？** 将(-∞, +∞) 映射到 (0,1)，得到有效概率，从指数族/最大熵框架自然导出。

**为什么用交叉熵损失？** 由最大似然估计（MLE）推导而来，使用MSE估计概率会导致梯度饱和（Sigmoid在极端值处很平坦），产生梯度消失。

**逻辑回归 vs. SVM：**
| | 逻辑回归 | SVM |
|--|---------|-----|
| 输出 | 校准的概率 | 类别标签（无自然概率）|
| 损失 | 对数损失（所有点）| Hinge（仅支持向量重要）|
| 异常值 | 敏感 | 鲁棒（仅支持向量重要）|

---

### Q3 ★★☆☆☆ — 决策树如何分裂节点？对比ID3、C4.5和CART。

**解答：**

| 算法 | 分裂准则 | 输出 | 关键优势 |
|------|---------|------|---------|
| **ID3** | 信息增益 | 多叉，仅分类 | 简单 |
| **C4.5** | 信息增益比（按特征熵归一化）| 多叉 | 处理连续特征，纠正多值特征偏差 |
| **CART** | 基尼不纯度（分类）或MSE（回归）| 二叉 | 适用于两类任务，随机森林的基础 |

**基尼不纯度：** G(D) = 1 − Σₖ pₖ²

**防止过拟合：**
- **预剪枝：** 信息增益<阈值时停止分裂，设置最小叶节点样本数
- **后剪枝：** 先生成完整树，然后用验证集剪枝（代价复杂度剪枝）

---

### Q4 ★★★☆☆ — 核技巧为什么不需要显式计算φ(x)？

**解答：** SVM的对偶优化问题只需要内积 xᵢᵀxⱼ。若能直接计算K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)，就无需实例化（可能是无限维的）特征映射φ。

**多项式核示例：** K(x, z) = (x·z + 1)² 对应6维特征映射，但通过核函数用O(d)时间计算，而非O(d²)。

**Mercer定理：** 函数K是有效核函数，当且仅当任意输入集合的Gram矩阵 [K(xᵢ, xⱼ)] 是半正定的。

---

### Q5 ★☆☆☆☆ — 逻辑回归 vs. SVM：各在什么场景下使用？

| 判断依据 | 用逻辑回归 | 用SVM |
|---------|-----------|------|
| 需要概率输出 | ✓ | ✗ |
| 类别分离良好 | 均可 | ✓（硬间隔）|
| 类别重叠 | ✓ | ✓（软间隔）|
| 高维稀疏（文本）| ✓ | ✓（线性核）|
| 大数据集 | ✓（扩展更好）| ✗（核SVM O(n²-n³)）|`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Optimization & Regularization
// Source: 百面机器学习 Ch.7 + 机器学习面试题 Ch.2–3
// ─────────────────────────────────────────────────────────────────────────────
export const interviewOptimization: TopicContent = {
  id: 'interview-optimization',
  title: { en: 'Interview: Optimization & Regularization', zh: '面试：优化与正则化' },
  contentType: 'article',
  content: {
    en: `## Optimization & Regularization Interview Questions

> Source: *百面机器学习* Ch.7 + *机器学习面试题* Ch.2–3
> Loss functions, gradient descent variants, and regularization — asked in almost every interview.

---

### Q1 ★☆☆☆☆ — What are the common loss functions and when to use each?

| Loss | Formula | Use Case | Property |
|------|---------|----------|----------|
| **MSE / L2** | Σ(yᵢ − ŷᵢ)² | Regression | Differentiable, sensitive to outliers |
| **MAE / L1** | Σ|yᵢ − ŷᵢ| | Robust regression | Not differentiable at 0, sparse residuals |
| **Huber** | L2 if |e|<δ, L1 otherwise | Robust regression | Best of L1 and L2 |
| **Cross-Entropy** | −Σyᵢ log(ŷᵢ) | Classification | MLE-derived, handles probabilities |
| **Hinge** | Σ max(0, 1 − yᵢŷᵢ) | SVM | Non-differentiable, convex upper bound of 0-1 loss |
| **KL Divergence** | Σ p log(p/q) | VAE, distillation | Measures distribution distance |

---

### Q2 ★★☆☆☆ — Compare BGD, SGD, and Mini-batch GD.

| Method | Update Uses | Pros | Cons |
|--------|------------|------|------|
| **Batch GD (BGD)** | Full dataset | Stable convergence, exact gradient | Slow for large datasets |
| **Stochastic GD (SGD)** | 1 sample | Fast updates, escapes local minima | Noisy, high variance |
| **Mini-batch GD** | B samples (32–512) | Balance of both, GPU-efficient | Tuning batch size required |

**SGD with momentum:**
$$v_{t+1} = \\beta v_t + (1-\\beta) \\nabla_\\theta L \\qquad \\theta \\leftarrow \\theta - \\alpha v_{t+1}$$

Momentum accumulates gradient history → faster convergence, smooths oscillations.

---

### Q3 ★★★☆☆ — Explain Adam optimizer. Why is it the default choice?

**Answer:** Adam (Adaptive Moment Estimation) combines Momentum and RMSProp:

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t \\quad \\text{(1st moment)}$$
$$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2 \\quad \\text{(2nd moment)}$$
$$\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t} \\quad \\text{(bias correction)}$$
$$\\theta_{t+1} = \\theta_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$

**Default hyperparameters:** β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3

**Why Adam?** Adapts learning rate per parameter — large parameters get smaller updates, small parameters get larger updates. Works well across diverse architectures without much tuning.

**Weakness:** Can generalize worse than SGD+momentum for some tasks (notably image classification). AdamW (Adam + decoupled weight decay) addresses this.

---

### Q4 ★★★☆☆ — How does gradient verification work?

**Answer:** To check that backpropagation is implemented correctly, compare the analytical gradient with the **numerical gradient** (finite differences):

$$g_{\\text{numerical}} = \\frac{L(\\theta + \\epsilon) - L(\\theta - \\epsilon)}{2\\epsilon}$$

**Relative error** should be < 1e-5 for correct implementation:

$$\\text{relative error} = \\frac{|g_{\\text{analytical}} - g_{\\text{numerical}}|}{\\max(|g_{\\text{analytical}}|, |g_{\\text{numerical}}|)}$$

**Use only for debugging** — numerical gradients are O(n) in parameters and too slow for production.

---

### Q5 ★★★☆☆ — L1 vs. L2 regularization: difference and intuition for sparsity.

**Answer:**

| | L1 (Lasso) | L2 (Ridge) |
|--|--|--|
| Penalty | λΣ|wᵢ| | λΣwᵢ² |
| Effect on weights | Drives many weights to exactly 0 | Shrinks all weights toward 0 (rarely exactly 0) |
| Solution shape | Sparse | Dense |
| Optimization | Non-differentiable at 0 (sub-gradient) | Differentiable everywhere (closed form) |
| Best for | Feature selection, sparse data | When all features matter |

**Why does L1 produce sparsity?** The L1 ball (constraint region) has corners on the axes. The loss function's contours tend to touch these corners, forcing some weights to exactly zero. The L2 ball is smooth — contours touch it off-axis, so weights approach but don't reach zero.

\`\`\`python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

lasso = Lasso(alpha=0.1)     # L1
ridge = Ridge(alpha=1.0)     # L2
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # L1 + L2
\`\`\`

---

### Q6 ★★☆☆☆ — How do you accelerate SGD? Compare Momentum, Nesterov, RMSProp, Adam.

| Optimizer | Key Idea | Formula Sketch |
|-----------|---------|----------------|
| **Momentum** | Accumulate past gradients | v = βv + α∇L; θ -= v |
| **Nesterov** | "Look ahead" gradient | Compute gradient at θ - βv (not θ) |
| **AdaGrad** | Per-parameter lr (sum of sq. grads) | θ -= α / (√G + ε) · g |
| **RMSProp** | AdaGrad with decaying sum | G = βG + (1-β)g²; adaptive lr |
| **Adam** | Momentum + RMSProp + bias correction | See Q3 above |
| **AdamW** | Adam + decoupled weight decay | Better generalization |`,

    zh: `## 优化与正则化面试题

> 来源：《百面机器学习》第7章 + 《机器学习面试题》第2-3章
> 损失函数、梯度下降变体和正则化 —— 几乎每次面试都会考到。

---

### Q1 ★☆☆☆☆ — 常见的损失函数有哪些？各适用什么场景？

| 损失函数 | 公式 | 应用场景 | 特性 |
|---------|------|---------|------|
| **MSE/L2** | Σ(yᵢ − ŷᵢ)² | 回归 | 可微，对异常值敏感 |
| **MAE/L1** | Σ\|yᵢ − ŷᵢ\| | 鲁棒回归 | 零点不可微，残差稀疏 |
| **Huber损失** | \|e\|<δ用L2，否则用L1 | 鲁棒回归 | 兼顾L1和L2优点 |
| **交叉熵** | −Σyᵢ log(ŷᵢ) | 分类 | 由MLE推导，处理概率 |
| **Hinge损失** | Σ max(0, 1 − yᵢŷᵢ) | SVM | 不可微，0-1损失的凸上界 |
| **KL散度** | Σ p log(p/q) | VAE、知识蒸馏 | 衡量分布距离 |

---

### Q2 ★★☆☆☆ — 对比BGD、SGD和小批量梯度下降。

| 方法 | 更新使用 | 优点 | 缺点 |
|------|---------|------|------|
| **批量梯度下降（BGD）** | 全部数据 | 收敛稳定，梯度精确 | 大数据集慢 |
| **随机梯度下降（SGD）** | 1个样本 | 更新快，可逃出局部极小 | 噪声大，方差高 |
| **小批量梯度下降** | B个样本（32-512）| 兼顾两者，GPU高效 | 需要调整批量大小 |

**带动量的SGD：** 通过累积历史梯度加速收敛，平滑振荡。

---

### Q3 ★★★☆☆ — 解释Adam优化器。为什么它是默认选择？

**解答：** Adam（自适应矩估计）结合了动量和RMSProp：
- 一阶矩：对梯度做指数移动平均（方向信息）
- 二阶矩：对梯度平方做指数移动平均（幅度信息）
- 偏差校正：早期步骤中补偿初始化为零的偏差

**默认超参数：** β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3

**为什么用Adam？** 自适应地为每个参数调整学习率——适用于多种架构，无需大量调参。

**缺点：** 在某些任务（如图像分类）上泛化可能不如SGD+动量。AdamW（Adam+解耦权重衰减）解决了这个问题。

---

### Q4 ★★★☆☆ — 梯度验证如何工作？

**解答：** 通过对比解析梯度和数值梯度（有限差分）来验证反向传播实现的正确性：

数值梯度 = [L(θ + ε) − L(θ − ε)] / 2ε

**相对误差**应小于1e-5，否则实现有问题。

**仅用于调试** —— 数值梯度需要O(n)次前向传播，生产环境太慢。

---

### Q5 ★★★☆☆ — L1与L2正则化的区别？为什么L1产生稀疏性？

**解答：**

| | L1（Lasso）| L2（Ridge）|
|--|-----------|-----------|
| 惩罚项 | λΣ\|wᵢ\| | λΣwᵢ² |
| 对权重的效果 | 将许多权重驱向精确为0 | 将所有权重向0压缩（很少精确为0）|
| 解的形状 | 稀疏 | 稠密 |

**为什么L1产生稀疏性？** L1球（约束区域）在坐标轴上有尖角。损失函数的等高线倾向于接触这些尖角，迫使某些权重精确为0。L2球是光滑的，等高线在离轴处接触，权重趋近但不到达0。

---

### Q6 ★★☆☆☆ — 如何加速SGD？对比动量、Nesterov、RMSProp和Adam。

| 优化器 | 核心思想 | 关键特点 |
|--------|---------|---------|
| **Momentum** | 累积历史梯度 | 加速收敛，减少振荡 |
| **Nesterov** | "预看"梯度 | 在更新后位置计算梯度，更精准 |
| **AdaGrad** | 参数自适应学习率 | 适合稀疏特征，学习率单调递减 |
| **RMSProp** | AdaGrad + 衰减 | 解决学习率消失问题 |
| **Adam** | Momentum + RMSProp + 偏差校正 | 工程首选，默认超参数效果好 |
| **AdamW** | Adam + 解耦权重衰减 | 更好的泛化性能 |`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Neural Networks & Deep Learning
// Source: 百面机器学习 Ch.9, 10 + 机器学习面试题 Ch.10–12
// ─────────────────────────────────────────────────────────────────────────────
export const interviewNeuralNetworks: TopicContent = {
  id: 'interview-neural-networks',
  title: { en: 'Interview: Neural Networks & Deep Learning', zh: '面试：神经网络与深度学习' },
  contentType: 'article',
  content: {
    en: `## Neural Networks & Deep Learning Interview Questions

> Source: *百面机器学习* Ch.9–10 + *机器学习面试题* Ch.10–12
> Activation functions, backprop, CNNs, RNNs, and attention.

---

### Q1 ★☆☆☆☆ — List common activation functions and their derivatives.

| Activation | Formula | Derivative | Pros / Cons |
|-----------|---------|-----------|-------------|
| **Sigmoid** | 1/(1+e⁻ˣ) | σ(1−σ) | Probabilistic output; vanishing gradient at extremes |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1−tanh² | Zero-centered; still vanishes |
| **ReLU** | max(0, x) | 0 or 1 | Simple, no vanishing; dying ReLU problem |
| **Leaky ReLU** | max(αx, x) α≈0.01 | α or 1 | Fixes dying ReLU |
| **ELU** | x if x>0 else α(eˣ−1) | 1 or α·eˣ | Smooth at 0, negative outputs |
| **GELU** | x·Φ(x) | Complex | Default in BERT, GPT |
| **Softmax** | eˣᵢ/Σeˣⱼ | pᵢ(δᵢⱼ−pⱼ) | Multi-class output |

**Why Sigmoid/Tanh cause vanishing gradients:** When |x| is large, σ'(x) ≈ 0 and tanh'(x) ≈ 0. Multiplied through many layers in backprop → gradients shrink exponentially.

**Why ReLU is preferred:** Gradient is always 1 for x>0 — no saturation in positive regime. But dead neurons (output always 0) occur when large negative bias pushes all activations below zero.

---

### Q2 ★☆☆☆☆ — Can you initialize all neural network weights to 0? Why not?

**Answer: No.** If all weights are 0, all neurons in the same layer compute identical outputs and receive identical gradients. Every weight in a layer updates identically — symmetry is never broken, and the network is equivalent to a single neuron no matter how many layers it has.

**Solution:** Random initialization
- **Xavier/Glorot:** Var(w) = 2/(nᵢₙ + nₒᵤₜ) — best for Sigmoid/Tanh
- **He/Kaiming:** Var(w) = 2/nᵢₙ — best for ReLU (accounts for the half-zero output)
- **Orthogonal:** Initialize weight matrices as random orthogonal matrices — good for RNNs

\`\`\`python
import torch.nn as nn

layer = nn.Linear(256, 128)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)
\`\`\`

---

### Q3 ★★☆☆☆ — Explain the vanishing gradient problem in RNNs. How does LSTM solve it?

**Answer:** In a vanilla RNN, the gradient at time step t flows back through T multiplications of the weight matrix W:

$$\\frac{\\partial L}{\\partial h_0} = \\prod_{t=1}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}} = \\prod_{t=1}^{T} W^T \\cdot \\text{diag}(\\sigma'(h_{t-1}))$$

If the spectral radius of W < 1 and σ' < 1 (sigmoid, tanh), this product **decays exponentially** → vanishing gradients over long sequences.

**LSTM solution:** The **cell state** cₜ acts as a "memory highway" with additive updates (not multiplicative):

$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$

The forget gate fₜ ∈ (0,1) controls what to keep. Crucially, the gradient flows back through addition: ∂cₜ/∂cₜ₋₁ = fₜ — if fₜ ≈ 1, gradient is preserved without decay.

---

### Q4 ★★☆☆☆ — What are sparse interactions and parameter sharing in CNNs?

**Answer:**

**Sparse interactions (local receptive fields):** A CNN filter of size k×k connects each output neuron to only k² input neurons, not all n². For a 3×3 filter on a 64×64 image: 9 weights per neuron instead of 4096. This captures local spatial structure efficiently.

**Parameter sharing:** The same k×k filter is applied at every spatial position — the same edge detector works everywhere in the image. This drastically reduces parameters: a convolutional layer has k×k×Cᵢₙ×Cₒᵤₜ parameters vs. (H×W×Cᵢₙ) × (H×W×Cₒᵤₜ) for a fully connected layer.

| Layer Type | Parameters | Receptive Field |
|-----------|-----------|----------------|
| FC (224×224×3 → 1000) | ~150M | Global |
| Conv 3×3 (3→64) | 1,728 | Local 3×3 |
| Conv 3×3 (64→128) | 73,728 | Local 3×3 |

---

### Q5 ★★★☆☆ — How does ResNet solve the vanishing gradient problem?

**Answer:** ResNet introduces **residual connections (skip connections)**:

$$h_{l+1} = \\mathcal{F}(h_l, W_l) + h_l$$

where F is the "residual" (what the layers need to learn). The gradient becomes:

$$\\frac{\\partial L}{\\partial h_l} = \\frac{\\partial L}{\\partial h_{l+1}} \\cdot \\left(1 + \\frac{\\partial \\mathcal{F}}{\\partial h_l}\\right)$$

The "+1" term provides a **gradient highway** — even if ∂F/∂h is small, gradients can flow back through the identity shortcut. This allows training networks of 100+ layers that would otherwise fail.

**Key insight:** If a layer is not needed, its residual F → 0 and the block becomes an identity mapping — easier to learn than the layer having to learn the identity itself.

---

### Q6 ★★★☆☆ — What is the attention mechanism? How does it solve Seq2Seq bottlenecks?

**Answer:**

**Seq2Seq bottleneck:** The encoder compresses an entire variable-length sequence into a single fixed-size context vector — information is lost, especially for long sequences.

**Attention solution:** The decoder, at each time step, computes a weighted sum over all encoder hidden states:

$$c_t = \\sum_i \\alpha_{ti} h_i \\quad \\text{where} \\quad \\alpha_{ti} = \\text{softmax}(e_{ti})$$
$$e_{ti} = \\text{score}(s_{t-1}, h_i) \\quad \\text{(alignment function)}$$

The decoder selectively focuses on relevant encoder positions — no bottleneck.

**Self-attention (Transformer):**
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Q, K, V are linear projections of the same input sequence. Every token attends to every other token in O(n²) time but O(1) path length — captures long-range dependencies directly.`,

    zh: `## 神经网络与深度学习面试题

> 来源：《百面机器学习》第9-10章 + 《机器学习面试题》第10-12章
> 激活函数、反向传播、CNN、RNN和注意力机制。

---

### Q1 ★☆☆☆☆ — 常用激活函数及其导数。

| 激活函数 | 公式 | 导数 | 优缺点 |
|---------|------|------|--------|
| **Sigmoid** | 1/(1+e⁻ˣ) | σ(1−σ) | 输出概率值；极端值处梯度消失 |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1−tanh² | 零均值；仍会消失 |
| **ReLU** | max(0, x) | 0或1 | 简单，无消失；Dead ReLU问题 |
| **Leaky ReLU** | max(αx, x) α≈0.01 | α或1 | 解决Dead ReLU |
| **GELU** | x·Φ(x) | 复杂 | BERT、GPT默认激活函数 |

**Sigmoid/Tanh导致梯度消失：** 当\|x\|较大时，导数接近0，经多层反向传播乘积后梯度指数级衰减。

**为什么ReLU更好：** x>0时梯度恒为1，不饱和。但存在Dead Neuron问题（大负偏置导致神经元永远不激活）。

---

### Q2 ★☆☆☆☆ — 神经网络参数全初始化为0可以吗？

**解答：不可以。** 若所有权重为0，同一层的所有神经元输出相同、接收相同梯度，每个权重更新相同——对称性永远无法打破，无论多少层，网络等价于单个神经元。

**解决方案 —— 随机初始化：**
- **Xavier/Glorot：** Var(w) = 2/(nᵢₙ + nₒᵤₜ) —— 适合Sigmoid/Tanh
- **He/Kaiming：** Var(w) = 2/nᵢₙ —— 适合ReLU（考虑半零输出）
- **正交初始化：** 初始化为随机正交矩阵 —— 适合RNN

---

### Q3 ★★☆☆☆ — 解释RNN中的梯度消失问题。LSTM如何解决？

**解答：** 在普通RNN中，梯度需经过T次权重矩阵W的乘法回传，若W的谱半径<1且σ'<1，乘积**指数级衰减** → 长序列梯度消失。

**LSTM的解决方案：** **单元状态**作为"记忆高速公路"，通过加法更新（而非乘法）：

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

遗忘门fₜ∈(0,1)控制保留什么。关键是梯度通过加法回流：∂cₜ/∂cₜ₋₁ = fₜ —— 若fₜ≈1，梯度无衰减。

---

### Q4 ★★☆☆☆ — CNN中的稀疏交互和参数共享各起什么作用？

**解答：**

**稀疏交互（局部感受野）：** 每个输出神经元只连接k²个输入神经元，而非全部n²个。捕捉局部空间结构，大幅减少参数。

**参数共享：** 同一个k×k滤波器在所有空间位置共享参数——同一个边缘检测器适用于图像任何位置。卷积层参数量：k×k×Cᵢₙ×Cₒᵤₜ，远少于全连接层的 (H×W×Cᵢₙ) × (H×W×Cₒᵤₜ)。

---

### Q5 ★★★☆☆ — ResNet如何解决梯度消失问题？

**解答：** ResNet引入**残差连接（跳跃连接）**：

hₗ₊₁ = F(hₗ, Wₗ) + hₗ

梯度变为：∂L/∂hₗ = ∂L/∂hₗ₊₁ · (1 + ∂F/∂hₗ)

"+1"项提供**梯度高速公路**——即使∂F/∂h很小，梯度也能通过恒等捷径流回。这使训练100+层的网络成为可能。

**关键洞察：** 若某层不需要，其残差F→0，该模块变为恒等映射——比让层自己学习恒等映射容易得多。

---

### Q6 ★★★☆☆ — 什么是注意力机制？如何解决Seq2Seq的瓶颈问题？

**解答：**

**Seq2Seq瓶颈：** 编码器将变长序列压缩为单个固定大小的上下文向量——信息丢失，对长序列尤其严重。

**注意力解决方案：** 解码器在每个时间步对所有编码器隐藏状态加权求和，selectively关注相关位置，无瓶颈。

**自注意力（Transformer）：**
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V

每个token关注序列中所有其他token，O(n²)时间复杂度，O(1)路径长度——直接捕捉长距离依赖。`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Unsupervised Learning & Probabilistic Models
// Source: 百面机器学习 Ch.4, 5, 6, 8
// ─────────────────────────────────────────────────────────────────────────────
export const interviewUnsupervised: TopicContent = {
  id: 'interview-unsupervised',
  title: { en: 'Interview: Unsupervised & Probabilistic Models', zh: '面试：非监督学习与概率模型' },
  contentType: 'article',
  content: {
    en: `## Unsupervised Learning & Probabilistic Models Interview Questions

> Source: *百面机器学习* Ch.4–6, 8
> Dimensionality reduction, clustering, probabilistic graphical models, and sampling.

---

### Q1 ★★☆☆☆ — Describe the K-means algorithm step by step.

**Answer:**

1. **Preprocessing:** Normalize features, handle outliers
2. **Initialize** K cluster centers μ₁, …, μₖ (random or K-means++ for better initialization)
3. **Repeat until convergence:**
   - **Assignment:** For each xᵢ, assign to nearest center: cᵢ = argminₖ ‖xᵢ − μₖ‖²
   - **Update:** Recompute centers: μₖ = mean of all xᵢ where cᵢ = k
4. **Convergence:** When assignments stop changing (objective J = Σᵢ ‖xᵢ − μ_{cᵢ}‖² no longer decreases)

**Choosing K:** Elbow method (plot J vs K), silhouette coefficient, gap statistic.

**K-means limitations:** Assumes spherical clusters of similar size; sensitive to initialization; struggles with non-convex shapes (use DBSCAN instead).

---

### Q2 ★★☆☆☆ — What is PCA? Explain both max-variance and min-reconstruction-error interpretations.

**Answer — PCA finds a linear subspace that best represents the data:**

**Max-variance interpretation:** Find unit vector w₁ such that the projected data has maximum variance. The solution is the **eigenvector corresponding to the largest eigenvalue** of the covariance matrix C = (1/n) XᵀX.

**Min-reconstruction-error interpretation:** Find the k-dimensional subspace minimizing the average squared distance between original points and their projections. Solution: same eigenvectors (equivalently, right singular vectors of X from SVD).

**Algorithm:**
1. Center the data: X ← X − mean(X)
2. Compute covariance matrix C = XᵀX / n (or use SVD for numerical stability)
3. Eigendecompose: C = VΛVᵀ — columns of V are principal components
4. Project: X_pca = X · V[:, :k]

\`\`\`python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)   # keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Components: {pca.n_components_}, Ratio: {pca.explained_variance_ratio_}")
\`\`\`

---

### Q3 ★★☆☆☆ — How does LDA differ from PCA? When to prefer each?

| | PCA | LDA |
|--|--|--|
| Supervised? | No | Yes (uses class labels) |
| Objective | Max variance | Max class separability (Fisher criterion) |
| Components | min(n-1, d) | min(c-1, d) where c = num classes |
| Use case | Dimensionality reduction without label | Feature extraction for classification |

**LDA criterion:** Maximize: J(w) = wᵀ Sᵦ w / wᵀ Sᵥᵥ w

where Sᵦ = between-class scatter, Sᵥᵥ = within-class scatter. Solution: eigenvectors of Sᵥᵥ⁻¹Sᵦ.

**LDA assumption:** Each class follows a Gaussian with the same covariance matrix (violated often in practice).

---

### Q4 ★★☆☆☆ — Explain Gaussian Mixture Models (GMM). How does EM work?

**Answer:** GMM models data as a mixture of K Gaussian components:

$$p(x) = \\sum_{k=1}^K \\pi_k \\mathcal{N}(x; \\mu_k, \\Sigma_k)$$

where πₖ are mixing weights (sum to 1), and each component has its own mean μₖ and covariance Σₖ.

**EM Algorithm (Expectation-Maximization):**

**E-step:** Compute responsibility (soft cluster assignment):
$$\\gamma(z_{ik}) = \\frac{\\pi_k \\mathcal{N}(x_i; \\mu_k, \\Sigma_k)}{\\sum_{j} \\pi_j \\mathcal{N}(x_i; \\mu_j, \\Sigma_j)}$$

**M-step:** Update parameters using responsibilities:
$$\\mu_k = \\frac{\\sum_i \\gamma_{ik} x_i}{\\sum_i \\gamma_{ik}}, \\quad \\pi_k = \\frac{1}{n} \\sum_i \\gamma_{ik}$$

**GMM vs. K-means:** K-means is hard assignment (each point belongs to one cluster). GMM is soft assignment (probabilistic membership). GMM handles elliptical clusters; K-means only spherical.

---

### Q5 ★★☆☆☆ — What is MCMC? Describe Metropolis-Hastings.

**Answer:** Markov Chain Monte Carlo (MCMC) draws samples from a complex target distribution p(x) by constructing a Markov chain whose stationary distribution is p(x).

**Metropolis-Hastings algorithm:**
1. Start at x⁰
2. At each step t: propose x* ~ q(x*|x^(t)) from a proposal distribution
3. Compute acceptance ratio: α = min(1, p(x*) q(x^(t)|x*) / p(x^(t)) q(x*|x^(t)))
4. Accept x* with probability α (set x^(t+1) = x*); otherwise stay at x^(t+1) = x^(t)

**Gibbs sampling** (special case): Sequentially sample each variable from its conditional p(xᵢ | x₋ᵢ) — acceptance rate is always 1.

**Key property:** After a burn-in period, samples are approximately from p(x) even if p is unnormalized. Independent samples are obtained by **thinning** (taking every k-th sample).

---

### Q6 ★★★☆☆ — How to handle imbalanced datasets? Compare resampling methods.

| Method | Description | Effect |
|--------|-------------|--------|
| **Oversampling (SMOTE)** | Synthesize new minority samples between existing ones | Increases minority class size |
| **Undersampling** | Remove majority class samples randomly or via Tomek links | Reduces majority class |
| **Class weights** | Penalize misclassification of minority class more | No data change, built into model |
| **Threshold tuning** | Lower classification threshold for positive class | Trade precision for recall |
| **Ensemble (EasyEnsemble)** | Train multiple models on balanced subsets | Combines bagging + undersampling |

**SMOTE (Synthetic Minority Over-sampling Technique):** For each minority sample, find k nearest neighbors, randomly interpolate: x_new = x + λ(x_neighbor − x).

\`\`\`python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
\`\`\``,

    zh: `## 非监督学习与概率模型面试题

> 来源：《百面机器学习》第4-6章、第8章
> 降维、聚类、概率图模型和采样。

---

### Q1 ★★☆☆☆ — 描述K均值算法的具体步骤。

**解答：**

1. **数据预处理：** 归一化特征，处理异常点
2. **初始化** K个聚类中心（随机选或K-means++更好）
3. **迭代直至收敛：**
   - **分配：** 将每个xᵢ分配到最近中心：cᵢ = argminₖ ‖xᵢ − μₖ‖²
   - **更新：** 重新计算每类中心：μₖ = 所有属于类k的xᵢ的均值
4. **收敛：** 分配不再变化（目标函数J = Σᵢ ‖xᵢ − μ_{cᵢ}‖² 不再下降）

**选择K：** 肘部法则（画J-K曲线）、轮廓系数、Gap统计量。

**K-means局限性：** 假设簇为球形且大小相近；对初始化敏感；不适合非凸形状（改用DBSCAN）。

---

### Q2 ★★☆☆☆ — 什么是PCA？解释最大方差和最小重建误差两种理解方式。

**解答 —— PCA寻找最能代表数据的线性子空间：**

**最大方差：** 寻找单位向量w₁使投影数据方差最大。解为协方差矩阵C = XᵀX/n 的**最大特征值对应的特征向量**。

**最小重建误差：** 寻找k维子空间使原始点与投影点的平均平方距离最小。解相同（等价于X的SVD分解的右奇异向量）。

**算法：**
1. 数据中心化：X ← X − mean(X)
2. 计算协方差矩阵（或用SVD提高数值稳定性）
3. 特征分解，特征向量即为主成分
4. 投影：X_pca = X · V[:, :k]

---

### Q3 ★★☆☆☆ — LDA与PCA有什么区别？各适用什么场景？

| | PCA | LDA |
|--|-----|-----|
| 是否监督 | 否 | 是（使用类别标签）|
| 目标 | 最大化方差 | 最大化类间可分性（Fisher准则）|
| 分量数 | min(n-1, d) | min(c-1, d)，c为类别数 |
| 适用场景 | 无标签降维 | 分类特征提取 |

**LDA假设：** 每类服从相同协方差矩阵的高斯分布（实践中常被违反）。

---

### Q4 ★★☆☆☆ — 解释高斯混合模型（GMM）。EM算法如何工作？

**解答：** GMM将数据建模为K个高斯分量的混合：

p(x) = Σₖ πₖ N(x; μₖ, Σₖ)

**EM算法：**

**E步（期望）：** 计算每个样本属于每个分量的后验概率（软分配）

**M步（最大化）：** 利用软分配权重更新各分量的均值、协方差和混合系数

**GMM vs. K-means：** K-means是硬分配（每点属于一个簇），GMM是软分配（概率隶属度）。GMM能处理椭圆形簇，K-means只能处理球形簇。

---

### Q5 ★★☆☆☆ — 什么是MCMC？描述Metropolis-Hastings算法。

**解答：** MCMC通过构造平稳分布为目标分布p(x)的马尔可夫链，从复杂分布中采样。

**Metropolis-Hastings算法：**
1. 从初始点x⁰出发
2. 每步：从提议分布 q(x*|x^(t)) 提议新样本x*
3. 计算接受率：α = min(1, p(x*)q(x^(t)|x*) / [p(x^(t))q(x*|x^(t))])
4. 以概率α接受x*，否则保持当前样本

**Gibbs采样**（特殊情况）：逐个从条件分布 p(xᵢ|x₋ᵢ) 采样，接受率始终为1。

**关键特性：** 经过预热期后，样本近似来自p(x)（即使p(x)未归一化）。通过**间隔采样**（每k步取一个）获得近似独立的样本。

---

### Q6 ★★★☆☆ — 如何处理类别不均衡数据集？对比各种重采样方法。

| 方法 | 描述 | 效果 |
|------|------|------|
| **过采样（SMOTE）** | 在少数类样本间合成新样本 | 增加少数类数量 |
| **欠采样** | 随机或用Tomek Links删除多数类样本 | 减少多数类数量 |
| **类别权重** | 对少数类误分类施加更大惩罚 | 无需改变数据 |
| **阈值调整** | 降低正类的分类阈值 | 以精确率换召回率 |
| **集成（EasyEnsemble）** | 在平衡子集上训练多个模型 | Bagging + 欠采样 |

**SMOTE：** 对每个少数类样本，找k个最近邻，在两点之间随机插值生成新样本。`,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Interview: Ensemble Learning & Advanced Topics
// Source: 百面机器学习 Ch.11–13
// ─────────────────────────────────────────────────────────────────────────────
export const interviewEnsembleAdvanced: TopicContent = {
  id: 'interview-ensemble-advanced',
  title: { en: 'Interview: Ensemble Methods & Advanced Topics', zh: '面试：集成学习与进阶主题' },
  contentType: 'article',
  content: {
    en: `## Ensemble Methods & Advanced Topics Interview Questions

> Source: *百面机器学习* Ch.11–13 + *机器学习面试题* Ch.7–8
> Random Forests, GBDT, XGBoost, Reinforcement Learning, and GANs.

---

### Q1 ★☆☆☆☆ — What are the types of ensemble methods?

| Type | Strategy | Examples |
|------|---------|---------|
| **Bagging** | Train independent models on bootstrap samples; aggregate by voting/averaging | Random Forest |
| **Boosting** | Train models sequentially; each corrects errors of the previous | AdaBoost, GBDT, XGBoost, LightGBM |
| **Stacking** | Train a meta-learner on the predictions of base learners | Stacked generalization |

**Key difference:**
- Bagging reduces **variance** (prevents overfitting) — parallel training
- Boosting reduces **bias** (improves accuracy) — sequential training, risk overfitting with too many rounds

---

### Q2 ★★☆☆☆ — How does Random Forest work? Why does it outperform a single decision tree?

**Answer:**

1. For each of T trees:
   - Draw a bootstrap sample from training data (sampling with replacement)
   - Grow a deep decision tree, but at each split, consider only a **random subset of m features** (typically m = √d)
2. **Predict:** Majority vote (classification) or average (regression) of all T trees

**Why it works better than a single tree:**
- Single tree: low bias, high variance (sensitive to training data perturbations)
- Random Forest: keeps low bias; variance reduced by averaging T trees (by √T theoretically)
- Feature randomization **decorrelates** trees — averaging correlated trees doesn't reduce variance much

**Key hyperparameters:** n_estimators (more = better, diminishing returns), max_features, max_depth, min_samples_leaf.

---

### Q3 ★★★☆☆ — Explain GBDT (Gradient Boosted Decision Trees). Derive the update rule.

**Answer:** GBDT builds an additive model by sequentially fitting trees to the **negative gradients** of the loss:

$$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$

where hₘ is a tree fit to the pseudo-residuals: rᵢₘ = −∂L(yᵢ, F_{m-1}(xᵢ)) / ∂F_{m-1}(xᵢ)

**For MSE loss:** rᵢₘ = yᵢ − F_{m-1}(xᵢ) — exactly the residuals! This generalizes to any differentiable loss (log-loss, Huber, etc.).

**Intuition:** Each new tree "corrects" whatever the current model gets wrong by moving in the direction of steepest descent on the loss surface.

\`\`\`python
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20
)
gbdt.fit(X_train, y_train)
\`\`\`

---

### Q4 ★★★☆☆ — How does XGBoost differ from GBDT? What innovations does it introduce?

| Feature | GBDT (sklearn) | XGBoost |
|---------|---------------|---------|
| **Objective** | 1st-order gradient (pseudo-residuals) | 2nd-order Taylor expansion (gradient + Hessian) |
| **Regularization** | None built-in | L1 (α) + L2 (λ) on leaf weights, min_child_weight |
| **Tree structure** | Level-wise growth | Leaf-wise growth (LightGBM) or level-wise |
| **Missing values** | Separate imputation step | Learns best direction automatically |
| **Parallelism** | Single-threaded | Multi-threaded, GPU support |
| **Speed** | Moderate | 10-100× faster via column subsampling, caching |

**XGBoost objective per step:**
$$\\mathcal{L}^{(t)} = \\sum_i [g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)] + \\Omega(f_t)$$

where gᵢ = ∂L/∂ŷᵢ^(t-1), hᵢ = ∂²L/∂(ŷᵢ^(t-1))² are the gradient and Hessian.

The **optimal leaf weight** for leaf j: w*ⱼ = −G_j / (H_j + λ), where G_j, H_j are gradient/Hessian sums for that leaf.

---

### Q5 ★★☆☆☆ — What is Reinforcement Learning? Define key components.

**Answer:**

| Component | Definition |
|-----------|-----------|
| **State** s ∈ S | Current observation of the environment |
| **Action** a ∈ A | Decision made by the agent |
| **Reward** r | Scalar feedback signal from environment |
| **Policy** π(a\|s) | Strategy mapping state to action distribution |
| **Value function** V^π(s) | Expected cumulative reward from state s under policy π |
| **Q-function** Q^π(s,a) | Expected cumulative reward starting with action a in state s |

**Core equation — Bellman:**
$$V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)[r(s,a,s') + \\gamma V^\\pi(s')]$$

**Exploration vs. Exploitation:** The fundamental tension — exploit known good actions or explore unknown ones for potential long-term gain.
- **ε-greedy:** Random action with probability ε
- **UCB (Upper Confidence Bound):** Choose action with highest: Q(a) + √(2 log t / n(a))
- **Thompson Sampling:** Sample from posterior belief over Q-values

---

### Q6 ★★☆☆☆ — Explain GANs. What are the training challenges?

**Answer:** GANs pit two networks against each other:
- **Generator G:** Learns to map noise z ~ p(z) to realistic data samples G(z)
- **Discriminator D:** Learns to distinguish real data from G's outputs

**Minimax objective:**
$$\\min_G \\max_D \\; \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]$$

**Training challenges:**

| Problem | Symptom | Solution |
|---------|---------|---------|
| **Mode collapse** | G generates only a few modes | WGAN, minibatch discrimination |
| **Vanishing gradients** | D easily classifies → G gets no gradient | WGAN (Wasserstein distance), label smoothing |
| **Training instability** | Oscillating loss, no convergence | Careful lr scheduling, spectral normalization |

**WGAN improvement:** Replaces JS divergence with Wasserstein distance (Earth Mover's distance) — provides meaningful gradient even when distributions don't overlap. Requires weight clipping or gradient penalty (WGAN-GP).`,

    zh: `## 集成学习与进阶主题面试题

> 来源：《百面机器学习》第11-13章 + 《机器学习面试题》第7-8章
> 随机森林、GBDT、XGBoost、强化学习和GAN。

---

### Q1 ★☆☆☆☆ — 集成学习有哪些类型？

| 类型 | 策略 | 典型例子 |
|------|------|---------|
| **Bagging** | 用Bootstrap样本独立训练多个模型，投票/平均聚合 | 随机森林 |
| **Boosting** | 串行训练，每个模型纠正前一个的错误 | AdaBoost、GBDT、XGBoost、LightGBM |
| **Stacking** | 用基学习器预测值训练元学习器 | 堆叠泛化 |

**关键区别：**
- Bagging减少**方差**（防止过拟合）—— 并行训练
- Boosting减少**偏差**（提升精度）—— 串行训练，轮数过多有过拟合风险

---

### Q2 ★★☆☆☆ — 随机森林如何工作？为什么比单棵决策树效果好？

**解答：**

1. 对每棵树T：
   - 有放回地从训练数据中抽取Bootstrap样本
   - 生成深度决策树，但每次分裂只考虑**随机选取的m个特征**（通常m=√d）
2. **预测：** 所有T棵树的多数投票（分类）或平均（回归）

**为什么比单棵树好：**
- 单棵树：低偏差、高方差（对训练数据扰动敏感）
- 随机森林：保持低偏差；通过平均T棵树理论上将方差降低√T倍
- 特征随机化使树之间**去相关**——平均相关树不能有效降低方差

---

### Q3 ★★★☆☆ — 解释GBDT。推导更新规则。

**解答：** GBDT通过串行地将树拟合到**损失函数的负梯度**来构建加法模型：

Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)

其中hₘ是拟合伪残差 rᵢₘ = −∂L(yᵢ, Fₘ₋₁(xᵢ)) / ∂Fₘ₋₁(xᵢ) 的树。

**对于MSE损失：** rᵢₘ = yᵢ − Fₘ₋₁(xᵢ) —— 正好是残差！可推广到任意可微损失。

**直觉：** 每棵新树"纠正"当前模型的错误，沿损失曲面的最速下降方向移动。

---

### Q4 ★★★☆☆ — XGBoost与GBDT有什么区别？有哪些创新？

| 特性 | GBDT | XGBoost |
|------|------|---------|
| **目标函数** | 一阶梯度（伪残差）| 二阶Taylor展开（梯度+Hessian）|
| **正则化** | 无内置 | L1(α) + L2(λ)叶权重正则 |
| **缺失值** | 需单独处理 | 自动学习最佳方向 |
| **速度** | 中等 | 通过列采样、缓存快10-100倍 |

**关键创新：** XGBoost用二阶Taylor展开，利用Hessian更精准地建立分裂标准，并推导出解析的最优叶权重：w*ⱼ = −G_j / (H_j + λ)。

---

### Q5 ★★☆☆☆ — 什么是强化学习？定义关键组件。

**解答：**

| 组件 | 定义 |
|------|------|
| **状态** s ∈ S | 环境的当前观测 |
| **动作** a ∈ A | 智能体做出的决策 |
| **奖励** r | 来自环境的标量反馈信号 |
| **策略** π(a\|s) | 状态到动作分布的映射 |
| **价值函数** V^π(s) | 策略π下从状态s的期望累积奖励 |

**核心方程——Bellman方程：** 当前状态的价值等于即时奖励加上折扣的后续状态价值的期望。

**探索与利用的权衡：** 核心张力——利用已知好动作还是探索未知动作以获得长期收益。
- **ε-贪婪：** 以概率ε随机选择动作
- **UCB（置信区间上界）：** 选择Q(a) + √(2 log t / n(a)) 最大的动作
- **Thompson采样：** 从Q值的后验置信度中采样

---

### Q6 ★★☆☆☆ — 解释GAN。训练挑战有哪些？

**解答：** GAN让两个网络对抗：
- **生成器G：** 学习将噪声z映射到真实数据样本G(z)
- **判别器D：** 学习区分真实数据和G的输出

**极小极大目标：** min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]

**训练挑战：**

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| **模式坍塌** | G只生成少数几种模式 | WGAN、小批量判别 |
| **梯度消失** | D轻易分类→G得不到梯度 | WGAN（Wasserstein距离）、标签平滑 |
| **训练不稳定** | 损失振荡，不收敛 | 学习率调度、谱归一化 |

**WGAN改进：** 用Wasserstein距离（推土机距离）替代JS散度——即使分布不重叠也能提供有意义的梯度。`,
  },
}
