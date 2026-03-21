import type { TopicContent } from '../types'

export const linearRegression: TopicContent = {
  id: 'linear-regression',
  title: { en: 'Linear & Logistic Regression', zh: '线性回归与逻辑回归' },
  contentType: 'code',
  content: {
    en: `Linear models are the workhorses of ML — simple, interpretable, and surprisingly powerful when features are well-engineered.

## Linear Regression from Scratch

\`\`\`python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionScratch:
    """
    Linear Regression: ŷ = Xw + b
    Loss: MSE = (1/n) Σ (ŷ_i - y_i)²
    Solution: w* = (X^T X)^{-1} X^T y  (Normal Equation)
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, d = X.shape
        # Add bias column
        X_aug = np.column_stack([X, np.ones(n)])
        # Normal equation: θ = (X^T X)^{-1} X^T y
        theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
        self.weights = theta[:-1]
        self.bias = theta[-1]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        return r2_score(y, self.predict(X))

# Generate data and evaluate
X, y = make_regression(n_samples=500, n_features=5, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegressionScratch()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")
print(f"Weights:  {model.weights.round(3)}")
\`\`\`

## Gradient Descent Training

\`\`\`python
class LinearRegressionGD:
    """Linear regression via gradient descent — scales to large datasets."""

    def __init__(self, lr=0.01, n_epochs=1000, batch_size=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size  # None = full gradient descent

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []

        for epoch in range(self.n_epochs):
            # Mini-batch SGD
            if self.batch_size:
                idx = np.random.choice(n, self.batch_size, replace=False)
                X_b, y_b = X[idx], y[idx]
            else:
                X_b, y_b = X, y

            # Forward pass
            y_pred = X_b @ self.w + self.b
            residuals = y_pred - y_b

            # Gradients
            dw = (2 / len(X_b)) * X_b.T @ residuals
            db = (2 / len(X_b)) * residuals.sum()

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                loss = mean_squared_error(y, X @ self.w + self.b)
                self.losses.append(loss)
                print(f"Epoch {epoch:4d}: MSE = {loss:.2f}")

        return self

model_gd = LinearRegressionGD(lr=0.01, n_epochs=500, batch_size=64)
model_gd.fit(X_train, y_train)
\`\`\`

## Logistic Regression for Classification

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score

# Production-ready logistic regression
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       stratify=y, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# L2 regularization (C=1/λ, smaller C = stronger regularization)
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_s, y_train)

y_pred = lr_model.predict(X_test_s)
y_prob = lr_model.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred,
      target_names=['malignant', 'benign']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
\`\`\`

## Regularization: Ridge, Lasso, ElasticNet

\`\`\`python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

X, y = make_regression(n_samples=200, n_features=100, n_informative=10,
                        noise=10, random_state=42)

models = {
    'OLS':        LinearRegressionScratch(),
    'Ridge':      Ridge(alpha=1.0),
    'Lasso':      Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    if name == 'OLS':
        model.fit(X, y)
        cv_scores = [r2_score(y, model.predict(X))]  # simplified
    else:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name:12s}: R² = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
\`\`\`

| Method | Penalty | Effect |
|---|---|---|
| **OLS** | None | Can overfit with many features |
| **Ridge (L2)** | λΣwᵢ² | Shrinks all weights, keeps all features |
| **Lasso (L1)** | λΣ\|wᵢ\| | Drives some weights to exact zero (feature selection) |
| **ElasticNet** | Mix of L1+L2 | Best of both worlds for correlated features |`,

    zh: `线性模型是ML的主力军——简单、可解释，当特征工程良好时出奇地强大。

## 从零实现线性回归

\`\`\`python
class LinearRegressionScratch:
    """
    线性回归: ŷ = Xw + b
    损失: MSE = (1/n) Σ (ŷ_i - y_i)²
    解: w* = (X^T X)^{-1} X^T y  (正规方程)
    """
    def fit(self, X, y):
        n, d = X.shape
        X_aug = np.column_stack([X, np.ones(n)])
        theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
        self.weights = theta[:-1]
        self.bias = theta[-1]
        return self
\`\`\`

| 方法 | 惩罚项 | 效果 |
|---|---|---|
| **OLS** | 无 | 特征多时可能过拟合 |
| **Ridge (L2)** | λΣwᵢ² | 缩小所有权重，保留所有特征 |
| **Lasso (L1)** | λΣ\|wᵢ\| | 将部分权重驱至零（特征选择） |
| **ElasticNet** | L1+L2混合 | 相关特征的最佳选择 |`,
  },
}

export const decisionTrees: TopicContent = {
  id: 'decision-trees',
  title: { en: 'Decision Trees & Ensembles', zh: '决策树与集成方法' },
  contentType: 'code',
  content: {
    en: `Decision trees are intuitive models that learn **if-then rules** from data. Ensemble methods combine many trees for dramatically better performance.

## Decision Tree: How it Works

\`\`\`python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A tree splits features to maximize information gain (or minimize Gini impurity)
tree = DecisionTreeClassifier(
    max_depth=4,                  # prevent overfitting
    min_samples_leaf=5,           # at least 5 samples per leaf
    criterion='gini',             # 'entropy' also available
    random_state=42
)
tree.fit(X_train, y_train)

print(f"Train accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test  accuracy: {tree.score(X_test, y_test):.3f}")
print("\\nTree rules (max depth 3):")
print(export_text(tree, feature_names=iris.feature_names, max_depth=3))
\`\`\`

## Random Forest

\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
import pandas as pd

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       stratify=y, random_state=42)

# Random Forest: bagging of trees, each trained on:
# - Random bootstrap sample of training data
# - Random subset of features at each split (sqrt(n_features) by default)
rf = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=None,        # grow full trees (pruned by min_samples_leaf)
    min_samples_leaf=2,
    max_features='sqrt',   # features to consider at each split
    n_jobs=-1,             # use all CPU cores
    random_state=42,
)
rf.fit(X_train, y_train)

print(f"RF Test Accuracy: {rf.score(X_test, y_test):.4f}")

# Feature importance
feature_names = load_breast_cancer().feature_names
importances = pd.Series(rf.feature_importances_, index=feature_names)
print("\\nTop 5 features:")
print(importances.sort_values(ascending=False).head())
\`\`\`

## Gradient Boosting (XGBoost)

\`\`\`python
# Gradient Boosting: sequential ensemble where each tree corrects previous errors
# XGBoost/LightGBM dominate structured data competitions

try:
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,       # shrinkage: controls contribution of each tree
        subsample=0.8,             # row sampling: prevents overfitting
        colsample_bytree=0.8,      # feature sampling per tree
        use_label_encoder=False,
        eval_metric='auc',
        early_stopping_rounds=20,  # stop if no improvement after 20 rounds
        random_state=42,
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
\`\`\`

## Algorithm Comparison

\`\`\`python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':      AdaBoostClassifier(n_estimators=100, random_state=42),
}

print(f"{'Model':<20} {'CV Accuracy'}")
print("-" * 35)
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name:<20} {scores.mean():.4f} ± {scores.std():.4f}")
\`\`\`

| Method | Key Idea | Pros | Cons |
|---|---|---|---|
| **Decision Tree** | Recursive splits | Interpretable | Overfits easily |
| **Random Forest** | Bagging + random features | Fast, robust | Less interpretable |
| **Gradient Boost** | Sequential boosting | Best accuracy | Slower to train |
| **AdaBoost** | Weighted examples | Simple | Sensitive to outliers |`,

    zh: `决策树是直观的模型，它从数据中学习**if-then规则**。集成方法结合多棵树以获得显著更好的性能。

## 随机森林

随机森林：每棵树的训练使用：
- 训练数据的随机bootstrap样本
- 每次分裂时随机的特征子集（默认为sqrt(n_features)）

\`\`\`python
rf = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_features='sqrt',   # 每次分裂考虑的特征数
    n_jobs=-1,             # 使用所有CPU核心
    random_state=42,
)
\`\`\`

| 方法 | 核心思想 | 优势 | 劣势 |
|---|---|---|---|
| **决策树** | 递归分裂 | 可解释 | 容易过拟合 |
| **随机森林** | 装袋+随机特征 | 快速、鲁棒 | 解释性较差 |
| **梯度提升** | 序列提升 | 最佳准确度 | 训练较慢 |`,
  },
}

export const svm: TopicContent = {
  id: 'svm-knn',
  title: { en: 'SVM & k-NN', zh: '支持向量机与k近邻' },
  contentType: 'code',
  content: {
    en: `SVMs find the optimal decision boundary; k-NN uses the simplest possible rule — look at your neighbors.

## Support Vector Machine

\`\`\`python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV

# SVM is sensitive to feature scale — always scale first
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM pipeline: scale → classify
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
])
svm_pipeline.fit(X_train, y_train)
print(f"SVM (RBF kernel) accuracy: {svm_pipeline.score(X_test, y_test):.4f}")

# Hyperparameter tuning: C (margin softness) and gamma (kernel width)
param_grid = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01],
}
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.best_estimator_.score(X_test, y_test):.4f}")
\`\`\`

## k-Nearest Neighbors

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# k-NN: classify by majority vote among k nearest neighbors
# No "training" — just stores all examples (lazy learner)

X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# Find optimal k
k_values = range(1, 30)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best k: {best_k}, CV Accuracy: {max(cv_scores):.4f}")

# Final model
knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X, y)
\`\`\`

## When to Use Which

| Algorithm | Best For | Avoid When |
|---|---|---|
| **Linear SVM** | High-dim text/gene data | Very large n (slow) |
| **RBF SVM** | Small/medium datasets, non-linear | > 10k samples |
| **k-NN** | Low-dim data, interpretability | High-dim, large n |`,

    zh: `SVM找到最优决策边界；k-NN使用最简单的规则——看你的邻居。

## 支持向量机

SVM对特征尺度敏感——**始终先进行缩放**。

\`\`\`python
# SVM流水线：缩放 → 分类
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale')),
])
\`\`\`

| 算法 | 最适合 | 避免使用 |
|---|---|---|
| **线性SVM** | 高维文本/基因数据 | 非常大的n（慢） |
| **RBF SVM** | 中小型数据集，非线性 | >10k样本 |
| **k-NN** | 低维数据，可解释性 | 高维，大n |`,
  },
}
