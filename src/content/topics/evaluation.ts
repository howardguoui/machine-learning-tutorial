import type { TopicContent } from '../types'

export const metrics: TopicContent = {
  id: 'evaluation-metrics',
  title: { en: 'Model Evaluation Metrics', zh: '模型评估指标' },
  contentType: 'code',
  content: {
    en: `Choosing the right metric is as important as choosing the right model. **Accuracy alone is deceptive** in imbalanced datasets.

## Classification Metrics

\`\`\`python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Imbalanced dataset: 95% negative, 5% positive
X, y = make_classification(n_samples=10000, n_features=20, weights=[0.95, 0.05],
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# --- Core Metrics ---
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  TN={tn:4d}  FP={fp:4d}")
print(f"  FN={fn:4d}  TP={tp:4d}")

# Why accuracy misleads on imbalanced data:
print(f"\\nAccuracy:   {accuracy_score(y_test, y_pred):.4f}")
# A dummy classifier predicting all-negative gets ~95% accuracy!
y_dummy = np.zeros(len(y_test), dtype=int)
print(f"Dummy (all-neg) accuracy: {accuracy_score(y_test, y_dummy):.4f}")

# Better metrics for imbalance:
print(f"\\nPrecision:  {precision_score(y_test, y_pred):.4f}")   # TP/(TP+FP)
print(f"Recall:     {recall_score(y_test, y_pred):.4f}")        # TP/(TP+FN)
print(f"F1:         {f1_score(y_test, y_pred):.4f}")            # harmonic mean
print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.4f}")       # area under ROC
print(f"Avg Prec:   {average_precision_score(y_test, y_prob):.4f}")  # area under PR
\`\`\`

## Regression Metrics

\`\`\`python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:   {mse:.2f}   (in units²)")
print(f"RMSE:  {rmse:.2f}  (same units as target)")
print(f"MAE:   {mae:.2f}   (robust to outliers)")
print(f"MAPE:  {mape:.4f}  (relative error)")
print(f"R²:    {r2:.4f}   (1.0 = perfect, 0 = baseline)")
\`\`\`

## Cross-Validation

\`\`\`python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Standard k-fold CV
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_results = cross_validate(
    clf, X, y, cv=5,
    scoring=['accuracy', 'roc_auc', 'f1'],
    return_train_score=True
)

print("5-Fold Cross-Validation Results:")
for metric in ['accuracy', 'roc_auc', 'f1']:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    print(f"  {metric:<10}: test={test_scores.mean():.4f}±{test_scores.std():.4f}"
          f"  train={train_scores.mean():.4f}")

# Stratified k-fold: preserves class balance in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    clf.fit(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    print(f"Fold {fold+1}: val_acc={val_acc:.4f}, class_balance={y_val.mean():.3f}")
\`\`\`

| Metric | When to Use | Formula |
|---|---|---|
| **Accuracy** | Balanced classes | (TP+TN)/(TP+TN+FP+FN) |
| **Precision** | Cost of false positives high | TP/(TP+FP) |
| **Recall** | Cost of false negatives high | TP/(TP+FN) |
| **F1** | Balance precision & recall | 2×P×R/(P+R) |
| **ROC-AUC** | Compare models, rank quality | Area under ROC curve |
| **RMSE** | Regression, outlier-sensitive | √(mean squared errors) |
| **MAE** | Regression, robust | mean(|y - ŷ|) |`,

    zh: `选择正确的指标与选择正确的模型同样重要。在不平衡数据集中，**准确率单独使用会误导人**。

## 分类指标

\`\`\`python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred, target_names=['负例', '正例']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
\`\`\`

| 指标 | 使用场景 | 公式 |
|---|---|---|
| **准确率** | 均衡类别 | (TP+TN)/(TP+TN+FP+FN) |
| **精确率** | 假阳性代价高 | TP/(TP+FP) |
| **召回率** | 假阴性代价高 | TP/(TP+FN) |
| **F1** | 平衡精确率和召回率 | 2×P×R/(P+R) |
| **ROC-AUC** | 比较模型，排名质量 | ROC曲线下面积 |`,
  },
}

export const biasVariance: TopicContent = {
  id: 'bias-variance',
  title: { en: 'Bias-Variance Tradeoff', zh: '偏差-方差权衡' },
  contentType: 'article',
  content: {
    en: `The bias-variance tradeoff is the **fundamental tension in supervised learning**: between models that are too simple (high bias) and models that are too complex (high variance).

## Understanding the Tradeoff

\`\`\`python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# True underlying function: y = sin(x) + noise
np.random.seed(42)
def true_function(x): return np.sin(x)
def generate_data(n, noise=0.3):
    x = np.linspace(-3, 3, n)
    y = true_function(x) + np.random.normal(0, noise, n)
    return x.reshape(-1, 1), y

X, y = generate_data(50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Diagnosis'}")
print("-" * 55)

for degree in [1, 2, 5, 10, 20]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr',   LinearRegression()),
    ])
    model.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse  = mean_squared_error(y_test,  model.predict(X_test))

    if test_mse < 0.15:     diagnosis = "✅ Good fit"
    elif train_mse > 0.2:   diagnosis = "⬆️  High bias (underfitting)"
    else:                   diagnosis = "⬆️  High variance (overfitting)"

    print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {diagnosis}")
\`\`\`

## Regularization as Variance Reduction

\`\`\`python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import validation_curve

X, y = generate_data(100)
X_features = PolynomialFeatures(degree=10).fit_transform(X)

alphas = np.logspace(-4, 2, 20)
train_scores, val_scores = validation_curve(
    Ridge(), X_features, y,
    param_name='alpha', param_range=alphas,
    cv=5, scoring='neg_mean_squared_error'
)

best_alpha = alphas[np.argmax(val_scores.mean(axis=1))]
print(f"Best regularization strength: α = {best_alpha:.5f}")

ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_features, y)
\`\`\`

## Learning Curves: Diagnosing Problems

\`\`\`python
from sklearn.model_selection import learning_curve

def diagnose_model(model, X, y, name="Model"):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)

    final_gap = val_mse[-1] - train_mse[-1]
    print(f"\\n{name}:")
    print(f"  Final train MSE: {train_mse[-1]:.4f}")
    print(f"  Final val MSE:   {val_mse[-1]:.4f}")
    print(f"  Gap:             {final_gap:.4f}")

    if train_mse[-1] > 0.1 and val_mse[-1] > 0.1:
        print("  → High Bias: add more features or increase model complexity")
    elif final_gap > 0.05:
        print("  → High Variance: regularize, add data, reduce complexity")
    else:
        print("  → Good fit!")

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)

diagnose_model(LinearRegression(), X, y, "Linear Regression")
from sklearn.tree import DecisionTreeRegressor
diagnose_model(DecisionTreeRegressor(max_depth=None), X, y, "Deep Decision Tree")
diagnose_model(DecisionTreeRegressor(max_depth=3), X, y, "Shallow Decision Tree")
\`\`\`

## The Tradeoff Summary

| Condition | Train Error | Test Error | Fix |
|---|---|---|---|
| **High Bias** (underfitting) | High | High | More features, complex model |
| **High Variance** (overfitting) | Low | High | Regularization, more data |
| **Good Fit** | Low | Low | Deploy! |
| **Both high** | High | Very High | Debug data quality |

> **Modern insight**: Very deep neural networks can achieve both low training *and* test error (beyond the classical bias-variance tradeoff) via the **double descent** phenomenon. Regularization techniques (dropout, weight decay, data augmentation) are key.`,

    zh: `偏差-方差权衡是**监督学习中的基本张力**：在太简单（高偏差）和太复杂（高方差）的模型之间。

## 理解权衡

- **欠拟合（高偏差）**：模型太简单，无法捕捉数据中的真实模式
- **过拟合（高方差）**：模型太复杂，记住了训练数据中的噪声

\`\`\`python
# 多项式回归不同次数的比较
for degree in [1, 2, 5, 10, 20]:
    # degree=1: 欠拟合（高偏差）
    # degree=5: 良好拟合
    # degree=20: 过拟合（高方差）
\`\`\`

| 情况 | 训练误差 | 测试误差 | 解决方法 |
|---|---|---|---|
| **高偏差**（欠拟合） | 高 | 高 | 增加特征，提高模型复杂度 |
| **高方差**（过拟合） | 低 | 高 | 正则化，增加数据 |
| **良好拟合** | 低 | 低 | 部署！ |`,
  },
}
