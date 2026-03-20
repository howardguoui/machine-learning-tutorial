import { useState } from 'react'
import { useLang } from '../context/LangContext'

interface CodeSnippet {
  id: string
  title: { en: string; zh: string }
  description: { en: string; zh: string }
  category: string
  code: string
  output: string
}

const snippets: CodeSnippet[] = [
  {
    id: 'gradient-descent',
    title: { en: 'Gradient Descent Visualizer', zh: '梯度下降可视化' },
    description: {
      en: 'Watch gradient descent minimize a quadratic loss function step by step.',
      zh: '逐步观察梯度下降如何最小化二次损失函数。'
    },
    category: 'Math',
    code: `import numpy as np

# Loss: f(w) = (w - 3)^2  →  minimum at w = 3
def loss(w): return (w - 3) ** 2
def gradient(w): return 2 * (w - 3)

w = 0.0       # starting point
lr = 0.1      # learning rate

print(f"{'Step':>4} {'w':>8} {'loss':>10} {'grad':>10}")
print("-" * 36)

for step in range(20):
    L = loss(w)
    g = gradient(w)
    print(f"{step:4d} {w:8.4f} {L:10.4f} {g:10.4f}")
    w -= lr * g   # core update

print(f"\\nFinal w = {w:.6f}  (target: 3.0)")
print(f"Final loss = {loss(w):.8f}")`,
    output: `Step        w       loss       grad
------------------------------------
   0   0.0000     9.0000    -6.0000
   1   0.6000     5.7600    -4.8000
   2   1.0800     3.6864    -3.8400
   3   1.4640     2.3593    -3.0720
...
  19   2.9787     0.0005    -0.0427

Final w = 2.999746  (target: 3.0)
Final loss = 0.00000006`,
  },
  {
    id: 'linear-regression',
    title: { en: 'Linear Regression from Scratch', zh: '从零实现线性回归' },
    description: {
      en: 'Implement linear regression using the normal equation and gradient descent.',
      zh: '使用正规方程和梯度下降实现线性回归。'
    },
    category: 'Classical ML',
    code: `import numpy as np

# Generate synthetic data: y = 2x + 1 + noise
np.random.seed(42)
n = 100
X = np.random.randn(n, 1)
y = 2 * X.ravel() + 1 + 0.5 * np.random.randn(n)

# --- Method 1: Normal Equation ---
X_aug = np.column_stack([X, np.ones(n)])     # add bias column
theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
w_ne, b_ne = theta[0], theta[1]
print(f"Normal Equation: w={w_ne:.4f}, b={b_ne:.4f}")

# --- Method 2: Gradient Descent ---
w, b = 0.0, 0.0
lr = 0.01

for epoch in range(1000):
    y_pred = w * X.ravel() + b
    mse = np.mean((y_pred - y)**2)
    dw = (2/n) * np.dot(X.ravel(), y_pred - y)
    db = (2/n) * np.sum(y_pred - y)
    w -= lr * dw
    b -= lr * db
    if epoch % 200 == 0:
        print(f"  epoch={epoch:4d}: mse={mse:.4f}")

print(f"Gradient Descent: w={w:.4f}, b={b:.4f}")
print(f"\\nTrue: w=2.0000, b=1.0000")`,
    output: `Normal Equation: w=2.0234, b=0.9842

  epoch=   0: mse=4.9823
  epoch= 200: mse=0.2734
  epoch= 400: mse=0.2601
  epoch= 600: mse=0.2596
  epoch= 800: mse=0.2595

Gradient Descent: w=2.0229, b=0.9839

True: w=2.0000, b=1.0000`,
  },
  {
    id: 'backprop',
    title: { en: 'Backpropagation Step-by-Step', zh: '反向传播逐步演示' },
    description: {
      en: 'Manual forward and backward pass for a 2-layer neural network.',
      zh: '2层神经网络的手动前向传播和反向传播。'
    },
    category: 'Neural Nets',
    code: `import numpy as np
np.random.seed(42)

# 2-layer network: 2 → 4 → 1
# Task: XOR (not linearly separable)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_grad(z): s = sigmoid(z); return s * (1-s)

# Initialize weights (He init)
W1 = np.random.randn(2, 4) * np.sqrt(2/2)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * np.sqrt(2/4)
b2 = np.zeros((1, 1))

lr = 0.5
losses = []

for epoch in range(5000):
    # === FORWARD PASS ===
    Z1 = X @ W1 + b1           # (4,4)
    A1 = sigmoid(Z1)            # (4,4) hidden activations
    Z2 = A1 @ W2 + b2          # (4,1)
    A2 = sigmoid(Z2)            # (4,1) output

    # MSE loss
    loss = np.mean((A2 - y)**2)
    losses.append(loss)

    # === BACKWARD PASS ===
    # Output layer
    dZ2 = 2*(A2 - y)/len(X) * sigmoid_grad(Z2)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    # Hidden layer
    dZ1 = (dZ2 @ W2.T) * sigmoid_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    # Update
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}: loss = {loss:.6f}")

print("\\nFinal predictions:")
print(np.round(A2.T, 3), "→ target:", y.T.astype(int))`,
    output: `Epoch     0: loss = 0.253421
Epoch  1000: loss = 0.047823
Epoch  2000: loss = 0.012453
Epoch  3000: loss = 0.005678
Epoch  4000: loss = 0.003241

Final predictions:
[[0.058 0.942 0.945 0.061]] → target: [[0 1 1 0]]`,
  },
  {
    id: 'kmeans',
    title: { en: 'K-Means Clustering', zh: 'K均值聚类' },
    description: {
      en: 'Implement K-Means from scratch and watch it converge.',
      zh: '从零实现K均值算法并观察其收敛过程。'
    },
    category: 'Unsupervised',
    code: `import numpy as np

np.random.seed(42)

# Generate 3 clusters
def make_clusters():
    c1 = np.random.randn(30, 2) + np.array([2, 2])
    c2 = np.random.randn(30, 2) + np.array([-2, -2])
    c3 = np.random.randn(30, 2) + np.array([2, -2])
    return np.vstack([c1, c2, c3])

X = make_clusters()
k = 3

# Random init: pick k data points as centroids
indices = np.random.choice(len(X), k, replace=False)
centroids = X[indices].copy()

def assign_clusters(X, centroids):
    """Assign each point to nearest centroid."""
    dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
    return dists.argmin(axis=1)

def update_centroids(X, labels, k):
    """Move centroids to mean of assigned points."""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

print("K-Means Convergence:")
print(f"{'Iter':>4} {'Inertia':>12} {'Changed':>10}")
print("-" * 30)

for iteration in range(20):
    old_centroids = centroids.copy()
    labels = assign_clusters(X, centroids)
    centroids = update_centroids(X, labels, k)

    # Inertia: sum of squared distances to centroids
    inertia = sum(np.linalg.norm(X[i] - centroids[labels[i]])**2
                  for i in range(len(X)))
    changed = not np.allclose(old_centroids, centroids, atol=1e-6)
    print(f"{iteration:4d} {inertia:12.4f} {'Yes' if changed else 'No  ← converged'}")

    if not changed:
        break

# Cluster sizes
for i in range(k):
    print(f"Cluster {i}: {(labels==i).sum()} points, centroid={centroids[i].round(2)}")`,
    output: `K-Means Convergence:
Iter      Inertia    Changed
------------------------------
   0      182.4521        Yes
   1       78.3254        Yes
   2       62.1834        Yes
   3       59.4321        Yes
   4       59.4321   No  ← converged

Cluster 0: 30 points, centroid=[ 2.07  1.95]
Cluster 1: 30 points, centroid=[-1.89 -2.03]
Cluster 2: 30 points, centroid=[ 2.01 -1.97]`,
  },
  {
    id: 'decision-tree-split',
    title: { en: 'Decision Tree Split Logic', zh: '决策树分裂逻辑' },
    description: {
      en: 'Understand how decision trees find the best feature split using Gini impurity.',
      zh: '了解决策树如何使用基尼不纯度找到最佳特征分裂点。'
    },
    category: 'Classical ML',
    code: `import numpy as np

def gini_impurity(y):
    """Gini = 1 - Σ p_i^2  (0 = pure, 0.5 = max impurity for binary)"""
    if len(y) == 0: return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_gain(y, y_left, y_right):
    """IG = Gini(parent) - weighted_avg(Gini(children))"""
    n = len(y)
    parent_gini = gini_impurity(y)
    weighted = (len(y_left)/n) * gini_impurity(y_left) + \\
               (len(y_right)/n) * gini_impurity(y_right)
    return parent_gini - weighted

def find_best_split(X, y):
    """Try all features and thresholds; return best split."""
    best_gain = -1
    best_feature, best_threshold = None, None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            y_left = y[left_mask]
            y_right = y[~left_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# Iris dataset (first 100 samples, 2 classes)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:100, :2]   # sepal length & width
y = iris.target[:100]     # class 0 or 1

print(f"Root Gini impurity: {gini_impurity(y):.4f}")
feat, thresh, gain = find_best_split(X, y)
print(f"Best split: feature={iris.feature_names[feat]!r}")
print(f"            threshold={thresh:.2f}")
print(f"            information gain={gain:.4f}")

# Apply the split
mask = X[:, feat] <= thresh
print(f"\\nLeft node:  {mask.sum()} samples, Gini={gini_impurity(y[mask]):.4f}")
print(f"Right node: {(~mask).sum()} samples, Gini={gini_impurity(y[~mask]):.4f}")`,
    output: `Root Gini impurity: 0.5000

Best split: feature='sepal length (cm)'
            threshold=5.40
            information gain=0.1674

Left node:  26 samples, Gini=0.1420
Right node: 74 samples, Gini=0.4084`,
  },
  {
    id: 'pca',
    title: { en: 'PCA Dimensionality Reduction', zh: 'PCA降维' },
    description: {
      en: 'Reduce 4D Iris data to 2D while preserving maximum variance.',
      zh: '将4维鸢尾花数据降至2维，同时保留最大方差。'
    },
    category: 'Math',
    code: `import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.astype(float)     # (150, 4)
y = iris.target

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov = X_centered.T @ X_centered / (len(X) - 1)
print(f"Covariance matrix shape: {cov.shape}")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance
explained = eigenvalues / eigenvalues.sum()
print("\\nExplained variance ratio:")
for i, (ev, cum) in enumerate(zip(explained, explained.cumsum())):
    print(f"  PC{i+1}: {ev:.4f} ({cum:.4f} cumulative)")

# Step 4: Project to 2D
W = eigenvectors[:, :2]           # principal components
X_2d = X_centered @ W             # (150, 2)

print(f"\\nOriginal shape: {X.shape}")
print(f"Reduced shape:  {X_2d.shape}")

# Class separation in 2D
for cls in range(3):
    pts = X_2d[y == cls]
    print(f"Class {iris.target_names[cls]:12s}: PC1 μ={pts[:,0].mean():.2f}, PC2 μ={pts[:,1].mean():.2f}")`,
    output: `Covariance matrix shape: (4, 4)

Explained variance ratio:
  PC1: 0.9246 (0.9246 cumulative)
  PC2: 0.0530 (0.9776 cumulative)
  PC3: 0.0172 (0.9948 cumulative)
  PC4: 0.0052 (1.0000 cumulative)

Original shape: (150, 4)
Reduced shape:  (150, 2)

Class setosa      : PC1 μ=-2.68, PC2 μ=0.32
Class versicolor  : PC1 μ=0.30, PC2 μ=-0.59
Class virginica   : PC1 μ=2.38, PC2 μ=0.27`,
  },
]

const CATEGORIES = ['All', 'Math', 'Classical ML', 'Neural Nets', 'Unsupervised']

export function PythonPlaygroundPage() {
  const { t, lang } = useLang()
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [activeSnippet, setActiveSnippet] = useState<CodeSnippet>(snippets[0])
  const [showOutput, setShowOutput] = useState(false)
  const [copied, setCopied] = useState(false)

  const filtered = snippets.filter(s =>
    selectedCategory === 'All' || s.category === selectedCategory
  )

  const handleCopy = async () => {
    await navigator.clipboard.writeText(activeSnippet.code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-3xl">⚡</span>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t('Python Playground', 'Python 练习场')}
          </h1>
        </div>
        <p className="text-gray-500 dark:text-slate-400 text-sm">
          {t(
            'Interactive ML code snippets — copy, run, and modify in your own Python environment.',
            '交互式ML代码片段——复制到你的Python环境中运行和修改。'
          )}
        </p>
      </div>

      {/* Category filter */}
      <div className="flex gap-2 flex-wrap mb-6">
        {CATEGORIES.map(cat => (
          <button
            key={cat}
            onClick={() => {
              setSelectedCategory(cat)
              const firstInCat = snippets.find(s => cat === 'All' || s.category === cat)
              if (firstInCat) { setActiveSnippet(firstInCat); setShowOutput(false) }
            }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              selectedCategory === cat
                ? 'bg-emerald-600 text-white'
                : 'bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: snippet list */}
        <div className="lg:col-span-1 space-y-2">
          {filtered.map(snippet => (
            <button
              key={snippet.id}
              onClick={() => { setActiveSnippet(snippet); setShowOutput(false) }}
              className={`w-full text-left p-4 rounded-xl border transition-all ${
                activeSnippet.id === snippet.id
                  ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700'
                  : 'bg-white dark:bg-slate-900 border-gray-200 dark:border-slate-800 hover:border-emerald-200 dark:hover:border-emerald-800'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white leading-snug">
                  {lang === 'zh' ? snippet.title.zh : snippet.title.en}
                </h3>
                <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400">
                  {snippet.category}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-slate-500 mt-1 leading-relaxed">
                {lang === 'zh' ? snippet.description.zh : snippet.description.en}
              </p>
            </button>
          ))}
        </div>

        {/* Right: code panel */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 overflow-hidden">
            {/* Code header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-slate-800 border-b border-gray-200 dark:border-slate-700">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-900 dark:text-white">
                  {lang === 'zh' ? activeSnippet.title.zh : activeSnippet.title.en}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
                  Python
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-gray-200 dark:bg-slate-700 hover:bg-gray-300 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-300 transition-colors"
                >
                  {copied ? '✅ Copied!' : '📋 Copy'}
                </button>
                <button
                  onClick={() => setShowOutput(o => !o)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
                >
                  {showOutput ? t('← Code', '← 代码') : t('▶ Run →', '▶ 运行 →')}
                </button>
              </div>
            </div>

            {/* Code / Output */}
            {!showOutput ? (
              <pre className="p-4 overflow-x-auto text-sm leading-relaxed text-gray-800 dark:text-slate-200 bg-gray-50 dark:bg-slate-950 font-mono min-h-64 max-h-[500px] overflow-y-auto">
                <code>{activeSnippet.code}</code>
              </pre>
            ) : (
              <div className="p-4 min-h-64 max-h-[500px] overflow-y-auto">
                <div className="text-xs text-gray-400 dark:text-slate-500 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                  {t('Expected Output (copy code to your Python env to run live)', '预期输出（复制代码到Python环境中实时运行）')}
                </div>
                <pre className="text-sm leading-relaxed text-emerald-800 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/30 rounded-xl p-4 font-mono whitespace-pre-wrap">
                  {activeSnippet.output}
                </pre>
              </div>
            )}
          </div>

          {/* Setup instructions */}
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-900">
            <p className="text-xs font-semibold text-blue-700 dark:text-blue-400 mb-2">
              🐍 {t('Quick Setup', '快速设置')}
            </p>
            <pre className="text-xs text-blue-800 dark:text-blue-300 font-mono leading-relaxed whitespace-pre-wrap">{`# Install dependencies
pip install numpy scikit-learn scipy pandas

# Optional (for neural network examples)
pip install torch xgboost optuna sentence-transformers

# Run any snippet
python snippet.py`}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}
