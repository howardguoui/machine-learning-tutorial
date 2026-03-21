import type { TopicContent } from '../types'

export const linearAlgebra: TopicContent = {
  id: 'linear-algebra',
  title: { en: 'Linear Algebra for ML', zh: '机器学习中的线性代数' },
  contentType: 'code',
  content: {
    en: `Linear algebra is the **language of machine learning**. Every neural network forward pass, every matrix decomposition, every embedding lookup is linear algebra in action.

## Vectors and Matrices

\`\`\`python
import numpy as np

# A data point is a vector
x = np.array([1.5, 2.3, 0.8])   # shape (3,) — 3 features
print(f"Shape: {x.shape}, Norm: {np.linalg.norm(x):.3f}")

# A dataset is a matrix — rows = samples, cols = features
X = np.array([
    [1.5, 2.3, 0.8],
    [0.2, 1.1, 3.4],
    [2.1, 0.5, 1.2],
])  # shape (3 samples, 3 features)
print(f"Data matrix shape: {X.shape}")

# Weight matrix in a neural network layer
W = np.random.randn(3, 4)  # maps from 3 features → 4 hidden units
b = np.zeros(4)             # bias vector

# Linear transformation: Z = XW + b
Z = X @ W + b   # @ is matrix multiplication
print(f"Output shape: {Z.shape}")  # (3, 4)
\`\`\`

## Dot Product and Similarity

\`\`\`python
# Dot product: measures alignment between vectors
a = np.array([1, 0, 0])  # unit vector along x
b = np.array([0.7, 0.7, 0])  # 45 degrees from x

dot = np.dot(a, b)  # = |a||b|cos(θ)
cos_sim = dot / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.3f}")  # ≈ 0.707 (45°)

# In transformers, attention is a scaled dot product:
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V

Q = np.random.randn(5, 8)  # 5 queries, dim=8
K = np.random.randn(5, 8)  # 5 keys
V = np.random.randn(5, 8)  # 5 values
attn_out = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {attn_out.shape}")
\`\`\`

## Eigenvalues and PCA

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data  # 150 samples, 4 features

# Principal Component Analysis via eigendecomposition
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")        # (150, 4)
print(f"Reduced shape: {X_reduced.shape}") # (150, 2)
print(f"Variance explained: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

# Under the hood: PCA finds eigenvectors of the covariance matrix
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
print(f"Top 2 eigenvalues: {eigenvalues[idx[:2]]}")
\`\`\`

## Matrix Factorization

\`\`\`python
# SVD: the Swiss army knife of linear algebra in ML
# Used in: PCA, LSA, collaborative filtering, compression

A = np.array([
    [4, 0, 0, 0, 1, 1],
    [0, 0, 3, 1, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
])  # User-item ratings matrix

U, S, Vt = np.linalg.svd(A, full_matrices=False)
# U: user embeddings, S: singular values, Vt: item embeddings

# Low-rank approximation (keep top-k singular values)
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"Reconstruction error: {np.linalg.norm(A - A_approx):.3f}")
\`\`\`

> **Key intuition**: A weight matrix in a neural network is a linear map between vector spaces. Training adjusts these maps so that semantically similar inputs map to similar output regions.`,

    zh: `线性代数是**机器学习的语言**。每次神经网络前向传播、每次矩阵分解、每次嵌入查找都是线性代数的实践。

## 向量和矩阵

\`\`\`python
import numpy as np

# 数据点是一个向量
x = np.array([1.5, 2.3, 0.8])   # 形状 (3,) — 3个特征

# 数据集是一个矩阵——行=样本，列=特征
X = np.array([
    [1.5, 2.3, 0.8],
    [0.2, 1.1, 3.4],
    [2.1, 0.5, 1.2],
])  # 形状 (3样本, 3特征)

# 神经网络层中的权重矩阵
W = np.random.randn(3, 4)  # 从3个特征映射到4个隐藏单元
b = np.zeros(4)             # 偏置向量

# 线性变换：Z = XW + b
Z = X @ W + b   # @ 是矩阵乘法
print(f"输出形状: {Z.shape}")  # (3, 4)
\`\`\`

## 点积与相似度

\`\`\`python
# 注意力机制中的缩放点积注意力
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V
\`\`\`

## 特征值与PCA

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # 150个样本，4个特征

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"原始形状: {X.shape}")        # (150, 4)
print(f"降维后形状: {X_reduced.shape}") # (150, 2)
print(f"解释方差: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")
\`\`\`

> **关键直觉**：神经网络中的权重矩阵是向量空间之间的线性映射。训练调整这些映射，使语义相似的输入映射到相似的输出区域。`,
  },
}

export const calcGradients: TopicContent = {
  id: 'calculus-gradients',
  title: { en: 'Calculus & Gradients', zh: '微积分与梯度' },
  contentType: 'code',
  content: {
    en: `Gradients are the **compass of optimization** — they point in the direction of steepest ascent. To minimize a loss, we follow the **negative gradient**.

## The Derivative: Slope at a Point

\`\`\`python
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Numerical gradient via finite differences
def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient of scalar function f at point x."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Example: loss = (w - 3)^2 + (b + 1)^2  (minimum at w=3, b=-1)
def loss(params):
    w, b = params
    return (w - 3)**2 + (b + 1)**2

params = np.array([0.0, 0.0])
grad = numerical_gradient(loss, params)
print(f"Gradient at (0, 0): {grad}")  # [-6.  2.]
# Gradient points away from minimum → step in -gradient direction
\`\`\`

## Gradient Descent

\`\`\`python
def gradient_descent(loss_fn, grad_fn, init_params, lr=0.1, n_steps=100):
    """
    Minimize loss_fn starting from init_params.

    Args:
        loss_fn: scalar loss function
        grad_fn: gradient function (or use numerical_gradient)
        init_params: starting point
        lr: learning rate (step size)
        n_steps: number of update steps
    """
    params = init_params.copy().astype(float)
    history = []

    for step in range(n_steps):
        loss_val = loss_fn(params)
        grad = grad_fn(params)

        # Core update: move opposite to gradient
        params -= lr * grad
        history.append({'step': step, 'loss': loss_val, 'params': params.copy()})

        if step % 20 == 0:
            print(f"Step {step:3d}: loss={loss_val:.4f}, params={params}")

    return params, history

# Analytical gradient for our quadratic loss
def loss_grad(params):
    w, b = params
    return np.array([2*(w - 3), 2*(b + 1)])

final_params, hist = gradient_descent(loss, loss_grad, np.array([0.0, 0.0]))
print(f"\\nFinal params: {final_params}")  # Should be ≈ [3, -1]
\`\`\`

## Backpropagation: Chain Rule in Action

\`\`\`python
class SimpleNeuron:
    """
    Single neuron: output = sigmoid(w*x + b)
    Demonstrates manual backprop via chain rule.
    """
    def __init__(self):
        self.w = 0.5
        self.b = 0.1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_grad(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)  # derivative of sigmoid

    def forward(self, x):
        self.x = x
        self.z = self.w * x + self.b     # pre-activation
        self.a = self.sigmoid(self.z)     # activation
        return self.a

    def backward(self, d_loss_d_a):
        """Chain rule: dL/dw = dL/da * da/dz * dz/dw"""
        d_a_d_z = self.sigmoid_grad(self.z)       # da/dz
        d_z_d_w = self.x                           # dz/dw = x
        d_z_d_b = 1.0                              # dz/db = 1

        d_loss_d_w = d_loss_d_a * d_a_d_z * d_z_d_w
        d_loss_d_b = d_loss_d_a * d_a_d_z * d_z_d_b
        return d_loss_d_w, d_loss_d_b

    def update(self, d_loss_d_w, d_loss_d_b, lr=0.1):
        self.w -= lr * d_loss_d_w
        self.b -= lr * d_loss_d_b

# Train: teach neuron to output 1 when input is 2.0
neuron = SimpleNeuron()
target = 1.0
x = 2.0

for step in range(50):
    prediction = neuron.forward(x)
    loss_val = (prediction - target) ** 2       # MSE loss
    d_loss_d_a = 2 * (prediction - target)      # dL/da

    dw, db = neuron.backward(d_loss_d_a)
    neuron.update(dw, db)

    if step % 10 == 0:
        print(f"Step {step}: pred={prediction:.4f}, loss={loss_val:.4f}")
\`\`\`

## Learning Rate: The Most Critical Hyperparameter

\`\`\`python
# Visualize effect of different learning rates
def try_lr(lr, n_steps=30):
    params = np.array([0.0, 0.0])
    losses = []
    for _ in range(n_steps):
        losses.append(loss(params))
        grad = loss_grad(params)
        params -= lr * grad
    return losses

for lr in [0.01, 0.1, 0.5, 1.1]:
    losses = try_lr(lr)
    final_loss = losses[-1]
    status = "converged" if final_loss < 0.01 else ("diverged" if final_loss > 1000 else "slow")
    print(f"lr={lr}: final_loss={final_loss:.4f} ({status})")

# lr=0.01: final_loss≈0.0001 (slow but stable)
# lr=0.1:  final_loss≈0.0000 (good)
# lr=0.5:  final_loss≈0.0000 (fast)
# lr=1.1:  final_loss=huge   (diverges - overshoots minimum)
\`\`\`

> **Intuition**: The gradient tells you which direction is uphill. The learning rate tells you how big a step to take. Too small → very slow convergence. Too large → oscillation or divergence.`,

    zh: `梯度是**优化的指南针**——它们指向最陡峭上升的方向。为了最小化损失，我们沿**负梯度**方向移动。

## 导数：某点处的斜率

\`\`\`python
import numpy as np

# 通过有限差分进行数值梯度计算
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# 损失函数示例：最小值在 w=3, b=-1
def loss(params):
    w, b = params
    return (w - 3)**2 + (b + 1)**2

params = np.array([0.0, 0.0])
grad = numerical_gradient(loss, params)
print(f"(0,0)处的梯度: {grad}")  # [-6.  2.]
\`\`\`

## 梯度下降

\`\`\`python
def gradient_descent(loss_fn, grad_fn, init_params, lr=0.1, n_steps=100):
    params = init_params.copy().astype(float)
    for step in range(n_steps):
        loss_val = loss_fn(params)
        grad = grad_fn(params)
        params -= lr * grad  # 核心更新：向梯度相反方向移动
        if step % 20 == 0:
            print(f"步骤 {step:3d}: loss={loss_val:.4f}, params={params}")
    return params
\`\`\`

## 反向传播：链式法则的实践

反向传播通过**链式法则**计算每个参数的梯度：

$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}$$

> **直觉**：梯度告诉你哪个方向是上坡。学习率告诉你步幅多大。太小→收敛非常慢。太大→振荡或发散。`,
  },
}

export const optimization: TopicContent = {
  id: 'optimization',
  title: { en: 'Optimization Algorithms', zh: '优化算法' },
  contentType: 'code',
  content: {
    en: `Modern ML optimizers go far beyond vanilla gradient descent. Understanding them helps you choose the right one and tune it effectively.

## SGD with Momentum

\`\`\`python
import numpy as np

class SGDMomentum:
    """
    Stochastic Gradient Descent with momentum.

    Momentum accumulates a velocity vector in directions of persistent gradient,
    dampening oscillations and accelerating convergence.

    v_t = β * v_{t-1} + (1-β) * g_t     # velocity
    θ_t = θ_{t-1} - lr * v_t            # parameter update
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * grads
        params -= self.lr * self.velocity
        return params
\`\`\`

## Adam Optimizer

\`\`\`python
class Adam:
    """
    Adaptive Moment Estimation (Adam).

    Combines momentum (1st moment) with RMSProp (2nd moment).
    - Maintains per-parameter adaptive learning rates
    - Bias correction for initialization warm-up

    m_t = β1 * m_{t-1} + (1-β1) * g_t           # 1st moment (mean)
    v_t = β2 * v_{t-1} + (1-β2) * g_t^2         # 2nd moment (variance)
    m̂_t = m_t / (1-β1^t)                         # bias correction
    v̂_t = v_t / (1-β2^t)
    θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        # Bias-corrected estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params

# Benchmark optimizers on the Rosenbrock function (tricky non-convex landscape)
def rosenbrock(params):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2. Global minimum at (1,1)."""
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(params):
    x, y = params
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

print("Optimizer comparison on Rosenbrock function:")
print(f"{'Optimizer':<15} {'Steps to <0.01':<20} {'Final loss'}")
print("-" * 50)

for OptimizerClass, kwargs, name in [
    (SGDMomentum, {'lr': 0.001, 'momentum': 0.9}, 'SGD+Momentum'),
    (Adam, {'lr': 0.01}, 'Adam'),
]:
    opt = OptimizerClass(**kwargs)
    params = np.array([-1.0, 1.0])  # start far from minimum
    steps_to_converge = None

    for step in range(5000):
        grads = rosenbrock_grad(params)
        params = opt.step(params, grads)
        loss_val = rosenbrock(params)
        if loss_val < 0.01 and steps_to_converge is None:
            steps_to_converge = step

    print(f"{name:<15} {str(steps_to_converge) + ' steps':<20} {rosenbrock(params):.6f}")
\`\`\`

## Learning Rate Scheduling

\`\`\`python
class WarmupCosineSchedule:
    """
    Linear warmup followed by cosine decay.
    Used in Transformers and large model training.
    """
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))

schedule = WarmupCosineSchedule(base_lr=1e-3, warmup_steps=100, total_steps=1000)
for step in [0, 50, 100, 300, 600, 1000]:
    print(f"Step {step:4d}: lr = {schedule.get_lr(step):.6f}")
\`\`\`

## Optimizer Choice Guide

| Situation | Recommended Optimizer |
|---|---|
| Computer vision (CNNs) | SGD + Momentum + LR schedule |
| NLP / Transformers | AdamW (Adam + weight decay) |
| Quick prototyping | Adam with default settings |
| Fine-tuning LLMs | AdamW with cosine warmup |
| Very large batches | LAMB or LARS |

> **Rule of thumb**: Adam is a safe default. For production training of large models, switch to AdamW (adds L2 regularization correctly) with a warmup cosine schedule.`,

    zh: `现代ML优化器远超简单梯度下降。理解它们有助于选择合适的并进行有效调优。

## 带动量的SGD

\`\`\`python
class SGDMomentum:
    """
    带动量的随机梯度下降。
    动量在持久梯度方向累积速度向量，抑制振荡并加速收敛。
    v_t = β * v_{t-1} + (1-β) * g_t
    θ_t = θ_{t-1} - lr * v_t
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * grads
        params -= self.lr * self.velocity
        return params
\`\`\`

## Adam优化器

Adam结合了动量（一阶矩）和RMSProp（二阶矩）：
- 维护每个参数的自适应学习率
- 偏差校正用于初始化预热

| 情况 | 推荐优化器 |
|---|---|
| 计算机视觉(CNN) | SGD + 动量 + LR调度 |
| NLP / Transformer | AdamW（Adam + 权重衰减） |
| 快速原型 | 使用默认设置的Adam |
| 微调LLM | AdamW + 余弦预热 |

> **经验法则**：Adam是安全的默认选择。对于大模型的生产训练，切换到AdamW（正确添加L2正则化）加余弦预热调度。`,
  },
}

export const tensors: TopicContent = {
  id: 'tensors',
  title: { en: 'Tensors & NumPy/PyTorch', zh: '张量与NumPy/PyTorch' },
  contentType: 'code',
  content: {
    en: `Tensors are **N-dimensional arrays** — the fundamental data structure of ML. Understanding tensor operations is essential for working with any modern ML framework.

## Tensor Fundamentals with NumPy

\`\`\`python
import numpy as np

# Scalar (0D tensor)
scalar = np.float32(3.14)
print(f"Scalar: {scalar}, shape: {scalar.shape}, ndim: {scalar.ndim}")

# Vector (1D tensor)
vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print(f"Vector shape: {vector.shape}")  # (3,)

# Matrix (2D tensor)
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(f"Matrix shape: {matrix.shape}")  # (2, 3)

# 3D tensor — e.g., batch of images (batch, height, width)
images = np.zeros((32, 28, 28), dtype=np.float32)
print(f"Images shape: {images.shape}")  # (32, 28, 28)

# 4D tensor — RGB image batch (batch, channels, height, width)
rgb_batch = np.zeros((32, 3, 224, 224), dtype=np.float32)
print(f"RGB batch shape: {rgb_batch.shape}")  # (32, 3, 224, 224)
print(f"Memory: {rgb_batch.nbytes / 1e6:.1f} MB")  # ~19 MB
\`\`\`

## Broadcasting: The Secret Weapon

\`\`\`python
# Broadcasting lets you operate on tensors of different shapes
# without explicit loops

# Normalize each of 1000 samples (each with 10 features)
X = np.random.randn(1000, 10).astype(np.float32)

# Per-feature mean and std (shape: (10,))
mean = X.mean(axis=0)  # mean over samples
std = X.std(axis=0)

# Broadcasting: (1000,10) - (10,) → NumPy broadcasts (10,) to (1000,10)
X_norm = (X - mean) / (std + 1e-8)
print(f"Normalized mean: {X_norm.mean():.6f}")    # ≈ 0
print(f"Normalized std:  {X_norm.std():.6f}")     # ≈ 1

# Batch matrix multiplication
batch_size = 16
Q = np.random.randn(batch_size, 8, 64)   # (B, seq_len, d_model)
K = np.random.randn(batch_size, 8, 64)   # (B, seq_len, d_model)

# Efficient batched attention scores: (B, seq, seq)
scores = Q @ K.transpose(0, 2, 1)  # transpose last two dims
print(f"Attention scores shape: {scores.shape}")  # (16, 8, 8)
\`\`\`

## PyTorch Tensors with Autograd

\`\`\`python
# PyTorch tensors support automatic differentiation
# Run this if you have PyTorch installed: pip install torch

try:
    import torch

    # Create tensor with gradient tracking
    w = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    x = torch.tensor([1.0, 2.0, 3.0])

    # Forward pass
    y = (w * x).sum()   # dot product: 2*1 + (-1)*2 + 0.5*3 = 1.5
    print(f"y = {y.item():.2f}")

    # Backward pass — computes dy/dw automatically
    y.backward()
    print(f"dy/dw = {w.grad}")  # [1., 2., 3.] = x

    # Manual SGD update
    with torch.no_grad():
        w -= 0.01 * w.grad
    w.grad.zero_()

    # Neural network layer from scratch
    def linear_layer(x, W, b):
        return x @ W + b

    batch = torch.randn(4, 3)      # 4 samples, 3 features
    W = torch.randn(3, 5, requires_grad=True)  # weight matrix
    b = torch.zeros(5, requires_grad=True)      # bias

    out = linear_layer(batch, W, b)
    loss = out.pow(2).mean()       # mean squared output (dummy loss)
    loss.backward()
    print(f"W.grad shape: {W.grad.shape}")  # (3, 5)
    print(f"b.grad shape: {b.grad.shape}")  # (5,)

except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
\`\`\`

## Efficient Tensor Operations

\`\`\`python
import numpy as np

# Vectorized vs loop comparison
n = 100_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

import time

# Python loop (slow)
start = time.perf_counter()
result_loop = sum(a[i] * b[i] for i in range(n))
loop_time = time.perf_counter() - start

# NumPy vectorized (fast)
start = time.perf_counter()
result_np = np.dot(a, b)
np_time = time.perf_counter() - start

print(f"Loop time:   {loop_time*1000:.1f} ms")
print(f"NumPy time:  {np_time*1000:.3f} ms")
print(f"Speedup:     {loop_time/np_time:.0f}x")
# Typical: NumPy is 100-1000x faster due to BLAS-optimized C routines
\`\`\`

> **Core principle**: Always prefer vectorized tensor operations over Python loops. ML frameworks are optimized to parallelize tensor ops on GPU/CPU SIMD units.`,

    zh: `张量是**N维数组**——ML的基本数据结构。理解张量操作对于使用任何现代ML框架都至关重要。

## NumPy张量基础

\`\`\`python
import numpy as np

# 标量（0D张量）
scalar = np.float32(3.14)

# 向量（1D张量）
vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# 矩阵（2D张量）
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 4D张量——RGB图像批次 (批次, 通道, 高度, 宽度)
rgb_batch = np.zeros((32, 3, 224, 224), dtype=np.float32)
print(f"RGB批次形状: {rgb_batch.shape}")  # (32, 3, 224, 224)
\`\`\`

## 广播：秘密武器

\`\`\`python
X = np.random.randn(1000, 10).astype(np.float32)
mean = X.mean(axis=0)  # 形状 (10,)
std = X.std(axis=0)

# 广播：(1000,10) - (10,) → NumPy将(10,)广播为(1000,10)
X_norm = (X - mean) / (std + 1e-8)
print(f"归一化均值: {X_norm.mean():.6f}")    # ≈ 0
\`\`\`

> **核心原则**：始终优先使用向量化张量操作而非Python循环。ML框架经过优化，可在GPU/CPU SIMD单元上并行处理张量操作。`,
  },
}

export const probability: TopicContent = {
  id: 'probability-stats',
  title: { en: 'Probability & Statistics', zh: '概率与统计' },
  contentType: 'code',
  content: {
    en: `Probability theory provides the **mathematical foundation** for uncertainty, which is central to all ML models.

## Probability Distributions

\`\`\`python
import numpy as np
from scipy import stats

# Normal (Gaussian) distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)

pdf = stats.norm.pdf(x, mu, sigma)      # probability density
cdf = stats.norm.cdf(x, mu, sigma)      # cumulative probability

# Key properties
samples = np.random.normal(mu, sigma, 10000)
print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")

# 68-95-99.7 rule
within_1_std = np.mean(np.abs(samples - mu) < sigma)
within_2_std = np.mean(np.abs(samples - mu) < 2*sigma)
print(f"Within 1σ: {within_1_std:.1%}")  # ≈ 68%
print(f"Within 2σ: {within_2_std:.1%}")  # ≈ 95%
\`\`\`

## Maximum Likelihood Estimation (MLE)

\`\`\`python
# MLE: find parameters that maximize P(data | params)
# Equivalent to minimizing negative log-likelihood

# Example: fit a Gaussian to data
data = np.array([2.1, 1.8, 2.3, 2.5, 1.9, 2.2, 2.0])

# Analytical MLE for Gaussian
mu_mle = data.mean()
sigma_mle = data.std()   # biased (divide by n), MLE estimate
print(f"MLE: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# Numerical MLE via minimizing NLL
from scipy.optimize import minimize

def neg_log_likelihood(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # ensure positive
    # sum of log N(x_i | mu, sigma)
    nll = -stats.norm.logpdf(data, mu, sigma).sum()
    return nll

result = minimize(neg_log_likelihood, x0=[0, 0])
mu_opt, sigma_opt = result.x[0], np.exp(result.x[1])
print(f"Numerical MLE: μ={mu_opt:.3f}, σ={sigma_opt:.3f}")
\`\`\`

## Cross-Entropy Loss

\`\`\`python
# Cross-entropy: the standard classification loss
# H(y, ŷ) = -Σ y_i * log(ŷ_i)

def cross_entropy(y_true, y_pred, eps=1e-10):
    """
    y_true: one-hot labels (n_samples, n_classes)
    y_pred: predicted probabilities from softmax
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # prevent log(0)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()

# Binary cross-entropy (binary classification)
def binary_cross_entropy(y_true, y_pred, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)).mean()

# Perfect predictions
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_perfect = np.array([[0.01, 0.98, 0.01], [0.98, 0.01, 0.01]])
y_random = np.array([[0.33, 0.34, 0.33], [0.33, 0.34, 0.33]])
y_wrong = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])

print(f"Perfect predictions: CE = {cross_entropy(y_true, y_perfect):.4f}")
print(f"Random predictions:  CE = {cross_entropy(y_true, y_random):.4f}")
print(f"Wrong predictions:   CE = {cross_entropy(y_true, y_wrong):.4f}")
\`\`\`

## Bayes' Theorem in ML

\`\`\`python
# Bayes: P(θ|data) ∝ P(data|θ) * P(θ)
# posterior ∝ likelihood * prior

# Naive Bayes classifier — applies Bayes theorem per feature
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()
clf = GaussianNB()
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(f"Naive Bayes CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
\`\`\`

> **Connection to loss functions**: Cross-entropy loss = negative log-likelihood under a categorical distribution. MSE loss = negative log-likelihood under a Gaussian distribution. **Every loss function is a probabilistic model in disguise.**`,

    zh: `概率论为不确定性提供了**数学基础**，这是所有ML模型的核心。

## 概率分布

\`\`\`python
import numpy as np
from scipy import stats

# 正态（高斯）分布
samples = np.random.normal(0, 1, 10000)
within_1_std = np.mean(np.abs(samples) < 1)
print(f"1σ内: {within_1_std:.1%}")  # ≈ 68%
\`\`\`

## 最大似然估计（MLE）

MLE：找到使P(数据|参数)最大化的参数，等价于最小化负对数似然。

## 交叉熵损失

\`\`\`python
def cross_entropy(y_true, y_pred, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()
\`\`\`

> **与损失函数的联系**：交叉熵损失=分类分布下的负对数似然。MSE损失=高斯分布下的负对数似然。**每个损失函数都是伪装的概率模型。**`,
  },
}
