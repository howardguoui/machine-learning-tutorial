import type { TopicContent } from '../types'

export const neuralNetworks: TopicContent = {
  id: 'neural-networks',
  title: { en: 'Neural Networks', zh: '神经网络' },
  contentType: 'code',
  content: {
    en: `Neural networks are **universal function approximators** — given enough neurons and layers, they can approximate any continuous function.

## Building a Neural Network from Scratch

\`\`\`python
import numpy as np

class NeuralNetwork:
    """
    Multi-layer perceptron with configurable architecture.
    Demonstrates forward pass and backprop from scratch.
    """
    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialization: prevents vanishing/exploding gradients
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z): return np.maximum(0, z)
    def relu_grad(self, z): return (z > 0).astype(float)

    def softmax(self, z):
        z_shifted = z - z.max(axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.pre_activations = []
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.pre_activations.append(Z)
            if i < len(self.weights) - 1:
                A = self.relu(Z)       # hidden layers: ReLU
            else:
                A = self.softmax(Z)    # output layer: Softmax
            self.activations.append(A)
        return A

    def backward(self, y_true):
        m = len(y_true)
        n_layers = len(self.weights)
        dW_list, db_list = [], []

        # Output layer gradient (cross-entropy + softmax combined)
        A_out = self.activations[-1]
        dZ = A_out.copy()
        dZ[range(m), y_true] -= 1
        dZ /= m

        for i in reversed(range(n_layers)):
            A_prev = self.activations[i]
            dW = A_prev.T @ dZ
            db = dZ.sum(axis=0, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:  # propagate to previous layer
                dA = dZ @ self.weights[i].T
                dZ = dA * self.relu_grad(self.pre_activations[i-1])

        # Parameter update
        for i in range(n_layers):
            self.weights[i] -= self.lr * dW_list[i]
            self.biases[i]  -= self.lr * db_list[i]

    def compute_loss(self, X, y_true):
        A = self.forward(X)
        log_probs = -np.log(A[range(len(y_true)), y_true] + 1e-10)
        return log_probs.mean()

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

# Train on XOR problem (not linearly separable)
np.random.seed(42)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# Augment dataset
X_train = np.tile(X_xor, (100, 1)) + np.random.randn(400, 2) * 0.1
y_train = np.tile(y_xor, 100)

net = NeuralNetwork(layer_sizes=[2, 8, 8, 2], lr=0.1)
for epoch in range(1000):
    net.forward(X_train)
    net.backward(y_train)
    if epoch % 200 == 0:
        loss = net.compute_loss(X_train, y_train)
        acc = net.accuracy(X_train, y_train)
        print(f"Epoch {epoch:4d}: loss={loss:.4f}, acc={acc:.3f}")
\`\`\`

## PyTorch Neural Network

\`\`\`python
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Production-style neural network with PyTorch
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),  # normalize layer outputs
                    nn.ReLU(),
                    nn.Dropout(dropout),          # regularization
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Create synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.LongTensor(y_tr)
    X_te_t = torch.FloatTensor(X_te)
    y_te_t = torch.LongTensor(y_te)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model, loss, optimizer
    model = MLPClassifier(input_dim=20, hidden_dims=[64, 32], output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(X_te_t)
            test_acc = (test_logits.argmax(1) == y_te_t).float().mean()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: loss={total_loss/len(loader):.4f}, test_acc={test_acc:.4f}")

except ImportError:
    print("PyTorch not installed. Run: pip install torch")
\`\`\`

> **Key concepts**: ReLU activation, batch normalization, dropout, and AdamW optimizer are the default building blocks for modern MLPs. The PyTorch pattern (forward → loss → backward → step) is universal across all deep learning architectures.`,

    zh: `神经网络是**通用函数近似器**——给定足够的神经元和层，它们可以近似任何连续函数。

## 从零构建神经网络

反向传播通过链式法则计算梯度：
- **前向传播**：计算每层的激活值
- **反向传播**：从输出层到输入层逐层计算梯度
- **参数更新**：用梯度更新权重和偏置

## PyTorch神经网络

\`\`\`python
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 批归一化
                nn.ReLU(),
                nn.Dropout(dropout),          # 正则化
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
\`\`\`

> **关键概念**：ReLU激活、批归一化、Dropout和AdamW优化器是现代MLP的默认构建块。`,
  },
}

export const cnnRnn: TopicContent = {
  id: 'cnn-rnn',
  title: { en: 'CNNs & RNNs', zh: '卷积神经网络与循环神经网络' },
  contentType: 'code',
  content: {
    en: `CNNs excel at spatial data (images); RNNs/LSTMs handle sequential data (text, time series).

## Convolutional Neural Network (CNN)

\`\`\`python
try:
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        """
        CNN for image classification.
        Architecture: Conv → Pool → Conv → Pool → FC → Output
        """
        def __init__(self, n_classes=10):
            super().__init__()
            # Feature extraction: shared-weight convolutions detect local patterns
            self.features = nn.Sequential(
                # Block 1: detect edges
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1,28,28)→(32,28,28)
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                           # →(32,14,14)

                # Block 2: detect higher-level patterns
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # →(64,14,14)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                          # →(64,7,7)
            )
            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    # Parameter count
    cnn = SimpleCNN(n_classes=10)
    total_params = sum(p.numel() for p in cnn.parameters())
    trainable_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(8, 1, 28, 28)   # batch of 8 grayscale 28×28 images
    out = cnn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")  # (8, 10) — class logits

except ImportError:
    print("PyTorch not installed.")
\`\`\`

## LSTM for Sequence Modeling

\`\`\`python
try:
    import torch
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        """
        LSTM for sequence classification (e.g., sentiment analysis).
        LSTM gates control information flow:
        - Forget gate: what to discard from cell state
        - Input gate:  what new info to add
        - Output gate: what to expose to hidden state
        """
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_classes, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,    # both directions capture more context
                dropout=dropout if n_layers > 1 else 0,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            # x shape: (batch, seq_len) — token indices
            embedded = self.embedding(x)                   # (B, L, E)
            output, (h_n, c_n) = self.lstm(embedded)      # (B, L, 2H)
            # Use final hidden states from both directions
            h_concat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
            return self.classifier(h_concat)

    model = LSTMClassifier(
        vocab_size=10000, embed_dim=128, hidden_dim=256,
        n_layers=2, n_classes=2
    )

    # Test forward pass
    x = torch.randint(1, 10000, (16, 50))   # batch of 16, seq_len=50
    out = model(x)
    print(f"LSTM output shape: {out.shape}")  # (16, 2)

except ImportError:
    print("PyTorch not installed.")
\`\`\`

## When to Use CNN vs RNN vs Transformer

| Architecture | Data Type | Key Strength | Example Task |
|---|---|---|---|
| **CNN** | Images, local patterns | Translation invariance | Image classification |
| **RNN/LSTM** | Short sequences | Memory over time | Sentiment, time series |
| **Transformer** | Long sequences | Global attention | Language models, BERT |
| **CNN + RNN** | Video, audio | Spatial + temporal | Speech recognition |`,

    zh: `CNN擅长空间数据（图像）；RNN/LSTM处理序列数据（文本、时间序列）。

## 卷积神经网络

CNN的核心思想：
- **权重共享**：同一卷积核在整个图像上滑动，减少参数
- **局部感受野**：每个神经元只关注局部区域
- **层次特征**：浅层检测边缘，深层检测高级特征

| 架构 | 数据类型 | 关键优势 | 示例任务 |
|---|---|---|---|
| **CNN** | 图像、局部模式 | 平移不变性 | 图像分类 |
| **RNN/LSTM** | 短序列 | 时间记忆 | 情感分析、时间序列 |
| **Transformer** | 长序列 | 全局注意力 | 语言模型、BERT |`,
  },
}
