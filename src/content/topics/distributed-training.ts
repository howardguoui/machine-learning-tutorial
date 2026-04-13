import type { TopicContent } from '../types'

export const dataVsModelParallelism: TopicContent = {
  id: 'dist-parallelism-overview',
  title: { en: 'Data vs Model Parallelism', zh: '数据并行 vs 模型并行' },
  contentType: 'article',
  content: {
    en: `Distributed training scales large models across multiple GPUs/TPUs. Three primary strategies exist, each suited to different problems.

## Data Parallelism (DP)
- **Concept**: Replicate model across devices, split batch across GPUs
- **Use case**: Model fits on one GPU; batch/data is bottleneck
- **Pros**: Simple, scales linearly until network saturation
- **Cons**: Communication overhead from gradient aggregation

Example: 70B model (140GB at fp16) replicated 8x requires 8 GPUs × 40GB = 320GB VRAM total, but model replicas = memory waste.

## Model Parallelism (MP)
- **Concept**: Split model layers across GPUs. Each GPU processes full batch through its portion.
- **Use case**: Model too large for one GPU (e.g., 70B Llama, GPT-3 175B)
- **Pros**: Enables giant models
- **Cons**: Pipeline bubbles, reduced GPU utilization (devices idle waiting for previous stage)

**Memory math for 70B at fp16**:
- Model params: 70B × 2 bytes = 140GB
- Activations during forward pass: ~140GB (stored for backprop)
- Optimizer state (Adam): 2 copies = 280GB
- **Total for full training: ~560GB per GPU**
- Naive DP: replicate × 8 = impossible
- **MP solution**: Shard weights across 4 GPUs = 140GB forward activations + 280GB optimizer = 420GB per GPU (still tight)

## Pipeline Parallelism (PP)
- **Concept**: Synchronize model stages in pipelines; use micro-batches to hide latency
- **Key insight**: While GPU-1 processes micro-batch-1 forward, GPU-2 processes micro-batch-0 backward
- **Schedule**: GPipe (all forward, then all backward) vs 1F1B (1 Forward, 1 Backward overlap)
- **1F1B achieves**: ~80% GPU utilization vs ~40% with GPipe

## Tensor Parallelism (TP)
- **Concept**: Split weight matrices row-wise or column-wise so each GPU computes part of matrix multiplication
- **Example**: QKV projection for attention: each GPU owns subset of output dims
- **Memory**: ~linear reduction (4-way TP ≈ 4x less activations per GPU)
- **Comm cost**: All-reduce after every linear layer

## 3D Parallelism (DP + TP + PP)
Modern mega-models combine all three:
- **Data Parallel**: 8 data-parallel groups
- **Tensor Parallel**: 4-way TP per group (shards weight matrices)
- **Pipeline Parallel**: 2 pipeline stages (encode/decode split)
- **Example Megatron-LM**: Train 175B GPT-3 on 1024 A100s using (DP=32, TP=4, PP=8)

## Decision Tree
1. **Model fits on GPU?** → Use Data Parallelism (simple, effective)
2. **Model 10x GPU VRAM?** → Add Tensor Parallelism
3. **Model 100x GPU VRAM?** → Add Pipeline Parallelism + coordinate all three

**Remember**: More parallelism = more communication. Total compute = computation + synchronization cost. Choose minimum parallelism needed.`,
    zh: `分布式训练跨多个GPU/TPU扩展大模型。三种主要策略各适应不同问题。

## 数据并行 (DP)
- **概念**: 在各设备复制模型，将批次分割到GPU
- **用途**: 模型适配单GPU；批数据是瓶颈
- **优点**: 简单，线性扩展（直到网络饱和）
- **缺点**: 梯度聚合通信开销大

例：70B模型（fp16下140GB）复制8倍需8个GPU × 40GB = 320GB VRAM，但模型副本浪费内存。

## 模型并行 (MP)
- **概念**: 将模型层分割到GPU。每GPU处理完整批通过其部分。
- **用途**: 模型太大无法在单GPU上运行（如70B Llama、GPT-3 175B）
- **优点**: 启用超大模型
- **缺点**: 流水线气泡，GPU利用率低（设备闲置等待前一阶段）

**70B fp16内存数学**：
- 模型参数：70B × 2字节 = 140GB
- 前向传播激活：~140GB（为反向保存）
- 优化器状态（Adam）：2份副本 = 280GB
- **训练总计：~560GB每GPU**
- 朴素DP：复制×8 = 不可行
- **MP方案**: 权重分片4个GPU = 140GB前向激活 + 280GB优化器 = 420GB每GPU（仍紧张）

## 流水线并行 (PP)
- **概念**: 同步管道模型阶段；用微批隐藏延迟
- **关键**: GPU-1处理微批-1前向时，GPU-2处理微批-0反向
- **调度**: GPipe（全前向后全反向）vs 1F1B（1前1反重叠）
- **1F1B达成**: ~80% GPU利用率 vs GPipe的~40%

## 张量并行 (TP)
- **概念**: 按行或列分割权重矩阵，各GPU计算矩阵乘法的部分
- **例**: 注意力QKV投影：各GPU拥有输出维度子集
- **内存**: ~线性降低（4路TP ≈ 4倍少激活每GPU）
- **通信成本**: 每线性层后的全约化

## 3D并行 (DP + TP + PP)
现代超大模型结合三种：
- **数据并行**: 8个数据并行组
- **张量并行**: 4路TP每组（分片权重矩阵）
- **流水线并行**: 2个流水线阶段（编码器/解码器分割）
- **例Megatron-LM**: 在1024个A100上训练175B GPT-3用(DP=32, TP=4, PP=8)

## 决策树
1. **模型适配GPU?** → 用数据并行（简单有效）
2. **模型10倍GPU VRAM?** → 加张量并行
3. **模型100倍GPU VRAM?** → 加流水线并行+协调三种

**记住**: 更多并行 = 更多通信。总计算 = 计算 + 同步成本。选择最少必需并行。`,
  },
}

export const ddpHandsOn: TopicContent = {
  id: 'dist-ddp-hands-on',
  title: { en: 'DDP: Hands-On Setup', zh: 'DDP：实战配置' },
  contentType: 'code',
  content: {
    en: `PyTorch Distributed Data Parallel (DDP) is the most practical starting point for multi-GPU training. Here's the complete workflow.

## Core Concepts

**Process Group**: Each GPU/process joins a process group. Ranks communicate via NCCL backend (NVIDIA Collective Communications Library).

**Gradient Synchronization**: After \`.backward()\`, DDP automatically averages gradients across all ranks before optimizer step.

**DistributedSampler**: Ensures each rank sees different data (rank-0: samples 0-999, rank-1: samples 1000-1999, etc.).

## Complete DDP Training Script

\`\`\`python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def setup_ddp():
    """Initialize DDP process group. Called once per process."""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(
        backend='nccl',
        init_method=f'env://',
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    print(f'Rank {rank}/{world_size} initialized on GPU {rank}')

def cleanup_ddp():
    """Clean up process group."""
    dist.destroy_process_group()

def main_worker(gpu, args):
    rank = args.nr * args.gpus_per_node + gpu
    setup_ddp()

    # Build model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model = model.cuda(gpu)

    # Wrap with DDP (gradient sync happens in backward)
    model = DDP(model, device_ids=[gpu])

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy dataset (MNIST-like)
    X_train = torch.randn(10000, 784)
    y_train = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X_train, y_train)

    # DistributedSampler: each rank gets unique subset
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
    )

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Important: reshuffle each epoch
        total_loss = 0

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.cuda(gpu), y.cuda(gpu)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()  # DDP synchronizes gradients here
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}')

        # Barrier: all ranks wait before next epoch
        dist.barrier()

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch} finished. Avg loss: {avg_loss:.4f}')

    cleanup_ddp()

if __name__ == '__main__':
    class Args:
        nr = 0  # node rank
        gpus_per_node = 2  # change if using 1, 4, 8 GPUs
        world_size = 2  # usually = num_nodes * gpus_per_node
        epochs = 3

    args = Args()
    main_worker(int(os.environ.get('LOCAL_RANK', 0)), args)
\`\`\`

## Launch with torchrun

\`\`\`bash
# Single machine, 2 GPUs
torchrun --nproc_per_node=2 train.py

# Multi-machine (node-0)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 train.py

# Multi-machine (node-1)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 train.py
\`\`\`

## Common Pitfalls & Debugging

**1. Unused Parameters Warning**
- Cause: Model has parameters not used in forward pass
- Fix: Set \`find_unused_parameters=True\` (slower) or redesign model

\`\`\`python
model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
\`\`\`

**2. Uneven Input Sizes (causes deadlock)**
- Cause: Rank-0 has 64 samples, rank-1 has 63 samples; barrier waits forever
- Fix: Use \`DistributedSampler\` (handles padding automatically)

**3. NCCL Timeouts**
- Symptom: Hangs after X batches, then times out
- Cause: Rank N crashed or too slow; others waiting at barrier
- Debug: Set \`NCCL_DEBUG=INFO\` for detailed collective comm logs
- Fix: Check for OOM on specific ranks, ensure all ranks have same GPU memory

**4. Rank Mismatch**
- Symptom: \`RuntimeError: expected scalarType Float but got Long\`
- Cause: Different dtypes on different ranks
- Fix: Ensure \`model.cuda(gpu)\` and data \`.cuda(gpu)\` match ranks

## Debugging Checklist

\`\`\`bash
# 1. Print rank/world_size on each process
export NCCL_DEBUG=INFO  # detailed NCCL logs

# 2. Log per-rank
if rank == 0:
    print(f"Loss: {loss.item()}")

# 3. Verify DistributedSampler is used (no duplicate data)
print(f"Rank {rank}: {len(loader)} batches")

# 4. Add barriers with logging
dist.barrier()
if rank == 0:
    print("All ranks synchronized")

# 5. Check GPU memory per rank
import nvidia_ml_py
handle = nvidia_ml_py.nvmlDeviceGetHandleByIndex(rank)
info = nvidia_ml_py.nvmlDeviceGetMemoryInfo(handle)
print(f"Rank {rank} GPU memory: {info.used / 1e9:.2f}GB")
\`\`\``,
    zh: `PyTorch分布式数据并行(DDP)是多GPU训练的最实用起点。以下是完整工作流程。

## 核心概念

**进程组**: 每个GPU/进程加入进程组。秩通过NCCL后端(NVIDIA集合通信库)通信。

**梯度同步**: 在\`.backward()\`后，DDP自动在所有秩间平均梯度后再优化器步。

**DistributedSampler**: 确保各秩看不同数据(秩-0: 样本0-999，秩-1: 样本1000-1999等)。

## 完整DDP训练脚本

\`\`\`python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def setup_ddp():
    """初始化DDP进程组。每进程调用一次。"""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(
        backend='nccl',
        init_method=f'env://',
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    print(f'秩 {rank}/{world_size} 初始化在GPU {rank}')

def cleanup_ddp():
    """清理进程组。"""
    dist.destroy_process_group()

def main_worker(gpu, args):
    rank = args.nr * args.gpus_per_node + gpu
    setup_ddp()

    # 构建模型
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model = model.cuda(gpu)

    # 用DDP包装(梯度同步在backward中发生)
    model = DDP(model, device_ids=[gpu])

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 虚拟数据集(MNIST类)
    X_train = torch.randn(10000, 784)
    y_train = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X_train, y_train)

    # DistributedSampler: 各秩得独特子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
    )

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 重要: 每轮重排
        total_loss = 0

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.cuda(gpu), y.cuda(gpu)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()  # DDP在此同步梯度
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0 and rank == 0:
                print(f'轮次 {epoch}, 批 {batch_idx}, 损失 {loss.item():.4f}')

        # 屏障: 所有秩等待后进入下轮
        dist.barrier()

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f'轮次 {epoch} 完成. 平均损失: {avg_loss:.4f}')

    cleanup_ddp()

if __name__ == '__main__':
    class Args:
        nr = 0  # 节点秩
        gpus_per_node = 2  # 改为1, 4, 8如需
        world_size = 2  # 通常 = num_nodes * gpus_per_node
        epochs = 3

    args = Args()
    main_worker(int(os.environ.get('LOCAL_RANK', 0)), args)
\`\`\`

## 用torchrun启动

\`\`\`bash
# 单机, 2个GPU
torchrun --nproc_per_node=2 train.py

# 多机(节点-0)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 train.py

# 多机(节点-1)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 train.py
\`\`\`

## 常见陷阱和调试

**1. 未使用参数警告**
- 原因: 模型有参数未在前向使用
- 修复: 设\`find_unused_parameters=True\`(更慢)或重设模型

\`\`\`python
model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
\`\`\`

**2. 不均匀输入大小(导致死锁)
- 原因: 秩-0有64样本，秩-1有63样本；屏障永远等待
- 修复: 用\`DistributedSampler\`(自动处理填充)

**3. NCCL超时**
- 症状: X批后挂起，然后超时
- 原因: 秩N崩溃或太慢；其他在屏障等待
- 调试: 设\`NCCL_DEBUG=INFO\`查看详细集合通信日志
- 修复: 检查特定秩OOM，确保各秩GPU内存相同

**4. 秩不匹配**
- 症状: \`RuntimeError: expected scalarType Float but got Long\`
- 原因: 不同秩不同dtypes
- 修复: 确保\`model.cuda(gpu)\`和数据\`.cuda(gpu)\`与秩匹配

## 调试检查清单

\`\`\`bash
# 1. 在各进程打印秩/world_size
export NCCL_DEBUG=INFO  # 详细NCCL日志

# 2. 按秩日志
if rank == 0:
    print(f"损失: {loss.item()}")

# 3. 验证DistributedSampler被用(无重复数据)
print(f"秩 {rank}: {len(loader)} 批")

# 4. 添加日志屏障
dist.barrier()
if rank == 0:
    print("所有秩同步")

# 5. 检查各秩GPU内存
import nvidia_ml_py
handle = nvidia_ml_py.nvmlDeviceGetHandleByIndex(rank)
info = nvidia_ml_py.nvmlDeviceGetMemoryInfo(handle)
print(f"秩 {rank} GPU内存: {info.used / 1e9:.2f}GB")
\`\`\``,
  },
}

export const fsdpStrategies: TopicContent = {
  id: 'dist-fsdp',
  title: { en: 'FSDP: Sharding Strategies', zh: 'FSDP：分片策略' },
  contentType: 'code',
  content: {
    en: `Fully Sharded Data Parallel (FSDP) automatically shards model parameters, gradients, and optimizer states across GPUs, dramatically reducing per-GPU memory.

## FSDP Sharding Strategies

**1. FULL_SHARD** (default, most aggressive)
- Shard params, gradients, optimizer states across all GPUs
- Memory: ~1/N (N = num GPUs)
- All-gather before forward, reduce-scatter after backward
- Latency: High communication

**2. SHARD_GRAD_OP**
- Shard gradients + optimizer states, but replicate params
- Memory: ~1/2 * 1/N (better than DDP, worse than FULL_SHARD)
- All-gather only before backward (gradients computed locally first)
- Latency: Lower than FULL_SHARD

**3. NO_SHARD**
- No sharding, only synchronize like DDP
- Memory: Baseline (no savings)
- Use for debugging or when model fits on one GPU

## Complete FSDP Example with Transformer

\`\`\`python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

class TransformerBlock(nn.Module):
    """Single transformer block: multi-head attention + FFN."""
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

class TransformerModel(nn.Module):
    """Multi-layer transformer for sequence classification."""
    def __init__(self, vocab_size=10000, seq_len=512, dim=768, num_layers=12, num_heads=12, hidden_dim=3072):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(dim, 10)  # 10 classes

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embed(x) + self.pos_embed[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        # Pool: take [CLS] token (first position)
        x = x[:, 0, :]
        logits = self.classifier(x)
        return logits

def setup_fsdp():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    dist.destroy_process_group()

def main():
    setup_fsdp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build model
    model = TransformerModel(dim=768, num_layers=12, num_heads=12)
    model = model.cuda()

    # Auto-wrap policy: wrap each TransformerBlock
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e6  # Wrap modules > 1M params
    )

    # Wrap with FSDP using FULL_SHARD strategy
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Most memory efficient
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=False),  # Set True to CPU-offload params (slower)
        backward_prefetch=True,  # Prefetch params during backward
        mixed_precision=torch.cuda.amp.autocast,  # Mixed precision (fp16 gradients)
    )

    # Optimizer (only wraps leaf modules after FSDP)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy dataset
    X = torch.randint(0, 10000, (1000, 512))
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()  # FSDP: reduce-scatter gradients, unshards for update
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}')

        dist.barrier()
        if rank == 0:
            print(f'Epoch {epoch}: Avg Loss = {total_loss / len(loader):.4f}')

    cleanup_fsdp()

if __name__ == '__main__':
    main()
\`\`\`

## Memory Savings Comparison

| Strategy | Memory per GPU | Communication | Use Case |
|----------|----------------|----------------|----------|
| **DDP** | 140GB (full model) | Reduce-scatter gradients | Model fits on GPU |
| **SHARD_GRAD_OP** | ~70GB (params replicated, grad+opt sharded) | Medium | Medium models |
| **FULL_SHARD** | ~35GB (all sharded, 4 GPUs) | High (all-gather + reduce-scatter) | Large models (70B+) |
| **FULL_SHARD + CPU Offload** | ~5GB GPU, 140GB CPU | Very high | Extreme cases (175B+) |

**Mixed Precision with FSDP**:
- Reduces gradients to fp16 before communication
- Keeps master weights in fp32 for numerical stability
- Example: 70B model at fp32 = 280GB → mixed precision = 140GB + 140GB (master) ≈ 70GB effective

**When to use CPU offload**:
- All GPUs have NVMe or fast CPU-GPU link (PCIe 4.0+)
- Sacrifice 50% throughput for 10x memory savings
- Useful for model inspection/debugging on single GPU`,
    zh: `完全分片数据并行(FSDP)自动将模型参数、梯度和优化器状态分片到GPU，大幅降低单GPU内存。

## FSDP分片策略

**1. FULL_SHARD**(默认，最激进)
- 将参数、梯度、优化器状态分片到所有GPU
- 内存: ~1/N (N = GPU数)
- 前向前全聚，反向后约化分散
- 延迟: 高通信

**2. SHARD_GRAD_OP**
- 分片梯度+优化器状态，复制参数
- 内存: ~1/2 * 1/N (比DDP好，比FULL_SHARD差)
- 仅在反向前全聚(梯度本地先计算)
- 延迟: 比FULL_SHARD低

**3. NO_SHARD**
- 无分片，仅同步如DDP
- 内存: 基线(无节省)
- 用于调试或模型适配单GPU

## 完整FSDP例子含Transformer

\`\`\`python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

class TransformerBlock(nn.Module):
    """单个transformer块: 多头注意力 + FFN."""
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # 自注意力含残差
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 前馈含残差
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

class TransformerModel(nn.Module):
    """多层transformer用于序列分类。"""
    def __init__(self, vocab_size=10000, seq_len=512, dim=768, num_layers=12, num_heads=12, hidden_dim=3072):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(dim, 10)  # 10类

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embed(x) + self.pos_embed[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        # 池化: 取[CLS]令牌(首位置)
        x = x[:, 0, :]
        logits = self.classifier(x)
        return logits

def setup_fsdp():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    dist.destroy_process_group()

def main():
    setup_fsdp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 构建模型
    model = TransformerModel(dim=768, num_layers=12, num_heads=12)
    model = model.cuda()

    # 自动包装策略: 包装各TransformerBlock
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e6  # 包装 > 1M参数的模块
    )

    # 用FSDP包装，使用FULL_SHARD策略
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 最内存高效
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=False),  # 设True卸载参数到CPU(更慢)
        backward_prefetch=True,  # 反向时预取参数
        mixed_precision=torch.cuda.amp.autocast,  # 混合精度(fp16梯度)
    )

    # 优化器(仅包装FSDP后叶模块)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 虚拟数据集
    X = torch.randint(0, 10000, (1000, 512))
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()  # FSDP: 约化分散梯度，为更新反分片
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0 and rank == 0:
                print(f'轮次 {epoch}, 批 {batch_idx}, 损失 {loss.item():.4f}')

        dist.barrier()
        if rank == 0:
            print(f'轮次 {epoch}: 平均损失 = {total_loss / len(loader):.4f}')

    cleanup_fsdp()

if __name__ == '__main__':
    main()
\`\`\`

## 内存节省对比

| 策略 | 单GPU内存 | 通信 | 用途 |
|---------|------------|------|------|
| **DDP** | 140GB(完整模型) | 约化分散梯度 | 模型适配GPU |
| **SHARD_GRAD_OP** | ~70GB(参数复制，梯度+优化器分片) | 中等 | 中等模型 |
| **FULL_SHARD** | ~35GB(全分片，4 GPU) | 高(全聚+约化分散) | 大模型(70B+) |
| **FULL_SHARD + CPU卸载** | ~5GB GPU, 140GB CPU | 极高 | 极端情况(175B+) |

**FSDP混合精度**:
- 通信前将梯度降到fp16
- 为数值稳定性保持fp32主权重
- 例: 70B模型fp32 = 280GB → 混合精度 = 140GB + 140GB(主) ≈ 70GB有效

**何时用CPU卸载**:
- 所有GPU有NVMe或快速CPU-GPU链接(PCIe 4.0+)
- 牺牲50%吞吐换10倍内存节省
- 用于单GPU模型检查/调试`,
  },
}

export const pipelineParallelism: TopicContent = {
  id: 'dist-pipeline',
  title: { en: 'Pipeline Parallelism', zh: '流水线并行' },
  contentType: 'article',
  content: {
    en: `Pipeline parallelism (PP) breaks a model into sequential stages, assigning each to a GPU. Unlike model parallelism (MP), PP overlaps forward and backward passes across stages to hide latency.

## Problem with Model Parallelism

With naive MP, GPUs sit idle:
1. GPU-0 forward: computes first N layers → outputs to GPU-1
2. GPU-1 forward: waits for GPU-0 → computes layers N+1 to 2N → outputs to GPU-2
3. GPU-0 idle, GPU-1 idle while GPU-2 processes

Result: only 1 GPU active at a time. Utilization = 1/num_stages.

## Pipeline Parallelism Solution: Micro-Batching

Split batch into micro-batches (e.g., batch=64 → 8 micro-batches of 8 samples each).

**Timeline (2 stages, 4 micro-batches)**:
\`\`\`
GPU-0       GPU-1
F(mb0)  →   F(mb0)
F(mb1)  →   F(mb1)   |   B(mb0) ← backward
F(mb2)  →   F(mb2)   |   B(mb1)
F(mb3)  →   F(mb3)   |   B(mb2)
            B(mb0)   ←
            B(mb1)   ←   B(mb3)
            B(mb2)   ←
            B(mb3)   ←
\`\`\`

With optimal pipelining (1F1B schedule), both GPUs are busy ≈80% of the time.

## GPipe vs 1F1B Schedule

**GPipe (Google)**:
- Forward: all micro-batches through all stages
- Backward: all micro-batches through all stages
- Simpler to implement
- Drawback: Large activation memory (stores all intermediate activations)
- Utilization: ~40%

**1F1B (1 Forward, 1 Backward)**:
- Interleave: F(mb0), F(mb1), B(mb0), F(mb2), B(mb1), F(mb3), B(mb2), B(mb3)
- Activations released immediately after backward
- Utilization: ~80%
- Used in practice (e.g., Megatron)

## Bubble Overhead

With K stages and M micro-batches, bubble time (idle cycles) is:

\`\`\`
Bubble ratio = (K - 1) / (M + K - 1)
\`\`\`

Example: 8 stages, 32 micro-batches:
- Bubble = 7 / (32 + 8 - 1) = 7/39 ≈ 18%
- Utilization = 82%

**To minimize bubble**:
- Increase M (more micro-batches) → reduces bubble ratio
- Decrease K (fewer stages) → faster sync
- Trade-off: More micro-batches = more communication + memory variance

## 3D Parallelism: Combining DP + TP + PP

Production models combine all three:

**Data Parallelism (DP)**:
- Replicate pipeline-parallel model across multiple nodes
- Synchronize gradients across DP groups

**Tensor Parallelism (TP)**:
- Within each pipeline stage, shard weight matrices column-wise
- Reduces activation memory per GPU
- Example: 4-way TP in each stage = 4x smaller activations

**Pipeline Parallelism (PP)**:
- Split model into K stages, each stage on different GPU
- Use 1F1B micro-batching for overlap

**Example Megatron-LM Configuration**:
- Model: GPT-175B
- Hardware: 1024 × A100 GPUs
- DP: 32 groups (32 data-parallel replicas)
- TP: 4 (each stage has 4-way tensor parallel)
- PP: 8 stages (model split into 8 pieces)
- Effective parallelism: 32 × 4 × 8 = 1024 ✓

**Memory math**:
- Batch size: 2048 global (64 per DP group per micro-batch)
- Per-GPU: ~200GB activations (fp16) + 40GB model params + 40GB optimizer = 280GB
- Fits on A100 40GB? No. But FSDP + TP reduce model params to 10GB/GPU. With TP, activations become 50GB. Total ≈ 100GB → feasible with checkpointing.

## When to Use Each Strategy

| Problem | Solution |
|---------|----------|
| Model fits on 1 GPU, need more throughput | Data Parallelism only |
| Model = 10x GPU VRAM | Add Tensor Parallelism (shard attention, MLPs) |
| Model = 100x GPU VRAM | Add Pipeline Parallelism (stage the model) |
| Activation memory explodes with long sequences | Gradient checkpointing + PP |
| Need 1000s of GPUs | 3D parallelism (DP + TP + PP) |

## Key Takeaways

- **PP hides latency** by overlapping forward/backward of different micro-batches
- **1F1B achieves 80% utilization** vs 40% with GPipe
- **Bubble overhead = (K-1)/(M+K-1)** — minimize by increasing M (micro-batches)
- **3D parallelism** is essential for models > 100B parameters across many GPUs`,
    zh: `流水线并行(PP)将模型分解为连续阶段，各分配到一个GPU。与朴素模型并行(MP)不同，PP跨阶段重叠前向和反向传播以隐藏延迟。

## 朴素模型并行的问题

用朴素MP，GPU闲置：
1. GPU-0前向: 计算前N层 → 输出到GPU-1
2. GPU-1前向: 等待GPU-0 → 计算层N+1至2N → 输出到GPU-2
3. GPU-0闲置，GPU-1闲置，仅GPU-2处理

结果: 一次只1个GPU活跃。利用率 = 1/阶段数。

## 流水线并行方案: 微批

将批分为微批(如：批=64 → 8微批各8样本)。

**时间线(2阶段, 4微批)**:
\`\`\`
GPU-0       GPU-1
F(微0)  →   F(微0)
F(微1)  →   F(微1)   |   B(微0) ←反向
F(微2)  →   F(微2)   |   B(微1)
F(微3)  →   F(微3)   |   B(微2)
            B(微0)   ←
            B(微1)   ←   B(微3)
            B(微2)   ←
            B(微3)   ←
\`\`\`

用最优流水线(1F1B调度)，两GPU忙碌≈80%时间。

## GPipe vs 1F1B调度

**GPipe(谷歌)**:
- 前向: 所有微批通过所有阶段
- 反向: 所有微批通过所有阶段
- 更简单实现
- 缺点: 大激活内存(存储所有中间激活)
- 利用率: ~40%

**1F1B(1前1反)**:
- 交错: F(微0), F(微1), B(微0), F(微2), B(微1), F(微3), B(微2), B(微3)
- 反向后立即释放激活
- 利用率: ~80%
- 实践中使用(如Megatron)

## 气泡开销

K阶段和M微批，气泡时间(闲置周期)为:

\`\`\`
气泡比 = (K - 1) / (M + K - 1)
\`\`\`

例: 8阶段, 32微批:
- 气泡 = 7 / (32 + 8 - 1) = 7/39 ≈ 18%
- 利用率 = 82%

**最小化气泡**:
- 增加M(更多微批) → 降低气泡比
- 减少K(更少阶段) → 更快同步
- 权衡: 更多微批 = 更多通信 + 内存方差

## 3D并行: 结合 DP + TP + PP

生产模型结合三种:

**数据并行(DP)**:
- 跨多节点复制流水线并行模型
- 同步DP组间梯度

**张量并行(TP)**:
- 各流水线阶段内，按列分片权重矩阵
- 降低单GPU激活内存
- 例: 各阶段4路TP = 4倍少激活

**流水线并行(PP)**:
- 将模型分为K阶段，各在不同GPU
- 用1F1B微批重叠

**例Megatron-LM配置**:
- 模型: GPT-175B
- 硬件: 1024 × A100 GPU
- DP: 32组(32数据并行副本)
- TP: 4(各阶段4路张量并行)
- PP: 8阶段(模型分为8块)
- 有效并行: 32 × 4 × 8 = 1024 ✓

**内存数学**:
- 批大小: 2048全局(64每DP组每微批)
- 单GPU: ~200GB激活(fp16) + 40GB模型参数 + 40GB优化器 = 280GB
- 适配A100 40GB? 否。但FSDP + TP将模型参数减到10GB/GPU。TP下激活变50GB。总≈100GB → 用检查点可行。

## 何时用各策略

| 问题 | 方案 |
|------|------|
| 模型适配1 GPU，需更多吞吐 | 仅数据并行 |
| 模型 = 10倍GPU VRAM | 加张量并行(分片注意力、MLP) |
| 模型 = 100倍GPU VRAM | 加流水线并行(分阶段) |
| 长序列激活内存爆炸 | 梯度检查点 + PP |
| 需1000s GPU | 3D并行(DP + TP + PP) |

## 关键要点

- **PP隐藏延迟**通过不同微批的前向/反向重叠
- **1F1B达成80%利用率**vs GPipe的40%
- **气泡开销 = (K-1)/(M+K-1)** — 通过增加M(微批)最小化
- **3D并行**对模型 > 100B参数跨多GPU必需`,
  },
}

export const debuggingDistributed: TopicContent = {
  id: 'dist-debugging',
  title: { en: 'Debugging Distributed Training', zh: '调试分布式训练' },
  contentType: 'code',
  content: {
    en: `Distributed training introduces new failure modes: deadlocks, rank crashes, silent data corruption, and NCCL errors. Here's a systematic debugging approach.

## Common Failure Modes

### 1. Deadlock: Training Hangs Forever

**Symptoms**:
- Process queue fills (NCCL timeout after 30min)
- Some ranks still running, others frozen
- No error message (silent failure)

**Root Causes**:
1. Uneven batch sizes → rank-0 has data, rank-1 doesn't → barrier waits forever
2. Conditional communications → if only rank-0 calls \`all_reduce\`, others waiting
3. Out-of-memory (OOM) on one rank → process dies → others wait at barrier

**Debugging**:

\`\`\`python
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - Rank %(process)d - %(message)s')

def safe_barrier(tag):
    """Barrier with logging to detect hangs."""
    logging.info(f'Entering barrier: {tag}')
    dist.barrier()
    logging.info(f'Exiting barrier: {tag}')

# In training loop:
for batch_idx, batch in enumerate(loader):
    logging.info(f'Rank {rank}: processing batch {batch_idx}')

    # Only call collective ops if all ranks will participate
    if batch is not None:
        loss = train_step(batch)
        logging.info(f'Rank {rank}: loss = {loss}')
    else:
        logging.warning(f'Rank {rank}: empty batch!')
        # Synchronize even with dummy computation
        loss = torch.tensor(0.0, device=device)

    # All ranks call barrier → if any hang, you see which batch
    safe_barrier(f'batch_{batch_idx}')
\`\`\`

### 2. NCCL Error: "NCCL operation timed out"

**Symptoms**:
\`\`\`
RuntimeError: NCCL operation timed out after 30m
\`\`\`

**Causes**:
1. Network congestion/flakiness
2. Rank crash without error propagation
3. Misconfigured NCCL (wrong backend, unsupported operation)

**Fix**:

\`\`\`bash
# Increase NCCL timeout (default 30min)
export NCCL_TIMEOUT=1800  # 30 minutes → 60 minutes
export NCCL_DEBUG=INFO     # Verbose NCCL logs
export NCCL_DEBUG_SUBSYS=COLL  # Log collective ops only

# Run training
torchrun --nproc_per_node=4 train.py 2>&1 | tee nccl_debug.log
\`\`\`

Parse logs for which rank is slow:
\`\`\`bash
grep "NCCL.*timed out" nccl_debug.log
grep "Rank.*all_reduce" nccl_debug.log | tail -20
\`\`\`

### 3. Rank Mismatch / Type Mismatch

**Symptoms**:
\`\`\`
RuntimeError: expected dtype Float but got Long on rank 2
\`\`\`

**Cause**: Different dtypes on different ranks before collective op.

**Fix**:

\`\`\`python
# Ensure all inputs are same dtype before all_reduce
x = x.float()  # Force consistent dtype
dist.all_reduce(x)
\`\`\`

### 4. OOM on Single Rank

**Symptoms**:
- Rank-0, rank-1, rank-2 training fine
- Rank-3 suddenly OOM and dies
- Other ranks deadlock

**Debugging**:

\`\`\`python
import torch

def log_gpu_memory(rank, tag=''):
    reserved = torch.cuda.memory_reserved() / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f'Rank {rank} [{tag}]: Reserved {reserved:.2f}GB, Allocated {allocated:.2f}GB')

# Call at strategic points:
log_gpu_memory(rank, 'after_model_init')
for batch_idx, batch in enumerate(loader):
    log_gpu_memory(rank, f'batch_{batch_idx}_start')
    loss.backward()
    log_gpu_memory(rank, f'batch_{batch_idx}_after_backward')
    optimizer.step()
    log_gpu_memory(rank, f'batch_{batch_idx}_after_optim_step')
\`\`\`

If rank-3 uses more memory, likely issue:
1. Uneven batch sampling (rank-3 gets larger batches)
2. Rank-3 accumulating gradients (forgot \`optimizer.zero_grad()\`)
3. Rank-3 model slightly different (extra layers not in other ranks)

## Complete Debugging Checklist

\`\`\`python
def debug_distributed_training(rank, world_size, args):
    \"\"\"Systematic debugging wrapper.\"\"\"

    # 1. VERIFY SETUP
    print(f'Rank {rank}/{world_size}')
    assert rank < world_size, 'Rank exceeds world_size'

    # 2. VERIFY DATA LOADING
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    # Check no data leakage
    all_indices = []
    for batch in loader:
        all_indices.extend(batch['index'].tolist())

    dist.all_gather_object([all_indices], all_indices)
    all_ranks_indices = [None] * world_size
    dist.all_gather_object(all_ranks_indices, all_indices)

    if rank == 0:
        # Verify no overlap
        flat = [idx for rank_idxs in all_ranks_indices for idx in rank_idxs]
        assert len(flat) == len(set(flat)), 'Data overlap detected!'
        print(f'✓ Data distribution verified')

    # 3. VERIFY MODEL IDENTICAL
    model = MyModel()

    # Broadcast model state from rank-0 to all
    for name, param in model.named_parameters():
        dist.broadcast(param, src=0)

    if rank == 0:
        print(f'✓ All ranks have identical models')

    # 4. VERIFY COLLECTIVE OPS BALANCED
    for batch_idx in range(3):  # Test first 3 batches
        batch = next(iter(loader))
        if batch is None:
            print(f'Rank {rank}: batch is None at idx {batch_idx}')

        # All ranks compute loss
        loss = model(batch)
        dist.barrier()

        # All ranks call backward
        loss.backward()
        dist.barrier()

        # All ranks call all_reduce (built into DDP.step)
        optimizer.step()
        dist.barrier()

        if rank == 0:
            print(f'✓ Batch {batch_idx}: all ranks synchronized')

    return True

# Usage:
debug_distributed_training(rank, world_size, args)
\`\`\`

## Advanced Debugging Tools

**1. torch.distributed.barrier() with Timeout**

\`\`\`python
try:
    dist.barrier(timeout=timedelta(seconds=60))
except RuntimeError as e:
    print(f'Rank {rank}: barrier timeout! {e}')
    # Log state and exit gracefully
    sys.exit(1)
\`\`\`

**2. Rank-Specific Logging**

\`\`\`python
import logging

# Only rank-0 logs to prevent file spam
if rank == 0:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)  # Suppress non-critical

logger.info('This only prints on rank-0')
\`\`\`

**3. Memory Monitor Thread**

\`\`\`python
import threading
import time

def monitor_memory(rank, interval=5):
    while True:
        reserved = torch.cuda.memory_reserved() / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f'Rank {rank}: {reserved:.1f}GB reserved, {allocated:.1f}GB allocated')
        time.sleep(interval)

# Start monitor in background
monitor_thread = threading.Thread(target=monitor_memory, args=(rank,), daemon=True)
monitor_thread.start()
\`\`\`

## Quick Reference: What to Check

| Symptom | Check |
|---------|-------|
| Hangs forever | Batch size uneven? \`DistributedSampler\` used? |
| NCCL timeout | NCCL_DEBUG=INFO? Rank crashed? Network down? |
| Type mismatch | All ranks same dtype before \`all_reduce\`? |
| Silent data corruption | Sampler overlap? Gradient accumulation bug? |
| OOM on rank-N only | Batch size per rank? Model params identical? |
| Slow/inefficient training | Communication overhead? Try NCCL_DEBUG profiling |`,
    zh: `分布式训练引入新失败模式: 死锁、秩崩溃、无声数据损坏和NCCL错误。以下是系统调试方法。

## 常见失败模式

### 1. 死锁: 训练永远挂起

**症状**:
- 进程队列满(NCCL超时30分钟后)
- 某些秩仍运行，其他冻结
- 无错误消息(无声失败)

**根本原因**:
1. 不均匀批大小 → 秩-0有数据，秩-1没有 → 屏障永远等待
2. 条件通信 → 仅秩-0调用\`all_reduce\`，其他等待
3. 单秩内存溢出 → 进程死亡 → 其他在屏障等待

**调试**:

\`\`\`python
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - Rank %(process)d - %(message)s')

def safe_barrier(tag):
    """屏障含日志以检测挂起。"""
    logging.info(f'进入屏障: {tag}')
    dist.barrier()
    logging.info(f'退出屏障: {tag}')

# 在训练循环中:
for batch_idx, batch in enumerate(loader):
    logging.info(f'秩 {rank}: 处理批 {batch_idx}')

    # 仅在所有秩参与时调用集合操作
    if batch is not None:
        loss = train_step(batch)
        logging.info(f'秩 {rank}: 损失 = {loss}')
    else:
        logging.warning(f'秩 {rank}: 空批!')
        # 即使虚拟计算也同步
        loss = torch.tensor(0.0, device=device)

    # 所有秩调用屏障 → 任何挂起，见到哪个批
    safe_barrier(f'batch_{batch_idx}')
\`\`\`

### 2. NCCL错误: "NCCL operation timed out"

**症状**:
\`\`\`
RuntimeError: NCCL operation timed out after 30m
\`\`\`

**原因**:
1. 网络拥塞/不稳定
2. 秩崩溃无错误传播
3. NCCL配置错误(后端错误、不支持操作)

**修复**:

\`\`\`bash
# 增加NCCL超时(默认30分钟)
export NCCL_TIMEOUT=1800  # 30分钟 → 60分钟
export NCCL_DEBUG=INFO     # 详细NCCL日志
export NCCL_DEBUG_SUBSYS=COLL  # 仅记录集合操作

# 运行训练
torchrun --nproc_per_node=4 train.py 2>&1 | tee nccl_debug.log
\`\`\`

解析日志找哪秩慢:
\`\`\`bash
grep "NCCL.*timed out" nccl_debug.log
grep "Rank.*all_reduce" nccl_debug.log | tail -20
\`\`\`

### 3. 秩不匹配 / 类型不匹配

**症状**:
\`\`\`
RuntimeError: expected dtype Float but got Long on rank 2
\`\`\`

**原因**: 集合操作前秩上不同dtype。

**修复**:

\`\`\`python
# 确保all_reduce前所有输入同dtype
x = x.float()  # 强制一致dtype
dist.all_reduce(x)
\`\`\`

### 4. 单秩OOM

**症状**:
- 秩-0、秩-1、秩-2训练正常
- 秩-3突然OOM和死亡
- 其他秩死锁

**调试**:

\`\`\`python
import torch

def log_gpu_memory(rank, tag=''):
    reserved = torch.cuda.memory_reserved() / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f'秩 {rank} [{tag}]: 保留 {reserved:.2f}GB, 分配 {allocated:.2f}GB')

# 在策略点调用:
log_gpu_memory(rank, 'after_model_init')
for batch_idx, batch in enumerate(loader):
    log_gpu_memory(rank, f'batch_{batch_idx}_start')
    loss.backward()
    log_gpu_memory(rank, f'batch_{batch_idx}_after_backward')
    optimizer.step()
    log_gpu_memory(rank, f'batch_{batch_idx}_after_optim_step')
\`\`\`

若秩-3用更多内存，可能问题:
1. 不均匀批采样(秩-3得更大批)
2. 秩-3积累梯度(忘记\`optimizer.zero_grad()\`)
3. 秩-3模型稍异(其他秩无额外层)

## 完整调试检查清单

\`\`\`python
def debug_distributed_training(rank, world_size, args):
    """系统调试包装。"""

    # 1. 验证设置
    print(f'秩 {rank}/{world_size}')
    assert rank < world_size, '秩超过world_size'

    # 2. 验证数据加载
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    # 检查无数据泄漏
    all_indices = []
    for batch in loader:
        all_indices.extend(batch['index'].tolist())

    dist.all_gather_object([all_indices], all_indices)
    all_ranks_indices = [None] * world_size
    dist.all_gather_object(all_ranks_indices, all_indices)

    if rank == 0:
        # 验证无重叠
        flat = [idx for rank_idxs in all_ranks_indices for idx in rank_idxs]
        assert len(flat) == len(set(flat)), '检测到数据重叠!'
        print(f'✓ 数据分布已验证')

    # 3. 验证模型相同
    model = MyModel()

    # 从秩-0广播模型状态到所有秩
    for name, param in model.named_parameters():
        dist.broadcast(param, src=0)

    if rank == 0:
        print(f'✓ 所有秩有相同模型')

    # 4. 验证集合操作均衡
    for batch_idx in range(3):  # 测试前3批
        batch = next(iter(loader))
        if batch is None:
            print(f'秩 {rank}: 批在索引 {batch_idx} 为空')

        # 所有秩计算损失
        loss = model(batch)
        dist.barrier()

        # 所有秩调用反向
        loss.backward()
        dist.barrier()

        # 所有秩调用all_reduce(内置DDP.step)
        optimizer.step()
        dist.barrier()

        if rank == 0:
            print(f'✓ 批 {batch_idx}: 所有秩已同步')

    return True

# 用法:
debug_distributed_training(rank, world_size, args)
\`\`\`

## 高级调试工具

**1. torch.distributed.barrier()含超时**

\`\`\`python
try:
    dist.barrier(timeout=timedelta(seconds=60))
except RuntimeError as e:
    print(f'秩 {rank}: 屏障超时! {e}')
    # 记录状态并优雅退出
    sys.exit(1)
\`\`\`

**2. 秩特定日志**

\`\`\`python
import logging

# 仅秩-0日志防止文件垃圾
if rank == 0:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)  # 抑制非关键

logger.info('这仅在秩-0打印')
\`\`\`

**3. 内存监视器线程**

\`\`\`python
import threading
import time

def monitor_memory(rank, interval=5):
    while True:
        reserved = torch.cuda.memory_reserved() / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f'秩 {rank}: {reserved:.1f}GB保留, {allocated:.1f}GB分配')
        time.sleep(interval)

# 后台启动监视器
monitor_thread = threading.Thread(target=monitor_memory, args=(rank,), daemon=True)
monitor_thread.start()
\`\`\`

## 快速参考: 需检查内容

| 症状 | 检查 |
|------|------|
| 永远挂起 | 批大小不均? 用\`DistributedSampler\`? |
| NCCL超时 | NCCL_DEBUG=INFO? 秩崩溃? 网络断? |
| 类型不匹配 | \`all_reduce\`前所有秩同dtype? |
| 无声数据损坏 | 采样器重叠? 梯度积累bug? |
| 仅秩-N OOM | 单秩批大小? 模型参数相同? |
| 慢/低效训练 | 通信开销? 试NCCL_DEBUG性能分析 |`,
  },
}
