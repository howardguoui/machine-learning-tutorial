import type { TopicContent } from '../types'

export const gpuMemoryHierarchy: TopicContent = {
  id: 'gpu-memory-hierarchy',
  title: { en: 'GPU Memory Hierarchy & Roofline Model', zh: 'GPU内存层次结构与屋顶线模型' },
  contentType: 'article',
  content: {
    en: `## GPU Memory Hierarchy & Roofline Model

Understanding GPU memory is essential for writing fast kernels. Every operation is bottlenecked by either compute or memory bandwidth — the roofline model tells you which.

---

### Memory Hierarchy (H100 specs)

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Registers | 256KB/SM | ~80 TB/s | 1 cycle |
| Shared Memory (SRAM) | 228KB/SM | ~33 TB/s | ~5 cycles |
| L2 Cache | 50MB | ~12 TB/s | ~200 cycles |
| HBM3 (DRAM) | 80GB | 3.35 TB/s | ~700 cycles |

**Key insight:** Shared memory is 10x faster than HBM. The goal of every optimization is to maximize work done per HBM byte accessed.

---

### The Roofline Model

An operation is either:
- **Compute-bound:** The GPU's FLOPs are the bottleneck (utilization > 80%)
- **Memory-bandwidth-bound:** HBM reads/writes are the bottleneck

**Arithmetic Intensity (AI):** FLOPs / bytes accessed from HBM

\`\`\`
Ridge point = Peak FLOPs / Peak Memory BW
H100: 989 TFLOPS (bf16) / 3.35 TB/s = ~295 FLOPs/byte

If your operation's AI < 295: memory-bandwidth-bound
If your operation's AI > 295: compute-bound
\`\`\`

**Examples:**
- Matrix multiply (N=4096): AI ≈ 4096/2 = 2048 → compute-bound ✓
- Element-wise ReLU: AI ≈ 1/4 = 0.25 → heavily memory-bound
- LayerNorm: AI ≈ 10/4 = 2.5 → memory-bound

---

### Practical Implications

**For memory-bound ops (element-wise, normalization):**
- Kernel fusion is critical — fuse multiple ops to reduce HBM round-trips
- FlashAttention works because it tiles to avoid materializing O(N²) attention matrix

**For compute-bound ops (large matmuls):**
- Focus on maximizing tensor core utilization
- Use bf16/fp16 for 2x throughput vs fp32
- Ensure matrix dimensions are multiples of 16 (tensor core alignment)`,
    zh: `## GPU内存层次结构与屋顶线模型

理解GPU内存对于编写快速内核至关重要。每个操作都受计算或内存带宽的瓶颈限制——屋顶线模型告诉你是哪个。

---

### 内存层次结构（H100规格）

| 级别 | 大小 | 带宽 | 延迟 |
|------|------|------|------|
| 寄存器 | 256KB/SM | ~80 TB/s | 1周期 |
| 共享内存(SRAM) | 228KB/SM | ~33 TB/s | ~5周期 |
| L2缓存 | 50MB | ~12 TB/s | ~200周期 |
| HBM3(DRAM) | 80GB | 3.35 TB/s | ~700周期 |

**关键洞察：** 共享内存比HBM快10倍。每次优化的目标都是最大化每个HBM字节访问所完成的工作量。

**屋顶线模型：** 算术强度(AI) = FLOPs / HBM访问字节数。H100的屋顶点约为295 FLOPs/字节。低于此值为内存带宽瓶颈，高于此值为计算瓶颈。`,
  },
}

export const torchCompileInternals: TopicContent = {
  id: 'gpu-torch-compile-internals',
  title: { en: 'torch.compile Internals', zh: 'torch.compile内部原理' },
  contentType: 'code',
  content: {
    en: `## torch.compile Internals

torch.compile (introduced in PyTorch 2.0) transforms eager Python code into optimized GPU kernels through a 3-stage pipeline.

---

### The 3-Stage Pipeline

\`\`\`
Stage 1: Dynamo (Graph Capture)
  - Intercepts Python bytecode execution
  - Traces through Python to build a computation graph
  - Handles Python control flow, dynamic shapes

Stage 2: AOTAutograd (Ahead-of-Time Autograd)
  - Ahead-of-time traces both forward and backward passes
  - Produces a flat FX graph for the entire forward+backward

Stage 3: Inductor (Kernel Generation)
  - Takes the FX graph
  - Decides which ops to fuse
  - Generates Triton kernels for GPU
  - Compiles via Triton → PTX → CUDA binary
\`\`\`

---

### Graph Breaks

Graph breaks occur when Dynamo can't trace through Python code:

\`\`\`python
import torch

# GOOD — no graph break
def model_forward(x, weight):
    x = torch.mm(x, weight)
    x = torch.nn.functional.relu(x)
    return x

# BAD — graph break on Python control flow
def model_forward_broken(x, weight, flag):
    x = torch.mm(x, weight)
    if flag:           # Graph break here! Dynamic Python condition
        x = x * 2
    return x

# Diagnose graph breaks
import torch._dynamo
explanation = torch._dynamo.explain(model_forward_broken)(
    torch.randn(4, 4), torch.randn(4, 4), True
)
print(explanation.break_reasons)
\`\`\`

---

### Compilation Modes

\`\`\`python
model = MyTransformer()

# default: balance between compile time and runtime
compiled = torch.compile(model)

# reduce-overhead: minimize Python overhead, good for small models
compiled = torch.compile(model, mode='reduce-overhead')

# max-autotune: exhaustive kernel search, slowest compile (~10min)
compiled = torch.compile(model, mode='max-autotune')

# Benchmark
import time
x = torch.randn(32, 512, dtype=torch.float16, device='cuda')

# Warmup (first run triggers compilation)
for _ in range(3):
    _ = compiled(x)

start = time.perf_counter()
for _ in range(100):
    _ = compiled(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print("Compiled: {:.2f} ms/iter".format(elapsed * 10))
\`\`\`

---

### Why Fusion Helps

Without fusion: ReLU reads tensor from HBM, applies relu, writes back to HBM.

With fusion (compiled): LayerNorm + Dropout + ReLU reads once, applies all ops in SRAM, writes once. Reduces HBM traffic by 3x.

**Typical speedups:** 1.5x-3x for transformer inference, 1.2x-2x for training.`,
    zh: `## torch.compile内部原理

torch.compile（PyTorch 2.0引入）通过3阶段流水线将急切Python代码转换为优化的GPU内核。

---

### 三阶段流水线

1. **Dynamo（图捕获）** — 拦截Python字节码，追踪构建计算图
2. **AOTAutograd（提前自动微分）** — 提前追踪前向和反向传播，生成平坦FX图
3. **Inductor（内核生成）** — 决定融合哪些算子，生成Triton内核并编译

---

### 图断裂

当Dynamo无法追踪Python代码时发生图断裂（如动态Python条件分支）。使用torch._dynamo.explain()诊断断裂原因。

---

### 为什么融合有帮助

没有融合：每个操作单独读写HBM。有融合：多个操作在SRAM中组合，一次读写HBM，减少3倍内存流量。**典型加速：** Transformer推理1.5x-3x，训练1.2x-2x。`,
  },
}

export const tritonKernelBasics: TopicContent = {
  id: 'gpu-triton-basics',
  title: { en: 'Triton Kernel Basics', zh: 'Triton内核基础' },
  contentType: 'code',
  content: {
    en: `## Triton Kernel Basics

Triton lets you write GPU kernels in Python-like syntax, abstracting away CUDA thread/warp management while giving tile-level control.

---

### Core Concepts

- **program_id:** Which tile this kernel instance processes (replaces CUDA blockIdx)
- **tl.load/tl.store:** Load/store a tile from/to global memory with bounds checking
- **Block size (BLOCK_SIZE):** Must be power of 2, controls the tile size

---

### Vector Addition Kernel

\`\`\`python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # Must be power of 2
):
    # Each program instance handles BLOCK_SIZE elements
    pid = tl.program_id(axis=0)

    # Compute the range of indices this instance handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to handle the last block (may be smaller)
    mask = offsets < n_elements

    # Load from global memory (HBM) into registers
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute in registers (fast!)
    output = x + y

    # Store result back to HBM
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024

    # Launch grid: how many program instances
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
\`\`\`

---

### Fused ReLU + Bias Kernel

\`\`\`python
@triton.jit
def fused_relu_bias_kernel(
    x_ptr, bias_ptr, output_ptr,
    n_cols, n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row from input + bias
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask)
    bias = tl.load(bias_ptr + col_offsets, mask=mask)

    # Fused: add bias then apply relu — no intermediate tensor written to HBM
    result = tl.maximum(x + bias, 0.0)

    tl.store(output_ptr + row_idx * n_cols + col_offsets, result, mask=mask)

# Benchmark vs PyTorch eager
import time

x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
bias = torch.randn(4096, device='cuda', dtype=torch.float16)

# PyTorch eager: 2 separate kernels, 2x HBM reads
t0 = time.perf_counter()
for _ in range(1000):
    out = torch.relu(x + bias)
torch.cuda.synchronize()
eager_ms = (time.perf_counter() - t0)

print("Eager: {:.2f}ms total".format(eager_ms * 1000))
# Triton version typically 1.5-2x faster for memory-bound ops
\`\`\`

Triton auto-tunes tile sizes and handles the complex CUDA thread/block mapping, letting you focus on the algorithm.`,
    zh: `## Triton内核基础

Triton允许用Python语法编写GPU内核，抽象CUDA线程/warp管理，同时提供块级控制。

---

### 核心概念

- **program_id：** 此内核实例处理哪个块（替代CUDA的blockIdx）
- **tl.load/tl.store：** 从全局内存加载/存储块，支持边界检查
- **BLOCK_SIZE：** 必须是2的幂，控制块大小

---

### 关键优势

与PyTorch急切模式相比，Triton融合内核通过减少HBM访问次数来提高内存带宽受限操作的速度。向量加法和融合ReLU+偏置是展示此优势的典型示例。

Triton自动调整块大小并处理复杂的CUDA线程/块映射，让你专注于算法本身。`,
  },
}

export const flashAttentionKernel: TopicContent = {
  id: 'gpu-flash-attention-kernel',
  title: { en: 'FlashAttention Kernel Walkthrough', zh: 'FlashAttention内核详解' },
  contentType: 'code',
  content: {
    en: `## FlashAttention Kernel Walkthrough

Standard attention materializes the full N×N attention matrix in HBM, requiring O(N²) memory. FlashAttention tiles the computation to stay in SRAM, achieving near-optimal IO.

---

### The Problem: Standard Attention IO

For N=4096, d=128:
- Q, K, V each: 4096 × 128 × 2 bytes = 1MB
- Attention matrix S = QK^T: 4096 × 4096 × 2 = 32MB written to HBM
- Total HBM IO: ~35MB just for attention weights

FlashAttention: ~3MB total (only Q, K, V; attention matrix stays in SRAM).

---

### The Online Softmax Trick

Standard softmax requires two passes: first compute max, then compute exp/sum.

FlashAttention computes softmax incrementally as new K,V blocks arrive:

\`\`\`
For each new K block:
  1. Compute new attention scores: s_new = q @ k_block.T
  2. Update running max: m_new = max(m_old, max(s_new))
  3. Rescale old accumulator: O = O * exp(m_old - m_new)
  4. Add new contribution: O += softmax(s_new) @ v_block
  5. Update normalizer: l_new = l_old * exp(m_old - m_new) + sum(exp(s_new - m_new))
\`\`\`

---

### Simplified Triton Implementation

\`\`\`python
import triton
import triton.language as tl

@triton.jit
def flash_attention_forward(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,  # Query block size
    BLOCK_N: tl.constexpr,  # Key/Value block size
    BLOCK_DHEAD: tl.constexpr,
):
    # Each program handles one query block for one head
    query_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Offsets for this query block
    q_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_DHEAD)

    # Load query block into SRAM
    Q = tl.load(Q_ptr + head_idx * seq_len * head_dim
                + q_offsets[:, None] * head_dim + d_offsets[None, :])

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    O_acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    # Iterate over K, V blocks (stays in SRAM)
    for j in range(0, seq_len, BLOCK_N):
        k_offsets = j + tl.arange(0, BLOCK_N)

        # Load K block
        K = tl.load(K_ptr + head_idx * seq_len * head_dim
                    + k_offsets[:, None] * head_dim + d_offsets[None, :])

        # Attention scores: (BLOCK_M, BLOCK_N)
        S = tl.dot(Q, tl.trans(K)) * scale

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(S, axis=1))

        # Load V block
        V = tl.load(V_ptr + head_idx * seq_len * head_dim
                    + k_offsets[:, None] * head_dim + d_offsets[None, :])

        # Update accumulator with rescaling
        alpha = tl.exp(m_i - m_new)
        O_acc = O_acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(tl.exp(S - m_new[:, None]), axis=1)
        O_acc += tl.dot(tl.exp(S - m_new[:, None]).to(tl.float16), V)
        m_i = m_new

    # Normalize and store
    O_acc = O_acc / l_i[:, None]
    tl.store(O_ptr + head_idx * seq_len * head_dim
             + q_offsets[:, None] * head_dim + d_offsets[None, :],
             O_acc.to(tl.float16))
\`\`\`

**Memory savings for N=4096:** Standard = 32MB for attention matrix. FlashAttention = 0MB (never materialized). Total IO reduction: ~10x.`,
    zh: `## FlashAttention内核详解

标准注意力将完整的N×N注意力矩阵实现在HBM中，需要O(N²)内存。FlashAttention将计算分块以保留在SRAM中，实现接近最优的IO。

---

### 在线softmax技巧

FlashAttention的核心创新是在线softmax：随着新的K,V块到来，增量计算softmax，无需预先知道所有注意力分数的最大值。

**内存节省（N=4096）：** 标准注意力矩阵32MB；FlashAttention为0MB（从不实现）。总IO减少约10倍。`,
  },
}

export const pytorchProfiler: TopicContent = {
  id: 'gpu-profiler',
  title: { en: 'PyTorch Profiler & Nsight', zh: 'PyTorch Profiler与Nsight' },
  contentType: 'code',
  content: {
    en: `## PyTorch Profiler & Nsight

Before optimizing, measure. PyTorch Profiler and Nsight Systems reveal where time is actually spent.

---

### PyTorch Profiler

\`\`\`python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyTransformer().cuda()
x = torch.randn(4, 512, 768, device='cuda')

# Basic profiling with chrome trace
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    with record_function("model_forward"):
        output = model(x)
        output.sum().backward()

# Print top ops by CUDA time
print(prof.key_averages().table(
    sort_by='cuda_time_total',
    row_limit=15,
))

# Export chrome trace (open at chrome://tracing)
prof.export_chrome_trace('trace.json')
\`\`\`

---

### CUDA Event-Based Timing

More accurate than Python time.perf_counter (accounts for async GPU execution):

\`\`\`python
def benchmark(fn, *args, warmup=10, iters=100):
    """Accurate GPU benchmark using CUDA events."""

    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    import statistics
    return {
        'mean_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'p95_ms': sorted(times)[int(0.95 * len(times))],
        'min_ms': min(times),
    }

# Usage
results = benchmark(model, x)
print("Mean: {:.2f}ms, P95: {:.2f}ms".format(results['mean_ms'], results['p95_ms']))
\`\`\`

---

### Memory Profiling

\`\`\`python
# Snapshot memory at peak
torch.cuda.memory._record_memory_history()

# Run forward/backward
output = model(x)
output.sum().backward()

# Save snapshot
torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')

# In Python after: use torch.cuda.memory_stats()
stats = torch.cuda.memory_stats()
peak_mb = stats['reserved_bytes.all.peak'] / 1024**2
print("Peak reserved memory: {:.1f} MB".format(peak_mb))
\`\`\`

**Nsight Systems workflow:** 1) Run \`nsys profile python train.py\` 2) Open .nsys-rep file in Nsight UI 3) Look for CPU↔GPU gaps (data loading bottleneck) and kernel serialization (missed parallelism).`,
    zh: `## PyTorch Profiler与Nsight

优化之前先测量。PyTorch Profiler和Nsight Systems揭示时间实际花费在哪里。

---

### PyTorch Profiler

使用profile上下文管理器捕获CPU和CUDA活动，生成chrome跟踪文件以可视化时间线，通过key_averages()找出最耗时的算子。

---

### CUDA事件计时

比Python time.perf_counter更准确（考虑GPU异步执行）：使用CUDA事件记录开始和结束时间，收集多次迭代的p50/p95/min延迟统计。

---

### Nsight Systems工作流

1. 运行 nsys profile python train.py
2. 在Nsight UI中打开.nsys-rep文件
3. 查找CPU↔GPU间隙（数据加载瓶颈）和内核序列化（错过的并行性）`,
  },
}

export const costPerformance: TopicContent = {
  id: 'gpu-cost-performance',
  title: { en: 'Cost/Performance at Scale', zh: '规模化成本与性能' },
  contentType: 'code',
  content: {
    en: `## Cost/Performance at Scale

Optimizing GPU training requires understanding FLOP counts, Model FLOP Utilization (MFU), and memory efficiency tradeoffs.

---

### FLOP Counting for Transformer Layers

\`\`\`python
def transformer_flops(
    seq_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    vocab_size: int,
    batch_size: int = 1,
) -> dict:
    """
    Count FLOPs for a transformer forward pass.
    (Multiply by 2 for backward pass in training)
    """
    d_head = d_model // n_heads

    # Attention per layer:
    # QKV projection: 3 * seq_len * d_model * d_model * 2
    qkv_flops = 3 * seq_len * d_model * d_model * 2

    # Attention scores: seq_len * seq_len * d_model * 2
    attn_flops = seq_len * seq_len * d_model * 2

    # Output projection: seq_len * d_model * d_model * 2
    out_proj_flops = seq_len * d_model * d_model * 2

    # FFN: 2 * seq_len * d_model * (4 * d_model) * 2
    ffn_flops = 2 * seq_len * d_model * 4 * d_model * 2

    per_layer = qkv_flops + attn_flops + out_proj_flops + ffn_flops
    total_flops = per_layer * n_layers * batch_size

    return {
        'total_flops': total_flops,
        'per_layer_flops': per_layer,
        'attention_flops': (qkv_flops + attn_flops + out_proj_flops) * n_layers * batch_size,
        'ffn_flops': ffn_flops * n_layers * batch_size,
    }

# Llama-3-8B example
flops = transformer_flops(
    seq_len=2048, d_model=4096, n_heads=32,
    n_layers=32, vocab_size=128256
)
print("FLOPs per forward pass: {:.2e}".format(flops['total_flops']))
# ≈ 2.1e12 FLOPs (2.1 TFLOPs)
\`\`\`

---

### Model FLOP Utilization (MFU)

\`\`\`python
import torch
import time

def measure_mfu(
    model,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    peak_flops: float,  # GPU peak FLOPs (e.g., H100 = 989e12)
) -> float:
    """Measure actual % of peak FLOPs being achieved."""

    x = torch.randint(0, 50000, (batch_size, seq_len), device='cuda')

    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(20):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 20

    # Compute achieved FLOPs
    model_flops = 6 * batch_size * seq_len * (12 * n_layers * d_model**2)
    achieved_flops = model_flops / elapsed

    mfu = achieved_flops / peak_flops
    print("MFU: {:.1f}% ({:.2e} FLOPs/s)".format(mfu * 100, achieved_flops))
    return mfu
\`\`\`

---

### Optimization Hierarchy

| Optimization | Memory Saving | Speed Impact | Complexity |
|-------------|--------------|-------------|-----------|
| bf16 mixed precision | 2x | +20-50% | Low |
| torch.compile | 0 | +30-100% | Low |
| Gradient checkpointing | 30-50% | -20-30% | Low |
| FSDP | N_gpu × | Required for >70B | High |
| Activation offloading | ~40% | -10-20% | Medium |

**Rule of thumb:** Always enable bf16 + gradient checkpointing first. Only reach for FSDP when the model doesn't fit in GPU memory.`,
    zh: `## 规模化成本与性能

优化GPU训练需要理解FLOP计数、模型FLOP利用率（MFU）和内存效率权衡。

---

### Transformer层的FLOP计数

每层主要FLOPs：
- QKV投影：3 × seq_len × d_model × d_model × 2
- 注意力分数：seq_len × seq_len × d_model × 2
- FFN：2 × seq_len × d_model × 4d_model × 2

Llama-3-8B（seq=2048）每次前向传播约2.1 TFLOPs。

---

### 模型FLOP利用率（MFU）

测量实际达到的峰值FLOPs百分比。良好训练运行的MFU通常为30-60%。低MFU表示数据加载、内核启动开销或内存瓶颈。

---

### 优化层次结构

始终先启用bf16+梯度检查点。只有在模型不适合GPU内存时才考虑FSDP。`,
  },
}
