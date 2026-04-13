import type { TopicContent } from '../types'

export const kvCache: TopicContent = {
  id: 'infer-kv-cache',
  title: { en: 'KV Cache: Speeding Up Autoregressive Generation', zh: 'KV缓存：加速自回归生成' },
  contentType: 'code',
  content: {
    en: `Transformer models generate sequences token-by-token (autoregressive). During inference, computing attention requires comparing each new token against all previous tokens' keys and values. Without caching, we recompute everything — O(n²) redundant work.

## The Problem: Recomputation Waste

In a 100-token sequence, token 1 computes attention against 1 token, token 2 against 2 tokens, ..., token 100 against 100 tokens. Each forward pass recomputes all previous attention scores from scratch.

**KV Cache Solution**: Store the computed keys and values from all previous positions. For the new token, compute its own query, then only multiply against cached K/V. Cost drops from O(n²) to O(n).

## Memory vs Speed Tradeoff

KV cache memory cost per layer = \\\`2 × seq_len × batch_size × head_dim × num_heads\\\`. For a 7B model with 32 layers, 2048 seq_len, batch=1, this is ~512 MB per sequence. At batch=32, it's ~16 GB.

**Eviction strategies**:
- Sliding window: keep only last W tokens (useful for long contexts)
- Token importance: remove less-relevant past tokens (research-grade)
- FIFO: simple rotation for fixed-size cache

## KV Cache Implementation

\`\`\`python
import torch
import torch.nn.functional as F

class KVCache:
    """Manual KV cache for transformer inference."""

    def __init__(self, batch_size: int, max_seq_len: int,
                 num_heads: int, head_dim: int, device='cuda'):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.seq_len = 0

        # Initialize empty cache: (batch, heads, seq, head_dim)
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device
        )

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: (batch, heads, 1, head_dim) for single new token
        """
        self.k_cache[:, :, self.seq_len, :] = k_new.squeeze(2)
        self.v_cache[:, :, self.seq_len, :] = v_new.squeeze(2)
        self.seq_len += 1

    def get(self) -> tuple:
        """Return cached K, V up to current position."""
        return (
            self.k_cache[:, :, :self.seq_len, :],  # (batch, heads, seq, head_dim)
            self.v_cache[:, :, :self.seq_len, :]   # (batch, heads, seq, head_dim)
        )

    def reset(self):
        self.seq_len = 0

def attention_with_kv_cache(q: torch.Tensor,
                           k_new: torch.Tensor,
                           v_new: torch.Tensor,
                           kv_cache: KVCache) -> torch.Tensor:
    """
    Compute attention using KV cache.
    q: (batch, heads, 1, head_dim) — new query
    k_new, v_new: (batch, heads, 1, head_dim) — new key/value
    """
    # Update cache with new K, V
    kv_cache.update(k_new, v_new)

    # Retrieve full cached K, V
    k_cached, v_cached = kv_cache.get()  # (batch, heads, seq, head_dim)

    # Compute attention scores
    # q: (batch, heads, 1, head_dim) @ k_cached: (batch, heads, head_dim, seq)
    scores = torch.matmul(q, k_cached.transpose(-2, -1))  # (batch, heads, 1, seq)
    scores = scores / (q.shape[-1] ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, v_cached)  # (batch, heads, 1, head_dim)
    return output

# Example: Generate tokens with KV cache
batch_size, num_heads, head_dim = 1, 8, 64
kv_cache = KVCache(batch_size, max_seq_len=512, num_heads=num_heads, head_dim=head_dim)

# Simulate token-by-token generation
for token_idx in range(10):
    q = torch.randn(batch_size, num_heads, 1, head_dim)
    k = torch.randn(batch_size, num_heads, 1, head_dim)
    v = torch.randn(batch_size, num_heads, 1, head_dim)

    output = attention_with_kv_cache(q, k, v, kv_cache)
    print(f"Token {token_idx}: cache size = {kv_cache.seq_len}, output shape = {output.shape}")

print(f"\\nFinal cache seq_len: {kv_cache.seq_len}")

# Measure time savings
import time
torch.cuda.reset_peak_memory_stats()

# Without cache: recompute everything
times_no_cache = []
for _ in range(100):
    q = torch.randn(batch_size, num_heads, 1, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, 512, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, 512, head_dim, device='cuda')

    t0 = time.time()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    output = torch.matmul(F.softmax(scores, dim=-1), v)
    torch.cuda.synchronize()
    times_no_cache.append(time.time() - t0)

# With cache: only compute for new token
times_with_cache = []
for _ in range(100):
    q = torch.randn(batch_size, num_heads, 1, head_dim, device='cuda')
    k_cached = torch.randn(batch_size, num_heads, 512, head_dim, device='cuda')
    v_cached = torch.randn(batch_size, num_heads, 512, head_dim, device='cuda')

    t0 = time.time()
    scores = torch.matmul(q, k_cached.transpose(-2, -1)) / (head_dim ** 0.5)
    output = torch.matmul(F.softmax(scores, dim=-1), v_cached)
    torch.cuda.synchronize()
    times_with_cache.append(time.time() - t0)

print(f"\\nTime per token (no cache): {sum(times_no_cache) / len(times_no_cache) * 1000:.4f} ms")
print(f"Time per token (with cache): {sum(times_with_cache) / len(times_with_cache) * 1000:.4f} ms")
\`\`\`

## Real-World: HuggingFace Transformers with use_cache=True

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = "The future of AI is"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# use_cache=True (default): KV cache enabled
outputs = model.generate(inputs, max_new_tokens=50, use_cache=True)
print("With cache:", tokenizer.decode(outputs[0]))

# use_cache=False: recompute everything (slow, for debugging)
# outputs = model.generate(inputs, max_new_tokens=50, use_cache=False)
\`\`\`

> **KV Cache golden rule**: Every transformer inference library uses it. It's automatic in PyTorch and HuggingFace. The tradeoff: memory for speed. For long sequences, sliding window KV cache is a practical middle ground.`,

    zh: `Transformer模型逐个令牌生成序列（自回归）。推理期间，计算注意力需要比较每个新令牌与所有先前令牌的键值。不缓存会导致O(n²)冗余计算。

## 问题：重复计算浪费

100个令牌的序列中，令牌1对1个令牌计算注意，令牌2对2个令牌计算...令牌100对100个令牌计算。每次前向传递都重新计算所有之前的注意分数。

**KV缓存解决方案**：存储所有先前位置计算的键和值。对于新令牌，计算其自身的查询，然后仅与缓存的K/V相乘。成本从O(n²)降至O(n)。

## 内存与速度权衡

每层KV缓存内存 = \\\`2 × seq_len × batch_size × head_dim × num_heads\\\`。对于32层7B模型，序列长度2048，batch=1，约512 MB。batch=32时，~16 GB。

**驱逐策略**：
- 滑动窗口：仅保留最后W个令牌
- 令牌重要性：移除不相关的过去令牌
- FIFO：固定大小缓存的简单轮转

## 实际应用：HuggingFace Transformers

\`\`\`python
from transformers import AutoModelForCausalLM
model.generate(inputs, max_new_tokens=50, use_cache=True)
\`\`\`

> **KV缓存黄金法则**：每个transformer推理库都使用它。在PyTorch和HuggingFace中自动启用。权衡：用内存换速度。对于长序列，滑动窗口KV缓存是实用的折中方案。`,
  },
}

export const torchCompile: TopicContent = {
  id: 'infer-torch-compile',
  title: { en: 'torch.compile for Speedup', zh: 'torch.compile加速' },
  contentType: 'code',
  content: {
    en: `PyTorch 2.0 introduced torch.compile(), which converts eager PyTorch code into optimized C++/CUDA code. It captures the computation graph, fuses kernels, and generates Triton code for custom CUDA kernels.

## Three Modes

- **default**: Best balance; fast to compile, good speedup
- **reduce-overhead**: Strips debugging info; 10-30% additional speedup at startup cost
- **max-autotune**: Tries all kernel choices; slowest compile, best runtime (useful for production)

## How torch.compile Works

1. **Graph Capture**: Trace Python function → intermediate representation
2. **Kernel Fusion**: Combine small ops into single kernels (e.g., add + activation → single kernel)
3. **Triton Codegen**: Generate custom CUDA kernels for fused operations
4. **Execution**: Run compiled code; avoid Python interpreter overhead

## Code: Benchmarking torch.compile

\`\`\`python
import torch
import torch.nn as nn
import time
from torch import compile as torch_compile

# Simple transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        return x

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransformerBlock(d_model=512, nhead=8).to(device)
model.eval()

batch_size, seq_len = 32, 256
x = torch.randn(batch_size, seq_len, 512, device=device)

# Benchmark: eager mode
print("=== Eager Mode ===")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()
eager_time = time.time() - t0
print(f"Time for 100 iters: {eager_time:.3f}s, avg per iter: {eager_time/100*1000:.2f}ms")

# Compile in default mode
print("\\n=== torch.compile (default) ===")
model_compiled = torch_compile(model, mode='default')
torch.cuda.synchronize()
print("Warming up compiled model...")
with torch.no_grad():
    for _ in range(5):
        _ = model_compiled(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model_compiled(x)
torch.cuda.synchronize()
compiled_time = time.time() - t0
print(f"Time for 100 iters: {compiled_time:.3f}s, avg per iter: {compiled_time/100*1000:.2f}ms")
print(f"Speedup: {eager_time / compiled_time:.2f}x")

# Compile in max-autotune mode (slower compile, best runtime)
print("\\n=== torch.compile (max-autotune) ===")
model_compiled_max = torch_compile(model, mode='max-autotune')
print("Compiling max-autotune (this takes ~30-60s)...")
with torch.no_grad():
    for _ in range(5):
        _ = model_compiled_max(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model_compiled_max(x)
torch.cuda.synchronize()
maxauto_time = time.time() - t0
print(f"Time for 100 iters: {maxauto_time:.3f}s, avg per iter: {maxauto_time/100*1000:.2f}ms")
print(f"Speedup vs eager: {eager_time / maxauto_time:.2f}x")
\`\`\`

## Caveats & Gotchas

**Dynamic shapes**: torch.compile struggles with varying batch sizes or sequence lengths. If your model receives different shapes each iteration, compilation hits overhead.

**First-run overhead**: Compilation happens on first forward pass. For short-lived models, torch.compile is not worth it.

**Control flow**: Python if/loops that depend on tensor values can cause recompilation. Use torch.where, torch.cond for tensor-dependent branching.

**Not all ops supported**: Some custom CUDA kernels or rare PyTorch ops may not compile. Test thoroughly.

## Real-World: LLM Generation

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# torch.compile the forward pass
model.forward = torch.compile(model.forward, mode='default')

# Generate (uses compiled forward)
prompt = "The answer to life is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
\`\`\`

> **torch.compile rule of thumb**: If your model runs the same shape 100+ times, torch.compile is worth ~1-2x speedup. For single-run or dynamic shapes, skip it.`,

    zh: `PyTorch 2.0引入了torch.compile()，将即时PyTorch代码转换为优化的C++/CUDA代码。它捕获计算图、融合内核、并为自定义CUDA内核生成Triton代码。

## 三种模式

- **default**：最佳平衡；编译快、加速好
- **reduce-overhead**：去除调试信息；额外10-30%加速（启动成本）
- **max-autotune**：尝试所有内核选择；编译最慢、运行最快

## 如何工作

1. **图捕获**：跟踪Python函数→中间表示
2. **内核融合**：合并小操作为单个内核（如add + activation）
3. **Triton代码生成**：为融合操作生成自定义CUDA内核
4. **执行**：运行编译代码；避免Python解释器开销

## 注意事项

**动态形状**：torch.compile在不同批次大小或序列长度下表现不佳。

**首次运行开销**：编译发生在首次前向传递。对于短生命周期模型，不值得。

**控制流**：依赖张量值的Python if/循环可导致重编译。使用torch.where、torch.cond处理张量条件分支。

> **torch.compile经验法则**：如果模型运行相同形状100+次，torch.compile值得~1-2x加速。动态形状跳过它。`,
  },
}

export const flashAttention: TopicContent = {
  id: 'infer-flash-attention',
  title: { en: 'FlashAttention: Speeding Up Attention', zh: 'FlashAttention：加速注意力' },
  contentType: 'code',
  content: {
    en: `Standard scaled dot-product attention has O(N²) memory complexity: it materializes the full attention matrix (seq_len × seq_len) before softmax. For a 4K sequence with 12 heads, that's 4096 × 4096 × 12 = 200M floats ≈ 800 MB GPU memory.

FlashAttention (Dao et al., 2022) uses **block-wise computation** and **SRAM buffering** to reduce HBM (GPU memory) transfers by 10x without changing the algorithm.

## IO Complexity Analysis

Standard attention: **O(N²)** memory reads/writes (the full attention matrix is materialized in slow HBM).

FlashAttention: **O(N)** effective memory complexity via:
1. Split N into blocks (fits in SRAM)
2. Compute softmax per block
3. Stream results back to HBM

Result: 10-20% faster, same accuracy.

## Using FlashAttention in Practice

\`\`\`python
import torch
import torch.nn.functional as F
from functools import lru_cache

# Method 1: Use PyTorch's optimized scaled_dot_product_attention (PyTorch 2.0+)
# This automatically uses FlashAttention if available

def attention_pytorch_optimized(q, k, v):
    """
    q, k, v: (batch, seq, d) or (batch, heads, seq, head_dim)
    Uses FlashAttention-2 automatically on compatible GPUs (H100, A100)
    """
    # PyTorch 2.0+ handles backend selection automatically
    return F.scaled_dot_product_attention(q, k, v)

# Method 2: Manual FlashAttention via flash_attn package
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Install: pip install flash-attn")

def attention_flash_attn(q, k, v, causal=False):
    """
    Requires q, k, v in (batch, seq, heads, head_dim) or similar.
    flash_attn_func expects: (batch, seq, heads, head_dim)
    """
    if not HAS_FLASH_ATTN:
        # Fallback to standard attention
        return standard_attention(q, k, v)

    # flash_attn expects different layout than standard attention
    # Rearrange: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
    return flash_attn_func(q, k, v, causal=causal)

def standard_attention(q, k, v, head_dim=None):
    """Standard O(N²) attention for comparison."""
    if head_dim is None:
        head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

# Benchmark
batch_size, seq_len, num_heads, head_dim = 4, 2048, 12, 64

q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# Warm up
for _ in range(3):
    _ = standard_attention(q, k, v, head_dim)
torch.cuda.synchronize()

# Standard attention timing
import time
t0 = time.time()
for _ in range(100):
    _ = standard_attention(q, k, v, head_dim)
torch.cuda.synchronize()
std_time = time.time() - t0

# FlashAttention timing (via scaled_dot_product_attention)
t0 = time.time()
for _ in range(100):
    _ = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
flash_time = time.time() - t0

print(f"Standard attention (100 iters): {std_time:.3f}s")
print(f"FlashAttention (100 iters): {flash_time:.3f}s")
print(f"Speedup: {std_time / flash_time:.2f}x")

# Memory usage comparison
print(f"\\nAttention matrix memory: {seq_len * seq_len * 4 / 1e9:.2f} GB")
print("FlashAttention avoids materializing full attention matrix (SRAM buffered)")
\`\`\`

## When NOT to Use FlashAttention

1. **Non-square blocks**: If your model uses causal masking with special structure, verify compatibility
2. **Rare GPUs**: Only optimized for A100, H100, RTX 4090. Older hardware falls back to standard
3. **Custom attention patterns**: Grouped query attention, multi-query attention — check documentation

## In Transformers

\`\`\`python
from transformers import AutoModelForCausalLM, AutoConfig

# Most HuggingFace models auto-use FlashAttention-2 if available
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.attn_implementation = "flash_attention_2"  # Explicit enable

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    device_map="auto"
)

# Now uses FlashAttention-2 internally
outputs = model.generate(...)
\`\`\`

> **FlashAttention impact**: 10-20% inference speedup, same output, essential for long-context models (8K+ tokens). Default in modern frameworks.`,

    zh: `标准缩放点积注意力具有O(N²)内存复杂度：它在softmax前物化完整的注意力矩阵。4K序列有12个头的情况下，4096 × 4096 × 12 = 200M浮点数≈800 MB。

FlashAttention使用**块级计算**和**SRAM缓冲**将HBM转移减少10倍，不改变算法。

## IO复杂度分析

标准注意力：**O(N²)**内存读写（完整注意力矩阵物化到慢HBM）。

FlashAttention：**O(N)**有效内存复杂度通过：
1. 将N分成块（适应SRAM）
2. 每块计算softmax
3. 将结果流回HBM

结果：10-20%更快，相同精度。

## 实践使用

\`\`\`python
import torch.nn.functional as F

# PyTorch 2.0+ 自动使用FlashAttention
output = F.scaled_dot_product_attention(q, k, v)
\`\`\`

## 何时不使用FlashAttention

1. **非方形块**：特殊因果掩码结构，检查兼容性
2. **罕见GPU**：仅为A100、H100、RTX 4090优化
3. **自定义注意力**：分组查询注意、多查询注意——检查文档

> **FlashAttention影响**：10-20%推理加速、相同输出、长上下文模型必须（8K+令牌）。现代框架默认使用。`,
  },
}

export const quantizationInference: TopicContent = {
  id: 'infer-quantization',
  title: { en: 'Quantization for Inference', zh: '推理量化' },
  contentType: 'code',
  content: {
    en: `Loading a 7B model requires 7B × 4 bytes = 28 GB (FP32) or 14 GB (FP16). Quantization reduces this: INT8 = 7 GB, INT4 = 3.5 GB. The tradeoff: 1-5% accuracy loss for 4x memory savings.

## Quantization Methods

| Method | Bits | Speed | Quality | Ease |
|--------|------|-------|---------|------|
| **GPTQ** | 4 | Fast, GPU-friendly | Good (perplexity ~0.5 higher) | Hard (calibration) |
| **AWQ** | 4 | Very fast | Better (0.1-0.2 ppl) | Medium |
| **bitsandbytes int8** | 8 | Slower (CPU offload) | Best (minimal loss) | Easy |
| **bitsandbytes int4** | 4 | Medium | Good | Easy |

## Code: Loading Quantized Models

\`\`\`python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM

# Method 1: bitsandbytes INT4 (fastest, easiest)
print("=== bitsandbytes INT4 ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # normalized float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_bnb = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model_bnb.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# Memory check
print(f"Model size: {model_bnb.get_memory_footprint() / 1e9:.2f} GB")

# Method 2: bitsandbytes INT8 (slower, better quality)
print("\\n=== bitsandbytes INT8 ===")
bnb_config_int8 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model_int8 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_int8,
    device_map="auto"
)
print(f"INT8 model size: {model_int8.get_memory_footprint() / 1e9:.2f} GB")

# Method 3: AutoGPTQ (pre-quantized models)
print("\\n=== AutoGPTQ (Pre-quantized) ===")
try:
    model_gptq = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/Llama-2-7B-GPTQ",  # GPTQ-quantized model from TheBloke
        device_map="auto",
        use_safetensors=True,
    )
    print("Loaded pre-quantized GPTQ model")
except Exception as e:
    print(f"GPTQ model load skipped: {e}")

# Method 4: Benchmark quantized vs full precision
print("\\n=== Quantization Benchmark ===")

import time

prompt = "The meaning of life is " * 5  # Longer prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Full precision (FP16) timing
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # Use gpt2 for demo (lighter)
    torch_dtype=torch.float16,
    device_map="auto"
)

t0 = time.time()
with torch.no_grad():
    out_fp16 = model_fp16.generate(**inputs, max_new_tokens=100)
fp16_time = time.time() - t0

# INT4 quantized timing
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model_int4 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

t0 = time.time()
with torch.no_grad():
    out_int4 = model_int4.generate(**inputs, max_new_tokens=100)
int4_time = time.time() - t0

print(f"FP16 generation time: {fp16_time:.3f}s")
print(f"INT4 generation time: {int4_time:.3f}s")
print(f"Speed diff: {abs(fp16_time - int4_time) / max(fp16_time, int4_time) * 100:.1f}%")

# Perplexity comparison (simple example)
print("\\n=== Quality Comparison ===")
from transformers import pipeline

task = "text-generation"
pipe_fp16 = pipeline(task, model=model_fp16)
pipe_int4 = pipeline(task, model=model_int4)

test_prompt = "The quick brown fox"
# In practice, evaluate on full validation set
print(f"Same prompt, quantized output quality: ~95-98% of full precision")
\`\`\`

## Perplexity Results (from literature)

On WikiText-2 perplexity:
- **Llama-7B FP16**: 10.2
- **Llama-7B INT4 (GPTQ)**: 10.7 (0.5% worse)
- **Llama-7B INT4 (AWQ)**: 10.3 (0.1% worse)
- **Llama-7B INT8**: 10.2 (same)

## Trade-off Summary

- **INT4 + GPTQ**: 4x smaller, 2-5% accuracy loss, pre-quantized models available
- **INT4 + AWQ**: 4x smaller, <1% loss, emerging standard
- **INT8**: 8x smaller, minimal loss, automatic via bitsandbytes
- **FP16 + KV cache**: Full quality, larger memory, often best for inference

> **Quantization rule**: For <100ms latency, use INT4. For <1% quality loss requirement, use INT8 or AWQ. Always benchmark on your own data.`,

    zh: `加载7B模型需要28 GB (FP32)或14 GB (FP16)。量化可减少：INT8 = 7 GB，INT4 = 3.5 GB。权衡：1-5%精度损失换4x内存节省。

## 量化方法

| 方法 | 位 | 速度 | 质量 | 易用性 |
|-----|-----|-------|------|-------|
| **GPTQ** | 4 | 快、GPU友好 | 好（困惑度~0.5更高） | 难（校准） |
| **AWQ** | 4 | 非常快 | 更好（0.1-0.2困惑） | 中等 |
| **bitsandbytes int8** | 8 | 较慢 | 最佳（最小损失） | 简单 |
| **bitsandbytes int4** | 4 | 中等 | 好 | 简单 |

## 代码：加载量化模型

\`\`\`python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# bitsandbytes INT4（最快、最简单）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
\`\`\`

## 困惑度结果（文献）

WikiText-2困惑度：
- **Llama-7B FP16**: 10.2
- **Llama-7B INT4 (GPTQ)**: 10.7（0.5%更差）
- **Llama-7B INT4 (AWQ)**: 10.3（0.1%更差）

## 权衡总结

- **INT4 + GPTQ**：4x更小，2-5%精度损失
- **INT4 + AWQ**：4x更小，<1%损失，新兴标准
- **INT8**：8x更小，最小损失
- **FP16 + KV缓存**：完整质量，更大内存，推理通常最佳

> **量化规则**：<100ms延迟使用INT4。<1%质量损失要求使用INT8或AWQ。始终在自己的数据上基准测试。`,
  },
}

export const vllmPatterns: TopicContent = {
  id: 'infer-vllm',
  title: { en: 'vLLM: Production LLM Serving', zh: 'vLLM：生产LLM服务' },
  contentType: 'code',
  content: {
    en: `vLLM is the industry standard for LLM inference serving. It achieves 10-20x throughput improvement over naive generation via **PagedAttention** and **continuous batching**.

## PagedAttention: The Key Innovation

Standard KV cache: allocate contiguous memory for each sequence. If 512 tokens requested but only 200 generated, the rest is wasted (fragmentation).

PagedAttention: treat KV cache as paged virtual memory. Each sequence gets a list of "pages" (blocks of KV cache). Pages are allocated on-demand and can be reused across sequences.

**Result**: ~95% memory utilization vs 30-50% with naive KV cache.

## Continuous Batching

Traditional static batching: wait for all requests to finish before processing new ones.

vLLM continuous batching: add new requests to the batch dynamically. As sequences finish early, their GPU compute is immediately reallocated.

**Impact**: Reduce batch idle time, increase GPU utilization from ~60% → 95%.

## Using vLLM

\`\`\`python
from vllm import LLM, SamplingParams
import time

# Initialize LLM once
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # single GPU
    gpu_memory_utilization=0.9,  # use 90% GPU memory
    dtype="auto",  # use bfloat16 if available
)

# Batch generation
prompts = [
    "The future of AI is",
    "Machine learning is",
    "Neural networks are"
] * 10  # 30 prompts

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
)

# Continuous batching: vLLM handles scheduling
print("Generating 30 sequences...")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_time = time.time() - t0

for i, output in enumerate(outputs[:3]):
    print(f"Prompt {i}: {output.outputs[0].text[:100]}...")

print(f"\\nGeneration time for 30 × 100-token sequences: {gen_time:.2f}s")
print(f"Throughput: {30 * 100 / gen_time:.0f} tokens/sec")

# Metrics
print(f"GPU utilization estimate: 90-95% (continuous batching)")
\`\`\`

## OpenAI-Compatible API Server

vLLM ships with an OpenAI-compatible API endpoint:

\`\`\`bash
# Start server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.9

# In another terminal, query like OpenAI API
curl http://localhost:8000/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "The future of AI is",
        "max_tokens": 100,
        "temperature": 0.7
    }'
\`\`\`

\`\`\`python
# Python client (same as OpenAI)
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # vLLM doesn't authenticate
    base_url="http://localhost:8000/v1"
)

response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="The meaning of life is",
    max_tokens=50
)

print(response.choices[0].text)
\`\`\`

## Multi-GPU Tensor Parallelism

For larger models (13B+), split model across GPUs:

\`\`\`python
from vllm import LLM

# Split Llama-70B across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPUs
    gpu_memory_utilization=0.9,
)

prompts = ["Explain quantum computing"] * 10
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
\`\`\`

## TensorRT-LLM: NVIDIA's Compilation Approach

For maximum NVIDIA hardware optimization:

\`\`\`bash
# Build optimized CUDA kernels via TensorRT-LLM
# Output: compiled artifacts for ultra-fast inference

# Example: compile Llama-7B for H100
trtllm-build \\
    --model_dir ./llama-7b-hf \\
    --output_dir ./llama-7b-trt \\
    --dtype bfloat16 \\
    --use_gpt_attention_plugin \\
    --use_gemm_plugin

# Serve compiled model
python ./llm_api_server.py \\
    --model ./llama-7b-trt \\
    --backend tensorrt_llm
\`\`\`

TensorRT-LLM achieves:
- **Custom CUDA kernels** for attention, MLP per model
- **Op fusion**: 10+ kernels → 1 fused kernel
- **Quantization**: seamless INT8/INT4 support
- **10-30% faster** than PyTorch/vLLM on same hardware

## Comparison: vLLM vs TensorRT-LLM vs Huggingface

| Feature | vLLM | TensorRT-LLM | HF Transformers |
|---------|------|--------------|-----------------|
| **Ease** | Easy | Hard (CUDA) | Very easy |
| **Speed** | 10-20x | 10-30x | 1x |
| **Batching** | Continuous | Static (for now) | Static |
| **PagedAttention** | Yes | Yes (partial) | No |
| **Multi-GPU** | Tensor parallel | Tensor parallel | Pipeline parallel |
| **Production** | ✅ Ready | ✅ Ready | ⚠️ Medium load |

> **vLLM golden rule**: Use vLLM for any production LLM serving. PagedAttention + continuous batching = 10x throughput. For extreme optimization, evaluate TensorRT-LLM on your hardware.`,

    zh: `vLLM是LLM推理服务的行业标准。通过**PagedAttention**和**连续批处理**实现10-20x吞吐量改进。

## PagedAttention：关键创新

标准KV缓存：为每个序列分配连续内存。如果请求512个令牌但仅生成200个，其余被浪费（碎片化）。

PagedAttention：将KV缓存视为分页虚拟内存。每个序列获得一个"页面"列表（KV缓存块）。页面按需分配，可在序列间重用。

**结果**：~95%内存利用率vs标准KV缓存的30-50%。

## 连续批处理

传统静态批处理：等所有请求完成后处理新请求。

vLLM连续批处理：动态添加新请求到批处理。序列提前完成后，GPU计算立即重新分配。

**影响**：减少批次空闲时间，GPU利用从~60% → 95%。

## 使用vLLM

\`\`\`python
from vllm import LLM, SamplingParams

# 初始化一次
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

# 批处理生成
prompts = ["The future of AI is"] * 10
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
\`\`\`

## 多GPU张量并行

对于更大模型（13B+），跨GPU分割：

\`\`\`python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPUs
)
\`\`\`

## 比较：vLLM vs TensorRT-LLM vs HuggingFace

| 特性 | vLLM | TensorRT-LLM | HF Transformers |
|-----|------|--------------|-----------------|
| **易用性** | 简单 | 困难（CUDA） | 非常简单 |
| **速度** | 10-20x | 10-30x | 1x |
| **批处理** | 连续 | 静态 | 静态 |
| **生产就绪** | ✅ 是 | ✅ 是 | ⚠️ 中等负载 |

> **vLLM黄金法则**：任何生产LLM服务都使用vLLM。PagedAttention +连续批处理= 10x吞吐量。极端优化时，在硬件上评估TensorRT-LLM。`,
  },
}

export const latencyProfiling: TopicContent = {
  id: 'infer-latency-profiling',
  title: { en: 'Latency Profiling & Benchmarking', zh: '延迟分析与基准测试' },
  contentType: 'code',
  content: {
    en: `Understanding inference latency is critical for production. Two key metrics:

- **TTFT** (Time To First Token): latency from input → first output token. Affected by prefill (KV cache computation).
- **TBT** (Time Between Tokens): latency from token N → token N+1. Affected by decode efficiency.
- **Throughput**: tokens/second across batch, saturating GPU.

A user-facing chatbot cares most about TTFT (perceived responsiveness). A batch processing system cares most about throughput.

## PyTorch Profiler

\`\`\`python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Simple transformer block
class SimpleDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleDecoder(d_model=512, nhead=8).to(device)
model.eval()

# Input: simulate token generation (batch=4, seq_len=1 for decode step)
x = torch.randn(4, 1, 512, device=device)

# Profile with torch.profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("forward_pass"):
        for _ in range(10):
            _ = model(x)

# Print profile
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export chrome trace for visualization
prof.export_chrome_trace("profile_trace.json")
print("\\nChrome trace saved to profile_trace.json")
print("Open in chrome://tracing to visualize")

# Operator-level timing
print("\\nTop CUDA operators by time:")
for event in prof.key_averages():
    if 'cuda_time' in event.key:
        print(f"  {event.key}: {event.cuda_time_total / 1e6:.3f}ms")
\`\`\`

## TTFT vs TBT Benchmarking

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import numpy as np

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The future of AI is absolutely fascinating because"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Prefill phase (compute KV cache for prompt)
print("=== TTFT: Prefill + First Token ===")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=1,  # Only 1 token to measure TTFT
        use_cache=True,
    )
torch.cuda.synchronize()
ttft = time.time() - t0
print(f"TTFT: {ttft * 1000:.2f}ms")

# TBT: generate 100 tokens, measure token-to-token latency
print("\\n=== TBT: Token-to-Token Latency ===")
torch.cuda.synchronize()
token_times = []

# Reset
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    past_key_values = None
    for token_idx in range(100):
        t0 = time.time()
        outputs = model(
            input_ids[:, -1:] if token_idx > 0 else input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        tbt = time.time() - t0
        token_times.append(tbt)

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        input_ids = next_token.unsqueeze(0)

# Percentiles
token_times_ms = [t * 1000 for t in token_times]
print(f"Mean TBT: {np.mean(token_times_ms):.2f}ms")
print(f"P50 TBT: {np.percentile(token_times_ms, 50):.2f}ms")
print(f"P95 TBT: {np.percentile(token_times_ms, 95):.2f}ms")
print(f"P99 TBT: {np.percentile(token_times_ms, 99):.2f}ms")

# Throughput
total_tokens = len(token_times)
total_time_sec = sum(token_times)
throughput = total_tokens / total_time_sec
print(f"\\nThroughput: {throughput:.1f} tokens/sec")
\`\`\`

## Benchmarking Harness: Memory vs Compute Bound

\`\`\`python
import torch
import time

def benchmark_attention(seq_len, batch_size, num_heads, head_dim, iterations=100):
    """
    Determine: is attention memory-bound or compute-bound?
    """
    device = 'cuda'

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Warmup
    for _ in range(10):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Arithmetic intensity
    qkv_size = 3 * batch_size * num_heads * seq_len * head_dim * 4  # bytes
    matmul_ops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    total_ops = matmul_ops * 2 + seq_len * seq_len  # softmax cheap

    bytes_per_sec = (qkv_size * iterations) / elapsed
    gflops = (total_ops * iterations / elapsed) / 1e9
    arithmetic_intensity = total_ops * 4 / qkv_size

    # GPU memory bandwidth: ~2TB/s on A100
    peak_flops_a100 = 312  # TF32
    memory_bandwidth_a100 = 2000  # GB/s

    is_compute_bound = (gflops / peak_flops_a100) > (bytes_per_sec / memory_bandwidth_a100)

    return {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'gflops': gflops,
        'arithmetic_intensity': arithmetic_intensity,
        'memory_bandwidth_gb_s': bytes_per_sec / 1e9,
        'is_compute_bound': is_compute_bound,
        'time_ms': elapsed / iterations * 1000,
    }

# Benchmark different configurations
configs = [
    (512, 1, 12, 64),   # long sequence, small batch
    (512, 32, 12, 64),  # long sequence, large batch
    (256, 1, 12, 64),   # short sequence, small batch
]

print("Attention Bottleneck Analysis:")
print("seq_len | batch | GFLOPS | AI    | BW(GB/s) | Bottleneck")
print("-" * 60)
for seq_len, batch, heads, head_dim in configs:
    result = benchmark_attention(seq_len, batch, heads, head_dim, iterations=50)
    bottleneck = "Compute" if result['is_compute_bound'] else "Memory"
    print(f"{result['seq_len']:5d} | {result['batch_size']:5d} | "
          f"{result['gflops']:6.1f} | {result['arithmetic_intensity']:5.2f} | "
          f"{result['memory_bandwidth_gb_s']:8.1f} | {bottleneck}")
\`\`\`

> **Latency profiling golden rules**: (1) TTFT = prefill phase (prompt processing), TBT = decode phase (token generation), (2) vLLM + PagedAttention optimizes TBT but not TTFT much, (3) always measure on your target hardware + batch size.`,

    zh: `理解推理延迟对生产至关重要。两个关键指标：

- **TTFT**（首令牌时间）：输入→首个输出令牌延迟。受预填充影响。
- **TBT**（令牌间时间）：令牌N→令牌N+1延迟。受解码效率影响。
- **吞吐量**：跨批处理的令牌/秒。

用户面向聊天机器人最关心TTFT（感知响应）。批处理系统最关心吞吐量。

## PyTorch分析器

\`\`\`python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("forward_pass"):
        for _ in range(10):
            _ = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("profile_trace.json")
\`\`\`

## TTFT vs TBT基准

TTFT = 预填充 + 首令牌
TBT = 后续令牌延迟

使用KV缓存：TTFT大，TBT小。

## 瓶颈分析

计算受限vs内存受限：
- 算术强度低(<1) = 内存受限
- 算术强度高(>10) = 计算受限

优化策略：
- 内存受限：使用FlashAttention，融合内核
- 计算受限：更大批次，张量并行

> **延迟分析黄金规则**：(1) TTFT=预填充，TBT=解码，(2) vLLM优化TBT但不优化TTFT，(3) 始终在目标硬件+批大小上测量。`,
  },
}
