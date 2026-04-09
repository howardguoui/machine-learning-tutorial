import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// PEFT Deep Dive + Scaling Laws + Quantization
// ─────────────────────────────────────────────────────────────────────────────

export const peftDeepDive: TopicContent = {
  id: 'peft-deep-dive',
  title: { en: 'PEFT Methods: LoRA, QLoRA & Beyond', zh: 'PEFT方法：LoRA、QLoRA及更多' },
  contentType: 'code',
  content: {
    en: `## PEFT — Parameter-Efficient Fine-Tuning

Full fine-tuning a 7B model needs ~56 GB VRAM. PEFT achieves comparable results training <1% of parameters.

---

### Why PEFT Works: The Low-Rank Hypothesis

**Observation (Aghajanyan et al. 2020):** Pre-trained models have a very low "intrinsic dimensionality" — the effective learning happens in a much smaller subspace than the full parameter space.

**Implication:** You don't need to update all 7 billion parameters to adapt a model. You only need to find the right low-dimensional adjustment.

---

### LoRA — Low-Rank Adaptation (The Standard)

**Core idea:** Instead of updating weight matrix W ∈ ℝ^{d×k} directly, decompose the update ΔW into two small matrices:

\`\`\`
W' = W + ΔW = W + B·A

Where:
  W  ∈ ℝ^{d×k}  (frozen, d=4096, k=4096)
  A  ∈ ℝ^{r×k}  (trained, r=16)
  B  ∈ ℝ^{d×r}  (trained, r=16)

Parameters trained:  r×k + d×r = 16×4096 + 4096×16 = 131,072
Parameters in W:     4096×4096 = 16,777,216
Reduction:           99.2%
\`\`\`

**Initialization:**
- A: random Gaussian (ensures non-zero gradients from the start)
- B: all zeros (so ΔW = B·A = 0 at start — no disruption to pretrained weights)

\`\`\`python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Replace a Linear layer with LoRA-augmented version.
    Only A and B are trained; W is frozen.
    """
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r   # scale factor applied to ΔW

        # Original frozen weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )

        # LoRA matrices — these are trained
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B=0 at init

        # Initialize A with kaiming (same as default Linear init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Original path (frozen) + LoRA path (trained)
        base = x @ self.weight.T
        lora = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

    def merge(self):
        """Merge LoRA weights into base weight for inference (zero overhead)."""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        # After merging, lora_A and lora_B can be deleted

# Hyperparameter guide:
# r (rank):   4=minimal, 8=small, 16=standard, 32=high fidelity, 64=heavy
# alpha:      Usually set to r or 2r; scaling = alpha/r
# target:     Usually ['q_proj', 'v_proj'] or all attention + FFN projections
\`\`\`

---

### QLoRA — Quantized LoRA (4-bit fine-tuning)

QLoRA (Dettmers et al. 2023) enables fine-tuning a 65B model on a single 48GB GPU by:
1. **Loading the base model in 4-bit** (NF4 quantization)
2. **Training LoRA adapters in full precision** (bfloat16)
3. **Double quantization** — quantize the quantization constants themselves

\`\`\`python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

# Step 1: 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — better than int4 for weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # computation in bf16
    bnb_4bit_use_double_quant=True,    # double quantization: saves ~0.4 bits/param
)

# Step 2: Load 4-bit model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 3: Prepare for k-bit training (patches certain layers)
model = prepare_model_for_kbit_training(model)

# Step 4: Add LoRA adapters in full precision
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 159,383,552 || all params: 13,175,865,344 || trainable%: 1.21

# Memory comparison for 13B model:
# Full fine-tune FP16:  ~104 GB
# QLoRA (4-bit + LoRA): ~16 GB  ← fits on 1× A100 40GB
\`\`\`

---

### PEFT Method Comparison

| Method | How It Works | Params Trained | Quality | Speed |
|--------|-------------|----------------|---------|-------|
| **Full SFT** | Update all weights | 100% | Best | Slowest |
| **LoRA** | Low-rank weight update | 0.1–1% | Near-full | Fast |
| **QLoRA** | LoRA + 4-bit base | 0.1–1% | Near-full | Fast, low mem |
| **Prefix Tuning** | Prepend trainable tokens to each layer | <0.1% | Moderate | Fast |
| **Prompt Tuning** | Trainable soft prefix tokens (input only) | <0.01% | Lower | Fastest |
| **Adapter** | Small bottleneck layers between Transformer layers | 0.5–3% | Good | Moderate |

**When to use which:**
- Production, full GPU available → **LoRA** (best quality-to-cost)
- Consumer GPU (24GB) → **QLoRA** (enables large models)
- Research, max efficiency → **Prompt Tuning**
- Multi-task serving (one base, many adapters) → **Adapter** (swap adapters at inference)

---

### Merging Multiple LoRA Adapters

A key advantage of LoRA: you can mathematically combine multiple domain adapters:

\`\`\`python
# Scenario: base model + medical adapter + coding adapter
# Serve one model that handles both domains

def merge_lora_adapters(base_weight, adapter_A1, adapter_B1, alpha1,
                                     adapter_A2, adapter_B2, alpha2):
    """Linear combination of LoRA adapters into base weight."""
    delta_W1 = (adapter_B1 @ adapter_A1) * (alpha1 / adapter_A1.shape[0])
    delta_W2 = (adapter_B2 @ adapter_A2) * (alpha2 / adapter_A2.shape[0])
    return base_weight + delta_W1 + delta_W2

# TIES-Merging and DARE (more advanced merging methods):
# Handle parameter conflicts by trimming & rescaling before merging
\`\`\`
`,
    zh: `## PEFT — 参数高效微调

全参数微调 7B 模型需要约 56 GB 显存。PEFT 训练不到 1% 的参数，却能达到接近的效果。

---

### 为什么 PEFT 有效：低秩假设

**观察（Aghajanyan et al. 2020）：** 预训练模型具有极低的"内在维度" — 有效学习发生在比完整参数空间小得多的子空间中。

**含义：** 不需要更新所有 70 亿个参数来适配模型。只需要找到正确的低维调整。

---

### LoRA — 低秩适应（标准方法）

**核心思想：** 不直接更新权重矩阵 W，而是将更新 ΔW 分解为两个小矩阵：

\`\`\`
W' = W + ΔW = W + B·A

其中：
  W  ∈ ℝ^{d×k}  （冻结，d=k=4096）
  A  ∈ ℝ^{r×k}  （训练，r=16）
  B  ∈ ℝ^{d×r}  （训练，r=16）

训练参数：r×k + d×r = 16×4096 + 4096×16 = 131,072
W 中的参数：4096×4096 = 16,777,216
减少比例：99.2%
\`\`\`

**初始化：**
- A：随机高斯（确保从一开始就有非零梯度）
- B：全零（使 ΔW = B·A = 0 — 不破坏预训练权重）

---

### QLoRA — 量化LoRA（4位微调）

通过以下方式在单张48GB GPU上微调65B模型：
1. **以4位加载基础模型**（NF4量化）
2. **以全精度训练LoRA适配器**（bfloat16）
3. **双重量化** — 连量化常数也量化

**内存比较（13B模型）：**
- 全参数微调 FP16：~104 GB
- QLoRA（4位 + LoRA）：~16 GB ← 单张A100 40GB可运行

---

### PEFT 方法比较

| 方法 | 工作原理 | 训练参数 | 质量 | 速度 |
|------|---------|---------|------|------|
| **全量SFT** | 更新所有权重 | 100% | 最佳 | 最慢 |
| **LoRA** | 低秩权重更新 | 0.1–1% | 接近全量 | 快 |
| **QLoRA** | LoRA + 4位基础模型 | 0.1–1% | 接近全量 | 快，低内存 |
| **Prefix Tuning** | 每层前置可训练token | <0.1% | 中等 | 快 |
| **Prompt Tuning** | 可训练软前缀token | <0.01% | 较低 | 最快 |
| **Adapter** | Transformer层之间的小瓶颈层 | 0.5–3% | 好 | 中等 |

**选择建议：**
- 生产环境，GPU充足 → **LoRA**（质量/成本最优）
- 消费级GPU（24GB）→ **QLoRA**（可运行大模型）
- 多任务服务（一个基础模型，多个适配器）→ **Adapter**
`,
  },
}

export const scalingQuantization: TopicContent = {
  id: 'scaling-quantization',
  title: { en: 'Scaling Laws & Quantization', zh: '规模定律与量化' },
  contentType: 'article',
  content: {
    en: `## Scaling Laws — Predicting LLM Performance

Scaling laws tell us how model performance improves as we scale compute, data, and parameters. Understanding them is crucial for senior ML engineer interviews.

---

### The Original Scaling Laws (Kaplan et al. 2020 — OpenAI)

\`\`\`
L(N) ∝ N^(-0.076)    — loss scales with parameter count
L(D) ∝ D^(-0.095)    — loss scales with dataset size
L(C) ∝ C^(-0.050)    — loss scales with compute budget

Where:
  N = number of non-embedding parameters
  D = number of training tokens
  C = total compute (FLOPs)
  L = cross-entropy loss (lower = better)
\`\`\`

**Key finding:** Model size matters more than dataset size (parameter exponent > data exponent). This led to training very large models on relatively few tokens.

---

### Chinchilla Scaling Law (Hoffmann et al. 2022 — DeepMind)

**Problem with Kaplan's law:** GPT-3 (175B params) was undertrained — it saw only 300B tokens, but optimal would be ~3.5 trillion tokens.

**Chinchilla's finding:** For a given compute budget, you should roughly **double the data** when you **double the model size**:

\`\`\`
Optimal N* = (C / 6)^0.5    ← optimal parameter count
Optimal D* = 6 × C / N*     ← optimal token count

Rule of thumb:
  N* ≈ D*   →  For every parameter, train on ~20 tokens
\`\`\`

| Model | Params | Tokens | Tokens/Param | Status |
|-------|--------|--------|--------------|--------|
| GPT-3 | 175B | 300B | 1.7 | Undertrained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA 2 7B | 7B | 2T | 286 | Overtrained (smaller, cheaper inference) |
| LLaMA 3 8B | 8B | 15T | 1,875 | Very overtrained |

**Why "overtrain"?** Training is expensive; inference happens billions of times. A smaller, well-trained model is cheaper to serve than a larger undertrained model with the same loss.

---

### Emergent Abilities

Some capabilities appear suddenly at certain scale thresholds — they're near zero below the threshold and suddenly non-zero above it.

| Capability | Emerged At Scale |
|------------|-----------------|
| 3-digit arithmetic | ~1B params |
| Chain-of-thought reasoning | ~100B params |
| Multi-step code generation | ~100B params |
| Instruction following (zero-shot) | ~100B params |

**Controversy:** Some researchers argue emergence is an artifact of measurement — using a smooth metric (calibration) instead of a threshold metric makes emergence disappear. The debate continues.

---

### Quantization — Making LLMs Smaller and Faster

**What is quantization?**
Represent model weights in lower precision to reduce memory and speed up computation.

\`\`\`
FP32:  32 bits per param  →  7B model = 28 GB
FP16:  16 bits per param  →  7B model = 14 GB
INT8:   8 bits per param  →  7B model =  7 GB
INT4:   4 bits per param  →  7B model =  3.5 GB
\`\`\`

---

### Post-Training Quantization (PTQ)

Quantize a trained model without retraining. Fast but some accuracy loss.

\`\`\`python
# Method 1: bitsandbytes INT8 (absmax quantization)
from transformers import AutoModelForCausalLM

model_int8 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,   # LLM.int8() by Tim Dettmers
    device_map="auto",
)
# How INT8 works:
# For each weight tensor, find max abs value (absmax)
# Scale all weights to [-127, 127] integer range
# Store as int8; multiply by scale at compute time
# Outlier weights (>6σ) kept in FP16 to prevent degradation

# Method 2: 4-bit NF4 (NormalFloat)
from transformers import BitsAndBytesConfig
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",   # NF4 matches normal distribution of weights
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_4bit,
    device_map="auto",
)
\`\`\`

---

### GPTQ — GPU-Optimized PTQ

More accurate than bitsandbytes, designed for efficient GPU inference:

\`\`\`python
# Requires: pip install auto-gptq optimum
from auto_gptq import AutoGPTQForCausalLM

# Load a pre-quantized GPTQ model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_triton=True,    # fast triton kernels
)
# GPTQ uses second-order (Hessian) information to minimize quantization error
# Layer-by-layer quantization with weight updates to compensate
\`\`\`

---

### AWQ — Activation-Aware Weight Quantization

**Key insight of AWQ:** Not all weights are equally important. Weights that correspond to large activations (salient channels) should be kept at higher precision.

\`\`\`
Standard PTQ:  quantize ALL weights equally  → some important weights degraded
AWQ:           identify salient channels (top 1% activation magnitude)
               scale UP those channels before quantization (×s)
               scale DOWN the corresponding activations (÷s)
               then quantize → important weights have smaller quantization error
\`\`\`

AWQ vs GPTQ comparison:

| | GPTQ | AWQ |
|--|------|-----|
| Method | Layer-wise, Hessian-based | Activation-aware scaling |
| Speed | Slow to quantize | Fast to quantize |
| Inference speed | Fast (Triton kernels) | Fast (custom CUDA) |
| Quality | Slightly better on text | Better on coding/reasoning |
| Memory | Same (~4-bit) | Same (~4-bit) |

---

### Quantization + Quality Trade-offs

| Quantization | Perplexity Increase | Memory Reduction | Inference Speed |
|---|---|---|---|
| FP16 (baseline) | 0% | 2× vs FP32 | 1.5× vs FP32 |
| INT8 (LLM.int8) | +0.1–0.5% | 4× vs FP32 | ~1× (activation overhead) |
| INT4 GPTQ | +0.5–2% | 8× vs FP32 | 2–3× vs FP32 |
| INT4 AWQ | +0.3–1.5% | 8× vs FP32 | 2–3× vs FP32 |
| INT2 | +5–15% | 16× vs FP32 | Mostly unusable |

**Rule of thumb:** INT8 is nearly lossless. INT4 with GPTQ/AWQ has minimal quality loss. INT4 naive quantization degrades significantly.

---

### Decoding Strategies Deep Dive

How the model generates each next token from the probability distribution:

\`\`\`python
import torch
import torch.nn.functional as F

def decode(logits, strategy="top_p", temperature=1.0, top_p=0.9, top_k=50):
    """
    logits: (vocab_size,) — raw model output before softmax

    Strategies:
      greedy:  always pick the highest probability token
      top_k:   sample from top-k most probable tokens
      top_p:   sample from smallest set covering probability mass p
      beam:    maintain k best sequences (not shown here, needs full state)
    """
    # Apply temperature first
    logits = logits / temperature

    if strategy == "greedy":
        return logits.argmax().item()

    if strategy == "top_k":
        # Zero out all but top-k logits
        top_k_vals, top_k_idx = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float('-inf'))
        filtered.scatter_(0, top_k_idx, top_k_vals)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, 1).item()

    if strategy == "top_p":
        # Nucleus sampling: keep tokens until cumulative prob >= p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=0)

        # Remove tokens where cumulative prob > p (keep the one that pushes it over)
        remove = cumprobs - probs > top_p
        sorted_logits[remove] = float('-inf')

        # Restore original order
        filtered = torch.full_like(logits, float('-inf'))
        filtered.scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, 1).item()

# Strategy selection guide:
# Factual tasks (QA, summarization):  temperature=0.3, top_p=0.9
# Creative tasks (story, brainstorm): temperature=0.8-1.0, top_p=0.95
# Code generation:                    temperature=0.2, top_p=0.95
# Beam search (k=4-8):                translation, code — maximize sequence probability
\`\`\`

**Temperature intuition:**
\`\`\`
Original probs:  [cat=0.8, dog=0.1, fish=0.05, bird=0.05]

T=0.5 (sharpen): [cat=0.97, dog=0.02, fish=0.005, bird=0.005]  ← more certain
T=1.0 (identity): [cat=0.8, dog=0.1, ...]
T=2.0 (flatten):  [cat=0.50, dog=0.25, fish=0.13, bird=0.12]   ← more random
\`\`\`
`,
    zh: `## 规模定律 — 预测LLM性能

规模定律告诉我们，随着计算量、数据量和参数量的增加，模型性能如何提升。理解它对于高级ML工程师面试至关重要。

---

### Chinchilla 规模定律（2022年，DeepMind）

**GPT-3 的问题：** 1750亿参数模型只训练了3000亿 token，但最优应该是约3.5万亿 token。

**Chinchilla 的发现：** 给定计算预算，每次模型大小翻倍，数据量也应该翻倍：

\`\`\`
经验法则：每个参数大约需要训练 ~20 个 token 才是计算最优的
\`\`\`

| 模型 | 参数量 | Token量 | Token/参数 | 状态 |
|------|--------|---------|-----------|------|
| GPT-3 | 175B | 300B | 1.7 | 训练不足 |
| Chinchilla | 70B | 1.4T | 20 | 计算最优 |
| LLaMA 2 7B | 7B | 2T | 286 | 过训练（推理更便宜） |
| LLaMA 3 8B | 8B | 15T | 1,875 | 大量过训练 |

**为什么要"过训练"？** 训练成本高但只发生一次；推理发生数十亿次。较小但训练充分的模型，推理成本远低于同等性能的大模型。

---

### 涌现能力（Emergent Abilities）

某些能力在特定规模阈值处突然出现：

| 能力 | 出现规模 |
|------|---------|
| 三位数算术 | ~10亿参数 |
| 思维链推理 | ~1000亿参数 |
| 多步代码生成 | ~1000亿参数 |
| 零样本指令遵循 | ~1000亿参数 |

---

### 量化 — 让LLM更小更快

**精度对比：**
\`\`\`
FP32：每参数32位 → 7B模型 = 28 GB
FP16：每参数16位 → 7B模型 = 14 GB
INT8：每参数8位  → 7B模型 =  7 GB
INT4：每参数4位  → 7B模型 =  3.5 GB
\`\`\`

**量化方法比较：**

| 方法 | 核心思路 | 质量损失 | 适用场景 |
|------|---------|---------|---------|
| **bitsandbytes INT8** | 绝对最大值缩放，离群值保留FP16 | 极小 | 训练/微调 |
| **GPTQ INT4** | 逐层优化，基于Hessian信息 | 小 | 推理 |
| **AWQ INT4** | 激活感知缩放，保护显著通道 | 极小 | 推理（代码/推理任务更好） |
| **NF4（QLoRA）** | 针对权重正态分布优化 | 极小 | 微调 |

**经验法则：** INT8几乎无损。GPTQ/AWQ INT4质量损失极小。朴素INT4量化会显著降质。

---

### 解码策略深度解析

**温度的直觉：**
\`\`\`
原始概率：[cat=0.8, dog=0.1, fish=0.05, bird=0.05]

T=0.5（锐化）：[cat=0.97, dog=0.02, ...]   ← 更确定
T=1.0（不变）：[cat=0.8, dog=0.1, ...]
T=2.0（平坦）：[cat=0.50, dog=0.25, ...]   ← 更随机
\`\`\`

**策略选择指南：**
- 事实类任务（问答、摘要）：temperature=0.3, top_p=0.9
- 创意类任务（故事、头脑风暴）：temperature=0.8-1.0, top_p=0.95
- 代码生成：temperature=0.2, top_p=0.95
- 束搜索（k=4-8）：翻译、代码 — 最大化序列概率
`,
  },
}
