import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// Transformer Architecture — the engine behind every LLM
// ─────────────────────────────────────────────────────────────────────────────

export const transformerArchitecture: TopicContent = {
  id: 'transformer-architecture',
  title: { en: 'Transformer Architecture Deep Dive', zh: 'Transformer架构深度解析' },
  contentType: 'article',
  content: {
    en: `## Transformer Architecture — The Engine Behind Every LLM

> Every modern LLM (GPT, LLaMA, Claude, Gemini) is a Transformer. Understanding this architecture cold is table stakes for any LLM interview.

---

### The Big Picture

\`\`\`
Input text: "The cat sat"
     ↓
[Tokenization]  →  [3, 47, 892]
     ↓
[Token Embeddings]  →  3 vectors of dim 512
     ↓
[+ Positional Encoding]
     ↓
[N × Transformer Blocks]
  ┌──────────────────────────┐
  │  Multi-Head Self-Attention│
  │  + Add & LayerNorm        │
  │  Feed-Forward Network     │
  │  + Add & LayerNorm        │
  └──────────────────────────┘
     ↓
[Linear + Softmax]  →  probability over vocabulary
     ↓
Next token prediction: "on"
\`\`\`

---

### Component 1: Self-Attention (The Core)

Self-attention lets every token look at every other token and decide how much to "attend" to it.

**Three learned projections per head:**
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What do I share if attended to?"

**Formula:**
\`\`\`
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
\`\`\`

**Why divide by √d_k?**
Without scaling, the dot products grow large as dimension increases → softmax saturates → gradients vanish. Dividing by √d_k keeps the variance stable.

\`\`\`python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    """
    d_k = Q.size(-1)

    # Step 1: dot product between queries and keys
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, heads, seq_len, seq_len)

    # Step 2: causal mask (decoder: token i can only see tokens 0..i)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: softmax to get attention weights
    weights = F.softmax(scores, dim=-1)

    # Step 4: weighted sum of values
    output = torch.matmul(weights, V)
    return output, weights

# Example: 1 sentence, 2 heads, 5 tokens, d_k=64
Q = torch.randn(1, 2, 5, 64)
K = torch.randn(1, 2, 5, 64)
V = torch.randn(1, 2, 5, 64)
out, attn = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")    # (1, 2, 5, 64)
print(f"Attn weights: {attn.shape}")   # (1, 2, 5, 5) — 5×5 attention matrix
\`\`\`

---

### Component 2: Multi-Head Attention (MHA)

Instead of one attention function, use **h parallel attention heads**, each with its own Q/K/V projections. Then concatenate all heads.

**Why multiple heads?**
Each head can specialize: one might focus on syntactic relations, another on coreference, another on semantic similarity.

\`\`\`python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads   # 64 per head
        self.num_heads = num_heads

        # One combined projection for efficiency
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection

    def split_heads(self, x):
        B, T, D = x.shape
        # Reshape: (B, T, D) → (B, num_heads, T, d_k)
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (B, heads, T, d_k) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
\`\`\`

---

### Component 3: Positional Encoding

Self-attention is **permutation-invariant** — it treats "cat sat the" the same as "the cat sat". Positional encoding adds position information.

**Sinusoidal PE (original Transformer):**
\`\`\`
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

**RoPE (Rotary Position Embedding) — used in LLaMA, GPT-NeoX:**
- Encodes position by *rotating* Q and K vectors
- Relative distances are preserved: attention score between pos i and j depends only on (i - j)
- Allows context length extension at inference time

**ALiBi (Attention with Linear Biases) — used in MPT, Falcon:**
- Adds a negative linear bias to attention scores based on distance
- m × |i - j| subtracted from score, where m is a per-head slope
- Generalizes to longer sequences without re-training

| Method | Used In | Context Extension |
|--------|---------|------------------|
| Sinusoidal | Original Transformer | Hard limit at training length |
| Learned PE | GPT-2, BERT | Hard limit at training length |
| RoPE | LLaMA 1/2/3, Mistral | Yes (NTK scaling) |
| ALiBi | MPT, Falcon | Yes (extrapolates naturally) |

---

### Component 4: Feed-Forward Network (FFN)

After attention, each token passes through a 2-layer MLP independently:

\`\`\`
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
\`\`\`

- Expands to 4× the model dim (d_model → 4·d_model → d_model)
- Uses ReLU or **SwiGLU** (LLaMA) or **GeLU** (GPT)
- This is where ~2/3 of model parameters live
- Each token processed *independently* — no cross-token interaction here

\`\`\`python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# SwiGLU variant (LLaMA-style):
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: swish(gate(x)) * up(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))
\`\`\`

---

### Component 5: Layer Normalization

**Pre-LN vs Post-LN — critical for training stability:**

\`\`\`
Post-LN (original): x → Sublayer(x) → Add → LayerNorm   (harder to train)
Pre-LN  (modern):   x → LayerNorm → Sublayer(x) → Add   (stable gradients)
\`\`\`

Pre-LN is used in GPT-2+, LLaMA — gradients flow more smoothly through the residual connection.

**RMSNorm** (LLaMA, Mistral) — simpler than LayerNorm, no mean subtraction:
\`\`\`
RMSNorm(x) = x / RMS(x) × γ
RMS(x) = sqrt(mean(x²))
\`\`\`

---

### Architecture Comparison: GPT vs BERT vs T5

| | GPT (decoder-only) | BERT (encoder-only) | T5 (encoder-decoder) |
|--|--|--|--|
| **Attention** | Causal (masked) | Bidirectional | Encoder: bi; Decoder: causal |
| **Pre-training** | Next token prediction | Masked LM + NSP | Span prediction |
| **Best for** | Generation | Classification, embedding | Seq2seq (translation) |
| **Examples** | GPT-4, LLaMA, Claude | BERT, RoBERTa | T5, BART, mT5 |
| **Context** | Single pass | Full sequence | Input → Output |

**Interview key point:** GPT uses **causal (autoregressive) attention** — each token can only see previous tokens. This is implemented with a triangular mask in the attention scores.

---

### Full Transformer Block (GPT-style)

\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff   = FeedForward(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN attention with residual connection
        x = x + self.drop(self.attn(self.ln1(x), mask))
        # Pre-LN FFN with residual connection
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq_len=1024):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        # Build causal mask: lower triangular
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

        x = self.embed(idx) + self.pos_emb(torch.arange(T))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab_size)
        return logits
\`\`\`

---

### Common Interview Questions

**Q: What is the time and space complexity of self-attention?**
- Time: O(n²·d) — quadratic in sequence length n
- Space: O(n²) for the attention matrix
- This is the scalability bottleneck for long context

**Q: What is KV cache and why does it matter for inference?**
- At inference, token i's K and V don't change when generating token i+1
- KV cache stores previously computed K and V — avoids recomputation
- Memory cost: 2 × num_layers × seq_len × d_model × batch_size × bytes_per_param

**Q: What is FlashAttention?**
- Reorders the attention computation to minimize HBM (slow) reads/writes
- Uses tiling: computes attention in blocks that fit in SRAM (fast)
- Same mathematical result, 2-4× faster, O(n) memory instead of O(n²)
`,
    zh: `## Transformer架构深度解析

> 每个现代 LLM（GPT、LLaMA、Claude、Gemini）都是 Transformer。冷静地理解这个架构是参加任何 LLM 面试的基本功。

---

### 整体结构

\`\`\`
输入文本："The cat sat"
     ↓
[分词]  →  [3, 47, 892]
     ↓
[词嵌入]  →  3个512维向量
     ↓
[+ 位置编码]
     ↓
[N × Transformer Block]
  ┌──────────────────────────┐
  │  多头自注意力机制         │
  │  + 残差连接 & 层归一化    │
  │  前馈神经网络             │
  │  + 残差连接 & 层归一化    │
  └──────────────────────────┘
     ↓
[线性层 + Softmax]  →  词表上的概率分布
     ↓
预测下一个词："on"
\`\`\`

---

### 组件1：自注意力机制（核心）

自注意力让每个 token 都能"看到"其他所有 token，并决定关注哪些。

**三种可学习映射：**
- **Q（查询）：** "我在寻找什么？"
- **K（键）：** "我包含什么？"
- **V（值）：** "被关注时我分享什么？"

**公式：**
\`\`\`
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
\`\`\`

**为什么除以 √d_k？**
不缩放的话，随着维度增大点积会变得很大 → softmax饱和 → 梯度消失。除以 √d_k 使方差保持稳定。

---

### 组件2：多头注意力

使用 h 个并行注意力头，每个头有独立的 Q/K/V 映射，最后将所有头的输出拼接。

**为什么用多个头？**
每个头可以专注于不同方面：一个可能关注句法关系，另一个关注共指，另一个关注语义相似性。

---

### 组件3：位置编码

自注意力是**排列不变的** — "猫坐垫" 和 "垫坐猫" 会产生同样的注意力分数。位置编码添加位置信息。

| 方法 | 使用模型 | 是否可以扩展上下文 |
|------|---------|-----------------|
| 正弦PE | 原始Transformer | 有硬限制 |
| 可学习PE | GPT-2, BERT | 有硬限制 |
| RoPE | LLaMA 1/2/3, Mistral | 可以（NTK缩放） |
| ALiBi | MPT, Falcon | 可以（自然外推） |

---

### 组件4：前馈网络（FFN）

注意力之后，每个 token 独立通过一个2层 MLP：
\`\`\`
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
\`\`\`
- 扩展到4倍模型维度（d_model → 4·d_model → d_model）
- 约2/3的模型参数在这里
- **没有跨token交互** — 每个token独立处理

---

### 组件5：层归一化

**Pre-LN vs Post-LN：**
\`\`\`
Post-LN（原始）：x → 子层(x) → Add → LayerNorm   （训练较难）
Pre-LN （现代）：x → LayerNorm → 子层(x) → Add   （梯度稳定）
\`\`\`

Pre-LN 用于 GPT-2+、LLaMA — 梯度通过残差连接流动更顺畅。

---

### 架构比较：GPT vs BERT vs T5

| | GPT（仅解码器） | BERT（仅编码器） | T5（编码-解码器） |
|--|--|--|--|
| **注意力** | 因果（有掩码） | 双向 | 编码器:双向; 解码器:因果 |
| **预训练** | 下一个词预测 | 掩码语言模型+NSP | Span预测 |
| **最适合** | 生成 | 分类、嵌入 | 序列到序列（翻译） |
| **代表模型** | GPT-4, LLaMA, Claude | BERT, RoBERTa | T5, BART, mT5 |

---

### 面试常见问题

**Q：自注意力的时间和空间复杂度是多少？**
- 时间：O(n²·d) — 序列长度 n 的二次方
- 空间：O(n²) 用于注意力矩阵
- 这是长上下文可扩展性的瓶颈

**Q：什么是 KV 缓存？为什么它对推理很重要？**
- 在推理时，生成第 i+1 个 token 时，token i 的 K 和 V 不会变化
- KV 缓存存储之前计算的 K 和 V，避免重复计算
- 内存代价：2 × 层数 × 序列长度 × 模型维度 × 批次大小 × 每参数字节数

**Q：什么是 FlashAttention？**
- 重排注意力计算，最小化慢速HBM（显存）的读写次数
- 使用分块：在能放入SRAM（快速片上内存）的块中计算注意力
- 数学结果相同，速度快2-4倍，内存从 O(n²) 降为 O(n)
`,
  },
}

export const tokenizationPromptEng: TopicContent = {
  id: 'tokenization-prompt-engineering',
  title: { en: 'Tokenization & Prompt Engineering', zh: '分词与提示工程' },
  contentType: 'code',
  content: {
    en: `## Tokenization — How Text Becomes Numbers

Every LLM interview touches tokenization. Understanding it explains why models fail on arithmetic, spacing, and rare words.

---

### What Is a Token?

A token is a chunk of text — could be a word, part of a word, or punctuation.

\`\`\`python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, world! ChatGPT is amazing."
tokens = tokenizer.encode(text)
decoded = [tokenizer.decode([t]) for t in tokens]

print(f"Token IDs: {tokens}")
# [15496, 11, 995, 0, 28208, 38, 11571, 318, 4998, 13]

print(f"Token strings: {decoded}")
# ['Hello', ',', ' world', '!', ' Chat', 'G', 'PT', ' is', ' amazing', '.']
# Note: "ChatGPT" splits into ['Chat', 'G', 'PT']
\`\`\`

---

### BPE (Byte-Pair Encoding) — The Standard Algorithm

Used by: GPT-2, GPT-4, LLaMA, Mistral, Claude

**How it works:**
1. Start with character-level vocabulary
2. Count all adjacent byte pairs in training corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size (e.g., 50,000 tokens)

\`\`\`python
# Toy BPE example
corpus = ["low low low low lower lower newest newest widest"]

# Initial: character-level with end-of-word marker
# l o w </w>  l o w e r </w>  n e w e s t </w>  w i d e s t </w>

# Iteration 1: most frequent pair = ('e', 's') → merge → 'es'
# Iteration 2: most frequent pair = ('es', 't') → merge → 'est'
# Iteration 3: most frequent pair = ('l', 'o') → merge → 'lo'
# Iteration 4: ('lo', 'w') → 'low'
# ...continues until vocab_size reached

# Result: 'newest' → ['new', 'est'], 'lower' → ['low', 'er']
\`\`\`

**Why BPE is good:**
- Handles unknown words by falling back to smaller subwords
- Never produces unknown tokens (worst case: byte-level)
- Efficient: common words = 1 token; rare words = multiple tokens

---

### Vocabulary Size Trade-offs

| Vocab Size | Tokens per sentence | Sequence length | Model size |
|------------|-------------------|-----------------|------------|
| Small (10K) | More (long sequences) | Expensive | Smaller embedding |
| Large (100K) | Fewer (short sequences) | Cheap | Larger embedding |

**Real vocab sizes:**
- GPT-2: 50,257 tokens
- LLaMA 2: 32,000 tokens
- LLaMA 3: 128,256 tokens (huge — better multilingual)
- Claude (via tiktoken): ~100K tokens

---

### Why Models Fail at Arithmetic — A Tokenization Issue

\`\`\`python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# "99 + 1" tokenizes to:
print(tokenizer.tokenize("99 + 1"))      # ['99', ' +', ' 1']

# But "9999 + 1" tokenizes to:
print(tokenizer.tokenize("9999 + 1"))    # ['9999', ' +', ' 1']
# GPT-2 vocab has 4-digit numbers as single tokens!

# However "9,999 + 1":
print(tokenizer.tokenize("9,999 + 1"))  # ['9', ',', '999', ' +', ' 1']
# Now "9,999" is split — model sees different "number" than "9999"

# The model learns statistical patterns over tokens,
# not actual digit arithmetic. That's why chain-of-thought helps:
# forcing step-by-step forces the model to process digits individually.
\`\`\`

---

### Special Tokens

Every model has reserved tokens with special meaning:

\`\`\`python
# GPT-2
print(tokenizer.special_tokens_map)
# {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>'}

# LLaMA 2 (instruction format)
# <s>         = BOS (beginning of sequence)
# </s>        = EOS (end of sequence)
# [INST]      = start of user instruction
# [/INST]     = end of user instruction

# Chat format example:
llama_prompt = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2 + 2? [/INST]"""

# These special tokens teach the model WHEN to respond (after [/INST])
# and WHEN to stop (at </s>)
\`\`\`

---

## Prompt Engineering Patterns

### Pattern 1: Zero-Shot

\`\`\`python
prompt = """Classify the sentiment of this review as Positive, Negative, or Neutral.

Review: "The battery lasts all day and the screen is sharp."

Sentiment:"""
# Model output: "Positive"
\`\`\`

### Pattern 2: Few-Shot (In-Context Learning)

\`\`\`python
prompt = """Classify sentiment as Positive, Negative, or Neutral.

Review: "Terrible product, broke after one day."
Sentiment: Negative

Review: "Works fine, nothing special."
Sentiment: Neutral

Review: "Absolutely love it! Best purchase of the year."
Sentiment: Positive

Review: "The battery lasts all day and the screen is sharp."
Sentiment:"""
# Model output: "Positive"
# Few-shot works by showing the pattern; model completes it.
\`\`\`

### Pattern 3: Chain-of-Thought (CoT)

Forces the model to reason step-by-step before answering:

\`\`\`python
# Without CoT — often wrong on multi-step problems
bad_prompt = "Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many balls does he have?"
# GPT might answer: "11" (correct) or "8" (wrong, forgetting to count original)

# With CoT — much more reliable
cot_prompt = """Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many tennis balls does he have now?

Let's think step by step:"""
# Model is forced to output:
# "Roger starts with 5 balls.
#  He buys 2 cans × 3 balls = 6 more balls.
#  Total: 5 + 6 = 11 balls."

# The reasoning forces the model to produce intermediate results as tokens,
# which it can then use as "scratchpad" context.
\`\`\`

### Pattern 4: Zero-Shot CoT

\`\`\`python
# Just add "Let's think step by step" to any prompt
prompt = f"{question}\\n\\nLet's think step by step:"
# This single phrase dramatically improves multi-step reasoning
# because it primes autoregressive generation to produce reasoning tokens
\`\`\`

### Pattern 5: System Prompt + Role Assignment

\`\`\`python
messages = [
    {
        "role": "system",
        "content": "You are an expert Python developer. "
                   "Always include type hints. Never use global variables. "
                   "Respond only with code and brief comments."
    },
    {
        "role": "user",
        "content": "Write a function to find the nth Fibonacci number."
    }
]
# System prompt constrains style and behavior across the entire conversation
\`\`\`

### Pattern 6: Structured Output (JSON Mode)

\`\`\`python
prompt = """Extract the key information from this job posting and return it as JSON.

Job posting: "Senior Python Engineer at TechCorp.
5+ years required. Salary: $150k-$180k. Remote OK."

Return JSON with keys: title, company, years_experience, salary_range, remote.
JSON:"""

# Expected output:
# {
#   "title": "Senior Python Engineer",
#   "company": "TechCorp",
#   "years_experience": 5,
#   "salary_range": "$150k-$180k",
#   "remote": true
# }

# Many APIs now support json_mode=True which guarantees valid JSON output
\`\`\`

---

### Prompt Engineering Best Practices

| Practice | Why |
|----------|-----|
| **Be specific and explicit** | LLMs follow instructions literally — ambiguity leads to unexpected outputs |
| **Put important instructions at the start and end** | "Lost in the middle" — models attend more to beginning and end of long prompts |
| **Use delimiters** (\`\`\`, ---) to separate sections | Reduces confusion between instruction and content |
| **Specify output format** | Avoids need for post-processing |
| **Use CoT for multi-step reasoning** | Externalizes computation as tokens |
| **Few-shot > zero-shot for domain-specific** | Provides concrete format examples |
| **Iterate and test** | Prompts are fragile — small changes can change output significantly |
`,
    zh: `## 分词 — 文本如何变成数字

每次 LLM 面试都会涉及分词。理解它能解释为什么模型在算术、空格处理和罕见词上会失败。

---

### 什么是 Token？

Token 是一段文本块 — 可以是一个单词、单词的一部分或标点符号。

\`\`\`python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "ChatGPT is amazing."
tokens = tokenizer.tokenize(text)
print(tokens)
# ['Chat', 'G', 'PT', 'Ġis', 'Ġamazing', '.']
# "ChatGPT" 被拆成 ['Chat', 'G', 'PT']
\`\`\`

---

### BPE（字节对编码）— 标准算法

使用此算法的模型：GPT-2、GPT-4、LLaMA、Mistral、Claude

**工作原理：**
1. 从字符级词表开始
2. 统计训练语料中所有相邻字节对的频次
3. 合并最频繁的对，生成新 token
4. 重复直到词表达到目标大小（如 50,000 个 token）

**词表大小权衡：**
- 小词表（10K）→ 每个句子 token 更多 → 序列更长 → 计算更贵
- 大词表（100K）→ 每个句子 token 更少 → 序列更短 → 嵌入矩阵更大

**真实词表大小：**
- GPT-2：50,257 tokens
- LLaMA 2：32,000 tokens
- LLaMA 3：128,256 tokens（大很多 — 多语言更好）

---

### 为什么模型算术会失败 — 分词问题

模型学习的是 token 上的统计模式，而不是真正的数字运算。这就是为什么思维链（CoT）有效：强制逐步推理迫使模型逐个处理数字 token。

---

### 特殊 Token

每个模型都有具有特殊含义的保留 token：
- \`<s>\` / \`</s>\` = 序列开始/结束
- \`[INST]\` / \`[/INST]\` = LLaMA 2 指令边界
- \`<|system|>\` = 系统提示开始

---

## 提示工程模式

### 模式1：零样本（Zero-Shot）
直接给出任务描述，不提供示例。

### 模式2：少样本（Few-Shot）
提供2-5个示例来展示格式，模型通过上下文学习模式。

### 模式3：思维链（Chain-of-Thought）

\`\`\`python
# 加一句 "让我们一步一步思考" 显著提升多步推理能力
prompt = f"{user_question}\\n\\n让我们一步一步思考："
# 这迫使模型将中间推理步骤输出为 token，作为"草稿纸"使用
\`\`\`

### 模式4：结构化输出

\`\`\`python
prompt = """从以下文本中提取信息并以JSON返回。

文本："张三，软件工程师，工资15万，支持远程办公。"

返回JSON，包含字段：name, title, salary, remote。
JSON:"""
\`\`\`

### 提示工程最佳实践

| 实践 | 原因 |
|------|------|
| **明确具体** | LLM 字面遵循指令，模糊会导致意外输出 |
| **重要指令放开头和结尾** | "中间遗失"现象 — 模型更关注长提示的首尾 |
| **用分隔符**（\`\`\`、---）分隔不同部分 | 减少指令和内容之间的混淆 |
| **指定输出格式** | 避免后处理 |
| **多步推理用CoT** | 将计算外化为 token |
| **领域特定任务用少样本** | 提供具体格式示例 |
`,
  },
}
