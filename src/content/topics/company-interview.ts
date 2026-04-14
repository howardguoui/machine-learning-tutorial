import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// Company-Specific AI Interview Questions: OpenAI · Google · Bloomberg
// Sources: 1point3acres, InterviewQuery, Glassdoor, Huru, Prepfully, community
// ─────────────────────────────────────────────────────────────────────────────

export const openaiTechnical: TopicContent = {
  id: 'company-openai-technical',
  title: { en: 'OpenAI — Technical Interview Questions', zh: 'OpenAI技术面试题' },
  contentType: 'code',
  content: {
    en: `## OpenAI — Technical Interview Questions

OpenAI interviews focus on **real engineering problems**: transformer debugging, scalable ML systems, and novel data pipelines — not pure LeetCode. Expect PyTorch code on the spot.

---

## Transformer / Deep Learning Debugging
*(Confirmed from 1point3acres phone screen reports)*

### Q1. Debug this broken PyTorch Transformer

You are handed a GPT-style decoder-only model. Find and fix all bugs:

\`\`\`python
import torch
import torch.nn as nn
import math

class BrokenAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # BUG 1: positional embedding applied inside attention (wrong place)
        pos = torch.arange(T).unsqueeze(0)
        x = x + pos  # shape mismatch: pos is (1, T), x is (B, T, C)

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head)

        Q = Q.transpose(1, 2)  # (B, H, T, d_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # BUG 2: missing scale factor
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # BUG 3: causal mask applied AFTER softmax (too late)
        attn = torch.softmax(scores, dim=-1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn = attn.masked_fill(mask, 0)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # BUG 4: output projection missing
        return out
\`\`\`

**Fixes required:**
1. Remove positional embedding from inside attention — it belongs in the embedding layer
2. Add scale: \`scores = scores / math.sqrt(self.d_head)\`
3. Apply causal mask BEFORE softmax: \`scores.masked_fill(mask, float('-inf'))\` then softmax
4. Apply \`self.W_o(out)\` before returning

---

### Q2. Implement KV Cache from scratch

\`\`\`python
class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # KV cache: grows each decode step
        self.cache_k = None  # (B, H, T_past, d_head)
        self.cache_v = None

    def forward(self, x, use_cache=True):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if use_cache and self.cache_k is not None:
            # Concatenate new K, V with cached past
            K = torch.cat([self.cache_k, K], dim=2)
            V = torch.cat([self.cache_v, V], dim=2)

        if use_cache:
            self.cache_k = K.detach()
            self.cache_v = V.detach()

        scale = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
\`\`\`

---

### Q3. Backpropagation through noisy/toxic data

You have a training batch where 20% of labels are **wrong** and 5% of inputs contain **adversarial/toxic** patterns. How does backprop behave? What do you change?

**Expected answer:**
- Noisy labels: gradient on wrong examples points in wrong direction; model memorizes noise late in training
- Fixes: label smoothing, early stopping, loss-based sample filtering (small-loss trick — clean samples have lower loss)
- Toxic inputs: adversarial gradients can be large, destabilizing training
- Fixes: gradient clipping, input sanitization, per-sample gradient norms (GradNorm), separate toxic detector

\`\`\`python
def small_loss_filter(loss_per_sample: torch.Tensor, keep_ratio: float = 0.8):
    """Keep only the lowest-loss samples (likely clean labels)."""
    k = int(len(loss_per_sample) * keep_ratio)
    threshold = torch.topk(loss_per_sample, k, largest=False).values[-1]
    return loss_per_sample <= threshold

# In training loop:
per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
mask = small_loss_filter(per_sample_loss, keep_ratio=0.8)
loss = per_sample_loss[mask].mean()
loss.backward()
\`\`\`

---

## Coding Problems *(from 1point3acres OpenAI problem list)*

### Q4. GPU Credit Tracker

Design a system tracking GPU credit allocation across concurrent jobs:

\`\`\`python
import threading
from collections import defaultdict

class GPUCreditTracker:
    def __init__(self, total_credits: float):
        self.total = total_credits
        self.available = total_credits
        self.jobs = {}          # job_id -> credits_allocated
        self.lock = threading.Lock()

    def allocate(self, job_id: str, credits: float) -> bool:
        with self.lock:
            if credits > self.available:
                return False
            self.available -= credits
            self.jobs[job_id] = self.jobs.get(job_id, 0) + credits
            return True

    def release(self, job_id: str):
        with self.lock:
            if job_id in self.jobs:
                self.available += self.jobs.pop(job_id)

    def consume(self, job_id: str, amount: float) -> bool:
        with self.lock:
            if self.jobs.get(job_id, 0) < amount:
                return False  # overage
            self.jobs[job_id] -= amount
            return True

    def balance(self, job_id: str) -> float:
        return self.jobs.get(job_id, 0.0)
\`\`\`

---

### Q5. OpenSheet — Spreadsheet with Cell Dependencies

\`\`\`python
class Spreadsheet:
    def __init__(self):
        self.values = {}       # cell -> raw value or formula string
        self.deps = {}         # cell -> set of cells it depends on
        self.rev_deps = {}     # cell -> set of cells that depend on it

    def set(self, cell: str, value):
        # Remove old reverse dep edges
        for dep in self.deps.get(cell, set()):
            self.rev_deps[dep].discard(cell)

        if isinstance(value, str) and value.startswith('='):
            refs = self._parse_refs(value)
            if self._has_cycle(cell, refs):
                raise ValueError("Circular dependency detected")
            self.deps[cell] = refs
            for dep in refs:
                self.rev_deps.setdefault(dep, set()).add(cell)
        else:
            self.deps[cell] = set()

        self.values[cell] = value
        self._recalculate(cell)

    def get(self, cell: str) -> float:
        val = self.values.get(cell, 0)
        if isinstance(val, str) and val.startswith('='):
            return self._eval(val)
        return float(val)

    def _parse_refs(self, formula: str):
        import re
        return set(re.findall(r'[A-Z][0-9]+', formula[1:]))

    def _has_cycle(self, cell: str, new_deps: set) -> bool:
        visited = set()
        def dfs(node):
            if node == cell:
                return True
            if node in visited:
                return False
            visited.add(node)
            return any(dfs(d) for d in self.deps.get(node, set()))
        return any(dfs(d) for d in new_deps)

    def _recalculate(self, start: str):
        # Topological recalculation of all downstream cells
        order = []
        visited = set()
        def topo(node):
            if node in visited:
                return
            visited.add(node)
            for dep in self.rev_deps.get(node, set()):
                topo(dep)
            order.append(node)
        topo(start)
        for cell in reversed(order):
            _ = self.get(cell)  # trigger recalculation

    def _eval(self, formula: str) -> float:
        expr = formula[1:]
        for ref in self._parse_refs(formula):
            expr = expr.replace(ref, str(self.get(ref)))
        return eval(expr)  # noqa: S307 — safe in sandbox
\`\`\`

---

### Q6. Mining Novel Data from Unlabeled Corpus

\`\`\`python
from datasketch import MinHash, MinHashLSH
import math

def mine_novel_data(
    documents: list[str],
    model,          # LM that returns perplexity per doc
    tokenizer,
    dedup_threshold: float = 0.8,
    quality_threshold: float = 50.0,   # max perplexity to keep
    max_docs: int = 10_000,
) -> list[str]:
    """
    Pipeline:
    1. Deduplicate with MinHash LSH
    2. Score quality via model perplexity
    3. Return top-k novel, high-quality docs
    """
    # Step 1: Deduplicate
    lsh = MinHashLSH(threshold=dedup_threshold, num_perm=128)
    unique = []
    for i, doc in enumerate(documents):
        m = MinHash(num_perm=128)
        for word in doc.lower().split():
            m.update(word.encode())
        if not lsh.query(m):
            lsh.insert(str(i), m)
            unique.append(doc)

    # Step 2: Score perplexity
    scored = []
    for doc in unique:
        inputs = tokenizer(doc, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs['input_ids']).loss
        ppl = math.exp(loss.item())
        if ppl <= quality_threshold:
            scored.append((ppl, doc))

    # Step 3: Return most informative (moderate perplexity = novel but learnable)
    scored.sort(key=lambda x: x[0])
    return [doc for _, doc in scored[:max_docs]]
\`\`\`

---

## System Design

### Q7. Design Sora-style Video Generation Scheduler

**Requirements:** Schedule diffusion inference jobs across a GPU cluster; handle job priorities, preemption, partial caching.

**Key components:**
- **Job queue**: priority queue (interactive > batch > background)
- **Resource manager**: tracks free GPU memory per node; packs small jobs together
- **Partial result cache**: for repeated prompts, cache U-Net activations at fixed denoising step
- **Preemption**: checkpoint denoising state to NVMe; resume from step k, not step 0
- **SLA monitoring**: p99 latency target, job aging to prevent starvation

\`\`\`python
from dataclasses import dataclass, field
from queue import PriorityQueue
import time

@dataclass(order=True)
class VideoJob:
    priority: int
    submit_time: float = field(compare=False)
    job_id: str = field(compare=False)
    prompt: str = field(compare=False)
    steps: int = field(compare=False, default=50)
    checkpoint_step: int = field(compare=False, default=0)

class VideoScheduler:
    def __init__(self, gpu_count: int, vram_per_gpu_gb: float):
        self.queue = PriorityQueue()
        self.gpus = {i: vram_per_gpu_gb for i in range(gpu_count)}
        self.running = {}   # job_id -> gpu_id

    def submit(self, job: VideoJob):
        self.queue.put(job)

    def schedule_next(self) -> tuple:
        if self.queue.empty():
            return None, None
        job = self.queue.get()
        # Find GPU with enough VRAM
        for gpu_id, vram in self.gpus.items():
            if vram >= 16.0:  # 16GB needed per job
                self.gpus[gpu_id] -= 16.0
                self.running[job.job_id] = gpu_id
                return job, gpu_id
        # No GPU available — re-queue
        self.queue.put(job)
        return None, None

    def complete(self, job_id: str):
        if job_id in self.running:
            gpu_id = self.running.pop(job_id)
            self.gpus[gpu_id] += 16.0
\`\`\``,

    zh: `## OpenAI技术面试题

OpenAI面试专注于**真实工程问题**：Transformer调试、可扩展ML系统和新型数据流水线——不是纯LeetCode。预计需要当场编写PyTorch代码。

---

## Transformer调试题

### Q1. 调试损坏的PyTorch Transformer

给你一个GPT风格的解码器模型，找出并修复所有bug：
1. 位置嵌入应在嵌入层而非注意力内部应用
2. 缺少缩放因子：scores = scores / sqrt(d_head)
3. 因果掩码必须在softmax之前应用（用-inf填充）
4. 缺少输出投影层W_o

---

### Q2. 从零实现KV缓存

KV缓存避免每次解码步骤重新计算所有历史token的K和V。缓存以(B, H, T_past, d_head)格式存储，每步追加新的K和V。

---

### Q3. 噪声/有毒数据中的反向传播

噪声标签：梯度在错误样本上指向错误方向，模型后期会记忆噪声。
修复：标签平滑、早停、小损失过滤（干净样本损失较低）。

有毒输入：对抗性梯度可能很大，破坏训练稳定性。
修复：梯度裁剪、输入清洗、每样本梯度范数监控。

---

## 编程题

### Q4. GPU积分追踪器
线程安全的GPU积分分配系统，支持并发作业的分配、消耗和释放。

### Q5. 带单元格依赖的电子表格
实现支持公式引用的电子表格，检测循环依赖，拓扑排序重新计算下游单元格。

### Q6. 从未标注语料库中挖掘新颖数据
流水线：MinHash去重 → 困惑度评分 → 返回高质量、非冗余样本。

---

## 系统设计

### Q7. Sora风格视频生成调度器
优先级队列调度扩散推理作业，支持抢占（检查点到NVMe）、部分结果缓存、SLA监控。`,
  },
}

export const googleTechnical: TopicContent = {
  id: 'company-google-technical',
  title: { en: 'Google — Technical Interview Questions', zh: 'Google技术面试题' },
  contentType: 'code',
  content: {
    en: `## Google — Technical Interview Questions

Google MLE interviews (L4/L5) run 4–5 rounds: 2 coding (DSA), 1 ML deep-dive, 1 ML system design, 1 behavioral. Python/C++/Java accepted. **Talk through your reasoning the entire time.**

---

## Coding / DSA *(actual reported questions)*

### Q1. Unique Work Days per Employee (SQL)

\`\`\`sql
-- Table: shifts(employee_id, start_date, end_date)
-- Find total unique calendar days each employee worked (merge overlapping ranges)

WITH merged AS (
  SELECT
    employee_id,
    start_date,
    end_date,
    MAX(end_date) OVER (
      PARTITION BY employee_id
      ORDER BY start_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_max_end
  FROM shifts
),
groups AS (
  SELECT *,
    SUM(CASE WHEN start_date > prev_max_end OR prev_max_end IS NULL THEN 1 ELSE 0 END)
      OVER (PARTITION BY employee_id ORDER BY start_date) AS grp
  FROM merged
),
intervals AS (
  SELECT employee_id, grp, MIN(start_date) AS gs, MAX(end_date) AS ge
  FROM groups GROUP BY employee_id, grp
)
SELECT employee_id,
       SUM(DATEDIFF(ge, gs) + 1) AS unique_days
FROM intervals
GROUP BY employee_id;
\`\`\`

---

### Q2. N-gram Frequency Dictionary

\`\`\`python
from collections import Counter
from typing import Dict

def ngram_freq(text: str, n: int) -> Dict[str, int]:
    """
    Return all n-grams and their frequency.
    Edge cases: n > len(words) -> return {}, empty string -> return {}
    """
    words = text.lower().split()
    if n <= 0 or n > len(words):
        return {}
    grams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return dict(Counter(grams))

# Example:
# ngram_freq("the cat sat on the mat", 2)
# -> {'the cat': 1, 'cat sat': 1, 'sat on': 1, 'on the': 1, 'the mat': 1}
\`\`\`

---

### Q3. Reservoir Sampling — Random from Stream

\`\`\`python
import random
from typing import Generator

def reservoir_sample(stream: Generator, k: int = 1) -> list:
    """
    Select k items uniformly at random from a stream of unknown length.
    Proof: item i is selected with probability k/i at each step.
    """
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

# Single item selection (k=1) — most commonly asked variant
def random_from_stream(stream: Generator):
    chosen = None
    for i, item in enumerate(stream):
        if random.random() < 1.0 / (i + 1):
            chosen = item
    return chosen
\`\`\`

---

### Q4. Text Justification

\`\`\`python
def justify(words: list[str], width: int) -> list[str]:
    lines, current, cur_len = [], [], 0

    for word in words:
        if cur_len + len(word) + len(current) > width:
            # Distribute spaces across current line
            lines.append(_distribute(current, cur_len, width))
            current, cur_len = [], 0
        current.append(word)
        cur_len += len(word)

    # Last line: left-justified
    lines.append(' '.join(current).ljust(width))
    return lines

def _distribute(words: list[str], word_len: int, width: int) -> str:
    if len(words) == 1:
        return words[0].ljust(width)
    gaps = len(words) - 1
    total_spaces = width - word_len
    space, extra = divmod(total_spaces, gaps)
    result = []
    for i, word in enumerate(words[:-1]):
        result.append(word + ' ' * (space + (1 if i < extra else 0)))
    result.append(words[-1])
    return ''.join(result)
\`\`\`

---

### Q5. Find Missing Integer (O(1) space)

\`\`\`python
def find_missing(X: list[int], Y: list[int]) -> int:
    """Y = X with one element removed. Find the missing one."""
    return sum(X) - sum(Y)
    # XOR variant (handles duplicates better):
    # from functools import reduce; import operator
    # return reduce(operator.xor, X + Y)
\`\`\`

---

## ML Domain *(asked in dedicated ML round)*

### Q6. L1 vs L2 Regularization — Geometric Proof

\`\`\`
L1 constraint: |w1| + |w2| <= C  -> diamond shape
L2 constraint: w1^2 + w2^2 <= C  -> circle shape

Loss contours are ellipses centered at the unconstrained optimum.
The intersection of the ellipse with the diamond hits a CORNER (where one
weight = 0) far more often than the intersection with the circle.
-> L1 produces sparsity; L2 does not.

Lasso (L1): dL/dw = dLoss/dw + lambda * sign(w)
Ridge (L2): dL/dw = dLoss/dw + 2*lambda * w
\`\`\`

---

### Q7. Derive the Gradient of Softmax + Cross-Entropy

\`\`\`
Forward:
  z_i = logit for class i
  p_i = exp(z_i) / sum_j exp(z_j)      <- softmax
  L = -log(p_y)                          <- cross-entropy for true class y

Backward:
  dL/dz_i = p_i - 1(i == y)
           = p_i - y_i    (where y is one-hot)

Proof:
  dL/dz_i = -d/dz_i [log p_y]
           = -1/p_y * dp_y/dz_i

  dp_y/dz_i = p_y*(1 - p_y)  if i == y
            = -p_y * p_i      if i != y

  -> dL/dz_i = p_i - y_i   (clean, elegant result)
\`\`\`

---

### Q8. Explain Vanishing Gradients + Fix

\`\`\`
In a network with L layers, gradient at layer k:
  dL/dW_k = (dL/dh_L) * prod_{i=k+1}^{L} (dh_i/dh_{i-1})

Each term dh_i/dh_{i-1} = W_i * sigma'(z_i)

For sigmoid: sigma'(z) <= 0.25 -> product of L terms -> 0.25^L -> 0 for L > 5
For tanh:    sigma'(z) <= 1.0  -> still vanishes for deep networks

Fixes:
1. ReLU: sigma'(z) = 1 for z > 0 (no saturation in positive half)
2. Residual connections: gradient highway dL/dW_k += dL/dh_k directly
3. Batch normalization: stabilizes activation scale per layer
4. Gradient clipping: prevents explosion (clip to max_norm=1.0)
\`\`\`

---

### Q9. BatchNorm vs LayerNorm — When to Use Each

\`\`\`python
# BatchNorm: normalize over BATCH dimension (per feature)
# Good for: CNNs, large batches
# Bad for: small batches, variable-length sequences, inference with batch=1
nn.BatchNorm1d(d_model)  # normalizes across batch

# LayerNorm: normalize over FEATURE dimension (per sample)
# Good for: Transformers, RNNs, any variable-length input
# Works correctly with batch=1
nn.LayerNorm(d_model)    # normalizes within each sample
\`\`\`

**Interview follow-up:** Why does LayerNorm have learnable gamma/beta?
-> Without them, normalization destroys the network's ability to represent the identity function. gamma and beta allow the network to undo normalization when beneficial.

---

## ML System Design *(reported from L5 onsite)*

### Q10. Design YouTube Shorts Recommendation

\`\`\`
Stage 1 — Candidate Generation (millions -> thousands)
  - Two-tower model: user embedding + video embedding
  - User features: watch history, demographics, device, time-of-day
  - Video features: title/description embeddings, engagement stats, recency
  - ANN retrieval (HNSW / ScaNN) over video embedding index

Stage 2 — Ranking (thousands -> hundreds)
  - Multi-task model: predict P(watch 80%), P(like), P(skip in 3s), P(share)
  - Features: user-video cross features, position bias correction
  - Serving: TensorFlow Serving / Vertex AI

Stage 3 — Re-ranking + Diversity
  - MMR (Maximal Marginal Relevance) for diversity
  - Freshness boost for videos < 24h old
  - Creator diversity constraint: cap same creator at 2 consecutive slots

Key challenge — Long vs short-term objective:
  Short-term: maximize swipe-away avoidance (immediate)
  Long-term: maximize 30-day return visits (delayed reward)
  Solution: multi-horizon reward model + offline RL for long-term policy
\`\`\`

---

### Q11. Design Fraud Detection with Real-Time Alerts

\`\`\`
Ingestion: Kafka topic per transaction type (< 10ms ingestion lag)

Feature Store:
  Online (Redis):  user transaction count last 1h/24h, avg spend, velocity
  Offline (BQ):    historical patterns, device fingerprints, graph features

Model:
  Tier 1: Rule engine (fast, < 1ms) — obvious fraud patterns
  Tier 2: GBDT (LightGBM, < 5ms) — feature-based scoring
  Tier 3: GNN (< 50ms) — fraud ring detection via transaction graph

Alert:
  If score > 0.85: block + SMS within 100ms
  If 0.6 < score < 0.85: soft decline + step-up auth

Feedback loop:
  Confirmed fraud labels -> retrain Tier 2/3 daily
  False positive rate monitored per merchant category
\`\`\`

---

### Q12. Design On-Device Smart Reply (Android)

\`\`\`
Constraints: model <= 10MB, latency < 20ms, no server call, privacy-preserving

Architecture:
  - Teacher: large T5/BART fine-tuned on reply pairs
  - Student: 3-layer transformer, d_model=128, vocab=8K (8-bit quantized)
  - Distillation loss: KL(teacher_logits, student_logits) + CE

Quantization:
  - Post-training int8 quantization via TFLite
  - Weights: 8-bit, activations: 8-bit
  - Size reduction: ~4x (from ~40MB to ~10MB)

On-device personalization:
  - Federated learning: update final layer on-device
  - Differential privacy: Gaussian noise on gradients before aggregation
  - Model never leaves device

Inference pipeline:
  Email text -> tokenize (SentencePiece, 8K vocab) ->
  student model (3 layers, int8) -> top-3 replies -> display
\`\`\``,

    zh: `## Google技术面试题

Google MLE面试（L4/L5）包含4-5轮：2轮编程（DSA）、1轮ML深度问答、1轮ML系统设计、1轮行为面试。Python/C++/Java均可，**全程大声思考**。

---

## 编程题

### Q1. 员工工作唯一天数（SQL）
合并重叠日期区间，计算每位员工工作的唯一日历天数。使用窗口函数检测区间断点，然后聚合。

### Q2. N-gram频率字典
返回文本中所有n-gram及其频率。边界情况：n大于词数返回{}，空字符串返回{}。

### Q3. 水库抽样
从未知长度的流中均匀随机抽取k个元素。第i个元素以k/i的概率被选中。

### Q4. 文本对齐
给定单词列表和行宽W，格式化使每行恰好W字符，空格均匀分布（最后一行左对齐）。

---

## ML深度问答

### Q6. L1 vs L2正则化几何证明
L1约束：菱形；L2约束：圆形。损失等高线与菱形相交更易落在角点（一个权重=0），产生稀疏性。

### Q7. Softmax + 交叉熵梯度推导
结果简洁：dL/dz_i = p_i - y_i（p_i为softmax输出，y_i为真实标签）。

### Q8. 梯度消失/爆炸
深层网络中梯度是各层Jacobian的乘积。sigmoid激活最大导数0.25，L层后趋近0。修复：ReLU、残差连接、批归一化、梯度裁剪。

---

## ML系统设计

### Q10. YouTube Shorts推荐
三阶段：候选生成（双塔+ANN）→ 排序（多任务预测）→ 重排（多样性+新鲜度）。关键挑战：短期跳过信号 vs 长期回访目标，用多时域奖励模型解决。

### Q11. 实时欺诈检测
规则引擎（<1ms）→ GBDT（<5ms）→ GNN（<50ms）三级架构。确认欺诈标签每日重训练。

### Q12. 端侧智能回复（Android）
知识蒸馏（T5→3层Transformer）→ int8量化（10MB内）→ 联邦学习个性化 → 差分隐私保护。`,
  },
}

export const bloombergTechnical: TopicContent = {
  id: 'company-bloomberg-technical',
  title: { en: 'Bloomberg — Technical Interview Questions', zh: 'Bloomberg技术面试题' },
  contentType: 'code',
  content: {
    en: `## Bloomberg — Technical Interview Questions

Bloomberg AI interviews blend **financial domain knowledge + ML engineering**. Expect NLP questions grounded in financial text (earnings calls, filings), time-series modeling, and coding that mirrors real Bloomberg engineering problems.

---

## Coding / DSA

### Q1. Implement HashMap from Scratch

\`\`\`python
class HashMap:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.size = 0
        self.load_factor = 0.75
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key) -> int:
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))
        self.size += 1
        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        idx = self._hash(key)
        self.buckets[idx] = [(k, v) for k, v in self.buckets[idx] if k != key]
        self.size -= 1

    def _resize(self):
        old = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old:
            for k, v in bucket:
                self.put(k, v)
# Worst case O(n) when all keys hash to same bucket.
# Amortized O(1) with good hash + load factor control.
\`\`\`

---

### Q2. Dijkstra's Shortest Path (Financial Instrument Graph)

\`\`\`python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: str) -> dict:
    """
    graph: {node: [(neighbor, weight), ...]}
    Returns: {node: shortest_distance_from_start}
    Application: shortest conversion path between currencies/instruments
    """
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dict(dist)

# Follow-up: detect negative cycles -> use Bellman-Ford
def has_negative_cycle(graph: dict, nodes: list) -> bool:
    dist = {n: float('inf') for n in nodes}
    dist[nodes[0]] = 0
    for _ in range(len(nodes) - 1):
        for u in nodes:
            for v, w in graph.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    # If we can still relax an edge, there's a negative cycle
    for u in nodes:
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                return True
    return False
\`\`\`

---

### Q3. Longest Increasing Subsequence — O(n log n)

\`\`\`python
import bisect

def lis(nums: list[int]) -> int:
    """Patience sorting approach — O(n log n)"""
    piles = []
    for n in nums:
        idx = bisect.bisect_left(piles, n)
        if idx == len(piles):
            piles.append(n)
        else:
            piles[idx] = n
    return len(piles)

def lis_with_sequence(nums: list[int]) -> list[int]:
    """Return the actual LIS, not just the length."""
    n = len(nums)
    if not n:
        return []
    tails, prev, indices = [], [-1] * n, []

    for i, num in enumerate(nums):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
        indices.append(pos)
        prev[i] = indices[pos - 1] if pos > 0 else -1

    # Reconstruct
    result = []
    idx = len(tails) - 1
    for i in range(len(nums) - 1, -1, -1):
        if indices[i] == idx:
            result.append(nums[i])
            idx -= 1
    return result[::-1]
\`\`\`

---

### Q4. Thread-Safe LRU Cache

\`\`\`python
from collections import OrderedDict
import threading

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()  # reentrant lock

    def get(self, key: int) -> int:
        with self.lock:
            if key not in self.cache:
                return -1
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # evict LRU
\`\`\`

---

## ML / NLP Technical *(finance-domain)*

### Q5. Stock Movement Prediction from Earnings Call Transcripts

**Full pipeline design** (asked as open-ended ML design):

\`\`\`python
# Step 1: Data preparation
# Label: direction of stock movement 24h after earnings call
# Features: transcript text, historical price, sector

from transformers import AutoTokenizer, AutoModel
import torch

# Step 2: FinBERT embeddings (pre-trained on financial text)
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert')

def encode_transcript(text: str) -> torch.Tensor:
    """Encode earnings call to CLS embedding."""
    inputs = tokenizer(
        text, max_length=512, truncation=True,
        return_tensors='pt', padding=True
    )
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :]  # CLS token

# Step 3: Fine-tune classifier head
class StockDirectionClassifier(torch.nn.Module):
    def __init__(self, hidden=768):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 2),  # up / down
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)

# Step 4: Key risks
# - Target leakage: call time vs market close — use only post-close calls
# - Look-ahead bias: ensure no future data in features
# - Evaluation: don't use accuracy alone; use information coefficient (IC)
#   and Sharpe ratio of a portfolio based on model predictions
\`\`\`

---

### Q6. Named Entity Recognition for Financial Text

\`\`\`python
# Fine-tune BERT for financial NER: companies, tickers, currencies, dates

from transformers import BertForTokenClassification, Trainer, TrainingArguments

LABELS = ['O', 'B-COMPANY', 'I-COMPANY', 'B-TICKER', 'I-TICKER',
          'B-CURRENCY', 'I-CURRENCY', 'B-DATE', 'I-DATE']
label2id = {l: i for i, l in enumerate(LABELS)}

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(LABELS),
    id2label={i: l for l, i in label2id.items()},
    label2id=label2id,
)

# Key challenges:
# 1. Abbreviations: "AAPL" must map to Apple Inc.
#    -> Add entity linking layer post-NER with Bloomberg ticker lookup
# 2. Multi-word spans: "Goldman Sachs Group Inc." = one entity
#    -> BIO tagging handles this
# 3. Nested entities: "Apple Inc. CEO Tim Cook"
#    -> Standard BERT NER can't handle — use span-based approach
\`\`\`

---

### Q7. Time-Series Anomaly Detection for Bond Prices

\`\`\`python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_anomaly_detector(prices: np.ndarray, contamination: float = 0.01):
    """
    ARIMA residual-based anomaly detection for bond price series.
    contamination: expected fraction of anomalies
    """
    # Fit ARIMA(1,1,1) on training window
    model = ARIMA(prices[:-100], order=(1, 1, 1))
    fit = model.fit()

    # Predict on held-out window
    forecasts = fit.forecast(steps=100)
    residuals = prices[-100:] - forecasts

    # Threshold: mean + 3*std of residuals
    mu, sigma = residuals.mean(), residuals.std()
    threshold = mu + 3 * sigma

    anomalies = np.where(np.abs(residuals) > threshold)[0]
    return anomalies

# Alternative: LSTM reconstruction error
# Train autoencoder on normal price windows
# Anomaly = reconstruction_error > percentile(99)
\`\`\`

---

### Q8. BLEU vs BERTScore — When Each Fails

\`\`\`
BLEU:
  - Counts n-gram overlap between hypothesis and reference
  - Fails when paraphrase is correct: "The dog ran fast" vs "The canine sprinted"
    -> BLEU = 0 (no 2-gram overlap), but meaning identical
  - Sensitive to tokenization, case, punctuation
  - Still standard for MT benchmarks (fast, reproducible)

BERTScore:
  - Computes cosine similarity of contextual BERT embeddings
  - Captures semantic similarity across paraphrases
  - Fails when: BERT lacks domain knowledge (financial jargon, tickers)
  - More expensive: requires BERT forward pass per sentence
  - Better for: summarization, paraphrase, dialogue quality

For Bloomberg:
  Use ROUGE-L for recall of key financial facts (numbers, dates, names)
  Use BERTScore for semantic coherence
  Use domain-specific factual accuracy metric: % of financial entities correctly retained
\`\`\`

---

### Q9. Detect Model Drift in Production Credit Risk Model

\`\`\`python
import numpy as np
from scipy.stats import ks_2samp

def detect_feature_drift(
    reference: np.ndarray,   # feature values at training time
    current: np.ndarray,     # feature values in production
    alpha: float = 0.05,
) -> dict:
    """
    Kolmogorov-Smirnov test for feature distribution shift.
    Returns: {feature_idx: (statistic, p_value, is_drifted)}
    """
    results = {}
    for i in range(reference.shape[1]):
        stat, p = ks_2samp(reference[:, i], current[:, i])
        results[i] = {
            'statistic': stat,
            'p_value': p,
            'drifted': p < alpha,
        }
    return results

def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                                n_bins: int = 10) -> float:
    """PSI > 0.25 indicates significant distribution shift."""
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected) + 1e-8
    act_pct = np.histogram(actual, bins=bins)[0] / len(actual) + 1e-8
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

# Thresholds: PSI < 0.1 = no drift, 0.1-0.25 = moderate, > 0.25 = retrain
\`\`\``,

    zh: `## Bloomberg技术面试题

Bloomberg AI面试结合**金融领域知识+ML工程**。预期NLP问题基于金融文本（财报电话会议、申报文件），时间序列建模，以及反映真实Bloomberg工程问题的编码题。

---

## 编程题

### Q1. 从零实现HashMap
链地址法处理冲突，负载因子0.75触发扩容，均摊O(1)。最坏情况O(n)（所有键哈希到同一桶）。

### Q2. Dijkstra最短路（金融工具图）
货币/工具之间的最短转换路径。扩展：用Bellman-Ford检测负环（对套利检测有用）。

### Q3. 最长递增子序列 O(n log n)
耐心排序方法：用二分查找维护"牌堆"序列。可重建实际LIS序列。

### Q4. 线程安全LRU缓存
OrderedDict + RLock实现线程安全的LRU缓存，支持并发读写。

---

## ML/NLP技术题（金融领域）

### Q5. 财报电话会议股票走势预测
完整ML流水线：FinBERT嵌入 → 分类器微调。关键风险：目标泄露（电话时间vs市场收盘）、前瞻偏差。用信息系数（IC）和夏普比率评估，而非单纯准确率。

### Q6. 金融文本命名实体识别
BERT微调，识别公司、股票代码、货币、日期。挑战：缩写解析（AAPL→Apple Inc.）、多词实体、嵌套实体。

### Q7. 债券价格时序异常检测
ARIMA残差法：预测值vs实际值，残差超过均值+3σ为异常。或用LSTM重建误差检测异常。

### Q8. BLEU vs BERTScore
BLEU：n-gram重叠，快速但无法捕捉语义等价（同义词改写得0分）。
BERTScore：上下文嵌入余弦相似度，捕捉语义，但对领域特定术语效果差。
Bloomberg场景：ROUGE-L召回关键金融事实，BERTScore评估语义连贯性。

### Q9. 信用风险模型漂移检测
KS检验检测特征分布偏移，PSI>0.25触发重训练。日常AUC和KS统计量监控，影子模型对比。`,
  },
}

export const bloombergAlgorithmic: TopicContent = {
  id: 'company-bloomberg-algorithmic',
  title: { en: 'Bloomberg — Algorithmic & DS Coding Questions', zh: 'Bloomberg算法与数据结构编程题' },
  contentType: 'code',
  content: {
    en: `## Bloomberg — Algorithmic & DS Coding Questions

Bloomberg engineering interviews emphasize **practical data structures, graph problems, and system design** problems similar to real Bloomberg Terminal engineering. Questions sourced from 1point3acres community reports.

---

## Linked Lists

### Add Two Numbers (Forward Order)
Numbers stored with the most-significant digit first — reverse the lists, add, then reverse result.

\`\`\`python
def addTwoNumbers(l1, l2):
    def reverse(head):
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev

    l1, l2 = reverse(l1), reverse(l2)
    dummy = ListNode(0)
    cur, carry = dummy, 0
    while l1 or l2 or carry:
        s = carry + (l1.val if l1 else 0) + (l2.val if l2 else 0)
        carry, s = divmod(s, 10)
        cur.next = ListNode(s)
        cur = cur.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return reverse(dummy.next)
\`\`\`

---

### Flatten a Multilevel Doubly Linked List

\`\`\`python
def flatten(head):
    if not head: return head
    cur = head
    while cur:
        if cur.child:
            child_head = flatten(cur.child)
            child_tail = child_head
            while child_tail.next:
                child_tail = child_tail.next
            # splice child list between cur and cur.next
            child_tail.next = cur.next
            if cur.next:
                cur.next.prev = child_tail
            cur.next = child_head
            child_head.prev = cur
            cur.child = None
        cur = cur.next
    return head
\`\`\`

---

## Graphs / BFS

### Check Connectivity Between Two Subway Stations

\`\`\`python
from collections import deque, defaultdict

def can_reach(edges: list[tuple], src: int, dst: int) -> bool:
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited = {src}
    q = deque([src])
    while q:
        node = q.popleft()
        if node == dst: return True
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return False
\`\`\`

---

### Grid Path Reachability with Refueling (Fuel-Constrained BFS)

\`\`\`python
from collections import deque

def can_reach_with_fuel(grid, fuel: int) -> bool:
    """BFS where state = (row, col, remaining_fuel)."""
    R, C = len(grid), len(grid[0])
    visited = set()
    q = deque([(0, 0, fuel)])
    while q:
        r, c, f = q.popleft()
        if r == R-1 and c == C-1: return True
        if (r, c, f) in visited: continue
        visited.add((r, c, f))
        for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#':
                nf = fuel if grid[nr][nc] == 'G' else f - 1  # 'G' = gas station
                if nf >= 0:
                    q.append((nr, nc, nf))
    return False
\`\`\`

---

### Find All Paths from Source to Target in DAG

\`\`\`python
def allPathsSourceTarget(graph: list[list[int]]) -> list[list[int]]:
    target = len(graph) - 1
    result = []
    def dfs(node, path):
        if node == target:
            result.append(path[:])
            return
        for nb in graph[node]:
            path.append(nb)
            dfs(nb, path)
            path.pop()
    dfs(0, [0])
    return result
\`\`\`

---

## Greedy / DP

### Two-City Onsite Flight Cost Minimization

\`\`\`python
def twoCitySchedCost(costs: list[list[int]]) -> int:
    # Greedy: send person to city A where (cost_A - cost_B) is smallest
    costs.sort(key=lambda x: x[0] - x[1])
    n = len(costs) // 2
    return sum(costs[i][0] for i in range(n)) + sum(costs[i][1] for i in range(n, 2*n))
\`\`\`

---

### Meeting Rooms II — Minimum Rooms

\`\`\`python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    intervals.sort()
    heap = []  # end times
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)
\`\`\`

---

### Candy Crush 1D — Stabilize by Repeatedly Removing Groups

\`\`\`python
def candyCrush1D(s: str, k: int = 3) -> str:
    """Remove consecutive groups of >= k, repeat until stable."""
    stack = []  # (char, count)
    for c in s:
        if stack and stack[-1][0] == c:
            stack[-1][1] += 1
        else:
            stack.append([c, 1])
        if stack[-1][1] == k:
            stack.pop()
    return ''.join(c * n for c, n in stack)
\`\`\`

---

### Partition Equal Subset Sum

\`\`\`python
def canPartition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp |= {x + n for x in dp if x + n <= target}
    return target in dp
\`\`\`

---

### Best Time to Buy and Sell Stock IV (at most k transactions)

\`\`\`python
def maxProfit(k: int, prices: list[int]) -> int:
    n = len(prices)
    if not n: return 0
    if k >= n // 2:  # unlimited transactions
        return sum(max(prices[i+1]-prices[i], 0) for i in range(n-1))
    dp = [[0]*n for _ in range(k+1)]
    for t in range(1, k+1):
        max_so_far = -prices[0]
        for d in range(1, n):
            dp[t][d] = max(dp[t][d-1], prices[d] + max_so_far)
            max_so_far = max(max_so_far, dp[t-1][d] - prices[d])
    return dp[k][-1]
\`\`\`

---

## Stack / Strings

### Remove Invalid Parentheses (Minimum Removal, All Valid Strings)

\`\`\`python
from collections import deque

def removeInvalidParentheses(s: str) -> list[str]:
    def is_valid(t):
        count = 0
        for c in t:
            if c == '(': count += 1
            elif c == ')':
                count -= 1
                if count < 0: return False
        return count == 0

    visited, result = {s}, []
    q = deque([s])
    found = False
    while q:
        cur = q.popleft()
        if is_valid(cur):
            result.append(cur)
            found = True
        if found: continue
        for i in range(len(cur)):
            if cur[i] not in '()': continue
            nxt = cur[:i] + cur[i+1:]
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return result
\`\`\`

---

### Decode String (Nested Patterns)

\`\`\`python
def decodeString(s: str) -> str:
    stack, cur_str, cur_num = [], '', 0
    for c in s:
        if c.isdigit():
            cur_num = cur_num * 10 + int(c)
        elif c == '[':
            stack.append((cur_str, cur_num))
            cur_str, cur_num = '', 0
        elif c == ']':
            prev_str, num = stack.pop()
            cur_str = prev_str + cur_str * num
        else:
            cur_str += c
    return cur_str
\`\`\`

---

### Gas Station — Can Complete Circuit?

\`\`\`python
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    if sum(gas) < sum(cost): return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            tank = 0
            start = i + 1
    return start
\`\`\`

---

### Validate Stack Sequences

\`\`\`python
def validateStackSequences(pushed: list[int], popped: list[int]) -> bool:
    stack, j = [], 0
    for val in pushed:
        stack.append(val)
        while stack and stack[-1] == popped[j]:
            stack.pop()
            j += 1
    return not stack
\`\`\`

---

### Longest Substring Without Repeating Characters

\`\`\`python
def lengthOfLongestSubstring(s: str) -> int:
    last = {}
    left = res = 0
    for right, c in enumerate(s):
        if c in last and last[c] >= left:
            left = last[c] + 1
        last[c] = right
        res = max(res, right - left + 1)
    return res
\`\`\`

---

## Data Structures / Design

### Insert Delete GetRandom O(1)

\`\`\`python
import random

class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.idx = {}   # val -> index in vals

    def insert(self, val: int) -> bool:
        if val in self.idx: return False
        self.idx[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.idx: return False
        i = self.idx[val]
        last = self.vals[-1]
        self.vals[i] = last
        self.idx[last] = i
        self.vals.pop()
        del self.idx[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)
\`\`\`

---

### Design a Lottery System with O(1) Delete

Same as RandomizedSet above — the key insight is swapping the deleted element with the last element to maintain a compact array. Use this pattern for any "delete from middle of array in O(1)" problem.

---

### Design Ordered Stream

\`\`\`python
class OrderedStream:
    def __init__(self, n: int):
        self.buf = [None] * (n + 1)
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> list[str]:
        self.buf[idKey] = value
        result = []
        while self.ptr < len(self.buf) and self.buf[self.ptr]:
            result.append(self.buf[self.ptr])
            self.ptr += 1
        return result
\`\`\`

---

### Design Top-K Frequent Elements (Insert / Delete / Query)

\`\`\`python
import heapq
from collections import defaultdict

class TopKFrequent:
    def __init__(self):
        self.freq = defaultdict(int)

    def insert(self, val: int):
        self.freq[val] += 1

    def delete(self, val: int):
        if self.freq[val] > 0:
            self.freq[val] -= 1

    def top_k(self, k: int) -> list[int]:
        return heapq.nlargest(k, self.freq, key=self.freq.get)
        # O(n log k) — for O(log n) per insert/delete use a sorted container
\`\`\`

---

### Trie Data Structure

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children: return False
            node = node.children[c]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children: return False
            node = node.children[c]
        return True
\`\`\`

---

## Tree

### Binary Tree Vertical Order Traversal

\`\`\`python
from collections import defaultdict, deque

def verticalOrder(root) -> list[list[int]]:
    if not root: return []
    col_map = defaultdict(list)
    q = deque([(root, 0)])
    min_col = max_col = 0
    while q:
        node, col = q.popleft()
        col_map[col].append(node.val)
        min_col = min(min_col, col)
        max_col = max(max_col, col)
        if node.left:  q.append((node.left, col - 1))
        if node.right: q.append((node.right, col + 1))
    return [col_map[c] for c in range(min_col, max_col + 1)]
\`\`\`

---

### Find Root Cause Node from Parent-to-Children Logs

\`\`\`python
def find_root_cause(logs: list[tuple]) -> int:
    """
    logs: [(parent, child), ...] — directed edges of an error propagation tree.
    The root cause is the node that appears only as a parent, never as a child.
    """
    children = {child for _, child in logs}
    parents  = {parent for parent, _ in logs}
    roots = parents - children
    assert len(roots) == 1, "Multiple or no roots found"
    return roots.pop()
\`\`\`

---

## Other Reported Problems (Brief Reference)

| Problem | Key Technique |
|---------|--------------|
| Van Eck-like Sequence N-th Term | Simulate with last-seen dict |
| Enumerate Round-Robin Match Schedules | Backtracking / combinatorics |
| Bucket Values Into Boundary Ranges | Binary search / sorting |
| Find Last Person in Circular Elimination Game | Josephus problem: \`(n-1, k) -> (result+k) % n\` |
| LRU Cache with BFS Enhancement | OrderedDict + BFS layer for proximity-aware eviction |
| Streaming Palindrome Checker (sublinear) | Rolling hash / Manacher online |
| Median of Two Sorted Arrays O(log(m+n)) | Binary search on partition |
| Spelling Bee Valid Words | Trie + bitmask (center letter + 7-letter set) |
| Count Islands in 2D Grid | DFS/BFS flood fill |
| Max Consecutive Ones II | Sliding window with at most one 0 flip |
| Check Meetings Scheduled Without Overlaps | Sort by start, check prev end ≤ cur start |
| Subsets | Bitmask or backtracking |
| Word Search in 2D Board | DFS with visited set |
| Implement atoi | Handle whitespace, sign, overflow carefully |
| Sort Words by Custom Alphabet | Map char → rank, sort with key |
| Simplified Grep / Wildcard Expansion | DFS or DP with \`.\` and \`*\` / \`?\` matching |
| Evaluate Reverse Polish Notation | Stack: push numbers, pop on operator |
| Remove Consecutive Letters | Stack: pop if top equals current |
| Find k-th Unique Character | Count freq, iterate for k-th with freq=1 |
| Decode Encoded String with Nested Patterns | Same as Decode String above |
| Ashley Loves Numbers | Simulate per problem constraints |
| Consecutive Sum (Intern) | Two-pointer / math: n*(n+1)/2 offset |
| Find Longest Chain | Sort by end, greedy — same as activity selection |
| Deck of Cards Shuffling | Fisher-Yates shuffle on 52-element array |
| Design 52-Card Deck for Blackjack | OOD: Card, Deck, Hand classes with draw/shuffle |`,

    zh: `## Bloomberg算法与数据结构编程题

Bloomberg工程面试强调**实用数据结构、图问题和系统设计**，类似真实Bloomberg终端工程问题。

---

## 链表

### 正向顺序两数相加
反转两个链表 → 相加 → 反转结果。关键：在反转辅助函数后执行标准进位加法。

### 多级双向链表展平
递归展平子列表，将子列表末尾与cur.next相连，清除child指针。

---

## 图/BFS

### 地铁站连通性检查
构建无向图，从src做BFS/DFS，判断是否能到达dst。

### 带补油的网格路径可达性
BFS状态=(行,列,剩余燃料)。遇到加油站恢复满油，否则每步减1，油量<0则剪枝。

### DAG所有路径
DFS+回溯，path到达目标时记录结果。

---

## 贪心/动态规划

### 双城飞行费用最小化
按(cost_A - cost_B)排序，前半派去A，后半派去B。

### 会议室II最少房间数
最小堆维护结束时间，若堆顶≤新会议开始则复用房间。

### 1D糖果消消乐
栈记录(字符, 计数)，连续k个则弹出，重复直到稳定。

### 分割等和子集
DP背包：dp = 可达目标值集合，逐个加入元素扩展。

### 买卖股票IV（最多k笔）
二维DP：dp[t][d] = 第d天完成t笔交易的最大利润。

---

## 栈/字符串

### 删除最少括号使其合法
BFS按层删除一个字符，找到有效字符串即停止（最少删除数）。

### 解码嵌套字符串
栈保存(当前字符串, 重复次数)，遇]时展开。

### 加油站能否完成环路
贪心：总油量≥总消耗则必有解，从tank<0的下一站重新开始。

### 验证栈序列
模拟入栈，每次入栈后尽量弹出匹配popped序列。

### 最长不重复子串
滑动窗口：last记录每字符最后位置，left跳到冲突字符之后。

---

## 数据结构/设计

### 常数时间插入删除随机获取
数组+哈希表：删除时将最后元素交换到被删位置，维护O(1)。

### 有序流
缓冲区+指针：插入后从ptr开始连续返回已填充的元素。

### Top-K频繁元素
频率字典+heapq.nlargest实现查询。高频insert/delete场景用有序容器。

### 字典树Trie
节点含children字典和is_end标记，支持insert/search/startsWith。

---

## 树

### 二叉树垂直顺序遍历
BFS携带列编号，按列聚合节点值，从min_col到max_col输出。

### 从父子日志找根因节点
所有child节点集合与所有parent节点集合做差集，结果即根因节点。

---

## 其他常见题（简要参考）

| 题目 | 核心技术 |
|------|---------|
| Van Eck序列第N项 | 用last_seen字典模拟 |
| 枚举循环赛日程 | 回溯/组合数学 |
| 按边界范围分桶 | 二分查找/排序 |
| 约瑟夫环最后一人 | 递推公式：(n-1, k) → (result+k) % n |
| 流式回文检测（次线性） | 滚动哈希/在线Manacher |
| 两个有序数组中位数 | 二分查找分区，O(log(m+n)) |
| 拼写蜂单词查找 | Trie + 位掩码（中心字母+7字母集） |
| 统计2D网格岛屿数 | DFS/BFS洪水填充 |
| 最多翻转一次0的最长连续1 | 滑动窗口 |`,
  },
}

export const universalAIQuestions: TopicContent = {
  id: 'company-universal-ai',
  title: { en: 'Universal AI Questions — All Companies', zh: '通用AI面试题（所有公司）' },
  contentType: 'code',
  content: {
    en: `## Universal AI Technical Questions

These core topics appear across OpenAI, Google, Bloomberg, and virtually every AI/ML role. Master both the formula AND the intuition.

---

## Transformers — Deep Technical

### Q1. Derive Scaled Dot-Product Attention

\`\`\`
Input: X ∈ R^{n×d}
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V   where W ∈ R^{d×d_k}

Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Why sqrt(d_k) scaling?
  Q @ K^T has entries q_i · k_j = sum of d_k products of random variables
  Variance of dot product ≈ d_k (if Q,K ~ N(0,1))
  Without scaling: variance grows with d_k -> softmax becomes peaky -> gradients vanish
  Dividing by sqrt(d_k) stabilizes variance at 1.0

Numerical stability: subtract row max before softmax
  softmax(x_i) = exp(x_i - max) / sum exp(x_j - max)  (mathematically identical, avoids overflow)
\`\`\`

\`\`\`python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
\`\`\`

---

### Q2. Why RoPE (Rotary Position Embedding) over learned absolute positions?

\`\`\`
Absolute learned embeddings:
  x_i = token_embed[i] + pos_embed[i]
  Problem: position 512 never seen during training — no generalization

ALiBi (linear bias):
  Subtract linear penalty m*|i-j| from attention scores
  Works but artificially decays long-range attention

RoPE (Su et al. 2021):
  Rotate Q and K by a position-dependent angle theta
  q_i = R(i * theta) @ q_i,   k_j = R(j * theta) @ k_j
  Dot product: q_i · k_j = f(i-j) — encodes RELATIVE position
  Advantage: position information appears in Q·K dot product naturally
  Generalizes to longer sequences (extrapolation works with YaRN / NTK scaling)
\`\`\`

---

### Q3. FlashAttention — Why it's faster despite more FLOPs

\`\`\`
Standard attention:
  1. Load Q, K from HBM -> compute S = Q@K^T (n×n) -> write to HBM (32MB for n=4096)
  2. Load S -> apply softmax -> write back
  3. Load softmax(S), V -> compute O = softmax(S)@V -> write to HBM
  Total HBM IO: O(n^2) — memory bandwidth bound for n > 512

FlashAttention:
  Tiles Q into blocks of size B_r, K/V into blocks of size B_c
  Fits tiles in SRAM (228KB/SM on H100)
  Online softmax: update running (max, sum) as new K blocks arrive
  NEVER writes n×n matrix to HBM
  Total HBM IO: O(n * d) — linear!

Result: 2-4x faster wall-clock time for n=2048+
        despite more total FLOPs (repeated loading of K,V per Q block)
Key insight: memory bandwidth, not compute, is the bottleneck
\`\`\`

---

## RLHF / Alignment

### Q4. Implement DPO Loss

\`\`\`python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple:
    """
    DPO loss (Rafailov et al. 2023).
    beta: KL penalty — higher = stay closer to reference model.

    Math: r(x,y) = beta * log(pi(y|x)/pi_ref(y|x))
    Loss = -E[log sigma(r(chosen) - r(rejected))]
    """
    r_chosen   = beta * (policy_chosen_logps   - ref_chosen_logps)
    r_rejected = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    acc  = (r_chosen > r_rejected).float().mean()
    return loss, acc

# Key insight vs RLHF:
# RLHF needs 4 models (policy, reference, reward, value)
# DPO needs only 2 (policy, reference) — same optimization target, simpler
\`\`\`

---

### Q5. What is Reward Hacking? How do you fix it?

\`\`\`
Reward hacking: the policy finds behaviors that maximize the proxy reward
but violate the true intent.

Examples:
  - Summarization: model produces confident-sounding but wrong summaries
    (reward model rates confident text higher)
  - Chatbot: model becomes sycophantic (agrees with user regardless of truth)
  - Game: RL agent spins in circles collecting bonuses instead of completing task

Root cause: reward model is an imperfect proxy for human preferences.
The policy, being optimized, finds and exploits the proxy's blind spots.

Fixes:
  1. KL penalty: total_reward = r_model - beta * KL(policy || reference)
     Constrains how far policy drifts from safe baseline
  2. Reward model ensemble: average multiple reward models (harder to jointly fool)
  3. Constitutional AI: model critiques its own output against principles
  4. Process reward model (PRM): reward at each step, not just final output
  5. Red-teaming: adversarially probe reward model to find exploitable gaps
\`\`\`

---

## RAG — Production Questions

### Q6. Hybrid RAG Pipeline with Reranking

\`\`\`python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

class HybridRAG:
    def __init__(self):
        self.bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.bm25 = None  # BM25Okapi from rank_bm25

    def retrieve(self, query: str, corpus: list[str], top_k: int = 20) -> list[str]:
        # Stage 1a: Dense retrieval (semantic)
        q_emb = self.bi_encoder.encode(query)
        c_emb = self.bi_encoder.encode(corpus)
        dense_scores = np.dot(c_emb, q_emb) / (
            np.linalg.norm(c_emb, axis=1) * np.linalg.norm(q_emb)
        )
        dense_top = np.argsort(dense_scores)[::-1][:top_k]

        # Stage 1b: Sparse retrieval (BM25 / keyword)
        bm25_scores = np.array(self.bm25.get_scores(query.split()))
        sparse_top = np.argsort(bm25_scores)[::-1][:top_k]

        # Fusion: Reciprocal Rank Fusion
        candidates = list(set(dense_top) | set(sparse_top))
        rrf_scores = {}
        for rank, idx in enumerate(dense_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rank + 60)
        for rank, idx in enumerate(sparse_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rank + 60)
        fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Stage 2: Reranking with cross-encoder (slower, more accurate)
        pairs = [(query, corpus[i]) for i in fused]
        rerank_scores = self.cross_encoder.predict(pairs)
        reranked = [fused[i] for i in np.argsort(rerank_scores)[::-1]]
        return [corpus[i] for i in reranked[:5]]
\`\`\`

---

### Q7. Detect and Fix Embedding Drift

\`\`\`python
import numpy as np
from scipy.stats import ks_2samp

def detect_embedding_drift(
    baseline_embeddings: np.ndarray,   # shape: (N, d) from training time
    current_embeddings: np.ndarray,    # shape: (M, d) from production
    threshold: float = 0.05,
) -> dict:
    """
    Detect if embedding distribution has shifted.
    Uses: pairwise cosine similarity distribution comparison (KS test)
    """
    # Sample pairwise cosine similarities (expensive to do all pairs)
    n = min(1000, len(baseline_embeddings))
    idx_b = np.random.choice(len(baseline_embeddings), (n, 2))
    idx_c = np.random.choice(len(current_embeddings), (n, 2))

    def cos_sim(a, b):
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    base_sims = cos_sim(baseline_embeddings[idx_b[:,0]], baseline_embeddings[idx_b[:,1]])
    curr_sims = cos_sim(current_embeddings[idx_c[:,0]], current_embeddings[idx_c[:,1]])

    stat, p_value = ks_2samp(base_sims, curr_sims)
    drifted = p_value < threshold

    return {
        'ks_statistic': stat,
        'p_value': p_value,
        'drifted': drifted,
        'action': 're-embed all documents with new model' if drifted else 'no action needed'
    }
\`\`\`

---

## Distributed Training

### Q8. DDP All-Reduce — What Happens Under the Hood

\`\`\`
Ring All-Reduce (NCCL default):

  Setup: N GPUs arranged in a ring.
  Each GPU starts with its own gradient tensor of size S.

  Phase 1 — Reduce-Scatter (N-1 steps):
    Each GPU sends S/N chunk to its right neighbor
    Each step: recv chunk + accumulate (sum) + send forward
    After N-1 steps: each GPU holds the FULLY REDUCED sum for 1/N of the tensor

  Phase 2 — All-Gather (N-1 steps):
    Each GPU broadcasts its reduced chunk to all others
    After N-1 steps: every GPU has the full reduced tensor

  Total data sent per GPU: 2 * (N-1)/N * S ≈ 2S
  This is OPTIMAL — cannot do better (2S is the lower bound for all-reduce)

  Bandwidth: N * bandwidth per link
  Latency: O(N) — why large clusters use hierarchical all-reduce (node-level then inter-node)
\`\`\`

---

### Q9. FSDP vs DDP — Memory Analysis

\`\`\`
Model: 7B parameters, fp16 (2 bytes each)
Model size: 7B * 2 = 14GB

DDP (8 GPUs):
  Each GPU holds: 14GB (model) + 14GB (grads) + optimizer states (~56GB for AdamW)
  Total per GPU: ~84GB
  Requires: H100 80GB — DOESN'T FIT

FSDP Full Shard (8 GPUs):
  Model sharded: 14GB / 8 = 1.75GB per GPU
  Gradients sharded: 1.75GB per GPU
  Optimizer sharded: ~7GB per GPU (56GB / 8)
  Total per GPU: ~10.5GB
  Requires: A100 40GB — FITS

FSDP communication overhead:
  Forward: all-gather parameters before each layer, free after
  Backward: all-gather for gradient computation, reduce-scatter after
  Total overhead: ~2x DDP communication, but enables training larger models
\`\`\``,

    zh: `## 通用AI技术面试题

这些核心主题出现在OpenAI、Google、Bloomberg及几乎所有AI/ML岗位。需要同时掌握公式和直觉。

---

## Transformer深度技术

### Q1. 推导缩放点积注意力
Q = X@W_Q, K = X@W_K, V = X@W_V。为何除以sqrt(d_k)：防止方差随d_k增长导致softmax过于尖锐、梯度消失。数值稳定性：softmax前减去行最大值。

### Q2. RoPE vs 绝对位置嵌入
绝对嵌入：训练时未见过的位置无法泛化。RoPE：用位置角度旋转Q和K，点积编码相对位置，自然泛化到更长序列。

### Q3. FlashAttention原理
标准注意力：n×n矩阵写HBM，O(n²)内存IO。FlashAttention：在SRAM中分块计算，在线softmax，从不将注意力矩阵写入HBM，总IO降为O(nd)。关键洞察：内存带宽是瓶颈，不是计算。

---

## RLHF/对齐

### Q4. 实现DPO损失
r(x,y) = beta * log(pi(y|x)/pi_ref(y|x))，损失 = -E[log σ(r(chosen) - r(rejected))]。与RLHF相比：只需2个模型而非4个。

### Q5. 奖励黑客及修复
策略利用代理奖励模型的盲点。修复：KL惩罚约束偏移、奖励模型集成、Constitutional AI、过程奖励模型（PRM）、红队测试。

---

## RAG生产问题

### Q6. 混合RAG+重排序
双编码器密集检索 + BM25稀疏检索，RRF融合，交叉编码器重排序（更准确但更慢），最终返回top-5上下文。

### Q7. 嵌入漂移检测
通过KS检验比较基线和当前嵌入的成对余弦相似度分布。p值<0.05触发重新嵌入所有文档。

---

## 分布式训练

### Q8. DDP All-Reduce原理
环形All-Reduce：(N-1)步Reduce-Scatter + (N-1)步All-Gather。每GPU发送~2S数据（最优下界），带宽随GPU数线性扩展。

### Q9. FSDP vs DDP内存分析
7B模型，DDP：每GPU~84GB（超出A100 80GB）。FSDP全分片（8GPU）：每GPU~10.5GB（参数+梯度+优化器状态均分片）。代价：通信量约为DDP的2倍。`,
  },
}
