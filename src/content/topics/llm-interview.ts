import type { TopicContent } from '../types'

// ─────────────────────────────────────────────────────────────────────────────
// LLM Interview: Expansion Nodes
// Source: 大模型面试题集 — 微调面 / 推理面 / 进阶面 / 强化学习面 / 评测面
// ─────────────────────────────────────────────────────────────────────────────

export const llmInterviewArchitecture: TopicContent = {
  id: 'llm-interview-architecture',
  title: { en: 'LLM Interview: Architectural Hard-Talk', zh: 'LLM面试：架构核心问题' },
  contentType: 'article',
  content: {
    en: `## Architectural Hard-Talk — Key Trade-offs

> Source: *大模型面试题集* — 微调面 + 强化学习面
> The hardest questions interviewers ask about LLM architecture, training pipelines, and design decisions.

---

### Q1 ★★★☆☆ — How much VRAM does full-parameter fine-tuning require?

**Answer:** VRAM requirements depend on four factors:

| Factor | Impact |
|--------|--------|
| **Model size** (# params) | Primary driver — each param ≈ 4 bytes in FP32, 2 bytes in FP16 |
| **Batch size** | Larger batch = more activations in memory |
| **Sequence length** | Longer sequences = larger attention maps (quadratic) |
| **Optimizer state** | Adam stores 2× model size in extra buffers |

**Rule of thumb:** A 7B-param model in FP16 needs ~14 GB just for weights. Add optimizer states (Adam: ×3) → ~42 GB minimum. Typical requirement: **≥ 16 GB for small models; 40–80 GB for 7B+ models**.

**Reducing VRAM:**
- Use FP16 / BF16 mixed precision (halves weight memory)
- Gradient accumulation (accumulate N micro-batches before updating)
- PEFT methods (LoRA, prefix tuning) — only train adapter parameters
- ZeRO optimizer (DeepSpeed) — shards optimizer state across GPUs

---

### Q2 ★★★★☆ — Should you use Base or Chat model as the SFT starting point?

**Answer:**

| Scenario | Recommended Starting Point |
|----------|---------------------------|
| Multi-turn conversation tasks | **Chat model** — already has dialogue structure |
| Single-turn generation / classification | **Base model** — more flexible, less constrained |
| Domain fine-tuning with specialized data | **Base model** — Chat model RLHF constraints can interfere |
| Fast prototyping / limited data | **Chat model** — pre-aligned, needs less SFT data |

**Key insight:** Chat models have been RLHF-aligned to refuse harmful instructions. If your fine-tuning data conflicts with that alignment, you may get unexpected refusals. Base models are "raw" and adapt more freely.

---

### Q3 ★★★☆☆ — Where is knowledge injected: pre-training or fine-tuning?

**Answer:** Knowledge is primarily injected during **pre-training**.

| Stage | Purpose | Knowledge Type |
|-------|---------|----------------|
| **Pre-training** | Learn language + world knowledge from massive text | Factual, linguistic, commonsense |
| **Fine-tuning (SFT)** | Learn task format + instruction following | Format / style / task-specific behavior |

**Practical implication:** If a model lacks factual knowledge about a domain, SFT alone won't fix it — you need **Continue Pre-Training (CPT)** on domain data first, then SFT for instruction format.

**Decision rule:**
- "Model needs to *know* domain facts" → **CPT first**
- "Model knows the facts but responds in the wrong format" → **SFT only**

---

### Q4 ★★★★☆ — What are the limitations of RLHF in practice?

**Answer:** Five key pain points:

1. **High annotation cost** — expert labelers must evaluate model outputs; doesn't scale cheaply
2. **Subjectivity** — different annotators give inconsistent preference labels
3. **Feedback delay & sparsity** — rewards arrive late; sparse signals slow convergence
4. **Reward hacking** — policy learns to exploit the reward model, not the actual goal
5. **Exploration-exploitation imbalance** — over-reliance on human feedback reduces the policy's ability to explore novel strategies

**The SFT → RM → PPO pipeline bottlenecks:**
- PPO requires **4 models simultaneously** in memory: Policy (train), Reference Policy (frozen), Value Model (train), Reward Model (frozen) — 4× normal VRAM
- Each RLHF iteration cycle is slow: collect rollouts → score with RM → PPO update

**Alternatives being explored:** DPO (Direct Preference Optimization) — skips the explicit RM; RLAIF (AI feedback instead of human).

---

### Q5 ★★★☆☆ — BERT vs LLaMA vs ChatGLM — how to choose?

**Answer:**

| Model Family | Best For | Avoid When |
|---|---|---|
| **BERT** (encoder-only) | Classification, NER, semantic similarity, embedding | Generation tasks, open-ended QA |
| **LLaMA** (decoder-only) | General generation, domain fine-tuning, research | Need Chinese-first performance out of box |
| **ChatGLM** (encoder-decoder) | Chinese dialogue, bilingual tasks, chat | English-only tasks, classification |

**Selection checklist:**
1. Is the task generation or understanding? → Generation: LLaMA/ChatGLM; Understanding: BERT
2. Is the primary language Chinese? → ChatGLM
3. Do you need to fine-tune extensively? → LLaMA (more open ecosystem)
4. Is it a dialogue/conversation task? → ChatGLM or fine-tuned LLaMA

---

### Q6 ★★★★☆ — What does the model actually learn during SFT?

**Answer:** SFT teaches the model **format, not facts**. Specifically:

1. **Output format** — follow instruction templates (JSON, markdown, step-by-step)
2. **Task-specific label prediction** — map input → expected label/response structure
3. **Context utilization patterns** — which parts of the prompt to attend to
4. **Behavior alignment** — polite refusals, tone, length constraints

**What SFT does NOT teach:** New factual knowledge. If you ask the model a question it didn't see in pre-training, SFT data won't make it suddenly know the answer.

**Common interview trap:** "Can SFT inject new knowledge?" → Answer: **minimally**. SFT can reinforce known facts but rarely creates new associations reliably.
`,
    zh: `## 架构核心问题 — 关键权衡

> 来源：*大模型面试题集* — 微调面 + 强化学习面
> 面试官最常问的 LLM 架构、训练流程与设计决策问题。

---

### Q1 ★★★☆☆ — 全参数微调需要多少显存？

**答：** 显存需求取决于四个因素：

| 因素 | 影响 |
|------|------|
| **模型大小**（参数量） | 主要驱动因素 — FP32 下每参数约 4 字节，FP16 约 2 字节 |
| **批次大小** | 批次越大，中间激活值占用显存越多 |
| **序列长度** | 序列越长，注意力矩阵越大（二次方增长） |
| **优化器状态** | Adam 需额外存储约 2× 模型大小的状态 |

**经验公式：** 7B 参数模型 FP16 权重约 14 GB，加 Adam 优化器状态（约 3×）→ 最少需要约 42 GB。典型要求：**小模型 ≥ 16 GB；7B+ 模型需 40–80 GB**。

**降低显存的方法：**
- 使用 FP16 / BF16 混合精度
- 梯度累积（N 个 micro-batch 后才更新参数）
- PEFT 方法（LoRA、prefix tuning）— 只训练 adapter 参数
- ZeRO 优化器（DeepSpeed）— 跨 GPU 分片优化器状态

---

### Q2 ★★★★☆ — SFT 时应选 Base 还是 Chat 模型？

**答：**

| 场景 | 推荐起点 |
|------|---------|
| 多轮对话任务 | **Chat 模型** — 已具备对话结构 |
| 单轮生成 / 分类任务 | **Base 模型** — 更灵活，约束更少 |
| 领域专项微调（数据特殊） | **Base 模型** — Chat 的 RLHF 对齐可能干扰 |
| 快速原型 / 数据有限 | **Chat 模型** — 已对齐，所需 SFT 数据更少 |

**核心洞察：** Chat 模型经过 RLHF 对齐，会拒绝有害指令。如果微调数据与该对齐冲突，可能出现意外拒绝。Base 模型是"原始"状态，适应性更强。

---

### Q3 ★★★☆☆ — 知识是在预训练还是微调阶段注入的？

**答：** 知识主要在**预训练阶段**注入。

| 阶段 | 目的 | 知识类型 |
|------|------|---------|
| **预训练** | 从海量文本学习语言和世界知识 | 事实、语言学、常识 |
| **微调 (SFT)** | 学习任务格式和指令遵循 | 格式 / 风格 / 任务特定行为 |

**实际影响：** 如果模型缺乏某领域的事实知识，仅靠 SFT 无法解决 — 需要先在领域数据上做**持续预训练 (CPT)**，再做 SFT 调整指令格式。

**决策规则：**
- "模型需要*了解*领域事实" → **先做 CPT**
- "模型知道事实但回答格式不对" → **只做 SFT**

---

### Q4 ★★★★☆ — RLHF 在实践中有哪些不足？

**答：** 五大痛点：

1. **标注成本高** — 需要专家评估模型输出，难以规模化
2. **主观性强** — 不同标注员给出的偏好标签不一致
3. **反馈延迟与稀疏** — 奖励信号来得晚，稀疏信号使收敛变慢
4. **奖励欺骗** — 策略学会利用奖励模型的漏洞，而非真正优化目标
5. **探索-利用失衡** — 过度依赖人类反馈会削弱策略探索新策略的能力

**SFT → RM → PPO 流程瓶颈：**
- PPO 需要同时在内存中维护 **4 个模型**：策略模型（训练）、参考策略（冻结）、价值模型（训练）、奖励模型（冻结）— 显存需求是普通训练的 4 倍
- 每次 RLHF 迭代周期很长：收集 rollout → RM 评分 → PPO 更新

**替代方案：** DPO（直接偏好优化）— 跳过显式奖励模型；RLAIF — 用 AI 反馈代替人工反馈。

---

### Q5 ★★★☆☆ — BERT vs LLaMA vs ChatGLM 如何选？

**答：**

| 模型家族 | 最适合 | 避免用于 |
|---------|-------|---------|
| **BERT**（仅编码器） | 分类、NER、语义相似度、向量化 | 生成任务、开放式问答 |
| **LLaMA**（仅解码器） | 通用生成、领域微调、研究 | 开箱即用中文性能要求高 |
| **ChatGLM**（编码器-解码器） | 中文对话、双语任务、聊天 | 纯英文任务、分类 |

---

### Q6 ★★★★☆ — SFT 期间模型真正学习到了什么？

**答：** SFT 教的是**格式，而非事实**。具体来说：

1. **输出格式** — 遵循指令模板（JSON、Markdown、分步推理）
2. **任务标签预测** — 将输入映射到期望的标签/响应结构
3. **上下文利用模式** — 关注 prompt 中哪些部分
4. **行为对齐** — 礼貌拒绝、语气、长度限制

**SFT 不教的内容：** 新的事实知识。如果模型预训练时没见过某问题的答案，SFT 数据不会让它突然知道答案。

**常见面试陷阱：** "SFT 能注入新知识吗？" → 答：**几乎不能**。SFT 可以强化已知事实，但很难可靠地创建新的知识关联。
`,
  },
}

export const llmInterviewEdgeCases: TopicContent = {
  id: 'llm-interview-edge-cases',
  title: { en: 'LLM Interview: Edge Cases & Failure Modes', zh: 'LLM面试：边缘情况与故障模式' },
  contentType: 'article',
  content: {
    en: `## Edge Cases & Failure Modes

> Source: *大模型面试题集* — 微调面 + 进阶面
> Production failures, known weaknesses, and how to mitigate them.

---

### Failure Mode 1: Catastrophic Forgetting

**What it is:** After SFT on domain data, the model loses general capabilities it had before.

**Root causes:**
- New task data distribution differs significantly from pre-training data
- Parameter updates for new task overwrite weights critical for old tasks
- Gradient conflicts between new and old task objectives

**Mitigation strategies:**

| Strategy | Mechanism | Cost |
|----------|-----------|------|
| **Replay Buffer** | Mix old task samples into new training batches | Storage for replay dataset |
| **Elastic Weight Consolidation (EWC)** | Regularize important weights — penalize large changes to params critical for old tasks | Extra compute for Fisher info matrix |
| **Incremental Learning** | Fine-tune only a small subset of params per stage | Slower convergence |
| **Multi-Task Learning** | Train on old + new tasks simultaneously | Need to curate mixed dataset |
| **Data Mixing** | Blend domain data with general corpus (common ratio: 1:1 to 1:3) | Need access to original data |

**Best practice in production:** Always mix ~10–30% general data (WebText, Wikipedia) with domain-specific data during SFT.

---

### Failure Mode 2: The Repetition / Parrot Problem

**What it is:** LLM generates repetitive, looping text — keeps copying the same phrases endlessly.

**Root causes:**
1. **Training data bias** — corpus contains many duplicate documents
2. **Auto-regressive objective** — predicting the next token encourages copying high-probability patterns
3. **Low temperature** — deterministic decoding collapses onto the most probable token repeatedly

**Mitigation strategies:**

| Strategy | How |
|----------|-----|
| **Increase temperature** (0.7–1.0) | Flatten probability distribution, encourage diversity |
| **Top-p / nucleus sampling** | Only sample from top tokens covering probability mass p |
| **Repetition penalty** | Multiply logit of already-generated tokens by penalty factor < 1 |
| **Beam search tuning** | Larger beam width with diversity constraint (diverse beam search) |
| **Post-processing filter** | Detect n-gram repetitions and truncate or re-generate |

\`\`\`python
# Hugging Face generation params to fight repetition
output = model.generate(
    input_ids,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.3,   # > 1.0 penalizes repeats
    do_sample=True,
    max_new_tokens=512,
)
\`\`\`

---

### Failure Mode 3: OOM (Out of Memory) During Training

**What it is:** Training crashes with CUDA OOM error when scaling batch size or sequence length.

**Diagnosis checklist:**
1. Is batch size too large? → Reduce by half
2. Are sequences too long? → Truncate to 512 or use sliding window
3. Is FP32 being used? → Switch to BF16/FP16
4. Is the optimizer caching too much? → Use 8-bit Adam (bitsandbytes)

**Memory reduction hierarchy (apply in order):**

\`\`\`
1. FP16/BF16 mixed precision          → 2× memory reduction
2. Gradient checkpointing             → trade compute for memory (~2× slower, ~4× less mem)
3. Gradient accumulation steps=N      → virtual batch size without extra mem
4. Reduce batch size                  → direct linear reduction
5. 8-bit Adam optimizer               → 4× less optimizer state memory
6. LoRA / PEFT                        → only train <1% of parameters
7. CPU offload (DeepSpeed ZeRO-3)     → offload optimizer state to CPU RAM
\`\`\`

---

### Failure Mode 4: Long Context Degradation

**What it is:** Model performance drops significantly on inputs longer than its training context window.

**Why it happens:**
- Positional encodings (absolute) were only trained up to a fixed length
- Attention is O(n²) — longer sequences hit memory limits fast
- "Lost in the middle" phenomenon — model ignores content in the middle of long inputs

**Handling strategies:**

| Approach | Best For |
|----------|---------|
| **Chunking + summarization** | Documents > 10K tokens; summarize each chunk |
| **Hierarchical modeling** | Documents with clear section structure |
| **RAG (Retrieval)** | Only retrieve the top-k relevant chunks |
| **Sliding window attention** | Local context tasks (Longformer, BigBird) |
| **RoPE scaling** (LLaMA 2+) | Extend context window without retraining |

---

### Failure Mode 5: Domain Capability vs General Capability Trade-off

**What it is:** After training on domain data, model improves on domain tasks but weakens on general benchmarks.

**The dilemma:** Optimizing for domain performance hurts general intelligence. You can't have both at full strength with limited fine-tuning.

**Mitigation:**
- **CPT data ratio:** Domain:General ≈ 30:70 → preserves general ability while gaining domain knowledge
- **Domain adapter (LoRA):** Keeps base weights frozen; domain knowledge in adapter
- **Two-model serving:** One general model + one domain-fine-tuned model; route queries by type
`,
    zh: `## 边缘情况与故障模式

> 来源：*大模型面试题集* — 微调面 + 进阶面
> 生产环境中的故障、已知缺陷及缓解方法。

---

### 故障模式 1：灾难性遗忘

**定义：** 在领域数据上做 SFT 后，模型丧失了之前具备的通用能力。

**根本原因：**
- 新任务数据分布与预训练数据差异显著
- 新任务的参数更新覆盖了旧任务所依赖的关键权重
- 新旧任务目标之间存在梯度冲突

**缓解策略：**

| 策略 | 机制 | 代价 |
|------|------|------|
| **重播缓冲区** | 将旧任务样本混入新训练批次 | 需要存储回放数据集 |
| **弹性权重共享 (EWC)** | 对旧任务关键权重加正则约束，惩罚大幅变动 | 需要计算 Fisher 信息矩阵 |
| **增量学习** | 每阶段只微调少量参数 | 收敛较慢 |
| **多任务学习** | 同时在新旧任务上训练 | 需要维护混合数据集 |
| **数据混合** | 领域数据与通用语料混合（推荐比例 1:1 至 1:3） | 需要保留原始通用数据 |

**生产最佳实践：** SFT 时始终混入约 10–30% 的通用数据（如 WebText、维基百科）。

---

### 故障模式 2：复读机问题

**定义：** LLM 生成重复、循环的文本 — 不断复制相同短语。

**根本原因：**
1. **训练数据偏差** — 语料中含大量重复文档
2. **自回归目标** — 预测下一个 token 会鼓励复制高概率模式
3. **低温度参数** — 确定性解码会反复坍缩到最高概率 token

**缓解策略：**

| 策略 | 方法 |
|------|------|
| **提高温度**（0.7–1.0） | 拉平概率分布，鼓励多样性 |
| **Top-p / nucleus 采样** | 只从覆盖概率质量 p 的 top token 中采样 |
| **重复惩罚** | 将已生成 token 的 logit 乘以 < 1 的惩罚系数 |
| **Beam search 调整** | 更大的 beam width 配合多样性约束 |
| **后处理过滤** | 检测 n-gram 重复并截断或重新生成 |

\`\`\`python
# HuggingFace 生成参数 — 对抗重复
output = model.generate(
    input_ids,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.3,   # > 1.0 惩罚重复
    do_sample=True,
    max_new_tokens=512,
)
\`\`\`

---

### 故障模式 3：训练时 OOM（显存不足）

**定义：** 扩大批次或序列长度时，训练崩溃并报 CUDA OOM 错误。

**诊断清单：**
1. 批次大小是否太大？→ 减半
2. 序列是否太长？→ 截断到 512 或使用滑动窗口
3. 是否在用 FP32？→ 切换到 BF16/FP16
4. 优化器缓存过多？→ 使用 8-bit Adam（bitsandbytes）

**内存降低层次（按顺序应用）：**

\`\`\`
1. FP16/BF16 混合精度             → 内存减少 2×
2. 梯度检查点                     → 用计算换内存（约慢 2×，内存减少 ~4×）
3. 梯度累积步数 = N               → 虚拟大批次，无需额外内存
4. 减少批次大小                   → 直接线性降低
5. 8-bit Adam 优化器              → 优化器状态内存减少 4×
6. LoRA / PEFT                    → 只训练 <1% 的参数
7. CPU offload（DeepSpeed ZeRO-3）→ 将优化器状态卸载到 CPU RAM
\`\`\`

---

### 故障模式 4：长上下文性能退化

**定义：** 当输入长度超过训练时的上下文窗口时，模型性能显著下降。

**原因：**
- 绝对位置编码只在固定长度内训练过
- 注意力是 O(n²) — 长序列会快速耗尽内存
- "Lost in the middle" 现象 — 模型会忽略长输入中间部分的内容

**处理策略：**

| 方法 | 最适合 |
|------|-------|
| **分块 + 摘要** | 超过 10K token 的文档；每块先总结 |
| **层次建模** | 有清晰段落结构的文档 |
| **RAG（检索）** | 只检索最相关的 top-k 块 |
| **滑动窗口注意力** | 局部上下文任务（Longformer、BigBird） |
| **RoPE 缩放**（LLaMA 2+） | 无需重训即可扩展上下文窗口 |

---

### 故障模式 5：领域能力与通用能力的权衡

**定义：** 在领域数据上训练后，模型在领域任务上提升，但通用基准测试下降。

**缓解方案：**
- **CPT 数据比例：** 领域:通用 ≈ 30:70 → 在获取领域知识同时保留通用能力
- **领域适配器（LoRA）：** 保持基础权重冻结；领域知识存在适配器中
- **双模型服务：** 一个通用模型 + 一个领域微调模型；按查询类型路由
`,
  },
}

export const llmInterviewSystemDesign: TopicContent = {
  id: 'llm-interview-system-design',
  title: { en: 'LLM Interview: System Design Layer', zh: 'LLM面试：系统设计层' },
  contentType: 'article',
  content: {
    en: `## The System Design Layer

> Source: *大模型面试题集* — 微调面 + 推理面 + 评测面
> End-to-end pipeline design, inference optimization, and evaluation frameworks.

---

### Design Topic 1: SFT Data Construction Pipeline

**Steps to build a high-quality SFT dataset:**

\`\`\`
1. RAW DATA COLLECTION
   └── Domain docs, web crawl, expert Q&A, synthetic (GPT-4 generated)

2. DATA CLEANING
   ├── Remove duplicates (MinHash / embedding dedup)
   ├── Filter low-quality (perplexity filter, length filter)
   └── PII removal (regex + NER)

3. ANNOTATION
   ├── Human annotation: instruction + reference answer pairs
   ├── Format: {"instruction": "...", "input": "...", "output": "..."}
   └── Quality check: inter-annotator agreement score

4. DATASET SPLITTING
   ├── Train: 80%
   ├── Validation: 10%  (for early stopping, hyperparam tuning)
   └── Test: 10%        (held out, never seen during training)

5. DOMAIN EVALUATION SET
   └── Curated by domain experts; covers edge cases + hard cases
\`\`\`

**Multi-turn conversation format:**
\`\`\`json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is gradient descent?"},
  {"role": "assistant", "content": "Gradient descent is..."},
  {"role": "user", "content": "What about stochastic gradient descent?"}
]
\`\`\`

---

### Design Topic 2: Inference Optimization Stack

**Why VRAM stays high during inference:**
- Model weights permanently resident in GPU memory
- KV-cache grows with each new token (stores past attention keys+values)
- Framework memory allocators hold freed memory as a pool for reuse

**Inference optimization hierarchy:**

| Technique | Speedup | Memory Savings | Quality Impact |
|-----------|---------|---------------|----------------|
| **INT8 quantization** | 1.5–2× | 2× | Minimal |
| **INT4 quantization** | 2–3× | 4× | Small degradation |
| **FP16 inference** | 1.5× | 2× | Negligible |
| **FlashAttention** | 2–4× | 3–5× | None |
| **KV-cache** | 5–10× | N/A (adds mem) | None |
| **Speculative decoding** | 2–3× | None | None |
| **Continuous batching** | Throughput ×3–5 | N/A | None |

**INT8 vs FP16 comparison:**

| | INT8 | FP16 |
|--|------|------|
| Memory per param | 1 byte | 2 bytes |
| Compute | Integer ops (fast on CPU) | Float ops (fast on GPU) |
| Accuracy | Slight drop possible | Near-lossless |
| Best for | CPU inference, edge | GPU inference |

---

### Design Topic 3: Domain Model Training Pipeline

\`\`\`
STAGE 1: Continue Pre-Training (CPT)
  Input:  Domain corpus (e.g., medical papers, legal docs)
  Goal:   Inject domain knowledge into base model
  Config: Lower LR (1e-5 to 5e-5), mix 20-30% general data

STAGE 2: Supervised Fine-Tuning (SFT)
  Input:  Instruction-response pairs in domain
  Goal:   Teach model to follow domain-specific instructions
  Config: Very low LR (1e-6 to 1e-5), small dataset OK

STAGE 3: RLHF / DPO (optional)
  Input:  Preference pairs (chosen vs rejected responses)
  Goal:   Align output style, safety, and quality
  Config: DPO is simpler; RLHF needs reward model + PPO trainer

STAGE 4: Evaluation
  Metrics: Domain-specific benchmark + general benchmark
  Watch:   If general score drops >5%, increase general data mixing
\`\`\`

---

### Design Topic 4: LLM Evaluation Framework

**5 evaluation dimensions:**

| Dimension | What to Measure | Tools |
|-----------|----------------|-------|
| **Fluency** | Grammar, readability, coherence | Perplexity, human eval |
| **Semantic Accuracy** | Is the answer factually correct? | Human eval, QA benchmarks |
| **Context Consistency** | Does long-form output stay consistent? | Human eval |
| **Honesty / Knowledge Boundary** | Does model say "I don't know" when appropriate? | TruthfulQA |
| **Creativity & Diversity** | N-gram diversity, unique responses | Self-BLEU |

**Perplexity (PPL):** Standard automatic metric for language model quality.

\`\`\`
PPL = exp(- (1/N) * Σ log P(token_i | context))
\`\`\`

Lower PPL = model assigns higher probability to real text = better language model.

**The honest principle — how to train a model that knows its knowledge boundary:**
1. Train on factually verified data (Wikipedia, textbooks)
2. Include "I don't know" examples for out-of-scope questions in SFT data
3. Calibrate confidence — use RLHF to reward admitting uncertainty
4. Evaluate with TruthfulQA: measures if model avoids confident wrong answers

---

### Design Topic 5: Vocabulary Expansion for Domain Models

**When is vocabulary expansion necessary?**

| Scenario | Expand? |
|----------|---------|
| Domain has many OOV (out-of-vocabulary) terms | Yes |
| Domain uses specialized notation (medical, legal, code) | Yes |
| Chinese model — domain uses many domain-specific characters | Yes |
| Domain vocabulary mostly overlaps base model's vocab | No |
| Limited compute budget | No — skip expansion, use byte-pair fallback |

**Trade-off:** Adding N new tokens requires re-training the embedding layer (N × hidden_dim params). For a 4096-dim model, adding 10K tokens = 40M new params to randomly initialize → needs more SFT data to train these embeddings.
`,
    zh: `## 系统设计层

> 来源：*大模型面试题集* — 微调面 + 推理面 + 评测面
> 端到端流程设计、推理优化与评测框架。

---

### 设计主题 1：SFT 数据构建流程

**构建高质量 SFT 数据集的步骤：**

\`\`\`
1. 原始数据收集
   └── 领域文档、网络爬取、专家问答、合成数据（GPT-4 生成）

2. 数据清洗
   ├── 去重（MinHash / 向量去重）
   ├── 过滤低质量数据（困惑度过滤、长度过滤）
   └── 删除个人隐私信息（正则 + NER）

3. 标注
   ├── 人工标注：指令 + 参考答案对
   ├── 格式：{"instruction": "...", "input": "...", "output": "..."}
   └── 质量检查：标注者间一致性分数

4. 数据集划分
   ├── 训练集：80%
   ├── 验证集：10%  （早停、超参调整用）
   └── 测试集：10%  （完全保留，训练时不可见）

5. 领域评测集
   └── 由领域专家整理；覆盖边缘案例和难例
\`\`\`

**多轮对话格式：**
\`\`\`json
[
  {"role": "system", "content": "你是一个有帮助的助手。"},
  {"role": "user", "content": "什么是梯度下降？"},
  {"role": "assistant", "content": "梯度下降是..."},
  {"role": "user", "content": "随机梯度下降又是什么？"}
]
\`\`\`

---

### 设计主题 2：推理优化栈

**为什么推理时显存会一直占着：**
- 模型权重永久驻留在 GPU 显存中
- KV 缓存随每个新 token 增长（存储历史注意力的 keys+values）
- 框架内存分配器将释放的内存以池的形式保留，供后续重用

**推理优化层次：**

| 技术 | 加速倍数 | 内存节省 | 质量影响 |
|------|---------|---------|---------|
| **INT8 量化** | 1.5–2× | 2× | 极小 |
| **INT4 量化** | 2–3× | 4× | 轻微下降 |
| **FP16 推理** | 1.5× | 2× | 可忽略 |
| **FlashAttention** | 2–4× | 3–5× | 无 |
| **KV 缓存** | 5–10× | 不适用（增加内存） | 无 |
| **投机解码** | 2–3× | 无 | 无 |
| **连续批处理** | 吞吐量 3–5× | 不适用 | 无 |

---

### 设计主题 3：领域模型训练流程

\`\`\`
阶段 1：持续预训练 (CPT)
  输入：领域语料（如医学论文、法律文件）
  目标：将领域知识注入基础模型
  配置：较低 LR（1e-5 到 5e-5），混入 20-30% 通用数据

阶段 2：有监督微调 (SFT)
  输入：领域内的指令-响应对
  目标：教模型遵循领域特定指令
  配置：极低 LR（1e-6 到 1e-5），数据集小也可以

阶段 3：RLHF / DPO（可选）
  输入：偏好对（chosen vs rejected 响应）
  目标：对齐输出风格、安全性和质量
  配置：DPO 更简单；RLHF 需要奖励模型 + PPO 训练器

阶段 4：评测
  指标：领域专项基准 + 通用基准
  注意：如果通用分数下降 >5%，增加通用数据混合比例
\`\`\`

---

### 设计主题 4：LLM 评测框架

**5 个评测维度：**

| 维度 | 衡量内容 | 工具 |
|------|---------|------|
| **流畅性** | 语法、可读性、连贯性 | 困惑度、人工评估 |
| **语义准确性** | 回答是否事实正确？ | 人工评估、QA 基准 |
| **上下文一致性** | 长篇输出是否保持一致？ | 人工评估 |
| **诚实性/知识边界** | 不知道时是否说"不知道"？ | TruthfulQA |
| **创造性与多样性** | N-gram 多样性、独特响应 | Self-BLEU |

**困惑度（PPL）：** 语言模型质量的标准自动指标。

\`\`\`
PPL = exp(- (1/N) * Σ log P(token_i | 上下文))
\`\`\`

PPL 越低 = 模型对真实文本赋予的概率越高 = 语言模型越好。

**诚实原则 — 如何训练知道知识边界的模型：**
1. 在事实验证的数据上训练（维基百科、教科书）
2. 在 SFT 数据中加入"不在范围内"问题的"我不知道"示例
3. 校准置信度 — 用 RLHF 奖励承认不确定性
4. 用 TruthfulQA 评测：衡量模型是否避免给出自信的错误答案
`,
  },
}

export const llmInterviewCodePatterns: TopicContent = {
  id: 'llm-interview-code-patterns',
  title: { en: 'LLM Interview: Code Snippets & Patterns', zh: 'LLM面试：代码片段与模式' },
  contentType: 'code',
  content: {
    en: `## Code Snippets & Patterns

> Source: *大模型面试题集* — 微调面 + 推理面
> Practical code patterns for LLM fine-tuning, inference optimization, and evaluation.

---

### Pattern 1: LoRA Fine-Tuning (Memory-Efficient SFT)

\`\`\`python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Configure LoRA adapter
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # rank — controls adapter size (8–64 typical)
    lora_alpha=32,      # scaling factor (alpha/r = effective LR scale)
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # inject into attention Q and V
    bias="none",
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# 4. SFT Training
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 4*4 = 16
    fp16=True,
    learning_rate=2e-4,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
\`\`\`

---

### Pattern 2: VRAM Estimation Formula

\`\`\`python
def estimate_vram(
    num_params_billion: float,
    dtype_bytes: int = 2,       # 2 = fp16/bf16, 4 = fp32, 1 = int8
    batch_size: int = 1,
    seq_len: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    training: bool = False,
) -> dict:
    """
    Estimate GPU VRAM requirements in GB.
    """
    # Model weights
    model_gb = num_params_billion * 1e9 * dtype_bytes / (1024 ** 3)

    # KV cache (inference) or activations (training)
    if training:
        # Optimizer state (Adam: 2x model), gradients (1x model)
        optimizer_gb = model_gb * 2  # Adam momentum + variance
        gradient_gb = model_gb
        # Activation memory (rough: batch * seq * hidden * layers * 2 bytes)
        activation_gb = (batch_size * seq_len * hidden_size * num_layers * 2) / (1024 ** 3)
        total_gb = model_gb + optimizer_gb + gradient_gb + activation_gb
        return {
            "weights": round(model_gb, 1),
            "optimizer_state": round(optimizer_gb, 1),
            "gradients": round(gradient_gb, 1),
            "activations": round(activation_gb, 1),
            "TOTAL_GB": round(total_gb, 1),
        }
    else:
        # KV cache: 2 (K+V) * layers * seq_len * hidden * dtype_bytes * batch
        kv_cache_gb = (2 * num_layers * seq_len * hidden_size * dtype_bytes * batch_size) / (1024 ** 3)
        total_gb = model_gb + kv_cache_gb
        return {
            "weights": round(model_gb, 1),
            "kv_cache": round(kv_cache_gb, 1),
            "TOTAL_GB": round(total_gb, 1),
        }

# Example: 7B model, FP16, inference
print(estimate_vram(7, dtype_bytes=2, training=False))
# {'weights': 13.0, 'kv_cache': 2.1, 'TOTAL_GB': 15.1}

# Example: 7B model, FP16, training with Adam
print(estimate_vram(7, dtype_bytes=2, training=True))
# {'weights': 13.0, 'optimizer_state': 26.0, 'gradients': 13.0, 'activations': 4.3, 'TOTAL_GB': 56.3}
\`\`\`

---

### Pattern 3: Gradient Accumulation for Memory-Efficient Training

\`\`\`python
from torch.optim import AdamW

model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)

accumulation_steps = 8   # effective batch = micro_batch * accumulation_steps

for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # normalize loss

    # Backward pass (accumulate gradients)
    loss.backward()

    # Only update weights every N steps
    if (step + 1) % accumulation_steps == 0:
        # Optional: clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
\`\`\`

---

### Pattern 4: Generation Parameter Tuning Cheat Sheet

\`\`\`python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# CONSERVATIVE: Factual, consistent outputs
factual_output = generator(
    prompt,
    max_new_tokens=256,
    temperature=0.3,       # low = more deterministic
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

# CREATIVE: Diverse, imaginative outputs
creative_output = generator(
    prompt,
    max_new_tokens=512,
    temperature=1.0,       # high = more random
    top_p=0.95,
    top_k=50,              # also sample from top-50 vocab
    repetition_penalty=1.3,
    do_sample=True,
)

# PARAMETER REFERENCE TABLE
params = {
    "temperature":        "0.1=deterministic, 0.7=balanced, 1.0=creative, >1.2=chaotic",
    "top_p":              "0.9-0.95 typical; lower = more conservative vocab",
    "top_k":              "0=disabled, 50=moderate, 10=restrictive",
    "repetition_penalty": "1.0=none, 1.1-1.3=light penalty, >1.5=aggressive",
    "max_new_tokens":     "Hard cap on output length; independent of input",
}
\`\`\`

---

### Pattern 5: EWC (Elastic Weight Consolidation) — Anti-Forgetting

\`\`\`python
import torch

class EWCTrainer:
    """
    Elastic Weight Consolidation: regularize against forgetting old tasks.
    Adds penalty term: lambda * sum(F_i * (theta_i - theta_old_i)^2)
    where F_i = Fisher information (importance of parameter i for old task)
    """
    def __init__(self, model, old_dataloader, ewc_lambda=1000):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.old_params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(old_dataloader)

    def _compute_fisher(self, dataloader):
        """Estimate Fisher information matrix (diagonal approximation)."""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()

        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(**batch)
            loss = output.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2  # squared gradient ≈ Fisher diag

        # Normalize by number of batches
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def ewc_loss(self):
        """Compute EWC penalty term."""
        penalty = 0.0
        for n, p in self.model.named_parameters():
            penalty += (self.fisher[n] * (p - self.old_params[n]) ** 2).sum()
        return self.ewc_lambda * penalty

    def training_step(self, new_batch):
        """Combined loss: task loss + EWC penalty."""
        output = self.model(**new_batch)
        task_loss = output.loss
        total_loss = task_loss + self.ewc_loss()
        return total_loss
\`\`\`

---

### Pattern 6: Simple LLM Evaluation Script

\`\`\`python
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """
    Compute average perplexity over a list of texts.
    Lower = better language model.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]

            outputs = model(**inputs, labels=input_ids)
            # outputs.loss = mean NLL per token
            nll = outputs.loss.item()
            n_tokens = input_ids.shape[1]

            total_nll += nll * n_tokens
            total_tokens += n_tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl

def evaluate_repetition(text: str, n: int = 4) -> float:
    """
    Measure n-gram repetition rate. Higher = more repetitive.
    Returns fraction of repeated n-grams.
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique = len(set(ngrams))
    total = len(ngrams)
    repetition_rate = 1 - (unique / total)
    return repetition_rate

# Usage
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

test_texts = ["The quick brown fox...", "Machine learning is..."]
ppl = compute_perplexity(model, tokenizer, test_texts)
print(f"Perplexity: {ppl:.2f}")  # e.g., Perplexity: 42.31

sample_output = "The model said the model said the model said the answer"
rep_rate = evaluate_repetition(sample_output, n=3)
print(f"Repetition rate: {rep_rate:.2%}")  # e.g., Repetition rate: 75.00%
\`\`\`
`,
    zh: `## 代码片段与模式

> 来源：*大模型面试题集* — 微调面 + 推理面
> LLM 微调、推理优化和评测的实用代码模式。

---

### 模式 1：LoRA 微调（内存高效的 SFT）

\`\`\`python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. 配置 LoRA 适配器
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # 秩 — 控制适配器大小（典型值 8–64）
    lora_alpha=32,      # 缩放因子（alpha/r = 有效 LR 缩放）
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 注入到注意力 Q 和 V
    bias="none",
)

# 3. 将 LoRA 应用到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# 4. SFT 训练
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # 有效批次 = 4*4 = 16
    fp16=True,
    learning_rate=2e-4,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
\`\`\`

---

### 模式 2：显存估算公式

\`\`\`python
def estimate_vram(
    num_params_billion: float,
    dtype_bytes: int = 2,       # 2 = fp16/bf16, 4 = fp32, 1 = int8
    batch_size: int = 1,
    seq_len: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    training: bool = False,
) -> dict:
    """估算 GPU 显存需求（GB）"""
    model_gb = num_params_billion * 1e9 * dtype_bytes / (1024 ** 3)

    if training:
        optimizer_gb = model_gb * 2  # Adam：动量 + 方差
        gradient_gb = model_gb
        activation_gb = (batch_size * seq_len * hidden_size * num_layers * 2) / (1024 ** 3)
        total_gb = model_gb + optimizer_gb + gradient_gb + activation_gb
        return {
            "权重": round(model_gb, 1),
            "优化器状态": round(optimizer_gb, 1),
            "梯度": round(gradient_gb, 1),
            "激活值": round(activation_gb, 1),
            "总计_GB": round(total_gb, 1),
        }
    else:
        kv_cache_gb = (2 * num_layers * seq_len * hidden_size * dtype_bytes * batch_size) / (1024 ** 3)
        total_gb = model_gb + kv_cache_gb
        return {
            "权重": round(model_gb, 1),
            "KV缓存": round(kv_cache_gb, 1),
            "总计_GB": round(total_gb, 1),
        }

# 示例：7B 模型，FP16，推理
print(estimate_vram(7, dtype_bytes=2, training=False))
# {'权重': 13.0, 'KV缓存': 2.1, '总计_GB': 15.1}

# 示例：7B 模型，FP16，Adam 训练
print(estimate_vram(7, dtype_bytes=2, training=True))
# {'权重': 13.0, '优化器状态': 26.0, '梯度': 13.0, '激活值': 4.3, '总计_GB': 56.3}
\`\`\`

---

### 模式 3：梯度累积（内存高效训练）

\`\`\`python
accumulation_steps = 8   # 有效批次 = micro_batch * accumulation_steps

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # 归一化损失
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
\`\`\`

---

### 模式 4：生成参数调优速查表

\`\`\`python
# 保守型：事实性、一致的输出
factual_config = dict(
    max_new_tokens=256,
    temperature=0.3,       # 低 = 更确定性
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

# 创意型：多样化、有想象力的输出
creative_config = dict(
    max_new_tokens=512,
    temperature=1.0,       # 高 = 更随机
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.3,
    do_sample=True,
)

# 参数参考表
# temperature: 0.1=确定性, 0.7=平衡, 1.0=创意, >1.2=混乱
# top_p:       0.9-0.95 典型值; 越低词汇越保守
# repetition_penalty: 1.0=无, 1.1-1.3=轻度惩罚, >1.5=强力惩罚
\`\`\`

---

### 模式 5：困惑度与重复率评测

\`\`\`python
import math
import torch

def compute_perplexity(model, tokenizer, texts: list) -> float:
    """计算平均困惑度。越低 = 语言模型越好。"""
    model.eval()
    total_nll, total_tokens = 0.0, 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            outputs = model(**inputs, labels=input_ids)
            total_nll += outputs.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    return math.exp(total_nll / total_tokens)

def repetition_rate(text: str, n: int = 4) -> float:
    """衡量 n-gram 重复率。越高 = 复读机问题越严重。"""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams)
\`\`\`
`,
  },
}
