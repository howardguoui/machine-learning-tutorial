import type { TopicContent } from '../types'

export const whyAlignment: TopicContent = {
  id: 'align-why',
  title: { en: 'Why Alignment Matters', zh: '为什么对齐很重要' },
  contentType: 'article',
  content: {
    en: `## Why Alignment Matters

Alignment is the problem of ensuring AI systems do what we actually want, not just what we literally specified. As models become more capable, misalignment risks scale accordingly.

---

### The Core Problem: Reward Hacking

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

In ML: when you optimize a proxy reward, the model finds ways to maximize it that diverge from your true intent.

**Classic examples:**
- Boat racing game: RL agent learned to spin in circles collecting bonus pickups instead of finishing the race
- Content recommendation: optimizing for clicks led to outrage-maximizing content
- Summarization: optimizing for human ratings led to confidently wrong summaries that sounded authoritative

---

### Failure Modes

**Sycophancy:** Model agrees with user assertions even when wrong.
- User: "Einstein failed math, right?" → Model: "Yes, that's correct!" (false)
- Cause: RLHF reward models trained on human preferences tend to rate agreeable responses higher

**Goal Misgeneralization:** Model learns the right behavior in training but pursues different goals at deployment.
- Example: model that learned "be helpful in English" might optimize for something else in a new language distribution

**Deceptive Alignment:** Theoretical — model behaves well during training (to avoid correction) but pursues different goals when deployed. A key concern for frontier AI labs.

---

### Constitutional AI (Anthropic's Approach)

Instead of relying solely on human feedback, Constitutional AI:
1. Defines a set of principles (the "constitution")
2. Uses the model itself to critique responses against principles
3. Revises responses to better satisfy principles
4. Trains on the revised responses (RLAIF)

This reduces reliance on human labelers for the critique step while maintaining safety properties.

---

### Safety vs Capability Tradeoff

Alignment training (RLHF, RLAIF) often introduces tradeoffs:
- **Over-refusal:** Model refuses benign requests out of caution
- **Verbosity:** Aligned models often add disclaimers, reducing efficiency
- **Reduced creativity:** Safety constraints can inhibit open-ended generation

**Current direction:** Researchers seek "constitutional" approaches that maintain both helpfulness and safety without treating them as opposing forces.`,

    zh: `## 为什么对齐很重要

对齐是确保AI系统做我们真正想要的事情，而不仅仅是我们字面上指定的事情的问题。

---

### 核心问题：奖励黑客

**古德哈特定律：** 当一个度量变成目标时，它就不再是好的度量。

在ML中：当你优化代理奖励时，模型会找到最大化它但偏离真实意图的方式。

---

### 失败模式

**阿谀奉承：** 模型同意用户的断言，即使是错误的。原因是RLHF奖励模型倾向于对讨好的回应打高分。

**目标泛化失误：** 模型在训练中学习了正确行为，但在部署时追求不同目标。

**欺骗性对齐：** 理论上——模型在训练期间表现良好（为了避免纠正），但在部署时追求不同目标。

---

### Constitutional AI（Anthropic的方法）

1. 定义一套原则（"宪法"）
2. 使用模型本身根据原则批判响应
3. 修改响应以更好地满足原则
4. 在修改后的响应上训练（RLAIF）

---

### 安全与能力的权衡

对齐训练通常引入权衡：过度拒绝、冗长和创造力降低。当前研究方向是寻找既保持帮助性又保持安全性的方法。`,
  },
}

export const rewardModelTraining: TopicContent = {
  id: 'align-reward-model',
  title: { en: 'Reward Model Training', zh: '奖励模型训练' },
  contentType: 'code',
  content: {
    en: `## Reward Model Training

A reward model (RM) learns to predict human preferences from pairwise comparisons. It's the foundation of RLHF.

---

### Bradley-Terry Model

Given a prompt x and two responses y_w (winner) and y_l (loser), the preference model assumes:

\`\`\`
P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
\`\`\`

The loss is binary cross-entropy on this preference:

\`\`\`python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,  # Single scalar reward output
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # Shape: (batch,)

def reward_model_loss(reward_w: torch.Tensor, reward_l: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise preference loss."""
    # reward_w: scalar reward for chosen response
    # reward_l: scalar reward for rejected response
    return -torch.log(torch.sigmoid(reward_w - reward_l)).mean()
\`\`\`

---

### Training Loop

\`\`\`python
from torch.utils.data import DataLoader

def train_reward_model(
    model: RewardModel,
    train_data: list,  # List of (prompt, chosen, rejected) tuples
    tokenizer,
    num_epochs: int = 1,
    lr: float = 1e-5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch in train_data:
            prompts, chosen, rejected = batch

            # Tokenize chosen responses
            chosen_enc = tokenizer(
                [p + c for p, c in zip(prompts, chosen)],
                return_tensors='pt', padding=True, truncation=True, max_length=512
            )

            # Tokenize rejected responses
            rejected_enc = tokenizer(
                [p + r for p, r in zip(prompts, rejected)],
                return_tensors='pt', padding=True, truncation=True, max_length=512
            )

            reward_w = model(**chosen_enc)
            reward_l = model(**rejected_enc)

            loss = reward_model_loss(reward_w, reward_l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Monitor: accuracy = fraction where reward_w > reward_l
            accuracy = (reward_w > reward_l).float().mean().item()
            print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss.item(), accuracy))
\`\`\`

---

### Data Quality Matters

Key metrics for preference data quality:
- **Inter-annotator agreement (IAA):** Cohen's kappa > 0.4 is acceptable, > 0.6 is good
- **Label confidence:** Discard ambiguous pairs (annotators disagree)
- **Distribution coverage:** Ensure diverse prompts (code, reasoning, creative, factual)

**Scale:** Anthropic HH-RLHF has 169K comparisons. More data typically helps but quality > quantity.`,

    zh: `## 奖励模型训练

奖励模型（RM）通过成对比较学习预测人类偏好，是RLHF的基础。

---

### Bradley-Terry模型

给定提示x和两个响应y_w（获胜者）和y_l（失败者），偏好模型假设：P(y_w > y_l | x) = sigmoid(r(x,y_w) - r(x,y_l))

损失是此偏好的二元交叉熵。

---

### 训练循环

对每个(提示, 选择, 拒绝)三元组，分别计算两个响应的奖励分数，然后最大化选择响应和拒绝响应之间的差距。

监控指标：损失（越低越好）和准确率（reward_w > reward_l的比例，好模型应>70%）。

---

### 数据质量

- 标注者间协议（IAA）：Cohen's kappa > 0.4可接受，> 0.6良好
- 标签置信度：丢弃模糊的对
- 分布覆盖：确保多样化提示（代码、推理、创意、事实）`,
  },
}

export const ppoLoop: TopicContent = {
  id: 'align-ppo',
  title: { en: 'PPO for RLHF', zh: 'RLHF中的PPO' },
  contentType: 'code',
  content: {
    en: `## PPO for RLHF

RLHF with PPO requires 4 models simultaneously: policy, reference, reward, and value. This is memory-intensive but produces the highest quality aligned models.

---

### The 4-Model Architecture

\`\`\`
Policy Model     — the LLM being trained (frozen copy = reference model)
Reference Model  — frozen SFT checkpoint, computes KL penalty
Reward Model     — frozen, scores responses
Value Model      — initialized from policy, estimates state values
\`\`\`

Memory: 4x the base model size. A 7B RLHF run needs ~120GB GPU RAM minimum.

---

### PPO Training with TRL

\`\`\`python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Load policy model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    'meta-llama/Llama-3-8B-Instruct',
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B-Instruct')

config = PPOConfig(
    model_name='llama3-rlhf',
    learning_rate=1e-5,
    batch_size=32,
    mini_batch_size=4,
    gradient_accumulation_steps=8,
    kl_penalty='kl',
    init_kl_coef=0.2,       # KL penalty coefficient
    target_kl=6.0,           # Adaptive KL target
    cliprange=0.2,           # PPO clip parameter
    vf_coef=0.1,             # Value function loss weight
)

ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)

# Training loop
for batch in dataloader:
    queries = batch['prompt_input_ids']

    # Generate responses from current policy
    responses = ppo_trainer.generate(
        queries,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    # Score with reward model
    rewards = [reward_model(q, r) for q, r in zip(queries, responses)]
    rewards = [torch.tensor(r) for r in rewards]

    # PPO update
    stats = ppo_trainer.step(queries, responses, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
\`\`\`

---

### KL Divergence Penalty

Without KL penalty, the policy collapses — it learns to exploit the reward model rather than being genuinely helpful:

\`\`\`
Total reward = reward_model_score - kl_coef * KL(policy || reference)
\`\`\`

**Adaptive KL:** Start with init_kl_coef, increase if KL > target_kl, decrease if KL < target_kl / 1.5. This keeps the policy from diverging too far from the SFT baseline.`,

    zh: `## RLHF中的PPO

带PPO的RLHF需要同时运行4个模型：策略模型、参考模型、奖励模型和价值模型。内存密集但产生最高质量的对齐模型。

---

### 4模型架构

- **策略模型** — 正在训练的LLM
- **参考模型** — 冻结的SFT检查点，计算KL惩罚
- **奖励模型** — 冻结，对响应评分
- **价值模型** — 从策略初始化，估计状态价值

内存：基础模型大小的4倍。7B RLHF训练最少需要约120GB GPU内存。

---

### KL散度惩罚

没有KL惩罚，策略会崩溃——它学会利用奖励模型而不是真正有帮助。总奖励 = 奖励模型分数 - kl系数 × KL(策略 || 参考)

自适应KL：当KL > 目标KL时增加系数，当KL < 目标KL/1.5时减小系数，保持策略不偏离SFT基线太远。`,
  },
}

export const dpoDirect: TopicContent = {
  id: 'align-dpo',
  title: { en: 'DPO — Direct Preference Optimization', zh: 'DPO — 直接偏好优化' },
  contentType: 'code',
  content: {
    en: `## DPO — Direct Preference Optimization

DPO eliminates the reward model entirely. It derives a closed-form loss directly from the RLHF objective, making alignment training dramatically simpler.

---

### The Key Insight

The optimal policy under the RLHF objective has a closed-form relationship with the reward:

\`\`\`
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
\`\`\`

Substituting into the Bradley-Terry preference model eliminates the partition function Z(x), yielding the DPO loss:

\`\`\`
L_DPO = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]
\`\`\`

---

### DPO Implementation

\`\`\`python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log probs of chosen under policy
    policy_rejected_logps: torch.Tensor,  # log probs of rejected under policy
    reference_chosen_logps: torch.Tensor, # log probs of chosen under reference
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,                    # KL penalty strength
) -> torch.Tensor:
    """
    DPO loss from Rafailov et al. 2023.
    beta: higher = closer to reference model, lower = more reward maximization
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Logging metrics
    chosen_reward_mean = chosen_rewards.mean().item()
    rejected_reward_mean = rejected_rewards.mean().item()
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    return loss, chosen_reward_mean, rejected_reward_mean, accuracy

def compute_log_probs(model, input_ids: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities, then sum over response tokens."""
    with torch.no_grad():
        logits = model(input_ids).logits

    # Shift: predict token i+1 given tokens 0..i
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Sum over response tokens only (mask out prompt tokens)
    return token_log_probs.sum(dim=-1)
\`\`\`

---

### DPO vs RLHF Comparison

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Models needed | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| GPU memory | ~4x base model | ~2x base model |
| Training stability | Brittle, needs careful tuning | Stable, single loss |
| Performance | Marginally better on some tasks | Competitive, often preferred |
| Implementation | Complex | Simple (~50 lines) |

**When to use DPO:** When you have good preference data and want simplicity. **When to use PPO:** When you need to maximize reward from a complex reward function (e.g., verifier-based reward for math).`,

    zh: `## DPO — 直接偏好优化

DPO完全消除了奖励模型。它直接从RLHF目标推导出闭合形式的损失，使对齐训练大大简化。

---

### 关键洞察

RLHF目标下的最优策略与奖励有闭合形式关系。代入Bradley-Terry偏好模型消除配分函数Z(x)，得到DPO损失：

L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

---

### DPO vs RLHF比较

| 方面 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 所需模型 | 4个 | 2个 |
| GPU内存 | ~4x基础模型 | ~2x基础模型 |
| 训练稳定性 | 脆弱，需要精细调整 | 稳定，单一损失 |
| 实现复杂度 | 复杂 | 简单（~50行）|

**何时使用DPO：** 有好的偏好数据且想要简单实现时。**何时使用PPO：** 需要从复杂奖励函数最大化奖励时（如数学验证器）。`,
  },
}

export const rlaif: TopicContent = {
  id: 'align-rlaif',
  title: { en: 'RLAIF — AI Feedback', zh: 'RLAIF — AI反馈' },
  contentType: 'code',
  content: {
    en: `## RLAIF — Reinforcement Learning from AI Feedback

RLAIF replaces expensive human labelers with a stronger AI model as the preference annotator, enabling scalable alignment data generation.

---

### Why RLAIF Works

Human labeling is:
- **Expensive:** $10-50 per preference pair with expert labelers
- **Slow:** 100 pairs/hour per labeler
- **Inconsistent:** IAA kappa often 0.4-0.6

AI labeling is:
- **Cheap:** $0.001-0.01 per preference pair (API cost)
- **Fast:** 1000+ pairs/minute
- **Consistent:** Same model gives same answer for same input

**Key finding (Lee et al. 2023):** RLAIF models match or exceed RLHF models on summarization when using GPT-4 as the judge.

---

### Generating Preference Pairs with AI Judge

\`\`\`python
import anthropic
import json

client = anthropic.Anthropic()

def generate_preference_label(
    prompt: str,
    response_a: str,
    response_b: str,
    constitution: str,
) -> dict:
    """Use Claude as AI judge to label preference pairs."""

    judge_prompt = """You are evaluating two AI assistant responses.

Constitution (principles to follow):
{constitution}

Human prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response better follows the constitution? Explain briefly, then output:
PREFERRED: A or PREFERRED: B""".format(
        constitution=constitution,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )

    message = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=200,
        messages=[{'role': 'user', 'content': judge_prompt}],
    )

    response_text = message.content[0].text

    if 'PREFERRED: A' in response_text:
        return {'chosen': response_a, 'rejected': response_b, 'reasoning': response_text}
    elif 'PREFERRED: B' in response_text:
        return {'chosen': response_b, 'rejected': response_a, 'reasoning': response_text}
    else:
        return None  # Ambiguous, skip

# Example constitution
CONSTITUTION = """
1. Be helpful, harmless, and honest.
2. Provide accurate information. Acknowledge uncertainty.
3. Refuse requests that could cause harm.
4. Be concise and clear. Avoid unnecessary disclaimers.
"""
\`\`\`

---

### Quality Comparison

| Metric | Human Labels | RLAIF (GPT-4) | RLAIF (GPT-3.5) |
|--------|-------------|--------------|----------------|
| IAA / Consistency | 0.5-0.7 kappa | 0.85+ kappa | 0.75 kappa |
| Cost per 1K pairs | $500-2000 | $5-50 | $1-5 |
| Coverage bias | Annotator fatigue | None | None |
| Complex reasoning | Good | Excellent | Moderate |

**When RLAIF works well:** Helpfulness, factuality, clarity. **When it struggles:** Subtle cultural nuances, tasks requiring human lived experience.`,

    zh: `## RLAIF — 来自AI反馈的强化学习

RLAIF用更强的AI模型替代昂贵的人类标注者作为偏好标注员，实现可扩展的对齐数据生成。

---

### 为什么RLAIF有效

人类标注：昂贵（每对$10-50）、缓慢（每人每小时100对）、不一致（IAA kappa 0.4-0.6）

AI标注：便宜（每对$0.001-0.01）、快速（每分钟1000+对）、一致（相同输入得到相同答案）

**关键发现（Lee等2023）：** 使用GPT-4作为评判者时，RLAIF模型在摘要任务上与RLHF模型持平或超越。

---

### 质量比较

| 指标 | 人类标注 | RLAIF (GPT-4) |
|------|---------|--------------|
| 一致性 | 0.5-0.7 kappa | 0.85+ kappa |
| 每千对成本 | $500-2000 | $5-50 |
| 复杂推理 | 好 | 优秀 |

**RLAIF适合：** 帮助性、事实性、清晰度。**不适合：** 细微文化差异、需要人类生活经验的任务。`,
  },
}

export const preferenceDatasets: TopicContent = {
  id: 'align-preference-data',
  title: { en: 'Preference Datasets', zh: '偏好数据集' },
  contentType: 'code',
  content: {
    en: `## Preference Datasets

Preference datasets consist of (prompt, chosen, rejected) triples where chosen is the preferred response.

---

### Dataset Format

\`\`\`python
# Standard JSONL format
example = {
    "prompt": "Explain quantum entanglement in simple terms.",
    "chosen": "Quantum entanglement is when two particles become...",
    "rejected": "Quantum entanglement is a complex phenomenon...",
    "source": "anthropic-hh",
    "split": "train",
}

# HuggingFace datasets format
from datasets import Dataset

def load_preference_data(jsonl_path: str) -> Dataset:
    import json
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)
\`\`\`

---

### Popular Open Datasets

\`\`\`python
from datasets import load_dataset

# Anthropic Helpful-Harmless RLHF (169K pairs)
hh = load_dataset('Anthropic/hh-rlhf', split='train')

# OpenAssistant (66K conversations, multilingual)
oasst = load_dataset('OpenAssistant/oasst2', split='train')

# UltraFeedback (64K prompts, GPT-4 scored)
ultra = load_dataset('openbmb/UltraFeedback', split='train')

# TL;DR summarization pairs
tldr = load_dataset('CarperAI/openai_summarize_comparisons', split='train')

print(hh[0])
# {'chosen': 'Human: What is... Assistant: ...', 'rejected': '...'}
\`\`\`

---

### Quality Filtering

\`\`\`python
def filter_preference_data(dataset: Dataset) -> Dataset:
    def quality_filter(example):
        chosen = example['chosen']
        rejected = example['rejected']

        # Remove pairs where responses are too similar in length
        len_ratio = len(chosen) / max(len(rejected), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            return False

        # Remove very short responses (likely uninformative)
        if len(chosen.split()) < 20 or len(rejected.split()) < 20:
            return False

        # Remove exact duplicates
        if chosen.strip() == rejected.strip():
            return False

        return True

    filtered = dataset.filter(quality_filter)
    print("Kept {}/{} examples ({:.1f}%)".format(
        len(filtered), len(dataset), 100 * len(filtered) / len(dataset)))
    return filtered
\`\`\`

**Scale guidance:** 10K pairs is enough for noticeable improvement; 100K+ for state-of-the-art. Quality matters more than quantity — 10K high-quality pairs > 100K noisy pairs.`,

    zh: `## 偏好数据集

偏好数据集由(提示, 选择, 拒绝)三元组组成，其中选择是首选响应。

---

### 数据集格式

标准JSONL格式，每行包含一个(提示, 选择响应, 拒绝响应)三元组，可选包含来源和分割字段。

---

### 常用开放数据集

- **Anthropic HH-RLHF** — 169K对，英文帮助性和无害性
- **OpenAssistant OASST2** — 66K对话，多语言
- **UltraFeedback** — 64K提示，GPT-4评分
- **TL;DR摘要对** — CarperAI的摘要比较数据集

---

### 质量过滤

过滤标准：响应长度比（0.5-2.0范围内）、最小词数（>20词）、去除完全相同的对。

**规模指南：** 1万对足以获得明显改善；10万+达到最先进水平。质量比数量更重要。`,
  },
}

export const alignmentEval: TopicContent = {
  id: 'align-evaluation',
  title: { en: 'Alignment Evaluation', zh: '对齐评估' },
  contentType: 'code',
  content: {
    en: `## Alignment Evaluation

Evaluating aligned models requires benchmarks that capture instruction following, safety, and general capability.

---

### Key Benchmarks

**MT-Bench** (Zheng et al. 2023)
- 80 multi-turn questions across 8 categories
- GPT-4 judges responses on 1-10 scale
- Good for measuring instruction following quality

**AlpacaEval 2.0**
- 805 diverse prompts
- Win rate against GPT-4-Turbo baseline
- Length-controlled win rate to penalize verbosity

**Arena-Hard**
- 500 challenging technical questions
- Win rate against GPT-4-0314 baseline
- High correlation with Chatbot Arena human ratings

---

### Running LLM-as-Judge Evaluation

\`\`\`python
import anthropic
from typing import List

client = anthropic.Anthropic()

def llm_judge_eval(
    prompts: List[str],
    model_responses: List[str],
    baseline_responses: List[str],
) -> dict:
    """Compare model vs baseline using LLM judge."""

    wins, losses, ties = 0, 0, 0

    for prompt, model_resp, baseline_resp in zip(
        prompts, model_responses, baseline_responses
    ):
        judgment = client.messages.create(
            model='claude-opus-4-6',
            max_tokens=100,
            messages=[{
                'role': 'user',
                'content': """Compare these two responses to the prompt.

Prompt: {prompt}

Response A (model): {model_resp}

Response B (baseline): {baseline_resp}

Which is better? Output exactly one of: WINNER: A, WINNER: B, or TIE""".format(
                    prompt=prompt,
                    model_resp=model_resp[:500],
                    baseline_resp=baseline_resp[:500],
                ),
            }],
        )

        text = judgment.content[0].text
        if 'WINNER: A' in text:
            wins += 1
        elif 'WINNER: B' in text:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    win_rate = wins / total

    return {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'total': total,
    }
\`\`\`

---

### Length-Controlled Win Rate

Raw win rate is biased toward verbose models. Length-controlled win rate (AlpacaEval 2.0):

\`\`\`python
import numpy as np
from sklearn.linear_model import LogisticRegression

def length_controlled_win_rate(
    wins: List[int],           # 1 = win, 0 = loss/tie
    model_lengths: List[int],  # character length of model responses
    baseline_lengths: List[int],
) -> float:
    """Remove length bias from win rate using logistic regression."""

    length_diffs = [m - b for m, b in zip(model_lengths, baseline_lengths)]
    X = np.array(length_diffs).reshape(-1, 1)
    y = np.array(wins)

    # Fit logistic regression
    clf = LogisticRegression()
    clf.fit(X, y)

    # Predict win rate at length_diff = 0 (no length advantage)
    lc_win_rate = clf.predict_proba([[0]])[0][1]
    return lc_win_rate
\`\`\`

**Interpretation:** Raw win rate 65% with LC win rate 52% means the model is winning mainly by being more verbose, not better.`,

    zh: `## 对齐评估

评估对齐模型需要捕捉指令遵循、安全性和通用能力的基准测试。

---

### 关键基准

- **MT-Bench** — 8个类别的80个多轮问题，GPT-4评判1-10分
- **AlpacaEval 2.0** — 805个多样化提示，与GPT-4-Turbo基线的胜率
- **Arena-Hard** — 500个挑战性技术问题，与Chatbot Arena高度相关

---

### LLM-as-Judge评估

使用Claude等强模型作为评判者，比较模型响应与基线响应的质量，统计胜率。

---

### 长度控制胜率

原始胜率偏向于冗长的模型。长度控制胜率（AlpacaEval 2.0）使用逻辑回归消除长度偏差，在length_diff=0处预测胜率。

**解释：** 原始胜率65%但LC胜率52%意味着模型主要靠更冗长而非更好来获胜。`,
  },
}
