import type { TopicContent } from '../types'

export const corpusPreprocessing: TopicContent = {
  id: 'pretrain-corpus',
  title: { en: 'Corpus Assembly & Preprocessing', zh: '语料库构建与预处理' },
  contentType: 'code',
  content: {
    en: `## Corpus Assembly & Preprocessing

Before training, you need clean, diverse, deduplicated text at scale. Quality beats quantity — a 100B token high-quality corpus outperforms a 1T token noisy one.

---

### Step 1: Source Selection

Good pretraining data mixes:
- **Web text** (CommonCrawl, C4, RefinedWeb) — scale but noisy
- **Books** (BookCorpus, Project Gutenberg) — long-range coherence
- **Code** (GitHub, The Stack) — structured reasoning
- **Academic** (ArXiv, PubMed) — technical depth
- **Wikipedia** — factual, clean

---

### Step 2: Text Cleaning

\`\`\`python
import re
import unicodedata

def clean_text(text: str) -> str:
    # Normalize unicode (NFC form)
    text = unicodedata.normalize('NFC', text)

    # Remove control characters except newlines/tabs
    text = re.sub(r'[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f\\x7f]', '', text)

    # Collapse excessive whitespace
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove boilerplate patterns
    text = re.sub(r'Click here to.*?\\n', '', text, flags=re.IGNORECASE)

    return text.strip()
\`\`\`

---

### Step 3: Deduplication (Critical)

Near-duplicate removal with MinHash:

\`\`\`python
from datasketch import MinHash, MinHashLSH

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Shingling: slide a 3-word window
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        m.update(shingle.encode('utf8'))
    return m

def deduplicate_corpus(documents: list) -> list:
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    unique_docs = []

    for i, doc in enumerate(documents):
        mh = get_minhash(doc['text'])
        key = str(i)

        # Check if near-duplicate exists
        result = lsh.query(mh)
        if not result:
            lsh.insert(key, mh)
            unique_docs.append(doc)

    return unique_docs
\`\`\`

---

### Step 4: Quality Scoring

\`\`\`python
def quality_score(text: str) -> float:
    score = 1.0
    words = text.split()

    # Too short or too long
    if len(words) < 50 or len(words) > 100_000:
        score *= 0.5

    # High punctuation ratio (spam indicator)
    punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if punct_ratio > 0.3:
        score *= 0.3

    # Average word length (gibberish detection)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word_len < 3 or avg_word_len > 10:
        score *= 0.5

    return score
\`\`\`

At 1T tokens with 512-token sequences, you have ~2B training examples. Deduplication typically removes 20-40% of web data but dramatically improves downstream performance.`,

    zh: `## 语料库构建与预处理

训练前需要大规模的干净、多样且去重的文本。质量胜于数量——1000亿高质量token的语料库优于1万亿嘈杂的token。

---

### 第一步：数据源选择

优质预训练数据混合：
- **网络文本**（CommonCrawl、C4、RefinedWeb）——规模大但噪声多
- **书籍**（BookCorpus、古腾堡计划）——长程连贯性
- **代码**（GitHub、The Stack）——结构化推理
- **学术**（ArXiv、PubMed）——技术深度
- **维基百科**——事实准确、干净

---

### 第二步：文本清洗

\`\`\`python
import re
import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f\\x7f]', '', text)
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
\`\`\`

---

### 第三步：去重（关键步骤）

使用MinHash进行近似重复删除，阈值0.8时通常去除网络数据的20-40%，但显著提升下游性能。

---

### 第四步：质量评分

通过词数、标点比例、平均词长等启发式指标过滤低质量文档，确保语料库质量。`,
  },
}

export const bpeTokenizer: TopicContent = {
  id: 'pretrain-bpe',
  title: { en: 'BPE Tokenizer from Scratch', zh: '从零实现BPE分词器' },
  contentType: 'code',
  content: {
    en: `## BPE Tokenizer from Scratch

Byte-Pair Encoding (BPE) starts with individual bytes/characters and iteratively merges the most frequent pair until reaching vocabulary size V.

---

### The BPE Algorithm

\`\`\`python
from collections import Counter, defaultdict

def get_vocab(corpus: list[str]) -> dict:
    """Convert corpus to word frequency dict with space-separated chars."""
    vocab = Counter()
    for text in corpus:
        for word in text.split():
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return dict(vocab)

def get_pairs(vocab: dict) -> Counter:
    """Count all adjacent symbol pairs."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair: tuple, vocab: dict) -> dict:
    """Merge all occurrences of the best pair."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def train_bpe(corpus: list[str], vocab_size: int = 1000) -> list[tuple]:
    """Train BPE and return list of merge rules."""
    vocab = get_vocab(corpus)
    merges = []

    # Initial vocabulary: all unique characters
    initial_symbols = set()
    for word in vocab:
        initial_symbols.update(word.split())
    current_size = len(initial_symbols)

    while current_size < vocab_size:
        pairs = get_pairs(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        current_size += 1

    return merges
\`\`\`

---

### Production: SentencePiece

For real pretraining, use SentencePiece (Google's implementation):

\`\`\`python
import sentencepiece as spm

# Train tokenizer on corpus
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # covers 99.995% of chars
    model_type='bpe',
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    shuffle_input_sentence=True,
    num_threads=16,
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

tokens = sp.encode('Hello, world!', out_type=str)
ids = sp.encode('Hello, world!', out_type=int)
print(tokens)  # ['▁Hello', ',', '▁world', '!']
print(ids)     # [8774, 11, 2370, 55]
\`\`\`

---

### Vocabulary Size Tradeoffs

| Vocab Size | Pros | Cons |
|-----------|------|------|
| 8K-16K | Small embedding table | Long sequences, poor OOV handling |
| 32K-64K | Good balance (GPT-2: 50K) | Standard choice |
| 100K+ | Short sequences | Large embedding table, rare tokens undertrained |

**Rule of thumb:** 32K-65K is the sweet spot for English-only models. Multilingual models need 100K-250K.`,

    zh: `## 从零实现BPE分词器

字节对编码（BPE）从单个字节/字符开始，迭代合并最频繁的字符对，直到达到词汇表大小V。

---

### BPE算法

算法从字符级词汇表开始，统计所有相邻符号对的频率，然后合并最高频的对，重复此过程直到达到目标词汇量。

---

### 生产环境：SentencePiece

实际预训练使用Google的SentencePiece实现，支持BPE和unigram模式，可配置词汇量、字符覆盖率等参数。

---

### 词汇表大小权衡

| 词汇量 | 优点 | 缺点 |
|-------|------|------|
| 8K-16K | 嵌入表小 | 序列长，OOV处理差 |
| 32K-64K | 平衡好（GPT-2: 50K）| 标准选择 |
| 100K+ | 序列短 | 嵌入表大，稀有token训练不足 |

**经验法则：** 英文模型32K-65K最佳；多语言模型需要100K-250K。`,
  },
}

export const dataPipeline: TopicContent = {
  id: 'pretrain-data-pipeline',
  title: { en: 'Data Pipeline at Scale', zh: '大规模数据流水线' },
  contentType: 'code',
  content: {
    en: `## Data Pipeline at Scale

A pretraining data pipeline must stream terabytes without loading everything into RAM, bucket sequences by length for efficiency, and shard across hundreds of GPUs.

---

### Streaming Dataset

\`\`\`python
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import json

class StreamingTextDataset(IterableDataset):
    """Streams tokenized documents from sharded JSON files."""

    def __init__(self, data_dir: str, tokenizer, seq_len: int = 2048,
                 rank: int = 0, world_size: int = 1):
        self.files = sorted(Path(data_dir).glob('*.jsonl'))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # Each rank processes a subset of files
        my_files = [f for i, f in enumerate(self.files)
                    if i % self.world_size == self.rank]

        buffer = []

        for filepath in my_files:
            with open(filepath, 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    tokens = self.tokenizer.encode(doc['text'])
                    buffer.extend(tokens)

                    # Pack tokens into fixed-length sequences
                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[:self.seq_len + 1]
                        buffer = buffer[self.seq_len:]

                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)

                        yield {'input_ids': input_ids, 'labels': labels}
\`\`\`

---

### Bucketing by Sequence Length

\`\`\`python
def bucket_batch_sampler(dataset, bucket_boundaries, batch_size):
    """
    Group similar-length sequences to minimize padding waste.
    bucket_boundaries: [128, 256, 512, 1024, 2048]
    """
    buckets = {b: [] for b in bucket_boundaries}

    for idx, sample in enumerate(dataset):
        length = len(sample['input_ids'])
        # Find appropriate bucket
        for boundary in sorted(bucket_boundaries):
            if length <= boundary:
                buckets[boundary].append(idx)
                break

        # Yield batch when bucket is full
        for boundary, indices in buckets.items():
            if len(indices) >= batch_size:
                yield indices[:batch_size]
                buckets[boundary] = indices[batch_size:]
\`\`\`

---

### Multi-GPU DataLoader Setup

\`\`\`python
from torch.utils.data.distributed import DistributedSampler

def build_dataloader(rank: int, world_size: int, batch_size: int):
    dataset = StreamingTextDataset(
        data_dir='/data/pretraining',
        tokenizer=tokenizer,
        seq_len=2048,
        rank=rank,
        world_size=world_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,          # Parallel data loading workers
        pin_memory=True,        # Faster GPU transfer
        prefetch_factor=2,      # Prefetch next 2 batches
    )
\`\`\`

**Key metrics to monitor:** tokens/second throughput, GPU utilization (should be >90%), data loading bottleneck (check if GPU is waiting for data).`,

    zh: `## 大规模数据流水线

预训练数据流水线必须能够流式处理TB级数据而不将所有内容加载到内存，按序列长度分桶以提高效率，并跨数百个GPU进行分片。

---

### 流式数据集

使用IterableDataset实现流式处理，避免将整个数据集加载到内存。每个GPU只处理文件的子集，将文档token化后打包成固定长度序列。

---

### 按序列长度分桶

将相似长度的序列分组到同一批次，最小化padding浪费，提高训练效率。

---

### 关键指标

监控每秒tokens吞吐量（应>90% GPU利用率）和数据加载瓶颈（确保GPU不在等待数据）。`,
  },
}

export const ddpSetup: TopicContent = {
  id: 'pretrain-ddp',
  title: { en: 'Multi-GPU Training with DDP', zh: '使用DDP进行多GPU训练' },
  contentType: 'code',
  content: {
    en: `## Multi-GPU Training with DDP

DistributedDataParallel (DDP) replicates the model on every GPU and synchronizes gradients after each backward pass via all-reduce.

---

### DDP Training Loop

\`\`\`python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank: int, world_size: int, model, dataset, config: dict):
    setup(rank, world_size)

    # Move model to GPU and wrap with DDP
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    dataloader = build_dataloader(rank, world_size, config['batch_size'])

    ddp_model.train()
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(rank)
        labels = batch['labels'].to(rank)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = ddp_model(input_ids)
            loss = compute_loss(outputs, labels)

        # Backward pass — DDP all-reduces gradients automatically
        loss.backward()

        # Gradient clipping (critical for stability)
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        if rank == 0 and step % 100 == 0:
            print('Step: {}, Loss: {:.4f}'.format(step, loss.item()))

    cleanup()

# Launch with torchrun:
# torchrun --nproc_per_node=8 train.py
\`\`\`

---

### Gradient Accumulation for Large Batches

\`\`\`python
accumulation_steps = 4  # Simulate 4x larger batch

for step, batch in enumerate(dataloader):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = ddp_model(**batch).loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
\`\`\`

**Global batch size** = per-GPU batch × GPUs × accumulation steps. GPT-3 used a global batch of ~3.2M tokens.`,

    zh: `## 使用DDP进行多GPU训练

DistributedDataParallel（DDP）在每个GPU上复制模型，并在每次反向传播后通过all-reduce同步梯度。

---

### DDP训练循环

初始化进程组，将模型移到对应GPU并用DDP包装。DDP会在反向传播时自动同步所有GPU的梯度。

关键参数：
- \`find_unused_parameters=False\` — 避免不必要的开销
- 梯度裁剪 \`max_norm=1.0\` — 防止梯度爆炸
- bfloat16混合精度 — 减少内存使用，提高速度

---

### 梯度累积

通过梯度累积模拟更大的全局批次大小：全局批次大小 = 单GPU批次 × GPU数量 × 累积步数。GPT-3使用约320万token的全局批次。`,
  },
}

export const lossMonitoring: TopicContent = {
  id: 'pretrain-loss-monitoring',
  title: { en: 'Loss Monitoring & LR Schedules', zh: '损失监控与学习率调度' },
  contentType: 'code',
  content: {
    en: `## Loss Monitoring & Learning Rate Schedules

Stable pretraining requires careful monitoring of loss curves, gradient norms, and learning rate schedules.

---

### Warmup + Cosine Decay Schedule

\`\`\`python
import math

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup (used by most LLMs)."""

    # Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # Cosine decay phase
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Apply to optimizer
for param_group in optimizer.param_groups:
    param_group['lr'] = get_lr(
        step=current_step,
        warmup_steps=2000,
        max_steps=100000,
        max_lr=3e-4,
        min_lr=3e-5,
    )
\`\`\`

---

### Gradient Norm Monitoring

\`\`\`python
def compute_grad_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return total_norm ** 0.5

# In training loop:
grad_norm = compute_grad_norm(model)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Log to WandB
if rank == 0:
    import wandb
    wandb.log({
        'train/loss': loss.item(),
        'train/perplexity': math.exp(loss.item()),
        'train/grad_norm': grad_norm,
        'train/lr': optimizer.param_groups[0]['lr'],
        'train/step': step,
    })
\`\`\`

---

### Loss Spike Detection

\`\`\`python
class LossSpikeDetector:
    def __init__(self, window: int = 100, threshold: float = 2.0):
        self.history = []
        self.window = window
        self.threshold = threshold

    def check(self, loss: float, step: int) -> bool:
        self.history.append(loss)
        if len(self.history) < self.window:
            return False

        recent = self.history[-self.window:]
        baseline = sum(recent[:-10]) / max(len(recent) - 10, 1)
        current = sum(recent[-10:]) / 10

        if current > baseline * self.threshold:
            print('Loss spike at step {}! Current: {:.4f}, Baseline: {:.4f}'.format(
                step, current, baseline))
            return True
        return False
\`\`\`

**Normal loss curve:** starts ~10-11 (random), drops to ~3-4 (word level), converges to ~2-3 (good model). Sudden spikes > 2x baseline indicate instability — roll back to last checkpoint.`,

    zh: `## 损失监控与学习率调度

稳定的预训练需要仔细监控损失曲线、梯度范数和学习率调度。

---

### 预热+余弦衰减调度

大多数LLM使用线性预热后接余弦衰减的学习率策略：预热阶段（约2000步）线性增加学习率到最大值，然后余弦衰减到最小值（约最大值的10%）。

---

### 梯度范数监控与WandB日志

监控关键指标：训练损失、困惑度（PPL）、梯度范数（正常范围0.1-1.0）和当前学习率。

---

### 损失尖峰检测

**正常损失曲线：** 从~10-11（随机）降至~3-4（词级），收敛到~2-3（优质模型）。突然尖峰>基线2倍表示不稳定——回滚到上一个检查点。`,
  },
}

export const checkpointRecovery: TopicContent = {
  id: 'pretrain-checkpoint',
  title: { en: 'Checkpoint Management & Recovery', zh: '检查点管理与恢复' },
  contentType: 'code',
  content: {
    en: `## Checkpoint Management & Recovery

Multi-week pretraining runs fail. Hardware crashes, OOM errors, loss spikes — you must checkpoint frequently and recover seamlessly.

---

### Checkpoint Save/Load

\`\`\`python
import torch
import os
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, step: int,
                    loss: float, output_dir: str):
    """Save full training state for resumable training."""

    ckpt_path = Path(output_dir) / 'step_{:06d}'.format(step)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Model weights (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save(raw_model.state_dict(), ckpt_path / 'model.pt')

    # Full training state
    torch.save({
        'step': step,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }, ckpt_path / 'training_state.pt')

    print('Saved checkpoint at step {} to {}'.format(step, ckpt_path))

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir: str):
    """Resume training from checkpoint."""

    ckpt_path = Path(checkpoint_dir)

    # Load model weights
    state_dict = torch.load(ckpt_path / 'model.pt', map_location='cpu')
    model.load_state_dict(state_dict)

    # Load training state
    state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu')
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and state['scheduler_state_dict']:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    # Restore RNG state for reproducibility
    torch.set_rng_state(state['rng_state'])
    torch.cuda.set_rng_state_all(state['cuda_rng_state'])

    return state['step'], state['loss']
\`\`\`

---

### Rotating Checkpoints

\`\`\`python
def rotate_checkpoints(output_dir: str, keep_last: int = 5):
    """Keep only the N most recent checkpoints."""
    checkpoints = sorted(Path(output_dir).glob('step_*'))

    # Always keep the best checkpoint separately
    to_delete = checkpoints[:-keep_last]
    for ckpt in to_delete:
        import shutil
        shutil.rmtree(ckpt)
        print('Deleted old checkpoint: {}'.format(ckpt))
\`\`\`

---

### EMA (Exponential Moving Average)

\`\`\`python
class EMA:
    def __init__(self, model, decay: float = 0.9999):
        self.shadow = {name: param.clone()
                       for name, param in model.named_parameters()}
        self.decay = decay

    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = (self.decay * self.shadow[name]
                                 + (1 - self.decay) * param.detach())

    def apply_shadow(self, model):
        """Copy EMA weights into model for evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
\`\`\`

**Checkpoint frequency:** every 1000-2000 steps for long runs. On 256 GPUs, checkpointing can take 5-10 minutes — use async saving.`,

    zh: `## 检查点管理与恢复

多周的预训练任务会失败。硬件崩溃、OOM错误、损失尖峰——必须频繁保存检查点并无缝恢复。

---

### 检查点保存与加载

保存完整训练状态：模型权重、优化器状态、调度器状态和RNG状态。这确保训练可以从任意步骤精确恢复。

---

### 轮转检查点

只保留最近N个检查点以节省存储空间，同时单独保存最佳检查点。

---

### EMA（指数移动平均）

EMA平均历史权重，通常比最后一个检查点的模型在评估时表现更稳定，衰减率0.9999适合大多数场景。

**检查点频率：** 长期运行每1000-2000步一次。在256个GPU上，保存检查点可能需要5-10分钟——使用异步保存。`,
  },
}

export const speculativeDecoding: TopicContent = {
  id: 'pretrain-speculative-decoding',
  title: { en: 'Speculative Decoding', zh: '推测解码' },
  contentType: 'code',
  content: {
    en: `## Speculative Decoding

Speculative decoding uses a small, fast **draft model** to generate K candidate tokens, then verifies them with the large **target model** in a single forward pass — achieving 2-4x speedup with identical output distribution.

---

### Why It Works

LLM inference is memory-bandwidth-bound, not compute-bound. The bottleneck is reading all model weights from HBM for every generated token. By verifying K tokens in one pass, we amortize the weight-loading cost.

---

### The Algorithm

\`\`\`python
import torch
import torch.nn.functional as F

def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    K: int = 5,           # Number of draft tokens per step
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Speculative decoding: draft K tokens, verify with target model.
    Produces identical distribution to target-only sampling.
    """

    generated = input_ids.clone()

    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        # Step 1: Draft model autoregressively generates K tokens
        draft_tokens = []
        draft_probs = []

        x = generated
        for _ in range(K):
            with torch.no_grad():
                draft_logits = draft_model(x).logits[:, -1, :] / temperature
                draft_p = F.softmax(draft_logits, dim=-1)
                token = torch.multinomial(draft_p, num_samples=1)
                draft_tokens.append(token)
                draft_probs.append(draft_p)
                x = torch.cat([x, token], dim=1)

        # Step 2: Target model verifies all K+1 positions in ONE forward pass
        candidate = torch.cat([generated] + draft_tokens, dim=1)

        with torch.no_grad():
            target_logits = target_model(candidate).logits[:, -K-1:, :] / temperature
            target_probs = F.softmax(target_logits, dim=-1)

        # Step 3: Accept/reject each draft token
        accepted = 0
        for i in range(K):
            token = draft_tokens[i].item()
            q = draft_probs[i][0, token].item()  # draft probability
            p = target_probs[0, i, token].item() # target probability

            # Accept with probability min(1, p/q)
            acceptance_prob = min(1.0, p / (q + 1e-8))

            if torch.rand(1).item() < acceptance_prob:
                generated = torch.cat([generated, draft_tokens[i]], dim=1)
                accepted += 1
            else:
                # Reject: sample a corrected token from target and stop
                corrected_p = torch.clamp(target_probs[0, i] - draft_probs[i][0], min=0)
                corrected_p = corrected_p / corrected_p.sum()
                corrected = torch.multinomial(corrected_p, num_samples=1).unsqueeze(0)
                generated = torch.cat([generated, corrected], dim=1)
                break  # Stop accepting this batch

        # If all K accepted, also add the bonus token from target
        if accepted == K:
            bonus_p = target_probs[0, K]
            bonus = torch.multinomial(bonus_p, num_samples=1).unsqueeze(0)
            generated = torch.cat([generated, bonus], dim=1)

    return generated
\`\`\`

---

### Practical Results

| Setup | Speedup | Acceptance Rate |
|-------|---------|----------------|
| 7B target + 68M draft | 1.5x-2x | 60-70% |
| 70B target + 7B draft | 2x-3x | 70-80% |
| 405B target + 13B draft | 3x-4x | 75-85% |

**Key insight:** Acceptance rate depends on how well the draft model approximates the target. Same family models work best (e.g., Llama-3-8B drafts for Llama-3-70B).`,

    zh: `## 推测解码

推测解码使用小型快速的**草稿模型**生成K个候选token，然后用大型**目标模型**在单次前向传播中验证它们——在保持相同输出分布的同时实现2-4倍加速。

---

### 原理

LLM推理受内存带宽限制而非计算限制。瓶颈在于每生成一个token都需要从HBM读取所有模型权重。通过一次验证K个token，我们分摊了权重加载的成本。

---

### 算法步骤

1. 草稿模型自回归生成K个token及其概率
2. 目标模型在ONE次前向传播中验证所有K+1个位置
3. 按接受概率 min(1, p/q) 接受或拒绝每个草稿token
4. 拒绝时从目标分布采样修正token并停止

---

### 实际结果

| 配置 | 加速比 | 接受率 |
|------|--------|--------|
| 7B目标 + 68M草稿 | 1.5x-2x | 60-70% |
| 70B目标 + 7B草稿 | 2x-3x | 70-80% |
| 405B目标 + 13B草稿 | 3x-4x | 75-85% |

**关键洞察：** 接受率取决于草稿模型对目标模型的近似程度。同系列模型效果最佳（如Llama-3-8B为Llama-3-70B起草）。`,
  },
}
