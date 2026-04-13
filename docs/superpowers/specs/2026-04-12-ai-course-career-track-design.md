# AI Course Career Track — Design Spec
**Date:** 2026-04-12  
**Project:** `machine-learning-tutorial`  
**Status:** Approved, ready for implementation

---

## Goal

Extend the existing `machine-learning-tutorial` React app to close the 5 biggest gaps between its current curriculum and what OpenAI / Anthropic / DeepMind / Meta AI actually hire for — and add a Resources page pointing to the best external bootcamps, YouTube channels, GitHub repos, and papers.

---

## Approach: Tiered Learning Path with Career Track Layer (Option B)

Add a lightweight **tier concept** (Junior / Mid-level / Senior) as a filter across the home page and sidebar. The 5 new gap modules are tagged to Mid/Senior tiers. Users filter by their target level to see a focused learning path. A new `/resources` route provides curated external learning materials, each tagged to a tier.

---

## Job Market Research Summary

### Top 5 Gaps in Existing Curriculum

| Gap | Existing Coverage | Job Market Demand | Target Tier |
|-----|------------------|-------------------|-------------|
| Pretraining from Scratch | Architecture only, no operational mechanics | Critical — 6/10 job postings | Senior |
| Distributed Training (DDP/FSDP) | Mentioned, no hands-on | Required for scaling roles | Mid/Senior |
| Inference Optimization (torch.compile, KV cache, vLLM) | Quantization only | Every inference role | Mid/Senior |
| RLHF / DPO / Alignment | Theory only | 7/10 postings, especially Anthropic | Senior |
| CUDA / GPU Kernel Optimization | None | Differentiator for senior/research roles | Senior |

### 3-Tier Job Market

| Tier | Years Exp | Key Skills | Current Coverage |
|------|-----------|-----------|-----------------|
| Junior | 0–2 yrs | RAG, fine-tuning, LLM APIs | Mostly covered ✅ |
| Mid-level | 2–5 yrs | Distributed training, optimization, custom implementations | Gaps ⚠️ |
| Senior/Research | 5+ yrs | Kernel optimization, pretraining, alignment, scaling | Major gaps ❌ |

---

## Data Model Changes

### `src/content/types.ts` — Add `Tier`

```ts
export type Tier = 'junior' | 'mid' | 'senior'

export interface TopicContent {
  // all existing fields unchanged
  tier?: Tier   // optional; defaults to 'junior' for existing topics
}
```

### `src/content/resources.ts` — New file

```ts
export type ResourceCategory = 'course' | 'youtube' | 'github' | 'paper'

export interface Resource {
  title: { en: string; zh: string }
  description: { en: string; zh: string }
  url: string
  category: ResourceCategory
  tier: Tier
  free: boolean
  author?: string
}

export const resources: Resource[] = [ /* populated during implementation */ ]
```

---

## New Curriculum Chapters (5 Modules)

All content: written articles + Python code examples. Fully bilingual (EN / 中文).

### 1. Pretraining a Transformer from Scratch (`senior`)
**Chapter ID:** `pretraining`  
**Topics:**
- Corpus assembly & preprocessing
- BPE tokenizer from scratch
- Data pipeline at scale (streaming, bucketing, sharding)
- Multi-GPU training with DDP
- Loss monitoring, convergence debugging
- Checkpoint management & recovery
- Inference scaling (speculative decoding intro)

### 2. Distributed Training: DDP & FSDP (`mid`)
**Chapter ID:** `distributed-training`  
**Topics:**
- Data Parallelism vs Model Parallelism
- DDP hands-on: setup, gradient sync, pitfalls
- FSDP: sharding strategies, memory savings
- Pipeline parallelism overview
- Debugging distributed training

### 3. Inference Optimization & Serving (`mid`)
**Chapter ID:** `inference-optimization`  
**Topics:**
- KV cache: implementation and management
- torch.compile modes (default, reduce-overhead, max-autotune)
- Attention optimization (FlashAttention-2 vs standard)
- Quantization for inference (AWQ, GPTQ)
- vLLM / TensorRT-LLM patterns
- Latency profiling and benchmarking

### 4. Alignment: RLHF, DPO & RLAIF (`senior`)
**Chapter ID:** `alignment`  
**Topics:**
- Why alignment matters (safety context)
- Reward model training
- PPO loop implementation
- DPO mechanics & comparison to RLHF
- RLAIF (AI feedback instead of human)
- Preference dataset generation
- Alignment evaluation metrics

### 5. GPU Optimization & CUDA Kernels (`senior`)
**Chapter ID:** `gpu-optimization`  
**Topics:**
- GPU memory hierarchy & roofline model
- torch.compile internals
- Triton kernel basics
- FlashAttention custom kernel walkthrough
- Profiling with PyTorch Profiler & Nsight
- Cost/performance tradeoffs at scale

---

## New Route: `/resources`

### Page Sections
1. **Learning Roadmap** — which resources match which tier (visual table)
2. **Courses & Bootcamps** — fast.ai, DeepLearning.AI, Hugging Face, Harvard, Udacity
3. **YouTube Channels** — Andrej Karpathy, 3Blue1Brown, Yannic Kilcher, Two Minute Papers
4. **GitHub Repos** — nanoGPT, LLMs-from-scratch (Raschka), transformers (HF), tinygrad
5. **Key Papers** — Attention Is All You Need, LoRA, QLoRA, DPO, InstructGPT, FlashAttention

### Filter
- Filter by tier: All / Junior / Mid / Senior
- Filter by category: All / Course / YouTube / GitHub / Paper
- Filter by free: toggle

---

## UI Changes

### Home Page
- Add tier filter badges below the hero: `All` `🟢 Junior` `🟡 Mid` `🔴 Senior`
- Chapters filtered by tier; "All" shows everything (default)
- Add "Resources" link in CTA buttons area

### Sidebar / AppLayout
- Add tier badge (colored dot) next to chapter titles for new chapters
- Add "📚 Resources" nav item at the bottom

### New Page Component
- `src/pages/ResourcesPage.tsx` — resources grid with filter controls

---

## Files to Create / Modify

### New files
- `src/content/resources.ts` — resource data
- `src/content/topics/pretraining.ts`
- `src/content/topics/distributed-training.ts`
- `src/content/topics/inference-optimization.ts`
- `src/content/topics/alignment.ts`
- `src/content/topics/gpu-optimization.ts`
- `src/pages/ResourcesPage.tsx`

### Modified files
- `src/content/types.ts` — add `Tier` type
- `src/content/curriculum.ts` — add 5 new chapters, import new topics
- `src/App.tsx` — add `/resources` route
- `src/pages/HomePage.tsx` — add tier filter UI
- `src/components/Layout/AppLayout.tsx` — add Resources nav + tier badges

---

## Out of Scope
- Interactive visualizations (attention maps, training curves)
- Python playground integration for new content
- Live demo page for new content
- Chinese-only content (always bilingual)

---

## Implementation Order (suggested)
1. Add `Tier` type to `types.ts`
2. Add 5 new topic files (content-heavy, no UI changes)
3. Register new chapters in `curriculum.ts`
4. Build `ResourcesPage.tsx` + `resources.ts` data
5. Add `/resources` route in `App.tsx`
6. Add tier filter to `HomePage.tsx`
7. Add Resources nav + tier badges to `AppLayout.tsx`
