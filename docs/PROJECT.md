# DL Final Project — LLM Fine-tuning for Bilingual Uyghur/English

> Course: Deep Learning — Jönköping University
> Compute: 1× **NVIDIA A100 80GB PCIe** per worker on `slurm.hj.se` (see §Compute environment)
> Dataset: CUTE corpus (Zhuang & Sun, COLING 2025)
> Status: Track 2 — LLM Instruction Fine-tuning

---

## Project Summary

We fine-tune two open-source LLMs — **Qwen2.5-7B-Instruct** (primary) and **LLaMA-3.1-8B-Instruct** (secondary) — on the CUTE parallel corpus (English↔Uyghur direction) using LoRA instruction tuning to produce bilingual assistants capable of translation, text classification, and general instruction-following in both languages. All comparison models are size-matched to the 7–8B class.

Our primary baseline is **CUTE-Llama-P** (Llama2-7B + vocabulary expansion, continued pretraining on CUTE-P) — we run inference on it ourselves on FLORES-200 EN↔UG since the paper only published ZH→UG numbers. The secondary and tertiary baselines are zero-shot LLaMA-3.1-8B-Instruct and Qwen2.5-7B-Instruct (the same models we fine-tune), which isolates the contribution of LoRA tuning.

We evaluate on FLORES-200 (translation, EN→UG and UG→EN), WCM-v2 (Uyghur text classification), and MiLiC-Eval (multi-task bilingual capability). The central research question is whether a modern multilingual LLM fine-tuned with LoRA on EN↔UG instruction data can match or surpass a model trained with full continued pretraining and vocabulary expansion on the same corpus.

---

## Data Available

All datasets and models used in this project are publicly accessible. Access notes are listed where a gated agreement (sharing contact info) is required.

| Resource | Role | HuggingFace / URL | Access | Size |
|----------|------|-------------------|--------|------|
| **CUTE-P** (parallel corpus) | Fine-tuning data (EN↔UG) | [`CMLI-NLP/CUTE-Datasets`](https://huggingface.co/datasets/CMLI-NLP/CUTE-Datasets) · [GitHub](https://github.com/CMLI-NLP/CUTE) | Open | ~934K EN+UG pairs, ~10.9 GB |
| **FLORES-200** | Evaluation — translation (EN↔UG) | [`facebook/flores`](https://huggingface.co/datasets/facebook/flores) | Open | 1,012 sentences per language, devtest split |
| **WCM-v2** | Evaluation — Uyghur text classification | [`hfl/wcm-v2`](https://huggingface.co/datasets/hfl/wcm-v2) | Gated (contact info) | 300 Uyghur test samples, 10 categories |
| **MiLiC-Eval** | Evaluation — multi-task bilingual capability | [`pkupie/milic-eval`](https://huggingface.co/datasets/pkupie/milic-eval) · [GitHub](https://github.com/luciusssss/MiLiC-Eval) | Gated (contact info) | 24K instances, 9 tasks, includes Uyghur |
| **CUTE-Llama-P** (baseline model) | Primary baseline — inference on FLORES-200 EN↔UG | [`CMLI-NLP/CUTE-Llama`](https://huggingface.co/CMLI-NLP/CUTE-Llama) | Open | Llama2-7B + vocab expansion, ~14 GB |
| **Qwen2.5-7B-Instruct** | Primary fine-tuning base; tertiary zero-shot baseline | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Open | ~15 GB (bf16) |
| **LLaMA-3.1-8B-Instruct** | Secondary fine-tuning base; secondary zero-shot baseline | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Gated (Meta license) | ~16 GB (bf16) |
| **FLAN subset** (English-only instructions) | Catastrophic forgetting prevention (10% mix) | [`Muennighoff/flan`](https://huggingface.co/datasets/Muennighoff/flan) | Open | Using ~50K samples |

**Notes:**
- CUTE-P is the parallel variant only. The non-parallel CUTE-NP is available in the same repository but is not used.
- WCM-v2 and MiLiC-Eval gating requires agreeing to share contact information on HuggingFace — no approval wait, instant access after agreement.
- LLaMA-3.1-8B gating requires accepting Meta's LLaMA 3 Community License on HuggingFace — also instant after agreement.
- CUTE-P EN+UG subset is ~10.9 GB on disk; manageable to download and preprocess in full on the worker node (128 GB RAM).

---

## Dataset: CUTE Corpus

**CUTE** (Chinese, Uyghur, Tibetan, English) was introduced by Zhuang & Sun at COLING 2025 ([GitHub](https://github.com/CMLI-NLP/CUTE)). It is the largest open-source corpus for Uyghur to date, produced by machine-translating SkyPile-150B (a large Chinese web corpus) into Uyghur, Tibetan, and English. Native-speaker human evaluation confirms translation quality of 8.5/10 average for Chinese→Uyghur, comparable to Chinese→English (9.1/10).

CUTE comes in two variants with a key distinction the paper establishes experimentally: **parallel data enables more effective cross-lingual knowledge transfer than non-parallel data**.

| Variant | Description | EN lines | UG lines | EN size | UG size |
|---------|-------------|----------|----------|---------|---------|
| **CUTE-P** | Parallel — all four languages aligned sentence-by-sentence (99.98% content similarity) | 933,989 | 934,002 | 3.49 GB | 7.37 GB |
| **CUTE-NP** | Non-parallel — same English, different source documents for UG/ZH/BO | 933,989 | 1,010,381 | 3.49 GB | 7.77 GB |

**We use CUTE-P exclusively** — the parallel variant — for the same reason the paper recommends it: sentence-level alignment directly supports cross-lingual representation alignment, which is the mechanism by which Uyghur capability is transferred from English.

**Practical subset**: The full CUTE-P EN+UG portion is ~10.86 GB. We use the full ~934K EN↔UG parallel pairs for fine-tuning.

---

## Baseline: CUTE-Llama-P (Zhuang & Sun, 2025)

The paper releases two models trained on CUTE. We use **CUTE-Llama-P** as our primary baseline — the stronger of the two, trained on the parallel corpus.

**Architecture**: Llama2-7B with vocabulary expansion. The original Llama2 tokenizer (32K tokens) was extended with 6,000 BPE tokens trained on Uyghur text, growing the Uyghur-specific vocabulary from 32K to **47,905 tokens**. Embeddings for new tokens are initialized with mean values from existing embeddings.

**Training**: Continued pre-training (not instruction fine-tuning) on CUTE-P. 8× H800 GPUs, ~18 hours, 1 epoch.

**Published results from the paper** are reported on **ZH→UG** (Chinese pivot) — the paper's training direction. We do **not** use these numbers as comparison targets for our project. They are recorded here only for context, since they characterize CUTE-Llama-P's best-case direction (the one most aligned with how it was trained).

FLORES-200 (ZH→UG, few-shot with 3 examples) — paper-reported, **not our comparison target**:

| Model | BLEU↑ | chrF↑ | TER↓ |
|-------|-------|-------|------|
| BLOOM-7.1B | 4.9 | 0.319 | 0.862 |
| Llama2-7B | 5.4 | 0.334 | 0.847 |
| Llama3.1-8B | 7.5 | 0.376 | 0.798 |
| CUTE-Llama-NP | 9.0 | 0.419 | 0.762 |
| **CUTE-Llama-P** | **10.2** | **0.443** | **0.738** |

WCM-v2 Uyghur text classification (paper, zero-shot transfer from Chinese training data) — used as a direct comparison target since both setups produce a Uyghur-language metric:

| Model | Accuracy | F1 |
|-------|----------|----|
| CINO-large | — | 28.8 |
| BLOOM-7.1B | 35.67 | 49.86 |
| Llama3.1-8B | 68.33 | 77.92 |
| Llama2-7B | 78.0 | 82.42 |
| CUTE-Llama-NP | 86.33 | 87.97 |
| **CUTE-Llama-P** | **87.0** | **89.08** |

> **Direction policy**: The paper trained and evaluated CUTE-Llama-P with Chinese as the pivot, so its published ZH→UG numbers reflect its best-case direction and are not informative for English↔Uyghur. Our project focuses on **EN↔UG** in **both directions**. We measure CUTE-Llama-P, Qwen2.5-7B-Instruct, and LLaMA-3.1-8B-Instruct ourselves on FLORES-200 EN→UG and UG→EN to obtain matched-direction baselines.

---

## Our Models

All models in this project are size-matched to the 7–8B class, so direct comparisons are not confounded by parameter count.

### Primary — Qwen2.5-7B-Instruct (Alibaba)

| Property | Value |
|----------|-------|
| Parameters | 7B |
| Architecture | Decoder-only transformer (GQA) |
| Context length | 128K tokens |
| Vocabulary size | 150K tokens |
| Uyghur in pretraining | Very likely — Alibaba's corpus covers Central Asian web data |
| Arabic-script tokenization | Good — Uyghur characters represented as 1–3 tokens |
| License | Qwen License (permissive research + commercial) |
| HuggingFace | `Qwen/Qwen2.5-7B-Instruct` |

**Why Qwen as primary**: The 150K vocabulary provides far more efficient Arabic-script tokenization than Llama2 (32K) — Uyghur words that require vocabulary surgery in CUTE-Llama are handled natively. Alibaba's training data almost certainly includes Uyghur content, meaning LoRA tuning is *unlocking* latent capability rather than teaching a language from scratch. This is the fundamental advantage over the CUTE-Llama approach and the model most likely to support the project's central claim.

### Secondary — LLaMA-3.1-8B-Instruct (Meta)

| Property | Value |
|----------|-------|
| Parameters | 8B |
| Architecture | Decoder-only transformer (GQA, RoPE) |
| Context length | 128K tokens |
| Vocabulary size | 128K tokens |
| Uyghur in pretraining | Minimal — Meta's data is heavily English/European |
| Arabic-script tokenization | Fair — wider vocabulary than Llama2 but still suboptimal for Uyghur |
| License | LLaMA 3 Community License |
| HuggingFace | `meta-llama/Llama-3.1-8B-Instruct` |

**Why LLaMA-3.1-8B as secondary**: Serves two roles. (1) A **same-family contrast against CUTE-Llama-P** — both are Meta-trained Llama models, but CUTE-Llama-P uses Llama2-7B with vocabulary surgery, while Llama-3.1-8B uses the modern Llama-3 tokenizer (128K vocab, no surgery). This isolates whether the original CUTE-Llama vocabulary expansion is still needed now that newer Llama tokenizers exist. (2) A **near-cold-start contrast against Qwen** — Llama-3.1's pretraining has far less Uyghur exposure than Qwen2.5, so the LoRA "lift" should be larger if our hypothesis holds. The published Llama3.1-8B zero-shot ZH→UG number from the paper (7.5 BLEU / 0.376 chrF) is not used as a comparison; we measure EN↔UG ourselves.

### Why no 3B models

We previously considered LLaMA-3.2-3B and Qwen2.5-3B as a "scale ablation" pair, but a 3B model competing against 7B baselines is confounded by parameter count: any gap could be attributed to scale rather than to the experimental variable. To keep comparisons clean, all models in this project are 7–8B. A scale ablation can be added later if compute permits.

---

## Open-source LLM Comparison

Five candidates in the 7–8B class were evaluated for this task. The two critical axes are: **vocabulary size** (determines Arabic-script tokenization efficiency) and **prior Uyghur exposure** (determines whether fine-tuning unlocks vs. teaches). 3B variants (LLaMA-3.2-3B, Qwen2.5-3B) were considered earlier but excluded because they introduce a parameter-count confound when compared against the 7B CUTE-Llama-P baseline.

| | Qwen2.5-7B | LLaMA-3.1-8B | Mistral-7B | Gemma-2-9B | Phi-3.5-mini |
|--|-----------|-------------|-----------|-----------|-------------|
| Parameters | 7B | 8B | 7B | 9B | 3.8B |
| Vocabulary size | 150K | 128K | **32K ❌** | 256K ✅ | **32K ❌** |
| Uyghur pretraining | Likely ✅ | Minimal ⚠️ | Unlikely ❌ | Possible ⚠️ | Unlikely ❌ |
| Arabic-script tokenization | Good ✅ | Fair ⚠️ | Poor ❌ | Excellent ✅ | Poor ❌ |
| Context window | 128K | 128K | 32K | **8K ⚠️** | 128K |
| Multilingual instruct | High ✅ | Medium ⚠️ | Low ⚠️ | Medium ⚠️ | Low ❌ |
| License | Permissive | Restrictive ⚠️ | Apache 2.0 ✅ | Custom | MIT ✅ |
| **Overall fit** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐** |

**Why Mistral and Phi were ruled out**: Both have 32K vocabularies. Arabic-script Uyghur characters decompose into byte-level fragments, inflating sequence length and degrading attention. This is a structural disadvantage that cannot be overcome by fine-tuning without vocabulary surgery — exactly the surgery CUTE-Llama required for Llama2.

**Why Gemma-2-9B is not selected despite its 256K vocabulary**: The 8K context window is a genuine limitation for instruction fine-tuning with system prompts and longer Uyghur passages. Gemma-2's alternating local/global attention also requires patching in standard fine-tuning libraries. Gemma-2-9B would be the best alternative if these issues are acceptable.

**Why LLaMA-3.1-8B over LLaMA-3.2-3B**: 3B vs the 7B CUTE-Llama-P baseline introduces a scale confound; any gap could be attributed to parameter count rather than to the experimental variable. Llama-3.1-8B is size-matched and uses the same Llama-3 tokenizer family.

---

## Fine-tuning Approach

### Instruction Formatting

CUTE-P parallel pairs are reformatted as instruction-response examples. Both translation directions are included:

```
System: You are a bilingual assistant fluent in English and Uyghur.
User: Translate to Uyghur: "The weather is cold today."
Assistant: بۈگۈن ھاۋا سوغاق.
```

```
System: You are a bilingual assistant fluent in English and Uyghur.
User: تەرجىمە قىلىڭ ئەنگلىزچىگە: "بۈگۈن ھاۋا سوغاق."
Assistant: The weather is cold today.
```

A small proportion (~10%) of English-only instruction data (FLAN subset) is mixed in to mitigate catastrophic forgetting of English capability.

### LoRA Configuration

| Hyperparameter | Value |
|---------------|-------|
| Method | **QLoRA** (4-bit NF4 base + bf16 LoRA adapters) via TRL `SFTTrainer` + PEFT |
| Rank (r) | 64 |
| Alpha (α) | 128 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | ~1–2% of total |
| Gradient checkpointing | Enabled (recomputes activations in backward to save VRAM) |

**Why QLoRA as the default**: our cluster job lands on a **MIG 1g.10gb** slice of the A100 (~10 GB visible VRAM, see §Compute environment), not the full 80 GB. Loading a 7B/8B base in bf16 alone takes ~14–16 GB, so bf16 LoRA does not fit. QLoRA loads the frozen base in 4-bit NF4 (~4 GB for 7B), keeps the trainable LoRA adapters and gradients in bf16, and uses gradient checkpointing to keep activation memory bounded. Cost: ~25–40% slower per step (dequantization + recomputation) and a small documented quality hit vs full-precision LoRA, which is the standard trade-off in the QLoRA literature (Dettmers et al., 2023).

LoRA itself is chosen over full fine-tuning and vocabulary expansion (the approach used by CUTE-Llama) because Qwen2.5's tokenizer already handles Uyghur natively — no vocabulary surgery is needed. This is a cleaner experimental setup: we isolate the effect of instruction fine-tuning on task performance.

If admins later grant a full A100 80GB (or a larger MIG profile such as `3g.40gb` / `7g.80gb`), training switches to **bf16 LoRA** without quantization (one config flag). All other LoRA hyperparameters are unchanged.

### Training Configuration

| Hyperparameter | Qwen2.5-7B (QLoRA on MIG 10 GB) | LLaMA-3.1-8B (QLoRA on MIG 10 GB) | If full A100 80GB granted (bf16 LoRA) |
|---------------|----------------------------------|-----------------------------------|----------------------------------------|
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| LR schedule | Cosine + 3% warmup | Cosine + 3% warmup | Cosine + 3% warmup |
| Per-device batch size | 1 | 1 | 4 |
| Gradient accumulation | 32 | 32 | 8 |
| Effective batch size | 32 | 32 | 32 |
| Epochs | 3 | 3 | 3 |
| Max sequence length | 512 tokens | 512 tokens | 512 tokens |
| Base precision | 4-bit NF4 + double-quant | 4-bit NF4 + double-quant | bf16 |
| LoRA adapter precision | bf16 | bf16 | bf16 |
| Optimizer | Paged AdamW 8-bit | Paged AdamW 8-bit | Paged AdamW 8-bit |
| Gradient checkpointing | On | On | Off |
| Hardware | 1× A100 MIG `1g.10gb` (~10 GB VRAM) | 1× A100 MIG `1g.10gb` | 1× A100 80GB |
| Estimated training time (3 epochs, full ~934K pairs) | ~16–24 h | ~20–28 h | ~8–12 h (Qwen) / ~10–14 h (Llama) |

The QLoRA timings above are first-order estimates. We will calibrate them with a smoke run (`main.py --mode train --sample-count 1000 --epochs 1`) before submitting full ablation jobs.

---

## Evaluation

### Benchmark 1: FLORES-200 (Translation Quality)

Standard benchmark for low-resource MT. Uyghur dev/test splits translated by professional translators. Primary metric is **chrF** (character F-score), preferred over BLEU for morphologically rich Arabic-script languages. We evaluate **both directions, EN→UG and UG→EN**, on `devtest` (1,012 sentences per direction).

All FLORES-200 numbers in our results — for our fine-tuned models *and* for the baselines — are produced by us under the same evaluation harness. The paper's published ZH→UG numbers are not used as comparison targets (see "Direction policy" above).

### Benchmark 2: WCM-v2 (Text Classification, Uyghur)

**WCM-v2** (Yang et al., 2022) is a multilingual text classification dataset with 10 categories, including a Uyghur test set (300 samples). The CUTE paper uses it to measure zero-shot cross-lingual transfer — models are fine-tuned on Chinese training data and tested on Uyghur.

We use it differently: we test whether our instruction-fine-tuned models can classify Uyghur text when prompted in Uyghur (no Chinese pivot), and compare to the paper's reported numbers for CUTE-Llama-P (87.0% accuracy / 89.08 F1 on Uyghur).

### Benchmark 3: MiLiC-Eval (Multi-task Bilingual Capability)

**MiLiC-Eval** (ACL 2025) is a benchmark designed specifically for Chinese minority languages including Uyghur. It covers 9 tasks across 24K instances: sentence topic classification, NLI, reading comprehension, QA, named entity recognition, summarization, and more. It is the most comprehensive structured benchmark for Uyghur available as of 2025.

MiLiC-Eval is the primary tool for answering the question the paper left open: *can our models handle Uyghur across diverse tasks, not just translation?*

### Metric Summary

| Evaluation | Metric | Tool | What it measures |
|-----------|--------|------|-----------------|
| FLORES-200 EN→UG | chrF, BLEU | `sacrebleu` | Translation quality to Uyghur |
| FLORES-200 UG→EN | chrF, BLEU | `sacrebleu` | Translation quality to English |
| WCM-v2 Uyghur | Accuracy, F1 | sklearn | Cross-lingual text classification |
| MiLiC-Eval (9 tasks) | Task-specific | HuggingFace | Broad bilingual task capability |
| Language ID rate | % correct lang | `langdetect` | Does the model respond in the prompted language? |
| Perplexity (held-out CUTE-P) | PPL | `evaluate` | Uyghur language modeling quality |

### Baselines

All FLORES-200 baselines are run by us on EN→UG and UG→EN. WCM-v2 numbers for CUTE-Llama-P are taken from the paper (Uyghur classification accuracy is comparable across our setup and theirs).

| Role | Model | Condition | FLORES-200 source | WCM-v2 source |
|------|-------|-----------|-------------------|---------------|
| Primary baseline | **CUTE-Llama-P** | Continued pretraining on CUTE-P, vocab expansion, ZH pivot training | Our measurement (EN↔UG) | Paper (Acc 87.0% / F1 89.08) |
| Secondary baseline | LLaMA-3.1-8B-Instruct (zero-shot) | Same model as our secondary fine-tune; isolates LoRA contribution | Our measurement (EN↔UG) | Our measurement |
| Tertiary baseline | Qwen2.5-7B-Instruct (zero-shot) | Same model as our primary fine-tune; isolates LoRA contribution | Our measurement (EN↔UG) | Our measurement |
| Optional | Qwen2.5-7B-Instruct (5-shot) | In-context examples, no weight update | Our measurement (EN↔UG) | Our measurement |

---

## Ablation: Data Mixing Ratio

The key variable is the proportion of Uyghur/English translation pairs versus English-only instruction data.

| Experiment | UG/EN CUTE-P pairs | EN-only data | Hypothesis |
|-----------|-------------------|--------------|------------|
| Mix-0 (UG only) | 100% | 0% | Maximum UG gain; catastrophic forgetting of English |
| Mix-10 | 90% | 10% | — |
| Mix-20 | 80% | 20% | Recommended default |
| Mix-50 | 50% | 50% | Strong English retention; reduced UG gain |

Each variant evaluated on:
- **chrF (EN↔UG)** — Uyghur capability gain
- **Perplexity on English WikiText-103** — English capability retention
- **WCM-v2 Uyghur accuracy** — task transfer quality

The trade-off curve between Uyghur gain and English retention is the primary scientific contribution of the ablation.

---

## Research Contribution

Relative to the CUTE paper (Zhuang & Sun, 2025), this project makes the following distinct contributions:

1. **EN↔UG direction**: The paper focuses on Chinese as a pivot language (ZH→UG). We evaluate English↔Uyghur directly, which is more practically relevant for international use and has not been published with CUTE data.

2. **LoRA vs. continued pretraining**: The paper uses full continued pretraining + vocabulary expansion on Llama2. We test whether parameter-efficient instruction fine-tuning (LoRA) on a model with native multilingual vocabulary (Qwen2.5) achieves comparable or better results with far less compute.

3. **Instruction capability beyond translation**: The paper measures translation, text classification, and reading comprehension via zero-shot cross-lingual transfer. We additionally measure direct Uyghur instruction-following, where prompts and responses are both in Uyghur.

---

## Project Timeline

| Week | Milestone |
|------|-----------|
| 1 | Download CUTE-P EN+UG; preprocess and format as instruction pairs; set up zero-shot baselines |
| 2 | LoRA fine-tune Qwen2.5-7B (Mix-20); evaluate on FLORES-200 and WCM-v2 |
| 3 | LoRA fine-tune LLaMA-3.1-8B (Mix-20); compare to Qwen and CUTE-Llama-P |
| 4 | Ablation runs (Mix-0, Mix-10, Mix-50); MiLiC-Eval evaluation |
| 5 | Analysis and write-up |

---

## Compute environment

Cluster: **`slurm.hj.se`** (Jönköping University), accessed via SSH alias `ju-compute-server` (`mach25ku@jth-ai-01.hj.se:50001`).

| Property | Value |
|----------|-------|
| Worker nodes | 7 (`worker1`…`worker7.slurm.hj.se`) |
| Per-node CPU | 16 cores |
| Per-node RAM | 128 GB |
| Per-node GPU (physical) | **NVIDIA A100 80GB PCIe** (`Gres=gpu:1`; Slurm does not name the model in `Gres`) |
| Per-job GPU (effective) | **MIG `1g.10gb`** — confirmed via `nvidia-smi -L` inside an `srun` job. The job sees ~10 GB VRAM, not the full 80 GB. `nvidia-smi --query-gpu=memory.total` returns "Insufficient Permissions" because querying the parent device is blocked from inside a MIG instance. |
| Partition `priority` | MaxTime **5 days**, default 2 h, no per-user `MaxTRESMins`/`MaxJobs` |
| Partition `scavenger` | unlimited time, preemptible (default partition) |
| Account / QoS | `tmls22` / `normal` (no quota set) |
| Concurrency | up to 7 parallel single-GPU jobs (one per worker) |

**Implications for the ablation**:
- **MIG 10 GB rules out bf16 LoRA on a 7–8B model.** A bf16 7B base alone is ~14 GB. We default to **QLoRA** (4-bit NF4 base, bf16 adapters, gradient checkpointing) which fits in ~6–9 GB at training time.
- A single `--partition priority` job (5-day cap) comfortably fits any one QLoRA fine-tune (estimated 16–28 h for 3 epochs on full CUTE-P).
- The full Mix-{0,10,20,50} × {Qwen-7B, Llama-3.1-8B} ablation = 8 fine-tunes can be submitted as 8 parallel jobs (one per idle worker), collapsing wall-clock to ~max(per-job time) instead of ~sum.
- **TODO — contact cluster admins**: ask whether `--gres=gpu:1` can grant the full A100 (or a larger MIG profile such as `3g.40gb` or `7g.80gb`). If granted, switch training back to bf16 LoRA via a config flag — same code path, ~2× faster, slightly better quality. Until then, plan as if MIG `1g.10gb` is permanent.
- **For each new job**: confirm allocation by reading `nvidia-smi -L` once at startup. The expected line is `MIG 1g.10gb Device 0`; if it ever changes (e.g. to `3g.40gb` or no MIG line), update training config.

---

*Last updated: May 2026 — physical GPU confirmed A100 80GB PCIe but Slurm assigns a **MIG `1g.10gb`** slice (~10 GB visible); training defaults to **QLoRA** (4-bit NF4 base + bf16 adapters + gradient checkpointing) with a bf16 LoRA flag for if/when admins grant a full GPU. Model line-up all 7–8B; FLORES-200 baselines all measured by us on EN↔UG.*
