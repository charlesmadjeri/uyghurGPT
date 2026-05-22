# DL Final Project — LLM Fine-tuning for Bilingual Uyghur/English

> Course: Deep Learning — Jönköping University
> Compute: 1× **NVIDIA A100 80GB PCIe** per worker on `slurm.hj.se` (~**24 GB MIG slice** assigned per job — see §Compute environment)
> Dataset: CUTE corpus (Zhuang & Sun, COLING 2025)
> Status: Track 2 — LLM Instruction Fine-tuning (scoped May 2026 — see `PROJECT_REFINEMENT.md`)

---

## Project Summary

We fine-tune two open-source LLMs — **Qwen2.5-7B-Instruct** (primary) and **LLaMA-3.1-8B-Instruct** (secondary) — on the CUTE parallel corpus (English↔Uyghur direction) using LoRA instruction tuning to produce bilingual assistants capable of translation, text classification, and general instruction-following in both languages. All comparison models are size-matched to the 7–8B class.

### Core experiment (must complete)
Fine-tune **Qwen2.5-7B-Instruct with Mix-20 QLoRA** and evaluate on **FLORES-200 (EN→UG and UG→EN chrF + BLEU)** and **WCM-v2 (Uyghur text classification)**. Compare against zero-shot Qwen2.5-7B-Instruct (isolates LoRA contribution) and zero-shot LLaMA-3.1-8B-Instruct (cross-architecture reference).

### Stretch goals (attempt only after core is complete and evaluated)
- Fine-tune LLaMA-3.1-8B-Instruct (Mix-20) and compare to Qwen2.5 fine-tune
- Ablation: Mix-{0, 10, 50} on Qwen2.5-7B
- MiLiC-Eval (9-task bilingual benchmark) — deferred to final report if time allows

> **Scope discipline:** the design presentation and interim milestones are assessed against the core experiment only. Stretch goals are reported as "completed" or "not reached" in the final report. Examiners are told upfront which is which.

---

## Pre-flight Sanity Checks (Day 1 — mandatory before any training)

These checks must pass before committing compute to any fine-tuning run. If any check fails, resolve it before proceeding.

| Check | Command / Method | Pass Condition | Fallback if Fail |
|-------|-----------------|----------------|-----------------|
| **Tokenizer — Uyghur segmentation** | Encode 50 Uyghur sentences from CUTE-P; compute token/byte ratio | Ratio < 0.6 (reasonable segmentation, not byte-fallback) | Re-evaluate "no vocabulary surgery" decision; consult `PROJECT_REFINEMENT.md` §Rec-4 |
| **QLoRA memory fit — Qwen2.5-7B** | Load 4-bit NF4 model + bf16 adapters + dummy forward pass on the assigned MIG slice (~24 GB) | Peak VRAM < 22 GB (leave ~2 GB headroom) | Reduce LoRA rank, disable flash-attention, or request larger MIG slice from admins |
| **QLoRA memory fit — LLaMA-3.1-8B** | Same as above | Same threshold | Same fallback |
| **CUTE-P EN+UG download + format** | Download, spot-check 100 lines, verify UTF-8 + Arabic script integrity | No mojibake, lines align EN↔UG | Re-download; check HuggingFace dataset viewer |
| **CUTE-Llama-P load test** | Attempt to load model weights + run 5 FLORES sentences | Inference produces Uyghur output | Fall back to zero-shot baselines only (see §CUTE-Llama-P Baseline) |

Record results of all checks in `results/preflight/preflight_report.md` before submitting any Slurm training job.

---

## CUTE-Llama-P Baseline

**Status: planned core baseline on the 24 GB MIG slice. Preflight check 5 has already loaded the model and produced Uyghur Arabic-script output for ≥3/5 test sentences. Fallback to zero-shot-only baselines remains documented for completeness.**

CUTE-Llama-P (Llama2-7B + vocabulary expansion + continued pretraining on CUTE-P) is the comparison baseline from the paper. The paper only publishes ZH→UG numbers; we run inference on FLORES-200 EN↔UG ourselves.

**Known constraints (not blockers on 24 GB):**
- Expanded ~155 K-token vocabulary → ~2.5 GB bf16 embedding tables; loads cleanly in 4-bit NF4 within the 24 GB slice.
- Base LM (not instruct), so it is prompted with **few-shot continuation** (`English: …\nUyghur:`) rather than the chat template used for Qwen/LLaMA. This is a known protocol difference, documented in the report.
- The reported ZH→UG paper numbers are not used for direct comparison — direction mismatch.

**Protocol:**
1. Confirm preflight check 5 PASS on the current cluster (already PASS in `results/preflight/check5.json`).
2. Evaluate on FLORES-200 EN↔UG and WCM-v2 using the same references and metrics as the other variants. Report its prompt style alongside the score.

**Fallback (kept for completeness, no longer expected):**
- If a future cluster change makes loading impossible again, fall back to zero-shot Qwen2.5 and zero-shot LLaMA-3.1 only. The contribution framing then narrows to "LoRA instruction tuning vs. zero-shot multilingual LLMs" — still valid.

---

## Expected Result Asymmetry — EN→UG vs UG→EN

**This is an anticipated outcome, not a risk or failure.**

UG→EN translation is expected to score significantly higher than EN→UG on chrF/BLEU. This asymmetry arises because:
- Generating fluent English (a resource-rich language with strong prior in both base models) is inherently easier than generating fluent Uyghur
- Both Qwen2.5 and LLaMA pre-training data is heavily English-weighted; the models have a strong English decoder prior
- CUTE-P provides ~934K EN↔UG pairs, but the model's ability to produce Uyghur morphology correctly (agglutinative, right-to-left Arabic script) is a harder generative problem

**How to handle it:**
- Report both directions separately in all tables — do not average them
- Discuss the asymmetry explicitly in the results section; frame it as a known challenge in low-resource generative NLP, not as a project failure
- If EN→UG chrF is low in absolute terms, compare the *relative gain* over zero-shot (the delta matters more than the absolute score)
- Consider a qualitative analysis: show 3–5 example outputs in both directions to illustrate where the model succeeds and fails

---

## Data Available

All datasets and models used in this project are publicly accessible. Access notes listed where a gated agreement is required.

| Resource | Role | Source | License | Notes |
|----------|------|---------|---------|-------|
| **CUTE-P** (EN + UG subset) | Fine-tuning corpus | [`CMLI-NLP/CUTE`](https://github.com/CMLI-NLP/CUTE) | Open | ~10.9 GB on disk; ~934K EN↔UG pairs |
| **FLORES-200** | Translation evaluation | HuggingFace `facebook/flores` | Open | 1012 sentences per language; use `eng_Latn` + `uig_Arab` |
| **WCM-v2** | Uyghur classification eval | HuggingFace (gated) | Gated (instant) | Agree to share contact info — no approval wait |
| **MiLiC-Eval** | Multi-task bilingual eval *(stretch)* | HuggingFace (gated) | Gated (instant) | Same gating as WCM-v2; defer to final report |
| **Qwen2.5-7B-Instruct** | Primary model | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache 2.0 | ~15 GB (bf16); ~8 GB (4-bit NF4) |
| **LLaMA-3.1-8B-Instruct** | Secondary model + zero-shot baseline | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Gated (Meta license, instant) | ~16 GB (bf16); ~8 GB (4-bit NF4) |
| **CUTE-Llama-P** | Comparison baseline | CMLI-NLP GitHub | Open | Custom 155 K vocabulary; fits in 4-bit NF4 on the 24 GB slice; few-shot prompted (base LM) |
| **FLAN subset** | Catastrophic forgetting prevention | [`Muennighoff/flan`](https://huggingface.co/datasets/Muennighoff/flan) | Open | ~50K samples for Mix-20 |

**Notes:**
- CUTE-P EN+UG subset only. CUTE-NP (non-parallel) and ZH/BO splits are not used.
- WCM-v2 and MiLiC-Eval: agree to share contact information on HuggingFace — instant access, no approval wait.
- LLaMA-3.1-8B: accept Meta's LLaMA 3 Community License on HuggingFace — instant after agreement.

---

## Dataset: CUTE Corpus

**CUTE** (Chinese, Uyghur, Tibetan, English) was introduced by Zhuang & Sun at COLING 2025 ([GitHub](https://github.com/CMLI-NLP/CUTE)). It is the largest open-source corpus for Uyghur to date, produced by machine-translating SkyPile-150B (a large Chinese web corpus) into Uyghur, Tibetan, and English.

Native-speaker human evaluation confirms average translation quality of **8.5/10 for Chinese→Uyghur**, comparable to Chinese→English (9.1/10). The paper establishes experimentally that **parallel data enables more effective cross-lingual knowledge transfer than non-parallel data** — this justifies using CUTE-P exclusively.

| Variant | ZH Lines | EN Lines | UG Lines | BO Lines | Total Size |
|---------|----------|----------|----------|----------|-----------|
| CUTE-P  | 933,946  | 933,989  | 934,002  | 934,140  | 24.70 GB  |
| CUTE-NP | 1,000,609 | 933,989 | 1,010,381 | 989,723 | 25.80 GB  |

We use **CUTE-P EN + UG only** (~10.9 GB).

---

## Models

### Primary — Qwen2.5-7B-Instruct
- Strong multilingual pre-training including CJK and Arabic-script languages
- Native Uyghur Arabic script handling confirmed in tokenizer vocabulary (verify in pre-flight check)
- Apache 2.0 license — no restrictions on research use
- Fine-tuned with QLoRA (4-bit NF4 base + bf16 LoRA adapters + gradient checkpointing)

### Secondary — LLaMA-3.1-8B-Instruct
- Used for Mix-20 fine-tune (stretch goal) and as zero-shot baseline (core)
- Meta LLaMA 3 Community License — gated but instant on HuggingFace
- Same QLoRA configuration as Qwen2.5

### Design decision — no vocabulary surgery
Neither Qwen2.5 nor LLaMA-3.1 undergoes vocabulary expansion or tokenizer modification. This is a deliberate contrast against CUTE-Llama-P's approach and one of the project's research contributions. The pre-flight tokenizer check validates that this decision holds (token/byte ratio < 0.6).

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Method | QLoRA | 4-bit NF4 base, bf16 adapters, gradient checkpointing |
| LoRA rank | 16 | Can be raised to 32 on the 24 GB slice if headroom allows |
| LoRA alpha | 32 | |
| LoRA target modules | `q_proj, v_proj` | Expand to `k_proj, o_proj` on 24 GB for better adaptation |
| Epochs | 3 | |
| Batch size | 4 (effective 16 with grad accum ×4) | 24 GB allows raising to 8 (effective 32) for faster epochs |
| Max sequence length | 512 tokens | Covers >95% of CUTE-P document lengths |
| Optimizer | paged AdamW 8-bit | Recommended for QLoRA; bf16 LoRA can use `adamw_torch` |
| LR | 2e-4 | Cosine decay, warmup 3% |
| bf16 LoRA flag | `--bf16-lora` | Now fits on 24 GB slice; ~2× faster than QLoRA |

---

## Data Mixing — Catastrophic Forgetting

| Variant | UG/EN CUTE-P | EN-only (FLAN) | Status |
|---------|-------------|-----------------|--------|
| Mix-0   | 100%        | 0%              | Stretch (ablation) |
| Mix-10  | 90%         | 10%             | Stretch (ablation) |
| Mix-20  | 80%         | 20%             | **Core — default** |
| Mix-50  | 50%         | 50%             | Stretch (ablation) |

Mix-20 is the core experiment. Ablation (Mix-0, Mix-10, Mix-50) runs only after core evaluation is complete.

---

## Evaluation Plan

### Core evaluation
| Benchmark | Task | Metric | Models evaluated |
|-----------|------|--------|-----------------|
| FLORES-200 | EN→UG translation | chrF, BLEU | Qwen fine-tune, Qwen zero-shot, LLaMA zero-shot, CUTE-Llama-P (if available) |
| FLORES-200 | UG→EN translation | chrF, BLEU | Same |
| WCM-v2 | Uyghur text classification | Accuracy | Same |
| Held-out English split (C4, 1K samples) | Perplexity | PPL | Qwen fine-tune vs Qwen zero-shot (catastrophic forgetting check) |

### Stretch evaluation
| Benchmark | Task | Status |
|-----------|------|--------|
| FLORES-200 (LLaMA fine-tune) | EN↔UG translation | After LLaMA fine-tune stretch goal |
| WCM-v2 (LLaMA fine-tune) | Classification | Same |
| FLORES-200 + WCM-v2 (ablation variants) | Translation + classification | After ablation stretch goal |
| MiLiC-Eval (9 tasks) | Multi-task bilingual | Deferred to final report |

### Success criteria (pre-registered)
| Level | Criterion |
|-------|-----------|
| **Minimum** | Fine-tuned Qwen2.5 beats zero-shot Qwen2.5 on FLORES chrF in at least one direction |
| **Target** | Fine-tuned Qwen2.5 within 2 chrF points of CUTE-Llama-P on EN→UG AND beats it on UG→EN |
| **Stretch** | Ablation reveals a statistically clear Mix-ratio sweet spot; LLaMA fine-tune results available |

### What we will NOT claim
- No human evaluation of translation quality
- No claims about languages other than English and Uyghur
- No production deployment or inference latency benchmarks
- No comparison against CUTE paper's ZH→UG numbers (direction mismatch)

---

## Research Contributions

Relative to the CUTE paper (Zhuang & Sun, COLING 2025):

1. **EN↔UG direction**: The paper focuses on Chinese as a pivot language (ZH→UG). We evaluate English↔Uyghur directly — more practically relevant for international use and not published with CUTE data.
2. **LoRA vs. continued pretraining**: The paper uses full continued pretraining + vocabulary expansion on Llama2. We test whether parameter-efficient instruction fine-tuning (QLoRA) on a model with native multilingual vocabulary achieves comparable results with far less compute.
3. **Instruction capability beyond translation**: We measure direct Uyghur instruction-following where prompts and responses are both in Uyghur (via WCM-v2 and, if reached, MiLiC-Eval).
4. **EN↔UG asymmetry analysis**: We document and analyze the expected performance gap between generation directions as a contribution to understanding low-resource generative NLP.

---

## Project Timeline

Order is set so we always have **comparable results** in hand before moving
on to the next harder block.

| Week | Milestone | Type |
|------|-----------|------|
| 1 | Pre-flight checks (tokenizer, VRAM, CUTE-Llama-P load test, CUTE-P download); evaluation harness runs on **zero-shot Qwen2.5** and **zero-shot LLaMA-3.1** → first FLORES-200 + WCM-v2 numbers in the artifacts | **Core** |
| 2 | QLoRA fine-tune Qwen2.5-7B Mix-20; evaluate on FLORES-200 + WCM-v2; perplexity check → fine-tuned vs zero-shot delta reported | **Core** |
| 3 | Add **CUTE-Llama-P** to the eval (few-shot prompted) and complete the full baseline table; design presentation. Optionally start LLaMA-3.1-8B QLoRA Mix-20 | Core + Stretch |
| 4 | Ablation Mix-{0,10,50} on Qwen2.5 (stretch); LLaMA-3.1 fine-tune finishes; MiLiC-Eval if time allows | Stretch |
| 5 | Analysis, EN↔UG asymmetry discussion, write-up | **Core** |

---

## Compute Environment

Cluster: **`slurm.hj.se`** (Jönköping University), accessed via SSH alias `ju-compute-server` (`mach25ku@jth-ai-01.hj.se:50001`).

| Property | Value |
|----------|-------|
| Worker nodes | 7 (`worker1`…`worker7.slurm.hj.se`) |
| Per-node CPU | 16 cores |
| Per-node RAM | 128 GB |
| Per-node GPU (physical) | **NVIDIA A100 80GB PCIe** |
| Per-job GPU (effective) | **~24 GB MIG slice** — confirmed via `nvidia-smi -L` inside an `srun` job. Replaces the earlier `1g.10gb` (~10 GB) profile after the admin-side upgrade. |
| Partition `priority` | MaxTime 5 days, default 2 h |
| Partition `scavenger` | Unlimited time, preemptible |
| Account / QoS | `tmls22` / `normal` |
| Concurrency | Up to 7 parallel single-GPU jobs |
| `scripts/push.py` default `--time` | `5-00:00:00` (5 days — the full `priority` partition cap) |

**MIG implications (24 GB slice):**
- **QLoRA stays the default** (4-bit NF4 base + bf16 adapters + gradient checkpointing) — peak VRAM ~8–12 GB, leaves comfortable headroom.
- **bf16 LoRA on a 7–8B model now fits** (bf16 7B ≈ 14 GB + bf16 adapters + activations ≈ 18–22 GB). Enable via `--bf16-lora` for ~2× faster training; same code path.
- A single `--partition priority` job at the push.py default (5 days, the partition cap) comfortably covers `preprocess + train + eval` for one Qwen QLoRA Mix-20 fine-tune (~6–10 h for 3 epochs on full CUTE-P + ~1 h evaluation). The headroom absorbs requeues, longer eval, or running multiple Mix variants in one job.
- Full ablation (Mix-{0,10,20,50} × {Qwen, LLaMA} = 8 jobs) can run in parallel across 7 workers (one queued) if stretch goals are reached.
- **For each new job:** read `nvidia-smi -L` at startup. Expected: ~24 GB visible. If the slice profile changes, update batch size / sequence length before proceeding.

---

## Per-run Artifacts

Each run writes to `results/run_<run_id>/experiment_<N>/`:
- `artifacts/run_config.json`, `run_status.json`
- `artifacts/eval_<benchmark>.json` per evaluation benchmark
- `artifacts/training_history.csv`
- `artifacts/preprocessed_dataset/` — HF-saved preprocess output
- `checkpoints/<model_label>/` — LoRA adapters per epoch
- `logs/<model_label>/` — TensorBoard / TRL training logs

Preflight artifacts (run once per cluster, not per experiment) live under
`results/preflight/`: `checkN.json`, `preflight_report.md`,
`cute_p_sample/`.

`run_status.json` is updated as the pipeline progresses (`started` →
`preprocessed` → `training` → `trained` → `evaluating` → `evaluated`).
`scripts/check.py` reads it and prints the current stage of the latest run.

---

*Last updated: May 2026 — scope refined per `PROJECT_REFINEMENT.md`. Core experiment: Qwen2.5-7B-Instruct QLoRA Mix-20 evaluated on FLORES-200 + WCM-v2. CUTE-Llama-P is back as a planned core baseline now that the 24 GB MIG slice and preflight check 5 confirm it loads and produces Uyghur output (few-shot prompted; protocol difference noted). EN↔UG asymmetry documented as expected outcome. MiLiC-Eval deferred to stretch. All stretch goals gated behind core completion. **Compute update:** MIG slice upgraded from `1g.10gb` (~10 GB) to ~24 GB — bf16 LoRA now feasible; QLoRA remains default; thresholds and headroom revised accordingly.*
