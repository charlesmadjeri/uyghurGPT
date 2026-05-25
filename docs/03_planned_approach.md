# Planned Approach

> Owner: Charles · Task 3 of the design deliverables (`docs/TASKS.md` §177–262)
> Companion file with canonical tables and rationale: `docs/PROJECT.md`
> Scope refinement log: `docs/PROJECT_REFINEMENT.md`

> **Compute update (May 2026):** the MIG slice has been upgraded from
> `1g.10gb` (~10 GB) to **~24 GB**. The two-plan A/B framing below is kept
> as historical context, but the project now operates under what was
> originally "Plan B" (full Qwen2.5 + LLaMA-3.1 + CUTE-Llama-P attempt,
> bf16 LoRA feasible). Canonical thresholds and timings live in
> `docs/PROJECT.md` §Compute environment.

This document specifies what we build and train. Sections 1 (data), 2 (model + learning algorithm) and 4 (Day-1 checks) are independent of the GPU allocation. Section 3 (compute) is presented as **two parallel plans** — **Plan A (10 GB MIG `1g.10gb`)** for the now-deprecated original allocation, and **Plan B (20 GB MIG `2g.20gb` / current ~24 GB)** for the larger slice. Section 5 (risk) is the union, with per-plan rows where they diverge.

The selection between A and B is made at training-launch time via a single CLI flag (`--slice-size {10g, 20g}`); the data pipeline, model identity, evaluation protocol and success criteria are identical.

---

## 1. Data Pipeline

**Source.** CUTE-P (Zhuang & Sun, COLING 2025), `CMLI-NLP/CUTE-Datasets` on HuggingFace, `parallel-corpus/{en,uy}.txt`. We use the EN + UG subset only: 933 989 EN lines aligned to 934 002 UG lines (≈ 10.9 GB on disk; ≈ 100 lines lost on the longer side after alignment — drop them).

> **Known data quality caveat (from Day-1 spot-check).** The EN side of CUTE-P is **machine-translated from Chinese**, not native English. The first 100 lines we inspected read as fluent but artifact-laden technical/encyclopedic prose. This means our fine-tune is learning "EN-from-ZH ↔ UG" rather than "native-EN ↔ UG"; FLORES-200 EN↔UG numbers will be depressed relative to a hypothetical native-EN training set. We document this in the final report's limitations section and frame the project as "best-effort low-resource EN↔UG using the only available parallel data at this scale."

**Preprocessing (`main.py --mode preprocess`) — implemented in `shared/data.py`.**

1. **Load CUTE-P** from `~/uyghurGPT/dataset/{en,uy}.txt`. If missing, `hf_hub_download` fetches only `parallel-corpus/{en,uy}.txt` from `CMLI-NLP/CUTE-Datasets` (~10.9 GB, one-time, atomic write). Line counts cached in `<file>.lines`.
2. **Pair-level train/test split** with `seed=42` *before* bidirectional expansion (`shared/data._split_pair_indices`; locked in by `tests/test_data_split.py`). A pair is held out as a whole — its EN→UG and UG→EN halves always land in the same split.
3. **Stream** each split through `Dataset.from_generator` (`_stream_cute_rows`): one row dict at a time, Arrow writer flushes in 1k-row batches — preprocess peak RAM **< 1 GB** on the full ~934k-pair corpus (no giant Python `list[dict]`).
4. **Expand** every kept pair into **both** directions (`en2ug` and `ug2en`) — not random direction sampling. Each row is conversational `{"messages": [...], "task": ...}`.
5. **Blend FLAN** (`Muennighoff/flan`, reservoir capped by `flan_subset_size=50_000`), same `test_split_pct` holdout on FLAN rows, `concatenate_datasets` into train/test.
6. **Shuffle** each split (`Dataset.shuffle`) and `save_to_disk` → `artifacts/preprocessed_dataset/` (~25–30 GB for full Mix-20).

> **Planned but not implemented (original design doc):** drop empty pairs, drop lines &gt; 512 tokens at preprocess time, Arabic-script filter. Long lines are handled at **train time** via `SFTConfig(max_length=512)` (mapped to `max_length` on TRL 1.4). Outlier lines may log a tokenizer warning during the one-time tokenize pass; they are truncated to 512 tokens in training.

**Instruction templating.** We use Qwen2.5's native ChatML template (the same template applies after a trivial rename for LLaMA-3.1, which uses an equivalent chat format). Both translation directions are formatted as one-turn user→assistant exchanges:

EN→UG (system prompt in code: “helpful bilingual assistant … Translate the English input to Uyghur”):

```
<|im_start|>system
You are a helpful bilingual assistant. Translate the English input to Uyghur.<|im_end|>
<|im_start|>user
{en_text}<|im_end|>
<|im_start|>assistant
{ug_text}<|im_end|>
```

UG→EN mirrors with source/target languages swapped. **Both directions** are emitted for every kept pair (balanced by construction).

At **train time**, `shared/training.py` templates `messages` → `text` and applies **`DataCollatorForCompletionOnlyLM`** with response template `<|im_start|>assistant\n` (TRL 1.4 on cluster). `assistant_only_loss=True` remains a fallback when the collator is unavailable (`PROJECT_REFINEMENT.md` §10).

**Data mix (Mix-20).** 80 % CUTE-P instruction pairs · 20 % English-only FLAN samples (`Muennighoff/flan`, 50 000 random instructions with `seed=42`). FLAN samples wear the same ChatML template (`user: {instruction}\nassistant: {response}`). Mix-20 is the **core experiment**; Mix-{0, 10, 50} are stretch ablation cells.

**Splits (three-way; implemented).**

| Split    | Source                                                              | Used for                                                                       | Code reference                       |
| -------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `train`  | ~95 % of CUTE-P pairs (pair-level) + matching FLAN rows             | Gradient updates                                                               | `shared/training.py`                 |
| `test`   | ~5 % held-out CUTE-P pairs + matching FLAN rows (configurable `test_split_pct`) | In-loop `eval_loss` every `eval_steps` → **overfit detector** in TensorBoard; also drives `EarlyStoppingCallback` and `load_best_model_at_end` | `shared/training.py`, `experiments/experiment_1/config.py` |
| `eval`   | **External, never seen**: FLORES-200 devtest (`eng_Latn` ↔ `uig_Arab`), WCM-v2 Uyghur (`hfl/wcm-v2` → `minority/ug.txt`), C4 EN held-out PPL | Final reported numbers: **`--experiment 0`** = zero-shot Qwen + Llama; **`--experiment 1`** = fine-tuned Qwen only (`eval_variants` in config) | `shared/evaluation.py`, `experiments/experiment_{0,1}/` |

The CUTE-P FLAN mix ratio (Mix-{0,10,20,50}) is computed against the **training pair count**, so the effective ratio is preserved after holding out the test pairs. Mix-20 means 80 % CUTE-P / 20 % FLAN among the rows the model trains on. See `PROJECT_REFINEMENT.md` §9 for why we moved away from the original `dataset/cute_p_valdev.jsonl` plan.

---

## 2. Model and Learning Algorithm

**Primary base model.** Qwen2.5-7B-Instruct (Apache 2.0). Day-1 evidence: UG token/byte ratio 0.396 (< 0.6 threshold) — tokenization is acceptable without vocabulary surgery. This is one of the project's explicit research positions — see PROJECT.md §"Design decision — no vocabulary surgery."

**Method.** QLoRA — 4-bit NF4 quantized base + bf16 LoRA adapters + gradient checkpointing. Reference: Dettmers et al. 2023, Section 2 (4-bit NormalFloat) for the choice of NF4; Hu et al. 2021, Figure 1 for the B·A rank-decomposition that LoRA inserts on top.

**Training configuration.** The full canonical table is in PROJECT.md §"Training Configuration." The choices that matter for the design critique:


| Parameter               | Value                     | One-line justification                                                                                                           |
| ----------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 4-bit quantization type | NF4                       | Information-theoretically optimal for normally-distributed weights (Dettmers §2); double-quant on top saves ≈ 0.4 bits/parameter |
| Compute dtype           | bf16                      | A100 supports it natively, no scale-loss issues vs fp16 on Ampere                                                                |
| LoRA rank               | 16                        | Empirically sufficient for instruction tuning at 7–8B scale; trainable params ≈ 1 % of base                                      |
| LoRA alpha              | 32                        | Standard 2× rank; effective scaling factor α/r = 2                                                                               |
| LoRA target modules     | `q_proj`, `v_proj`        | Standard LoRA-on-attention; cheapest configuration that still updates attention's dominant subspaces                             |
| Epochs                  | 3                         | Trades off catastrophic forgetting against fitting; same as CUTE-Llama-P paper for comparability                                 |
| Optimizer               | paged AdamW 8-bit         | Required by the QLoRA recipe; quantizes optimizer state to avoid VRAM blow-up                                                    |
| Learning rate           | 2e-4                      | Cosine decay, 3 % warmup — QLoRA standard from Dettmers Table 4                                                                  |
| Gradient checkpointing  | on, `use_reentrant=False` | Saves activation memory at ~20 % wall-clock cost; non-reentrant variant has smaller saved-state footprint                        |
| Response masking        | yes (assistant-only loss) | `DataCollatorForCompletionOnlyLM` on templated `text` (primary); `assistant_only_loss` fallback (`PROJECT_REFINEMENT.md` §10) |
| Seeding                 | `transformers.set_seed(42)` + `SFTConfig(seed=42, data_seed=42)` | Reproducible data shuffling, model init, dropout, and DataLoader order. GPU is still best-effort due to cuDNN nondeterminism. |
| In-loop validation      | `eval_strategy="steps"`, `eval_steps=50` on held-out `test` split  | Overfit detector in TensorBoard (`train/loss` vs `eval/loss`); see splits table above |
| Early stopping          | `EarlyStoppingCallback(patience=3)` + `load_best_model_at_end=True, metric_for_best_model="eval_loss"` | Stops when `eval_loss` stalls; final adapter is the best checkpoint seen, not the last (`PROJECT_REFINEMENT.md` §10) |


Batch size, sequence length and attention implementation depend on the slice (Plan A vs Plan B below).

**Why no vocabulary surgery.** The CUTE-Llama paper adds ~100 K Uyghur/Tibetan/Chinese tokens on top of Llama 2 and continues pretraining — a multi-week effort. Day-1 evidence shows Qwen2.5's native tokenizer already segments UG at byte ratios comparable to its native EN handling (×2 worse, not ×10 worse). The trade-off "spend a multi-week vocab+CPT phase vs. spend nothing and live with 2× longer UG sequences" lands clearly on the latter for a 5-week project. This is the contrast that gives the project its research framing (see §Research Contributions in PROJECT.md).

---

## 3. Compute Plan — Two Variants

The Day-1 QLoRA memory checks failed at the PROJECT.md-default config (`bsz=4, seq=512, eager attention`) on the current MIG `1g.10gb` allocation: peak VRAM at OOM was **9.118 GB on Qwen2.5-7B** and **8.551 GB on LLaMA-3.1-8B** (during transformers' `_initialize_missing_keys`, before forward). CUTE-Llama-P (155 K-vocab continued-pretrained Llama 2) fails to load entirely at peak **9.284 GB** during weight materialization.

This produced two parallel paths: **Plan A** (engineer around the slice — tune QLoRA for sub-9 GB peak) and **Plan B** (request a larger slice).

### 3.A — Plan A: 10 GB MIG `1g.10gb` (current allocation)

**Headline:** Qwen2.5-7B Mix-20 QLoRA fine-tune is achievable on the current slice with three zero-quality-cost QLoRA config tweaks. LLaMA-3.1-8B fine-tune and CUTE-Llama-P baseline are **not** achievable on this slice; both are dropped from this plan.

**Memory-saving QLoRA tweaks applied (vs PROJECT.md defaults).**


| Knob                            | PROJECT.md default   | Plan A override                                  | Estimated saving (bsz=1, seq=512)         | Quality cost                    |
| ------------------------------- | -------------------- | ------------------------------------------------ | ----------------------------------------- | ------------------------------- |
| `attn_implementation`           | `eager`              | `sdpa` (FlashAttention-2 on A100)                | 300–400 MB                                | None                            |
| `PYTORCH_CUDA_ALLOC_CONF`       | unset                | `expandable_segments:True,max_split_size_mb:128` | 150–300 MB effective (less fragmentation) | None                            |
| `gradient_checkpointing_kwargs` | `use_reentrant=True` | `use_reentrant=False`                            | 80–150 MB                                 | None (different impl)           |
| `batch_size`                    | 4                    | **1**                                            | 75 % activation reduction                 | Compensated by `grad_accum=32`  |
| `grad_accum`                    | 4 (effective bs 16)  | **32** (effective bs 32)                         | —                                         | Same effective batch as default |


If after the three "free-win" knobs Qwen forward still exceeds 9 GB peak, the next, cheapest fallback is `**seq_len 512 → 384`** (saves ~150 MB; truncates < 5 % of CUTE-P pairs). Beyond that we go to `seq_len=256`, which truncates ~30 % of CUTE-P and damages training quality more meaningfully — only invoke that if every prior knob has been exercised.

**Configuration (Plan A).**


| Parameter              | Value                                     |
| ---------------------- | ----------------------------------------- |
| Base model             | Qwen2.5-7B-Instruct (only)                |
| Batch size             | 1                                         |
| Gradient accumulation  | 32 (effective batch 32)                   |
| Max sequence length    | 512 (fallback 384)                        |
| Attention              | SDPA (FlashAttention-2)                   |
| Gradient checkpointing | non-reentrant                             |
| Allocator              | `cudaMallocAsync` + `expandable_segments` |
| Other QLoRA params     | as in PROJECT.md §Training Configuration  |


**Wall-clock estimate (Plan A).** ~28 h per 3-epoch core fine-tune on Mix-20 (slower than PROJECT.md's "16–28 h" range because bsz=1 + grad-accum-32 has more optimizer overhead than bsz=4 + grad-accum-4 even at the same effective batch). Comfortably under the `priority` partition's 5-day cap.

**What Plan A explicitly drops or demotes.**

- **LLaMA-3.1-8B fine-tune.** The 8 B model OOMs during weight init at 8.5 GB before any of the activation-saving knobs apply. We could rescue it via CPU-offload of `embed_tokens` and `lm_head` (knob 8 in our memory analysis: saves ~2 GB on GPU, costs ~5× forward-pass latency). At ~5× slower, the fine-tune wall-clock blows the timeline → **drop from Plan A**. Zero-shot LLaMA-3.1-8B as a baseline is still possible because inference does not require optimizer state.
- **CUTE-Llama-P baseline.** Structurally impossible to load on 10 GB at any QLoRA config — its expanded vocabulary (~155 K tokens) means the unquantized embedding tables alone are ~2.5 GB in bf16. CPU offload makes it loadable but inference becomes prohibitively slow. **Plan-A fallback (historical):** declare baseline FAILED and use zero-shot Qwen2.5 + zero-shot LLaMA-3.1 as the only baselines. *(Superseded: on the current ~24 GB slice, CUTE-Llama-P loads in 4-bit NF4 and is back as a core baseline — see `PROJECT.md` §CUTE-Llama-P Baseline.)*
- **Stretch ablation cells.** Mix-{0, 10, 50} on Qwen are still runnable serially (4 cells × ~28 h ≈ 4.7 days), but not in parallel because we only have 3 concurrent-job quota on the cluster. Drop to 2 cells (Mix-0 and Mix-50) if timeline gets tight.

### 3.B — Plan B: 20 GB MIG `2g.20gb` (requested from admins)

**Headline:** All three models — Qwen2.5-7B fine-tune, LLaMA-3.1-8B fine-tune, CUTE-Llama-P baseline — are reachable with comfortable headroom. PROJECT.md's default training config applies as-written. Wall-clock is roughly 1.5× faster than Plan A on the same fine-tune.

**Estimated memory headroom (Plan B).**


| Model                                      | Estimated peak VRAM (default config) | Slice headroom |
| ------------------------------------------ | ------------------------------------ | -------------- |
| Qwen2.5-7B QLoRA (bsz=4, seq=512, eager)   | ~9–10 GB                             | ~10 GB free    |
| LLaMA-3.1-8B QLoRA (bsz=4, seq=512, eager) | ~10–11 GB                            | ~9 GB free     |
| CUTE-Llama-P inference (4-bit, no LoRA)    | ~10–11 GB                            | ~9 GB free     |


**Configuration (Plan B).** PROJECT.md §"Training Configuration" verbatim (`bsz=4, grad_accum=4, seq=512, eager attention OK, paged AdamW 8-bit, LoRA r=16, q+v`). The Plan A knobs (SDPA, allocator config, non-reentrant ckpt) are still recommended as defensive hygiene but no longer load-bearing.

**Wall-clock estimate (Plan B).** ~16–20 h per 3-epoch core fine-tune (close to PROJECT.md's original 16–28 h estimate, lower end of range).

**What Plan B enables that Plan A doesn't.**

- **LLaMA-3.1-8B Mix-20 fine-tune** moves from stretch to core — we get two fine-tuned models compared head-to-head on FLORES-200 + WCM-v2.
- **CUTE-Llama-P as a live baseline** — recovers the "LoRA vs. continued pretraining" research framing PROJECT.md was originally built around. If CUTE-Llama-P's inference test produces valid Uyghur (Day-1 check 5), it becomes the strong baseline; if not, fall back to zero-shot LLaMA-3.1-8B as the strong baseline.
- **Parallel ablation cells** — `priority` partition with the 3-concurrent-job quota lets us run 3 of 4 Mix-{0, 10, 50} cells simultaneously; ablation completes in one wall-clock window instead of four.

### 3.C — Decision Rule

The plan is chosen at training-launch time, not now.


| Event                                                  | Action                                                                                                                                                |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Admin grants `2g.20gb` (or larger) within Week 1       | Run under Plan B. Re-attempt CUTE-Llama-P load (Day-1 check 5) on the new slice.                                                                      |
| Admin response is "no" or no response by end of Week 1 | Commit to Plan A. Lock the demoted scope (LLaMA-3.1-8B fine-tune and CUTE-Llama-P baseline → not pursued). Document the demotion in the final report. |


The CLI flag `--slice-size {10g, 20g}` selects between the two configs; the rest of the training code is identical.

### 3.D — Slurm Submission (both plans)

`--partition priority`, `--gres=gpu:1`, `--time=5-00:00:00`, `--cpus-per-task=8` (push default), `--mem=24G` (matches 24 GB VRAM; see `SERVER_CONFIG.md` §4.0.1). Job submission via `scripts/run_preflight.py` (preflight) and `scripts/push.py` (preprocess / train / eval). The wrap sources `.env` so `HF_TOKEN` reaches the compute node; `python -u` + `PYTHONUNBUFFERED=1` for live Slurm logs.

---

## 4. Day-1 Sanity Checks (Results)

The five mandatory checks from PROJECT.md §"Pre-flight Sanity Checks." Performed; aggregate report in `results/preflight/preflight_report.md`. Individual JSONs in `results/preflight/check{N}.json`.


| #   | Check                           | Pass condition                                 | Current state                                                                                                                           |
| --- | ------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Tokenizer Uyghur segmentation   | UG token/byte ratio < 0.6 for both base models | **PASS** — Qwen 0.396, LLaMA-3.1 0.460                                                                                                  |
| 2   | QLoRA memory fit — Qwen2.5-7B   | Peak VRAM < 9.5 GB on MIG `1g.10gb`            | **FAIL** at 9.118 GB on PROJECT.md-default config → re-run with Plan A knobs expected to PASS                                           |
| 3   | QLoRA memory fit — LLaMA-3.1-8B | Same threshold                                 | **FAIL** at 8.551 GB during `_initialize_missing_keys` → structural; **Plan A drops LLaMA fine-tune**, Plan B fits without intervention |
| 4   | CUTE-P EN+UG download + format  | No mojibake; ≥ 80 % UG in Arabic script        | **PASS** — 100 / 100 UG lines in Arabic script; 1 / 100 stray U+FFFD (within tolerance)                                                 |
| 5   | CUTE-Llama-P load + inference   | Loads and produces UG output                   | **FAIL** — OOM at 9.284 GB during weight load on 10 GB slice → **Plan A demotes baseline**, Plan B retries on the larger slice          |


**Interpretation.** Tokenizer + data integrity checks (1, 4) are clean; we have a sound corpus and the tokenizer choice is validated. The three GPU-fit checks (2, 3, 5) tell a coherent story: 7–8 B models with QLoRA are right at the edge of the 10 GB slice. Check 2 is recoverable on the same slice via config tuning (Plan A); checks 3 and 5 require the larger slice (Plan B) to be reachable at design-relevant wall-clock.

---

## 5. Risk and Mitigation

Risks marked **(A)** apply to Plan A only; **(B)** to Plan B only; otherwise both.


| Risk                                                          | Probability                           | Impact         | Mitigation                                                                                                                                           |
| ------------------------------------------------------------- | ------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen forward still OOMs after Plan A knobs (~0.4 GB shortage) | Low (≈ 15 %)                          | Medium         | Drop `seq_len` 512 → 384 (knob 4 in memory analysis); ~5 % CUTE-P truncation                                                                         |
| Qwen forward OOMs even at `seq_len=384`                       | Very low (≈ 5 %)                      | High (A)       | Drop to `seq_len=256` and document the data-truncation cost; or escalate Plan B request                                                              |
| Admin denies / delays `2g.20gb` request                       | Medium (≈ 40 %)                       | Medium         | Plan A is fully scoped and runnable on current allocation; project completes core experiment regardless                                              |
| **(B)** CUTE-Llama-P still won't load on the 24 GB slice      | Very low (preflight check 5 PASS)     | Low            | Fallback to zero-shot-only baselines documented in `PROJECT.md` §CUTE-Llama-P Baseline                                                              |
| Tokenizer fragments Uyghur (byte-level)                       | Already resolved — Day-1 check 1 PASS | Low            | None needed                                                                                                                                          |
| **(A)** LLaMA-3.1 fine-tune dropped from scope                | Certain (Plan A)                      | Medium         | Zero-shot LLaMA-3.1 is still a viable cross-architecture baseline; demotion documented in the report                                                 |
| **(A)** CUTE-Llama-P baseline dropped                         | Plan A only (historical)              | Low            | Contribution reframes from "LoRA vs. CPT" to "LoRA vs. zero-shot multilingual LLM"; still publishable (per PROJECT.md §CUTE-Llama-P Baseline fallback) |
| EN side of CUTE-P is auto-translated from ZH                  | Certain                               | Low (expected) | Document as a known data limitation in the final report; framed honestly as "best-effort given the only available large-scale EN↔UG parallel corpus" |
| UG→EN >> EN→UG (expected asymmetry)                           | Certain                               | Low (expected) | Pre-empt in presentation; report directions separately, never averaged (PROJECT.md §Expected Result Asymmetry)                                       |
| Wall-clock exceeds `priority` 5-day cap                       | Very low (any single run is < 30 h)   | Low            | Submit on `priority` first; if preempted, requeue or fall back to `scavenger`                                                                        |


---

## Slide Notes

### Slide 5 — "Approach Overview" (~1 min)

**Visual:** left-to-right pipeline diagram.

```
[CUTE-P EN↔UG, 934K pairs]   [FLAN 20%, 50K]
              \                    /
               \                  /
                ↓                ↓
       [Qwen ChatML formatting, response masking]
                          ↓
     [Qwen2.5-7B-Instruct + QLoRA (4-bit NF4 + bf16 LoRA r=16)]
                          ↓
      [FLORES-200 EN↔UG  +  WCM-v2 UG-classification]
```

**Caption (one sentence, placed below the diagram):**
"Single fine-tune, two compute paths (10 GB / 20 GB MIG) depending on cluster slice."

**Speaker notes:**

- Start at the data sources (left). CUTE-P is the only large-scale EN↔UG parallel corpus available — ~934 K aligned sentence pairs from `CMLI-NLP/CUTE-Datasets`. We add a 20 % FLAN slice (English-only) to defend the model's existing English instruction-following from catastrophic forgetting.
- Center: every training example is formatted into Qwen's native ChatML template (system + user + assistant). The loss is masked to **assistant tokens only** — the model is never penalised for failing to reproduce the prompt template.
- Right: evaluation is on two **unseen** datasets — FLORES-200 devtest for EN↔UG translation (chrF first, BLEU second), WCM-v2 for Uyghur text classification. Neither overlaps with training data.
- End on the caption: one training pipeline, two compute paths. Which path we take is a function of whether cluster admins grant the larger MIG slice (Slide 7).

---

### Slide 6 — "Why QLoRA + Qwen2.5" (~1 min)

**Visual:** two side-by-side boxes, with a small LoRA B·A figure (Hu et al. 2021, Figure 1) in the corner.

| QLoRA | Qwen2.5-7B-Instruct |
|---|---|
| Fits a 7 B model in ~6–9 GB | Apache 2.0 — no licensing friction |
| Trains adapters only (~1 % of params, rank 16) | Native multilingual vocabulary inc. Arabic script |
| No vocabulary surgery needed | Strong zero-shot baseline built into the **same** model |
| Day-1 check: UG token/byte ratio 0.396 < 0.6 ✓ | Same model used fine-tuned and zero-shot ⇒ clean LoRA isolation |

**Caption:**
"We compare to ourselves zero-shot to isolate what LoRA actually buys us."

**Speaker notes:**

- QLoRA box: the recipe is 4-bit NF4 quantized base + bf16 LoRA adapters + gradient checkpointing. This is what lets a 7 B model train in ~10 GB instead of the ~80 GB a full fp32 fine-tune would need.
- Qwen2.5 box: the Apache 2.0 license matters practically — no gating, no agreement, no friction. The Day-1 tokenizer check is the empirical justification for skipping vocabulary surgery: Qwen segments Uyghur at ~2× the rate of English, not ~10× like a Latin-only tokenizer would.
- The deeper design choice: by using zero-shot Qwen as one of our baselines, we control for everything the base model already knows. The delta between fine-tuned-Qwen and zero-shot-Qwen is the *causal* contribution of LoRA training on CUTE-P.
- Anticipated Q: "Why not full fine-tune?" → (a) doesn't fit on the slice, (b) ~50× more compute, (c) catastrophically forgets EN; LoRA preserves it by construction (base is frozen).

---

### Slide 7 — "Compute and Risks" (~1 min)

**Visual:** two-row compute table + three-bullet risk list.

| Plan | Slice | Models trained | Baselines | Wall-clock (core) |
|---|---|---|---|---|
| A | 10 GB MIG `1g.10gb` | Qwen2.5 only | Zero-shot Qwen + zero-shot LLaMA-3.1 | ~28 h |
| B | 20 GB MIG `2g.20gb` | Qwen2.5 + LLaMA-3.1 | + CUTE-Llama-P (if loads) | ~16–20 h |

**Top three risks:**

- *Memory budget on 10 GB slice* — mitigated with SDPA attention, `expandable_segments` allocator, non-reentrant gradient checkpointing (~530–850 MB saved at zero quality cost).
- *EN side of CUTE-P is ZH-translated* — documented limitation; affects absolute FLORES scores, not our delta-over-zero-shot framing.
- *EN→UG < UG→EN expected* — pre-empted; directions reported separately, never averaged.

**Caption:**
"Two compute paths; the core Qwen2.5 fine-tune ships under either."

**Speaker notes:**

- Lead with: Plan A is fully scoped and runnable **today** on the current allocation. Plan B is what we *would* run if admins grant a `2g.20gb` slice (request pending).
- The decision between A and B is made at training-launch time, not now. A single CLI flag (`--slice-size`) selects the config; the data pipeline, model identity, and evaluation are identical.
- What we lost under the historical Plan A: LLaMA-3.1-8B fine-tune (stretch goal anyway) and CUTE-Llama-P baseline. On the current 24 GB slice both are back in scope (LLaMA fine-tune remains stretch, CUTE-Llama-P is a planned core baseline — see `PROJECT.md` §CUTE-Llama-P Baseline). The core experiment — Qwen2.5 Mix-20 vs. zero-shot Qwen on FLORES + WCM-v2 — ships either way.
- Anticipated Q: "What if the 20 GB request is denied?" → Plan A is the plan. We lose stretch goals; the core contribution and evaluation are unchanged.
- Anticipated Q: "Why is one run 28 h and the other 16–20 h?" → Plan A uses `bsz=1, grad_accum=32` to fit memory; the optimizer overhead per effective batch is higher than Plan B's `bsz=4, grad_accum=4`, even though the effective batch size is the same.

---

*Document version: written after Day-1 preflight checks (5/5 attempted, 2 PASS / 3 FAIL with diagnosis). Last updated: 2026-05-12.*