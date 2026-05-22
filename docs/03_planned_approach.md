# Planned Approach

> Owner: Charles · Task 3 of the design deliverables (`docs/TASKS.md` §177–262)
> Companion file with canonical tables and rationale: `docs/PROJECT.md`
> Scope refinement log: `docs/PROJECT_REFINEMENT.md`

This document specifies what we build and train. Sections 1 (data), 2 (model + learning algorithm) and 4 (Day-1 checks) are independent of the GPU allocation. Section 3 (compute) is presented as **two parallel plans** — **Plan A (10 GB MIG `1g.10gb`)** for the current allocation, and **Plan B (20 GB MIG `2g.20gb`)** for the larger slice we are requesting from cluster admins. Section 5 (risk) is the union, with per-plan rows where they diverge.

The selection between A and B is made at training-launch time via a single CLI flag (`--slice-size {10g, 20g}`); the data pipeline, model identity, evaluation protocol and success criteria are identical.

---

## 1. Data Pipeline

**Source.** CUTE-P (Zhuang & Sun, COLING 2025), `CMLI-NLP/CUTE-Datasets` on HuggingFace, `parallel-corpus/{en,uy}.txt`. We use the EN + UG subset only: 933 989 EN lines aligned to 934 002 UG lines (≈ 10.9 GB on disk; ≈ 100 lines lost on the longer side after alignment — drop them).

> **Known data quality caveat (from Day-1 spot-check).** The EN side of CUTE-P is **machine-translated from Chinese**, not native English. The first 100 lines we inspected read as fluent but artifact-laden technical/encyclopedic prose. This means our fine-tune is learning "EN-from-ZH ↔ UG" rather than "native-EN ↔ UG"; FLORES-200 EN↔UG numbers will be depressed relative to a hypothetical native-EN training set. We document this in the final report's limitations section and frame the project as "best-effort low-resource EN↔UG using the only available parallel data at this scale."

**Preprocessing (`main.py --mode preprocess`).**

1. Drop pairs where either side is empty after stripping whitespace.
2. Drop pairs where either side exceeds **512 tokens under the chosen base model's tokenizer** (Qwen2.5-7B-Instruct). Day-1 tokenizer evidence: UG token/byte ratio = 0.396, EN = 0.202 — so a ≈ 600-character UG line ≈ 240 tokens. We expect < 5 % of pairs to exceed 512 tokens.
3. Drop any UG line that fails the Arabic-script check (no Arabic characters present) — the Day-1 spot-check found 100 / 100 lines in Arabic script, so this filter is a defensive null-op.
4. Shuffle with `seed=42`; write to `dataset/cute_p_clean.jsonl` (one JSON object per line: `{en, ug}`).
5. Compute and freeze a 1 000-pair held-out **in-domain validation split** (`dataset/cute_p_valdev.jsonl`) sampled before shuffling — used only for periodic during-training loss checks. **Not used for final reporting.**

**Instruction templating.** We use Qwen2.5's native ChatML template (the same template applies after a trivial rename for LLaMA-3.1, which uses an equivalent chat format). Both translation directions are formatted as one-turn user→assistant exchanges:

EN→UG:

```
<|im_start|>system
You are a translator between English and Uyghur.<|im_end|>
<|im_start|>user
Translate to Uyghur: {en_text}<|im_end|>
<|im_start|>assistant
{ug_text}<|im_end|>
```

UG→EN:

```
<|im_start|>system
You are a translator between English and Uyghur.<|im_end|>
<|im_start|>user
Translate to English: {ug_text}<|im_end|>
<|im_start|>assistant
{en_text}<|im_end|>
```

Direction is sampled per-example with `p=0.5` (so each training batch is approximately balanced). The loss is masked to **the assistant tokens only** using `trl.DataCollatorForCompletionOnlyLM` with the response template `<|im_start|>assistant\n` — the model is not penalised for failing to reproduce the system+user preamble.

**Data mix (Mix-20).** 80 % CUTE-P instruction pairs · 20 % English-only FLAN samples (`Muennighoff/flan`, 50 000 random instructions with `seed=42`). FLAN samples wear the same ChatML template (`user: {instruction}\nassistant: {response}`). Mix-20 is the **core experiment**; Mix-{0, 10, 50} are stretch ablation cells.

**Splits.**

- Train: all of `cute_p_clean.jsonl` + 50 K FLAN samples, mixed Mix-20.
- In-domain validation (sanity only): 1 000 held-out CUTE-P pairs.
- Final evaluation: **FLORES-200 devtest** (1 012 sentences `eng_Latn` ↔ `uig_Arab`) + **WCM-v2** (Uyghur text classification). Both are unseen by the training data.

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
| Response masking        | yes                       | We never want to fit the prompt template                                                                                         |


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
- **CUTE-Llama-P baseline.** Structurally impossible to load on 10 GB at any QLoRA config — its expanded vocabulary (~155 K tokens) means the unquantized embedding tables alone are ~2.5 GB in bf16. CPU offload makes it loadable but inference becomes prohibitively slow. **Declare baseline FAILED per PROJECT.md §Baseline Risk and use zero-shot Qwen2.5 + zero-shot LLaMA-3.1 as the only baselines.**
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

`--partition priority`, `--gres=gpu:1`, `--time=5-00:00:00`, `--cpus-per-task=4`, `--mem=32G`. Job submission via `scripts/run_preflight.py` (preflight) and `scripts/run_train.py` (training). The wrap command sources a local `.env` so that `HF_TOKEN` reaches the compute node — verified end-to-end on the most recent preflight run.

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
| **(B)** CUTE-Llama-P still won't load on 20 GB                | Low (≈ 15 %)                          | Low            | Already documented as high-risk baseline (PROJECT.md §Baseline Risk) with hard 2-day budget and zero-shot LLaMA-3.1 fallback                         |
| Tokenizer fragments Uyghur (byte-level)                       | Already resolved — Day-1 check 1 PASS | Low            | None needed                                                                                                                                          |
| **(A)** LLaMA-3.1 fine-tune dropped from scope                | Certain (Plan A)                      | Medium         | Zero-shot LLaMA-3.1 is still a viable cross-architecture baseline; demotion documented in the report                                                 |
| **(A)** CUTE-Llama-P baseline dropped                         | Certain (Plan A)                      | Low            | Contribution reframes from "LoRA vs. CPT" to "LoRA vs. zero-shot multilingual LLM"; still publishable (per PROJECT.md §Baseline Risk fallback plan)  |
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
- What we lose under Plan A: LLaMA-3.1-8B fine-tune (stretch goal anyway) and CUTE-Llama-P baseline (declared FAILED per `PROJECT.md` §Baseline Risk). The core experiment — Qwen2.5 Mix-20 vs. zero-shot Qwen on FLORES + WCM-v2 — ships either way.
- Anticipated Q: "What if the 20 GB request is denied?" → Plan A is the plan. We lose stretch goals; the core contribution and evaluation are unchanged.
- Anticipated Q: "Why is one run 28 h and the other 16–20 h?" → Plan A uses `bsz=1, grad_accum=32` to fit memory; the optimizer overhead per effective batch is higher than Plan B's `bsz=4, grad_accum=4`, even though the effective batch size is the same.

---

*Document version: written after Day-1 preflight checks (5/5 attempted, 2 PASS / 3 FAIL with diagnosis). Last updated: 2026-05-12.*