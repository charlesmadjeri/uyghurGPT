# Project Results — Run-by-run log

This file is the single source of truth for **what we measured, when, and
why it changed**. Each entry is a `## YYYY-MM-DD — run_<id>` section
containing the snapshot of external benchmark numbers for that run plus a
short analysis of how they moved against the previous entry.

**Convention.** Sections are listed in chronological order (oldest first);
the **last section's table is therefore always the current latest result**.
A new run = a new section appended to the bottom, never a rewrite of an
older one. If a re-evaluation supersedes a number for a given run (e.g.
WCM-v2 backfilled later) the new value is added inside the same section
with a dated sub-bullet so the original snapshot stays auditable.

**Column legend.** `chrF` / `BLEU` from `sacrebleu` on FLORES-200 devtest
(1012 sentences per direction); `WCM` = Uyghur classification accuracy on
`hfl/wcm-v2 → minority/ug.txt` (300 rows); `C4 PPL` = English perplexity on
`allenai/c4/en` validation (1k samples). Variants:

- `qwen_zs` — Qwen2.5-7B-Instruct, zero-shot (no adapter).
- `llama_zs` — LLaMA-3.1-8B-Instruct, zero-shot.
- `qwen_ft` — Qwen2.5-7B-Instruct + the run's LoRA adapter.

---

## 2026-05-24 — run_20260524_020432 (Slurm job 2650)

**Setup.** First successful full pipeline run. Qwen2.5-7B-Instruct QLoRA
Mix-20 on 100k CUTE-P pairs + 50k FLAN rows. `lora_rank=16`, `α=32`,
`lr=2e-4`, `bs=4 × grad_accum=4 = 16`, `max_seq_length=512`,
`epochs=3` planned. Packing + FlashAttention 2 enabled.
`early_stopping_patience=3` over `eval_steps=50`.

**Training health.**

- Preprocess: `num_train=32670`, `num_test=1717` (pair-level split).
- Stopped at **step 1550 / 3138 (~1.48 epochs)** by `EarlyStoppingCallback`
  (not a crash).
- In-loop `eval_loss`: **2.159 → ~1.523** (best). `final/` adapter is the
  best checkpoint via `load_best_model_at_end=True`.
- Last train loss at stop: ~1.721 (`epoch≈1.48`).

**External benchmarks** (from `eval_summary.json`):

| Variant     | FLORES EN→UG chrF | FLORES EN→UG BLEU | FLORES UG→EN chrF | FLORES UG→EN BLEU | WCM Uyghur acc. | C4 EN PPL |
|-------------|-------------------|-------------------|-------------------|-------------------|-----------------|-----------|
| qwen_zs     | 9.96              | 0.23              | **30.29**         | 4.09              | n/a (ERROR)     | 16.59     |
| llama_zs    | 0.84              | 0.45              | 4.71              | 1.36              | n/a (ERROR)     | 13.69     |
| **qwen_ft** | **14.18**         | 0.04              | 9.38              | 0.14              | n/a (ERROR)     | 16.17     |

**Analysis.**

- **EN→UG is the gain direction.** Fine-tuning lifts chrF by **+4.22**
  over zero-shot Qwen (9.96 → 14.18); zero-shot LLaMA stays at chrF ~0.84
  (essentially "no Uyghur"), confirming Qwen's UG tokenization head-start.
- **UG→EN regresses sharply** (chrF 30.29 → 9.38). Mix-20 over-fits the
  *generate-Uyghur* direction; the EN-side decoder is partly forgotten
  despite the 20 % FLAN buffer. The C4 PPL gap is small (16.59 → 16.17),
  so the regression is task-shaped, not catastrophic-forgetting-shaped —
  more likely prompt template / stop-token mismatch on UG→EN. Worth
  re-checking decoding before retraining.
- **BLEU is uniformly tiny.** chrF (character n-gram F-score) is the
  right primary metric for low-resource Uyghur where token-level BLEU is
  near zero for everyone (including the zero-shot baseline). We report
  both but interpret chrF.
- **WCM-v2 missing.** Loader called `load_dataset("hfl/wcm-v2",
  split="test")`, which returns the Chinese parquet with a single `text`
  column. Fix applied (load `minority/ug.txt` via `hf_hub_download`); a
  re-evaluation will backfill the WCM column in this section.
- **Early stop at ~1.5 epochs** is reasonable given the loss curve was
  still trending down very slowly; a longer-patience run is a candidate
  for the next iteration once decoding for UG→EN is sanity-checked.

---

<!--
APPEND NEW SECTIONS BELOW THIS LINE — chronological order, never rewrite
older snapshots. Use this template:

## YYYY-MM-DD — run_<id> (Slurm job <jobid>)

**Setup.** <delta from previous run: config/code changes that matter>

**Training health.** <steps, eval_loss best, early stop, etc.>

**External benchmarks.**

| Variant  | FLORES EN→UG chrF | … | WCM acc. | C4 PPL |
|----------|-------------------|---|----------|--------|
| qwen_zs  | …                 | … | …        | …      |
| llama_zs | …                 | … | …        | …      |
| qwen_ft  | …                 | … | …        | …      |

**Analysis.** <2–6 bullets focused on how each cell moved vs the
previous section and what we believe caused the change>
-->
