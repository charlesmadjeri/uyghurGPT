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

- **2026-05-26 WCM/eval backfill (Slurm job 2715).** Re-ran `--experiment 1 --mode eval`
  on this run id (resumed adapter `checkpoints/qwen_mix20/final`) with the fixed
  WCM loader (`minority/ug.txt` via `hf_hub_download`). FLORES EN→UG chrF / UG→EN
  chrF and C4 PPL reproduced **byte-identically** (deterministic `do_sample=False`
  decoding holds). Only new cell: **WCM `qwen_ft` accuracy = 7.33 %** (22/300)
  with the *legacy* free-form prompt + substring match — known broken, see the
  next sub-bullet for the post-fix number.
  *Caveat — WCM-v2 is methodologically broken at this protocol.* `minority/ug.txt`
  is heavily imbalanced (256/300 rows = label `1`, i.e. 85.3 % majority-class
  baseline; paper reports CUTE-Llama-P at 87.0 %). All three of our variants land
  **below the 16.7 % random baseline**, i.e. the chat-template free-form prompt
  in `shared/evaluation._classify_uyghur` is making the model emit free-form text
  that incidentally contains a digit, not actually pick a label. Δ +1.0 % between
  `qwen_ft` and `qwen_zs` is noise. Prompt + scoring fix tracked in
  `docs/tasks/02_wcm_v2_reevaluation.md`.

- **2026-05-26 post-fix re-eval (Slurm job 2744).** Re-ran
  `--experiment 1 --mode eval --run-id 20260524_020432` against the **same**
  adapter with both fixes active: WCM constrained log-likelihood scoring
  (commit `6d4197c`) and FLORES chat-marker stop-token + post-decode trim
  (commit `da8e8d8`). Results:

  | Metric | Pre-fix (May 24) | Post-fix (Slurm 2744) | Δ |
  |---|---|---|---|
  | FLORES EN→UG chrF | 14.18  | **14.1762** | ~0.00 |
  | FLORES EN→UG BLEU | 0.04   | **0.0354**  | ~0.00 |
  | FLORES UG→EN chrF | 9.38   | **9.385**   | +0.00 |
  | FLORES UG→EN BLEU | 0.14   | **0.1387**  | ~0.00 |
  | WCM `qwen_ft` accuracy | 7.33 % (free-form) | **21.00 % (63/300)** | +13.7 pp |
  | C4 EN PPL | 16.17 | **16.1667** | ~0.00 |

  Two findings — **both important and one is a hypothesis-falsifier:**

  1. **FLORES chrF / BLEU are byte-identical.** The chat-marker stop-token
     + hard-trim path landed in `da8e8d8` had **zero measurable effect**
     on the corpus chrF in either direction. Mechanistically that means
     `skip_special_tokens=True` was already stripping the *token-id*
     form of `<|im_end|>` cleanly on this adapter, and the adapter is
     *not* emitting the *literal-string* form `"<|im_end|>"` as plain
     text the way we hypothesised in `PROJECT_REFINEMENT.md` §13.
     **Conclusion: the UG→EN regression 30.29 → 9.38 is genuine**, not a
     decoding artifact — Task 03 §Step 4 success criterion lands in the
     "OR the analysis concludes the regression is genuine" branch.
     The stop-token fix is kept as a defensive measure (it would catch
     the failure mode if a future adapter learns it) but is not the
     explanation for what we observed.
  2. **WCM `qwen_ft` rises from 7.33 % → 21.00 %.** Switching from
     free-form generation + substring match to constrained log-likelihood
     scoring quadruples accuracy. The model is now always returning one
     of the legal labels (no fallback). 21 % is still well below the
     85.3 % majority floor, so the FT model isn't refusing — it is
     producing a non-trivial label distribution that is only marginally
     above the 16.7 % random baseline on the 6-class task. WCM-v2 is
     no longer methodologically broken; what remains is a real
     "the chat-template Uyghur prompt does not anchor the model on the
     right label" finding. Whether this is a fine-tune-side or
     prompt-side effect is now answerable by repeating the same eval on
     `qwen_zeroshot` / `llama_zeroshot` with the constrained-LL path
     (Task 02 step 3 — outstanding).

  C4 PPL byte-identical confirms determinism. The April-style
  "no fine-tune row in this section yet" caveat in the next-section
  bullet is superseded by the table above; the canonical
  `qwen_finetuned` row for `run_20260524_020432` is this sub-bullet.

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

## 2026-05-26 — run_20260525_143722 (Slurm job 2714)

**Setup.** First clean **experiment-0-only** zero-shot eval after the
exp-0 / exp-1 split landed (`docs/PROJECT_REFINEMENT.md` §12). No
training, no adapter loaded. Evaluates `qwen_zeroshot` + `llama_zeroshot`
on the same external benchmarks as exp 1 (FLORES+ devtest EN↔UG,
WCM-v2 Uyghur `minority/ug.txt`, C4 EN PPL on 1k samples). Replaces the
legacy combined eval in `run_20260524_020432` for the zero-shot rows.

**External benchmarks** (from `experiment_0/artifacts/eval_summary.json`):

| Variant  | FLORES EN→UG chrF | FLORES EN→UG BLEU | FLORES UG→EN chrF | FLORES UG→EN BLEU | WCM Uyghur acc. | C4 EN PPL |
|----------|-------------------|-------------------|-------------------|-------------------|-----------------|-----------|
| qwen_zs  | 9.96              | 0.23              | **30.29**         | 4.09              | 6.33 % (19/300) | 16.59     |
| llama_zs | 0.84              | 0.45              | 4.71              | 1.36              | 0.67 % (2/300)  | **13.69** |

**Analysis.**

- **FLORES + C4 PPL match the 2026-05-24 combined-run cells to four
  decimals** — same models, same prompts, deterministic decoding.
  Confirms the exp-0/exp-1 refactor introduced no eval drift; the
  zero-shot baseline is now produced by experiment 0 alone and can be
  pinned for the report.
- **WCM-v2 numbers now populate without `ERROR`**, but the absolute
  values are not informative: `minority/ug.txt` is 85.3 % majority-class
  `1`; both zero-shot variants score *below random* (16.7 %). The fix is
  not in the loader (loader is correct, returns 300 rows × `text`/`label`)
  but in `_classify_uyghur` — it uses free-form generation + substring
  matching, which can't return a constrained label. Tracked in
  `docs/tasks/02_wcm_v2_reevaluation.md` (scope extended to also fix the
  prompt + scoring path).
- **No fine-tune row in this run.** `qwen_ft` lives in the
  `run_20260524_020432` section above; the 2026-05-26 sub-bullet there
  is the canonical place for the back-filled `qwen_ft` numbers.
- **Outstanding gap before the minimum-results comparison can ship:**
  Task 01 (CUTE-Llama-P few-shot baseline) still has no run dir.
  Without it the FT-vs-CUTE-Llama-P column of the consolidated table
  (`docs/tasks/04_consolidated_results_table.md`) is missing.

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
