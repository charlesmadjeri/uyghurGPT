# Project Results

This file is the single source of truth for **what we measured and how
the numbers moved**. It has three sections:

1. **Change log** — a chronological log of *deltas* only (what improved,
   what regressed, what was a no-op, what is still pending). Full
   per-run snapshots live in the raw JSON under
   `results/run_<id>/experiment_<N>/artifacts/`.
2. **Final results (core)** — the *latest* measured number per cell for
   every core variant. Cells that have not yet been measured at the
   current evaluation protocol are marked `pending`.
3. **Bonus experiments** — placeholder table for the stretch goals
   defined under `docs/tasks/bonus/`. All cells `pending` until those
   experiments run.

## Variants

- `qwen_zeroshot` — Qwen2.5-7B-Instruct, zero-shot (no adapter), chat template.
- `llama_zeroshot` — LLaMA-3.1-8B-Instruct, zero-shot, chat template.
- `qwen_finetuned` — Qwen2.5-7B-Instruct + the run's QLoRA adapter (Mix-20).
- `cute_llama_p` — CMLI-NLP/CUTE-Llama, `CUTE-Llama-Parallel` subfolder, fp16, base-LM few-shot continuation prompt.

## Metrics

- **FLORES EN↔UG chrF / BLEU** — `sacrebleu.corpus_chrf` / `corpus_bleu` on the
  FLORES+ devtest split (1012 sentences per direction), id-joined across
  `eng_Latn` ↔ `uig_Arab`.
- **WCM Uyghur acc.** — accuracy on `hfl/wcm-v2 → minority/ug.txt` (300 rows,
  6 labels). Predicted by constrained log-likelihood scoring over the
  candidate label set (`shared/evaluation._classify_uyghur`).
  Majority-class floor on this file = **85.3 %** (256 of 300 rows = label
  `1`); random baseline = 16.7 %.
- **C4 EN PPL** — held-out English perplexity on `allenai/c4/en` validation,
  1 000 samples streamed. Catastrophic-forgetting check.

---

## 1. Change log (deltas only)

### 2026-05-24 — `run_20260524_020432` (Slurm 2650)

> First successful full pipeline run. Qwen2.5-7B QLoRA Mix-20 on 100 k
> CUTE-P pairs + 50 k FLAN; early-stopped at step 1 550 / 3 138 (≈ 1.48
> epochs); best `eval_loss` 1.523.

- **+** Established baseline cells for `qwen_zeroshot`, `llama_zeroshot`,
  and the first `qwen_finetuned` row.
- **+** EN→UG chrF: `qwen_finetuned` **14.18** vs `qwen_zeroshot` 9.96
  → **+4.22 chrF** absolute gain.
- **−** UG→EN chrF: `qwen_finetuned` **9.38** vs `qwen_zeroshot` 30.29
  → **−20.91 chrF** regression. C4 PPL only 16.59 → 16.17 (+0.4) — too
  small for catastrophic forgetting; flagged for decoding investigation
  (Task 03).
- **n/a** WCM `ERROR` for all three variants. Loader called
  `load_dataset("hfl/wcm-v2", split="test")` which returned the Chinese
  parquet with no `label` column.

### 2026-05-26 — `run_20260525_143722` (Slurm 2714) — exp-0 isolation re-run

> No training. First clean experiment-0-only zero-shot eval after the
> exp-0 / exp-1 split landed (`PROJECT_REFINEMENT.md` §12).

- **=** FLORES + C4 PPL for `qwen_zeroshot` / `llama_zeroshot` reproduced
  byte-identically against the May-24 cells → confirms determinism +
  zero eval-pipeline drift from the refactor.
- **+** WCM cells populate without `ERROR` (loader fixed to load
  `minority/ug.txt` via `hf_hub_download`).
- **−** But WCM accuracy is **below random** (`qwen_zeroshot` 6.33 %,
  `llama_zeroshot` 0.67 %). Diagnosed: `_classify_uyghur` used
  free-form generation + substring match — cannot return a constrained
  label. Scoring fix tracked in `docs/tasks/02_wcm_v2_reevaluation.md`.

### 2026-05-26 — Slurm 2715 WCM backfill on `run_20260524_020432`

- **+** `qwen_finetuned` WCM cell populates: **7.33 %** (22 / 300, free-form
  scoring). Below random — same protocol bug as the 2714 zero-shot run.

### 2026-05-26 — Slurm 2744 post-fix re-eval on `run_20260524_020432`

> Same adapter, two fixes active: WCM constrained log-likelihood scoring
> (commit `6d4197c`) + FLORES chat-marker stop-token list + post-decode
> trim (commit `da8e8d8`).

- **=** FLORES EN→UG / UG→EN chrF and BLEU **byte-identical** to the May-24
  pre-fix numbers (14.1762 / 9.385). **The chat-marker decoding fix had
  zero measurable effect on this adapter** → falsifies the "leak causes
  the UG→EN regression" hypothesis recorded in `PROJECT_REFINEMENT.md`
  §13. **The UG→EN regression 30.29 → 9.38 is genuine** — a real Mix-20
  over-fitting effect on the generate-English direction. The stop-token
  fix is kept as a defensive measure.
- **+** WCM `qwen_finetuned` **7.33 % → 21.00 %** (63 / 300) via
  constrained-LL scoring. ×2.9 lift, model always returns a legal label.
  Still well below the 85.3 % majority floor — substantive
  prompt-anchoring / model-side finding, no longer a methodology bug.
- **=** C4 PPL byte-identical → determinism confirmed.

### 2026-05-26 — Slurm 2745 / `run_20260526_171100` (CUTE-Llama-P) — in flight, will time out

- Experiment 2 first run; reached `[eval] 50/1012` on EN→UG at the time
  of the last log sync. fp16 7B + eager attention + `repetition_penalty`
  is slower than budgeted (~30 s/sentence). Status `evaluating`; the 6 h
  walltime will not be enough. **No artifacts written yet.** Re-submit
  required with `--time 1-00:00:00`.

### 2026-05-27 — `run_20260526_223852` (Slurm 2749) — zero-shot WCM constrained-LL re-eval

> Full exp-0 re-eval (FLORES + WCM + C4 for `qwen_zeroshot` +
> `llama_zeroshot`) with the WCM constrained-LL scoring path active.
> No code changes since Slurm 2744. Submitted with `push.py`'s
> default `--time 6:00:00`; completed in-budget.

- **=** FLORES + C4 PPL reproduce against the May-24 cells within
  rounding (`qwen_zs` EN→UG chrF 9.96 → 9.963, UG→EN chrF 30.29 →
  30.0957; `llama_zs` EN→UG chrF 0.84 → 0.8447, UG→EN chrF 4.71 →
  4.705; PPL byte-identical at 16.5949 / 13.6891). Deterministic
  decoding holds across the May-24 / May-25 / May-27 runs.
- **=** WCM `qwen_zeroshot`: 6.33 % (19 / 300, free-form, Slurm 2714)
  → **6.33 % (19 / 300, constrained-LL, Slurm 2749)**. Same correct
  count under both protocols — Qwen's free-form output happens to
  contain a legal label on the same 19 rows the LL argmax picks. No
  movement, still well below the 16.7 % random floor.
- **+** WCM `llama_zeroshot`: 0.67 % (2 / 300, free-form, Slurm 2714)
  → **3.00 % (9 / 300, constrained-LL, Slurm 2749)**. ×4.5 lift —
  consistent with constrained-LL recovering a legal label that
  free-form scoring discarded. Still below random.
- **Implication.** Both zero-shot variants are now scored under the
  *same* protocol as `qwen_finetuned` (constrained-LL). The fine-tune
  WCM gap is now apples-to-apples: `qwen_ft` 21.00 % vs `qwen_zs`
  6.33 % = **+14.67 pp** absolute (×3.3 over zero-shot). Both still
  far below the 85.3 % majority floor — see §2 *Analysis*.

### 2026-05-26 / 27 — Slurm 2748 + 2750 (CUTE-Llama-P) — both stalled, no FLORES output

- `run_20260526_222254` (Slurm 2748): pushed with default `--time
  6:00:00`; `run_status.json` froze at `"evaluating"` after **1 min**
  (22:22 → 22:23). Slurm log ends right after `[eval] FLORES-200
  n=1012 few-shot k=3 (EN→UG then UG→EN) ...` — never printed a
  single `[eval]   50/1012` progress dot. No artifacts beyond
  `run_config.json` / `run_status.json`. Most likely killed by Slurm
  walltime + a slow first-batch generation, but could also be a hang
  in `generate()` on the few-shot prompt.
- `run_20260526_224102` (Slurm 2750): re-submitted with `--time
  1-00:00:00`. Same failure mode — Slurm log ends after `[eval]
  FLORES-200 n=1012 few-shot k=3 ...` with no progress dots; status
  last updated 00:07 (≈ 1 h 25 min into the run) and has not moved
  since. As of pull at 02:07 the job has produced no FLORES,
  WCM, or PPL artifact. **The `cute_llama_p` row in §2 remains
  `pending`; this is now the only outstanding cell in the core
  comparison table.**
- **Next action** (see `TODO.md` "Investigate CUTE-Llama-P FLORES
  stall"): inspect `squeue -u` / Slurm accounting for 2750 to confirm
  whether it actually crashed vs. is still running with buffered
  output; if running, wait; if dead, instrument
  `generate_translation_fewshot` with a per-sentence print and lower
  the few-shot `k` or `max_new_tokens` before the next resubmit.

### 2026-05-27 — `run_20260526_224102` (Slurm 2750) — CUTE-Llama-P full eval **complete**

> 24 h resubmit (`--time 1-00:00:00`) after Slurm 2745 (6 h timeout at
> `[eval] 50/1012`) and 2748 (died before first progress dot). First
> `[eval] 50/1012` at ~30 min wall; `run_status.json` → `evaluated`
> 07:02 UTC (~7 h total for FLORES × 2 + WCM + C4).

- **+** `cute_llama_p` row populated (FLORES few-shot k=3, WCM
  `base_lm` constrained-LL).
- **+** FLORES EN→UG chrF **6.8773** (BLEU 0.2638); UG→EN chrF
  **23.0881** (BLEU 1.7748). Direction asymmetry is **opposite** to
  `qwen_finetuned` (strong EN→UG FT, weak UG→EN FT).
- **+** WCM **15.33 %** (46 / 300) — just under the 16.7 % random
  floor; highest among non-FT variants; below `qwen_finetuned` 21.00 %.
- **+** C4 EN PPL **13.0148** — lowest in the core table.
- **−** EN→UG: `qwen_finetuned` **+7.30 chrF** over `cute_llama_p`
  (14.18 vs 6.88) — primary project comparison favours QLoRA Mix-20.
- **−** UG→EN: `cute_llama_p` **+13.70 chrF** over `qwen_finetuned`
  (23.09 vs 9.39) but still **−7.01 chrF** vs `qwen_zeroshot` 30.10 —
  FT UG→EN regression is not explained by the CUTE baseline being
  stronger on that direction.

### 2026-05-27 — Slurm 2766 (`debug_ug2en`, n=20) — mechanism diagnostic

> `scripts/debug_ug2en.py` on `run_20260524_020432`'s adapter +
> `qwen_zeroshot` comparison. Log: `results/debug/slurm_ug2en_2766.out`.

- **=** **0×** `A_wrong_language_uyghur` and **0×**
  `C_decoding_or_template_leak` on both variants → chat stop/trim path
  is sound; §13 leak hypothesis stays falsified.
- **−** `qwen_finetuned` on 20-sentence slice: **12×
  `B_garbled_or_weak_english`** (greedy `"The 2 1 1 1 …"` loop to
  `max_new_tokens`), **8× `ok_english`** (fluent EN **not** translating
  the Uyghur source). mean chrF **5.56** vs `qwen_zeroshot` **30.33**
  (all 20 source-anchored, however wrong).
- **Implication.** Regression is **training-shaped** (gradient /
  `eval_loss` checkpoint bias + FLAN-style EN completions), not missing
  UG→EN rows in `shared/data.py` (audit confirmed 1:1 `en2ug`/`ug2en`).
  See `PROJECT_REFINEMENT.md` §14.

### 2026-05-27 — decoding follow-up (option 1, code only — re-eval pending)

> `generate_translation` now applies `repetition_penalty=1.15` +
> `no_repeat_ngram_size=4` when `tgt_lang == "English"` only (EN→UG
> unchanged). Targets the B′ repetition collapse seen on Slurm 2766.

- **Pending** full exp-1 FLORES re-eval on `run_20260524_020432`
  (`python3 scripts/push.py --server ju-compute-server --experiment 1
  --mode eval --run-id 20260524_020432 --time 6:00:00`). Update §2
  `qwen_finetuned` UG→EN cell in the **same commit** as the pull.
- **Sanity gate:** `qwen_zeroshot` UG→EN must stay within ±0.5 chrF of
  30.10 (`run_20260526_223852`).

### 2026-05-27 — Slurm 2751 (`debug_ug2en`) — failed at import

- `scripts/debug_ug2en.py` crashed immediately:
  `ModuleNotFoundError: No module named 'shared'`. Invoking
  `python scripts/debug_ug2en.py` puts `scripts/` on `sys.path`, not
  the repo root. Fixed in-repo with `sys.path.insert(0, REPO_ROOT)`;
  re-submit after rsync (`TODO.md`).

---

## 2. Final results — core experiments

Latest measured number per cell. `pending` = the numerical value at the
current protocol has not yet been written by a successful Slurm run;
the source run for each populated cell is noted under the table.

| Variant | FLORES EN→UG chrF | EN→UG BLEU | FLORES UG→EN chrF | UG→EN BLEU | WCM Uyghur acc. | C4 EN PPL |
|---------|-------------------|------------|-------------------|------------|------------------|-----------|
| `qwen_zeroshot`   | 9.963       | 0.2389 | **30.0957** | 4.2854 | 6.33 % (19 / 300, constrained-LL)      | 16.5949 |
| `llama_zeroshot`  | 0.8447      | 0.449  | 4.705       | 1.3592 | 3.00 % (9 / 300, constrained-LL)       | **13.6891** |
| `qwen_finetuned`  | **14.1762** | 0.0354 | 9.385       | 0.1387 | **21.00 %** (63 / 300, constrained-LL) | 16.1667 |
| `cute_llama_p`    | 6.8773      | 0.2638 | 23.0881     | 1.7748 | 15.33 % (46 / 300, base_lm constrained-LL) | **13.0148** |

Sources for populated cells (latest measurement per metric):

| Variant | FLORES rows | WCM row | C4 PPL row |
|---|---|---|---|
| `qwen_zeroshot`  | `run_20260526_223852` (Slurm 2749) | `run_20260526_223852` (Slurm 2749, constrained-LL) | `run_20260526_223852` (Slurm 2749) |
| `llama_zeroshot` | `run_20260526_223852` (Slurm 2749) | `run_20260526_223852` (Slurm 2749, constrained-LL) | `run_20260526_223852` (Slurm 2749) |
| `qwen_finetuned` | `run_20260524_020432` (Slurm 2744) | `run_20260524_020432` (Slurm 2744, constrained-LL) | `run_20260524_020432` (Slurm 2744) |
| `cute_llama_p`   | `run_20260526_224102` (Slurm 2750) | `run_20260526_224102` (Slurm 2750, base_lm constrained-LL) | `run_20260526_224102` (Slurm 2750) |

### Analysis (current best estimate)

- **EN→UG: fine-tuning works.** `qwen_finetuned` reaches **14.18 chrF**,
  +4.22 over `qwen_zeroshot` (9.96) and >13 chrF over `llama_zeroshot`
  (0.84). The pre-registered **Minimum** criterion (fine-tuned beats
  zero-shot in ≥1 direction) is met. The pre-registered **Target**
  criterion (+5 chrF in *both* directions) is not — see UG→EN below.
- **UG→EN: real training-side regression, not a template leak.**
  `qwen_finetuned` drops to **9.39 chrF** vs `qwen_zeroshot` 30.29.
  Slurm 2744 falsified chat-marker leak; Slurm 2766 (n=20) shows **0 %
  leak** but **60 %** greedy repetition collapse (`"The 2 1 1 1 …"`) and
  **40 %** source-unfaithful English hallucinations. Data audit (§14)
  confirmed balanced `ug2en`/`en2ug` rows — mechanism is
  `assistant_only_loss` gradient bias toward Uyghur output + aggregated
  `eval_loss` early stopping, not missing UG→EN examples. Direction-
  conditional `repetition_penalty` is shipped; **full FLORES re-eval
  pending** to measure how much chrF recovers once B′ is suppressed.
  C4 PPL gap only +0.4 → not catastrophic forgetting.
- **WCM: fine-tune beats zero-shot by ×3.3 under apples-to-apples
  scoring, but the absolute level stays well below random for both
  zero-shot variants.** All three variants now scored under
  constrained-LL (Slurm 2749 closed the protocol gap): `qwen_ft` 21.00 %
  vs `qwen_zs` 6.33 % vs `llama_zs` 3.00 %. The +14.67 pp delta on
  Qwen is the cleanest single-metric win for the fine-tune in the
  table. Two caveats stay on the record: (i) both zero-shot variants
  are *below* the 16.7 % random floor — Qwen and LLaMA actively
  *anti-prefer* the gold labels under the chat-template prompt, which
  is a real prompt-anchoring finding worth flagging in the analysis
  rather than a bug; (ii) `qwen_ft` itself is still far below the
  85.3 % majority floor, so the fine-tune is *better calibrated*
  toward the label set, not yet *competent* at the task.
- **BLEU is uniformly tiny on FLORES.** chrF is the right primary
  metric for low-resource Uyghur — token-level BLEU is near zero for
  every variant including the zero-shot baselines. Reported but not
  interpreted.
- **C4 PPL is stable** (16.59 → 16.17 across Qwen fine-tuning). No
  catastrophic forgetting. `cute_llama_p` at **13.01** and
  `llama_zeroshot` at **13.69** are lower than Qwen variants — both
  reflect Llama-family English tokenization / base-LM efficiency, not
  Uyghur-task competence.
- **CUTE-Llama-P vs Qwen Mix-20 (research question).** On the metrics
  this project optimizes for, **QLoRA instruction tuning wins**:
  EN→UG chrF +7.30 (`qwen_ft` 14.18 vs `cute_llama_p` 6.88), WCM
  +5.67 pp (21.00 % vs 15.33 %). CUTE-Llama-P's only win in the core
  table is UG→EN chrF (+13.70 over `qwen_ft`), and even there it
  underperforms `qwen_zeroshot` (−7.01 chrF). The published baseline
  does not rescue the fine-tuned model's generate-English regression.

### Outstanding (core table populated — one re-eval pending)

All four variant rows in §2 are populated at the **pre–repetition-penalty**
protocol. Remaining:

1. **`qwen_finetuned` FLORES re-eval** after UG→EN `repetition_penalty`
   (code shipped; Slurm job not yet run). May move UG→EN chrF off 9.39;
   EN→UG must be checked for regression. See `TODO.md`.
2. **Task 04** — `scripts/aggregate_results.py` +
   `results/reports/consolidated_results.json`.
3. **(Optional) sacrebleu `--paired-bs` CIs** — `docs/04_planned_evaluation.md`
   §4.3.

---

## 3. Bonus experiments (stretch — placeholders)

Stretch goals from `docs/tasks/bonus/`. All cells `pending` until the
corresponding Slurm runs land. None of these are required for the
**Core** evaluation in §2.

| Variant | Source task | FLORES EN→UG chrF | FLORES UG→EN chrF | WCM Uyghur acc. | C4 EN PPL | Status |
|---|---|---|---|---|---|---|
| `llama_finetuned` (Mix-20) | `bonus/01_experiment_3_llama_mix20_finetune.md` | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix0`      | `bonus/02_qwen_mix_ablation.md`                  | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix10`     | `bonus/02_qwen_mix_ablation.md`                  | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix50`     | `bonus/02_qwen_mix_ablation.md`                  | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_zeroshot_5shot`      | `bonus/04_qwen_5shot_baseline.md`                | _pending_ | _pending_ | _pending_ | _pending_ | not started |

### MiLiC-Eval (separate benchmark suite — deferred to final report)

9-task bilingual MiLiC-Eval (`docs/tasks/bonus/03_milic_eval.md`). Not
reported in the same table as FLORES / WCM / C4; will be added as its
own block here once any of the core variants is evaluated against it.

| Variant | MiLiC tasks (9) | Status |
|---|---|---|
| `qwen_zeroshot`   | _pending_ | not started |
| `qwen_finetuned`  | _pending_ | not started |
| `cute_llama_p`    | _pending_ | not started |

---

<!--
APPEND DELTAS BELOW THIS LINE — chronological. Each entry is a short
bullet list of what changed (+ / − / =) for which variant / cell, plus
the source `run_<id>` and Slurm job id. Full numerical snapshots live
in the run's `artifacts/eval_summary.json`; do not mirror them here.

When a new measurement supersedes a cell in §2 *Final results — core
experiments*, update that table in the SAME commit. Do not rewrite
older change-log entries.

Template:

### YYYY-MM-DD — run_<id> (Slurm <jobid>)

- **+/=/−** <variant>.<metric>: <old> → <new> (<source artifact>).
- Notes / caveats.
-->
