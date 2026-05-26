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

### Pending re-runs (in flight)

- `qwen_zeroshot` + `llama_zeroshot` WCM under constrained-LL — **Slurm
  2749 / `run_20260526_223852`** (full exp-0 re-eval; FLORES + C4 are
  re-computed but expected byte-identical, the new information is the
  WCM column). Submitted with `push.py`'s default `--time 6:00:00`.
- `cute_llama_p` full eval (Task 01) — **Slurm 2750 /
  `run_20260526_222254`** (re-submitted with `--time 1-00:00:00` after
  Slurm 2745 timed out at 6 h on `[eval] 50/1012` EN→UG).
- *(Deferred, not on the cluster yet — see `TODO.md`)* UG→EN
  per-sentence failure-mode diagnostic (`scripts/debug_ug2en.py`).
  Waiting for a queue slot to free; informational, not on the critical
  path for the §2 table.

---

## 2. Final results — core experiments

Latest measured number per cell. `pending` = the numerical value at the
current protocol has not yet been written by a successful Slurm run;
the source run for each populated cell is noted under the table.

| Variant | FLORES EN→UG chrF | EN→UG BLEU | FLORES UG→EN chrF | UG→EN BLEU | WCM Uyghur acc. | C4 EN PPL |
|---------|-------------------|------------|-------------------|------------|------------------|-----------|
| `qwen_zeroshot`   | 9.96      | 0.23   | **30.29** | 4.09   | 6.33 % *(pending re-eval — free-form scoring)* | 16.59 |
| `llama_zeroshot`  | 0.84      | 0.45   | 4.71      | 1.36   | 0.67 % *(pending re-eval — free-form scoring)* | **13.69** |
| `qwen_finetuned`  | **14.1762** | 0.0354 | 9.385   | 0.1387 | **21.00 %** (63 / 300, constrained-LL) | 16.1667 |
| `cute_llama_p`    | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

Sources for populated cells (latest measurement per metric):

| Variant | FLORES rows | WCM row | C4 PPL row |
|---|---|---|---|
| `qwen_zeroshot`  | `run_20260525_143722` (Slurm 2714) | `run_20260525_143722` (Slurm 2714, free-form — *to be superseded*) | `run_20260525_143722` |
| `llama_zeroshot` | `run_20260525_143722`              | `run_20260525_143722` (free-form — *to be superseded*)              | `run_20260525_143722` |
| `qwen_finetuned` | `run_20260524_020432` (Slurm 2744) | `run_20260524_020432` (Slurm 2744, constrained-LL)                   | `run_20260524_020432` (Slurm 2744) |
| `cute_llama_p`   | n/a (Slurm 2745 timed out)         | n/a                                                                  | n/a |

### Analysis (current best estimate)

- **EN→UG: fine-tuning works.** `qwen_finetuned` reaches **14.18 chrF**,
  +4.22 over `qwen_zeroshot` (9.96) and >13 chrF over `llama_zeroshot`
  (0.84). The pre-registered **Minimum** criterion (fine-tuned beats
  zero-shot in ≥1 direction) is met. The pre-registered **Target**
  criterion (+5 chrF in *both* directions) is not — see UG→EN below.
- **UG→EN: real Mix-20 over-fitting, not a decoding artifact.**
  `qwen_finetuned` drops to **9.39 chrF** vs `qwen_zeroshot` 30.29. The
  May-26 post-fix re-eval (Slurm 2744) confirmed the chat-marker fix
  was a no-op on this adapter — the regression is genuine. Mix-20 over-
  anchors on the generate-Uyghur direction at the cost of generate-
  English fluency, despite the 20 % FLAN buffer. C4 PPL gap is only
  +0.4 (16.59 → 16.17), so this is task-shaped, not catastrophic-
  forgetting-shaped. Reported as the headline finding rather than
  engineered away.
- **WCM: constrained scoring lifts `qwen_finetuned` to 21 %**, still
  far below the 85.3 % majority floor. The model now always returns a
  legal label (scoring path is sound); the gap to majority is a real
  "the chat-template prompt does not anchor the Uyghur classifier on
  the right label" finding. The zero-shot WCM cells are still the
  pre-fix free-form numbers and **must be re-run with constrained-LL
  scoring** before any qwen_zeroshot vs qwen_finetuned WCM Δ is
  reported in the paper. Command listed in §1 *Pending re-runs*.
- **BLEU is uniformly tiny on FLORES.** chrF is the right primary
  metric for low-resource Uyghur — token-level BLEU is near zero for
  every variant including the zero-shot baselines. Reported but not
  interpreted.
- **C4 PPL is stable** (16.59 → 16.17 across fine-tuning;
  `llama_zeroshot` lower at 13.69 reflects the LLaMA tokenizer's English
  efficiency, unrelated to Uyghur training). No catastrophic forgetting.

### Outstanding before the core comparison is complete

1. **`cute_llama_p` row.** Required for the CUTE-Llama-P comparison
   that motivates the project (`docs/01_prob_describtion.md` §1.5).
   Blocked on a successful experiment-2 Slurm run (re-submit with
   `--time 1-00:00:00`).
2. **`qwen_zeroshot` / `llama_zeroshot` WCM under constrained-LL
   scoring.** Without this, the WCM column mixes two different scoring
   protocols and the Δ between `qwen_zeroshot` and `qwen_finetuned` is
   not interpretable.
3. **(Optional) per-direction chrF confidence intervals via sacrebleu
   `--paired-bs`.** Currently the eval pipeline reports point estimates
   only; tracked in `docs/04_planned_evaluation.md` §4.3.

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
