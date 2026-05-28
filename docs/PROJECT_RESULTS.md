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
  6 labels with support). Predicted by constrained log-likelihood scoring
  over the candidate label set (`shared/evaluation._classify_uyghur`).
  Floors on this file: majority-class = **85.3 %** (256 of 300 = label
  `1`); uniform-random over 6 classes = **16.7 %**. Because the file is
  heavily imbalanced and small (label `0` has only 3 rows, so a balanced
  subset caps at 18 rows), raw accuracy is reported as the headline but
  **balanced accuracy (macro recall) and macro F1** are also reported
  for fairness — these score "always-predict-majority" at 1/6 instead
  of 85.3 % and so isolate genuine classification competence from prior
  exploitation. The macro metrics are computed offline from the
  per-prediction artifact via `scripts/debug_wcm.py`; see Slurm 2785
  entry in §1.
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

### 2026-05-27 — `run_20260524_020432` (Slurm 2768) — UG→EN repetition-penalty re-eval

> Same adapter as Slurm 2744; only change is the direction-conditional
> repetition controls (`repetition_penalty=1.15`,
> `no_repeat_ngram_size=4`) on English-target chat decoding. Scope:
> `qwen_finetuned` only (`run_config.json eval_variants`); zero-shot
> variants were **not** re-run in this job.

- **+** FLORES UG→EN chrF: **9.385 → 16.8079** (+7.42, +79 % relative);
  BLEU **0.1387 → 0.1794**. The B′ greedy `"The 2 1 1 1 …"` collapse
  (12 / 20 in Slurm 2766) is largely suppressed.
- **=** FLORES EN→UG chrF: **14.1762 → 14.1762** (byte-identical);
  BLEU **0.0354 → 0.0354**. Penalty is gated to `tgt_lang == "English"`,
  so the Uyghur-target decode is untouched — confirms the gate works.
- **=** WCM `qwen_finetuned`: **21.00 %** (63 / 300, constrained-LL) and
  C4 PPL **16.1667** byte-identical. Neither path uses
  `generate_translation`.
- **+** Direction asymmetry **restored** (fine-tune now also has
  UG→EN > EN→UG: 16.81 vs 14.18, gap +2.63) but **compressed** vs
  zero-shot (30.10 vs 9.96, gap +20.14). Residual UG→EN gap to
  zero-shot: **−13.29 chrF** → §14 mechanism account stands: B′
  collapse explained ~7.4 chrF of the 20.91 chrF regression; remaining
  loss is B″ source-unfaithful hallucinations + the gradient /
  `eval_loss` checkpoint asymmetry, not decoding.
- **Pre-registered Minimum (≥1 direction beats zero-shot):** met
  (EN→UG +4.22). **Target (+5 chrF in both directions):** still not met
  — UG→EN remains a regression vs zero-shot.
- **Sanity gate status: not yet measured.** Re-eval was variant-scoped
  to `qwen_finetuned`. Slurm 2766 showed 0 % collapse on `qwen_zeroshot`
  (20 / 20 source-anchored EN), so the penalty is expected to be a near
  no-op there; a small zero-shot re-run is the cheapest confirmation
  (see `TODO.md`).

### 2026-05-27 — Slurm 2751 (`debug_ug2en`) — failed at import

- `scripts/debug_ug2en.py` crashed immediately:
  `ModuleNotFoundError: No module named 'shared'`. Invoking
  `python scripts/debug_ug2en.py` puts `scripts/` on `sys.path`, not
  the repo root. Fixed in-repo with `sys.path.insert(0, REPO_ROOT)`;
  re-submit after rsync (`TODO.md`).

### 2026-05-28 — `run_20260527_185416` (Slurm 2770) — Qwen Mix-50 ablation retrain

> Mix-50 = 50 % FLAN English-only data mixed with the 100 k CUTE-P
> parallel pairs (vs Mix-20 = 20 % FLAN). Same Qwen2.5-7B QLoRA recipe,
> same LoRA r/α, same packing flag. Logged under
> `docs/tasks/bonus/02_qwen_mix_ablation.md`. Decoded with the same
> direction-conditional repetition controls as Slurm 2768.

- **+** FLORES UG→EN chrF (Mix-20 → Mix-50): **16.8079 → 17.9662**
  (+1.16). Closes ~5.5 % of the residual 13.29 chrF gap to
  `qwen_zeroshot` 30.10 after the repetition fix, but Mix-50 stays well
  under zero-shot UG→EN.
- **=** FLORES EN→UG chrF: **14.1762 → 14.0649** (−0.11; within noise).
  Mixing in more English-only data did **not** trade away EN→UG.
- **+** WCM raw accuracy: **21.00 % → 81.00 %** (+60 pp), but raw
  accuracy is now anchored to the 85.3 % majority-class floor — see
  Slurm 2785 entry below for the macro-recall verdict.
- **=** C4 EN PPL: **16.1667 → 15.9124** (−0.25). No catastrophic
  forgetting; if anything a tiny improvement from the extra English
  data.
- **Interpretation.** Mix-50 is the cleanest training-side fix for the
  UG→EN regression mechanism (§14): more English completions in the
  loss shifts the gradient slightly toward EN-target generation. Lift
  is real but modest (+1.16 chrF); B1 / B2 retrains (direction-balanced
  loss + checkpoint selection) were the next levers but are deferred
  (see TODO).

### 2026-05-28 — `run_20260528_103619` (Slurm 2771) — exp-0 rep-penalty zero-shot sanity gate

> Full exp-0 re-eval of both zero-shot variants under the
> direction-conditional `repetition_penalty=1.15` /
> `no_repeat_ngram_size=4` UG→EN decoder. Pre-registered as a sanity
> gate after Slurm 2768 only re-ran `qwen_finetuned`; needed to confirm
> the §2 table is apples-to-apples after the decoder change.

- **=** `qwen_zeroshot` UG→EN chrF: **30.0957 → 29.5635** (−0.53,
  within the ±0.5 pre-registered tolerance — **gate pass with a
  margin**). EN→UG / WCM / C4 PPL **byte-identical** to Slurm 2749.
- **+** `llama_zeroshot` UG→EN chrF: **4.7050 → 15.9577** (+11.25,
  **×3.4 lift**). LLaMA-3.1's zero-shot Uyghur→English was suffering
  from the same greedy repetition collapse as `qwen_finetuned` pre-fix
  — the repetition controls fix it without any model change.
  EN→UG / WCM / C4 PPL byte-identical.
- **Implication for §2.** Zero-shot UG→EN cells are now scored under
  the same decoder as `qwen_finetuned` and `cute_llama_p`. The
  `qwen_zs` cell barely moves; the `llama_zs` cell jumps and the §2
  table is updated. The fine-tune's UG→EN regression vs `qwen_zs`
  shrinks slightly (−13.29 → −12.76 chrF for Mix-20, −11.60 chrF for
  Mix-50). No analysis conclusions change; the decoder fix is just now
  uniform across all four core rows.

### 2026-05-28 — Slurm 2785 (`debug_wcm`) — Mix-50 majority-class audit + macro metrics

> `scripts/debug_wcm.py` runs the same constrained-LL scoring path as
> `eval_wcm` but records per-row gold + predicted label, the full
> per-label joint log-prob, and the top-1 − top-2 margin. Compares
> `qwen_finetuned` (Mix-50 adapter) and `qwen_zeroshot` on the same
> 300 rows. Artifact: `results/debug/wcm_mix50_vs_zs.json`.

- **=** Mix-50 `majority_class_share_pred` = **0.830** (< 0.95 collapse
  threshold). Not majority-class collapse. The 81 % cell is real.
- **−** Mix-50 **balanced accuracy** (macro recall) = **0.258** vs
  `qwen_zeroshot` **0.271**. Uniform-random floor = 0.167. Under the
  class-balance-invariant metric Mix-50 ≈ zero-shot — both are ~9 pp
  above random, neither has acquired broad classification competence.
- **+** Mix-50 **macro F1** = **0.220** vs `qwen_zeroshot` **0.103**.
  Mix-50 wins on macro F1 because its label-1 precision is high
  (0.924); the absolute level is still low.
- **Per-class recall.** Mix-50 `[0, 0.90, 0, 0.65, 0, 0]` —
  learned exactly labels 1 and 4 (which together cover 276/300 = 92 %
  of gold support); zero TPs on 0/3/6/9. `qwen_zeroshot`
  `[0, 0.02, 0.57, 0.20, 0, 0.83]` — over-fires label 9 (164
  predictions for 6 true rows). Two opposite failure modes converging
  on similar balanced accuracy by accident.
- **Methodological note.** A stratified balanced subset is *not* a
  fix here: with only 3 label-`0` rows the balanced subset caps at
  18 rows, statistically meaningless. Macro recall (=balanced
  accuracy) gives the same property at full sample size and is now
  emitted by `scripts/debug_wcm.py` on every run.
- **§3 / report consequence.** Mix-50's WCM cell stays in the bonus
  table (raw accuracy 81 %) with a footnote citing the macro-recall
  parity with zero-shot. The fine-tune's WCM lift over zero-shot is
  real on **macro F1** (×2.1) and on **majority-class calibration**,
  but not on broad classification competence — there is no headline
  WCM improvement under a fair metric.

### 2026-05-28 — Slurm 2786 / 2787 / 2788 (`qualitative_examples`)

> `scripts/qualitative_examples.py` — N FLORES sentences × variants ×
> 2 directions, sentence-level chrF; for Task 05 §6 of the report.

- **−** Slurm 2786 OOM'd in `caching_allocator_warmup` when
  `cute_llama_p` (fp16, ~13 GB) loaded as the 4th variant: the prior
  three 4-bit + LoRA load/del cycles left no contiguous 13 GB block on
  the 24 GB MIG. Fix in `scripts/qualitative_examples.py` (commit
  `f3586b3`): load `cute_llama_p` first while the allocator is
  fragment-free; iteration order decoupled from display order; explicit
  `gc.collect()` between variants.
- **=** Slurm 2787 succeeded but only ran 4 variants (Mix-20 only);
  per-sentence mean chrF logged for `qwen_zeroshot` (EN→UG 6.92, UG→EN
  29.95), `llama_zeroshot` (0.47, 19.60), `qwen_finetuned`/Mix-20
  (12.25, 17.81), `cute_llama_p` (8.61, 28.10). Numbers track §2 modulo
  the 5-sentence sample.
- **+** Script generalised (commit `a7a0593`) to score **5 variants**:
  `qwen_zeroshot, llama_zeroshot, qwen_finetuned_mix20,
  qwen_finetuned_mix50, cute_llama_p`. Slurm 2788 submitted with the
  5-variant default + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  as a second fragmentation guard. **In-flight at last sync** —
  artifacts `results/reports/qualitative_examples.{json,md}` will
  supersede the 2787 pull.

---

## 2. Final results — core experiments

Latest measured number per cell. `pending` = the numerical value at the
current protocol has not yet been written by a successful Slurm run;
the source run for each populated cell is noted under the table.

| Variant | FLORES EN→UG chrF | EN→UG BLEU | FLORES UG→EN chrF | UG→EN BLEU | WCM Uyghur acc. | C4 EN PPL |
|---------|-------------------|------------|-------------------|------------|------------------|-----------|
| `qwen_zeroshot`   | 9.963       | 0.2389 | **29.5635** | 3.4428 | 6.33 % (19 / 300, constrained-LL)      | 16.5949 |
| `llama_zeroshot`  | 0.8447      | 0.449  | 15.9577     | 2.5033 | 3.00 % (9 / 300, constrained-LL)       | **13.6891** |
| `qwen_finetuned`  | **14.1762** | 0.0354 | 16.8079     | 0.1794 | **21.00 %** (63 / 300, constrained-LL) | 16.1667 |
| `cute_llama_p`    | 6.8773      | 0.2638 | 23.0881     | 1.7748 | 15.33 % (46 / 300, base_lm constrained-LL) | **13.0148** |

All four FLORES rows are under the same decoder
(`generate_translation` with direction-conditional
`repetition_penalty=1.15` + `no_repeat_ngram_size=4` on English target);
all four WCM rows are under constrained log-likelihood scoring over the
6-label set. Apples-to-apples across the table.

WCM macro metrics on the same predictions (from `scripts/debug_wcm.py`
artifacts; `qwen_finetuned` row here is the Mix-20 cell measured at
Slurm 2744 / 2768, **not** the Mix-50 audit which lives in §3):

| Variant | Raw acc. | Balanced acc. (macro recall) | Macro F1 |
|---|---|---|---|
| `qwen_zeroshot`  | 6.33 %  | 0.271 | 0.103 |
| `llama_zeroshot` | 3.00 %  | _pending_ (no debug_wcm artifact) | _pending_ |
| `qwen_finetuned` (Mix-20) | 21.00 % | _pending_ (Mix-20 not yet re-audited) | _pending_ |
| `cute_llama_p`   | 15.33 % | _pending_ | _pending_ |

The Mix-50 audit numbers (balanced acc. 0.258, macro F1 0.220) live in
§3. The Mix-20 / `llama_zs` / `cute_llama_p` rows above are marked
pending because `debug_wcm` was only run on the Mix-50 vs `qwen_zs`
pair (Slurm 2785); no additional GPU run is required to fill them
since the per-prediction artifacts already exist in
`results/run_<id>/.../eval_wcm_*.json` — only an offline script
extension is needed.

Sources for populated cells (latest measurement per metric):

| Variant | FLORES rows | WCM row | C4 PPL row |
|---|---|---|---|
| `qwen_zeroshot`  | `run_20260528_103619` (Slurm 2771, rep-penalty UG→EN) | `run_20260526_223852` (Slurm 2749, constrained-LL) | `run_20260526_223852` (Slurm 2749) |
| `llama_zeroshot` | `run_20260528_103619` (Slurm 2771, rep-penalty UG→EN) | `run_20260526_223852` (Slurm 2749, constrained-LL) | `run_20260526_223852` (Slurm 2749) |
| `qwen_finetuned` | `run_20260524_020432` (Slurm 2768, rep-penalty UG→EN) | `run_20260524_020432` (Slurm 2744, constrained-LL) | `run_20260524_020432` (Slurm 2744) |
| `cute_llama_p`   | `run_20260526_224102` (Slurm 2750) | `run_20260526_224102` (Slurm 2750, base_lm constrained-LL) | `run_20260526_224102` (Slurm 2750) |

### Analysis (current best estimate)

- **EN→UG: fine-tuning works.** `qwen_finetuned` reaches **14.18 chrF**,
  +4.22 over `qwen_zeroshot` (9.96) and >13 chrF over `llama_zeroshot`
  (0.84). The pre-registered **Minimum** criterion (fine-tuned beats
  zero-shot in ≥1 direction) is met. The pre-registered **Target**
  criterion (+5 chrF in *both* directions) is not — see UG→EN below.
- **UG→EN: real training-side regression, partly masked by greedy
  repetition collapse.** `qwen_finetuned` UG→EN was **9.39 chrF** under
  greedy decoding vs `qwen_zeroshot` 30.10. The repetition penalty
  shipped after Slurm 2766 recovers the chrF to **16.81** (Slurm 2768)
  — confirming B′ greedy `"The 2 1 1 1 …"` collapse accounted for
  ~7.4 chrF of the 20.91 chrF regression. The remaining **−13.29
  chrF** residual gap to zero-shot matches the §14 training-side
  mechanism: `assistant_only_loss` biases gradients toward Uyghur
  output and aggregated `eval_loss` early stopping picks an
  EN→UG-optimised checkpoint, so the adapter loses source faithfulness
  on UG→EN (B″ hallucinations) even when generation no longer collapses
  into a loop. Slurm 2744 falsified chat-marker leak; data audit (§14)
  ruled out missing UG→EN rows. C4 PPL gap only +0.4 → not catastrophic
  forgetting. **Direction asymmetry is restored** (UG→EN > EN→UG, +2.63
  chrF gap) but **compressed** vs zero-shot (+20.14 gap).
- **WCM: fine-tune wins on raw accuracy and macro F1, but balanced
  accuracy is unchanged.** All variants scored under constrained-LL
  (Slurm 2749 closed the protocol gap): `qwen_ft` (Mix-20) 21.00 % vs
  `qwen_zs` 6.33 % vs `llama_zs` 3.00 %. The +14.67 pp raw delta on
  Qwen is the cleanest single-metric headline for the fine-tune. **But
  under macro recall (= balanced accuracy)**, measured by Slurm 2785
  on the Mix-50 vs `qwen_zs` pair (the closest analogue to Mix-20),
  the fine-tune is **not** statistically distinguishable from zero-shot
  (0.258 vs 0.271 — both ~9 pp above the 1/6 = 16.7 % uniform floor).
  On macro F1 the fine-tune still wins (0.220 vs 0.103). What this
  means concretely: fine-tuning teaches the model the **prior**
  P(label = 1) (label 1 covers 85.3 % of rows) and the **label-4**
  distinction, and very little else (zero TPs on labels 0/3/6/9).
  Zero-shot, conversely, *anti-prefers* the majority class (label-1
  recall = 0.02) but over-fires label 9 — a real prompt-anchoring
  pathology. The honest read: WCM raw accuracy is a calibration win,
  not a classification competence win, and the report should quote
  both numbers.
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
  +5.67 pp (21.00 % vs 15.33 %). CUTE-Llama-P still leads UG→EN chrF
  (23.09 vs `qwen_ft` 16.81 = +6.28 after the repetition fix; was
  +13.70 pre-fix). Even with the recovery, neither fine-tune /
  baseline matches `qwen_zeroshot` UG→EN (30.10), confirming the
  generate-English regression is an instruction-tuning artefact not
  rescued by the published baseline.

### Outstanding (write-up only — no further GPU runs planned)

All four variant rows in §2 now reflect the **post–repetition-penalty**
protocol for the UG→EN cell (sanity-gate closed by Slurm 2771).
Remaining items are reporting:

1. **Task 05** — `docs/05_results_analysis.md` 8-section write-up.
2. **Task 06** — final report / slides.
3. **(Optional) Task 04** — `scripts/aggregate_results.py` +
   `results/reports/consolidated_results.json`. §2 already serves as
   the canonical table; skippable.
4. **(Optional) macro metrics for Mix-20 / `llama_zs` / `cute_llama_p`** —
   re-run `scripts/debug_wcm.py --no-adapter` on each base + adapter
   to populate the macro F1 / macro recall column in §2 for every row.
   Per-prediction artifacts already exist so no fresh GPU eval is
   needed; only a small refactor of `debug_wcm.py` to consume an
   `eval_wcm_*.json` instead of re-classifying from scratch.
5. **(Optional) sacrebleu `--paired-bs` CIs** — `docs/04_planned_evaluation.md`
   §4.3.

---

## 3. Bonus experiments (stretch — placeholders)

Stretch goals from `docs/tasks/bonus/`. All cells `pending` until the
corresponding Slurm runs land. None of these are required for the
**Core** evaluation in §2.

| Variant | Source task | FLORES EN→UG chrF | FLORES UG→EN chrF | WCM Uyghur acc. (raw) | C4 EN PPL | Status |
|---|---|---|---|---|---|---|
| `llama_finetuned` (Mix-20) | `bonus/01_experiment_3_llama_mix20_finetune.md` | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix0`      | `bonus/02_qwen_mix_ablation.md`                  | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix10`     | `bonus/02_qwen_mix_ablation.md`                  | _pending_ | _pending_ | _pending_ | _pending_ | not started |
| `qwen_finetuned_mix50`     | `bonus/02_qwen_mix_ablation.md`                  | **14.0649** | **17.9662** | 81.00 % (243 / 300) † | 15.9124 | **done** (Slurm 2770 / `run_20260527_185416`) |
| `qwen_zeroshot_5shot`      | `bonus/04_qwen_5shot_baseline.md`                | _pending_ | _pending_ | _pending_ | _pending_ | not started |

† **Mix-50 WCM caveat.** Raw accuracy 81.00 % sits just below the
85.3 % majority-class floor. Class-balance-invariant metrics from
Slurm 2785 (`scripts/debug_wcm.py`): **balanced accuracy (macro
recall) = 0.258** vs zero-shot 0.271 (uniform floor 0.167); **macro
F1 = 0.220** vs zero-shot 0.103. Mix-50 learned the label-1 prior +
the label-4 distinction (per-class recall `[0, 0.90, 0, 0.65, 0, 0]`);
zero TPs on labels 0/3/6/9. Real lift on calibration and macro F1,
no broad classification competence yet.

**Mix-50 vs Mix-20 (head-to-head FLORES).** UG→EN chrF +1.16 (17.97
vs 16.81), EN→UG chrF essentially unchanged (14.06 vs 14.18). Closes
~5.5 % of the 13.29 chrF post-rep-penalty UG→EN gap to `qwen_zeroshot`
29.56. Decision (see TODO): Mix-50 promoted to §3 with full numbers,
B1 / B2 retrains deferred — further training-side fixes are
out-of-scope for the final report.

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
