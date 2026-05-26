# Project Refinement Log

> Date: May 2026
> Context: Design presentation planning session — review of PROJECT.md and RESEARCH.md
>          against course requirements and practical constraints.

This document records every decision that was changed from the original project plan,
with the rationale for each change. It serves as a transparency log for the course
examiner and as a reference for the team.

---

## 1. Core vs. Stretch Goal Restructuring

### What changed
The original plan treated the full ablation (Mix-{0,10,20,50}) × 2 models (Qwen2.5, Llama-3.1),
CUTE-Llama-P baseline reproduction, and MiLiC-Eval (9 tasks) as first-class deliverables.

The revised plan defines a strict two-tier structure:
- **Core experiment** (must complete): Qwen2.5-7B Mix-20 fine-tuned, evaluated on
  FLORES-200 and WCM-v2, compared against zero-shot Qwen2.5 and zero-shot Llama-3.1.
- **Stretch goals** (if time permits): Llama-3.1 fine-tune, CUTE-Llama-P baseline,
  ablation variants, MiLiC-Eval.

### Why
The original scope was too large for the compute constraints (originally MIG
`1g.10gb` per job; later upgraded to a ~24 GB slice — see `docs/PROJECT.md`
§Compute environment). Even with the larger slice an 8-cell ablation × 2 models
serial on a single worker exceeds the 5-day priority partition limit. More
importantly, course examiners reward a clean, complete, well-analysed core
experiment over a partially executed grand plan. Presenting an incomplete
ablation at the design stage creates expectations the project may not meet.

---

## 2. CUTE-Llama-P Baseline Status (originally demoted, now back as core)

### What changed
- **Original plan:** CUTE-Llama-P was the primary baseline.
- **First refinement:** demoted to a *stretch* baseline with a 2-day load
  budget and a documented fallback to zero-shot-only baselines, because
  the MIG `1g.10gb` (~10 GB) slice could not load the model's expanded
  ~155 K-token embedding tables and the load story was unknown.
- **Current state (May 2026, 24 GB MIG slice):** preflight check 5 has
  already loaded the model in 4-bit NF4 and produced Uyghur Arabic-script
  output for ≥3/5 sentences. The "high-risk" framing and the 2-day budget
  are dropped. CUTE-Llama-P is a planned core baseline again, evaluated on
  the same FLORES-200 + WCM-v2 test sets as the instruct models, with a
  base-LM-appropriate few-shot prompt (documented as a protocol
  difference, not hidden).

### Why the fallback is still recorded
If a future cluster change makes loading impossible again, the fallback
plan (zero-shot Qwen2.5 + zero-shot LLaMA-3.1 only) is still valid and
documented. It is no longer the expected path.

---

## 3. Day-1 Sanity Checks Added as Mandatory Gate

### What changed
The original plan had no explicit pre-training checks. The revised plan adds three
mandatory sanity checks that must pass before any training job is submitted:
1. Tokenizer Uyghur segmentation test (token/byte ratio).
2. QLoRA memory test (forward+backward on MIG slice).
3. CUTE-Llama-P load test (determines feasibility of stretch baseline).

### Why
Two risks were identified that could invalidate the entire experimental design
without being caught until late:

- **Tokenizer fragmentation risk**: Both Qwen2.5 and Llama-3.1 were trained on
  Latin/CJK-heavy corpora. Uyghur uses Arabic script. If the tokenizer fragments
  Uyghur into byte-level tokens (token/byte ratio >> English ratio), the "no vocabulary
  surgery" design decision is flawed and must be revisited before wasting compute
  on a broken fine-tune.

- **Memory risk**: QLoRA on the ~24 GB MIG slice is expected to fit (~8–12 GB)
  with comfortable headroom; bf16 LoRA (~18–22 GB) also fits. On the earlier
  `1g.10gb` (~10 GB) profile QLoRA was tight (~6–9 GB) and bf16 LoRA was
  infeasible. Either way the budget depends on sequence length, batch size,
  and gradient checkpointing configuration. A failed memory test on day 1 is
  far better than a failed training job on day 3.

Catching either issue on day 1 allows a week-1 fix. Catching it on week 2 or 3
would derail the timeline.

---

## 4. MiLiC-Eval Deferred to Final Report (Stretch)

### What changed
MiLiC-Eval (9-task bilingual benchmark) was listed as a primary evaluation benchmark
in the original plan. It is now a stretch goal, deferred to the final report if at all.

### Why
The design presentation evaluation plan should be simple, concrete, and completable
within the project timeline. FLORES-200 + WCM-v2 already covers translation quality
and downstream Uyghur task performance — the two most important axes.
MiLiC-Eval adds coverage breadth but not depth, and its 9-task structure makes
result reporting and analysis significantly more complex for marginal additional insight
at the course level.

---

## 5. Translation Direction Asymmetry Explicitly Documented

### What changed
The original plan treated EN→UG and UG→EN symmetrically, with no mention of
the expected performance gap between the two directions.

The revised plan explicitly notes: EN→UG generation requires the model to *produce*
Uyghur fluently (much harder), while UG→EN only requires understanding Uyghur input
and producing English. EN→UG scores will likely be substantially lower.
This asymmetry is now flagged in the evaluation plan and will be discussed in the
presentation and report.

### Why
Not flagging this would make the EN→UG results look like a failure rather than
an expected characteristic of generation in a low-resource target language.
Presenting it proactively shows methodological awareness and pre-empts a likely
examiner question.

---

## 6. Pre-registered Success Criteria Added

### What changed
The original plan had no pre-defined success criteria. The revised plan adds three
explicitly tiered, numeric success criteria (minimum / target / stretch), defined
before any results are seen.

### Why
Pre-registration is a standard scientific practice that prevents post-hoc
re-framing of results. For a course project, it also demonstrates systematic
thinking — a graded criterion in the course. The numeric thresholds (e.g.,
"≥5 chrF++ improvement", "within 2 chrF++ of CUTE-Llama-P") are set based on
what is known from the literature about LoRA fine-tuning gains on low-resource tasks,
not pulled from results.

---

## 7. "No Vocabulary Surgery" Framing Made Conditional

### What changed
The original plan stated as a fixed design decision: "No vocabulary surgery on
Qwen / Llama — both already handle Uyghur Arabic script in their native tokenizer."

The revised plan keeps this as the default and intended approach, but conditions it
on the tokenizer sanity check (item 3 above). If the check fails, a minimal vocabulary
extension (top-5K Uyghur unigrams) will be added and the change documented.

### Why
Making an absolute claim about tokenizer quality without verifying it first is
methodologically weak. The conditional framing is more honest and more defensible
in the design presentation Q&A.

---

## 8. Ablation Scope Reduced and Re-prioritised

### What changed
The original ablation: Mix-{0, 10, 20, 50} × {Qwen2.5, Llama-3.1} = 8 fine-tunes.

The revised ablation: Mix-{0, 10, 20, 50} × {Qwen2.5 only} = 4 fine-tunes (stretch).
Further reduced to Mix-{0, 20} only if time is short (2 fine-tunes).

### Why
Running the ablation on both models doubles the compute without adding scientific clarity.
The ablation's research question is about data mixing ratio, not about model architecture.
Fixing the model to Qwen2.5 (the primary model) keeps the experimental variable clean.
Running all 8 cells on 7 workers is theoretically fast (~1.5 days parallelised), but
this assumes all jobs start immediately, no job fails, and evaluation runs smoothly —
all optimistic assumptions for a first-time Slurm user on a shared cluster.

---

## 9. Train / Test / Eval Split Formalised (Pair-Level, In-Pipeline)

### What changed

- **Original plan** (`03_planned_approach.md` v1 §1): write a separate
  `dataset/cute_p_valdev.jsonl` of 1 000 held-out CUTE-P pairs as the
  in-domain validation set, fed to a periodic "loss check" outside the
  training loop. No formal three-way split was specified.
- **Revised plan (implemented):** the split happens inside
  `--mode preprocess`, at **parallel-pair level**, *before* bidirectional
  expansion. The result is saved as a single HF `DatasetDict`:

  | Split   | Role                                                                                       |
  |---------|--------------------------------------------------------------------------------------------|
  | `train` | gradient updates                                                                           |
  | `test`  | in-loop `eval_loss` curve in TensorBoard (overfit detector) + early stopping / best-checkpoint signal |
  | `eval`  | external FLORES-200 devtest + WCM-v2 Uyghur (`minority/ug.txt`) + C4 PPL (never seen during training, different domain); experiment 0 = zero-shot baselines, experiment 1 = fine-tuned adapter only |

  The default `test_split_pct` is 5 %, controlled by
  `experiments/experiment_1/config.Experiment1Config`. FLAN rows get an
  independent row-level split at the same percentage, and the Mix-{N}
  ratio is computed against the **training pair count** so the effective
  ratio is preserved after holdout.

### Why

Three reasons:

1. **No leakage across translation directions.** A naive row-level
   shuffle puts one CUTE-P pair's `en2ug` row in `train` and its
   `ug2en` row in `test`. The model has then already seen the source
   and target tokens (in opposite roles) during training, so `eval_loss`
   becomes an optimistic in-distribution memorisation signal rather
   than a real overfit detector. Splitting at pair level removes this
   structurally; the invariant is enforced by
   `tests/test_data_split.py::test_no_pair_leakage_after_bidirectional_expansion`.
2. **No separate file to keep in sync.** The original plan implied
   maintaining `dataset/cute_p_clean.jsonl` and `dataset/cute_p_valdev.jsonl`
   side-by-side. Storing the split *inside* the preprocessed dataset
   removes a class of "which `valdev` matches which `clean`?" bugs and
   keeps the entire preprocess output under `artifacts/preprocessed_dataset/`,
   which is what every other stage already reads from.
3. **Reproducible from the run config.** The split is a deterministic
   function of `(test_split_pct, flan_seed)`, both captured in
   `artifacts/run_config.json`. Re-running `--mode preprocess` for the
   same run id reproduces the split bit-for-bit, which is required
   for `--run-id` resume semantics to be honest.

### What we explicitly did not do

- **No length-stratified split.** CUTE-P pairs are not balanced by token
  length across train/test; random pair-level shuffle is sufficient at
  the 5 % holdout scale (variance is dominated by content, not length).
- **No language-direction stratification.** Each kept pair contributes
  one `en2ug` and one `ug2en` row, so train and test are direction-
  balanced by construction.

---

## 10. Reproducibility & Best-Checkpoint Policy

### What changed

The original plan trained for a fixed `epochs=3` and saved the
checkpoint from the last epoch. The revised plan adds three coupled
training-hygiene controls:

| Control                | Where                                         | Effect                                                                                      |
|------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| **Seeded RNGs**        | `transformers.set_seed(cfg.flan_seed)` + `SFTConfig(seed=…, data_seed=…)` | Python `random`, NumPy, Torch (CPU+CUDA), DataLoader order and dropout masks are reproducible from the run config. |
| **In-loop eval**       | `SFTConfig(eval_strategy="steps", eval_steps=50)` on the `test` split    | Produces `eval/loss` in TensorBoard alongside `train/loss`.                                 |
| **Best-checkpoint + early stopping** | `load_best_model_at_end=True, metric_for_best_model="eval_loss"` + `EarlyStoppingCallback(patience=3)` | Final adapter is the lowest-`eval_loss` checkpoint, not the last; training stops if eval_loss plateaus for 3 evaluations. |

Assistant-only loss masking moved from a best-effort path (legacy
`DataCollatorForCompletionOnlyLM`, which was removed in newer TRL
versions) to a first-class path: the dataset is stored in
**conversational form** (`{"messages": [...]}`) and TRL applies the
chat template plus assistant-token masking natively via
`SFTConfig(assistant_only_loss=True)`. The text+collator path remains
as an automatic fallback for older TRL.

### Why

- **Seeding** is the cheapest reproducibility win available; without it,
  two runs of the same config produce different LoRA adapters and
  different `eval_loss` curves, which makes ablation comparisons
  ambiguous. Full GPU determinism is intentionally out of scope (cuDNN
  + paged optimizer + bf16 ≠ bitwise reproducible without large speed
  loss).
- **Early stopping + best-checkpoint** turn the in-loop `eval_loss`
  signal from passive ("look at the curve") into actionable ("stop
  when it stops helping, keep what worked"). For a course project this
  prevents wasted GPU hours past the overfit point and ensures the
  reported FLORES/WCM numbers come from the genuinely best fine-tune,
  not the most-trained one.
- **Native assistant-only loss** is the documented modern path in TRL
  and removes our dependency on a deprecated collator API. The
  fallback path is preserved so older TRL builds (and the smoke
  laptop env) still work.

---

## 11. Unit-Test Contract for the Data Split

### What changed

Introduced a `tests/` directory with `pytest` contract tests
(`pytest >= 8`, listed in `requirements.txt`). The initial suite covers:

| Test                                                                | Invariant locked in                                                                                  |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `test_pair_split_disjoint_and_sized`                                | The pair-level splitter returns disjoint train/test sets of the requested size.                       |
| `test_pair_split_reproducible_from_seed`                            | Same `(n_pairs, test_pct, seed)` → bit-identical split; different seed → different split.            |
| `test_pair_split_zero_pct`, `test_pair_split_too_few_pairs`         | Edge cases collapse to "all train, no test" without raising.                                          |
| `test_no_pair_leakage_after_bidirectional_expansion`                | **The deep-learning hygiene invariant**: a CUTE-P pair is never simultaneously in train and test after the `en2ug` / `ug2en` expansion. |
| `test_each_pair_appears_in_both_directions`                         | Bidirectional expansion is symmetric: every kept pair contributes one `en2ug` and one `ug2en` row.    |
| `test_flan_count_for_mix_*`                                         | Mix ratio math is correct (and the Mix-100 degenerate case is rejected).                              |
| `test_experiment1_defaults_are_production`                          | The committed defaults are not a "smoke" config (no `sample_count`, sane LoRA rank, `test_split_pct ∈ (0, 0.5)`, etc.). |

All tests run **without HuggingFace downloads or GPU** — `shared/data`
lazy-imports `datasets` and the leakage test stubs `load_cute_parallel_lines`
to feed a synthetic 200-pair corpus.

### Why

The two failure modes these tests catch are silent and expensive:

- **Pair leakage** is invisible from the training curves — `eval_loss`
  just looks suspiciously good. Without the test, a refactor that
  reverts to a row-level `train_test_split` would not be detected
  until somebody noticed unrealistically low `eval_loss` weeks later.
- **Accidentally shipped smoke defaults** (e.g., a forgotten
  `sample_count=100`) waste a 5-day Slurm reservation on a useless
  fine-tune. A one-line config test makes this a CI-catchable mistake.

The suite is intentionally **small and fast** (≈ 2 s on CPU). It is a
contract suite, not a coverage exercise; expanding it is welcome where
similar silent failure modes are identified.

---

## 13. Decoding & Scoring Bugs Caught After `run_20260524_020432`

### What changed

The first full fine-tune run (`run_20260524_020432`, Slurm 2650, Mix-20)
revealed two evaluation-side bugs that had **nothing to do with training**
but were corrupting the reported numbers. Both are now fixed in
`shared/evaluation.py` and locked in with unit tests:

| Bug | Symptom in `run_20260524_020432` | Code patch | Test |
|-----|-----------------------------------|------------|------|
| **FLORES UG→EN chat-marker leak** | `qwen_finetuned` UG→EN chrF collapsed 30.29 → 9.38 while EN→UG *improved* 9.96 → 14.18. C4 PPL barely moved (16.59 → 16.17), so the regression was not catastrophic forgetting. | `generate_translation` now passes a *list* of stop ids (`eos_token_id=[<|endoftext|>, <|im_end|>, <|im_start|>, <|eot_id|>, <|start_header_id|>, <|end_header_id|>]`, transformers ≥ 4.45) and hard-trims the decoded string at the first occurrence of any chat-marker literal or fresh assistant/user/system turn header. Commit `da8e8d8`. | `tests/test_evaluation_translation.py` (11 tests). |
| **WCM-v2 free-form prompt + substring match** | All three variants scored *below* the 16.7 % random baseline on a 6-class task where the majority-class floor is 85.3 %. Δ between `qwen_ft` (7.33 %) and `qwen_zs` (6.33 %) was noise. | `_classify_uyghur` switched from free-form generation + substring match to **constrained log-likelihood scoring** — for each candidate label, score `log P(label | chat_prompt(text))` and return the argmax. Always returns one of the supplied labels. Commit `6d4197c`. | `tests/test_evaluation_wcm.py` (9 tests). |

### Why the FLORES leak only damaged one direction

The fine-tuned adapter learned to emit chat-template scaffolding past the
natural answer (a known failure mode when the response is short and the
adapter is over-anchored on the template). The leak itself is
**direction-agnostic** — the FT model bleeds chat markers in both
directions — but the *chrF impact* is direction-asymmetric for three
compounding reasons:

1. **Script-mismatch hides the noise on EN→UG.** chrF is the F-score of
   character n-gram overlap. The leaked content (`<|im_end|>`,
   `\nassistant\n`, a fresh `user\n…\nassistant\n…` second turn) is all
   ASCII / Latin. For EN→UG the reference is Uyghur Arabic script
   (U+0600..U+06FF), so the leaked Latin n-grams do **not** dilute the
   match between the Arabic translation portion of the hypothesis and
   the Arabic reference. Precision drops slightly (denominator grows)
   but the matched n-gram count is essentially unchanged. For UG→EN the
   reference is English Latin, the *same script* as the noise — the
   leaked `\nassistant\nuser\n…` n-grams compete directly with the
   English reference's character n-grams and tank precision.
2. **The FT model's leak rate is itself asymmetric.** Mix-20 is 80 %
   CUTE-P pairs (`en2ug` + `ug2en`) + 20 % FLAN (EN-only, EN-output).
   The adapter saw far more "output Uyghur under chat template" than
   "output English under chat template", so it has cleaner internal
   stop behaviour when producing Uyghur and is more likely to spill past
   `<|im_end|>` when producing English. Zero-shot Qwen, having no such
   imbalance, did *not* show the asymmetry — its UG→EN chrF stayed at
   30.29.
3. **Qwen exposes two end markers.** `<|endoftext|>` (151643) and
   `<|im_end|>` (151645). The pre-fix code passed only
   `tokenizer.eos_token_id` (= `<|endoftext|>`), so generation continued
   on `<|im_end|>` until `max_new_tokens=256`, appending 100+ tokens of
   garbage per sentence × 1012 FLORES sentences = the 21-point chrF
   collapse we observed. Adding `<|im_end|>` to the stop set is the
   primary fix; the post-decode trim catches the residual literal-text
   variant the adapter learned to emit as normal text (because
   `skip_special_tokens=True` strips token ids, not the literal string
   `"<|im_end|>"`).

### Why neither bug surfaced earlier

- **WCM**: the loader called `load_dataset("hfl/wcm-v2", split="test")`,
  which returned a Chinese parquet with no `label` column. All three
  variants logged `WCM = ERROR`. The error was *visible* (and tracked
  separately, see §12), so the prompt bug behind the error was masked —
  it took fixing the loader to expose the scoring path failure.
- **FLORES UG→EN**: the regression was visible in the chrF table but
  initially read as a Mix-20 over-fitting artifact (anchor on
  generate-Uyghur direction at the cost of English fluency). The clue
  that ruled out forgetting was the C4 PPL gap of only +0.4
  (`PROJECT_RESULTS.md` 2026-05-24 §Analysis, bullet 2): a 21-point
  chrF collapse with a stable English language-model perplexity is a
  decoding-shape signature, not a forgetting-shape signature.

### Why we did not retrain

The decoding fix is purely on the inference path; the adapter from
`run_20260524_020432` is reused as-is. The validation contract
(`docs/tasks/03_ug2en_decoding_fix.md` §Step 4) is that the fix is
*additive*: `qwen_zeroshot` UG→EN must reproduce within ±0.5 chrF of
30.29 (and `llama_zeroshot` within ±0.5 of 4.71). If `qwen_finetuned`
UG→EN climbs back toward the zero-shot Qwen baseline after the fix, the
regression was decoding and training is correct. If it does **not**, the
regression is a real Mix-20 over-fitting effect and is reported as the
headline finding rather than engineered away.

### Empirical update (2026-05-26 re-eval, Slurm 2744)

The post-fix re-eval reproduced FLORES EN→UG chrF / UG→EN chrF /
C4 PPL **byte-identically** to the May-24 pre-fix numbers (see
`PROJECT_RESULTS.md` 2026-05-26 sub-bullet under
`run_20260524_020432`). The chat-marker fix had **zero measurable
effect** on translation quality. Concretely this falsifies the
leak-causes-regression part of the hypothesis above:

- `skip_special_tokens=True` was already stripping the *token-id*
  form of `<|im_end|>`. The adapter is *not* emitting the
  *literal-string* form `"<|im_end|>"` as plain text the way we
  hypothesised, so there is nothing for the post-decode trim to
  remove on this run.
- The new stop-token list still removes the failure mode as a
  potential cause of future regressions, but for **this** adapter it
  was a no-op.

The UG→EN regression 30.29 → 9.38 is therefore **genuine** — a real
Mix-20 over-fitting effect on the generate-English direction — and is
reported as the headline finding rather than engineered away (Task 03
§Step 4 success criterion, "OR the analysis concludes the regression
is genuine" branch).

The WCM half of §13 is **not** affected by this update: switching to
constrained log-likelihood scoring raised `qwen_ft` accuracy from
7.33 % to 21.00 % on the same adapter (×2.9). That fix was real and
landed as documented; what remains is a model-side / prompt-side gap
between 21 % and the 85.3 % majority floor, which is no longer a
methodology bug but a substantive finding.

### What the test suite now guarantees

- **No more silent chat-marker leaks**:
  `tests/test_evaluation_translation.py::test_stop_token_ids_adds_known_chat_markers`
  ensures every known marker the tokenizer recognises is in the stop
  set, and `test_clean_translation_output_*` (×6) lock in the
  post-decode trim semantics across `<|im_end|>`, `<|eot_id|>`,
  `\nassistant\n`, and the earliest-marker rule.
- **No more free-form WCM**:
  `tests/test_evaluation_wcm.py::test_classify_uyghur_returns_argmax_label_from_candidate_set`
  asserts the return is always one of the candidate labels under
  argmax-log-likelihood scoring, with a stub model that has *no*
  string-generation surface at all.

---

## Summary Table

| # | Change | Direction | Primary Reason |
|---|--------|-----------|----------------|
| 1 | Core/stretch split introduced | Scope reduction | Compute constraints + examiner expectations |
| 2 | CUTE-Llama-P demoted to stretch | Risk reduction | Engineering risk + direction mismatch |
| 3 | Day-1 sanity checks made mandatory | Risk mitigation | Tokenizer + memory failure modes |
| 4 | MiLiC-Eval deferred | Scope reduction | Complexity vs. value tradeoff |
| 5 | Direction asymmetry documented | Clarity | Pre-empts misinterpretation of results |
| 6 | Pre-registered success criteria added | Scientific rigour | Standard practice + course grading criteria |
| 7 | "No vocab surgery" made conditional | Honesty | Unverified claim removed |
| 8 | Ablation scope reduced | Scope reduction | Compute constraints + experimental clarity |
| 9 | Three-way pair-level split formalised in-pipeline | Methodological rigour | Removes leakage across `en2ug` / `ug2en`; replaces external `valdev.jsonl` file |
| 10 | Seeded RNGs + early stopping + best-checkpoint + native assistant-only loss | Reproducibility + waste reduction | Final adapter = lowest `eval_loss` checkpoint, not the last; runs are deterministic from `run_config.json` |
| 11 | Pytest contract suite for the data split | Regression prevention | Locks in the no-leakage invariant + guards against shipped smoke defaults |
| 12 | Experiment 0 for zero-shot eval; experiment 1 eval = `qwen_finetuned` only | Compute / workflow | Zero-shot FLORES/WCM/C4 numbers are invariant across fine-tunes; re-running them on every experiment-1 eval wasted ~25 min per variant. WCM loader fixed to `minority/ug.txt` (HF `split=test` was Chinese-only, no labels). |
| 13 | FLORES stop-token list + post-decode trim; WCM constrained log-likelihood scoring | Decoding/scoring correctness | UG→EN chrF collapse 30.29 → 9.38 was a chat-marker leak past `<|im_end|>` (asymmetric because the noise is Latin-script — invisible to chrF on Arabic-script EN→UG, catastrophic on Latin-script UG→EN). WCM below-chance accuracy was free-form generation + substring match instead of constrained scoring. Both fixed without retraining the adapter. |

---

*No changes were made to the core research question, the choice of Qwen2.5-7B as the
primary model, the use of CUTE-P as the training corpus, or the FLORES-200 + WCM-v2
evaluation benchmarks. These remain as originally planned.*
