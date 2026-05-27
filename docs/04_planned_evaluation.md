# 4. Planned Evaluation

To rigorously assess whether our parameter-efficient fine-tuning approach successfully imparts English↔Uyghur capabilities to Qwen2.5-7B, we pre-register the following systematic evaluation protocol.

## 4.1 Evaluation Matrix
Our experimental results will be reported according to the following evaluation matrix. CUTE-Llama-P is now a planned core baseline (`--experiment 2`, few-shot continuation; see `PROJECT_REFINEMENT.md` §2 + §13 for the protocol notes).

| Model | FLORES-200 (EN↔UG) | WCM-v2 (Uyghur) | English Perplexity (C4) |
|---|---|---|---|
| Qwen2.5-7B-Instruct (`--experiment 0`) | Zero-shot MT (chat prompt) | Constrained-LL classification (chat prompt) | Base PPL |
| Qwen2.5-7B-Instruct + QLoRA Mix-20 (`--experiment 1`) | Fine-tuned MT (chat prompt) | Constrained-LL classification (chat prompt) | Fine-tuned PPL |
| LLaMA-3.1-8B-Instruct (`--experiment 0`) | Zero-shot MT (chat prompt) | Constrained-LL classification (chat prompt) | Base PPL |
| CUTE-Llama-P (`--experiment 2`) | Few-shot continuation MT (base LM) | Constrained-LL classification (flat prompt) | Base PPL |



## 4.2 Benchmarks and Metrics
* **FLORES-200 Devtest (Translation)**
Our primary benchmark for translation is the FLORES+ devtest set, which contains 1012 sentences with high-quality parallel data for both English and Uyghur (Costa-jussà et al., 2022). We evaluate the EN→UG and UG→EN directions separately. Our primary metric is **chrF** (Popović, 2015) computed by `sacrebleu.corpus_chrf(...)` (default `char_order=6, word_order=0` — i.e. plain chrF, not chrF++/chrF-2). chrF is well-suited to morphologically rich, low-resource languages like Uyghur where token-level BLEU under-rewards correct content with surface-form variation. BLEU (Papineni et al., 2002) is reported as a secondary metric.

* **WCM-v2 (Uyghur Text Classification)**
To evaluate downstream Uyghur language understanding, we use the WCM-v2 dataset (`hfl/wcm-v2` → `minority/ug.txt`, 300 rows). We classify by **constrained log-likelihood scoring**: for each candidate label *l* in the dataset's label set, the model scores `log P(l | chat_prompt(text))` under teacher forcing, and the prediction is `argmax_l`. This guarantees the prediction is always one of the legal labels and removes the free-form-generation failure mode that produced below-random results on `run_20260524_020432` (see `PROJECT_REFINEMENT.md` §13). The primary metric is **Accuracy**. Macro-F1 is not currently computed; if it is needed for the final report it will be added on top of the same prediction array, with no change to the inference path.

* **English Perplexity (Catastrophic Forgetting Check)**
To ensure that tuning on Uyghur data does not destroy the model's core English capabilities, we evaluate perplexity on a held-out set of 1,000 English sentences from `allenai/c4/en` (streaming, validation split). We compute `model.eval()` perplexity before and after fine-tuning. A substantial relative increase (> 20 %) flags catastrophic forgetting. The C4 PPL gap on `run_20260524_020432` (16.59 → 16.17) ruled out forgetting; the UG→EN chrF regression is training-shaped (see `PROJECT_REFINEMENT.md` §14), not a C4-forgetting artifact.

* **MiLiC-Eval (Stretch Benchmark)**
As a stretch goal, we may evaluate the models on the 9-task bilingual MiLiC-Eval benchmark. Deferred to the final report; not part of the design-stage scope.

## 4.3 Statistical Reporting and Rigor
To ensure reproducible and statistically sound claims, we will report the following:

**Single fixed seed (limitation).** Due to strict time and compute constraints, our primary training runs use a single fixed random seed (`flan_seed=42` + `SFTConfig(seed=42, data_seed=42)`). This will be explicitly stated as a limitation in the final report. We do not run multi-seed averages.

**Deterministic decoding.** Translation eval uses `do_sample=False` (greedy). Chat-template paths pass a multi-id stop list + post-decode trim (`§4.7`). For **UG→EN only**, `generate_translation` also applies `repetition_penalty=1.15` and `no_repeat_ngram_size=4` (same values as the CUTE-Llama-P few-shot path) to suppress the greedy `"token loop"` collapse seen on the fine-tuned adapter (Slurm 2766 diagnostic). EN→UG generation is unchanged. Slurm 2768 re-eval: `qwen_finetuned` UG→EN chrF 9.385 → 16.8079 (+7.42); EN→UG byte-identical 14.1762 (gate works). Zero-shot sanity-gate re-run pending (`TODO.md`).

**Reproducibility signature.** Run config (`flan_seed`, `test_split_pct`, `eval_steps`, early-stopping patience, …) is frozen in `artifacts/run_config.json`. The exact sacrebleu version is pinned in `requirements.txt`; chrF / BLEU are reported as `sacrebleu.corpus_chrf` / `sacrebleu.corpus_bleu` default settings.

**Paired bootstrap (deferred).** sacrebleu's `--paired-bs` (n=1000) was in the original plan as a significance test on chrF deltas. It is currently **not** wired into the eval pipeline (`shared/evaluation.py` reports point estimates only). For the design presentation it is sufficient to report point chrF / BLEU; if statistical significance becomes a graded requirement we add a single `_paired_bootstrap_chrF` helper over the saved per-sentence hypotheses without rerunning the model. Tracked alongside the consolidated-results task in `docs/tasks/04_consolidated_results_table.md`.


## 4.4 Translation Direction Asymmetry (Expected vs. Observed)

**Expected (zero-shot).** EN→UG is harder than UG→EN because the model must
*generate* fluent Uyghur; UG→EN only requires generating English, which the
base instruct model already handles. We therefore expect zero-shot EN→UG chrF
to be lower than UG→EN chrF.

**Observed (fine-tuned, Mix-20).** The directions **invert**: `qwen_finetuned`
wins EN→UG (+4.22 chrF vs zero-shot) but **regresses** UG→EN (30.29 → 9.39).
This is not the generic script-difficulty asymmetry above — it is a
training-side bias (`assistant_only_loss` + aggregated `eval_loss`
checkpointing; balanced `ug2en`/`en2ug` data confirmed in
`PROJECT_REFINEMENT.md` §14). Report as the headline finding for Task 05.


## 4.5 Pre-Registered Success Criteria
We define three levels of success for this project (all chrF figures are plain chrF, not chrF++ — see §4.2):

1. **Minimum Criteria.** The fine-tuned Qwen2.5-7B+QLoRA outperforms the zero-shot base Qwen2.5-7B in ≥ 1 translation direction.
Rationale: any measurable benefit from fine-tuning validates the core pipeline and data formatting.

2. **Target Criteria.** A ≥ 5 chrF absolute improvement over the baseline in both directions, and the WCM-v2 accuracy improves over the same baseline.
Rationale: a 5-point chrF gain is widely considered a meaningful and perceptible improvement in low-resource machine translation literature.

3. **Stretch Criteria.** Our QLoRA model scores within 2 chrF points of the CUTE-Llama-P model on EN→UG, and beats it on UG→EN.
Rationale: if achieved, this shows that parameter-efficient LoRA on a multilingual instruct base matches full continued pre-training + vocabulary expansion (CUTE-Llama-P) at roughly 10× less compute.

## 4.6 Scope of Claims (What We Will NOT Claim)
To maintain academic honesty, we explicitly state that this study will not include:

- Human evaluation of translation fluency or adequacy.
- Claims about deployment readiness or runtime stability.
- Generalization of this specific recipe to other low-resource languages.
- Direct comparisons on ZH↔UG translation (direction mismatch with the primary focus of the CUTE baseline paper).

## 4.7 Decoding Pitfalls Caught After the First Run
The first fine-tune run (`run_20260524_020432`) surfaced two evaluation-side
bugs that, if left unfixed, would have made the reported numbers
unreliable. Both are documented at length in `PROJECT_REFINEMENT.md` §13;
the short version that belongs in the evaluation plan is:

- **FLORES — stop-token set must include all chat markers, not just
  `eos_token_id`.** Qwen exposes both `<|endoftext|>` and `<|im_end|>`;
  the fine-tuned adapter learned to emit `<|im_end|>` past the natural
  answer and the original `generate_translation` only stopped on
  `<|endoftext|>`, letting generation continue to `max_new_tokens=256`.
  The trailing Latin garbage tanks chrF on Latin-script UG→EN
  (the noise n-grams compete with the English reference) while
  remaining nearly invisible on Arabic-script EN→UG (cross-script noise
  doesn't dilute the Arabic n-gram match). Fix: pass a list of stop
  ids (`[<|endoftext|>, <|im_end|>, <|im_start|>, <|eot_id|>,
  <|start_header_id|>, <|end_header_id|>]`) + post-decode hard-trim on
  literal markers. Guarded by `tests/test_evaluation_translation.py`.

- **WCM-v2 — free-form generation cannot return a label.** The legacy
  `_classify_uyghur` generated a sentence and substring-matched against
  the candidate labels. On a 6-class task with a 85.3 % majority floor
  this produced *below-random* accuracy across all three variants. Fix:
  constrained log-likelihood scoring — score each candidate label under
  teacher forcing and return the argmax. Guarded by
  `tests/test_evaluation_wcm.py`.

Both fixes are inference-only and do not require retraining; the same
`run_20260524_020432` adapter is re-evaluated against the patched
pipeline.

- **UG→EN repetition collapse (2026-05-27).** Slurm 2766 (`debug_ug2en`,
  n=20) showed the fine-tuned adapter often enters a greedy `"The 2 1 1 1 …"`
  loop on UG→EN while zero-shot stays source-anchored. Fix:
  `repetition_penalty=1.15` + `no_repeat_ngram_size=4` in
  `generate_translation` when the target language is English. Slurm 2768
  re-eval: `qwen_finetuned` UG→EN chrF 9.385 → 16.8079 (+7.42, +79 %)
  with EN→UG byte-identical; residual −13.29 chrF gap to zero-shot is
  training-shaped. Zero-shot sanity-gate re-run pending (`TODO.md`).
  Mechanism, data audit, and pre/post table: `PROJECT_REFINEMENT.md` §14.