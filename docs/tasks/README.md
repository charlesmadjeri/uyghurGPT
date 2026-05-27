# Tasks index

Implementation-ready tasks that take the project from the current state
(experiments 0 + 1 implemented and run once) to a finished final report.

## Scope of "main" vs. "bonus"

The minimum the report must include is a comparison of our **fine-tuned
Qwen2.5-7B-Instruct** (experiment 1) against:

1. **CUTE-Llama-P** — the published baseline from Zhuang & Sun (COLING 2025)
   we are positioning against, **run by us on FLORES-200 EN↔UG + WCM-v2 + C4**
   with a base-LM-appropriate few-shot prompt.
2. **Qwen2.5-7B-Instruct zero-shot** — isolates the contribution of the LoRA
   fine-tune (same model family, no weight updates).

Everything beyond that comparison — LLaMA-3.1-8B fine-tune, Mix-{0,10,50}
ablations, MiLiC-Eval, few-shot ICL baselines — is in [`bonus/`](bonus/).

## Already implemented (reference, not a task)

| Experiment | What it does | Code | Latest run |
|------------|--------------|------|------------|
| **0** | Zero-shot FLORES + WCM + C4 PPL for `qwen_zeroshot` + `llama_zeroshot` (eval only, no training) | `experiments/experiment_0/`, `shared/evaluation.py` | `results/run_20260524_020432/experiment_1/artifacts/eval_summary.json` includes the `qwen_zeroshot` and `llama_zeroshot` rows from the legacy combined run |
| **1** | Qwen2.5-7B-Instruct QLoRA Mix-20 (preprocess + train + eval); eval runs `qwen_finetuned` only | `experiments/experiment_1/`, `shared/{data,training,evaluation}.py` | `results/run_20260524_020432/experiment_1/` (early-stopped at step 1550/3138, best `eval_loss` ≈ 1.523) |

## Main tasks (must complete for the report)

In implementation order. Each task is self-contained and references the
files / commands it touches.

| # | Task | File |
|---|------|------|
| 01 | **Experiment 2 — CUTE-Llama-P few-shot baseline.** Eval-only; `CMLI-NLP/CUTE-Llama` fp16 few-shot FLORES + `base_lm` WCM. **Done** — Slurm 2750 / `run_20260526_224102`: EN→UG chrF 6.88, UG→EN 23.09, WCM 15.33 %, C4 PPL 13.01 (~7 h wall). | [`01_experiment_2_cute_llama_p_baseline.md`](01_experiment_2_cute_llama_p_baseline.md) |
| 02 | **WCM-v2 re-evaluation.** Loader + constrained-LL scoring fixes (`PROJECT_REFINEMENT.md` §12–13). **Done** — all 4 variants: `qwen_ft` 21.00 %, `qwen_zs` 6.33 %, `llama_zs` 3.00 %, `cute_llama_p` 15.33 %. Core §2 WCM column is protocol-consistent. | [`02_wcm_v2_reevaluation.md`](02_wcm_v2_reevaluation.md) |
| 03 | **UG→EN decoding regression fix.** Stop/trim + WCM constrained-LL (§13); leak falsified (2744). Slurm 2766: repetition collapse + EN hallucinations; data audit ruled out missing UG→EN rows (§14). `repetition_penalty` for UG→EN shipped + re-eval'd (Slurm 2768): UG→EN chrF 9.385 → 16.8079 (+7.42); residual −13.29 chrF gap to zero-shot is training-shaped. Zero-shot sanity-gate re-run pending (`TODO.md`). | [`03_ug2en_decoding_fix.md`](03_ug2en_decoding_fix.md) |
| 04 | **Consolidated results table.** Aggregate experiments 0 + 1 + 2 artifacts into a single canonical comparison table (`qwen_zs` / `llama_zs` / `qwen_ft` / `cute_llama_p` × FLORES EN→UG / UG→EN chrF + BLEU / WCM acc / C4 PPL) for the report and append a `PROJECT_RESULTS.md` entry. | [`04_consolidated_results_table.md`](04_consolidated_results_table.md) |
| 05 | **Results analysis.** EN↔UG asymmetry write-up, 3–5 qualitative examples per direction per model, delta-over-zero-shot framing, success-criteria check (minimum / target / stretch). | [`05_results_analysis.md`](05_results_analysis.md) |
| 06 | **Final report.** Compile the academic write-up (intro, related work, approach, experiments, results, analysis, limitations, conclusions). | [`06_final_report.md`](06_final_report.md) |

## Bonus tasks (run only after the main path is green)

See [`bonus/README.md`](bonus/README.md) for the index. Each bonus task
ships independently and is *not* a prerequisite for any main task.

| # | Task | File |
|---|------|------|
| B1 | LLaMA-3.1-8B-Instruct Mix-20 QLoRA fine-tune (experiment 3) | [`bonus/01_experiment_3_llama_mix20_finetune.md`](bonus/01_experiment_3_llama_mix20_finetune.md) |
| B2 | **Mix-50 retrain active** (Mix-0/10 deferred); training-side UG→EN fix per §14; A1 beams paused | [`bonus/02_qwen_mix_ablation.md`](bonus/02_qwen_mix_ablation.md) |
| B3 | MiLiC-Eval 9-task bilingual benchmark add-on (experiment 4) | [`bonus/03_milic_eval.md`](bonus/03_milic_eval.md) |
| B4 | Qwen2.5-7B 5-shot in-context baseline (experiment 5) | [`bonus/04_qwen_5shot_baseline.md`](bonus/04_qwen_5shot_baseline.md) |
