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
| 01 | **Experiment 2 — CUTE-Llama-P few-shot baseline.** New eval-only experiment dir mirroring experiment 0, but loading `CMLI-NLP/CUTE-Llama` (subfolder `CUTE-Llama-Parallel`) and prompting it as a base LM (FLORES few-shot continuation). | [`01_experiment_2_cute_llama_p_baseline.md`](01_experiment_2_cute_llama_p_baseline.md) |
| 02 | **WCM-v2 re-evaluation.** Re-run WCM-v2 on `qwen_zeroshot`, `llama_zeroshot`, `qwen_finetuned`, `cute_llama_p` with the fixed `minority/ug.txt` loader (run_20260524_020432 returned `ERROR` for WCM on every variant). | [`02_wcm_v2_reevaluation.md`](02_wcm_v2_reevaluation.md) |
| 03 | **UG→EN decoding regression fix.** Diagnose and fix the chrF 30.29 → 9.38 UG→EN regression observed on the Mix-20 fine-tune (PROJECT_RESULTS.md flags it as likely template / stop-token mismatch, not catastrophic forgetting). Re-evaluate `qwen_finetuned` on FLORES after the fix. | [`03_ug2en_decoding_fix.md`](03_ug2en_decoding_fix.md) |
| 04 | **Consolidated results table.** Aggregate experiments 0 + 1 + 2 artifacts into a single canonical comparison table (`qwen_zs` / `llama_zs` / `qwen_ft` / `cute_llama_p` × FLORES EN→UG / UG→EN chrF + BLEU / WCM acc / C4 PPL) for the report and append a `PROJECT_RESULTS.md` entry. | [`04_consolidated_results_table.md`](04_consolidated_results_table.md) |
| 05 | **Results analysis.** EN↔UG asymmetry write-up, 3–5 qualitative examples per direction per model, delta-over-zero-shot framing, success-criteria check (minimum / target / stretch). | [`05_results_analysis.md`](05_results_analysis.md) |
| 06 | **Final report.** Compile the academic write-up (intro, related work, approach, experiments, results, analysis, limitations, conclusions). | [`06_final_report.md`](06_final_report.md) |

## Bonus tasks (run only after the main path is green)

See [`bonus/README.md`](bonus/README.md) for the index. Each bonus task
ships independently and is *not* a prerequisite for any main task.

| # | Task | File |
|---|------|------|
| B1 | LLaMA-3.1-8B-Instruct Mix-20 QLoRA fine-tune (experiment 3) | [`bonus/01_experiment_3_llama_mix20_finetune.md`](bonus/01_experiment_3_llama_mix20_finetune.md) |
| B2 | Qwen Mix-{0, 10, 50} catastrophic-forgetting ablation (experiment 1 reruns) | [`bonus/02_qwen_mix_ablation.md`](bonus/02_qwen_mix_ablation.md) |
| B3 | MiLiC-Eval 9-task bilingual benchmark add-on (experiment 4) | [`bonus/03_milic_eval.md`](bonus/03_milic_eval.md) |
| B4 | Qwen2.5-7B 5-shot in-context baseline (experiment 5) | [`bonus/04_qwen_5shot_baseline.md`](bonus/04_qwen_5shot_baseline.md) |
