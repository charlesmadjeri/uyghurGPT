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
| 01 | **Experiment 2 — CUTE-Llama-P few-shot baseline.** New eval-only experiment dir mirroring experiment 0, but loading `CMLI-NLP/CUTE-Llama` (subfolder `CUTE-Llama-Parallel`) and prompting it as a base LM (FLORES few-shot continuation). **Code shipped; results still pending — three submissions (Slurm 2745 / 2748 / 2750) have all stalled before any FLORES progress dots, even at 24 h walltime. Now the only outstanding cell in `PROJECT_RESULTS.md` §2. Investigation handoff in `TODO.md` "Investigate CUTE-Llama-P FLORES stall".** | [`01_experiment_2_cute_llama_p_baseline.md`](01_experiment_2_cute_llama_p_baseline.md) |
| 02 | **WCM-v2 re-evaluation.** Both bugs fixed in code: the `minority/ug.txt` loader (caught run_20260524_020432's `ERROR`) **and** the free-form-generation scoring bug that drove the first WCM backfill below random (`PROJECT_REFINEMENT.md` §13). Now uses constrained log-likelihood scoring. **3 of 4 variants done** under the same protocol: `qwen_finetuned` 21.00 % (Slurm 2744), `qwen_zeroshot` 6.33 %, `llama_zeroshot` 3.00 % (Slurm 2749). Fine-tune Δ over zero-shot is now apples-to-apples (+14.67 pp / ×3.3). `cute_llama_p` cell still blocked on Task 01. | [`02_wcm_v2_reevaluation.md`](02_wcm_v2_reevaluation.md) |
| 03 | **UG→EN decoding regression fix.** Decoding fix shipped (stop-token list + post-decode chat-marker trim in `shared/evaluation.py::generate_translation`). Slurm 2744 re-eval returned chrF / BLEU **byte-identical** to the May-24 pre-fix numbers, which **falsifies** the leak hypothesis: the 30.29 → 9.38 regression is genuine Mix-20 over-fitting on the generate-English direction, not a template artifact (`PROJECT_REFINEMENT.md` §13). Headline finding carried into Task 05. | [`03_ug2en_decoding_fix.md`](03_ug2en_decoding_fix.md) |
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
