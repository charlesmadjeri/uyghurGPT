# Bonus tasks

Stretch goals listed in `docs/PROJECT.md` §"Stretch goals" and
`docs/PROJECT_REFINEMENT.md` §1 / §8. **None of these is required for
the minimum results** (which only needs fine-tuned Qwen vs. CUTE-Llama-P
vs. zero-shot Qwen — see `docs/tasks/README.md`). They are tracked here
so they can be picked up if the main path lands ahead of schedule.

## Gating

Each bonus task assumes:

- All main tasks (`docs/tasks/01_*` through `06_*`) are at least
  drafted (writing in parallel is fine, but no bonus task should
  *delay* the main report path).
- The cluster has free capacity on `slurm.hj.se` `priority` partition.
  Bonus runs go on `scavenger` (preemptible, unlimited time) if
  `priority` is queued.

## Index

| # | Task | Experiment id (if any) | File |
|---|------|------------------------|------|
| B1 | LLaMA-3.1-8B-Instruct Mix-20 QLoRA fine-tune (secondary model) | 3 | [`01_experiment_3_llama_mix20_finetune.md`](01_experiment_3_llama_mix20_finetune.md) |
| B2 | Qwen Mix-{0, 10, 50} catastrophic-forgetting ablation (reruns of experiment 1) | 1 (reruns) | [`02_qwen_mix_ablation.md`](02_qwen_mix_ablation.md) |
| B3 | MiLiC-Eval 9-task bilingual benchmark on every variant | 4 (new eval-only) | [`03_milic_eval.md`](03_milic_eval.md) |
| B4 | Qwen2.5-7B 5-shot ICL baseline (no weight updates) | 5 (new eval-only) | [`04_qwen_5shot_baseline.md`](04_qwen_5shot_baseline.md) |

## What goes in the final report if any of these complete

The report's §Experiments and §Results sections both have explicit
slots for additional experiments — see `docs/tasks/06_final_report.md`
§"Optional — incorporating bonus results". The rule is binary: a bonus
experiment is **either** fully run-evaluated-analyzed (and goes into
the report with its own row in the Task-04 table) **or** it is left
out entirely. Half-finished ablations are worse than no ablations.
