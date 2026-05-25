# Bonus B3 — Experiment 4: MiLiC-Eval 9-task bilingual benchmark

> **Status:** not started.
> **Depends on:** main path stable (the four variants from Tasks 01–03
> are what MiLiC-Eval scores; without them this benchmark has nothing
> to compare against).
> **Estimated wall-clock:** ~6 h per variant (9 tasks × ~300 rows each
> × ~1 s/row generation). Total for 4 variants: ~24 h, parallelisable.

## Goal

Add the **MiLiC-Eval** 9-task bilingual benchmark
(`pkupie/milic-eval`) as a stretch evaluation, scored on every variant
already covered by FLORES + WCM + C4 PPL. MiLiC-Eval adds *breadth* —
9 task types (reading comprehension, NLI, sentiment, etc.) — over
WCM-v2's single classification task.

This is the only stretch eval explicitly named in `docs/PROJECT.md`
§Stretch goals (bullet 3) and §Stretch evaluation. It is *not* a new
fine-tune; it is a new benchmark evaluated on existing checkpoints.

## Deliverables

1. `experiments/experiment_4/` package mirroring `experiments/experiment_0/`
   (eval-only, no preprocess/train). The variants evaluated are
   whatever subset of `(qwen_zeroshot, llama_zeroshot, qwen_finetuned,
   cute_llama_p, llama_finetuned)` is available at run time.
2. `shared/evaluation_milic.py` — separate file (do not bloat
   `shared/evaluation.py`) implementing one function per MiLiC task,
   plus a `run_milic(model, tokenizer, max_samples)` dispatcher that
   returns a `dict[str, dict]` shape `{task_name: {accuracy/f1/em/...}}`.
3. Per-variant artifacts written under
   `results/run_<id>/experiment_4/artifacts/milic_<variant>.json` and
   an aggregated `eval_summary.json`.
4. `scripts/aggregate_results.py` extended to merge a MiLiC column /
   sub-table into the canonical report — best handled as a separate
   `mil_table.md` rather than widening the main 8-column table.
5. A new section in `docs/PROJECT_RESULTS.md`.

## Implementation plan

### Step 1 — confirm dataset access

MiLiC-Eval is gated on HuggingFace (`pkupie/milic-eval`, gating is
instant — same flow as WCM-v2; `docs/PROJECT.md` §Data Available
notes this). Verify locally:

```python
from datasets import load_dataset
ds = load_dataset("pkupie/milic-eval", split="test", token=os.environ["HF_TOKEN"])
print(ds)
```

If access fails, ungate via `huggingface.co/datasets/pkupie/milic-eval`
and re-confirm.

### Step 2 — implement per-task evaluators

MiLiC-Eval has 9 tasks (full list in the dataset card). Group them by
output type:

- **Multiple-choice / classification** → score by exact-match against
  the gold label, using a few-shot prompt + label-set constraint
  similar to `shared/evaluation.py::_classify_uyghur`.
- **Extractive / short-form generation** → score by exact-match and
  token-F1 against the gold answer.
- **Generation / free-form** → score by chrF against the reference
  (re-use `sacrebleu.corpus_chrf`).

Implement the per-task scorers in a single dispatch table keyed by
task name:

```python
TASK_SCORERS = {
    "task1_name": _score_multiple_choice,
    "task2_name": _score_short_form,
    ...
}
```

This keeps the file linear and testable.

### Step 3 — base-LM vs chat-template prompting

`cute_llama_p` cannot use the chat template; route it through
`generate_translation_fewshot`-style continuation prompts the same
way `tasks/01` did. The two prompt paths share the same scoring code.

### Step 4 — wire CLI + Slurm

- `main.py::run_experiment` branch for `args.experiment == 4`.
- `scripts/push.py` accepts `--experiment 4` with `--time 1-00:00:00`
  by default (more generous than experiment 0 because there are 9
  tasks per variant, not 3 benchmarks).
- One Slurm run per variant (or one combined run if memory swap-out
  is reliable; the existing `run_eval` already does
  `del model; torch.cuda.empty_cache()` between variants).

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 4 --mode eval --new-run --time 1-00:00:00
```

### Step 5 — report integration

MiLiC-Eval's 9-task table is too wide for the canonical comparison
table. Emit it as a separate `results/reports/milic_table.md` with
shape:

| Variant         | task1 | task2 | … | task9 | mean |
|-----------------|-------|-------|---|-------|------|
| qwen_zeroshot   | …     | …     | … | …     | …    |
| llama_zeroshot  | …     | …     | … | …     | …    |
| qwen_finetuned  | …     | …     | … | …     | …    |
| cute_llama_p    | …     | …     | … | …     | …    |

Reference this from the report's §Results as a separate sub-table
(or appendix if space is tight).

## Validation / success criteria

1. Every variant's `milic_<variant>.json` exists and contains a score
   for all 9 tasks (no `"status": "ERROR"`).
2. The mean column for `qwen_finetuned` is **above** the mean for
   `qwen_zeroshot` — otherwise the Uyghur fine-tune is not
   transferring to the broader Uyghur task suite and the report needs
   to lead with that explanation.
3. `cute_llama_p`'s mean is reported alongside ours with the same
   prompt-style disclaimer used everywhere else.
4. `pytest tests/` still passes; a new
   `tests/test_milic_loader.py` smoke test (network-skipped if
   `HF_TOKEN` is missing) validates the per-task field selectors.

## References

- Benchmark id + access: `docs/PROJECT.md` §Data Available
  (MiLiC-Eval row).
- Why stretch: `docs/PROJECT_REFINEMENT.md` §4 (deferred from primary
  to stretch — keep the 9-task complexity out of the main report
  unless time allows).
- Reusable evaluation primitives: `shared/evaluation.py::_corpus_scores`,
  `_classify_uyghur`, `generate_translation`,
  and the Task-01 base-LM `generate_translation_fewshot`.
