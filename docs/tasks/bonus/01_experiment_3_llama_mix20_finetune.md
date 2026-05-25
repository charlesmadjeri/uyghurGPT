# Bonus B1 — Experiment 3: LLaMA-3.1-8B Mix-20 QLoRA fine-tune

> **Status:** not started.
> **Depends on:** main path stable (Tasks 01–04 ideally landed so the
> baselines this is compared against are not in flux).
> **Estimated wall-clock:** ~20 h on the 24 GB MIG slice for the same
> Mix-20 100k-pair recipe (LLaMA-3.1-8B is slightly heavier than
> Qwen2.5-7B; budget +30 % over the experiment-1 observed wall of
> ~15h52m). Use `--time 1-12:00:00` for the train pass and the
> experiment-1 eval default for the eval pass.

## Goal

Fine-tune the secondary model — LLaMA-3.1-8B-Instruct — with the same
QLoRA Mix-20 recipe used for Qwen in experiment 1, and evaluate it as a
fourth variant `llama_finetuned` on FLORES + WCM + C4.

This is a stretch goal (`docs/PROJECT.md` §Stretch goals, bullet 1). It
adds a *cross-architecture* data point: same training recipe, different
base model. Useful for the analysis ("Qwen's stronger native UG
tokenization carries through to a bigger LoRA gain") and explicitly
listed in the success criteria as the stretch tier.

## Deliverables

1. `experiments/experiment_3/` package mirroring `experiments/experiment_1/`:
   - `config.py` — `Experiment3Config` identical to `Experiment1Config`
     except `experiment_id = 3`, `model = "llama"`,
     `eval_variants = ("llama_finetuned",)`. Same Mix-20 default,
     same sample_count cap, same LoRA r/α, same packing flag.
   - `run.py` — copy of `experiments/experiment_1/run.py` with class +
     stage labels relabelled.
2. `MODEL_IDS` already has `"llama": "meta-llama/Llama-3.1-8B-Instruct"`;
   no `shared/models.py` change required.
3. `shared/evaluation.py` — extend `ALL_EVAL_VARIANTS` and
   `_variant_specs` with `"llama_finetuned"`, mirroring the existing
   `qwen_finetuned` branch (looks up adapter via `cfg.model_label`).
4. `shared/training.py` — already model-agnostic via
   `Experiment1Config.model`; only verify the LLaMA chat-template
   response-template path works:
   `shared/models.py::response_template("llama")` returns
   `"<|start_header_id|>assistant<|end_header_id|>\n\n"`. Smoke-test
   locally with `--sample-count 100 --epochs 1` before submitting a
   real run.
5. `main.py::run_experiment` branch for `args.experiment == 3`.
6. `scripts/push.py` accepts `--experiment 3` and picks
   `--time 1-12:00:00` by default (longer than experiment 1's
   1-00:00:00 because LLaMA is ~15 % heavier and the timeline cushion
   has bitten before).
7. One full Slurm run producing
   `results/run_<id>/experiment_3/artifacts/eval_summary.json` with
   the `llama_finetuned` row populated.
8. A new section in `docs/PROJECT_RESULTS.md` for this run.

## Implementation plan

### Step 1 — clone experiment 1

```
cp -r experiments/experiment_1 experiments/experiment_3
```

Then in `experiments/experiment_3/config.py`:

- `experiment_id: int = 3`
- `model: str = "llama"`
- `eval_variants: tuple[str, ...] = ("llama_finetuned",)`

In `experiments/experiment_3/run.py`, replace every `Experiment1`
reference with `Experiment3` and every `Experiment 1 —` log label
with `Experiment 3 —`.

### Step 2 — local smoke test

The full pipeline must run end-to-end on the laptop (CPU is fine for
preprocess, GPU optional but recommended) at `--sample-count 100
--epochs 1` before any cluster time is burned.

```bash
python3 main.py --experiment 3 --model llama --mix 20 \
  --epochs 1 --sample-count 100 --run-id smoke_llama
```

Expected at the end: `results/run_smoke_llama/experiment_3/artifacts/
eval_summary.json` with a `llama_finetuned` block (FLORES + C4 PPL
populated; WCM may say ERROR locally without HF token — that is fine
in smoke mode).

### Step 3 — submit on the cluster

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 3 --model llama --new-run --time 1-12:00:00
```

Monitor + pull the same way as experiment 1.

### Step 4 — re-aggregate

Re-run `python3 scripts/aggregate_results.py` (Task 04's script) so
`results/reports/consolidated_results.json` gains the `llama_finetuned`
row. The aggregator should handle a new variant transparently as long
as it appears in `eval_summary.json` and was added to
`ALL_EVAL_VARIANTS`.

## Validation / success criteria

1. `eval_summary.json` for the run has a `llama_finetuned` block with
   FLORES EN→UG / UG→EN, WCM, and C4 PPL populated and no
   `"status": "ERROR"`.
2. `llama_finetuned.flores.en2ug.chrF` ≥ `llama_zeroshot.flores.en2ug.chrF
   + 5` — LLaMA-3.1's zero-shot UG capability is essentially 0 (`docs/
   PROJECT_RESULTS.md` 2026-05-24 has `llama_zs` at chrF 0.84), so a
   working fine-tune **must** move it. If it does not, the recipe is
   broken or the adapter loading is wrong; do not ship this row to
   the report.
3. C4 PPL stays within 50 % of `llama_zeroshot.perplexity` (13.69 in
   2026-05-24). A blow-up means catastrophic forgetting; document it
   if it happens, but the report's headline should not lead with a
   broken LLaMA fine-tune.
4. `pytest tests/` still passes; a new `tests/test_experiment3_config.py`
   mirroring `tests/test_config.py` for the new dataclass is a fair
   addition.
5. The Task-04 aggregator picks up `llama_finetuned` without code
   changes.

## References

- Experiment-1 recipe to clone: `experiments/experiment_1/`,
  `shared/training.py`, `shared/data.py`.
- LLaMA-3.1 chat template / response template:
  `shared/models.py::response_template("llama")`.
- Stretch-goal scope:
  `docs/PROJECT.md` §Stretch goals (bullet 1), §Success criteria
  (Stretch tier).
- Wall-clock baseline:
  `docs/PROJECT_RESULTS.md` 2026-05-24 (experiment 1 observed
  ~15h52m on the 24 GB slice).
