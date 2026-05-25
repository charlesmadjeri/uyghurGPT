# Bonus B2 — Qwen Mix-{0, 10, 50} ablation

> **Status:** not started.
> **Depends on:** main path stable. Each Mix cell is an independent
> Slurm job; they can run in parallel across workers (the cluster has
> 7 workers, our quota allows 3 concurrent).
> **Estimated wall-clock:** ~16 h per cell on the 24 GB MIG slice
> (same recipe as experiment 1's Mix-20 run; FLAN tokenization is the
> dominant variable, but at sample-count 100k the difference between
> Mix-0 and Mix-50 is < 30 min). Total serial budget: ~48 h for the
> three cells; ~16–24 h parallelised.

## Goal

Measure how the CUTE-P / FLAN mixing ratio trades off **Uyghur
capability gain** (chrF EN↔UG, WCM-v2 acc) against **English retention**
(C4 PPL, plus a peek at the catastrophic-forgetting signal in
FLORES UG→EN where the model has to generate English).

Run three additional Qwen fine-tunes at Mix-0, Mix-10, Mix-50 — *same
code, same data, same seeds*, only the `--mix` value changes.
Compare against the Mix-20 cell from experiment 1.

If time forces a cut, drop Mix-10 first (Mix-0 and Mix-50 bracket the
ratio space — `docs/PROJECT_REFINEMENT.md` §8 explicitly endorses this
fallback).

## Deliverables

1. Three Slurm runs:
   - `run_<id>_mix0/experiment_1/` (Mix-0, no FLAN)
   - `run_<id>_mix10/experiment_1/` (Mix-10)
   - `run_<id>_mix50/experiment_1/` (Mix-50)
   Each has its own `eval_summary.json` with a `qwen_finetuned` row
   (the variant label is unchanged; the run *id* is what distinguishes
   them downstream).
2. The aggregator (Task 04) extended to also recognise the
   `(variant=qwen_finetuned, mix=N)` axis. Concretely, the
   per-variant key becomes `(variant, mix)`; the canonical comparison
   table still lists `qwen_finetuned` = Mix-20, while the ablation
   table is a separate output.
3. `results/reports/mix_ablation.md` — table indexed by Mix ratio:

   | Mix | FLORES EN→UG chrF | UG→EN chrF | WCM-v2 acc | C4 PPL | Δ C4 PPL vs ZS |
   |-----|-------------------|------------|------------|--------|----------------|
   | 0   | … | … | … | … | … |
   | 10  | … | … | … | … | … |
   | 20  | … | … | … | … | … |
   | 50  | … | … | … | … | … |

4. A new section in `docs/PROJECT_RESULTS.md` per cell (one per run id,
   per the file's append-only convention).
5. If the report (Task 06) is being written, an ablation subsection
   referencing this table.

## Implementation plan

### Step 1 — three cluster runs

No code changes are needed; the `--mix` flag is already wired through
`experiments/experiment_1/config.Experiment1Config.from_namespace`.
Submit three independent runs with `--new-run` so each gets its own
run directory and `eval_summary.json`:

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 0  --new-run --time 1-00:00:00

python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 10 --new-run --time 1-00:00:00

python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 50 --new-run --time 1-00:00:00
```

Monitor + pull each with `scripts/check.py` as usual.

### Step 2 — extend the aggregator

In `scripts/aggregate_results.py` (Task 04):

- When iterating runs, also read each `run_config.json`'s `mix` field
  if the variant is `qwen_finetuned`. Group results by the
  `(variant, mix)` tuple instead of just `variant`.
- The default canonical table keeps `(qwen_finetuned, mix=20)` as the
  headline row (the main fine-tune).
- Add a `--ablation` flag that emits `results/reports/mix_ablation.md`
  with rows for `mix ∈ {0, 10, 20, 50}` instead of the four-variant
  table.

### Step 3 — update PROJECT_RESULTS.md

One section per Mix cell, dated with its run id, following the
template at the bottom of `docs/PROJECT_RESULTS.md`. Each section's
Analysis paragraph should compare to the Mix-20 reference (chrF
delta, PPL delta) — i.e. answer "did adding 0/10/50 % FLAN buffer
help or hurt?"

## Validation / success criteria

1. Three `eval_summary.json` files exist, one per cell, each with a
   numeric `qwen_finetuned` block (FLORES + WCM + PPL).
2. `results/reports/mix_ablation.md` exists with 4 rows (Mix-0/10/20/50)
   all numerically populated.
3. The Mix-0 cell shows lower C4 PPL retention (i.e. *higher* PPL than
   Mix-20) — or, if not, that is a real surprise and goes into the
   analysis honestly.
4. The Mix-50 cell shows lower FLORES EN→UG chrF than Mix-20 — or, if
   not, the report's "Mix-20 is the sweet spot" framing must be
   softened.
5. `pytest tests/` still passes; the existing `test_flan_count_for_mix_*`
   tests already cover the Mix-ratio math on the data side.

## References

- Mix table definition: `docs/PROJECT.md` §Data Mixing —
  Catastrophic Forgetting.
- Scope rationale: `docs/PROJECT_REFINEMENT.md` §8 (Mix-{0, 20} only
  fallback if time is short).
- Variant + adapter discovery code:
  `shared/evaluation.py::_variant_specs` and `_find_adapter_path`
  (lines 42–51, 312–325).
