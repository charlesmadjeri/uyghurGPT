# Bonus B2 — Qwen Mix-{0, 10, 50} ablation

> **Status:** **Mix-50 active** (priority retrain after Slurm 2768 / §14).
> Mix-0 and Mix-10 remain **deferred** until Mix-50 lands or time forces a
> bracket-only cut (`PROJECT_REFINEMENT.md` §8 fallback).
> **Depends on:** main path stable; in-flight exp-0 rep-penalty sanity gate
> may finish in parallel.
> **Estimated wall-clock:** ~15–20 h per cell on the 24 GB MIG slice
> (same recipe as `run_20260524_020432` Mix-20; Mix-50 adds FLAN rows,
> not more CUTE-P pairs).

## Why Mix-50 is priority now (2026-05-27)

Slurm 2768 recovered UG→EN chrF **9.39 → 16.81** with direction-conditional
`repetition_penalty` only. The remaining **−13.29 chrF** gap to
`qwen_zeroshot` (30.10) is **training-shaped** (`assistant_only_loss`
gradient bias + aggregated `eval_loss` early stopping), not missing UG→EN
rows and not fixable by eval-only tricks without re-running every §2 variant
under the same decoder (beam search was **paused** for that reason — see
`TODO.md`).

Mix-50 is the **single-variable** training change aligned with §14: more
FLAN EN-only rows → more gradient on English assistant spans → checkpoint
less dominated by EN→UG (Uyghur-output) CE. It does **not** change FLORES /
WCM / C4 eval protocol vs Slurm 2768.

**Realistic expectations** (not guarantees):

| Metric | Mix-20 (Slurm 2768) | Plausible Mix-50 |
|--------|---------------------|------------------|
| FLORES EN→UG chrF | 14.18 | 11–13 (may drop) |
| FLORES UG→EN chrF | 16.81 | 20–25 |
| C4 EN PPL | 16.17 | ~flat |
| WCM | 21.00 % | ~18–22 % |

Matching zero-shot UG→EN (**30 chrF**) in one Mix-50 run is **unlikely**
without additionally changing loss weighting (B1) or checkpoint selection
(B2). If Mix-50 UG→EN **< 18**, plan B1+B2 retrain (`TODO.md`).

**200k / 300k CUTE-P pairs** are **out of scope** for this cell: Mix-20
early-stopped at ~1.48 epochs on 100 k pairs — data quantity is not the
binding constraint until mix / checkpoint mechanics are tested.

## Goal

Measure how the CUTE-P / FLAN mixing ratio trades off **Uyghur
capability gain** (chrF EN↔UG, WCM-v2 acc) against **English retention**
(C4 PPL, plus FLORES UG→EN where the model generates English).

**Immediate deliverable:** one Mix-50 run (`--new-run`) compared to the
existing Mix-20 reference (`run_20260524_020432`, Slurm 2768 numbers).

**Stretch (if time):** Mix-0 and Mix-10 for a full `{0, 10, 20, 50}` table.

## Deliverables

1. **Active:** `run_<id>_mix50/experiment_1/` with `mix: 50` in
   `run_config.json`, checkpoint `checkpoints/qwen_mix50/final`, and
   `eval_summary.json` (`qwen_finetuned` variant label unchanged — **run id**
   distinguishes cells).
2. **Deferred:** `run_<id>_mix0/`, `run_<id>_mix10/`.
3. The aggregator (Task 04) extended to recognise `(variant, mix)` — see
   Step 2 below.
4. `results/reports/mix_ablation.md` — at minimum one Mix-50 row plus Mix-20
   reference; full 4-row table when Mix-0/10 land.
5. `docs/PROJECT_RESULTS.md` §1 entry per run id (append-only).

## Implementation plan

### Step 1 — Mix-50 cluster run (priority)

No code changes required; `--mix` is wired through
`experiments/experiment_1/config.Experiment1Config.from_namespace`.
Checkpoint label: `qwen_mix50` (`config.model_label`).

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/

python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 50 --new-run \
  --mode all --time 1-00:00:00
```

**Eval protocol:** do **not** set `UYGHUR_UG2EN_NUM_BEAMS` (default 1).
Post-train eval uses the same chat decode path as Slurm 2768 (rep-penalty
on English target only).

Monitor + pull with `scripts/check.py`. Log §1 + §2 (or §3 bonus row) in
the **same commit** as artifacts.

### Step 1b — optional parallel cells (deferred)

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 0  --new-run --time 1-00:00:00

python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --model qwen --mix 10 --new-run --time 1-00:00:00
```

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
template at the bottom of `docs/PROJECT_RESULTS.md`. Mix-50 Analysis
should compare to Mix-20 (Slurm 2768) on UG→EN / EN→UG / PPL and state
whether the §14 mechanism prediction held.

## Post–Mix-50 decision (B1 / B2)

| Mix-50 UG→EN chrF vs 16.81 | Action |
|----------------------------|--------|
| ≥ 22 | Document trade-off; optional Mix-0 bracket; defer B1+B2 |
| 18–22 | Consider B2-only retrain (direction-stratified `eval_loss`) |
| < 18 | B1+B2 combined retrain (`ug2en` weight 2× + per-direction checkpoint) |

Optional A2 diagnostic (`debug_ug2en.py --fewshot-k 3`) does not replace
this table — see `TODO.md`.

## Validation / success criteria

1. `eval_summary.json` exists with numeric `qwen_finetuned` block.
2. `run_config.json` shows `"mix": 50` and `sample_count` 100000 (unless
   explicitly overridden).
3. FLORES evaluated with rep-penalty UG→EN path, **no** beam env var.
4. Analysis honestly records EN→UG vs UG→EN trade-off vs Mix-20.
5. `pytest tests/` still passes (`test_flan_count_for_mix_*` covers Mix math).

## References

- Mix table definition: `docs/PROJECT.md` §Data Mixing.
- UG→EN mechanism: `docs/PROJECT_REFINEMENT.md` §14.
- Mix-20 reference run: `run_20260524_020432` (Slurm 2768 FLORES cells).
- Scope rationale: `docs/PROJECT_REFINEMENT.md` §8.
- Paused beam eval: `TODO.md` §Deferred A1.
