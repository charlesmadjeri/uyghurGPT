# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

## Priority — Mix-50 retrain (training-side fix, no eval-protocol bias)

**Why this instead of A1 (beams).** Slurm 2768 fixed the greedy repetition
collapse (+7.42 UG→EN chrF) but left a **−13.29 chrF** gap to zero-shot
(30.10). §14 attributes that residual to gradient / `eval_loss` checkpoint
bias toward Uyghur-output rows and FLAN-shaped English completions — a
**training** problem. Beams change decoding for some variants only and
would require re-running every §2 row in parallel to stay fair; Mix-50
changes only the recipe, not the eval protocol (same `generate_translation`
path as Slurm 2768 for all models).

**Hypothesis.** More FLAN dilution (Mix-50 vs Mix-20) shifts the
checkpoint toward English assistant spans and should lift UG→EN chrF at
some cost to EN→UG chrF. Realistic target: UG→EN **20–25** chrF (not
guaranteed to reach 30 — that likely needs B1/B2 if Mix-50 stalls).

**Row mix at 100 k CUTE-P pairs** (same `sample_count` as Mix-20 run):

| Bucket | Mix-20 | Mix-50 |
|--------|--------|--------|
| EN→UG / UG→EN (each) | 100 k | 100 k |
| FLAN EN-only | 25 k (~11 %) | **100 k (~33 %)** |

**Not in scope yet:** 200 k / 300 k pair counts — Mix-20 early-stopped at
~1.48 epochs; quantity is unlikely to be the binding constraint until the
mix / checkpoint mechanism is tested. Revisit only if Mix-50 plateaus.

**Rsync + push** (full pipeline: preprocess + train + eval; ~15–20 h):

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

**Pull + log** (same commit for §1 + new row):

- `results/run_<id>/experiment_1/artifacts/eval_summary.json`
- Checkpoint dir: `checkpoints/qwen_mix50/final`
- Add §2 row `qwen_finetuned_mix50` (or document under §3 bonus table until
  Task 04 aggregator exists) with FLORES / WCM / C4 vs Mix-20 (Slurm 2768)
  and `qwen_zeroshot` (30.10 / 9.96).
- **Do not** enable `UYGHUR_UG2EN_NUM_BEAMS` — eval must match Slurm 2768
  protocol (rep-penalty only, default beams=1).

**Decision after Mix-50 lands:**

| UG→EN chrF vs Slurm 2768 (16.81) | Next step |
|----------------------------------|-----------|
| **≥ 22** (+5 chrF) | Write up mix trade-off; optional Mix-0 for bracket; skip B1+B2 for now |
| **18–22** (+1 to +5) | Consider **B2** alone on a new run (direction-stratified `eval_loss`) |
| **< 18** (no meaningful lift) | **B1 + B2** retrain (`ug2en` row weight 2× + per-direction early stopping) |

Full ablation spec: `docs/tasks/bonus/02_qwen_mix_ablation.md`.

---

## In-flight: exp-0 rep-penalty-only zero-shot sanity gate

> Submitted **before** commit `9b6141d`. Cluster code: rep-penalty +
> `no_repeat_ngram_size` on UG→EN; **no** beam env hook.

- **Pass:** `qwen_zeroshot` UG→EN within ±0.5 of **30.10**; `llama_zeroshot`
  UG→EN within ±0.5 of **4.71**.
- **Pull + log:** §1 entry; closes Slurm 2768 open sanity item.
- **Do not cancel** for Mix-50 — orthogonal measurement.

---

## Optional — A2 diagnostic only (does not change §2)

Runs while Mix-50 trains or after rsync. Informs whether the adapter still
has a UG→EN circuit (prompt vs weights); **not** a substitute for Mix-50.

```bash
ssh ju-compute-server 'cd ~/uyghurGPT && \
  $HOME/micromamba/envs/uyghur_env/bin/python scripts/debug_ug2en.py \
    --fewshot-k 3 -n 50 --compare-zeroshot \
    --out results/debug/ug2en_fewshot_$(date -u +%Y%m%d_%H%M%S).json'
```

Log mean chrF + failure modes in §1 (diagnostic paragraph only).

| A2 FT mean chrF (n=50) | Note for report |
|------------------------|-----------------|
| ≥ 27 | Circuit intact; Mix-50 + prompt shape matter |
| 22–27 | Partly intact |
| < 22 | Weights shifted; Mix-50 / B1+B2 more urgent |

---

## Deferred — A1 beam search (eval-protocol change)

Code remains in repo (`UYGHUR_UG2EN_NUM_BEAMS`, default **off**). **Paused**
because adopting beams in §2 requires parallel re-eval of **all** chat-path
variants (`qwen_zeroshot`, `llama_zeroshot`, `qwen_finetuned`, and ideally
a few-shot-path policy for `cute_llama_p`) — otherwise the table is biased.

Revisit only if Mix-50 + B1/B2 plateau and we need a decoding-only lift with
full parity re-run budget (~3 h).

---

## Deferred — B1 + B2 (retrain mechanics)

See Mix-50 decision table above. Spec: `PROJECT_REFINEMENT.md` §14.

---

## Done (remove when read)

- ~~Slurm 2768 `qwen_finetuned` UG→EN re-eval~~ — §2 UG→EN **16.8079**.
- ~~Slurm 2766 `debug_ug2en`~~ — mechanism in §14.
- ~~Training-data audit~~ — balanced `ug2en`/`en2ug`.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 (Mix-20)~~ — Slurm 2750 / 2749.
- ~~A1/A2 implementation (code)~~ — commit `9b6141d`; A1 eval **not** run.
