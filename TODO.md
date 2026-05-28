# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

> **Experimentation closed.** No further fine-tunes or eval-protocol
> changes are planned — see `docs/PROJECT_RESULTS.md` §2 + §3 for the
> measured tables. Remaining items are inference verification, qualitative
> data capture, and write-up. The two open Slurm jobs below are the
> **last** GPU runs of the project.

## In-flight: exp-0 rep-penalty-only zero-shot sanity gate (Slurm 2771)

`run_20260528_103619`. Qwen done (FLORES UG→EN **29.5635**, within ±0.5
of 30.10 — gate **pass**). `llama_zeroshot` FLORES was at **700/1012**
in the pulled log; WCM + PPL still to come.

- **Pull when done:** `eval_*_llama_zeroshot.json`, slurm log.
- **Pass:** `llama_zeroshot` UG→EN chrF within ±0.5 of **4.71** (Slurm
  2749) and WCM/PPL byte-identical.
- **Log:** §1 entry (single paragraph) closing Slurm 2768's open sanity
  item.

## Verification — WCM Mix-50 audit (debug_wcm.py)

Mix-50 reported WCM **81.00 %** (243/300). Majority-class floor is
**85.3 %** (256/300 = label `1`). `scripts/debug_wcm.py` records the
full per-label log-prob distribution + confusion matrix to decide
whether this is real classification or majority-class collapse.

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/

ssh ju-compute-server "cd ~/uyghurGPT && mkdir -p results results/debug && sbatch \
  --job-name=debug_wcm --time=00:30:00 --ntasks=1 --cpus-per-task=8 \
  --mem=24G --gres=gpu:1 --partition=priority --requeue \
  --output=results/slurm_debug_wcm_%j.out \
  --wrap='cd \$HOME/uyghurGPT && set -a && source .env && set +a && \
    export HF_HOME=\$HOME/uyghurGPT/hf_cache && \
    export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN && \
    export CUDA_VISIBLE_DEVICES=0 && \
    export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync && \
    export PYTHONUNBUFFERED=1 && \
    \$HOME/micromamba/envs/uyghur_env/bin/python -u scripts/debug_wcm.py \
      --compare-zeroshot \
      --out results/debug/wcm_mix50_vs_zs.json'"
```

**Decision after pull:**

| Mix-50 `majority_class_share_pred` | Reading |
|------------------------------------|---------|
| ≥ 0.95 | **Collapse.** §3 bonus annotates the 81 % cell as such; report flags as Mix-50 over-fit signal, not classification competence. |
| 0.6–0.95 + non-majority F1 ≥ 0.2 | Real (mixed). Keep cell; describe in §05 §8 honestly. |
| < 0.6 + spread predictions | Real classification. Headline-worthy. |

Same script for free zero-shot comparison: `pred_dist` should be diverse
on `qwen_zeroshot` (which scores 6.33 % — its predictions definitionally
miss most rows; we want to see *how* they miss).

## Qualitative examples — Option B (final inference job)

`scripts/qualitative_examples.py` runs 4 variants × 2 directions × 5
FLORES devtest sentences (default ids `0 1 2 3 4`) and emits both:

- `results/reports/qualitative_examples.json` (structured rows)
- `results/reports/qualitative_examples.md` (4-variant tables per
  direction; suitable for the report)

Per-cell content: hypothesis + sentence chrF. Variants reuse the same
decode paths as §2 (rep-penalty chat for instruct variants; 3-shot
base-LM continuation for `cute_llama_p`). FT adapter defaults to the
Mix-20 `run_20260524_020432` checkpoint that produces the §2 row; pass
`--ft-adapter` to swap if needed.

**Cost:** ~15 min wall (10 generations per variant; cute_llama_p
dominates at ~30 s/sentence in fp16). 1 h walltime is generous.

```bash
ssh ju-compute-server "cd ~/uyghurGPT && mkdir -p results results/reports && sbatch \
  --job-name=qualitative --time=01:00:00 --ntasks=1 --cpus-per-task=8 \
  --mem=24G --gres=gpu:1 --partition=priority --requeue \
  --output=results/slurm_qualitative_%j.out \
  --wrap='cd \$HOME/uyghurGPT && set -a && source .env && set +a && \
    export HF_HOME=\$HOME/uyghurGPT/hf_cache && \
    export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN && \
    export CUDA_VISIBLE_DEVICES=0 && \
    export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync && \
    export PYTHONUNBUFFERED=1 && \
    \$HOME/micromamba/envs/uyghur_env/bin/python -u scripts/qualitative_examples.py'"
```

**Pull:**

```bash
rsync -avz ju-compute-server:~/uyghurGPT/results/reports/qualitative_examples.{json,md} results/reports/ \
  && rsync -avz ju-compute-server:~/uyghurGPT/results/slurm_qualitative_*.out results/
```

`results/` is gitignored — once pulled, copy `qualitative_examples.md`
into `docs/qualitative_examples.md` (or `git add -f`) if the markdown
table needs to ship with the report.

---

## Write-up phase (no more GPU jobs)

After the three pulls above (`2771`, `debug_wcm`, `qualitative_examples`):

1. **§1 + §3 log updates** (`PROJECT_RESULTS.md`, ~30 min).
2. **Task 05** — `docs/05_results_analysis.md` (8-section structure per
   `docs/tasks/05_results_analysis.md`).
3. **Task 06** — final report / slides.
4. **Optional Task 04** — `scripts/aggregate_results.py` (§2 already
   serves as the canonical table; skippable).

---

## Deferred (do not run unless write-up demands a number)

- **A1 beams** — code-only, default off. §2 would require parallel re-run
  of all chat-path variants. Don't enable.
- **B1 + B2 retrain** — Mix-50 sits at UG→EN 17.97 (just under the
  "≥ 18" line); B1+B2 was the next training fix if we kept going.
- **Mix-0 / Mix-10 bracket** — `docs/tasks/bonus/02_qwen_mix_ablation.md`.
- **A2 chat-fewshot diagnostic** — not needed for §2; useful only as
  side-evidence in the report if there's time.
- **200 k / 300 k pair count** — Mix-20 early-stopped at 1.48 epochs;
  quantity not the binding constraint.

---

## Done (remove when read)

- ~~Mix-50 retrain~~ — `run_20260527_185416`, Slurm 2770; UG→EN
  9.39 → 17.97 (+1.16 over Mix-20).
- ~~Slurm 2768 `qwen_finetuned` UG→EN re-eval~~ — §2 UG→EN **16.8079**.
- ~~Slurm 2766 `debug_ug2en`~~ — mechanism in §14.
- ~~Training-data audit~~ — balanced `ug2en`/`en2ug`.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 (Mix-20)~~ — Slurm 2750 / 2749.
- ~~A1/A2 implementation (code)~~ — commit `9b6141d`; A1 eval **not** run.
