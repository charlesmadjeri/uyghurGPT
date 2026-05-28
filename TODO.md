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

## Verification — WCM Mix-50 audit — **Pulled, verdict: real-but-majority-biased**

Slurm 2785 (`results/debug/wcm_mix50_vs_zs.json`, log
`results/slurm_debug_wcm_2785.out`). 300 rows, 6 classes with support
(gold: `1:256, 4:20, 6:8, 3:7, 9:6, 0:3`). The natural class
imbalance (label `1` covers 85.3 % of rows) makes raw accuracy a
biased headline. Class-balance-invariant metrics tell the real story:

| Metric | Mix-50 | Zero-shot | Floor |
|--------|--------|-----------|-------|
| Raw accuracy | **0.810** | 0.063 | 0.853 (always-maj) |
| **Balanced accuracy** = macro recall | **0.258** | 0.271 | 0.167 (uniform 1/6) |
| Macro precision | 0.203 | 0.216 | – |
| Macro F1 | **0.220** | 0.103 | – |
| Per-class recall `[0,1,3,4,6,9]` | `[0.00, 0.90, 0.00, 0.65, 0.00, 0.00]` | `[0.00, 0.02, 0.57, 0.20, 0.00, 0.83]` | – |
| `majority_class_share_pred` | 0.830 (< 0.95 collapse threshold) | 0.023 | – |
| Top-1 − top-2 log-prob margin | 0.29 (low conf) | 1.04 (confidently wrong) | – |

**Verdict.** Under the metric you asked for (class-balance-invariant),
**Mix-50 ≈ zero-shot**: 25.8 % vs 27.1 % balanced accuracy, both ~9 pp
above the 16.7 % uniform-random floor. Mix-50 has learned exactly two
things: (a) the prior `P(label = 1)`, and (b) the label-4 distinction
(R 0.65, F1 0.41). Labels 0/3/6/9 → 0 TPs. Mix-50 still beats zero-shot
on **macro F1** (0.22 vs 0.10), because its high majority-class
precision lifts the unweighted mean.

Zero-shot is the mirror image: almost never picks label 1 (R 0.02) but
over-fires label 9 (164 predictions for 6 true rows, R 0.83). Two
opposite failure modes converge on the same balanced accuracy by
accident.

### Why not just resample to a balanced set?

The natural file `minority/ug.txt` has only **300 rows** and label `0`
has only 3 samples. A stratified balanced subset is therefore capped
at **3 × 6 = 18 rows** — single-error swings ≥ 5 pp, useless for
ranking. **Macro recall** (above) is the textbook fix: it gives
exactly the property you want (defaulting to label 1 contributes
only `1/N_classes` to the score) **without** discarding 282 rows.

`scripts/debug_wcm.py` now emits `balanced_accuracy_macro_recall`,
`macro_precision`, `macro_f1`, and `uniform_floor_acc` on every run.
No GPU rerun needed for Mix-50 — the values above are derived from
the existing JSON.

### For the write-up

- `PROJECT_RESULTS.md` §3 Mix-50 row keeps `0.81` but adds a footnote:
  *"Raw accuracy near the 85.3 % majority floor. Balanced accuracy
  (macro recall) = 0.258, macro F1 = 0.220; zero-shot = 0.271 / 0.103.
  Mix-50's lift is concentrated on labels 1 and 4."*
- Task 05 §8 (negative results) gets the same line as the WCM signal.

## Qualitative examples — Option B (final inference job, **re-submission**)

`scripts/qualitative_examples.py` runs **5 variants** × 2 directions × 5
FLORES devtest sentences (default ids `0 1 2 3 4`):

| Variant | Adapter |
|---------|---------|
| `qwen_zeroshot` / `llama_zeroshot` | none |
| `qwen_finetuned_mix20` | `run_20260524_020432/.../qwen_mix20/final` (§2) |
| `qwen_finetuned_mix50` | `run_20260527_185416/.../qwen_mix50/final` (§3) |
| `cute_llama_p` | none (fp16 base-LM few-shot) |

Emits `results/reports/qualitative_examples.{json,md}`.

**Slurm 2787 (pulled).** OOM fix worked; log shows 4 variants only
(label `qwen_finetuned` = Mix-20). Mean sentence chrF: mix20 EN→UG
12.25 / UG→EN 17.81. **Re-run** after rsync to add `qwen_finetuned_mix50`
(5-variant table). Incremental option if you want Mix-50 only:
`--variants qwen_finetuned_mix50 --out-json results/reports/qualitative_mix50.json`
(then merge manually or re-run full default).

**Cost:** ~20–25 min wall (12 generations per FT variant; cute_llama_p
~30 s/sentence fp16). Use **1:30:00** walltime.

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/

ssh ju-compute-server "cd ~/uyghurGPT && mkdir -p results results/reports && sbatch \
  --job-name=qualitative --time=01:30:00 --ntasks=1 --cpus-per-task=8 \
  --mem=24G --gres=gpu:1 --partition=priority --requeue \
  --output=results/slurm_qualitative_%j.out \
  --wrap='cd \$HOME/uyghurGPT && set -a && source .env && set +a && \
    export HF_HOME=\$HOME/uyghurGPT/hf_cache && \
    export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN && \
    export CUDA_VISIBLE_DEVICES=0 && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    export PYTHONUNBUFFERED=1 && \
    \$HOME/micromamba/envs/uyghur_env/bin/python -u scripts/qualitative_examples.py'"
```

Note: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a second
belt-and-braces fragmentation guard (other jobs use
`backend:cudaMallocAsync`; here the warmup wants contiguous space so
the expandable-segments allocator is the safer pick).

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

After the two remaining pulls (`2771` + the
qualitative re-submission; `debug_wcm` already analysed above):

1. **§1 + §3 log updates** (`PROJECT_RESULTS.md`, ~30 min). Include
   the WCM real-but-majority-biased verdict as a caveat under the
   81 % cell; quote the 29.55 % non-majority recall identity vs
   zero-shot.
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
