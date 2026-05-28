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

## Verification — WCM Mix-50 audit (debug_wcm.py) — **Pulled, verdict: real-but-majority-biased**

Slurm 2785 (`results/debug/wcm_mix50_vs_zs.json`, log
`results/slurm_debug_wcm_2785.out`). Mix-50 falls in the **middle** row
of the decision table: not collapse, but driven by the label-1 prior.

| Metric | `qwen_finetuned` (Mix-50) | `qwen_zeroshot` |
|--------|---------------------------|-----------------|
| Accuracy | 0.810 (243/300) | 0.063 (19/300) |
| Always-predict-majority floor | 0.853 | 0.853 |
| `pred_dist` (top) | `1: 249, 4: 44, …` | `9: 164, 4: 94, …` |
| `majority_class_share_pred` | **0.830** | 0.023 |
| `majority_class_collapse_detected` | **False** (threshold ≥ 0.95) | False |
| Acc on majority rows (label 1) | 0.898 | 0.023 |
| Acc on non-majority rows | **0.296** | **0.296** |
| Label-1 F1 | 0.911 (P 0.924, R 0.898) | 0.046 |
| Label-4 F1 | 0.406 (P 0.296, R 0.650) | 0.070 |
| Other labels (0/3/6/9) F1 | **0 / None** (≤ 3 TP combined) | mostly 0 |
| Mean top-1 − top-2 log-prob margin | 0.29 (low confidence) | 1.04 (confidently wrong) |

**Reading.** §3 cell stays — 81 % is real, not collapse — but the report
must caveat that **non-majority recall is unchanged from zero-shot**
(both at 13/44 = 29.55 %). The +75 pp lift comes entirely from
"default to label 1 instead of guessing label 9". Mix-50 has learned
the prior + the **label-4** sub-distinction (F1 = 0.41), and nothing
else. Label-1 + label-4 cover 276/300 = 92 % of the gold support, so
two-class competence ≈ overall accuracy.

Zero-shot is the mirror image: it almost never picks label 1 (recall
2.3 %) but over-fires label 9 (164 predictions for 6 true rows,
recall 83.3 %). Two completely different failure modes converge on
the same 29.55 % non-majority accuracy — coincidence, not signal.

**For the write-up.** Quote both lines in `PROJECT_RESULTS.md` §3
caveats and Task 05 §8 (negative results): "Mix-50's WCM headline is
mostly the label-1 prior; non-majority recall is identical to
zero-shot." This is honest and turns a suspicious cell into a clean
finding.

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
