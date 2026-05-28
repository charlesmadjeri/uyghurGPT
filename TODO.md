# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

> **Experimentation closed.** All GPU eval / training runs for the
> project are done; only `qualitative_examples` (Slurm 2788) is still
> in flight. See `docs/PROJECT_RESULTS.md` §1 / §2 / §3 for the
> measured numbers. Remaining work is reporting.

## Write-up phase (no more GPU jobs)

All Slurm runs are done and pulled. Remaining is reporting.

1. **Task 05** — `docs/05_results_analysis.md` 8-section write-up.
   Reuse:
   - §2 + §3 core / bonus tables verbatim.
   - §1 Slurm 2770 / 2771 / 2785 / 2788 entries for *why* each row
     says what it says.
   - WCM macro-metric verdict (raw 81 % real but balanced-accuracy
     parity with zero-shot; macro F1 win is real).
   - §14 UG→EN regression mechanism: greedy collapse partial fix
     (Slurm 2768) + training-side residual + Mix-50 retraining lift
     (+1.16 chrF).
   - Qualitative table (`results/reports/qualitative_examples.md`,
     5 variants × 5 sentences × 2 directions) for Task 05 §6 — copy
     into `docs/qualitative_examples.md` or `git add -f` if the
     markdown table needs to ship with the report (`results/` is
     gitignored).
2. **Task 06** — final report / slides.
3. **(Optional) Task 04** — `scripts/aggregate_results.py`. §2 already
   serves as the canonical table; skippable.
4. **(Optional)** Backfill the §2 macro F1 / macro recall column for
   `qwen_finetuned` Mix-20, `llama_zeroshot`, and `cute_llama_p` by
   running `scripts/debug_wcm.py --no-adapter` (and with each
   adapter) — per-prediction artifacts already exist; ~30 min total
   on a free MIG, or skip and quote only the Mix-50 vs `qwen_zs` pair
   from Slurm 2785.

---

## Deferred (do not run unless write-up demands a number)

- **A1 beams** — code-only, default off. §2 would require parallel
  re-run of all chat-path variants; not enabled.
- **B1 + B2 retrain** — Mix-50 sits at UG→EN 17.97 (just under the
  "≥ 18" line); B1+B2 was the next training fix if we kept going.
- **Mix-0 / Mix-10 bracket** — `docs/tasks/bonus/02_qwen_mix_ablation.md`.
- **A2 chat-fewshot diagnostic** — not needed for §2; useful only as
  side-evidence in the report if there's time.
- **200 k / 300 k pair count** — Mix-20 early-stopped at 1.48 epochs;
  quantity not the binding constraint.

---

## Done (remove when read)

- ~~Slurm 2788~~ — qualitative 5-variant pulled
  (`results/reports/qualitative_examples.{json,md}`). Per-variant
  mean chrF: `qwen_zs` 6.92 / 29.95; `llama_zs` 0.47 / 19.60;
  Mix-20 12.25 / 17.81; **Mix-50 12.59 / 21.38**; `cute_llama_p`
  8.61 / 28.10. Table in `PROJECT_RESULTS.md` §1 (2026-05-28 entry).
- ~~Slurm 2787~~ — superseded by 2788 (was Mix-20 only).
- ~~Slurm 2786~~ — qualitative OOM, fixed by load-order change
  (commit `f3586b3`).
- ~~Slurm 2785 (`debug_wcm`)~~ — Mix-50 WCM audit + macro metrics in
  `PROJECT_RESULTS.md` §1 and §3 footnote. **Verdict:**
  balanced acc 0.258 vs zero-shot 0.271 (parity); macro F1 0.220 vs
  0.103 (×2.1 win); raw 81 % real but anchored to majority prior.
- ~~Slurm 2771~~ — exp-0 rep-penalty zero-shot sanity gate.
  `qwen_zs` UG→EN 30.10 → 29.56 (gate pass), `llama_zs` UG→EN
  **4.71 → 15.96** (+11.25 chrF, same repetition-collapse pathology
  as `qwen_ft` pre-fix). §2 zero-shot UG→EN cells updated.
- ~~Mix-50 retrain~~ — `run_20260527_185416`, Slurm 2770;
  UG→EN 16.81 → 17.97 (+1.16), EN→UG byte-stable, PPL −0.25.
  Numbers in §3 bonus table; macro-WCM audit done by Slurm 2785.
- ~~Slurm 2768 `qwen_finetuned` UG→EN re-eval~~ — §2 UG→EN
  **16.8079** (rep-penalty UG→EN only).
- ~~Slurm 2766 `debug_ug2en`~~ — mechanism in
  `PROJECT_REFINEMENT.md` §14.
- ~~Training-data audit~~ — balanced `ug2en`/`en2ug`.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 (Mix-20)~~ — Slurm 2750 / 2749.
- ~~A1/A2 implementation (code)~~ — commit `9b6141d`; A1 eval **not** run.
