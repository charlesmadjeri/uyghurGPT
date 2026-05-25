# Task 06 — Final project report

> **Status:** not started.
> **Depends on:** Tasks 01–05. Task 04's `consolidated_results.md` is
> the single source of numerical truth; Task 05's
> `05_results_analysis.md` is the single source of prose claims.
> **Blocks:** project submission.
> **Estimated wall-clock:** 2–3 days of writing + 1 day of integrated
> editing / diagram polish.

## Goal

Produce the final report at `docs/FINAL_REPORT.md` (or
`docs/FINAL_REPORT.pdf` if the course requires PDF — confirm format
before the final pass). The report tells the complete story: research
question → method → experiments → results → analysis → conclusions.
It must stand alone — a reader who has not seen any other file should
be able to follow it end to end.

## Deliverables

1. `docs/FINAL_REPORT.md` with the structure below. Sections may be
   reordered to match the course template if one is provided; the
   *content* obligations are non-negotiable.
2. `docs/figures/` containing the figures referenced in the report
   (training curve from TensorBoard export, EN↔UG asymmetry bar chart,
   delta-over-zero-shot bar chart, pipeline diagram). Use the
   matplotlib + pandas stack that is already in `requirements.txt`.
3. (Optional, if the course requires PDF) `docs/FINAL_REPORT.pdf`
   produced via `pandoc` from the Markdown source. Pinned command
   committed in a short `scripts/build_report.sh`.
4. README.md gains a "Final report" section linking to the report.

## Report structure (fixed)

1. **Title + authors + course + date** (1 line each).
2. **Abstract** (~200 words). State the research question, the method
   (QLoRA Mix-20 on Qwen2.5-7B-Instruct), the baselines (zero-shot
   Qwen, zero-shot LLaMA, CUTE-Llama-P), the headline finding, the
   single most important caveat.
3. **Introduction** (~1 page). Why Uyghur, why now. Why parallel
   data, why CUTE-P. The research question in one paragraph: "Can a
   modern multilingual LLM fine-tuned with LoRA on EN↔UG instruction
   data match or surpass a model trained with full continued
   pretraining and vocabulary expansion (CUTE-Llama-P) on the same
   corpus?" (verbatim from `PROJECT.md`).
4. **Related work** (~1 page). Lift from `docs/02_related_work.md`;
   focus on (i) CUTE / CUTE-Llama-P (Zhuang & Sun, 2025), (ii) QLoRA
   (Dettmers et al., 2023), (iii) the broader low-resource MT
   landscape (Costa-jussà et al., 2022 — NLLB / FLORES-200; Joshi et
   al., 2020 — language inclusion taxonomy).
5. **Method**. Subsections:
   - 5.1 Data — CUTE-P EN+UG subset (~934K pairs); pair-level
     train/test split (5 %); bidirectional expansion; FLAN mix
     (Mix-20). Lift from `docs/03_planned_approach.md` §1.
   - 5.2 Model — Qwen2.5-7B-Instruct (primary), no vocabulary surgery,
     justified by the Day-1 token/byte ratio of 0.396. Lift from
     `docs/03_planned_approach.md` §2.
   - 5.3 Training — QLoRA NF4 + bf16 adapters, rank 16, paged AdamW
     8-bit, assistant-only loss, in-loop early stopping. Lift from
     `PROJECT.md` §Training Configuration.
   - 5.4 Baselines — zero-shot Qwen2.5, zero-shot LLaMA-3.1,
     CUTE-Llama-P (3-shot continuation, *protocol-difference disclaimer
     here*).
   - 5.5 Evaluation — FLORES+ devtest (EN↔UG chrF + BLEU), WCM-v2
     Uyghur classification accuracy, C4 English perplexity. State that
     in-loop `eval_loss` is the overfit detector, not a reported
     number.
6. **Experiments** (1 page). One subsection per experiment, mapping
   1-to-1 to the experiment dirs:
   - 6.1 Experiment 0 — zero-shot baselines.
   - 6.2 Experiment 1 — Qwen Mix-20 QLoRA fine-tune.
   - 6.3 Experiment 2 — CUTE-Llama-P few-shot baseline.
   - (6.4+ if any bonus experiments completed — see §Optional bonus.)
   Each subsection: dataset, hyperparameters, wall-clock, hardware
   (Slurm + 24 GB MIG slice). The numbers themselves go in §Results.
7. **Results** (1–2 pages). Lead with the Task-04 consolidated table.
   Then the per-direction asymmetry chart (Task 05's analysis backs
   this), the delta-over-zero-shot chart, the catastrophic-forgetting
   table (Δ C4 PPL vs Δ FLORES). Quote 1–2 qualitative examples from
   `results/reports/qualitative_examples.md`; the full set goes in an
   appendix.
8. **Analysis** (1–2 pages). Lift the eight-section structure from
   `docs/05_results_analysis.md`. Do not duplicate numbers — refer
   back to §Results.
9. **Limitations** (~½ page).
10. **Conclusions** (~½ page). Restate the research question and
    answer it in two sentences. Identify the most defensible
    contribution (the one with the cleanest evidence).
11. **References** (BibTeX-style; pull from `docs/02_related_work.md`
    and `docs/papers/`).
12. **Appendix A — full qualitative examples** (`results/reports/
    qualitative_examples.md` verbatim).
13. **Appendix B — training curve** (TensorBoard screenshot or
    matplotlib export of `train/loss` + `eval/loss` from the chosen
    run's `logs/`).
14. **Appendix C — reproducibility checklist**: seeds, run ids, Slurm
    job ids, code commit hashes, dataset versions / hashes.

## Implementation plan

### Step 1 — outline pass (½ day)

Draft the entire report at section-heading + one-sentence-per-bullet
granularity. **Do not write prose yet.** This pass exists to catch
missing data: if a section needs a number, write the cell coordinate
from `consolidated_results.md` as the bullet (e.g. "EN→UG chrF: Q-FT
vs Q-ZS Δ = +X.X"). Anywhere the bullet has no number to point to,
the report cannot be written yet and that gap goes back to the
upstream task.

### Step 2 — prose pass on §§1–6 (1 day)

Write Abstract, Introduction, Related Work, Method, Experiments in
order. These sections are mostly lifts from existing docs (clearly
sourced above), so the work is editorial — paragraphing,
abbreviation introductions, citation insertion.

### Step 3 — figures (½ day)

Three figures, one each:

- `docs/figures/pipeline.png` — adapt the ASCII pipeline diagram from
  `docs/03_planned_approach.md` §Slide 5 into a polished figure.
- `docs/figures/asymmetry.png` — grouped bar chart, x-axis
  `{qwen_zs, llama_zs, qwen_ft, cute_llama_p}`, y-axis chrF, two bars
  per group (EN→UG, UG→EN).
- `docs/figures/delta_vs_zeroshot.png` — single bar chart, x-axis
  `{qwen_ft, cute_llama_p}`, y-axis "Δ chrF over zero-shot Qwen", two
  bars per group (EN→UG, UG→EN).

All three loaded from `results/reports/consolidated_results.json` in
matplotlib — no hand-edited numbers.

### Step 4 — prose pass on §§7–10 (1 day)

Results, Analysis, Limitations, Conclusions. These lean heavily on
Task 05's `docs/05_results_analysis.md`; the report's role is to
*present* the analysis, not redo it.

### Step 5 — integration pass (½ day)

End-to-end read-through; check every cross-reference resolves, every
abbreviation introduced before use, every figure / table captioned,
every claim citation-backed.

### Step 6 — build PDF if required (optional)

```bash
pandoc docs/FINAL_REPORT.md \
  --pdf-engine=xelatex \
  --citeproc \
  --bibliography docs/references.bib \
  -o docs/FINAL_REPORT.pdf
```

(Decide between BibTeX management and inline references upfront; do
not mix.)

## Optional — incorporating bonus results

If any bonus task in `docs/tasks/bonus/` completes before the report,
add a corresponding subsection in §Experiments + an extra row to the
Task-04 table. Do **not** include bonus results unless the bonus
experiment is fully run, evaluated, and analyzed — half-finished
ablation cells in the report are worse than no ablation.

## Validation / success criteria

1. `docs/FINAL_REPORT.md` exists, covers all 14 sections, and every
   numeric claim cites a cell in `consolidated_results.md`.
2. Every figure renders from the JSON in `results/reports/` — no
   hand-edited values in matplotlib scripts.
3. PASS/PARTIAL/FAIL row of the Success-criteria table is present
   verbatim (so the examiner can find it without searching).
4. README.md links to the final report (and to the PDF if built).
5. `pytest tests/` still passes; if the report build introduced a
   helper script, that script's smoke test passes too.

## References

- The whole task hierarchy this report consumes:
  `docs/tasks/01_…` through `docs/tasks/05_…`.
- Course rubric — confirm word count / page count / required
  sections against the course instructions PDF the team has;
  override the structure above only when the rubric demands a
  different shape.
- Prior project artifacts the report lifts from: `PROJECT.md`,
  `PROJECT_REFINEMENT.md`, `02_related_work.md`,
  `03_planned_approach.md`, `RESEARCH.md`.
