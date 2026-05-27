# Task 05 — Results analysis & qualitative examples

> **Status:** not started.
> **Depends on:** Tasks 01–02 (done), 04 (consolidated table). Task 03
> decoding fixes are done; post-`repetition_penalty` FLORES re-eval
> landed (Slurm 2768: UG→EN 9.385 → 16.8079). Optional zero-shot
> sanity-gate re-run is queued in `TODO.md` but not blocking.
> Mechanism narrative + pre/post table: `PROJECT_REFINEMENT.md` §14;
> diagnostic log: `results/debug/slurm_ug2en_2766.out`.
> **Blocks:** Task 06 (final report).
> **Estimated wall-clock:** 1 day (writing + a few short adapter-loaded
> inference batches for the qualitative examples).

## Goal

Convert the consolidated numbers into the narrative that will fill the
"Results & Analysis" section of the final report. The course grades on
*systematic thinking*, so this task is where the project's claims —
about LoRA vs. continued pretraining, about EN↔UG asymmetry, about
catastrophic forgetting — get explicitly defended (or refuted) against
the numbers.

## Deliverables

1. `docs/05_results_analysis.md` (NOT this file — the analysis output;
   this `tasks/` file is the task spec). The structure is fixed:

   1. **Headline result** (one paragraph; the single claim the report
      makes, e.g. "LoRA fine-tuning of Qwen2.5-7B on CUTE-P matches
      CUTE-Llama-P on EN→UG chrF while using ~1 % of its trainable
      parameters and no vocabulary surgery").
   2. **Success-criteria check** (table, three rows: minimum / target /
      stretch — verbatim from `PROJECT.md` §Success criteria, with
      PASS / PARTIAL / FAIL and one-line justification each).
   3. **Delta-over-zero-shot per benchmark** (Δ chrF, Δ BLEU, Δ acc, Δ
      PPL — the *causal* contribution of LoRA training, as motivated by
      `docs/03_planned_approach.md` Slide 6).
   4. **EN↔UG asymmetry analysis** (1–2 paragraphs; this is the
      pre-empted question per `PROJECT.md` §Expected Result Asymmetry
      and §Slide 7).
   5. **Catastrophic-forgetting analysis** (Δ C4 PPL vs Δ FLORES EN→UG —
      the Mix-20 buffer's effectiveness).
   6. **Qualitative examples** — exactly **5 sentences per direction**
      (EN→UG and UG→EN), each showing all four variants' outputs side
      by side + the FLORES reference. Sentences are picked
      deterministically by FLORES `id` (record the ids so the table is
      reproducible).
   7. **Limitations** (CUTE-P EN side is auto-translated from Chinese;
      no human eval; eval-only baseline for CUTE-Llama-P with
      protocol-difference disclaimer; single-seed run; absolute chrF
      depressed by translationese — see
      `03_planned_approach.md` §1 caveat).
   8. **Negative / surprising results** (direction asymmetry restored
      but compressed vs zero-shot — Slurm 2768 UG→EN 16.81 vs zero-shot
      30.10; Slurm 2766 failure-mode split (B′ collapse + B″
      hallucinations); decoding fix recovered ~7.4 chrF, training-side
      residual −13.29 chrF; LLaMA-3.1 near-zero UG capability; WCM
      below random for both zero-shot variants).

2. `scripts/qualitative_examples.py` — small script that, given a list
   of FLORES `id`s and the run-id-per-variant mapping from Task 04,
   loads each model once and emits the 4-variant × 5-example × 2-direction
   table as `results/reports/qualitative_examples.md`. Idempotent.

3. The qualitative examples markdown table (deliverable 2's output)
   committed under `results/reports/qualitative_examples.md` so the
   final report can `\include` / copy-paste it.

## Implementation plan

### Step 1 — pick the qualitative examples

Pick 5 EN sentences from FLORES `devtest` covering a spread of:
- short (≤ 12 words) declarative
- short interrogative
- medium-length (~25 words) with a proper noun
- medium-length with technical / scientific vocabulary
- long (~50 words) discursive sentence

Record their FLORES `id`s in the script as constants. Use the same 5
EN sentences (and their aligned UG references) for both directions —
that way the EN→UG row and the UG→EN row of the table sit in 1-to-1
correspondence, which lets the reader compare per-sentence asymmetry
directly.

### Step 2 — run inference once per variant

Reuse `shared/evaluation.py::load_eval_model` and
`generate_translation` / `generate_translation_fewshot` (post-Task 01).
The script loops over the four variants, loads each, generates the 10
outputs (5 sentences × 2 directions), and unloads before loading the
next variant — same pattern as `shared/evaluation.py::run_eval`.

Output format:

```markdown
### Example 1 — FLORES id 42

EN (source): "..."
UG (reference): "..."

| Variant            | EN→UG output                            | UG→EN output                            |
|--------------------|-----------------------------------------|-----------------------------------------|
| qwen_zeroshot      | ...                                     | ...                                     |
| llama_zeroshot     | ...                                     | ...                                     |
| **qwen_finetuned** | ...                                     | ...                                     |
| cute_llama_p       | ...                                     | ...                                     |
```

### Step 3 — write the analysis

Open `docs/05_results_analysis.md` with the structure from
"Deliverables" §1. Each subsection must:

- cite an exact cell or delta from `results/reports/consolidated_results.md`
  (Task 04 output) by quoting the value — no hand-edited numbers, ever
- prefer comparing **deltas vs zero-shot Qwen** over absolute values
  (this is the framing called out in `03_planned_approach.md` Slide 6)
- avoid claims the data does not support — if Δ chrF EN→UG is +4.2 and
  CUTE-Llama-P is +1.5 chrF below us on the same direction, the
  headline is "matches CUTE-Llama-P on EN→UG", not "beats SOTA"
- when reporting CUTE-Llama-P, repeat the prompt-style disclaimer
  ("evaluated with 3-shot continuation prompting because CUTE-Llama-P
  is a base LM, not instruct; the chat-template models had no
  exemplars in their prompt — protocol difference noted").

### Step 4 — defend the pre-registered success criteria

Lift the table from `PROJECT.md` §Success criteria verbatim, then
fill PASS / PARTIAL / FAIL:

| Tier | Criterion | Outcome | One-line justification |
|------|-----------|---------|------------------------|
| Minimum | FT Qwen beats ZS Qwen on FLORES chrF in ≥ 1 direction | PASS/FAIL | "EN→UG: 9.96 → X (Δ +Y); UG→EN: 30.29 → X (Δ +/−Y)" |
| Target  | FT Qwen within 2 chrF of CUTE-Llama-P on EN→UG **and** beats it on UG→EN | PASS/FAIL | quote both cells |
| Stretch | Ablation reveals statistically clear Mix sweet spot; LLaMA FT done | PARTIAL/FAIL | depends on bonus tasks |

This is the part the examiner will look for first; do not bury it.

### Step 5 — limitations + honesty

A short, bulleted limitations subsection covering:

- CUTE-P EN side is machine-translated from Chinese
  (`03_planned_approach.md` §1 caveat) — depresses absolute FLORES
  chrF / BLEU; argue from deltas, not absolutes.
- Single seed (42); no variance estimate. Discuss in one sentence.
- CUTE-Llama-P protocol difference (few-shot continuation vs. chat
  template). Already mentioned everywhere; restate here.
- No human evaluation of translation quality.
- EN→UG BLEU stays near zero across all systems because chrF is the
  right metric at this resource level; we report BLEU for completeness
  only.

## Validation / success criteria

1. `docs/05_results_analysis.md` exists, follows the 8-section
   structure exactly, and **every numeric claim** in the prose links to
   a cell in `results/reports/consolidated_results.md`.
2. `results/reports/qualitative_examples.md` exists with 5 examples ×
   2 directions × 4 variants, FLORES ids recorded, all outputs decoded
   without leading/trailing chat markers.
3. The Success-criteria table is filled with PASS/PARTIAL/FAIL plus a
   one-line justification each.
4. `pytest tests/` still passes (this task only touches docs +
   `scripts/qualitative_examples.py`; no test changes required).

## References

- Pre-registered claims: `docs/PROJECT.md` §Research Contributions and
  §Success criteria.
- EN↔UG asymmetry framing: `docs/PROJECT.md` §Expected Result Asymmetry,
  `docs/03_planned_approach.md` Slide 7 anticipated Q.
- Mix-20 / catastrophic-forgetting framing: `docs/PROJECT.md`
  §Data mixing ratio.
- Previous-run analysis we are extending: `docs/PROJECT_RESULTS.md`
  2026-05-24 §Analysis (bullets on the UG→EN regression, BLEU
  pathology, WCM missing).
