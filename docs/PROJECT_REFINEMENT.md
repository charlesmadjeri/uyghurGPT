# Project Refinement Log

> Date: May 2026
> Context: Design presentation planning session — review of PROJECT.md and RESEARCH.md
>          against course requirements and practical constraints.

This document records every decision that was changed from the original project plan,
with the rationale for each change. It serves as a transparency log for the course
examiner and as a reference for the team.

---

## 1. Core vs. Stretch Goal Restructuring

### What changed
The original plan treated the full ablation (Mix-{0,10,20,50}) × 2 models (Qwen2.5, Llama-3.1),
CUTE-Llama-P baseline reproduction, and MiLiC-Eval (9 tasks) as first-class deliverables.

The revised plan defines a strict two-tier structure:
- **Core experiment** (must complete): Qwen2.5-7B Mix-20 fine-tuned, evaluated on
  FLORES-200 and WCM-v2, compared against zero-shot Qwen2.5 and zero-shot Llama-3.1.
- **Stretch goals** (if time permits): Llama-3.1 fine-tune, CUTE-Llama-P baseline,
  ablation variants, MiLiC-Eval.

### Why
The original scope was too large for the compute constraints (MIG 1g.10gb per job).
An 8-cell ablation × 2 models serial on a single worker exceeds the 5-day priority
partition limit. More importantly, course examiners reward a clean, complete, well-analysed
core experiment over a partially executed grand plan. Presenting an incomplete ablation
at the design stage creates expectations the project may not meet.

---

## 2. CUTE-Llama-P Demoted to Stretch Baseline

### What changed
The original plan set CUTE-Llama-P as the **primary baseline**, to be reproduced
by running inference on FLORES-200 EN↔UG ourselves.

The revised plan demotes it to a **stretch baseline** with an explicit fallback:
if loading the model fails or takes more than 2 days of engineering time, it is dropped
from the comparison entirely.

### Why
CUTE-Llama-P uses an expanded vocabulary and a custom tokenizer. Loading it with
standard HuggingFace `AutoModel` is not guaranteed. The paper only published
ZH→UG numbers; EN↔UG inference requires verifying the model produces coherent output
in a direction never evaluated by the authors. The direction mismatch already limits
the comparison's scientific value. The zero-shot Qwen2.5 and Llama-3.1 baselines are
sufficient to isolate the contribution of LoRA fine-tuning, which is the core
research question.

---

## 3. Day-1 Sanity Checks Added as Mandatory Gate

### What changed
The original plan had no explicit pre-training checks. The revised plan adds three
mandatory sanity checks that must pass before any training job is submitted:
1. Tokenizer Uyghur segmentation test (token/byte ratio).
2. QLoRA memory test (forward+backward on MIG slice).
3. CUTE-Llama-P load test (determines feasibility of stretch baseline).

### Why
Two risks were identified that could invalidate the entire experimental design
without being caught until late:

- **Tokenizer fragmentation risk**: Both Qwen2.5 and Llama-3.1 were trained on
  Latin/CJK-heavy corpora. Uyghur uses Arabic script. If the tokenizer fragments
  Uyghur into byte-level tokens (token/byte ratio >> English ratio), the "no vocabulary
  surgery" design decision is flawed and must be revisited before wasting compute
  on a broken fine-tune.

- **Memory risk**: QLoRA on a MIG 1g.10gb slice is expected to fit (~6–9 GB),
  but this depends on sequence length, batch size, and gradient checkpointing
  configuration. A failed memory test on day 1 is far better than a failed training
  job on day 3.

Catching either issue on day 1 allows a week-1 fix. Catching it on week 2 or 3
would derail the timeline.

---

## 4. MiLiC-Eval Deferred to Final Report (Stretch)

### What changed
MiLiC-Eval (9-task bilingual benchmark) was listed as a primary evaluation benchmark
in the original plan. It is now a stretch goal, deferred to the final report if at all.

### Why
The design presentation evaluation plan should be simple, concrete, and completable
within the project timeline. FLORES-200 + WCM-v2 already covers translation quality
and downstream Uyghur task performance — the two most important axes.
MiLiC-Eval adds coverage breadth but not depth, and its 9-task structure makes
result reporting and analysis significantly more complex for marginal additional insight
at the course level.

---

## 5. Translation Direction Asymmetry Explicitly Documented

### What changed
The original plan treated EN→UG and UG→EN symmetrically, with no mention of
the expected performance gap between the two directions.

The revised plan explicitly notes: EN→UG generation requires the model to *produce*
Uyghur fluently (much harder), while UG→EN only requires understanding Uyghur input
and producing English. EN→UG scores will likely be substantially lower.
This asymmetry is now flagged in the evaluation plan and will be discussed in the
presentation and report.

### Why
Not flagging this would make the EN→UG results look like a failure rather than
an expected characteristic of generation in a low-resource target language.
Presenting it proactively shows methodological awareness and pre-empts a likely
examiner question.

---

## 6. Pre-registered Success Criteria Added

### What changed
The original plan had no pre-defined success criteria. The revised plan adds three
explicitly tiered, numeric success criteria (minimum / target / stretch), defined
before any results are seen.

### Why
Pre-registration is a standard scientific practice that prevents post-hoc
re-framing of results. For a course project, it also demonstrates systematic
thinking — a graded criterion in the course. The numeric thresholds (e.g.,
"≥5 chrF++ improvement", "within 2 chrF++ of CUTE-Llama-P") are set based on
what is known from the literature about LoRA fine-tuning gains on low-resource tasks,
not pulled from results.

---

## 7. "No Vocabulary Surgery" Framing Made Conditional

### What changed
The original plan stated as a fixed design decision: "No vocabulary surgery on
Qwen / Llama — both already handle Uyghur Arabic script in their native tokenizer."

The revised plan keeps this as the default and intended approach, but conditions it
on the tokenizer sanity check (item 3 above). If the check fails, a minimal vocabulary
extension (top-5K Uyghur unigrams) will be added and the change documented.

### Why
Making an absolute claim about tokenizer quality without verifying it first is
methodologically weak. The conditional framing is more honest and more defensible
in the design presentation Q&A.

---

## 8. Ablation Scope Reduced and Re-prioritised

### What changed
The original ablation: Mix-{0, 10, 20, 50} × {Qwen2.5, Llama-3.1} = 8 fine-tunes.

The revised ablation: Mix-{0, 10, 20, 50} × {Qwen2.5 only} = 4 fine-tunes (stretch).
Further reduced to Mix-{0, 20} only if time is short (2 fine-tunes).

### Why
Running the ablation on both models doubles the compute without adding scientific clarity.
The ablation's research question is about data mixing ratio, not about model architecture.
Fixing the model to Qwen2.5 (the primary model) keeps the experimental variable clean.
Running all 8 cells on 7 workers is theoretically fast (~1.5 days parallelised), but
this assumes all jobs start immediately, no job fails, and evaluation runs smoothly —
all optimistic assumptions for a first-time Slurm user on a shared cluster.

---

## Summary Table

| # | Change | Direction | Primary Reason |
|---|--------|-----------|----------------|
| 1 | Core/stretch split introduced | Scope reduction | Compute constraints + examiner expectations |
| 2 | CUTE-Llama-P demoted to stretch | Risk reduction | Engineering risk + direction mismatch |
| 3 | Day-1 sanity checks made mandatory | Risk mitigation | Tokenizer + memory failure modes |
| 4 | MiLiC-Eval deferred | Scope reduction | Complexity vs. value tradeoff |
| 5 | Direction asymmetry documented | Clarity | Pre-empts misinterpretation of results |
| 6 | Pre-registered success criteria added | Scientific rigour | Standard practice + course grading criteria |
| 7 | "No vocab surgery" made conditional | Honesty | Unverified claim removed |
| 8 | Ablation scope reduced | Scope reduction | Compute constraints + experimental clarity |

---

*No changes were made to the core research question, the choice of Qwen2.5-7B as the
primary model, the use of CUTE-P as the training corpus, or the FLORES-200 + WCM-v2
evaluation benchmarks. These remain as originally planned.*
