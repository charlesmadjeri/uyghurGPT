# Design Presentation — Task Plan

> Presentation format: 10 min talk + 5 min Q&A
> Deliverables: 4 documentation files + 1 slide deck (~10 slides)

---

## Deliverable Overview

| File | Content | Owner |
|------|---------|-------|
| `docs/01_problem_description.md` | What problem, why it matters, DL task definition | Eric |
| `docs/02_related_work.md` | Prior work, comparison table, gap statement | Charles |
| `docs/03_planned_approach.md` | Data pipeline, model, training config, risks | Charles |
| `docs/04_planned_evaluation.md` | Benchmarks, metrics, success criteria | Eric |
| Slide deck (~10 slides) | Built directly from slide notes in each task | Charles & Eric |

**Slide time budget:** Problem 2 min · Related Work 2 min · Approach 3 min · Evaluation 3 min

---

## Task 1 — `docs/01_problem_description.md`

### Task Objective
Define what problem we are solving, why it matters, and what the learning task is in
deep-learning terms. By the end of this section, the audience must understand: the
input, the output, the data, the user need, and the gap we are filling.

### Research Section

**Literature required (cite in report AND on slides):**
- Zhuang & Sun, COLING 2025 (CUTE paper) — the dataset we use and the gap we extend.
  Read: abstract, introduction, Section 3 (dataset construction).
- Haddow et al. 2022, *Survey of Low-Resource Machine Translation*, Computational
  Linguistics — justifies "low-resource" framing and the data scarcity problem.
  Read: abstract + Section 1 (introduction) is sufficient for one citation.
- One Uyghur NLP / morphology reference — e.g. Tursun & Cakici 2017 or any recent
  paper mentioning Uyghur tokenization challenges.
  Read: abstract only; cite for the "agglutinative morphology" claim.

**Learn-only (no citation needed — watch/read for background understanding):**
- What instruction fine-tuning is: HuggingFace blog post
  "Instruction-tuning LLMs" (~15 min read). No citation needed.
- LoRA at a high level: the original LoRA paper (Hu et al. 2021) figure and
  abstract only (~10 min). The full method is covered in Task 3.

### What to Write

1. **Real-world problem (1 paragraph).** Uyghur has ~12M speakers but receives
   minimal support from current LLMs. Frame the user need: a bilingual
   English↔Uyghur assistant for translation and instruction-following tasks.

2. **Why it is hard (bullet list, 4–5 points).**
   - Agglutinative morphology: many suffixes per word, very different from English.
   - Arabic script, written right-to-left.
   - Scarce parallel corpora compared to major languages.
   - Tokenizers trained on Latin/CJK-heavy data segment Uyghur poorly.
   - Limited published baselines for the EN↔UG direction specifically.

3. **The deep learning task (formal definition).**
   Supervised instruction fine-tuning of a causal language model on EN↔UG instruction
   pairs. Input: prompt (instruction + source sentence). Output: target sequence.
   Loss: standard next-token cross-entropy over response tokens only.

4. **Scope statement (what we do and do not do).**
   EN↔UG only; 7–8B models; QLoRA/LoRA only; CUTE-P as primary data.
   No ZH↔UG, no NLLB fine-tuning, no models under 7B.

5. **Research question (1 sentence, bold).**
   Can parameter-efficient LoRA fine-tuning on a modern multilingual LLM (Qwen2.5-7B)
   match continued pretraining + vocabulary expansion (CUTE-Llama-P) on EN↔UG
   translation and Uyghur classification, at a fraction of the compute?

### Slide Notes

**Slide 1 — "The Problem" (~1 min)**
- Visual: one Uyghur sentence in Arabic script + its English translation.
- Key stat: "~12M Uyghur speakers — <1% of typical LLM pre-training tokens."
- Bullet: Arabic script, agglutinative, scarce data, poor tokenizer support.

**Slide 2 — "The DL Task and Research Question" (~1 min)**
- Diagram: [Prompt: instruction + source] → [Qwen2.5-7B + LoRA adapter] → [Response].
- One box labelling the loss: "CE loss on response tokens only."
- Research question in bold at the bottom of the slide.

**Estimated effort:** ~0.5 day.

---

## Task 2 — `docs/02_related_work.md`

### Task Objective
Position the project in the existing literature. Show we know what has been done,
what works, and where the gap is. This is the section examiners use to judge whether
the novelty claim is credible.

### Research Section

**Literature required (cite in report; name on slides):**

1. **CUTE paper** — Zhuang & Sun, COLING 2025.
   Read: full paper (already read for Task 1). Summarise: ZH pivot, continued
   pretraining + vocab expansion, ZH→UG focus, no EN↔UG published numbers.

2. **NLLB-200** — Costa-jussà et al. 2022, "No Language Left Behind."
   Read: abstract + Table 1 (Uyghur FLORES scores). Summarise in 2–3 sentences:
   dedicated translation model, strong Uyghur support, but not an instruction-following LLM.

3. **LoRA** — Hu et al. 2021. Read: abstract + Figure 1 (the B·A decomposition).
   Cite for the method; deeper understanding covered in Task 3.

4. **QLoRA** — Dettmers et al. 2023. Read: abstract + Section 2 (4-bit NF4).
   Cite for the quantization approach.

5. **Qwen2.5 technical report** — Qwen Team 2024.
   Read: abstract + multilingual coverage section. Note: does the report explicitly
   mention Uyghur? If yes, quote it. If no, note the absence.

6. **One additional Uyghur / Turkic LLM paper** beyond CUTE — e.g. MC²-LLaMA
   (Li et al. 2024) or a recent multilingual Turkic paper.
   Read: abstract only. Purpose: show we surveyed beyond a single prior work.

**Learn-only (no citation needed):**
- The standard low-resource LLM recipe (pretrain → CPT → SFT → RLHF).
  One blog post or diagram is enough. Used to show where our approach sits
  in the typical pipeline (we skip CPT).

### What to Write

1. **Three subsections:**

   **a. Low-resource MT and Uyghur NLP**
   Cover: NLLB-200 (dedicated MT), CUTE-Llama-P (continued pretraining + vocab),
   one additional Uyghur paper. End with: what none of these do (EN↔UG + LoRA
   on a modern multilingual LLM).

   **b. Parameter-efficient fine-tuning**
   Cover: LoRA (low-rank adapter decomposition), QLoRA (4-bit base + bf16 adapters).
   Emphasise: enables fine-tuning 7B models on ~10 GB VRAM — the enabling technology
   for this project's compute constraints.

   **c. Multilingual LLMs**
   Cover: Qwen2.5-7B (our primary model — native multilingual vocab, Apache 2.0),
   Llama-3.1-8B (secondary — Meta license). Note: both handle Uyghur Arabic script
   natively — this is to be verified by the Day-1 tokenizer check.

2. **Comparison table** (this is the core visual for the slide):

   | Method | Data type | Training cost | Vocab handling | EN↔UG support | Published? |
   |--------|-----------|---------------|----------------|---------------|------------|
   | NLLB-200 | Parallel MT | Full training | Native Uyghur | ✓ | ✓ |
   | CUTE-Llama-P | Parallel (ZH pivot) | Full CPT + vocab expansion | Expanded | ZH→UG only | ✓ |
   | MC²-LLaMA | Mixed | Continued PT | Expanded | Partial | ✓ |
   | **Ours (Qwen2.5+QLoRA)** | **Parallel EN↔UG** | **QLoRA only** | **Native (to verify)** | **✓ both directions** | **This work** |

3. **Gap statement (1 paragraph).**
   No published work combines: (a) a modern Qwen2.5-class multilingual base,
   (b) QLoRA-only tuning (no CPT, no vocab surgery), (c) EN↔UG direction,
   (d) the CUTE-P corpus. Each prior work addresses at most 2 of these 4.

### Slide Notes

**Slide 3 — "Related Work" (~1 min)**
- The comparison table (4 rows). Highlight the "Ours" row.
- One sentence caption: "All prior work misses at least one of: modern base / LoRA-only /
  EN↔UG direction."

**Slide 4 — "The Gap" (~1 min)**
- Simple diagram: 2×2 matrix with axes "Training cost" (low/high) and
  "EN↔UG support" (yes/no). Place NLLB-200, CUTE-Llama-P, and Ours on the matrix.
  Ours sits in the low-cost + EN↔UG quadrant, currently empty.

**Estimated effort:** ~1 day (mostly reading abstracts and intros).

---

## Task 3 — `docs/03_planned_approach.md`

### Task Objective
Specify what we will actually build and train, with enough detail that any classmate
could critique the design choices. Cover: data pipeline, model and training config,
compute plan, and risk mitigations.

This task is owned by the person who will do the engineering.

### Research Section

**Literature required (cite in report; 1 figure on slides):**
- LoRA paper (Hu et al. 2021): Figure 1 (B·A decomposition) — use on the slide.
- QLoRA paper (Dettmers et al. 2023): Table showing memory usage at 4-bit NF4 —
  use to justify "fits in 10 GB." Read Section 2 fully.

**Learn-only (docs / tutorials — understand well enough to implement):**
- HuggingFace `peft` documentation: `LoraConfig` parameters (rank, alpha, target
  modules, task type). Spend ~30 min. You will configure this directly.
- HuggingFace `trl` documentation: `SFTTrainer` and `DataCollatorForCompletionOnlyLM`
  (response masking). Spend ~30 min.
- `bitsandbytes` 4-bit loading: `BitsAndBytesConfig` with `load_in_4bit=True`,
  `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`.
  Spend ~20 min.
- Qwen2.5 chat template: find the Jinja template in the model card and write out
  the exact prompt format for EN→UG and UG→EN. This goes directly into the doc.
- Alpaca/ChatML instruction formatting — understand the concept (15 min), then
  adapt Qwen's native template.

### What to Write

1. **Data pipeline.**
   - Source: CUTE-P EN + UG columns, ~934K pairs, ~10.9 GB.
   - Preprocessing: filter pairs where either side is empty or > 512 tokens
     (estimated < 5% of data). Shuffle with fixed seed.
   - Instruction templating: show the exact Qwen chat template for both directions.
   - Data mix: 80% CUTE-P pairs (alternating EN→UG and UG→EN per batch),
     20% FLAN English-only samples (~50K selected randomly, fixed seed).
   - Split: all CUTE-P for training; FLORES-200 devtest (unseen) for evaluation.
     Hold out 1K CUTE-P pairs as in-domain validation (sanity check during training).

2. **Model and learning algorithm.**
   - Full training config table (copied and explained from PROJECT.md §Training Configuration).
   - Explain each key choice in 1 sentence: why NF4, why rank 16, why cosine schedule,
     why response masking.

3. **Compute plan.**
   - MIG 1g.10gb → QLoRA fits ~6–9 GB.
   - Core fine-tune: ~16–28 h per run.
   - Stretch ablation (4 cells, 1 model): ~4 × 20h = ~80h serial; ~20h parallel
     (4 workers simultaneously).
   - Slurm submission: `--partition priority`, `--gres=gpu:1`, `--time=5-00:00:00`.

4. **Day-1 sanity checks (reference PROJECT.md §Day-1 Sanity Checks).**
   State the three checks and their pass/fail criteria. State the fallback for each.

5. **Risk and mitigation table.**

   | Risk | Probability | Impact | Mitigation |
   |------|------------|--------|------------|
   | Tokenizer fragments Uyghur (byte-level) | Medium | High | Minimal vocab extension (top-5K Uyghur unigrams); document change |
   | QLoRA OOM on MIG 1g.10gb | Low | High | Reduce seq length to 384; reduce batch size |
   | CUTE-Llama-P won't load | High | Low | Drop stretch baseline; zero-shot only |
   | Wall-clock exceeds priority partition | Low | Medium | Submit ablation cells as parallel jobs; reduce to 2 ablation cells |
   | UG→EN >> EN→UG (expected asymmetry) | Certain | Low (expected) | Pre-empt in presentation; frame as expected, not failure |

### Slide Notes

**Slide 5 — "Approach Overview" (~1 min)**
- Pipeline diagram: [CUTE-P EN↔UG] + [FLAN 20%] → [Instruction formatting]
  → [Qwen2.5-7B + QLoRA] → [FLORES-200 / WCM-v2 evaluation].
- Keep visual, minimal text.

**Slide 6 — "Why QLoRA + Qwen2.5" (~1 min)**
- Two side-by-side boxes:
  - *QLoRA*: fits 7B in 10 GB; trains adapters only (~1% of params); no vocab surgery needed.
  - *Qwen2.5-7B*: Apache 2.0; native multilingual vocab; strong zero-shot baseline built-in.
- LoRA B·A figure (small, from the paper).

**Slide 7 — "Compute and Risks" (~1 min)**
- Small table: core run ~20h, 4-cell ablation ~20h parallel.
- Three-row risk list (top 3 risks + mitigations in one bullet each).

**Estimated effort:** ~1.5 days (mostly reading docs + writing the pipeline spec).

---

## Task 4 — `docs/04_planned_evaluation.md`

### Task Objective
Define how we will know the project worked, in a way that is fair, reproducible,
and pre-registered. This section is explicitly graded: the course requires "a systematic
approach to evaluate a solution."

### Research Section

**Literature required (cite in report; name metrics on slides):**
- FLORES-200: Costa-jussà et al. 2022. Read: abstract + dataset card (HuggingFace).
  Note: 1012 sentences, covers both EN and UG, cc-by-sa license.
- chrF: Popović 2015. Read: abstract only. Key claim: character n-gram overlap
  outperforms BLEU for morphologically rich and agglutinative target languages. Cite this.
- BLEU: Papineni et al. 2002. Abstract only. Cite for secondary metric.
- WCM-v2: cite the originating paper or HuggingFace dataset card.
  Read: task description, label set, evaluation metric used in prior work.

**Learn-only (no citation needed):**
- Why chrF is better than BLEU for Uyghur: 5-min read on any chrF tutorial.
  Understand: BLEU counts word n-grams; for agglutinative languages, word boundaries
  are uninformative because one stem can have dozens of suffixed forms.
  chrF counts character n-grams, which capture partial morpheme overlap.
- Paired bootstrap resampling for MT evaluation: sacrebleu `--paired-bs` flag.
  Read the sacrebleu README section on paired testing (~10 min).
  Understand: it resamples test sentence pairs 1000× to estimate whether a difference
  in chrF scores is statistically significant.
- English perplexity as a catastrophic forgetting proxy: load a held-out English
  set (C4 1K sentences), compute `model.eval()` perplexity before and after fine-tuning.
  One blog post on perplexity as LM evaluation is sufficient.

### What to Write

1. **Evaluation matrix** — full table as in PROJECT.md §Evaluation Plan, with all
   models × benchmarks × metrics filled in. Clearly mark stretch rows.

2. **Benchmarks and metrics — one paragraph each:**

   **FLORES-200 devtest (1012 sentences):**
   Primary benchmark for translation. We evaluate EN→UG and UG→EN separately.
   Primary metric: chrF++ (sacrebleu). Secondary: BLEU. Full sacrebleu signature
   reported in all tables. Why chrF: character n-gram overlap is more robust than
   word n-gram overlap for Uyghur's agglutinative morphology.

   **WCM-v2 (Uyghur text classification):**
   Downstream Uyghur language understanding task. Metric: accuracy (primary),
   macro-F1 (secondary). Evaluated zero-shot: the fine-tuned model is prompted
   with the classification task in Uyghur.

   **English perplexity (catastrophic forgetting check):**
   Held-out C4 1K English sentences. Report perplexity before fine-tuning (base model)
   and after. A large increase (> 20% relative) flags catastrophic forgetting.

   **MiLiC-Eval (stretch):**
   9-task bilingual benchmark. Deferred to final report. Not presented at design stage.

3. **Statistical reporting.**
   - Paired bootstrap resampling (sacrebleu `--paired-bs`, n=1000) for all translation comparisons.
   - 95% confidence intervals on all primary metrics.
   - Single training seed (time constraint) — flagged as a limitation.
   - Sacrebleu signature reported in full for all BLEU/chrF scores.

4. **Translation direction asymmetry note.**
   EN→UG requires generating Uyghur fluently (hard). UG→EN requires understanding
   Uyghur input and generating English (easier). Expect a substantial gap between
   the two direction scores. This is not a failure — it is an expected property of
   generation in a low-resource target language. It will be discussed explicitly
   in the presentation and report.

5. **Pre-registered success criteria** — copy from PROJECT.md §Pre-registered Success Criteria
   and expand with the rationale for each threshold:
   - *Minimum*: fine-tuned Qwen2.5 beats zero-shot Qwen2.5 in ≥1 direction.
     Rationale: any fine-tuning benefit at all validates the approach.
   - *Target*: ≥5 chrF++ improvement in both directions + WCM-v2 accuracy gain.
     Rationale: 5 chrF++ is a meaningful gap in low-resource MT literature.
   - *Stretch*: within 2 chrF++ of CUTE-Llama-P on EN→UG and beats it on UG→EN.
     Rationale: would demonstrate LoRA parity with full CPT at ~10× less compute.

6. **What we will NOT claim.**
   - No human evaluation.
   - No production deployment readiness.
   - No generalisation to other low-resource languages.
   - No direct comparison on ZH↔UG (direction mismatch with the CUTE paper).

### Slide Notes

**Slide 8 — "Evaluation Plan" (~1.5 min)**
- The evaluation matrix table (simplified: 4 model rows × 3 benchmark columns).
- Mark core rows bold, stretch rows greyed out.
- One bullet: "Primary metric: chrF++ — preferred for agglutinative target languages."

**Slide 9 — "Success Criteria + Direction Note" (~1 min)**
- Three-level success criteria (minimum / target / stretch) as a visual ladder or
  stacked bar.
- One highlighted warning box: "EN→UG scores will be lower than UG→EN — expected,
  not a failure."

**Slide 10 — "Plan + Q&A" (~0.5 min)**
- Plan: Mark core vs. stretch with colour or icon.
- "Questions?" at the bottom.

**Estimated effort:** ~1 day.

---

## Slide Deck Layout (10 slides, ~1 min each)

| # | Title | Section | Time |
|---|-------|---------|------|
| 1 | Title + team | — | 0:15 |
| 2 | The Problem | Task 1 | 1:00 |
| 3 | The DL Task + Research Question | Task 1 | 1:00 |
| 4 | Related Work (comparison table) | Task 2 | 1:00 |
| 5 | The Gap (2×2 diagram) | Task 2 | 1:00 |
| 6 | Approach Overview (pipeline diagram) | Task 3 | 1:00 |
| 7 | Why QLoRA + Qwen2.5 | Task 3 | 1:00 |
| 8 | Compute and Risks | Task 3 | 1:00 |
| 9 | Evaluation Plan (matrix + success criteria) | Task 4 | 1:30 |
| 10 | Plan + Q&A | Task 4 | 0:15 |

**Rehearsal rule:** if any slide takes > 75 seconds on first rehearsal, cut content — not pace.

**Q&A preparation (5 min):**
Anticipate these questions:
- "Why not just use NLLB-200?" → We target instruction-following, not pure MT.
- "How do you know Qwen handles Uyghur?" → Day-1 tokenizer test; fallback documented.
- "What if CUTE-Llama-P beats you?" → Expected; framed as stretch; zero-shot comparison is the core.
- "Why not train on ZH↔UG?" → EN↔UG is the novel direction; ZH pivot already published.
