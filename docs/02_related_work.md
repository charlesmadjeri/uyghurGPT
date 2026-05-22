# Related Work

> **Owner:** Charles
> **Task:** Task 2 — Design Presentation
> **Estimated effort:** ~1 day
> **Slide coverage:** Slides 3–4 (~2 min)

---

## 2.1 Low-Resource MT and Uyghur NLP

Uyghur is a Turkic language spoken by approximately 12 million people, written in
Arabic script (right-to-left), and characterised by agglutinative morphology — a
single Uyghur word such as ئۆيلىرىمىزدىن (*öylirimizdın*, "from our houses") can
encode stem, possessive, plural, and case in one token sequence that would require
several words in English. These properties make Uyghur a challenging target for NLP
systems trained predominantly on Latin or CJK script data.

The most widely cited dedicated translation system for Uyghur is **NLLB-200**
(Costa-jussà et al., 2022), a 200-language seq2seq model trained on large-scale
mined bitext and human-curated parallel data. NLLB-200 natively supports Uyghur
(`ug_Arab`), achieving perfect language identification on FLORES-200 (F1 = 100.0,
precision = 99.9, recall = 100.0 on `uig_Arab`, Table 49 of the paper). The paper
does not publish EN↔UG translation chrF or BLEU scores directly; FLORES-200
translation performance for the Uyghur direction can be obtained by running
NLLB-200 inference on the FLORES-200 devtest, which we do not do in this project
as NLLB-200 is not one of our evaluated models. Regardless of its translation
performance, NLLB-200 is a pure machine translation system with no
instruction-following capability; it cannot be prompted for classification, question
answering, or general dialogue in Uyghur.

A more recent line of work extends large language models to Uyghur through continued
pretraining. **CUTE-Llama-P** (Zhuang & Sun, 2025), introduced alongside the CUTE
corpus at COLING 2025, continues pretraining a Llama2-7B base with vocabulary
expansion on the CUTE-P parallel corpus (~934K lines each of Chinese, English,
Uyghur, and Tibetan, ~24.7 GB total). The model achieves strong results on ZH→UG translation (FLORES-200 BLEU 10.2,
chrF 0.443, Table 9) and Uyghur downstream tasks (WCM-v2 accuracy 87.0%, F1 89.08,
Table 6). Critically, the paper uses **Chinese as the pivot language** and reports
only ZH→UG FLORES numbers; no EN↔UG evaluation appears anywhere in the paper.
The EN↔UG direction — more practically relevant for international use — is not
evaluated, and the full continued pretraining + vocabulary surgery pipeline is
computationally expensive relative to adapter-based methods.

**MC²-LLaMA** (Zhang et al., 2024), introduced in "MC²: Towards Transparent and
Culturally-Aware NLP for Minority Languages in China" (Zhang, Tao, Huang, Lin,
Chen, & Feng; Peking University), takes a similar continued-pretraining approach
across multiple minority languages of China, including Uyghur, Tibetan, and
Mongolian, with an expanded multilingual vocabulary. It demonstrates partial Uyghur
capability but does not focus on the EN↔UG direction and does not report FLORES
EN↔UG numbers.

None of these prior works combine: (a) a modern Qwen2.5-class multilingual base
model with native Uyghur script support, (b) parameter-efficient LoRA-only tuning
without vocabulary surgery, and (c) direct evaluation of the EN↔UG translation
direction using the CUTE-P corpus. This is the gap the present project addresses.

---

## 2.2 Parameter-Efficient Fine-Tuning

Full fine-tuning of a 7–8B parameter language model requires storing and updating
all model weights, which is prohibitive under the MIG 1g.10gb (~10 GB VRAM) compute
constraint of this project. Two complementary techniques make fine-tuning feasible.

**LoRA** (Hu et al., 2021) addresses the update cost by constraining weight updates
to a low-rank factorisation. Instead of updating the full weight matrix W, LoRA
injects two small trainable matrices whose product approximates the update; only
these adapter matrices are trained while the original weights remain frozen. For a
7B model with LoRA rank 16, the number of trainable parameters is approximately
1% of the total — dramatically reducing GPU memory and training time with minimal
loss in task performance (Hu et al., 2021).

**QLoRA** (Dettmers et al., 2023) extends this by quantising the frozen base model
to 4-bit NF4 (NormalFloat4), a data type optimised for normally distributed neural
network weights. The base model is loaded in 4-bit precision while the LoRA adapters
are kept in bf16; gradient checkpointing is applied throughout. In principle, this
reduces the base-model memory footprint of a 7B model substantially relative to
bf16 loading (~14 GB). In practice, Day-1 preflight checks on this project's MIG
`1g.10gb` slice (~10 GB VRAM) show that Qwen2.5-7B peaks at 9.118 GB and
LLaMA-3.1-8B peaks at 8.551 GB during weight initialisation at the default training
configuration — right at the edge of the available VRAM. Fine-tuning on the 10 GB
slice is achievable for Qwen2.5-7B with targeted configuration adjustments
(SDPA attention, expandable memory allocator, non-reentrant gradient checkpointing,
batch size 1 + gradient accumulation 32), but LLaMA-3.1-8B fine-tuning requires
the larger `2g.20gb` slice requested from cluster admins (Plan B in `03_planned_approach.md`).

Together, LoRA and QLoRA allow fine-tuning Qwen2.5-7B-Instruct on the full CUTE-P
EN↔UG corpus (~934K pairs) in a single priority-partition Slurm job (~28 hours on
Plan A, ~16–20 hours on Plan B), with no vocabulary surgery and no need for
full-model weight updates.

---

## 2.3 Multilingual LLMs as Fine-Tuning Bases

Unlike dedicated translation models (§2.1), modern large language models are
instruction-following systems that can handle translation, classification, and
open-ended generation within a single unified interface. Two open-weight models
are used in this project.

**Qwen2.5-7B-Instruct** (Qwen Team, 2024) is the primary fine-tuning base and the
primary zero-shot baseline. Developed by Alibaba, it is trained on a large
multilingual corpus with strong CJK and Arabic-script coverage, and is released
under the Apache 2.0 license. Notably, the Qwen2.5 technical report does not
explicitly name Uyghur among its supported languages — Uyghur capability is
inferred from the broad Arabic-script and multilingual pre-training coverage, and
validated empirically by the Day-1 tokenizer sanity check (Uyghur token/byte ratio
0.396, below the 0.6 threshold, confirming acceptable segmentation without
vocabulary surgery).

**LLaMA-3.1-8B-Instruct** (Meta, 2024) is the secondary fine-tuning base (under
Plan B, pending a larger MIG slice) and a zero-shot baseline (available under
both plans). It belongs to the same model family as CUTE-Llama-P (which extends
Llama2-7B), making it a useful architectural reference point. It is available under
the Meta LLaMA 3 Community License (gated, instant access on HuggingFace). Its
Uyghur tokenizer segmentation is verified by the Day-1 check (token/byte ratio
0.460, below the 0.6 threshold). Fine-tuning it on the 10 GB MIG slice is not
feasible at any reasonable training configuration due to the model's 8B parameter
count; it is a fine-tuning target only under Plan B.

A deliberate design decision distinguishes both models from CUTE-Llama-P: **neither
undergoes vocabulary expansion or tokenizer modification**. This is the primary
research contrast — we test whether a modern multilingual base with native
multilingual coverage can match a model that was explicitly extended for Uyghur
through expensive vocabulary surgery, using only parameter-efficient adapters.

---

## Comparison Table

| Method | Data type | Training cost | Vocab handling | EN↔UG support | Published? |
|--------|-----------|---------------|----------------|---------------|------------|
| NLLB-200 (Costa-jussà et al., 2022) | Parallel MT (mined bitext) | Full model training | Native Uyghur (LID F1=100) | ✓ (MT only; no EN↔UG scores published) | ✓ |
| CUTE-Llama-P (Zhuang & Sun, 2025) | Parallel (ZH pivot) | Full CPT + vocab expansion | Expanded tokenizer | ZH→UG only | ✓ |
| MC²-LLaMA (Zhang et al., 2024) | Mixed multilingual | Continued pretraining | Expanded tokenizer | Partial | ✓ |
| **Ours — Qwen2.5-7B + QLoRA** | **Parallel EN↔UG (CUTE-P)** | **QLoRA only (~1% params)** | **Native (verified: UG ratio 0.396)** | **✓ both directions** | **This work** |

*All prior work misses at least one of: modern multilingual base / LoRA-only training /
EN↔UG direction.*

---

## Gap Statement

No published work to date combines all four of the following properties: (a) a
modern Qwen2.5-class multilingual base model with native Uyghur Arabic script
handling, (b) parameter-efficient QLoRA-only fine-tuning without continued
pretraining or vocabulary surgery, (c) direct training and evaluation on the
EN↔UG translation direction, and (d) use of the CUTE-P corpus as the fine-tuning
data source. NLLB-200 satisfies (c) but is a pure translation system with no
instruction-following capability and requires full model training. CUTE-Llama-P
satisfies (d) but uses Chinese as a pivot, ignores (c), and requires expensive
vocabulary surgery that contradicts (b). MC²-LLaMA partially satisfies (a) and (b)
but does not target the EN↔UG direction. Each prior work addresses at most two of
the four properties; this project occupies the currently empty intersection.

---

## References

- Costa-jussà, M. R., et al. (2022). No Language Left Behind: Scaling Human-Centered
  Machine Translation. *arXiv:2207.04672*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient
  Finetuning of Quantized LLMs. *arXiv:2305.14314*.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
  Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  *arXiv:2106.09685*.
- Zhang, C., Tao, M., Huang, Q., Lin, J., Chen, Z., & Feng, Y. (2024). MC²: Towards
  Transparent and Culturally-Aware NLP for Minority Languages in China. Peking University.
- Qwen Team. (2024). Qwen2.5 Technical Report. Alibaba Group.
- Zhuang, Y., & Sun, M. (2025). CUTE: A Corpus of Chinese, Uyghur, Tibetan and
  English for Low-Resource Multilingual NLP. In *Proceedings of COLING 2025*.

---

## Slide Notes

### Slide 3 — "Related Work" (~1 min)

**Visual:** The comparison table above (4 rows × 6 columns). Bold and highlight
the "Ours" row in the slide deck.

**Caption (one sentence, placed below the table):**
"All prior work misses at least one of: modern multilingual base / LoRA-only
training / EN↔UG direction."

**Speaker notes:**
- Point to NLLB-200: strong MT, but no instruction-following.
- Point to CUTE-Llama-P: same corpus, but Chinese pivot + expensive vocab surgery
  (and structurally unloadable on the 10 GB MIG slice).
- Point to MC²-LLaMA: shows Uyghur LLMs exist, but no EN↔UG.
- Land on "Ours": the only row with QLoRA + EN↔UG + modern base.

---

### Slide 4 — "The Gap" (~1 min)

**Visual:** 2×2 matrix diagram.
- X-axis: **EN↔UG support** (No → Yes, left to right)
- Y-axis: **Training cost** (High → Low, bottom to top)
- Positions:
  - NLLB-200: bottom-right (high cost, EN↔UG yes — MT only)
  - CUTE-Llama-P: bottom-left (high cost, EN↔UG no — ZH pivot only)
  - MC²-LLaMA: bottom-left area (high cost, partial EN↔UG)
  - **Ours**: top-right (low cost, EN↔UG yes) — **this quadrant is currently empty**

**Caption:** "Low training cost + EN↔UG support: currently unoccupied."

**Speaker notes:**
- The top-right quadrant is the research gap.
- QLoRA makes "low cost" achievable on a single MIG slice.
- Anticipate Q: "Why not just use NLLB-200?" → We target instruction-following,
  not pure MT. NLLB-200 cannot be prompted for classification or dialogue.
