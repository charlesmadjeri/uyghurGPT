# Glossary:
Translation quality metrics:
- BLEU: Bilingual Evaluation Understudy
    - Papineni et al., 2002
    - The standard MT metric for ~20 years
    - Operates on words (KO for UG)
- chrF (character n-gram F-score)
    - Popović, 2015
    - The current best-practice metric for low-resource and morphologically rich languages 
    - Operates on characters (OK for UG)

## Datasets

### NLLB-200 (Costa-jussà et al., 2022):
- most widely cited dedicated translation system for Uyghur
- natively supports Uyghur
- paper does not publish EN↔UG translation chrF or BLEU scores directly
- BUT NLLB-200 is a pure machine translation system with no instruction-following capability; it cannot be prompted for classification, question answering, or general dialogue in Uyghur

### CUTE-Llama-P (Zhuang & Sun, 2025):
- introduced with the CUTE dataset
- continuous pretraining a Llama2-7B base with vocabulary expansion on the CUTE-P parallel corpus
- strong results on ZH→UG translation (FLORES-200 BLEU 10.2, chrF 0.443)
- BUT the paper uses Chinese as the pivot language and reports only ZH→UG FLORES numbers

### MC2
- demonstrates partial Uyghur capability but does not focus on the EN↔UG direction and does not report FLORES EN↔UG numbers

None of these prior works combine:
- a modern Qwen2.5-class multilingual base model with native Uyghur script support
- parameter-efficient LoRA-only tuning without vocabulary surgery
- direct evaluation of the EN↔UG translation direction using the CUTE-P corpus

This is the gap the present project addresses.

## Fine-tuning LLMs

Full fine-tuning of a 7–8B parameter language model requires storing and updating
all model weights, which is too much under the (~10 GB VRAM) slices of the compute server.
Two complementary techniques make fine-tuning feasible:

- **LoRA** (Hu et al., 2021) addresses the update cost by constraining weight updates
to a low-rank factorisation.
- Instead of updating the full weight matrix W, **LoRA injects two small trainable matrices** whose product approximates the update; **only these adapter matrices are trained** while the original weights remain frozen.
- For a 7B model with LoRA rank 16, the number of trainable parameters is approximately
1% of the total — **reducing GPU memory and training time with minimal loss in task performance** (Hu et al., 2021).

- **QLoRA** (Dettmers et al., 2023) extends this by quantising the frozen base model
to 4-bit NF4 (NormalFloat4), a data type optimised for normally distributed neural
network weights.
- The base model is loaded in 4-bit precision while the LoRA adapters are kept in bf16

## Choosing LLM base to fine-tune

Unlike dedicated translation models, modern LLMs are instruction-following systems that can handle translation, classification, and open-ended generation within a single unified interface.
Two open-weight models are used in this project:

### Qwen2.5-7B-Instruct (Qwen Team, 2024)
- Our primary fine-tuning base and the primary zero-shot baseline
- Developed by Alibaba, it is trained on a large multilingual corpus with strong CJK and Arabic-script coverage
- Apache 2.0 license

### LLaMA-3.1-8B-Instruct (Meta, 2024)
- Our secondary fine-tuning base and a zero-shot baseline.
- It belongs to the same model family as CUTE-Llama-P (which extends Llama2-7B), useful architectural reference point.

# Slides notes:
## Slide "Related Work"
- **NLLB-200**: strong MT, but no instruction-following.
- **CUTE-Llama-P**: same corpus, but Chinese pivot + expensive vocab surgery
- **MC²-LLaMA**: shows Uyghur LLMs exist, but no EN↔UG.
- **Ours**: the only row with QLoRA + EN↔UG + modern base.

## Slide "The Gap"
- The bottom-right area is the **research gap**
- QLoRA makes "low cost" achievable.
- "Why not just using NLLB-200?" → We target instruction-following,not pure MT. NLLB-200 cannot be prompted for classification or dialogue.


## Slide "Approach Overview"
- CUTE-P is the only large-scale EN↔UG parallel corpus available
- We add a 20 % FLAN slice (English-only) to avoid English instruction-following forgetting

- All training examples are formatted into Qwen's native ChatML template (system + user + assistant)

- Evaluation is on two unseen datasets:
    - FLORES-200 devtest for EN↔UG translation (chrF first, BLEU second)
    - WCM-v2 for Uyghur text classification. Neither overlaps with training data.

- We have one training pipeline, two compute paths depending of the possibility to get 20GB VRAM compute slices

## Slide "Why QLoRA + Qwen2.5​"
QLoRA box:
- Using a quantized base + LoRA adapters, is what makes a 7B model train in ~10 GB instead of the ~80 GB
Qwen2.5 box:
- Open-source
- We checked the tokenizer: Qwen is segmenting Uyghur at ~2× the rate of English, not ~10× like a Latin-only tokenizer would.

The deeper design choice:
- by using zero-shot Qwen as one of our baselines, we control for everything the base model already knows.
- the difference between fine-tuned-Qwen and zero-shot-Qwen is measuring the effect of LoRA training on CUTE-P
