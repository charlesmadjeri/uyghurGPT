# 1. Problem Description

## 1.1 The Real-World Problem
Uyghur is a minority language in china spoken by approximately 13~15 million people globally, yet current Large Language Models (LLMs) are not sufficent enough to use Uyghur language. In the context of global natural language processing, Uyghur suffers from severe data scarcity and is classified as a low-resource language (Haddow et al., 2022). Consequently, Uyghur speakers are largely excluded from the benefits of modern generative AI. There is an urgent user need for a capable, bilingual English↔Uyghur AI assistant that can perform accurate translations and follow complex instructions in both languages.

## 1.2 Why It Is Hard
Developing an effective LLM for Uyghur presents several unique linguistic and technical challenges:

*   **Agglutinative Morphology:** Uyghur is a highly agglutinative language where multiple suffixes are attached to a root word to convey grammatical relationships (Tursun & Cakici, 2017). This complex morphology is very different from English.
*   **Orthography and Directionality:** The language is written in an Arabic-based script and read from right to left, adding complexity and recognition confusion to text processing and generation.
*   **Scarce Parallel Corpora:** High-quality parallel data between English and Uyghur is extremely limited compared to major languages like Spanish or Mandarin.
*   **Suboptimal Tokenization:** Modern LLM tokenizers, which are heavily optimized for Latin and CJK (Chinese, Japanese, Korean) characters, tend to fragment Uyghur words poorly, leading to inefficient generation and degraded performance.
*   **Lack of Published Baselines:** There is a severe lack of established, published baseline models specifically targeting the direct English↔Uyghur (EN↔UG) translation and instruction-following direction.

## 1.3 The Deep Learning Task
Formally, we frame this problem as Supervised Instruction Fine-Tuning (SFT) of a causal language model using EN↔UG instruction pairs.

*   **Input:** A constructed prompt comprising a task instruction paired with a source sentence.
*   **Output:** The generated target sequence in the requested language.
*   **Loss Function:** Standard next-token cross-entropy loss, computed only over the response tokens (response masking), ensuring the model learns to generate the answer rather than memorize the prompt.

## 1.4 Scope Statement
To ensure this project remains feasible within our computational constraints, we strictly define our scope:

*   **In Scope:** Direct EN↔UG translation and instruction tuning; utilizing 7B-8B parameter class models (primarily Qwen2.5-7B); employing parameter-efficient tuning via QLoRA/LoRA; using the CUTE-P dataset (Zhuang & Sun, 2025) as our primary training corpus.
*   **Out of Scope:** We will not address Chinese↔Uyghur (ZH↔UG) translation; we will not fine-tune dedicated MT models like NLLB; we will not train models under 7B parameters; and we will not perform full continued pre-training (CPT) or vocabulary expansion.

## 1.5 Research Question
**Can parameter-efficient LoRA fine-tuning on a modern multilingual LLM (Qwen2.5-7B) match the performance of continued pretraining combined with vocabulary expansion (CUTE-Llama-P) on EN↔UG translation and Uyghur classification, at a fraction of the compute cost?**