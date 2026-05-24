## Task 4 — `docs/04_planned_evaluation.md`

## 4.To rigorously assess whether our parameter-efficient fine-tuning approach successfully imparts English↔Uyghur capabilities to Qwen2.5-7B, we pre-register the following systematic evaluation protocol.



## 4.1 Evaluation Matrix
Our experimental results will be reported according to the following evaluation matrix. Stretch goals (dependent on compute and model loading feasibility) are marked in grey/italics.
Model,                     FLORES-200 (EN↔UG),     WCM-v2 (Uyghur),           English Perplexity (C4)
Qwen2.5-7B (Base),         Zero-shot MT,           Zero-shot Classification,  Base PPL
Qwen2.5-7B + QLoRA (Ours), Fine-tuned MT,          Zero-shot Classification,  Fine-tuned PPL
CUTE-Llama-P (Stretch),    Zero-shot MT,           Zero-shot Classification,  Base PPL



## 4.2 Benchmarks and Metrics
* **FLORES-200 Devtest (Translation)**
Our primary benchmark for translation is the FLORES-200 devtest set, which contains 1012 sentences with high-quality parallel data for both English and Uyghur (Costa-jussà et al., 2022). We evaluate the EN→UG and UG→EN directions separately. Our primary metric is chrF++ (Popović, 2015), with BLEU (Papineni et al., 2002) as a secondary metric. We focus on chrF++ because it has been shown to be much better than other methods for languages like Uyghur, where the words can change a lot because of the suffixes added to the end of words.

* **WCM-v2 (Uyghur Text Classification)**
To evaluate downstream Uyghur language understanding, we use the WCM-v2 dataset. The fine-tuned model will be prompted zero-shot to perform document classification purely in Uyghur. The primary metric is Accuracy, with Macro-F1 reported as a secondary metric to account for class imbalances.

* **English Perplexity (Catastrophic Forgetting Check)**
To ensure that tuning on Uyghur data does not destroy the model's core English capabilities, we evaluate perplexity on a held-out set of 1,000 English sentences from the C4 corpus. We will compute the model.eval() perplexity before and after fine-tuning. A substantial relative increase (> 20%) will flag catastrophic forgetting.

* **MiLiC-Eval (Stretch Benchmark)**
As a stretch goal, we will evaluate the models on the 9-task bilingual MiLiC-Eval benchmark. However, this is deferred to the final report and will not be presented during the design stage.

## 4.3 Statistical Reporting and Rigor
To ensure reproducible and statistically sound claims, we will report the following:

Paired Bootstrap Resampling: We will use sacrebleu's --paired-bs flag (n=1000) for all translation comparisons to estimate whether differences in chrF++ scores are statistically significant.

Confidence Intervals: 95% confidence intervals will be provided for all primary metrics.

Limitation (Single Seed): Due to strict time and compute constraints, our primary training runs will use a single fixed random seed. This will be explicitly stated as a limitation in our final report.

Reproducibility: The exact sacrebleu signature will be provided for all reported BLEU and chrF++ scores.


## 4.4 Translation Direction Asymmetry (Expected Behavior)
We explicitly note an expected performance gap between translation directions. The EN→UG direction requires the model to generate fluent, morphologically correct Uyghur (a significantly harder task). Conversely, UG→EN requires understanding Uyghur but generating English, a language the base model already masters. So we expect EN→UG scores to be substantially lower than UG→EN scores. This asymmetry is an inherent property of text generation in low-resource target languages and does not constitute a failure of the fine-tuning process.


## 4.5 Pre-Registered Success CriteriaWe define three levels of success for this project:
1.Minimum Criteria: The fine-tuned Qwen2.5-7B+QLoRA outperforms the zero-shot base Qwen2.5-7B in ≥ 1 translation direction.
Rationale: Any measurable benefit from fine-tuning validates the core pipeline and data formatting.

2.Target Criteria: A ≥ 5 chrF++ absolute improvement over the baseline in both directions, and the WCM-v2 accuracy improved too.
Rationale: A 5-point chrF++ gain is widely considered a meaningful and perceptible improvement in low-resource machine translation literature.

3.Stretch Criteria: Our QLoRA model scores within 2 chrF++ points of the CUTE-Llama-P model on EN→UG, and beats it on UG→EN.
Rationale:  If we can achieve, it will show that our LoRA approach, which is efficient in terms of parameters, is as good as a model trained with full continued pre-training (CPT) and vocabulary expansion, but uses approximately 10 times less computing power

## 4.6 Scope of Claims (What We Will NOT Claim)
To maintain academic honesty, we explicitly state that this study will not include:

Human evaluation of translation fluency or adequacy.

Claims about how well the product is ready to be made and how stable it is when it is being used.

Generalization of this specific recipe to other low-resource languages.

Direct comparisons on ZH↔UG translation, as this presents a direction mismatch with the primary focus of the CUTE baseline paper.