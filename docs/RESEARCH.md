# DL Final Project — Uyghur NLP Research Recap

> Research session: April 2026  
> Context: Deep Learning course (Jönköping University), H100 compute available

---

## 1. The Problem Space

**Uyghur** is a low-resource Turkic language (~12M speakers) with specific NLP challenges:

- Uses Arabic script, written right-to-left
- Agglutinative morphology (many suffixes per word — very different from English)
- Limited parallel corpora compared to major languages
- Existing tokenizers (trained on Latin/CJK-heavy data) handle it poorly

An H100 is a significant asset: most prior Uyghur NLP work was done under much tighter compute constraints.

---

## 2. Three Research Tracks (Full Landscape)

### Track 1 — English ↔ Uyghur Translation

| Option | Description | Notes |
|--------|-------------|-------|
| **A — Fine-tune NLLB-200** | Meta's No Language Left Behind (600M distilled) already supports Uyghur | Recommended starting point |
| **B — Fine-tune mBART-50** | Older multilingual seq2seq, supports Uyghur (`ug_UG`) | Useful as comparison baseline |
| **C — Seq2seq from scratch** | Train a small Transformer encoder-decoder with custom BPE tokenizer | Educational; unlikely to beat NLLB |
| **D — Distillation** | Distill a large translation model into a smaller Uyghur-specific one | Complex pipeline |

### Track 2 — Uyghur-Capable LLM

| Option | Description | Notes |
|--------|-------------|-------|
| **A — Continual pretraining** | Continue pretraining a small multilingual LLM (e.g. Llama-3.2-1B/3B) on Uyghur text | Corpus is genuinely small (~80MB CC-100) |
| **B — LoRA/QLoRA instruction fine-tuning** | Instruction-tune a multilingual base (e.g. Qwen2.5) for Uyghur QA/chat | No ready Uyghur instruction dataset exists |
| **C — Tokenizer surgery** | Extend an existing model's vocabulary with Uyghur-specific tokens, retrain embedding layer | Underexplored, legitimate research contribution |

### Track 3 — Other Ideas

| Idea | What it involves | Status |
|------|-----------------|--------|
| Uyghur ASR | Fine-tune Whisper on Uyghur audio | Common Voice Uyghur ~2-4h only |
| Script normalization | Arabic ↔ Latin Uyghur conversion | Mostly algorithmic, limited DL scope |
| Cross-lingual transfer | Use Turkish (high-resource Turkic) to bootstrap Uyghur | Linguistically interesting |
| RAG in Uyghur | Retrieval-augmented system over Uyghur documents | Unclear embedding quality |
| Benchmark creation | Curate/translate a small Uyghur eval set | Manual, labor-intensive |

---

## 3. Data Collection Filter

**Criterion applied:** eliminate any option where data collection requires more than pulling and combining existing datasets.

### Eliminated

| Option | Reason |
|--------|--------|
| Distillation (1D) | Requires building a full inference pipeline to generate teacher outputs first |
| Instruction fine-tuning (2B) | No Uyghur instruction dataset exists; would need to translate Alpaca/etc. = separate pipeline |
| ASR (3) | Common Voice Uyghur is ~2-4 hours — too small, scraping needed to supplement |
| Benchmark creation (3) | Manual curation, not a pull |
| RAG (3) | Uyghur document corpus is thin; embedding quality unknown |

### Survivors

| Option | Data sources | Where |
|--------|-------------|-------|
| NLLB-200 fine-tune (1A) | FLORES-200, CCAligned, WikiMatrix | HuggingFace / OPUS |
| mBART-50 fine-tune (1B) | Same as above | HuggingFace / OPUS |
| Seq2seq from scratch (1C) | Same parallel data + CC-100 `ug` for tokenizer | HuggingFace |
| Continual pretraining (2A) | CC-100 `ug` (~80MB), Uyghur Wikipedia dump | HuggingFace / wikidump |
| Tokenizer surgery (2C) | CC-100 `ug` + model tokenizer | HuggingFace |
| Cross-lingual transfer (3) | Turkish data (abundant on OPUS) + Uyghur eval | OPUS / HuggingFace |

---

## 4. NLLB-200 — What It Is

**No Language Left Behind** — a multilingual seq2seq translation model released by Meta in 2022.  
Supports 200 languages including Uyghur. Available in multiple sizes; the distilled 600M version fits easily on an H100.

**Training data (what it was trained on):**
- Mined bitext from Common Crawl via LASER3 sentence encoders
- NLLB-Seed: human-curated pairs created by Meta
- Existing public translation datasets (biblical translations, research corpora)

**Key caveat:** NLLB was released in 2022. It has not seen:
- **CUTE (2025)** — Chinese-Uyghur-Tibetan-English, ~25GB, currently the largest open-source Uyghur corpus. Generated via machine translation (synthetic).
- **MC2 corpus** — multilingual minority languages in China, on HuggingFace
- **Chinese-Uyghur parallel corpus** — 4.72M pairs (commercial but referenced)

---

## 5. Data Landscape for Uyghur

### Parallel corpora (pullable)

| Dataset | Pair | Size | Quality | Notes |
|---------|------|------|---------|-------|
| FLORES-200 | EN↔UG (and many others) | ~1K sentences | High (human) | Evaluation benchmark only |
| CCAligned | EN↔UG | ~200K pairs | Noisy | Web-mined |
| WikiMatrix | EN↔UG | ~50-100K pairs | Moderate | Wikipedia-based |
| OPUS / OpenSubtitles | EN↔UG | Small | Mixed | Subtitles |
| CUTE | ZH/EN↔UG | ~25GB | Synthetic (MT) | Post-NLLB, unseen by NLLB |
| MC2 corpus | Multi↔UG | Variable | Mixed | HuggingFace |
| Chinese-Uyghur parallel | ZH↔UG | 4.72M pairs | Cleaned | Primarily Chinese-centric |

### Monolingual Uyghur

| Dataset | Size | Notes |
|---------|------|-------|
| CC-100 `ug` | ~80MB | Web crawl, available on HuggingFace |
| Uyghur Wikipedia | ~4K articles | Small but clean |

**Total high-quality EN↔UG parallel data: ~100–300K sentence pairs.** This is the ceiling for English-focused work.

---

## 6. Language Pair Choice — Framework

Four factors to weigh:

1. **Data volume** — more data = easier training, but less room for novel contribution
2. **Existing baseline quality** — check NLLB FLORES-200 scores; high baseline = less room to improve
3. **Evaluation infrastructure** — FLORES-200 covers all three candidate pairs ✅
4. **Research angle clarity** — what question does the pair motivate?

### Candidate pairs compared

| Pair | Parallel Data | Baseline Gap | Research Angle |
|------|--------------|-------------|----------------|
| EN↔UG | ~100-300K | NLLB already decent | Classic low-resource fine-tuning |
| ZH↔UG | ~4.7M+ (CUTE + others) | Less studied in western NLP | Richer data; pivot language possible |
| TR→UG | Very sparse | Large gap | Cross-lingual Turkic transfer |

**Key tension:** Chinese-Uyghur has the most data but is less standard for a western DL course context. English-Uyghur is the most natural framing but has less room to beat NLLB. Turkish-Uyghur is the most linguistically interesting but data is very sparse.

---

## 7. Open Questions (Unresolved)

- [ ] What is the actual NLLB-200 BLEU/chrF score on Uyghur in FLORES-200? (determines how much room to improve)
- [ ] Is the CUTE dataset usable as fine-tuning data despite being machine-translated?
- [ ] Does the Chinese-centric nature of most new Uyghur data make ZH↔UG a better target than EN↔UG for this project?
- [ ] What language pair to target — EN, ZH, or TR as the pivot/source?
- [ ] What is the research question: data quantity effects? tokenization quality? cross-lingual transfer?

---

## 8. Final Decision

After this research recap, the project moved away from the NLLB-200 fine-tuning track (Track 1) and into the LLM instruction fine-tuning track (Track 2). The reasoning:

- **CUTE-Llama-P (released 2025) is a stronger, more directly comparable baseline than NLLB-200.** It was trained on the same corpus we plan to use (CUTE-P) and reports published numbers. Beating an LLM baseline trained with vocabulary surgery is a sharper claim than beating a generic seq2seq translator.
- **Modern multilingual LLMs (Qwen2.5) likely have latent Uyghur capability already.** Fine-tuning becomes "unlock latent capability with LoRA" rather than "teach a language from scratch", which matches the single-GPU LoRA + TRL `SFTTrainer` toolchain available on the JU compute cluster.
- **Instruction fine-tuning generalizes beyond translation.** WCM-v2 (classification) and MiLiC-Eval (9 tasks) become viable evaluations alongside FLORES-200, which a pure NLLB fine-tune does not enable.

### Final scope

| Decision | Value |
|----------|-------|
| Track | 2 — LLM instruction fine-tuning with LoRA |
| Language pair | English ↔ Uyghur, **both directions** |
| Fine-tune corpus | CUTE-P (Zhuang & Sun, COLING 2025), full ~934K EN↔UG pairs |
| Catastrophic-forgetting mix | ~20% English-only FLAN subset (Mix-20 default) |
| Primary model | **Qwen2.5-7B-Instruct** |
| Secondary model | **LLaMA-3.1-8B-Instruct** |
| Primary baseline | **CUTE-Llama-P** (we run inference on FLORES-200 EN↔UG ourselves) |
| Secondary / tertiary baselines | Llama-3.1-8B (zero-shot), Qwen2.5-7B (zero-shot) |
| Evaluation | FLORES-200 (chrF, BLEU; both directions) · WCM-v2 (Uyghur classification) · MiLiC-Eval (9 tasks) |
| Ablation | Mix-{0, 10, 20, 50} on the primary model |

### What we do **not** do

- **No NLLB-200 fine-tuning** — replaced by LoRA on a multilingual LLM.
- **No comparison against the paper's published ZH→UG numbers** — direction mismatch makes them uninformative for our task.
- **No 3B models** — competing 3B against the 7B CUTE-Llama-P baseline would confound parameter count with the experimental variable. All models are size-matched to 7–8B.
- **No vocabulary surgery on Qwen / Llama** — both already handle Uyghur Arabic script in their native tokenizer, which is a deliberate contrast against CUTE-Llama-P's approach.

See **`PROJECT.md`** for the full project plan, baselines, ablation design, and compute environment.

---

*Decision recorded May 2026.*
