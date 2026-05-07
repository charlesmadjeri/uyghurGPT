# UyghurGPT — Bilingual Uyghur/English LLM Fine-tuning

Final project for the Deep Learning course (Jönköping University). We fine-tune
**Qwen2.5-7B-Instruct** (primary) and **LLaMA-3.1-8B-Instruct** (secondary) on
the **CUTE-P** parallel corpus (English↔Uyghur, both directions) using LoRA
instruction tuning, and benchmark them against the **CUTE-Llama-P** baseline
from Zhuang & Sun (COLING 2025) on FLORES-200, WCM-v2, and MiLiC-Eval. All
four comparison models are size-matched to the 7–8B class.

See [`docs/PROJECT.md`](docs/PROJECT.md) for the full project plan and
[`docs/RESEARCH.md`](docs/RESEARCH.md) for the prior research notes that led
to this scope.

## Research question

> Can a modern multilingual LLM (Qwen2.5) fine-tuned with LoRA on EN↔UG
> instruction data match or surpass a model trained with full continued
> pretraining and vocabulary expansion (CUTE-Llama-P) on the same corpus?

## Project structure

```
uyghurGPT/
├── main.py                  # CLI entrypoint (train + evaluate)
├── docs/
│   ├── PROJECT.md           # full project plan
│   ├── RESEARCH.md          # research recap that led to this project
│   └── papers/              # reference PDFs (gitignored)
├── scripts/
│   ├── push.py              # rsync code + submit Slurm job
│   └── check.py             # monitor jobs + pull results
├── shared/                  # data / model / eval modules (TBD)
├── experiments/             # per-experiment entrypoints (TBD)
├── results/                 # per-run artifacts (gitignored)
└── requirements.txt
```

## Runtime requirements

- Python 3.10+
- 1× **NVIDIA A100 80GB PCIe** per worker on `slurm.hj.se`, **but Slurm assigns a
  MIG `1g.10gb` slice (~10 GB VRAM)** to each `--gres=gpu:1` job. Training therefore
  defaults to **QLoRA** (4-bit NF4 base + bf16 LoRA adapters + gradient checkpointing),
  which fits in ~6–9 GB. A bf16-LoRA flag is available for if/when admins grant a
  full GPU. See `docs/PROJECT.md` §Compute environment.
- HuggingFace stack (`transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`,
  `datasets`) — see `requirements.txt`

## Conda setup and install

```bash
conda create -n uyghurgpt python=3.11
conda activate uyghurgpt
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Venv setup and install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run locally (smoke test)

```bash
python3 main.py --model qwen --mix 20 --epochs 1 --sample-count 100
```

Run on the full pipeline:

```bash
python3 main.py --model qwen --mix 20 --epochs 3
```

## Run on compute server

```bash
python3 scripts/push.py --server <ssh-alias> --model qwen --epochs 3
```

Force a new run instead of resuming the latest incomplete one:

```bash
python3 scripts/push.py --server <ssh-alias> --model qwen --epochs 3 --new-run
```

Bootstrap missing Python packages on the server before training (the full
HuggingFace fine-tuning stack):

```bash
python3 scripts/push.py --server <ssh-alias> --model qwen --epochs 3 --new-run --install-deps
```

## Monitor and pull results

```bash
python3 scripts/check.py --server <ssh-alias>
python3 scripts/check.py --server <ssh-alias> --pull
```

## Evaluation

| Benchmark | Direction | Metric | Tool |
|-----------|-----------|--------|------|
| FLORES-200 | EN↔UG | chrF, BLEU | `sacrebleu` |
| WCM-v2 (Uyghur) | classification | Accuracy, F1 | `scikit-learn` |
| MiLiC-Eval | 9 tasks | task-specific | HuggingFace |

All FLORES-200 numbers — for both our fine-tuned models and the baselines —
are produced by us on **EN→UG and UG→EN**. The paper only publishes ZH→UG
numbers for CUTE-Llama-P, which are its best-case direction (Chinese was the
pivot during continued pretraining) and not informative about EN↔UG; we
ignore them.

Baselines (all run by us on FLORES-200 EN↔UG):

- **CUTE-Llama-P** (Zhuang & Sun, 2025) — primary baseline; load from
  HuggingFace and run inference. Our EN→UG and UG→EN numbers are new data
  points not reported by the paper.
- **Qwen2.5-7B-Instruct, zero-shot** — isolates the contribution of LoRA
  fine-tuning on Uyghur capability.
- **LLaMA-3.1-8B-Instruct, zero-shot** — same isolation for the secondary model
  (and the same Llama family as CUTE-Llama-P, but without the vocabulary surgery).
- *(Optional)* **Qwen2.5-7B-Instruct, 5-shot** — in-context examples without
  weight updates, as a non-fine-tuning point of comparison.

For WCM-v2 (Uyghur classification) and MiLiC-Eval, we compare directly to the
paper's reported CUTE-Llama-P numbers (Acc 87.0% / F1 89.08 on WCM-v2 Uyghur),
since those are Uyghur-language evaluations regardless of which pivot the
model was trained through.

## Per-run artifacts

Each run writes to `results/run_<run_id>/`:

- `artifacts/run_config.json`, `run_status.json`
- `artifacts/eval_<benchmark>.json` per evaluation benchmark
- `artifacts/training_history.csv`
- `checkpoints/<model_label>/` — LoRA adapters per epoch
- `logs/<model_label>/` — TensorBoard / TRL training logs

## Ablation: data mixing ratio

| Variant | UG/EN CUTE-P | EN-only (FLAN) |
|---------|---------------|-----------------|
| Mix-0 | 100% | 0% |
| Mix-10 | 90% | 10% |
| Mix-20 (default) | 80% | 20% |
| Mix-50 | 50% | 50% |

Each variant is evaluated on Uyghur capability gain (chrF EN↔UG, WCM-v2
accuracy) versus English retention (perplexity on a held-out English split).
