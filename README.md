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
├── main.py                  # CLI entrypoint (preflight + experiment dispatch)
├── experiments/             # per-experiment pipelines
│   └── experiment_1/        # core Qwen Mix-20 QLoRA (config + run)
├── shared/                  # cross-experiment modules
│   ├── preflight.py         # Day-1 sanity checks
│   ├── data.py              # CUTE-P + FLAN loading and instruction formatting
│   ├── models.py            # model ids, QLoRA + tokenizer helpers
│   ├── training.py          # QLoRA fine-tune (SFTTrainer)
│   └── evaluation.py        # FLORES-200, WCM-v2, C4 perplexity
├── utils/                   # run I/O (artifacts layout), terminal logging
├── scripts/
│   ├── push.py              # rsync code + submit Slurm job
│   ├── check.py             # job status + pipeline stage + result pull
│   └── run_preflight.py     # submit Day-1 preflight as one Slurm job
├── docs/
│   ├── PROJECT.md           # full project plan (canonical)
│   ├── PROJECT_REFINEMENT.md
│   ├── 03_planned_approach.md
│   ├── RESEARCH.md
│   └── papers/              # reference PDFs (gitignored)
├── results/                 # per-run artifacts (gitignored)
└── requirements.txt
```

## Runtime requirements

- Python 3.10+
- 1× **NVIDIA A100 80GB PCIe** per worker on `slurm.hj.se`, with each
  `--gres=gpu:1` job receiving a **~24 GB MIG slice** (upgraded from the
  earlier `1g.10gb` ~10 GB profile). Training defaults to **QLoRA** (4-bit
  NF4 base + bf16 LoRA adapters + gradient checkpointing), peak ~8–12 GB.
  **bf16 LoRA also fits** on the 24 GB slice (~18–22 GB) via `--bf16-lora`
  for ~2× speed. Slurm jobs default to `--time=5-00:00:00` (the full
  `priority` partition cap). See `docs/PROJECT.md` §Compute environment.
- HuggingFace stack (`transformers`, `peft`, `trl`, `accelerate`,
  `bitsandbytes`, `datasets`) — see `requirements.txt`

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

## CLI overview

`main.py` is the single entrypoint and has two modes:

| Invocation | What it does |
|------------|--------------|
| `--mode preflight` | Day-1 sanity checks (tokenizer, QLoRA VRAM, CUTE-P sample, CUTE-Llama-P load) — writes `results/preflight/` |
| `--experiment N --mode {preprocess,train,eval,all}` | Run a stage (or all stages) of experiment `N`. Currently `N=1` is the core Qwen Mix-20 QLoRA pipeline. |

Stage semantics for an experiment:

- **`preprocess`** — load CUTE-P (local files or HF Hub fallback), build
  bidirectional EN↔UG instructions, blend Mix-{0/10/20/50} FLAN EN-only
  examples, save to `artifacts/preprocessed_dataset/`.
- **`train`** — QLoRA fine-tune Qwen using the preprocessed dataset; cosine LR,
  3% warmup, paged AdamW 8-bit, gradient checkpointing; checkpoints saved per
  epoch under `checkpoints/<model_label>/`.
- **`eval`** — FLORES-200 EN↔UG (chrF, BLEU), WCM-v2 accuracy, C4 perplexity
  (catastrophic forgetting). Evaluates zero-shot Qwen, zero-shot LLaMA, and
  the fine-tuned Qwen adapter; results to `artifacts/eval_*.json`.
- **`all`** — runs the three above sequentially in the same run directory.

## Day-1 preflight

```bash
python3 main.py --mode preflight                       # local (CPU/GPU)
python3 scripts/run_preflight.py --server compute-server  # as a single Slurm job
python3 main.py --mode preflight --check 1,2           # subset of checks
```

Pass condition for memory checks: peak VRAM **< 22 GB** on the ~24 GB MIG slice.

## Run locally (smoke test)

```bash
python3 main.py --experiment 1 --model qwen --mix 20 --epochs 1 --sample-count 100
```

Run the full pipeline:

```bash
python3 main.py --experiment 1 --model qwen --mix 20 --epochs 3
```

Run a single stage:

```bash
python3 main.py --experiment 1 --mode preprocess --run-id myrun
python3 main.py --experiment 1 --mode train      --run-id myrun
python3 main.py --experiment 1 --mode eval       --run-id myrun
```

## Run on compute server

```bash
python3 scripts/push.py --server compute-server --model qwen --epochs 3
```

Force a new run instead of resuming the latest incomplete one:

```bash
python3 scripts/push.py --server compute-server --model qwen --epochs 3 --new-run
```

Bootstrap missing Python packages on the server before training (the full
HuggingFace fine-tuning stack):

```bash
python3 scripts/push.py --server compute-server --new-run --install-deps
```

Useful flags (defaults shown): `--mode all`, `--experiment 1`, `--time
5-00:00:00`, `--cpus 8`, `--gpus 1`, `--partition priority`.

## Monitor and pull results

```bash
# Status only (squeue/sacct + pipeline stage from run_status.json)
python3 scripts/check.py --server compute-server

# Fast TensorBoard sync (only logs/ + run_status.json + run_config.json)
python3 scripts/check.py --server compute-server --logs

# Full pull (excludes checkpoints by default)
python3 scripts/check.py --server compute-server --pull

# Full pull including adapter weights
python3 scripts/check.py --server compute-server --pull --pull-checkpoints
```

`run_status.json` advances through:
`started → preprocessed → training → trained → evaluating → evaluated`.
TB scalars (loss, learning rate, grad norm) are emitted every 10 steps:

```bash
tensorboard --logdir results
```

## Per-run artifacts

Each run writes to `results/run_<run_id>/experiment_<N>/`:

- `artifacts/run_config.json` — frozen hyperparameters
- `artifacts/run_status.json` — current pipeline stage and timestamp
- `artifacts/preprocessed_dataset/` — HF-saved preprocess output
- `artifacts/eval_<benchmark>.json` — one file per benchmark and variant
- `checkpoints/<model_label>/` — LoRA adapters per epoch (e.g. `qwen_mix20`)
- `logs/<model_label>/` — TensorBoard / TRL training logs

Preflight runs once per cluster (not per experiment) and writes to
`results/preflight/` instead.

## Evaluation

| Benchmark | Direction / Task | Metric | Tool |
|-----------|------------------|--------|------|
| FLORES-200 (devtest) | EN→UG | chrF, BLEU | `sacrebleu` |
| FLORES-200 (devtest) | UG→EN | chrF, BLEU | `sacrebleu` |
| WCM-v2 (Uyghur) | classification | Accuracy | HF datasets |
| C4 (en, 1K samples) | held-out perplexity | PPL | `transformers` |
| MiLiC-Eval | 9 tasks (stretch) | task-specific | HF datasets |

All FLORES-200 numbers — for both our fine-tuned models and the baselines —
are produced by us on **EN→UG and UG→EN**. The paper only publishes ZH→UG
numbers for CUTE-Llama-P, which are its best-case direction (Chinese was the
pivot during continued pretraining) and not informative about EN↔UG; we
ignore them.

Baselines (all run by us on FLORES-200 EN↔UG):

- **CUTE-Llama-P** (Zhuang & Sun, 2025) — core baseline; loads in 4-bit NF4
  on the 24 GB MIG slice (preflight check 5 PASS). Prompted with few-shot
  `English: …\nUyghur:` continuation since it is a base LM, not instruct;
  protocol difference is reported alongside its score. Fall back to
  zero-shot-only baselines only if a future cluster change breaks loading.
- **Qwen2.5-7B-Instruct, zero-shot** — isolates the contribution of LoRA
  fine-tuning on Uyghur capability.
- **LLaMA-3.1-8B-Instruct, zero-shot** — same isolation for the secondary
  model (and the same Llama family as CUTE-Llama-P, without the vocabulary
  surgery).
- *(Optional)* **Qwen2.5-7B-Instruct, 5-shot** — in-context examples without
  weight updates.

For WCM-v2 (Uyghur classification) and MiLiC-Eval, we compare directly to the
paper's reported CUTE-Llama-P numbers (Acc 87.0% / F1 89.08 on WCM-v2 Uyghur),
since those are Uyghur-language evaluations regardless of which pivot the
model was trained through.

## Data mixing ratio (catastrophic forgetting)

| Variant | UG/EN CUTE-P | EN-only (FLAN) | Status |
|---------|---------------|-----------------|--------|
| Mix-0   | 100%          | 0%              | Stretch (ablation) |
| Mix-10  | 90%           | 10%             | Stretch (ablation) |
| **Mix-20** | **80%**    | **20%**         | **Core — default** |
| Mix-50  | 50%           | 50%             | Stretch (ablation) |

Each variant is evaluated on Uyghur capability gain (chrF EN↔UG, WCM-v2
accuracy) versus English retention (C4 perplexity delta vs. the base model).
