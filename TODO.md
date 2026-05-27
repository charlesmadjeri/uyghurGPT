# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

## Re-submit UG→EN failure-mode diagnostic (Slurm 2751 failed)

Slurm **2751** crashed immediately with `ModuleNotFoundError: No module
named 'shared'`. Fixed in `scripts/debug_ug2en.py` (`sys.path.insert(0,
REPO_ROOT)`). Re-submit after rsync.

**Step 1 — push (quote globs for zsh):**

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/
```

**Step 2 — submit:**

```bash
ssh ju-compute-server 'cd ~/uyghurGPT && mkdir -p results/debug && sbatch \
  --job-name=debug_ug2en --time=2:00:00 --ntasks=1 --cpus-per-task=4 \
  --mem=24G --gres=gpu:1 --partition=priority \
  --output=results/debug/slurm_ug2en_%j.out \
  --wrap="cd \$HOME/uyghurGPT && set -a && source .env && set +a && \
    export HF_HOME=\$HOME/uyghurGPT/hf_cache && \
    export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN && \
    export CUDA_VISIBLE_DEVICES=0 && export PYTHONUNBUFFERED=1 && \
    \$HOME/micromamba/envs/uyghur_env/bin/python -u \
      scripts/debug_ug2en.py --compare-zeroshot -n 20"'
```

**Step 3 — pull:**

```bash
rsync -avz ju-compute-server:~/uyghurGPT/results/debug/ results/debug/
```

**Decision table** (after `results/debug/ug2en_<timestamp>.json`):

| Dominant bucket | Next step |
|---|---|
| `A_wrong_language_uyghur` | Stronger EN cue at eval; else Mix-50 (`docs/tasks/bonus/02_qwen_mix_ablation.md`) |
| `B_garbled_or_weak_english` | Higher Mix or LoRA capacity |
| `C_decoding_or_template_leak` | Re-open `docs/tasks/03_ug2en_decoding_fix.md` |
| `ok_english` | Inspect aggregation / sentence-length |

## Done (remove when read)

- ~~CUTE-Llama-P exp 2 (Slurm 2750)~~ — `run_20260526_224102`, logged in
  `PROJECT_RESULTS.md` §1 + §2.
- ~~Zero-shot WCM constrained-LL (Slurm 2749)~~ — `run_20260526_223852`.
- ~~Tasks 01 + 02~~ — marked `done` in `docs/tasks/`.
