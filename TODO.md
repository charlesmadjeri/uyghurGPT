# TODO

Short-lived, actionable items that don't belong in `docs/tasks/` (those
are larger task plans with status, deliverables, validation criteria).
Pop entries from here as they land; once `TODO.md` is empty, delete it.

## Pending Slurm runs

### UG→EN failure-mode diagnostic on the compute server

> Defer until at least one of the current jobs frees a queue slot (the
> association limit is 2 concurrent submitted jobs). Don't cancel Slurm
> `2749` (experiment-0 zero-shot WCM constrained-LL re-eval) or `2750`
> (experiment-2 CUTE-Llama-P) just to run this — they are required for
> the final results table; the diagnostic is informational.

Script + the latest `shared/evaluation.py` are already synced to
`~/uyghurGPT/` on `ju-compute-server`.

**Submit (when a slot is free):**

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

**Pull the report:**

```bash
rsync -avz ju-compute-server:~/uyghurGPT/results/debug/ results/debug/
```

**Decision after reading `results/debug/ug2en_<timestamp>.json`:**

| Dominant `failure_mode_counts` bucket | Next step |
|---|---|
| `A_wrong_language_uyghur` | Try stronger EN direction cue at eval; if that fails → Mix-50 retrain (bonus task `docs/tasks/bonus/02_qwen_mix_ablation.md`). |
| `B_garbled_or_weak_english` | Retrain at higher Mix or larger LoRA capacity. |
| `C_decoding_or_template_leak` | Re-open `docs/tasks/03_ug2en_decoding_fix.md`. |
| `ok_english` | Regression is in the corpus aggregation; look at sentence-length / repetition. |

## Pending experiment hygiene (cluster-side, after the in-flight jobs land)

- Pull `run_20260526_223852` (exp 0, Slurm 2749) once `evaluated`; add a
  delta entry to `docs/PROJECT_RESULTS.md` §1 and update the
  `qwen_zeroshot` / `llama_zeroshot` WCM cells in §2 in the **same**
  commit (per the file's convention).
- Pull `run_20260526_222254` (exp 2, Slurm 2750) once `evaluated`; same
  treatment for the `cute_llama_p` row.
