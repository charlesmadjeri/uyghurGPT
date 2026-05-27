# TODO

Short-lived, actionable items that don't belong in `docs/tasks/` (those
are larger task plans with status, deliverables, validation criteria).
Pop entries from here as they land; once `TODO.md` is empty, delete it.

## Pending Slurm runs

### Investigate CUTE-Llama-P FLORES stall (Slurm 2745 / 2748 / 2750)

> **Now the most important item.** `cute_llama_p` is the only remaining
> `pending` cell in `PROJECT_RESULTS.md` §2 *Final results — core
> experiments*. Three exp-2 submissions have all stalled at the same
> point — right after `[eval] FLORES-200 n=1012 few-shot k=3 (EN→UG
> then UG→EN) ...`, with **zero** per-50-sentence progress dots
> printed before walltime / silence. The 24 h resubmit (Slurm 2750 /
> `run_20260526_224102`) still has not moved as of the 02:07 pull.

**Step 1 — confirm whether Slurm 2750 is dead or just slow:**

```bash
ssh ju-compute-server 'squeue --me; sacct -j 2750 --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList'
```

If `State=RUNNING`, give it another 30 min and re-pull — fp16 7B +
eager attention with a 3-shot prompt may legitimately need that long
for the first 50-sentence batch (worst-case estimate from Slurm 2745
telemetry was ~30 s/sentence ⇒ 25 min per batch of 50, but the first
batch also pays the CUDA graph / kv-cache warm-up cost). If `State`
is `TIMEOUT` / `OUT_OF_MEMORY` / `FAILED`, jump to Step 2.

**Step 2 — instrument before the next resubmit.** The current
`generate_translation_fewshot` only prints once per 50 sentences;
that's too coarse for "is the first call hung or just slow?".
Minimum patch (don't ship to repo unless you're going to keep it —
revert after diagnosis):

```python
# shared/evaluation.py::generate_translation_fewshot (TEMP)
import time
t0 = time.time()
out = model.generate(...)
print(f"[eval] sentence {i} took {time.time()-t0:.1f}s", flush=True)
```

Re-submit a **smoke run** first (`--sample-count 5` if it threads, or
hard-cap `flores_max_samples=10` in `Experiment2Config` for one
submission), not the full 1 012 × 2.

**Step 3 — if generation is the bottleneck, reduce inference cost
before the next full submission:**

- Drop `repetition_penalty` (cheapest change; preflight check 5 set
  it defensively, may not be load-bearing on FLORES-style prompts).
- Lower few-shot `k` from 3 to 1 (cuts prompt length ~3×, which
  dominates KV-cache cost on fp16 7B).
- Cap `max_new_tokens` from 200 → 80 (FLORES single sentences rarely
  need more than ~60 BPE tokens of target text).

Open in `docs/tasks/01_experiment_2_cute_llama_p_baseline.md` as a
sub-deliverable if any of the three is kept in the final config.

---

### UG→EN failure-mode diagnostic on the compute server (deferred)

> Lower priority than the CUTE-Llama-P stall above. The fine-tune
> regression is logged + analysed (`PROJECT_RESULTS.md` §2 *Analysis*
> bullet 2); the diagnostic only deepens the explanation.
> Defer until at least one queue slot is free (association limit is
> 2 concurrent submitted jobs).

**Step 1 — push code (mirrors `scripts/push.py`'s rsync invocation; the
diagnostic is a stand-alone script, not a `main.py` mode, so it doesn't
go through `push.py` directly):**

```bash
# Quote globs — zsh expands unquoted *.pyc / *.ipynb locally and errors
# with "no matches found" when none exist in cwd.
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/
```

**Step 2 — submit (only once a queue slot is free):**

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

**Step 3 — pull the report once the job completes:**

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

## Pending experiment hygiene (cluster-side)

- ~~Pull `run_20260526_223852` (exp 0, Slurm 2749) once `evaluated`.~~
  **Done 2026-05-27 02:07.** Delta logged in `PROJECT_RESULTS.md` §1
  "2026-05-27 — `run_20260526_223852`"; §2 cells updated in the same
  commit.
- ~~Pull `run_20260526_222254` (exp 2, Slurm 2750) once `evaluated`.~~
  Pulled, but the run never produced any FLORES output — see the
  *Investigate CUTE-Llama-P FLORES stall* item above for the
  follow-up. (Also note: the job submitted to the cluster as
  Slurm **2750** corresponds to `run_20260526_224102`, not
  `run_20260526_222254` — the earlier dir is Slurm **2748**, which
  also stalled identically.)
