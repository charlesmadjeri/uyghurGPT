# Cluster setup — from zero to a green preflight run

End-to-end log of everything needed to take a fresh `slurm.hj.se` account
(or any equivalent SLURM cluster with an NVIDIA L4 / A100 MIG node) to a
fully **PASS** preflight run for UyghurGPT. Each step is what we actually
did to get the run captured in `results/preflight/preflight_report.md`.

Whenever a step has a "verify" command, run it before moving to the next
step. If the verify fails, fix that step before continuing — most of the
downstream checks assume the previous ones succeeded.

---

## 0. Local prerequisites

Done once on your laptop.

### 0.1 Clone the repo and create a Python venv

```bash
git clone <repo-url> uyghurGPT
cd uyghurGPT
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The local venv is only used for code editing / dry runs. The heavy
training and preflight runs happen on the cluster.

### 0.2 Hugging Face account + token

Create a token at <https://huggingface.co/settings/tokens> (read scope is
enough). Accept the terms of use for every gated repository the project
touches — preflight check 6 will fail otherwise:

| Repo | Why we need it |
|------|----------------|
| <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> | Secondary base model + zero-shot baseline |
| <https://huggingface.co/datasets/openlanguagedata/flores_plus> | FLORES-200 EN↔UG eval (devtest) |
| <https://huggingface.co/datasets/hfl/wcm-v2> | Uyghur classification eval |
| <https://huggingface.co/datasets/pkupie/milic-eval> | Stretch eval (optional) |

Acceptance is instant for all of these.

### 0.3 Create `.env`

```bash
cp .env.example .env
# edit .env and replace hf_your_huggingface_token_here with your real token
```

`.env` is **gitignored** but is rsynced to the cluster by the push scripts
and sourced by the SLURM wrap.

### 0.4 SSH alias

Add an entry to `~/.ssh/config` so the push scripts can reach the cluster:

```sshconfig
Host ju-compute-server
    HostName slurm.hj.se
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Verify:

```bash
ssh ju-compute-server 'hostname && nvidia-smi -L'
```

You should see the cluster hostname and at least one CUDA-capable GPU
(L4 or A100 MIG slice). If `nvidia-smi` isn't on the login node, that's
fine — GPUs are only visible inside a SLURM allocation. In that case
verify with:

```bash
ssh ju-compute-server 'sinfo -o "%n %P %G"'
```

---

## 1. One-time cluster bootstrap

Run all commands in this section on the **login node** of the cluster:

```bash
ssh ju-compute-server
```

### 1.1 Install micromamba

`~/.local` is root-owned on this cluster, so `pip install --user` fails.
Micromamba gives us a per-user Python install in `$HOME/micromamba` with
no `sudo` required.

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# Follow prompts: install to $HOME/micromamba, shell hook into ~/.bashrc

source ~/.bashrc
micromamba --version
```

### 1.2 Create the `uyghur_env` Python environment

```bash
micromamba create -y -n uyghur_env python=3.11
micromamba activate uyghur_env
python -c "import sys; print(sys.executable)"
# expected: $HOME/micromamba/envs/uyghur_env/bin/python
```

Every later command — both push scripts and SLURM wraps — invokes
`$HOME/micromamba/envs/uyghur_env/bin/python` explicitly. Do **not**
rely on `python` being on `PATH` inside SLURM jobs.

### 1.3 Sync the repo once so `requirements.txt` is available remotely

Back on your laptop:

```bash
python3 scripts/push.py --server ju-compute-server --new-run --install-deps
```

`--install-deps` triggers the cluster-side install sequence described
below; you can also abort the SLURM job with `scancel <jobid>` right
after submission — the install runs *inside* the wrap so you'd need a
running job. Easier: do the install by hand the first time:

```bash
ssh ju-compute-server
cd ~/uyghurGPT
micromamba activate uyghur_env

# 1. Hugging Face stack + utilities (sentencepiece, sacrebleu, tensorboard, ...)
python -m pip install -r requirements.txt jinja2 huggingface_hub

# 2. PyTorch built for the cluster's CUDA 12.2 driver.
#    The default torch wheel is built for cu13.0 and "No CUDA device" at
#    runtime; cu121 wheels are ABI-compatible with cu122 drivers.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch

# 3. Pin fsspec back into datasets's supported range.
#    torch's install bumps fsspec past 2026.2, but datasets 4.x requires
#    fsspec[http]<=2026.2.0.
python -m pip install 'fsspec[http]>=2023.1.0,<=2026.2.0'
```

### 1.4 Verify the install

Submit a 5-minute interactive sanity check:

```bash
srun --gres=gpu:1 --time=00:05:00 --partition=priority --pty bash
# inside the allocation:
$HOME/micromamba/envs/uyghur_env/bin/python -c "
import torch, transformers, peft, bitsandbytes, datasets, sacrebleu, trl, tensorboard
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('trl', trl.__version__)
"
exit
```

Expected:

```
torch 2.x.x+cu121 cuda True
device NVIDIA L4
transformers 4.4x.x
datasets 4.x.x
```

If `torch.cuda.is_available()` is `False`, reinstall torch from the
cu121 index. If you see an `fsspec` `ImportError` from `datasets`,
re-run step 1.3 line 3.

### 1.5 (Done by push scripts) Per-job environment exports

Both `scripts/push.py` and `scripts/run_preflight.py` set these inside
the SLURM `--wrap`. You only need to know they exist if a future job
misbehaves:

| Variable | Why |
|----------|-----|
| `CUDA_VISIBLE_DEVICES=0` | Forces accelerate to skip its multi-device probe (which crashes on MIG with `NVML_SUCCESS == r`). |
| `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | Avoids torch's caching allocator's NVML init crash on MIG slices. |
| `HF_HOME=$HOME/uyghurGPT/hf_cache` | Keeps the multi-GB HF cache off the small home quota. |
| `HUGGING_FACE_HUB_TOKEN=$HF_TOKEN` | Legacy clients still read this name. |

---

## 2. Run the preflight

### 2.1 Submit (from your laptop)

```bash
python3 scripts/run_preflight.py --server ju-compute-server
```

What this does:

1. Rsyncs the repo (including `.env`) to `~/uyghurGPT` on the cluster.
2. Submits a single SLURM job (`--gres=gpu:1`, `--time=01:30:00`,
   `--partition=priority`) whose wrap sources `.env`, exports the
   variables in §1.5, and runs `main.py --mode preflight --check all`.
3. Prints the job id; logs stream to
   `~/uyghurGPT/results/preflight/slurm_<jobid>.out`.

Subset of checks (e.g. skip the long CUTE-Llama load):

```bash
python3 scripts/run_preflight.py --server ju-compute-server --check 1,2,3,4,6
```

Valid ids are 1–6 (see `shared/preflight.py`).

### 2.2 Monitor

```bash
python3 scripts/check.py --server ju-compute-server          # status only
python3 scripts/check.py --server ju-compute-server --pull   # fetch artifacts
```

`--pull` includes TensorBoard event files (both `logs/` and the
`checkpoints/<label>/runs/` paths) and eval JSON by default. It skips
`preprocessed_dataset/` (reproducible on the server, often 250 MB+),
`hf_cache/`, and the adapter weight directories
(`checkpoints/<label>/checkpoint-*/`, `checkpoints/<label>/final/`).
Use `--pull-checkpoints` to also fetch the adapter weights, or `--logs`
for the leanest TB-only sync.

The preflight takes ≈90 seconds end-to-end on a clean cache, ≈3 minutes
on the very first run (model downloads dominate).

### 2.3 Expected output

Once SLURM marks the job COMPLETED and you pull, you should see this in
`results/preflight/preflight_report.md`:

| # | Status | Check |
|---|--------|-------|
| 1 | PASS | Tokenizer Uyghur segmentation |
| 2 | PASS | QLoRA memory fit — Qwen2.5-7B |
| 3 | PASS | QLoRA memory fit — LLaMA-3.1-8B |
| 4 | PASS | CUTE-P EN+UG download + format spot-check |
| 5 | FAIL | CUTE-Llama-P load + inference test |
| 6 | PASS | HuggingFace repo access |

Check 5 is **expected to FAIL** on a clean infrastructure — the paper
baseline (`CMLI-NLP/CUTE-Llama`) produces degenerate outputs on FLORES+
eval sentences regardless of the cluster setup. A FAIL here means
"CUTE-Llama-P is not a usable baseline", not "the cluster is broken".
Checks 1–4 and 6 are the real gate. See PROJECT.md §Baseline Risk for
the fallback path.

---

## 3. Things that broke and how we fixed them

In order of how we hit them, so a future setup can short-circuit the same
debugging:

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: sacrebleu` (or transformers, peft, …) inside SLURM | Wrap was using cluster `python3`, not the micromamba env. Confirm scripts call `$HOME/micromamba/envs/uyghur_env/bin/python`. |
| `pip install --user` → `Permission denied` on `~/.local` | `~/.local` is root-owned. Use micromamba (§1.1). |
| Job dies on `cd: $HOME/uyghurGPT: No such file or directory` | sbatch `--wrap` was single-quoted, so `$HOME` didn't expand. Both push scripts now use double-quoted wraps. |
| `RuntimeError: No CUDA device` inside preflight | Torch was built for cu13.0; cluster driver is cu12.2. Reinstall torch from the cu121 index (§1.3 line 2). |
| `ImportError: fsspec ...` from `datasets` after torch install | Torch bumps fsspec past 2026.2; pin it back (§1.3 line 3). |
| `c10 NVML_SUCCESS == r INTERNAL ASSERT FAILED` (torch on MIG) | Set `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (already in the wraps). |
| `Dataset scripts are no longer supported, but found flores.py` (check 5) | `datasets >= 2.20` refuses scripts. Switched to `openlanguagedata/flores_plus` per-language configs, joined by `id`. Same path used by `shared/evaluation.py`. |
| Check 6 reports `GATED_NO_ACCESS` for `hfl/wcm-v2` or `flores_plus` | Re-accept terms on the HF website with the **same** account whose token is in `.env`. |

### Experiment 1 train — additional failures (after preflight)

These appeared when running `--mode train` with current PyPI `trl`
(≈0.20+). `shared/training.py` tolerates version differences, but you
still need `tensorboard` installed for TensorBoard logging:

| Symptom | Fix |
|---------|-----|
| `cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'` | Upgrade `shared/training.py` (uses `assistant_only_loss` or full-sequence fallback). |
| `SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'` | Same — code maps to `max_length` on modern TRL. |
| `PeftModel` instance together with a `peft_config` | Same — do not pass `peft_config` when model is already wrapped. |
| `assistant_only_loss=True` but dataset is not conversational | Same — our data uses pre-templated `text` strings, not `messages` lists. |
| `TensorBoardCallback requires tensorboard to be installed` | `pip install tensorboard` in `uyghur_env`, or rely on code fallback to `report_to="none"`. Listed in `requirements.txt`. |
| Train job uses a **new** run dir; `FileNotFoundError: preprocessed_dataset` | **Must pass `--run-id`** (see §4). Omitting `--new-run` does **not** auto-resume. |

---

## 4. After preflight passes

You're cleared to run experiment 1.

### 4.1 Fresh run (preprocess + train + eval)

```bash
# smoke (~20 min)
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode all --model qwen --mix 20 \
  --epochs 1 --sample-count 64 --new-run --time 1:00:00

# full (5-day walltime default)
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode all --model qwen --mix 20 \
  --epochs 3 --new-run
```

Add `--install-deps` on first train after pulling code changes (installs
`requirements.txt` + cu121 torch + fsspec pin inside the job wrap).

### 4.2 Resume after preprocess succeeded but train failed

The preprocessed HF dataset lives under
`results/run_<id>/experiment_1/artifacts/preprocessed_dataset` **on the
server only** (not pulled by default). Reuse that run id explicitly:

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode train \
  --model qwen --mix 20 --epochs 3 \
  --run-id 20260523_182843
```

Replace `20260523_182843` with your run's timestamp from
`results/run_*/experiment_1/artifacts/run_config.json`.

Then eval (after train completes):

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval \
  --model qwen --mix 20 \
  --run-id 20260523_182843 \
  --time 4:00:00
```

### 4.3 Monitor

```bash
python3 scripts/check.py --server ju-compute-server
python3 scripts/check.py --server ju-compute-server --pull
python3 scripts/check.py --server ju-compute-server --logs   # TB scalars only
```

See the README §"Run on compute server" for more flags.

### 4.4 Train / test / eval split

The CUTE-P + FLAN mix is split at `--mode preprocess` time:

| Split | Source | Used for |
|-------|--------|----------|
| `train` | 95% of CUTE-P+FLAN mix (configurable, `test_split_pct`) | gradient updates |
| `test`  | 5% held-out from the same mix | in-loop `eval/loss` curve in TensorBoard — **the overfit detector** (compare against `train/loss`) |
| `eval`  | external FLORES+ devtest, WCM-v2 test, C4 PPL | final benchmark numbers (`--mode eval`), data the model has never seen, different domain |

TensorBoard will show `train/loss` (every `logging_steps=10`) and
`eval/loss` (every `eval_steps=50`). A widening gap = overfitting. The
splits are saved as a single `DatasetDict` under
`results/run_<id>/experiment_1/artifacts/preprocessed_dataset/`.

> Runs whose `preprocess` step happened **before** the train/test split
> landed will be loaded as `{"train": <full mix>}` (no `test` split). The
> training loop logs a warning and skips in-loop eval. To get the
> overfit curve, re-run `--mode preprocess` with a fresh `--new-run`.

