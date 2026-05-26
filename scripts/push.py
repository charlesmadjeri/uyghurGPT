"""
Sync UyghurGPT to a compute server and submit a Slurm run.

Usage examples:
  python scripts/push.py --server my-hpc --epochs 1
  python scripts/push.py --server my-hpc --epochs 3 --new-run --install-deps
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[96m"
BOLD = "\033[1m"; RESET = "\033[0m"
def ok(s):   return f"{G}{BOLD}✔ {s}{RESET}"
def info(s): return f"{B}{s}{RESET}"
def warn(s): return f"{Y}{BOLD}⚠ {s}{RESET}"
def err(s):  return f"{R}{BOLD}✘ {s}{RESET}"

# Cluster: ~/.local is root-owned (pip --user fails). Use micromamba env instead.
REMOTE_PYTHON = "$HOME/micromamba/envs/uyghur_env/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH alias from ~/.ssh/config")
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama"],
                        help="Which base model to fine-tune")
    parser.add_argument("--mix", type=int, default=20,
                        help="Percentage of EN-only data mixed into training (0/10/20/50)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--time",
        default=None,
        help=(
            "Slurm walltime. If unset, defaults are derived from the "
            "observed wall time of run_20260524_020432 (+50%% margin): "
            "experiment 0 = 6:00:00 (~3h36m observed), experiment 1 = "
            "1-00:00:00 (~15h52m observed), experiment 2 = 6:00:00 "
            "(same shape as exp 0; first run will calibrate). Priority "
            "partition cap is 5-00:00:00; raise only if you have evidence "
            "the run will exceed."
        ),
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--mem",
        default="24G",
        help=(
            "Slurm host RAM. Default 24G matches the 24 GB VRAM ceiling "
            "(see docs/SERVER_CONFIG.md §4.0.1). Streaming preprocess "
            "fits in <1 GB; train + eval fit in <16 GB with "
            "low_cpu_mem_usage=True. Raise (e.g. --mem 32G) only if a "
            "log clearly shows a real host-side OOM."
        ),
    )
    parser.add_argument("--partition", default="priority")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Re-install deps into ~/micromamba/envs/uyghur_env at job start (normally one-time setup)",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Create a new run id (timestamp) and pass it to main.py",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing run directory (e.g. 20260523_182843 after preprocess)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        help=(
            "Experiment id for main.py (0 = zero-shot eval only; "
            "1 = core Qwen Mix-20 preprocess/train/eval; "
            "2 = CUTE-Llama-P few-shot baseline eval)"
        ),
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["preprocess", "train", "eval", "all"],
        help="Pipeline stage(s) to run on the cluster",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Subsample CUTE-P / eval for smoke tests (passed to main.py)",
    )
    args = parser.parse_args()

    # Wall-time default depends on the experiment (observed time × 1.5).
    # See docs/PROJECT_RESULTS.md for the source measurements.
    if args.time is None:
        default_time_by_experiment = {
            0: "6:00:00",
            1: "1-00:00:00",
            # Experiment 2 is eval-only (FLORES devtest 1012 × 2 + WCM 300 +
            # C4 PPL 1000) on a fp16 7B base LM. Same shape as exp 0 so we
            # reuse its 6 h budget; tighten after the first observed wall.
            2: "6:00:00",
        }
        args.time = default_time_by_experiment.get(args.experiment, "1-00:00:00")
        print(info(f"--time not set; using default {args.time} for experiment {args.experiment}"))

    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    remote_dir = "~/uyghurGPT"
    if args.new_run and args.run_id:
        print(err("Use either --new-run or --run-id, not both."))
        sys.exit(2)
    if args.new_run:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = args.run_id

    exclude = [
        "--exclude=results/",
        "--exclude=results.archive/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=*.ipynb",
        "--exclude=docs/papers/",
        "--exclude=dataset/",
        "--exclude=models/",
        "--exclude=checkpoints/",
    ]

    print(info(f"Syncing code {local_dir} -> {args.server}:{remote_dir} ..."))
    rsync_cmd = ["rsync", "-avz", "--progress"] + exclude + [f"{local_dir}/", f"{args.server}:{remote_dir}/"]
    result = subprocess.run(rsync_cmd)
    if result.returncode != 0:
        print(err("rsync failed."))
        sys.exit(1)
    print(ok("Code synced."))

    print(info("Submitting job..."))
    install_parts = []
    py = REMOTE_PYTHON
    if args.install_deps:
        install_parts.extend([
            f"{py} -m pip install -q -r requirements.txt jinja2 huggingface_hub",
            f"{py} -m pip install -q --index-url https://download.pytorch.org/whl/cu121 torch",
            # torch pulls fsspec>2026.2; datasets 4.x requires fsspec[http]<=2026.2.0
            f"{py} -m pip install -q 'fsspec[http]>=2023.1.0,<=2026.2.0'",
            # flash-attn must compile against the torch wheel above; failures
            # are logged but do not abort the job (training falls back to SDPA).
            f"{py} -u scripts/ensure_flash_attn.py",
        ])
    install_cmd = " && ".join(install_parts)
    if install_cmd:
        install_cmd += " && "
    run_arg = f"--run-id {run_id} " if run_id is not None else ""
    sample_arg = f"--sample-count {args.sample_count} " if args.sample_count else ""
    job_tag = run_id if run_id is not None else "resume"
    slurm_run_label = run_id if run_id is not None else "resume"
    # Double-quoted --wrap (same as run_preflight.py): single-quoted wrap leaves
    # $HOME literal and the job dies on "cd $HOME/uyghurGPT: No such file".
    sbatch_inline = (
        f"cd {remote_dir} && mkdir -p results && sbatch "
        f"--job-name=uyghur_{args.model}_{job_tag} "
        f"--time={args.time} --ntasks=1 --cpus-per-task={args.cpus} "
        f"--mem={args.mem} "
        f"--gres=gpu:{args.gpus} --partition={args.partition} --requeue "
        f"--output=results/slurm_{slurm_run_label}_%j.out "
        f"--wrap=\"cd \\$HOME/uyghurGPT && set -a && source .env && set +a && "
        f"export HF_HOME=\\$HOME/uyghurGPT/hf_cache && "
        f"export HUGGING_FACE_HUB_TOKEN=\\$HF_TOKEN && "
        f"export CUDA_VISIBLE_DEVICES=0 && "
        f"export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync && "
        f"export PYTHONUNBUFFERED=1 && "
        f"{install_cmd}{REMOTE_PYTHON} -u main.py --experiment {args.experiment} "
        f"--mode {args.mode} {run_arg}{sample_arg}"
        f"--model {args.model} --mix {args.mix} --epochs {args.epochs}\""
    )
    result = subprocess.run(["ssh", args.server, sbatch_inline], capture_output=True, text=True)
    if result.returncode != 0:
        print(err(result.stderr.strip()))
        sys.exit(1)

    print(ok(result.stdout.strip()))
    if run_id is None:
        print(info("Remote script will auto-resume latest incomplete run (or create a new one)."))
    else:
        print(info(f"Remote results will be written under: {remote_dir}/results/run_{run_id}"))
    print(info("Next: python scripts/check.py --server <alias> --pull"))


if __name__ == "__main__":
    main()
