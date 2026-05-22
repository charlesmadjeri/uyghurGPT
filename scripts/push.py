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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH alias from ~/.ssh/config")
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama"],
                        help="Which base model to fine-tune")
    parser.add_argument("--mix", type=int, default=20,
                        help="Percentage of EN-only data mixed into training (0/10/20/50)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--time", default="1-00:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--partition", default="priority")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install HuggingFace fine-tuning stack (transformers, peft, trl, accelerate, "
             "bitsandbytes, datasets, sacrebleu) at job start",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Force creation of a new run instead of auto-resuming latest incomplete run",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        help="Experiment id passed to main.py (default: 1 = core Qwen Mix-20)",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["preprocess", "train", "eval", "all"],
        help="Pipeline stage(s) to run on the cluster",
    )
    args = parser.parse_args()

    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    remote_dir = "~/uyghurGPT"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if args.new_run else None

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
    if args.install_deps:
        install_parts.append(
            "python3 -m pip install --user -q "
            "'transformers>=4.44' 'peft>=0.12' 'trl>=0.10' 'accelerate>=0.34' "
            "'bitsandbytes>=0.43' 'datasets>=2.20' sacrebleu sentencepiece"
        )
    install_cmd = " && ".join(install_parts)
    if install_cmd:
        install_cmd += " && "
    run_arg = f"--run-id {run_id} " if run_id is not None else ""
    job_tag = run_id if run_id is not None else "resume"
    slurm_run_label = run_id if run_id is not None else "resume"
    sbatch_inline = (
        f"cd {remote_dir} && mkdir -p results && sbatch "
        f"--job-name=uyghur_{args.model}_{job_tag} "
        f"--time={args.time} --ntasks=1 --cpus-per-task={args.cpus} "
        f"--gres=gpu:{args.gpus} --partition={args.partition} --requeue "
        f"--output=results/slurm_{slurm_run_label}_%j.out "
        f"--wrap='{install_cmd}python3 main.py --experiment {args.experiment} "
        f"--mode {args.mode} {run_arg}"
        f"--model {args.model} --mix {args.mix} --epochs {args.epochs}'"
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
