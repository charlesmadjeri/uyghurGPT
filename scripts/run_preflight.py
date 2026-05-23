"""Submit Day-1 preflight checks as a SINGLE Slurm job that runs sequentially.

The original design submitted one sbatch per check, but two real constraints make
the single-job design simpler and more robust:

  1. The cluster QoS caps concurrent submitted jobs (we hit AssocMaxSubmitJobLimit
     at the 4th submission).
  2. Bash quoting around the install command + pip version specifiers truncates
     the wrap command when nested single quotes collide with `>=` redirection.

Single job: one GPU allocation, checks run in order (4 -> 1 -> 2 -> 3 -> 5),
all logs go to one slurm output file, results JSONs go to results/preflight/.

Pre-requisites (one-time setup on the cluster login node):

  1. Install the fine-tuning stack (this also installs torch as a dep):
       python3 -m pip install --user 'transformers>=4.44' 'peft>=0.12' 'trl>=0.10' \\
         'accelerate>=0.34' 'bitsandbytes>=0.43' 'datasets>=2.20' sentencepiece

  2. Install a torch wheel matching the cluster CUDA driver. The default
     `pip install transformers` pulls a torch built for the newest CUDA,
     which fails on older drivers. The compute nodes here report a
     CUDA-12.2-class driver, so install cu121 wheels explicitly:
       python3 -m pip install --user --index-url https://download.pytorch.org/whl/cu121 torch
     Verify with:
       python3 -c "import torch; print(torch.cuda.is_available())"

  3. Create a local .env file (cp .env.example .env) and fill in HF_TOKEN.
     This script rsyncs .env to the server alongside the code, and the
     sbatch wrap sources it before invoking python.

Usage:
  # Submit all five checks in one job
  python scripts/run_preflight.py --server ju-compute-server

  # Skip a check (e.g. Llama license not accepted yet)
  python scripts/run_preflight.py --server ju-compute-server --check 1,2,4,5

After submission, monitor with `python scripts/check.py --server <alias>`
and pull results once SLURM marks the job COMPLETED.
"""

import argparse
import os
import subprocess
import sys


G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[96m"
BOLD = "\033[1m"; RESET = "\033[0m"
def ok(s):   return f"{G}{BOLD}✔ {s}{RESET}"
def info(s): return f"{B}{s}{RESET}"
def warn(s): return f"{Y}{BOLD}⚠ {s}{RESET}"
def err(s):  return f"{R}{BOLD}✘ {s}{RESET}"

REMOTE_PYTHON = "$HOME/micromamba/envs/uyghur_env/bin/python"

VALID_CHECKS = {1, 2, 3, 4, 5, 6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH alias from ~/.ssh/config")
    parser.add_argument(
        "--check",
        default="all",
        help="Which check(s) to run: 'all', a single number (e.g. '3'), or a comma list (e.g. '1,2,4,5')",
    )
    parser.add_argument("--partition", default="priority")
    parser.add_argument("--time", default="01:30:00",
                        help="Slurm walltime for the single job (default 1h30m)")
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip rsync (use if code is already in sync)")
    args = parser.parse_args()

    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    remote_dir = "~/uyghurGPT"

    env_path = os.path.join(local_dir, ".env")
    if not os.path.exists(env_path):
        print(err(".env not found at " + env_path))
        print(info("Run:  cp .env.example .env  and fill in HF_TOKEN"))
        sys.exit(2)
    with open(env_path) as f:
        env_body = f.read()
    if "hf_your_huggingface_token_here" in env_body or "HF_TOKEN=" not in env_body:
        print(err(".env does not contain a real HF_TOKEN — edit it before submitting."))
        sys.exit(2)

    if not args.no_sync:
        print(info(f"Syncing code {local_dir} -> {args.server}:{remote_dir} ..."))
        exclude = [
            "--exclude=results/", "--exclude=results.archive/",
            "--exclude=__pycache__/", "--exclude=*.pyc",
            "--exclude=.git/", "--exclude=.venv/", "--exclude=*.ipynb",
            "--exclude=docs/papers/", "--exclude=dataset/",
            "--exclude=models/", "--exclude=checkpoints/",
        ]
        # .env is NOT excluded — it is shipped to the server so the sbatch
        # wrap can source it. (.env is gitignored locally; it never leaves
        # via git, only via this rsync to the user's private homedir on the
        # cluster.)
        rsync = ["rsync", "-avz", "--progress"] + exclude + [f"{local_dir}/", f"{args.server}:{remote_dir}/"]
        if subprocess.run(rsync).returncode != 0:
            print(err("rsync failed.")); sys.exit(1)
        print(ok("Code + .env synced."))

    if args.check == "all":
        check_arg = "all"
    else:
        ids = sorted({int(x.strip()) for x in args.check.split(",") if x.strip()})
        for c in ids:
            if c not in VALID_CHECKS:
                print(err(f"Unknown check id: {c} (valid: 1-6)")); sys.exit(2)
        check_arg = ",".join(str(c) for c in ids)

    # sbatch wrap, executed on the compute node:
    #   - cd into the synced repo
    #   - source .env (HF_TOKEN, etc) with `set -a` so each KEY=VALUE is exported
    #   - mirror HF_TOKEN into HUGGING_FACE_HUB_TOKEN for legacy clients
    #   - put ~/.local/bin on PATH (user-installed pip console scripts)
    #   - run the preflight via main.py
    # All $VARs are escaped with \$ so they expand on the REMOTE shell, not locally.
    # sbatch wrap, executed on the compute node:
    #   - cd into the synced repo
    #   - source .env (HF_TOKEN, etc) with `set -a` so each KEY=VALUE is exported
    #   - mirror HF_TOKEN into HUGGING_FACE_HUB_TOKEN for legacy clients
    #   - put ~/.local/bin on PATH (user-installed pip console scripts)
    #   - CUDA_VISIBLE_DEVICES=0 makes the MIG slice the only visible device,
    #     bypassing accelerate's multi-device probing
    #   - PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync switches torch from
    #     its native caching allocator (which calls NVML on init and crashes
    #     with NVML_SUCCESS == r INTERNAL ASSERT FAILED on MIG slices) to the
    #     CUDA driver's stream-ordered allocator, which works on MIG.
    #     This is the fix for the bug we hit on the previous run.
    #   - run the preflight via main.py
    # All $VARs are escaped with \$ so they expand on the REMOTE shell, not locally.
    sbatch_inline = (
        f"cd {remote_dir} && mkdir -p results/preflight && "
        f"sbatch --job-name=preflight "
        f"--time={args.time} --ntasks=1 --cpus-per-task={args.cpus} --mem={args.mem} "
        f"--gres=gpu:1 --partition={args.partition} --requeue "
        f"--output=results/preflight/slurm_%j.out "
        f"--wrap=\"cd \\$HOME/uyghurGPT && "
        f"set -a && source .env && set +a && "
        f"export HF_HOME=\\$HOME/uyghurGPT/hf_cache && "
        f"export HUGGING_FACE_HUB_TOKEN=\\$HF_TOKEN && "
        f"export CUDA_VISIBLE_DEVICES=0 && "
        f"export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync && "
        f"echo HF_TOKEN_set=\\$([ -n \\\"\\$HF_TOKEN\\\" ] && echo yes || echo no) && "
        f"{REMOTE_PYTHON} main.py --mode preflight --check {check_arg}\""
    )
    print(info(f"Submitting preflight job (checks={check_arg}) ..."))
    proc = subprocess.run(["ssh", args.server, sbatch_inline], capture_output=True, text=True)
    if proc.returncode != 0:
        print(err(proc.stderr.strip())); sys.exit(1)
    out = proc.stdout.strip()
    print(ok(out))
    print(info("\nMonitor: python scripts/check.py --server " + args.server))
    print(info("Pull:    python scripts/check.py --server " + args.server + " --pull"))


if __name__ == "__main__":
    main()
