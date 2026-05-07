"""
Check UyghurGPT jobs and optionally pull results.

Usage:
  python scripts/check.py --server my-hpc
  python scripts/check.py --server my-hpc --pull
"""

import argparse
import os
import subprocess

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[96m"; C = "\033[96m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
def ok(s):      return f"{G}{BOLD}✔ {s}{RESET}"
def info(s):    return f"{B}{s}{RESET}"
def section(s): return f"\n{BOLD}{C}{'─'*4} {s} {'─'*4}{RESET}"
def warn(s):    return f"{Y}{BOLD}⚠ {s}{RESET}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH alias from ~/.ssh/config")
    parser.add_argument("--pull", action="store_true", help="Pull results and logs locally")
    args = parser.parse_args()

    remote_dir = "~/uyghurGPT"
    local_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_results = os.path.join(local_project, "results")

    print(section("Current jobs"))
    result = subprocess.run(["ssh", args.server, "squeue --me"], capture_output=True, text=True)
    print(result.stdout.strip() or warn("No jobs currently running."))

    print(section("Recent job history"))
    result = subprocess.run(
        ["ssh", args.server, "sacct --format=JobID,JobName,State,Elapsed,End -X"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().splitlines():
        if "RUNNING" in line:
            print(f"{G}{line}{RESET}")
        elif "FAILED" in line or "CANCELLED" in line:
            print(f"{R}{line}{RESET}")
        elif "TIMEOUT" in line:
            print(f"{Y}{line}{RESET}")
        elif "COMPLETED" in line:
            print(f"{G}{DIM}{line}{RESET}")
        else:
            print(line)

    print(section("Latest server logs"))
    logs = subprocess.run(
        ["ssh", args.server, f"ls -t {remote_dir}/results/slurm_*.out 2>/dev/null | head -5"],
        capture_output=True,
        text=True,
    )
    log_files = [x.strip() for x in logs.stdout.splitlines() if x.strip()]
    if not log_files:
        print(warn("No slurm logs found yet."))
    else:
        for lf in log_files:
            print(f"- {lf}")

    if args.pull:
        print(section("Pull results"))
        os.makedirs(local_results, exist_ok=True)
        pull = subprocess.run(
            ["rsync", "-avz", "--progress",
             "--exclude=*/checkpoints/",
             f"{args.server}:{remote_dir}/results/", f"{local_results}/"]
        )
        if pull.returncode == 0:
            print(ok(f"Pulled results to {local_results}"))
            print(info("View TensorBoard with: tensorboard --logdir results"))
        else:
            print(warn("Rsync pull failed."))


if __name__ == "__main__":
    main()
