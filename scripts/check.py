"""
Check UyghurGPT jobs and optionally pull results.

Usage:
  python scripts/check.py --server my-hpc                # status only
  python scripts/check.py --server my-hpc --logs         # fast TB sync (logs only)
  python scripts/check.py --server my-hpc --pull         # full pull (no checkpoints)
  python scripts/check.py --server my-hpc --pull --pull-checkpoints  # include weights

Reports:
  - Slurm job state (squeue, sacct)
  - Pipeline stage of the latest run (run_status.json)
  - Latest slurm_*.out path
"""

import argparse
import json
import os
import subprocess

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[96m"; C = "\033[96m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
def ok(s):      return f"{G}{BOLD}✔ {s}{RESET}"
def info(s):    return f"{B}{s}{RESET}"
def section(s): return f"\n{BOLD}{C}{'─'*4} {s} {'─'*4}{RESET}"
def warn(s):    return f"{Y}{BOLD}⚠ {s}{RESET}"


def _ssh(server, cmd):
    return subprocess.run(["ssh", server, cmd], capture_output=True, text=True)


def _print_pipeline_state(server, remote_dir):
    print(section("Pipeline stage (run_status.json)"))
    # Find the latest run, then the latest experiment_* inside it.
    find = _ssh(
        server,
        f"ls -1dt {remote_dir}/results/run_*/experiment_*/artifacts/run_status.json "
        f"2>/dev/null | head -1",
    )
    path = find.stdout.strip()
    if not path:
        # Legacy layout (no experiment_<N> subdir).
        find = _ssh(
            server,
            f"ls -1dt {remote_dir}/results/run_*/artifacts/run_status.json "
            f"2>/dev/null | head -1",
        )
        path = find.stdout.strip()
    if not path:
        print(warn("No run_status.json yet — pipeline not started."))
        return

    cat = _ssh(server, f"cat {path}")
    try:
        payload = json.loads(cat.stdout)
    except Exception:
        print(warn(f"Found {path} but could not parse it."))
        print(cat.stdout)
        return

    status = payload.get("status", "?")
    updated = payload.get("updated_at", "?")
    colour = G if status in ("trained", "evaluated") else (Y if status == "evaluating" else B)
    print(f"  path:    {path}")
    print(f"  status:  {colour}{BOLD}{status}{RESET}")
    print(f"  updated: {updated}")
    for k, v in payload.items():
        if k in ("status", "updated_at"):
            continue
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH alias from ~/.ssh/config")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull results and logs locally (excludes checkpoints by default)",
    )
    parser.add_argument(
        "--pull-checkpoints",
        action="store_true",
        help="Include checkpoint adapter weights when pulling (large)",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Fast TensorBoard sync: pulls only logs/ and run_status.json (skip checkpoints, datasets, eval json)",
    )
    args = parser.parse_args()

    remote_dir = "~/uyghurGPT"
    local_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_results = os.path.join(local_project, "results")

    print(section("Current jobs"))
    result = _ssh(args.server, "squeue --me")
    print(result.stdout.strip() or warn("No jobs currently running."))

    print(section("Recent job history"))
    result = _ssh(
        args.server,
        "sacct --format=JobID,JobName,State,Elapsed,End -X",
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

    _print_pipeline_state(args.server, remote_dir)

    print(section("Latest server logs"))
    logs = _ssh(
        args.server,
        f"ls -t {remote_dir}/results/slurm_*.out 2>/dev/null | head -5",
    )
    log_files = [x.strip() for x in logs.stdout.splitlines() if x.strip()]
    if not log_files:
        print(warn("No slurm logs found yet."))
    else:
        for lf in log_files:
            print(f"- {lf}")

    if args.logs:
        print(section("Pull TensorBoard logs + status"))
        os.makedirs(local_results, exist_ok=True)
        cmd = [
            "rsync", "-avz", "--progress",
            "--include=*/", "--include=logs/***",
            "--include=run_status.json", "--include=run_config.json",
            "--exclude=*",
            f"{args.server}:{remote_dir}/results/", f"{local_results}/",
        ]
        if subprocess.run(cmd).returncode == 0:
            print(ok(f"Logs synced to {local_results}"))
            print(info("View TensorBoard with: tensorboard --logdir results"))
        else:
            print(warn("Rsync pull failed."))

    if args.pull:
        print(section("Pull results"))
        os.makedirs(local_results, exist_ok=True)
        excludes = []
        if not args.pull_checkpoints:
            excludes.append("--exclude=*/checkpoints/")
        pull = subprocess.run(
            ["rsync", "-avz", "--progress", *excludes,
             f"{args.server}:{remote_dir}/results/", f"{local_results}/"]
        )
        if pull.returncode == 0:
            print(ok(f"Pulled results to {local_results}"))
            print(info("View TensorBoard with: tensorboard --logdir results"))
            if not args.pull_checkpoints:
                print(info("Tip: pass --pull-checkpoints to also fetch adapter weights."))
        else:
            print(warn("Rsync pull failed."))


if __name__ == "__main__":
    main()
