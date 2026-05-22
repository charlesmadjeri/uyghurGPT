"""UyghurGPT — bilingual Uyghur/English LLM fine-tuning.

CLI entrypoint:
  --mode preflight              Day-1 sanity checks (shared/preflight.py)
  --experiment 1 --mode <stage> Core experiment (experiments/experiment_1/)

Stages for experiments: preprocess | train | eval | all

See docs/PROJECT.md for the full plan.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="UyghurGPT — fine-tune + evaluate")
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        help="Experiment id (1 = core Qwen Mix-20 QLoRA). Omit for legacy top-level modes.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["preflight", "preprocess", "train", "eval", "all"],
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=["qwen", "llama"],
        help="Which base model to fine-tune",
    )
    parser.add_argument(
        "--mix",
        type=int,
        default=20,
        choices=[0, 10, 20, 50],
        help="Percentage of EN-only (FLAN) data mixed into training",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="If set, train/eval on a subsample (smoke testing)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run id to resume; if not set, a new run id is created",
    )
    parser.add_argument("--results-root", default="results")

    parser.add_argument(
        "--check",
        default="all",
        help="Preflight only: 'all', a single id (e.g. '3'), or comma list (e.g. '1,2,4,5')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Preflight QLoRA memory check batch size (default 1 for MIG 1g.10gb)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Preflight QLoRA memory check sequence length",
    )
    return parser.parse_args()


def run_preflight(args):
    from shared import preflight

    preflight.run(args)


def run_experiment(args):
    if args.experiment == 1:
        from experiments.experiment_1 import run as exp1

        exp1.run(args)
        return
    print(f"Unknown experiment id: {args.experiment}", file=sys.stderr)
    sys.exit(2)


def main():
    args = parse_args()

    if args.mode == "preflight":
        run_preflight(args)
        return

    if args.experiment is not None:
        run_experiment(args)
        return

    print(
        "No --experiment set. Use --experiment 1 for the core pipeline "
        "or --mode preflight for Day-1 checks.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
