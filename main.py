"""UyghurGPT — bilingual Uyghur/English LLM fine-tuning.

CLI entrypoint:
  --mode preflight              Day-1 sanity checks (shared/preflight.py)
  --experiment 0 --mode eval       Zero-shot baselines (experiments/experiment_0/)
  --experiment 1 --mode <stage>  Core Qwen Mix-20 QLoRA (experiments/experiment_1/)
  --experiment 2 --mode eval       CUTE-Llama-P few-shot baseline (experiments/experiment_2/)

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
        help=(
            "Experiment id. 0 = zero-shot baselines (qwen+llama, eval only). "
            "1 = core Qwen Mix-20 QLoRA (evaluates fine-tuned variant only). "
            "2 = CUTE-Llama-P few-shot baseline (eval only). "
            "Omit for legacy top-level modes."
        ),
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
        help="Preflight QLoRA memory check batch size (default 1; safe on ~24 GB MIG slice)",
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
    if args.experiment == 0:
        from experiments.experiment_0 import run as exp0

        exp0.run(args)
        return
    if args.experiment == 1:
        from experiments.experiment_1 import run as exp1

        exp1.run(args)
        return
    if args.experiment == 2:
        from experiments.experiment_2 import run as exp2

        exp2.run(args)
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
        "No --experiment set. Use --experiment 0 (zero-shot eval), "
        "--experiment 1 (fine-tune pipeline), --experiment 2 (CUTE-Llama-P "
        "few-shot baseline), or --mode preflight for Day-1 checks.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
