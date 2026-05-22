"""UyghurGPT — bilingual Uyghur/English LLM fine-tuning.

CLI entrypoint. Dispatches to one of five stages:
  --mode preflight     Day-1 sanity checks (tokenizer, QLoRA fit, data, baseline)
  --mode preprocess    Download CUTE-P + format as instruction pairs
  --mode train         LoRA fine-tune the chosen base model
  --mode eval          Evaluate the fine-tuned adapter on FLORES-200, WCM-v2, MiLiC-Eval
  --mode all           Run preprocess + train + eval sequentially (no preflight)

See docs/PROJECT.md for the full plan.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="UyghurGPT — fine-tune + evaluate")
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
        help="Existing run id to resume; if not set, a new run is created",
    )
    parser.add_argument("--results-root", default="results")

    # Preflight-only options
    parser.add_argument(
        "--check",
        default="all",
        help="Which Day-1 preflight check to run: 'all', a single id (e.g. '3'), or a comma list (e.g. '1,2,4,5')",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Per-device batch size for QLoRA memory checks (default 1 matches PROJECT.md on MIG 1g.10gb)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512,
        help="Max sequence length for QLoRA memory checks (default 512)",
    )
    return parser.parse_args()


def run_preflight(args):
    from shared import preflight
    preflight.run(args)


def run_preprocess(args):
    raise NotImplementedError("preprocess stage not implemented yet")


def run_train(args):
    raise NotImplementedError("train stage not implemented yet")


def run_eval(args):
    raise NotImplementedError("eval stage not implemented yet")


def main():
    args = parse_args()

    if args.mode == "preflight":
        run_preflight(args)
        return

    if args.mode in ("preprocess", "all"):
        run_preprocess(args)

    if args.mode in ("train", "all"):
        run_train(args)

    if args.mode in ("eval", "all"):
        run_eval(args)


if __name__ == "__main__":
    main()
