"""Experiment 1 orchestration (preprocess / train / eval stages)."""

from __future__ import annotations

from experiments.experiment_1.config import Experiment1Config
from utils import io


def run(args) -> None:
    """Dispatch stages for experiment 1. Training/eval filled in later commits."""
    cfg = Experiment1Config.from_namespace(args)
    run_id = io.resolve_run_id(args.run_id, cfg.results_root)
    root = io.ensure_run_layout(cfg.results_root, run_id)
    io.write_run_config(root, {"experiment": cfg.experiment_id, **cfg.to_dict(), "run_id": run_id})

    if args.mode in ("preprocess", "all"):
        raise NotImplementedError(
            "Experiment 1 preprocess is implemented in the training commit."
        )

    if args.mode in ("train", "all"):
        raise NotImplementedError(
            "Experiment 1 training is implemented in the training commit."
        )

    if args.mode in ("eval", "all"):
        raise NotImplementedError(
            "Experiment 1 evaluation is implemented in the evaluation commit."
        )
