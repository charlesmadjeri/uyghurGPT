"""Experiment 1 orchestration (preprocess / train / eval stages)."""

from __future__ import annotations

from experiments.experiment_1.config import Experiment1Config
from utils import io
from utils.logging import stage


def run(args) -> None:
    cfg = Experiment1Config.from_namespace(args)
    run_id = io.resolve_run_id(args.run_id, cfg.results_root)
    root = io.ensure_run_layout(cfg.results_root, run_id, cfg.experiment_id)
    io.write_run_config(root, {"experiment": cfg.experiment_id, **cfg.to_dict(), "run_id": run_id})
    io.write_run_status(root, "started", {"mode": args.mode})

    if args.mode in ("preprocess", "all"):
        stage("Experiment 1 — preprocess")
        from shared import training

        training.preprocess(cfg, root)

    if args.mode in ("train", "all"):
        stage("Experiment 1 — train")
        from shared import training

        training.train(cfg, root)

    if args.mode in ("eval", "all"):
        stage("Experiment 1 — eval")
        from shared import evaluation

        evaluation.run_eval(cfg, root)
