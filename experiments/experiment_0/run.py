"""Experiment 0 orchestration — zero-shot eval only.

Skips ``preprocess`` and ``train``: there is nothing to train. Modes other
than ``eval`` / ``all`` are coerced to ``eval`` with a warning so that
existing CLIs / push.py wrappers keep working.
"""

from __future__ import annotations

from experiments.experiment_0.config import Experiment0Config
from utils import io
from utils.logging import stage


def run(args) -> None:
    cfg = Experiment0Config.from_namespace(args)
    run_id = io.resolve_run_id(args.run_id, cfg.results_root)
    root = io.ensure_run_layout(cfg.results_root, run_id, cfg.experiment_id)
    io.write_run_config(root, {"experiment": cfg.experiment_id, **cfg.to_dict(), "run_id": run_id})

    mode = getattr(args, "mode", "eval")
    if mode in ("preprocess", "train"):
        print(
            f"[exp0] mode={mode!r} is a no-op for the zero-shot experiment; "
            "running eval instead."
        )
        mode = "eval"

    io.write_run_status(root, "started", {"mode": mode})

    if mode in ("eval", "all"):
        stage("Experiment 0 — zero-shot eval (qwen + llama)")
        from shared import evaluation

        evaluation.run_eval(cfg, root)
