"""Experiment 2 orchestration — CUTE-Llama-P few-shot baseline (eval-only).

Skips ``preprocess`` and ``train``: there is nothing to train. Modes other
than ``eval`` / ``all`` are coerced to ``eval`` with a warning so existing
CLIs / push.py wrappers keep working — same pattern as experiment 0.
"""

from __future__ import annotations

from experiments.experiment_2.config import Experiment2Config
from utils import io
from utils.logging import stage


def run(args) -> None:
    cfg = Experiment2Config.from_namespace(args)
    run_id = io.resolve_run_id(args.run_id, cfg.results_root)
    root = io.ensure_run_layout(cfg.results_root, run_id, cfg.experiment_id)
    io.write_run_config(root, {"experiment": cfg.experiment_id, **cfg.to_dict(), "run_id": run_id})

    mode = getattr(args, "mode", "eval")
    if mode in ("preprocess", "train"):
        print(
            f"[exp2] mode={mode!r} is a no-op for the CUTE-Llama-P baseline; "
            "running eval instead."
        )
        mode = "eval"

    io.write_run_status(root, "started", {"mode": mode})

    if mode in ("eval", "all"):
        stage("Experiment 2 — CUTE-Llama-P few-shot baseline (eval)")
        from shared import evaluation

        evaluation.run_eval(cfg, root)
