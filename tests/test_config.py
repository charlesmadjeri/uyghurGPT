"""Sanity checks for the Experiment 1 config defaults.

These guard against accidentally checking in a "smoke" or "debug" default
that would silently nerf a real training run.
"""

from __future__ import annotations


def test_experiment1_defaults_are_production():
    from experiments.experiment_1.config import Experiment1Config

    cfg = Experiment1Config()
    assert cfg.model == "qwen"
    assert cfg.mix == 20
    assert cfg.epochs >= 1
    assert cfg.sample_count is None, "smoke-only sample_count must not be the default"
    assert cfg.lora_rank > 0 and cfg.lora_alpha > 0
    assert 0.0 < cfg.test_split_pct < 0.5, "test_split_pct must be in (0, 0.5)"
    assert cfg.eval_steps > 0
    assert cfg.early_stopping_patience >= 0


def test_experiment1_to_dict_is_json_safe():
    import json
    from experiments.experiment_1.config import Experiment1Config

    payload = Experiment1Config().to_dict()
    json.dumps(payload)
