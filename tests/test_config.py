"""Sanity checks for the Experiment 1 config defaults.

These guard against accidentally checking in a "smoke" or "debug" default
that would silently nerf a real training run.
"""

from __future__ import annotations

import pytest


def test_experiment1_defaults_are_production():
    from experiments.experiment_1.config import Experiment1Config

    cfg = Experiment1Config()
    assert cfg.model == "qwen"
    assert cfg.mix == 20
    assert cfg.epochs >= 1
    # sample_count may be capped (default 100k since we can't fit the full
    # 934k-pair corpus in the 5-day Slurm walltime even with packing+FA2),
    # but it must never be a smoke-test value that would silently nerf a
    # real training run.
    assert cfg.sample_count is None or cfg.sample_count >= 10_000, (
        "sample_count default is a smoke-test value"
    )
    assert cfg.lora_rank > 0 and cfg.lora_alpha > 0
    assert 0.0 < cfg.test_split_pct < 0.5, "test_split_pct must be in (0, 0.5)"
    assert cfg.eval_steps > 0
    assert cfg.early_stopping_patience >= 0
    assert cfg.enable_packing is True, "throughput knob regressed"


def test_experiment1_to_dict_is_json_safe():
    import json
    from experiments.experiment_1.config import Experiment1Config

    payload = Experiment1Config().to_dict()
    json.dumps(payload)


def test_resolve_completion_only_collator_always_available():
    pytest.importorskip("transformers")
    from shared.training import _resolve_completion_only_collator

    cls = _resolve_completion_only_collator()
    assert cls is not None


def test_completion_collator_masks_prompt_labels():
    pytest.importorskip("transformers")
    from shared.completion_collator import DataCollatorForCompletionOnlyLM

    class _Tok:
        pad_token = "<pad>"
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "<|im_start|>assistant\n":
                return [99]
            return [1, 2, 3]

        def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
            import torch

            ids = [[1, 2, 3, 99, 4, 5, 0], [1, 2, 99, 6, 7, 0, 0]]
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor([[1] * 7, [1] * 7]),
            }

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=_Tok(),
        mlm=False,
    )
    batch = collator([{"text": "a"}, {"text": "b"}])
    assert batch["labels"][0, 0].item() == -100
    assert batch["labels"][0, 3].item() == 99
    assert batch["labels"][1, 2].item() == 99
