"""Configuration for experiment 1 (core Qwen Mix-20 QLoRA)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Experiment1Config:
    """Hyperparameters aligned with docs/PROJECT.md §Training Configuration."""

    experiment_id: int = 1
    model: str = "qwen"
    mix: int = 20
    epochs: int = 3
    sample_count: int | None = None
    results_root: str = "results"

    lora_rank: int = 16
    lora_alpha: int = 32
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    flan_subset_size: int = 50_000
    flan_seed: int = 42
    preprocess_num_proc: int = 8

    # Held-out fraction from the CUTE-P+FLAN mix for in-loop overfit detection.
    # NOT the final "eval" — that's external benchmarks (FLORES+/WCM/C4 PPL).
    # Split is at parallel-pair level for CUTE-P (see shared/data.py).
    test_split_pct: float = 0.05
    eval_steps: int = 50
    # Stop training when eval_loss hasn't improved for N evaluations.
    # 0 disables early stopping (still loads best checkpoint at end).
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    flores_max_samples: int | None = None
    wcm_max_samples: int | None = None
    ppl_max_samples: int = 1000

    @classmethod
    def from_namespace(cls, args) -> "Experiment1Config":
        return cls(
            model=args.model,
            mix=args.mix,
            epochs=args.epochs,
            sample_count=args.sample_count,
            results_root=args.results_root,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def model_label(self) -> str:
        return f"{self.model}_mix{self.mix}"
