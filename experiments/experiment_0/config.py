"""Configuration for experiment 0 (zero-shot baselines only — no training).

Zero-shot benchmark numbers for ``qwen_zeroshot`` and ``llama_zeroshot`` do not
change between fine-tuning runs, so they only need to be computed once and can
then be reused by every fine-tune experiment. This experiment isolates that
work: it runs the same FLORES-200 + WCM-v2 + C4 PPL harness as experiment 1
but skips ``preprocess`` and ``train`` and evaluates only the zero-shot
variants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Experiment0Config:
    """Eval-only config shared with ``shared/evaluation.py``."""

    experiment_id: int = 0
    # Kept for compatibility with shared/evaluation helpers that reference
    # ``cfg.model_label`` even when no adapter is loaded.
    model: str = "qwen"
    mix: int = 0
    epochs: int = 0
    sample_count: int | None = None
    results_root: str = "results"

    flores_max_samples: int | None = None
    wcm_max_samples: int | None = None
    ppl_max_samples: int = 1000

    eval_variants: tuple[str, ...] = field(
        default_factory=lambda: ("qwen_zeroshot", "llama_zeroshot")
    )

    @classmethod
    def from_namespace(cls, args) -> "Experiment0Config":
        kwargs: dict[str, Any] = {
            "results_root": getattr(args, "results_root", "results"),
        }
        if getattr(args, "sample_count", None) is not None:
            kwargs["sample_count"] = args.sample_count
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def model_label(self) -> str:
        # Never used (no adapter is loaded for zero-shot variants) but kept
        # so ``shared.evaluation`` helpers don't need to special-case us.
        return f"{self.model}_mix{self.mix}"
