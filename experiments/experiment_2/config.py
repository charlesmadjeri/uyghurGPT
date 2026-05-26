"""Configuration for experiment 2 (CUTE-Llama-P few-shot baseline; eval-only).

Mirrors ``experiments/experiment_0/config.py`` since both experiments are
eval-only with no preprocess/train stages. The only meaningful difference
is the evaluated variant: ``cute_llama_p`` (a base LM loaded from
``CMLI-NLP/CUTE-Llama`` / ``CUTE-Llama-Parallel`` in fp16) instead of the
zero-shot instruct pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Experiment2Config:
    """Eval-only config shared with ``shared/evaluation.py``."""

    experiment_id: int = 2
    # ``model`` is referenced by ``shared/evaluation`` helpers (e.g.
    # ``cfg.model_label`` when ``_find_adapter_path`` is consulted). It is
    # never used to load a model here: CUTE-Llama-P is loaded via the
    # ``cute_llama_p`` branch in ``load_eval_model``.
    model: str = "cute_llama_p"
    mix: int = 0
    epochs: int = 0
    sample_count: int | None = None
    results_root: str = "results"

    flores_max_samples: int | None = None
    wcm_max_samples: int | None = None
    ppl_max_samples: int = 1000

    eval_variants: tuple[str, ...] = field(default_factory=lambda: ("cute_llama_p",))

    @classmethod
    def from_namespace(cls, args) -> "Experiment2Config":
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
        # Used only by ``shared.evaluation._find_adapter_path`` when looking
        # for a LoRA adapter; no adapter exists for CUTE-Llama-P so the
        # value is functionally a no-op. We expose a sensible label so
        # downstream tooling that inspects ``cfg.model_label`` does not
        # crash.
        return "cute_llama_p"
