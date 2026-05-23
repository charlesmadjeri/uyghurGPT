"""Completion-only loss masking collator (vendored for TRL >= 1.0).

TRL 1.4 removed the top-level ``DataCollatorForCompletionOnlyLM`` export.
We keep a minimal copy here so train and eval_loss use the same label
masking (assistant tokens only) without relying on
``assistant_only_loss=True`` on conversational data, which can log
``eval_loss=NaN`` on some batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling


def _find_subsequence(sequence: list[int], subseq: list[int]) -> int:
    """Return start index of *subseq* in *sequence*, or -1."""
    if not subseq:
        return -1
    n = len(subseq)
    for i in range(len(sequence) - n + 1):
        if sequence[i : i + n] == subseq:
            return i
    return -1


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Mask labels to the assistant completion (response template onward)."""

    response_template: str | list[int]
    instruction_template: str | list[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tokenizer is None:
            raise ValueError("DataCollatorForCompletionOnlyLM requires a tokenizer.")

        if isinstance(self.response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = list(self.response_template)

        if self.instruction_template is None:
            self.instruction_token_ids = None
        elif isinstance(self.instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            self.instruction_token_ids = list(self.instruction_template)

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()

        for i in range(labels.shape[0]):
            seq = labels[i].tolist()
            start = _find_subsequence(seq, self.response_token_ids)
            if start < 0:
                continue
            if self.instruction_token_ids is not None:
                inst = _find_subsequence(seq, self.instruction_token_ids)
                if inst >= 0:
                    start = inst + len(self.instruction_token_ids)
            labels[i, :start] = -100

        batch["labels"] = labels
        return batch
