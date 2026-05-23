"""Model identifiers and chat/QLoRA helpers shared across experiments."""

from __future__ import annotations

import os

from transformers import AutoTokenizer

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def attn_implementation() -> str:
    """Pick the best available attention backend.

    Order: FlashAttention 2 (if installed) > SDPA (when CUDA is
    available) > eager. Set ``UYGHURGPT_ATTN`` to force a backend.
    """
    override = os.environ.get("UYGHURGPT_ATTN")
    if override:
        return override
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "sdpa"
    except Exception:
        pass
    return "eager"


def dtype_kwarg(value) -> dict:
    """Return the ``{'dtype': value}`` kwarg using the current transformers name.

    transformers >= 4.56 renamed ``torch_dtype`` to ``dtype`` and emits a
    DeprecationWarning on the old name. Older builds still expect
    ``torch_dtype``. We pick the right key at import time so every
    ``AutoModelForCausalLM.from_pretrained(..., **dtype_kwarg(...))`` call
    is silent on both APIs.
    """
    import inspect

    from transformers import AutoModelForCausalLM

    try:
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        key = "dtype" if "dtype" in params else "torch_dtype"
    except (TypeError, ValueError):
        key = "torch_dtype"
    return {key: value}


def model_id(choice: str) -> str:
    if choice not in MODEL_IDS:
        raise ValueError(f"Unknown model choice {choice!r}; expected one of {list(MODEL_IDS)}")
    return MODEL_IDS[choice]


def response_template(model_choice: str) -> str:
    if model_choice == "qwen":
        return "<|im_start|>assistant\n"
    if model_choice == "llama":
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    raise ValueError(f"Unknown model choice: {model_choice}")


def bnb_config():
    """4-bit NF4 quantization config; returns None when CUDA is unavailable.

    Lazy-imports torch + BitsAndBytesConfig so CPU-only flows (preprocess,
    preflight checks 1/4) can run without installing torch.
    """
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_choice: str):
    tok = AutoTokenizer.from_pretrained(model_id(model_choice), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
