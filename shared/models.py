"""Model identifiers and chat/QLoRA helpers shared across experiments.

Heavy ML deps (``transformers``) are imported lazily inside the helpers
that need them so pure-Python utilities (``MODEL_IDS``, ``model_id``,
``model_load_kwargs``, ``attn_implementation``, ``response_template``) can
be inspected from unit tests without installing the full eval stack.
"""

from __future__ import annotations

import os

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    # CUTE-Llama-P (Zhuang & Sun, COLING 2025): Llama2-7B + ~155 K vocab
    # expansion + continued pretraining on CUTE-P. Lives in a subfolder of
    # the CMLI-NLP/CUTE-Llama HF repo (the same repo hosts the NP variant);
    # use ``model_load_kwargs("cute_llama_p")`` to get the matching
    # ``subfolder=`` and ``trust_remote_code=`` kwargs.
    "cute_llama_p": "CMLI-NLP/CUTE-Llama",
}

# Extra ``from_pretrained`` kwargs that must accompany ``MODEL_IDS[choice]``
# for HF to resolve the right snapshot / behaviour. Empty for the
# instruct-tuned models (Qwen, Llama-3.1) which sit at the repo root.
_MODEL_LOAD_KWARGS = {
    "cute_llama_p": {
        "subfolder": "CUTE-Llama-Parallel",
        "trust_remote_code": True,
    },
}


def model_load_kwargs(choice: str) -> dict:
    """Extra ``from_pretrained`` kwargs (subfolder, trust_remote_code, …)
    that must be passed alongside ``model_id(choice)``.
    """
    return dict(_MODEL_LOAD_KWARGS.get(choice, {}))

# TRL packing / padding-free training requires a flash-attn backend so
# flattened sequences do not cross-contaminate between packed samples.
PACKING_SAFE_ATTN = frozenset({
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
})


def attn_supports_packing(impl: str) -> bool:
    return impl in PACKING_SAFE_ATTN


def flash_attn_import_error() -> str | None:
    """Return None if ``flash_attn`` imports; otherwise a short reason."""
    try:
        import flash_attn  # noqa: F401

        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def attn_implementation() -> str:
    """Pick the best available attention backend.

    Order: FlashAttention 2 (if installed) > SDPA (when CUDA is
    available) > eager. Set ``UYGHURGPT_ATTN`` to force a backend.
    """
    override = os.environ.get("UYGHURGPT_ATTN")
    if override:
        return override
    if flash_attn_import_error() is None:
        return "flash_attention_2"
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


def load_tokenizer(model_choice: str, max_seq_length: int | None = None):
    from transformers import AutoTokenizer

    extra = model_load_kwargs(model_choice)
    tok = AutoTokenizer.from_pretrained(
        model_id(model_choice), use_fast=True, **extra
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if max_seq_length is not None:
        # Match SFTConfig max_seq_length so TRL tokenization truncates here
        # instead of warning on the model default (131072 for Qwen).
        tok.model_max_length = max_seq_length
    return tok


_MISSING = object()


def align_special_tokens(model, tokenizer) -> None:
    """Sync model.config + generation_config with the tokenizer's special tokens.

    Qwen2.5 ships ``pad_token=<|endoftext|>`` (id 151643) in the
    tokenizer but leaves ``model.config.pad_token_id=None`` and
    ``model.config.bos_token_id=None``. transformers >= 4.45 detects
    the drift on first forward/save and prints an "aligned accordingly"
    warning. Aligning explicitly here (including pushing ``None`` from the
    tokenizer back onto the model where Qwen ships a stale ``bos_token_id``)
    keeps the train log clean and the config persisted to checkpoints is
    consistent with how we actually tokenized.
    """
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        tok_value = getattr(tokenizer, attr, _MISSING)
        if tok_value is _MISSING:
            continue
        if getattr(model.config, attr, _MISSING) != tok_value:
            setattr(model.config, attr, tok_value)
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, attr, _MISSING) != tok_value:
            setattr(gen_cfg, attr, tok_value)
