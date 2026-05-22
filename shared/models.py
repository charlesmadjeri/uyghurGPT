"""Model identifiers and chat/QLoRA helpers shared across experiments."""

from __future__ import annotations

from transformers import AutoTokenizer

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


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
