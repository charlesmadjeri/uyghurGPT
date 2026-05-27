"""FLORES-200, WCM-v2, and English perplexity evaluation (docs/PROJECT.md §Evaluation).

Heavy ML deps (``sacrebleu``, ``datasets``, ``huggingface_hub``, ``peft``,
``transformers``) are imported lazily inside the helpers that need them so
that pure-CPU unit tests can exercise WCM scoring / prompt builders without
installing the full eval stack. ``torch`` is imported at module load because
the WCM scorer uses tensor operations and tests already require ``torch``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from utils.io import checkpoint_dir, write_eval_artifact, write_run_status

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_SPLIT = "devtest"
FLORES_EN_CODE = "eng_Latn"
FLORES_UG_CODE = "uig_Arab"
C4_REPO = "allenai/c4"
C4_CONFIG = "en"

WCM_REPO_DEFAULT = "hfl/wcm-v2"
WCM_UG_FILE = os.environ.get("WCM_V2_UG_FILE", "minority/ug.txt").strip() or "minority/ug.txt"
WCM_CANDIDATES = [
    os.environ.get("WCM_V2_DATASET", "").strip(),
    WCM_REPO_DEFAULT,
    "CMLI-NLP/WCM-v2",
    "wcm-v2",
]


def _find_adapter_path(run_root: Path, model_label: str) -> Path | None:
    base = checkpoint_dir(run_root, model_label)
    for candidate in (base / "final", base):
        if candidate.is_dir() and any(candidate.glob("adapter_*")):
            return candidate
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt in checkpoints:
        if any(ckpt.glob("adapter_*")):
            return ckpt
    return None


def load_eval_model(model_choice: str, adapter_path: Path | None = None):
    """Load a model + tokenizer for evaluation.

    Two paths:

    - ``cute_llama_p`` — base LM (Llama2-7B + vocab expansion). **No 4-bit
      quantization** (preflight check 5 confirmed NF4 produces degenerate
      output on this vocab-expanded base; fp16 is the validated path) and
      no LoRA adapter. Uses ``subfolder=CUTE-Llama-Parallel`` +
      ``trust_remote_code=True`` from ``shared.models.model_load_kwargs``.
    - ``qwen`` / ``llama`` — instruct models. 4-bit NF4 quantization via
      ``bnb_config()`` + optional LoRA adapter (used by ``qwen_finetuned``).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from shared.models import (
        align_special_tokens,
        bnb_config,
        dtype_kwarg,
        load_tokenizer,
        model_id,
        model_load_kwargs,
    )

    mid = model_id(model_choice)
    extra = model_load_kwargs(model_choice)

    if model_choice == "cute_llama_p":
        if adapter_path is not None:
            raise ValueError(
                "cute_llama_p is a base LM published as-is; loading a LoRA "
                "adapter on top is not supported."
            )
        base = AutoModelForCausalLM.from_pretrained(
            mid,
            device_map={"": 0} if torch.cuda.is_available() else None,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            **dtype_kwarg(torch.float16 if torch.cuda.is_available() else torch.float32),
            **extra,
        )
    else:
        quant = bnb_config()
        base = AutoModelForCausalLM.from_pretrained(
            mid,
            quantization_config=quant,
            device_map={"": 0} if torch.cuda.is_available() else None,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            **dtype_kwarg(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            **extra,
        )
        if adapter_path is not None:
            print(f"[eval] Loading adapter from {adapter_path}")
            base = PeftModel.from_pretrained(base, str(adapter_path))
    base.eval()
    tokenizer = load_tokenizer(model_choice)
    align_special_tokens(base, tokenizer)
    # CUTE-Llama-P ships ``generation_config.max_length=100000`` baked into
    # the checkpoint; transformers then warns once per ``model.generate``
    # call (≈ 2024× per FLORES eval) about the latent ``max_length`` vs
    # our explicit ``max_new_tokens``. Behaviour is correct either way —
    # the warning is noise — so clear ``max_length`` at load time. Same
    # for any other checkpoint that ships a stale default.
    gen_cfg = getattr(base, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.max_length = None
    return base, tokenizer


# Chat / role markers we want generation to stop on. Qwen's ChatML
# tokenizer maps each of these to a single special id; LLaMA-3.1's chat
# template uses ``<|eot_id|>`` / ``<|start_header_id|>``. ``_stop_token_ids``
# below resolves whichever ones the tokenizer actually knows about.
_STOP_TOKEN_STRINGS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
)

# Substrings we hard-trim from the decoded output after generation. Even
# with ``skip_special_tokens=True`` an adapter can learn to emit the
# *literal text* "<|im_end|>", or a fresh "\nassistant\n" turn header, when
# it has been over-trained on the chat template. These markers must never
# leak into a translation hypothesis or chrF / BLEU collapses (this is the
# root cause of the UG→EN regression diagnosed in
# ``docs/tasks/03_ug2en_decoding_fix.md``).
_POST_DECODE_TRIM_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "\nassistant\n",
    "\nsystem\n",
    "\nuser\n",
)


def _stop_token_ids(tokenizer) -> list[int]:
    """List of stop token ids derived from the tokenizer's special vocabulary.

    Always includes ``tokenizer.eos_token_id`` when present. Adds any chat
    marker (``<|im_end|>``, ``<|eot_id|>``, …) the tokenizer knows about.
    Order is stable for reproducibility and the list is deduplicated; the
    return is safe to hand to ``model.generate(eos_token_id=...)`` which
    accepts either an int or a list of ints from transformers >= 4.45.
    """
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        ids.append(eos)
    for marker in _STOP_TOKEN_STRINGS:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids


def _clean_translation_output(text: str) -> str:
    """Hard-trim chat markers and assistant-turn headers from a hypothesis.

    Splits on the **first** occurrence of any marker in
    ``_POST_DECODE_TRIM_MARKERS`` and keeps the prefix. Idempotent; safe
    on outputs that already lack any marker.
    """
    earliest = len(text)
    for marker in _POST_DECODE_TRIM_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest].strip()


# UG→EN chat decoding: greedy FT outputs on Slurm 2766 showed a stable
# "The 2 1 1 1 …" loop to max_new_tokens. Base-LM few-shot already uses
# repetition_penalty=1.15 in generate_translation_fewshot; apply the same
# knobs only when generating English so EN→UG (Uyghur script) is unchanged.
_UG2EN_REPETITION_PENALTY = 1.15
_UG2EN_NO_REPEAT_NGRAM_SIZE = 4


def _is_english_target(tgt_lang: str) -> bool:
    return tgt_lang.strip().lower() == "english"


def _chat_generate_extra_kwargs(tgt_lang: str) -> dict:
    """Extra ``model.generate`` kwargs for chat-template translation."""
    if _is_english_target(tgt_lang):
        return {
            "repetition_penalty": _UG2EN_REPETITION_PENALTY,
            "no_repeat_ngram_size": _UG2EN_NO_REPEAT_NGRAM_SIZE,
        }
    return {}


@torch.inference_mode()
def generate_translation(model, tokenizer, source: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful bilingual assistant. "
                f"Translate the {src_lang} input to {tgt_lang}."
            ),
        },
        {"role": "user", "content": source},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    stop_ids = _stop_token_ids(tokenizer)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids if stop_ids else tokenizer.eos_token_id,
        **_chat_generate_extra_kwargs(tgt_lang),
    )
    out = model.generate(**inputs, **gen_kwargs)
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _clean_translation_output(decoded)


# ──────────────────────────────────────────────────────────────────────────
# Few-shot continuation decoding (for base LMs without a chat template)
# ──────────────────────────────────────────────────────────────────────────

# Boundary literals that mark the *next* prompted continuation in a
# few-shot continuation prompt. A base LM (e.g. CUTE-Llama-P) will happily
# keep generating past its own answer into a fabricated next exemplar;
# truncating at the first occurrence of any of these strings is the
# documented preflight check 5 strategy (``shared/preflight.py::_generate_one``).
_FEWSHOT_BOUNDARY_MARKERS = (
    "\nEnglish:",
    "\nUyghur:",
    "\nChinese:",
    "\n\n",
)


def build_fewshot_translation_prompt(
    source: str,
    src_lang: str,
    tgt_lang: str,
    exemplars: list[tuple[str, str]],
) -> str:
    """k-shot translation continuation prompt.

    Format mirrors ``shared/preflight.py::_build_fewshot_prompt`` (validated
    by preflight check 5):

        {src_lang}: <exemplar source 1>
        {tgt_lang}: <exemplar target 1>

        {src_lang}: <exemplar source 2>
        {tgt_lang}: <exemplar target 2>

        ...

        {src_lang}: <source>
        {tgt_lang}:

    Each exemplar string is truncated to 400 characters to keep the prompt
    well under the model's context (CUTE-Llama-P inherits Llama 2's 4096 cap).
    """
    parts: list[str] = []
    for ex_src, ex_tgt in exemplars:
        parts.append(
            f"{src_lang}: {ex_src[:400].strip()}\n"
            f"{tgt_lang}: {ex_tgt[:400].strip()}"
        )
    parts.append(f"{src_lang}: {source}\n{tgt_lang}:")
    return "\n\n".join(parts)


def _trim_at_fewshot_boundary(text: str) -> str:
    """Trim a few-shot continuation output at the first exemplar boundary.

    Idempotent. Splits on the *earliest* occurrence of any string in
    ``_FEWSHOT_BOUNDARY_MARKERS`` so a single helper covers all of
    ``\\nEnglish:`` / ``\\nUyghur:`` / ``\\nChinese:`` / ``\\n\\n``.
    """
    earliest = len(text)
    for marker in _FEWSHOT_BOUNDARY_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest].strip()


@torch.inference_mode()
def generate_translation_fewshot(
    model,
    tokenizer,
    source: str,
    src_lang: str,
    tgt_lang: str,
    exemplars: list[tuple[str, str]],
    max_new_tokens: int = 200,
    repetition_penalty: float = 1.15,
) -> str:
    """Few-shot continuation translation for base LMs (CUTE-Llama-P et al.).

    Stop strategy (validated by preflight check 5 in ``shared/preflight.py``):

    1. ``eos_token_id = tokenizer.eos_token_id`` — Llama 2's ``</s>`` survives
       the vocab expansion.
    2. ``repetition_penalty = 1.15`` — base LMs loop without it on few-shot
       prompts; this value preserves Uyghur Arabic-script fluency.
    3. **Post-decode hard-trim at the first exemplar boundary** —
       see ``_trim_at_fewshot_boundary``. This is the equivalent of the
       chat-marker hard-trim ``_clean_translation_output`` does for
       instruct models.
    """
    prompt = build_fewshot_translation_prompt(source, src_lang, tgt_lang, exemplars)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _trim_at_fewshot_boundary(decoded)


def _corpus_scores(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "bleu": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
        "num_sentences": len(hypotheses),
    }


def load_flores_pairs(
    max_samples: int | None = None,
    split: str = FLORES_SPLIT,
) -> tuple[list[str], list[str]]:
    """Id-aligned EN↔UG sentence pairs from one FLORES+ split.

    Defaults to ``devtest`` (the 1012-sentence public test set used by the
    final eval). Pass ``split="dev"`` for the 997-sentence exemplar pool —
    that is what ``load_flores_dev_exemplars`` reads to supply few-shot
    examples to base-LM variants (CUTE-Llama-P).

    Uses ``openlanguagedata/flores_plus`` per-language configs and joins on
    ``id`` so no dataset script is required (datasets >= 2.20 refuses
    scripts) and the same source is shared with the preflight check.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds_en = load_dataset(FLORES_REPO, FLORES_EN_CODE, split=split, token=token)
    ds_ug = load_dataset(FLORES_REPO, FLORES_UG_CODE, split=split, token=token)
    ug_by_id = {str(row["id"]): row["text"].strip() for row in ds_ug}
    en, ug = [], []
    for row in ds_en:
        rid = str(row["id"])
        if rid in ug_by_id:
            en.append(row["text"].strip())
            ug.append(ug_by_id[rid])
    if max_samples is not None:
        en, ug = en[:max_samples], ug[:max_samples]
    return en, ug


def load_flores_dev_exemplars(
    k: int = 3,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(en2ug_exemplars, ug2en_exemplars)`` from the FLORES+ ``dev`` split.

    Each list is ``k`` ``(source, target)`` pairs aligned by FLORES id. The
    ``dev`` split is disjoint from ``devtest`` (test set), so using its first
    ``k`` items as exemplars introduces no leakage.
    """
    en_dev, ug_dev = load_flores_pairs(max_samples=k, split="dev")
    en2ug = list(zip(en_dev, ug_dev))
    ug2en = list(zip(ug_dev, en_dev))
    return en2ug, ug2en


def eval_flores(
    model,
    tokenizer,
    max_samples: int | None = None,
    *,
    prompt_style: str = "chat",
    fewshot_k: int = 3,
) -> dict:
    """Evaluate FLORES+ devtest EN↔UG with the requested prompt style.

    ``prompt_style="chat"`` (default) — chat-template + assistant-turn
    decoding (Qwen, LLaMA-3.1, fine-tuned Qwen).

    ``prompt_style="fewshot"`` — base-LM-friendly few-shot continuation
    (CUTE-Llama-P). Exemplars come from the FLORES+ ``dev`` split (disjoint
    from ``devtest``) per ``load_flores_dev_exemplars``.
    """
    en, ug = load_flores_pairs(max_samples)
    if prompt_style == "fewshot":
        en2ug_ex, ug2en_ex = load_flores_dev_exemplars(k=fewshot_k)
        print(
            f"[eval] FLORES-200 n={len(en)} few-shot k={fewshot_k} "
            f"(EN→UG then UG→EN) ..."
        )
    elif prompt_style == "chat":
        en2ug_ex, ug2en_ex = ([], [])
        print(f"[eval] FLORES-200 n={len(en)} chat (EN→UG then UG→EN) ...")
    else:
        raise ValueError(
            f"Unknown FLORES prompt_style {prompt_style!r}; "
            "expected 'chat' or 'fewshot'"
        )

    en2ug_hyps, ug2en_hyps = [], []
    for i, (e, u) in enumerate(zip(en, ug)):
        if prompt_style == "fewshot":
            en2ug_hyps.append(
                generate_translation_fewshot(
                    model, tokenizer, e, "English", "Uyghur", en2ug_ex
                )
            )
            ug2en_hyps.append(
                generate_translation_fewshot(
                    model, tokenizer, u, "Uyghur", "English", ug2en_ex
                )
            )
        else:
            en2ug_hyps.append(
                generate_translation(model, tokenizer, e, "English", "Uyghur")
            )
            ug2en_hyps.append(
                generate_translation(model, tokenizer, u, "Uyghur", "English")
            )
        if (i + 1) % 50 == 0:
            print(f"[eval]   {i + 1}/{len(en)}")
    return {
        "en2ug": _corpus_scores(en2ug_hyps, ug),
        "ug2en": _corpus_scores(ug2en_hyps, en),
        "prompt_style": prompt_style,
        "fewshot_k": fewshot_k if prompt_style == "fewshot" else 0,
    }


def _parse_wcm_tab_file(path: Path):
    from datasets import Dataset

    rows: dict[str, list[str]] = {"text": [], "label": []}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"WCM line missing tab separator: {line[:80]!r}...")
            text, label = line.rsplit("\t", 1)
            rows["text"].append(text.strip())
            rows["label"].append(label.strip())
    if not rows["text"]:
        raise ValueError(f"WCM file is empty: {path}")
    return Dataset.from_dict(rows)


def _load_wcm_dataset(max_samples: int | None):
    """Uyghur WCM-v2 eval uses minority/ug.txt (text\\tlabel), not HF test split."""
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    repos = [r for r in WCM_CANDIDATES if r]
    if not repos:
        repos = [WCM_REPO_DEFAULT]
    for repo in repos:
        try:
            path = hf_hub_download(
                repo,
                WCM_UG_FILE,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
            ds = _parse_wcm_tab_file(Path(path))
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))
            return ds, f"{repo}:{WCM_UG_FILE}"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load WCM-v2 Uyghur split from {repos}: {last_err}")


def _wcm_columns(ds) -> tuple[str, str]:
    cols = set(ds.column_names)
    text_keys = ["text", "content", "sentence", "uyghur", "input"]
    label_keys = ["label", "labels", "category", "class"]
    text_col = next((c for c in text_keys if c in cols), None)
    label_col = next((c for c in label_keys if c in cols), None)
    if text_col and label_col:
        return text_col, label_col
    if len(cols) >= 2:
        return ds.column_names[0], ds.column_names[1]
    raise ValueError(f"Unrecognized WCM-v2 schema: {ds.column_names}")


def _wcm_messages(
    text: str,
    labels: list[str],
    exemplars: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Chat-template messages for WCM-v2 Uyghur classification.

    The user turn ends with a literal ``Label:`` cue so the *next* token the
    model is asked to predict is the label. ``_classify_uyghur`` then scores
    each candidate label by its (joint) log-likelihood under teacher forcing
    instead of generating freely, which guarantees the prediction is one of
    ``labels`` and removes the substring-match fallback that produced the
    below-chance numbers in ``run_20260525_143722`` (see
    ``docs/PROJECT_RESULTS.md`` 2026-05-26 §Analysis).
    """
    label_str = ", ".join(labels[:20])
    sections: list[str] = [
        f"Classify the following Uyghur text. Reply with exactly one label "
        f"from this set: {{{label_str}}}.",
    ]
    if exemplars:
        sections.append("Examples:")
        for ex_text, ex_label in exemplars:
            sections.append(f"Text: {ex_text}\nLabel: {ex_label}")
    sections.append(f"Text: {text}\nLabel:")
    return [
        {
            "role": "system",
            "content": (
                "You classify Uyghur text. Reply with exactly one label "
                "from the provided set."
            ),
        },
        {"role": "user", "content": "\n\n".join(sections)},
    ]


@torch.inference_mode()
def _score_label_logprob(model, prompt_ids, label_ids) -> float:
    """Joint log-likelihood of ``label_ids`` continuing ``prompt_ids``.

    ``prompt_ids`` and ``label_ids`` are both ``[1, T]`` tensors already on
    the model's device. Uses one forward pass over ``cat(prompt, label)``
    and gathers the per-token log-softmax of the label positions.
    """
    full = torch.cat([prompt_ids, label_ids], dim=1)
    out = model(input_ids=full)
    plen = prompt_ids.shape[1]
    n_label = label_ids.shape[1]
    if n_label <= 0:
        return float("-inf")
    label_logits = out.logits[0, plen - 1 : plen - 1 + n_label]
    log_probs = torch.log_softmax(label_logits, dim=-1)
    target = full[0, plen : plen + n_label].unsqueeze(-1)
    return log_probs.gather(-1, target).squeeze(-1).sum().item()


def build_wcm_base_lm_prompt(
    text: str,
    labels: list[str],
    exemplars: list[tuple[str, str]] | None = None,
) -> str:
    """Flat WCM-v2 classification prompt for base LMs (no chat template).

    Mirrors the structure of ``_wcm_messages`` but emits a single string
    that can be tokenized directly. Ends on a literal ``Label:`` cue so
    the next-token log-likelihood scoring path is the same on both
    instruct and base LMs.
    """
    label_str = ", ".join(labels[:20])
    sections: list[str] = [
        f"Classify the following Uyghur text. Reply with exactly one label "
        f"from this set: {{{label_str}}}.",
    ]
    if exemplars:
        sections.append("Examples:")
        for ex_text, ex_label in exemplars:
            sections.append(f"Text: {ex_text}\nLabel: {ex_label}")
    sections.append(f"Text: {text}\nLabel:")
    return "\n\n".join(sections)


@torch.inference_mode()
def _classify_uyghur(
    model,
    tokenizer,
    text: str,
    labels: list[str],
    exemplars: list[tuple[str, str]] | None = None,
    *,
    prompt_style: str = "chat",
) -> str:
    """Pick ``argmax_{l in labels} log P(l | prompt(text))``.

    Constrained classification: the return value is always one of ``labels``,
    even if the model would otherwise emit free-form text. Deterministic
    under fixed weights (no sampling, no temperature).

    ``prompt_style="chat"`` (default) routes through
    ``tokenizer.apply_chat_template`` (Qwen / LLaMA-3.1 / fine-tuned Qwen).
    ``prompt_style="base_lm"`` builds the flat string from
    ``build_wcm_base_lm_prompt`` (CUTE-Llama-P).
    """
    if not labels:
        raise ValueError("labels must be a non-empty list")

    if prompt_style == "chat":
        messages = _wcm_messages(text, labels, exemplars)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif prompt_style == "base_lm":
        prompt = build_wcm_base_lm_prompt(text, labels, exemplars)
    else:
        raise ValueError(
            f"Unknown WCM prompt_style {prompt_style!r}; "
            "expected 'chat' or 'base_lm'"
        )

    prompt_enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    prompt_ids = prompt_enc["input_ids"].to(model.device)

    best_label = labels[0]
    best_score = float("-inf")
    for label in labels:
        label_enc = tokenizer(label, return_tensors="pt", add_special_tokens=False)
        label_ids = label_enc["input_ids"].to(model.device)
        score = _score_label_logprob(model, prompt_ids, label_ids)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def eval_wcm(
    model,
    tokenizer,
    max_samples: int | None = None,
    *,
    prompt_style: str = "chat",
) -> dict:
    ds, repo = _load_wcm_dataset(max_samples)
    text_col, label_col = _wcm_columns(ds)
    labels = sorted({str(x) for x in ds[label_col]})
    correct = 0
    total = len(ds)
    print(f"[eval] WCM-v2 ({repo}) n={total} style={prompt_style} ...")
    for row in ds:
        pred = _classify_uyghur(
            model, tokenizer, str(row[text_col]), labels,
            prompt_style=prompt_style,
        )
        if pred == str(row[label_col]):
            correct += 1
    acc = correct / max(total, 1)
    return {
        "dataset": repo,
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "text_column": text_col,
        "label_column": label_col,
        "prompt_style": prompt_style,
    }


def load_c4_snippets(max_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(C4_REPO, C4_CONFIG, split="validation", streaming=True)
    texts = []
    for row in ds:
        texts.append(row["text"])
        if len(texts) >= max_samples:
            break
    return texts


@torch.inference_mode()
def eval_english_perplexity(model, tokenizer, max_samples: int = 1000) -> dict:
    texts = load_c4_snippets(max_samples)
    nlls = []
    print(f"[eval] English PPL on C4 n={len(texts)} ...")
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        nlls.append(out.loss.item())
        if (i + 1) % 200 == 0:
            print(f"[eval]   {i + 1}/{len(texts)}")
    mean_nll = sum(nlls) / len(nlls)
    return {
        "perplexity": round(math.exp(mean_nll), 4),
        "mean_nll": round(mean_nll, 4),
        "num_samples": len(texts),
        "dataset": f"{C4_REPO}/{C4_CONFIG}",
    }


ALL_EVAL_VARIANTS = (
    "qwen_zeroshot",
    "llama_zeroshot",
    "qwen_finetuned",
    "cute_llama_p",
)


def _variant_specs(cfg, run_root: Path) -> list[dict]:
    """Build evaluation variant specs, optionally filtered by ``cfg.eval_variants``.

    ``cfg.eval_variants`` is an iterable of labels from ``ALL_EVAL_VARIANTS``.
    When unset (None) we keep the historical behaviour and run every variant
    whose model/adapter is available.

    Each spec records the ``prompt_style`` to use for FLORES and WCM:

    - ``"chat"`` — chat-template prompt + chat-marker stop tokens
      (Qwen, LLaMA-3.1, fine-tuned Qwen).
    - ``"fewshot"`` — few-shot continuation prompt + exemplar-boundary
      hard-trim (CUTE-Llama-P; see ``generate_translation_fewshot``).
    """
    requested = getattr(cfg, "eval_variants", None)
    if requested is None:
        wanted = set(ALL_EVAL_VARIANTS)
    else:
        wanted = {v for v in requested}
        unknown = wanted - set(ALL_EVAL_VARIANTS)
        if unknown:
            raise ValueError(
                f"Unknown eval_variants {sorted(unknown)}; expected subset of {ALL_EVAL_VARIANTS}"
            )

    specs: list[dict] = []
    if "qwen_zeroshot" in wanted:
        specs.append({
            "label": "qwen_zeroshot", "model": "qwen", "adapter": None,
            "flores_prompt_style": "chat", "wcm_prompt_style": "chat",
        })
    if "llama_zeroshot" in wanted:
        specs.append({
            "label": "llama_zeroshot", "model": "llama", "adapter": None,
            "flores_prompt_style": "chat", "wcm_prompt_style": "chat",
        })
    if "qwen_finetuned" in wanted:
        adapter = _find_adapter_path(run_root, cfg.model_label)
        if adapter:
            specs.append({
                "label": "qwen_finetuned", "model": "qwen", "adapter": adapter,
                "flores_prompt_style": "chat", "wcm_prompt_style": "chat",
            })
        else:
            print("[eval] WARNING: no fine-tuned adapter found; skipping qwen_finetuned")
    if "cute_llama_p" in wanted:
        specs.append({
            "label": "cute_llama_p", "model": "cute_llama_p", "adapter": None,
            "flores_prompt_style": "fewshot", "wcm_prompt_style": "base_lm",
        })
    return specs


def run_eval(cfg, run_root: Path) -> None:
    write_run_status(run_root, "evaluating")
    all_results = {}

    for spec in _variant_specs(cfg, run_root):
        label = spec["label"]
        flores_style = spec.get("flores_prompt_style", "chat")
        wcm_style = spec.get("wcm_prompt_style", "chat")
        print(f"\n[eval] === {label} (flores={flores_style}, wcm={wcm_style}) ===")
        model, tokenizer = load_eval_model(spec["model"], spec.get("adapter"))
        variant = {
            "model": spec["model"],
            "adapter": str(spec["adapter"]) if spec["adapter"] else None,
        }

        flores = eval_flores(
            model, tokenizer,
            cfg.flores_max_samples or cfg.sample_count,
            prompt_style=flores_style,
        )
        write_eval_artifact(run_root, f"flores_{label}", {"variant": variant, **flores})
        variant["flores"] = flores

        try:
            wcm = eval_wcm(
                model, tokenizer,
                cfg.wcm_max_samples or cfg.sample_count,
                prompt_style=wcm_style,
            )
            write_eval_artifact(run_root, f"wcm_{label}", {"variant": variant, **wcm})
            variant["wcm"] = wcm
        except Exception as e:
            variant["wcm"] = {"status": "ERROR", "error": str(e)}
            write_eval_artifact(run_root, f"wcm_{label}", variant)
            print(f"[eval] WCM-v2 failed for {label}: {e}")

        ppl = eval_english_perplexity(model, tokenizer, cfg.ppl_max_samples)
        write_eval_artifact(run_root, f"ppl_{label}", {"variant": variant, **ppl})
        variant["perplexity"] = ppl

        all_results[label] = variant
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_eval_artifact(run_root, "summary", all_results)
    write_run_status(run_root, "evaluated", {"variants": list(all_results)})
    print(f"\n[eval] Summary written to {run_root / 'artifacts'}")
