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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from shared.models import (
        align_special_tokens,
        bnb_config,
        dtype_kwarg,
        load_tokenizer,
        model_id,
    )

    mid = model_id(model_choice)
    quant = bnb_config()
    base = AutoModelForCausalLM.from_pretrained(
        mid,
        quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        **dtype_kwarg(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    )
    if adapter_path is not None:
        print(f"[eval] Loading adapter from {adapter_path}")
        base = PeftModel.from_pretrained(base, str(adapter_path))
    base.eval()
    tokenizer = load_tokenizer(model_choice)
    align_special_tokens(base, tokenizer)
    return base, tokenizer


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
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _corpus_scores(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "bleu": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
        "num_sentences": len(hypotheses),
    }


def load_flores_pairs(max_samples: int | None = None) -> tuple[list[str], list[str]]:
    """FLORES+ devtest (the public test set, 1012 sentences), id-aligned EN↔UG.

    Uses openlanguagedata/flores_plus per-language configs and joins on `id`,
    so no dataset script is required (datasets >= 2.20 refuses scripts) and
    the same source/splits are used by the preflight check 5 sanity test.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds_en = load_dataset(FLORES_REPO, FLORES_EN_CODE, split=FLORES_SPLIT, token=token)
    ds_ug = load_dataset(FLORES_REPO, FLORES_UG_CODE, split=FLORES_SPLIT, token=token)
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


def eval_flores(model, tokenizer, max_samples: int | None = None) -> dict:
    en, ug = load_flores_pairs(max_samples)
    en2ug_hyps, ug2en_hyps = [], []
    print(f"[eval] FLORES-200 n={len(en)} (EN→UG then UG→EN) ...")
    for i, (e, u) in enumerate(zip(en, ug)):
        en2ug_hyps.append(generate_translation(model, tokenizer, e, "English", "Uyghur"))
        ug2en_hyps.append(generate_translation(model, tokenizer, u, "Uyghur", "English"))
        if (i + 1) % 50 == 0:
            print(f"[eval]   {i + 1}/{len(en)}")
    return {
        "en2ug": _corpus_scores(en2ug_hyps, ug),
        "ug2en": _corpus_scores(ug2en_hyps, en),
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


@torch.inference_mode()
def _classify_uyghur(
    model,
    tokenizer,
    text: str,
    labels: list[str],
    exemplars: list[tuple[str, str]] | None = None,
) -> str:
    """Pick ``argmax_{l in labels} log P(l | chat_prompt(text))``.

    Constrained classification: the return value is always one of ``labels``,
    even if the model would otherwise emit free-form text. Deterministic
    under fixed weights (no sampling, no temperature).
    """
    if not labels:
        raise ValueError("labels must be a non-empty list")

    messages = _wcm_messages(text, labels, exemplars)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


def eval_wcm(model, tokenizer, max_samples: int | None = None) -> dict:
    ds, repo = _load_wcm_dataset(max_samples)
    text_col, label_col = _wcm_columns(ds)
    labels = sorted({str(x) for x in ds[label_col]})
    correct = 0
    total = len(ds)
    print(f"[eval] WCM-v2 ({repo}) n={total} ...")
    for row in ds:
        pred = _classify_uyghur(model, tokenizer, str(row[text_col]), labels)
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


ALL_EVAL_VARIANTS = ("qwen_zeroshot", "llama_zeroshot", "qwen_finetuned")


def _variant_specs(cfg, run_root: Path) -> list[dict]:
    """Build evaluation variant specs, optionally filtered by ``cfg.eval_variants``.

    ``cfg.eval_variants`` is an iterable of labels from ``ALL_EVAL_VARIANTS``.
    When unset (None) we keep the historical behaviour and run every variant
    whose model/adapter is available.
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
        specs.append({"label": "qwen_zeroshot", "model": "qwen", "adapter": None})
    if "llama_zeroshot" in wanted:
        specs.append({"label": "llama_zeroshot", "model": "llama", "adapter": None})
    if "qwen_finetuned" in wanted:
        adapter = _find_adapter_path(run_root, cfg.model_label)
        if adapter:
            specs.append(
                {"label": "qwen_finetuned", "model": "qwen", "adapter": adapter}
            )
        else:
            print("[eval] WARNING: no fine-tuned adapter found; skipping qwen_finetuned")
    return specs


def run_eval(cfg, run_root: Path) -> None:
    write_run_status(run_root, "evaluating")
    all_results = {}

    for spec in _variant_specs(cfg, run_root):
        label = spec["label"]
        print(f"\n[eval] === {label} ===")
        model, tokenizer = load_eval_model(spec["model"], spec.get("adapter"))
        variant = {"model": spec["model"], "adapter": str(spec["adapter"]) if spec["adapter"] else None}

        flores = eval_flores(model, tokenizer, cfg.flores_max_samples or cfg.sample_count)
        write_eval_artifact(run_root, f"flores_{label}", {"variant": variant, **flores})
        variant["flores"] = flores

        try:
            wcm = eval_wcm(model, tokenizer, cfg.wcm_max_samples or cfg.sample_count)
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
