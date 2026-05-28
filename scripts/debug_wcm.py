#!/usr/bin/env python3
"""Inspect WCM-v2 classification outputs and detect majority-class collapse.

Mix-50 (Slurm 2770) reported ``qwen_finetuned`` WCM accuracy **81.00 %**
(243/300) vs **21.00 %** for Mix-20 (Slurm 2744). The Uyghur file's
majority class covers **85.3 %** of rows (256/300 of label ``1``), so
that jump is suspiciously close to the always-predict-majority floor.

This script reproduces the constrained log-likelihood scoring path
(``shared/evaluation._classify_uyghur``) on a fixed model + label set
and records, **per row**:

- gold + predicted label
- full per-label joint log-prob (the distribution the argmax picked from)
- top-1 minus top-2 margin (decision confidence)
- a short text snippet for human-readable rows

It then summarises:

- prediction distribution vs gold distribution
- confusion matrix (gold × pred)
- per-label precision / recall / F1
- accuracy on majority-gold rows vs non-majority-gold rows
- "always-predict-majority" baseline accuracy (sanity floor)
- a `majority_class_collapse_detected` flag (pred share of majority class ≥ 95 %)

Defaults target the Mix-50 adapter. Pass ``--no-adapter`` for zero-shot.
Pass ``--compare-zeroshot`` to score zero-shot qwen on the same rows in
the same job (single model swap; ~6 min extra).

Examples::

    python scripts/debug_wcm.py
    python scripts/debug_wcm.py --compare-zeroshot
    python scripts/debug_wcm.py --no-adapter
    python scripts/debug_wcm.py --adapter path/to/final -n 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ADAPTER = (
    REPO_ROOT
    / "results/run_20260527_185416/experiment_1/checkpoints/qwen_mix50/final"
)


def _per_label_logprobs(model, tokenizer, prompt: str, labels: list[str]) -> dict[str, float]:
    """Joint log-prob of each candidate ``label`` continuing ``prompt``.

    Mirrors the inner loop of ``shared.evaluation._classify_uyghur`` but
    keeps the **full distribution** instead of returning only argmax.
    """
    from shared.evaluation import _score_label_logprob

    prompt_enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    prompt_ids = prompt_enc["input_ids"].to(model.device)
    scores: dict[str, float] = {}
    for label in labels:
        label_enc = tokenizer(label, return_tensors="pt", add_special_tokens=False)
        label_ids = label_enc["input_ids"].to(model.device)
        scores[label] = _score_label_logprob(model, prompt_ids, label_ids)
    return scores


def _build_prompt(tokenizer, text: str, labels: list[str], prompt_style: str) -> str:
    from shared.evaluation import _wcm_messages, build_wcm_base_lm_prompt

    if prompt_style == "chat":
        return tokenizer.apply_chat_template(
            _wcm_messages(text, labels),
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_style == "base_lm":
        return build_wcm_base_lm_prompt(text, labels)
    raise ValueError(f"Unknown prompt_style {prompt_style!r}")


def _score_one(
    model,
    tokenizer,
    rows: list[dict],
    label: str,
    prompt_style: str,
    labels: list[str],
) -> list[dict]:
    out_rows: list[dict] = []
    for i, row in enumerate(rows):
        prompt = _build_prompt(tokenizer, str(row["text"]), labels, prompt_style)
        scores = _per_label_logprobs(model, tokenizer, prompt, labels)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        pred = ranked[0][0]
        gold = str(row["label"])
        margin = (
            ranked[0][1] - ranked[1][1]
            if len(ranked) > 1
            else float("inf")
        )
        out_rows.append(
            {
                "index": i,
                "variant": label,
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "margin": (
                    round(margin, 4) if margin != float("inf") else None
                ),
                "label_logprobs": {k: round(v, 4) for k, v in scores.items()},
                "text_snippet": str(row["text"])[:120],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i + 1}/{len(rows)} processed")
    return out_rows


def _summarize(rows: list[dict], labels: list[str], variant: str) -> dict:
    if not rows:
        return {"variant": variant, "n": 0}
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = round(correct / total, 4)

    gold_dist = Counter(r["gold"] for r in rows)
    pred_dist = Counter(r["pred"] for r in rows)

    maj_label, maj_count = gold_dist.most_common(1)[0]
    maj_gold_share = round(maj_count / total, 4)
    pred_maj_share = round(pred_dist.get(maj_label, 0) / total, 4)
    always_maj_acc = round(maj_count / total, 4)

    maj_rows = [r for r in rows if r["gold"] == maj_label]
    non_maj_rows = [r for r in rows if r["gold"] != maj_label]
    acc_majority = (
        round(sum(1 for r in maj_rows if r["correct"]) / len(maj_rows), 4)
        if maj_rows
        else None
    )
    acc_non_majority = (
        round(
            sum(1 for r in non_maj_rows if r["correct"]) / len(non_maj_rows), 4
        )
        if non_maj_rows
        else None
    )

    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in labels} for g in labels
    }
    for r in rows:
        if r["gold"] in confusion and r["pred"] in confusion[r["gold"]]:
            confusion[r["gold"]][r["pred"]] += 1

    per_label: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for r in rows if r["gold"] == label and r["pred"] == label)
        fp = sum(1 for r in rows if r["gold"] != label and r["pred"] == label)
        fn = sum(1 for r in rows if r["gold"] == label and r["pred"] != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec is not None and rec is not None and (prec + rec) > 0
            else None
        )
        per_label[label] = {
            "support": gold_dist.get(label, 0),
            "predicted": pred_dist.get(label, 0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    margins = [r["margin"] for r in rows if r["margin"] is not None]
    mean_margin = (
        round(sum(margins) / len(margins), 4) if margins else None
    )

    labels_with_support = [l for l in labels if per_label[l]["support"] > 0]
    n_eff = len(labels_with_support)

    def _macro(field: str) -> float | None:
        if not labels_with_support:
            return None
        vals = [
            per_label[l][field] if per_label[l][field] is not None else 0.0
            for l in labels_with_support
        ]
        return round(sum(vals) / len(vals), 4)

    macro_recall = _macro("recall")
    macro_precision = _macro("precision")
    macro_f1 = _macro("f1")
    uniform_floor = round(1 / n_eff, 4) if n_eff else None

    return {
        "variant": variant,
        "n": total,
        "accuracy": acc,
        "always_predict_majority_acc": always_maj_acc,
        "uniform_floor_acc": uniform_floor,
        "balanced_accuracy_macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "n_classes_with_support": n_eff,
        "majority_class": maj_label,
        "majority_class_share_gold": maj_gold_share,
        "majority_class_share_pred": pred_maj_share,
        "accuracy_on_majority_rows": acc_majority,
        "accuracy_on_non_majority_rows": acc_non_majority,
        "majority_class_collapse_detected": pred_maj_share >= 0.95,
        "mean_top1_minus_top2_logprob": mean_margin,
        "gold_distribution": dict(gold_dist),
        "pred_distribution": dict(pred_dist),
        "confusion_matrix_gold_x_pred": confusion,
        "per_label": per_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Debug WCM-v2 outputs (collapse / confusion / log-prob "
            "distribution)."
        )
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
        help=(
            "LoRA adapter directory (default: Mix-50 final, Slurm 2770). "
            "Ignored when --no-adapter is passed."
        ),
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run on the base model only (zero-shot variant).",
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=["qwen", "llama"],
        help="Base model (default qwen).",
    )
    parser.add_argument(
        "--compare-zeroshot",
        action="store_true",
        help=(
            "Also score the base model (no adapter) on the same rows. "
            "Adds a model reload; ~6 minutes extra on a 24 GB MIG."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Limit to first N WCM rows (default: all 300).",
    )
    parser.add_argument(
        "--prompt-style",
        default="chat",
        choices=["chat", "base_lm"],
        help=(
            "Prompt format (matches shared.evaluation.eval_wcm). 'chat' for "
            "Qwen/LLaMA instruct + qwen_finetuned; 'base_lm' for CUTE-Llama-P."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: results/debug/wcm_<timestamp>.json).",
    )
    args = parser.parse_args()

    adapter: Path | None
    if args.no_adapter:
        adapter = None
    else:
        adapter = args.adapter
        if not adapter.is_dir():
            print(f"Adapter not found: {adapter}", file=sys.stderr)
            print(
                "Pass --no-adapter for zero-shot or --adapter PATH explicitly.",
                file=sys.stderr,
            )
            return 1

    from shared.evaluation import (
        _load_wcm_dataset,
        _wcm_columns,
        load_eval_model,
    )

    print(f"[wcm-debug] Loading WCM-v2 (max_samples={args.num_samples}) …")
    ds, repo = _load_wcm_dataset(args.num_samples)
    text_col, label_col = _wcm_columns(ds)
    labels = sorted({str(x) for x in ds[label_col]})
    rows = [
        {"text": r[text_col], "label": r[label_col]} for r in ds
    ]
    print(
        f"[wcm-debug] repo={repo} n={len(rows)} labels={labels} "
        f"text_col={text_col} label_col={label_col}"
    )

    primary_label = (
        f"{args.model}_finetuned" if adapter is not None
        else f"{args.model}_zeroshot"
    )

    print(f"[wcm-debug] Loading {primary_label} (adapter={adapter}) …")
    model, tok = load_eval_model(args.model, adapter_path=adapter)
    primary_rows = _score_one(
        model, tok, rows, primary_label, args.prompt_style, labels
    )
    summaries = [_summarize(primary_rows, labels, primary_label)]
    all_rows = list(primary_rows)
    del model

    if args.compare_zeroshot and adapter is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        zs_label = f"{args.model}_zeroshot"
        print(f"[wcm-debug] Loading {zs_label} …")
        zs_model, zs_tok = load_eval_model(args.model, adapter_path=None)
        zs_rows = _score_one(
            zs_model, zs_tok, rows, zs_label, args.prompt_style, labels
        )
        summaries.append(_summarize(zs_rows, labels, zs_label))
        all_rows.extend(zs_rows)
        del zs_model

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter": str(adapter.resolve()) if adapter else None,
        "model": args.model,
        "prompt_style": args.prompt_style,
        "num_samples": args.num_samples,
        "dataset": repo,
        "labels": labels,
        "summaries": summaries,
        "rows": all_rows,
    }

    out_path = args.out
    if out_path is None:
        out_dir = REPO_ROOT / "results" / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"wcm_{stamp}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[wcm-debug] Wrote {out_path}")
    print("[wcm-debug] Summaries:")
    for s in summaries:
        flag = " *COLLAPSE*" if s.get("majority_class_collapse_detected") else ""
        print(
            f"  {s['variant']}: acc={s['accuracy']}  "
            f"always-majority={s['always_predict_majority_acc']}  "
            f"pred[maj]={s['majority_class_share_pred']}  "
            f"acc_maj={s['accuracy_on_majority_rows']}  "
            f"acc_non_maj={s['accuracy_on_non_majority_rows']}{flag}"
        )
        print(
            f"    balanced_acc(macro_recall)={s['balanced_accuracy_macro_recall']}  "
            f"macro_f1={s['macro_f1']}  "
            f"uniform_floor={s['uniform_floor_acc']} "
            f"(n_classes={s['n_classes_with_support']})"
        )
        print(f"    pred_dist={s['pred_distribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
