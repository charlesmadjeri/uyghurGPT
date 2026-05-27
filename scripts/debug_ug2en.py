#!/usr/bin/env python3
"""Inspect UG→EN FLORES hypotheses and classify failure modes.

Runs the same ``generate_translation`` path as external eval on a small
FLORES+ devtest slice, then scores each hypothesis for:

  - Arabic-script ratio (failure mode **A**: model outputs Uyghur, not English)
  - Chat-marker leaks (failure mode **C**: decoding / template artefacts)
  - Sentence chrF vs reference (failure mode **B**: English but garbled)

Requires GPU + the fine-tuned adapter on disk. Default adapter:
``results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final``.

Examples::

    python scripts/debug_ug2en.py
    python scripts/debug_ug2en.py --compare-zeroshot -n 20
    python scripts/debug_ug2en.py --adapter path/to/final --out results/debug/ug2en.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ADAPTER = (
    REPO_ROOT
    / "results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final"
)

_CHAT_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "\nassistant\n",
    "\nuser\n",
    "\nsystem\n",
)


def _arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = sum(
        1
        for ch in text
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    return arabic / len(text)


def _repetition_ratio(text: str, n: int = 5) -> float:
    if len(text) < 2 * n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    seen: set[str] = set()
    dup = 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / max(1, len(grams))


def _has_chat_leak(text: str) -> bool:
    return any(marker in text for marker in _CHAT_MARKERS)


def _sentence_chrf(hypothesis: str, reference: str) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    try:
        return round(
            sacrebleu.metrics.CHRF().sentence_score(hypothesis, [reference]).score,
            2,
        )
    except Exception:
        return None


def _classify_failure_mode(
    hypothesis: str, reference: str, chrf: float | None
) -> str:
    """Heuristic label for the investigation note in Task 03."""
    if _arabic_char_ratio(hypothesis) >= 0.6:
        return "A_wrong_language_uyghur"
    if _has_chat_leak(hypothesis):
        return "C_decoding_or_template_leak"
    if chrf is not None and chrf < 5.0:
        return "B_garbled_or_weak_english"
    return "ok_english"


def _score_one(
    model,
    tokenizer,
    ug_sources: list[str],
    en_refs: list[str],
    label: str,
) -> list[dict]:
    from shared.evaluation import generate_translation

    rows: list[dict] = []
    for i, (src, ref) in enumerate(zip(ug_sources, en_refs)):
        hyp = generate_translation(model, tokenizer, src, "Uyghur", "English")
        chrf = _sentence_chrf(hyp, ref)
        row = {
            "index": i,
            "variant": label,
            "source_uy": src,
            "reference_en": ref,
            "hypothesis": hyp,
            "hypothesis_repr": repr(hyp),
            "arabic_char_ratio": round(_arabic_char_ratio(hyp), 3),
            "repetition_ratio": round(_repetition_ratio(hyp), 3),
            "chrf": chrf,
            "has_chat_leak": _has_chat_leak(hyp),
            "failure_mode": _classify_failure_mode(hyp, ref, chrf),
            "hypothesis_len_chars": len(hyp),
        }
        rows.append(row)
        print(
            f"[{label} {i:02d}] mode={row['failure_mode']} "
            f"arabic={row['arabic_char_ratio']:.2f} chrf={chrf} "
            f"len={row['hypothesis_len_chars']}"
        )
        print(f"  ref: {ref[:120]}{'…' if len(ref) > 120 else ''}")
        print(f"  hyp: {hyp[:120]}{'…' if len(hyp) > 120 else ''}")
        if row["has_chat_leak"]:
            print("  ** chat marker leak detected in decoded text")
        print()
    return rows


def _summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"variant": label, "n": 0}
    modes: dict[str, int] = {}
    for r in rows:
        modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    chrf_vals = [r["chrf"] for r in rows if r["chrf"] is not None]
    return {
        "variant": label,
        "n": len(rows),
        "failure_mode_counts": modes,
        "mean_arabic_char_ratio": round(
            sum(r["arabic_char_ratio"] for r in rows) / len(rows), 3
        ),
        "mean_chrf": round(sum(chrf_vals) / len(chrf_vals), 2) if chrf_vals else None,
        "chat_leak_count": sum(1 for r in rows if r["has_chat_leak"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug UG→EN FLORES outputs (failure-mode classification)."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
        help="Path to LoRA adapter directory (default: run_20260524_020432 final)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
        help="Number of FLORES+ devtest UG→EN sentence pairs (default 20)",
    )
    parser.add_argument(
        "--compare-zeroshot",
        action="store_true",
        help="Also run qwen zero-shot on the same sentences (loads model twice)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report here (default: results/debug/ug2en_<timestamp>.json)",
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        print(f"Adapter not found: {args.adapter}", file=sys.stderr)
        print("Pull checkpoints or pass --adapter explicitly.", file=sys.stderr)
        return 1

    from shared.evaluation import load_eval_model, load_flores_pairs

    print(f"[debug] Loading FLORES+ devtest (n={args.num_samples}) …")
    en_refs, ug_sources = load_flores_pairs(max_samples=args.num_samples)

    all_rows: list[dict] = []
    summaries: list[dict] = []

    print(f"[debug] Loading fine-tuned qwen (adapter={args.adapter}) …")
    ft_model, ft_tok = load_eval_model("qwen", adapter_path=args.adapter)
    ft_rows = _score_one(ft_model, ft_tok, ug_sources, en_refs, "qwen_finetuned")
    all_rows.extend(ft_rows)
    summaries.append(_summarize(ft_rows, "qwen_finetuned"))
    del ft_model

    if args.compare_zeroshot:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[debug] Loading qwen zero-shot …")
        zs_model, zs_tok = load_eval_model("qwen", adapter_path=None)
        zs_rows = _score_one(zs_model, zs_tok, ug_sources, en_refs, "qwen_zeroshot")
        all_rows.extend(zs_rows)
        summaries.append(_summarize(zs_rows, "qwen_zeroshot"))
        del zs_model

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter": str(args.adapter.resolve()),
        "num_samples": args.num_samples,
        "summaries": summaries,
        "rows": all_rows,
        "interpretation": {
            "A_wrong_language_uyghur": (
                "Hypothesis is mostly Arabic script — model answered in Uyghur. "
                "Try stronger EN direction cue at eval or higher FLAN mix (Mix-50)."
            ),
            "B_garbled_or_weak_english": (
                "Hypothesis is Latin but chrF < 5 — fluent English decoding "
                "problem or severe content mismatch; retrain / capacity."
            ),
            "C_decoding_or_template_leak": (
                "Chat markers or turn headers in hypothesis — decoding fix "
                "target (may be invisible to corpus chrF if stripped as tokens)."
            ),
            "ok_english": "Plausible English fragment; regression is elsewhere.",
        },
    }

    out_path = args.out
    if out_path is None:
        out_dir = REPO_ROOT / "results" / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"ug2en_{stamp}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[debug] Wrote {out_path}")
    print("[debug] Summary:")
    for s in summaries:
        print(f"  {s['variant']}: {s['failure_mode_counts']} mean_chrf={s.get('mean_chrf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
