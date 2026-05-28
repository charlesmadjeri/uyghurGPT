#!/usr/bin/env python3
"""Qualitative examples — N FLORES sentences × all variants × 2 directions.

The final report (Task 05 §6) needs human-readable hypothesis examples to
illustrate the failure-mode and recovery story documented in
``PROJECT_REFINEMENT.md`` §14. ``shared.evaluation.eval_flores`` discards
per-sentence hypotheses after computing corpus chrF, so they are not
preserved by the main eval pipeline; this script regenerates them on a
small fixed FLORES devtest subset and emits a JSON + markdown table.

Variant matrix:

- ``qwen_zeroshot`` — Qwen2.5-7B-Instruct, no adapter, chat template.
- ``llama_zeroshot`` — LLaMA-3.1-8B-Instruct, no adapter, chat template.
- ``qwen_finetuned_mix20`` — Qwen + Mix-20 LoRA (``run_20260524_020432``);
  §2 core row. Same decode path as Slurm 2768 (rep-penalty UG→EN).
- ``qwen_finetuned_mix50`` — Qwen + Mix-50 LoRA (``run_20260527_185416``);
  §3 bonus ablation row.
- ``cute_llama_p`` — CMLI-NLP/CUTE-Llama-Parallel, fp16, 3-shot base-LM.

For each variant and each picked FLORES devtest sentence the script
generates an EN→UG and a UG→EN hypothesis, computes sentence-level
chrF against the FLORES reference, and writes:

- ``results/reports/qualitative_examples.json`` — full structured rows.
- ``results/reports/qualitative_examples.md`` — per-direction tables of
  all variants × N sentences (suitable for copy-paste / \\include into the
  final report).

Reproducibility: the picked indices are FLORES devtest row positions
(0-indexed). The default ``0 1 2 3 4`` keeps the script idempotent.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MIX20_ADAPTER = (
    REPO_ROOT
    / "results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final"
)
DEFAULT_MIX50_ADAPTER = (
    REPO_ROOT
    / "results/run_20260527_185416/experiment_1/checkpoints/qwen_mix50/final"
)
DEFAULT_INDICES = (0, 1, 2, 3, 4)
ALL_VARIANTS = (
    "qwen_zeroshot",
    "llama_zeroshot",
    "qwen_finetuned_mix20",
    "qwen_finetuned_mix50",
    "cute_llama_p",
)


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


def _generate(
    model,
    tokenizer,
    source: str,
    src_lang: str,
    tgt_lang: str,
    prompt_style: str,
    fewshot_exemplars: list[tuple[str, str]] | None,
) -> str:
    if prompt_style == "chat":
        from shared.evaluation import generate_translation

        return generate_translation(model, tokenizer, source, src_lang, tgt_lang)
    if prompt_style == "fewshot":
        from shared.evaluation import generate_translation_fewshot

        return generate_translation_fewshot(
            model,
            tokenizer,
            source,
            src_lang,
            tgt_lang,
            fewshot_exemplars or [],
        )
    raise ValueError(f"Unknown prompt_style {prompt_style!r}")


def _score_variant(
    model,
    tokenizer,
    label: str,
    en_list: list[str],
    ug_list: list[str],
    indices: list[int],
    prompt_style: str,
    en2ug_exemplars: list[tuple[str, str]],
    ug2en_exemplars: list[tuple[str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for i, idx in enumerate(indices):
        en_src = en_list[idx]
        ug_src = ug_list[idx]

        en2ug_hyp = _generate(
            model, tokenizer, en_src, "English", "Uyghur",
            prompt_style, en2ug_exemplars,
        )
        ug2en_hyp = _generate(
            model, tokenizer, ug_src, "Uyghur", "English",
            prompt_style, ug2en_exemplars,
        )

        rows.append(
            {
                "flores_id": idx,
                "variant": label,
                "source_en": en_src,
                "reference_ug": ug_src,
                "source_ug": ug_src,
                "reference_en": en_src,
                "en2ug": {
                    "hypothesis": en2ug_hyp,
                    "chrf": _sentence_chrf(en2ug_hyp, ug_src),
                },
                "ug2en": {
                    "hypothesis": ug2en_hyp,
                    "chrf": _sentence_chrf(ug2en_hyp, en_src),
                },
            }
        )
        print(
            f"  [{label} {i + 1}/{len(indices)}] "
            f"EN→UG chrF={rows[-1]['en2ug']['chrf']} "
            f"UG→EN chrF={rows[-1]['ug2en']['chrf']}"
        )
    return rows


def _emit_markdown(report: dict, out_md: Path) -> None:
    indices = report["indices"]
    variants = [s["label"] for s in report["variant_specs"]]
    rows_lookup: dict[tuple[str, int], dict] = {
        (r["variant"], r["flores_id"]): r for r in report["rows"]
    }

    lines: list[str] = []
    lines.append("# Qualitative examples")
    lines.append("")
    lines.append(
        f"FLORES+ devtest ids `{indices}`. Generated {report['generated_at']}."
    )
    lines.append("")
    lines.append(
        "Sentence-level chrF reported per cell. See "
        "`docs/PROJECT_RESULTS.md` §2 for corpus-level numbers and "
        "`PROJECT_REFINEMENT.md` §14 for the UG→EN regression mechanism."
    )
    lines.append("")

    direction_meta = [
        ("en2ug", "EN → UG (English source → Uyghur hypothesis)",
         "source_en", "reference_ug", "English source", "Uyghur reference"),
        ("ug2en", "UG → EN (Uyghur source → English hypothesis)",
         "source_ug", "reference_en", "Uyghur source", "English reference"),
    ]

    for direction, heading, src_key, ref_key, src_label, ref_label in direction_meta:
        lines.append(f"## {heading}")
        lines.append("")
        for idx in indices:
            any_row = next(
                (r for r in report["rows"] if r["flores_id"] == idx), None
            )
            if any_row is None:
                continue
            lines.append(f"### FLORES id {idx}")
            lines.append("")
            lines.append(f"- **{src_label}**: {any_row[src_key]}")
            lines.append(f"- **{ref_label}**: {any_row[ref_key]}")
            lines.append("")
            lines.append("| Variant | chrF | Hypothesis |")
            lines.append("|---------|------|------------|")
            for v in variants:
                row = rows_lookup.get((v, idx))
                if row is None:
                    lines.append(f"| `{v}` | – | _not scored_ |")
                    continue
                hyp = (
                    row[direction]["hypothesis"]
                    .replace("|", "\\|")
                    .replace("\n", " ")
                )
                chrf = row[direction]["chrf"]
                lines.append(f"| `{v}` | {chrf} | {hyp} |")
            lines.append("")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate qualitative examples table for Task 05 §6."
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_INDICES),
        help="FLORES+ devtest row indices (0-indexed). Default: 0 1 2 3 4.",
    )
    parser.add_argument(
        "--mix20-adapter",
        type=Path,
        default=DEFAULT_MIX20_ADAPTER,
        help=(
            "Mix-20 LoRA adapter for qwen_finetuned_mix20 "
            f"(default: {DEFAULT_MIX20_ADAPTER.relative_to(REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--mix50-adapter",
        type=Path,
        default=DEFAULT_MIX50_ADAPTER,
        help=(
            "Mix-50 LoRA adapter for qwen_finetuned_mix50 "
            f"(default: {DEFAULT_MIX50_ADAPTER.relative_to(REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ALL_VARIANTS),
        choices=list(ALL_VARIANTS),
        help="Variants to score (default: all four core variants).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help=(
            "JSON output path "
            "(default: results/reports/qualitative_examples.json)."
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help=(
            "Markdown output path "
            "(default: results/reports/qualitative_examples.md)."
        ),
    )
    args = parser.parse_args()

    adapter_checks = (
        ("qwen_finetuned_mix20", args.mix20_adapter, "--mix20-adapter"),
        ("qwen_finetuned_mix50", args.mix50_adapter, "--mix50-adapter"),
    )
    for variant, adapter_path, flag in adapter_checks:
        if variant in args.variants and not adapter_path.is_dir():
            print(f"Adapter not found for {variant}: {adapter_path}", file=sys.stderr)
            print(f"Drop {variant} from --variants or pass {flag} PATH.", file=sys.stderr)
            return 1

    from shared.evaluation import (
        load_eval_model,
        load_flores_dev_exemplars,
        load_flores_pairs,
    )

    print("[qual] Loading FLORES+ devtest …")
    en_list, ug_list = load_flores_pairs(max_samples=None)
    n_pairs = len(en_list)
    for idx in args.indices:
        if not 0 <= idx < n_pairs:
            print(
                f"Index out of range: {idx} (FLORES devtest has {n_pairs} pairs)",
                file=sys.stderr,
            )
            return 1
    print(f"[qual] Picked {len(args.indices)} indices: {args.indices}")

    en2ug_exemplars, ug2en_exemplars = load_flores_dev_exemplars(k=3)
    print(
        f"[qual] Loaded {len(en2ug_exemplars)} EN↔UG exemplars from FLORES "
        "dev (for fewshot variants)"
    )

    variant_specs: list[dict] = []
    if "qwen_zeroshot" in args.variants:
        variant_specs.append(
            {
                "label": "qwen_zeroshot",
                "model": "qwen",
                "adapter": None,
                "prompt_style": "chat",
            }
        )
    if "llama_zeroshot" in args.variants:
        variant_specs.append(
            {
                "label": "llama_zeroshot",
                "model": "llama",
                "adapter": None,
                "prompt_style": "chat",
            }
        )
    if "qwen_finetuned_mix20" in args.variants:
        variant_specs.append(
            {
                "label": "qwen_finetuned_mix20",
                "model": "qwen",
                "adapter": args.mix20_adapter,
                "prompt_style": "chat",
            }
        )
    if "qwen_finetuned_mix50" in args.variants:
        variant_specs.append(
            {
                "label": "qwen_finetuned_mix50",
                "model": "qwen",
                "adapter": args.mix50_adapter,
                "prompt_style": "chat",
            }
        )
    if "cute_llama_p" in args.variants:
        variant_specs.append(
            {
                "label": "cute_llama_p",
                "model": "cute_llama_p",
                "adapter": None,
                "prompt_style": "fewshot",
            }
        )

    # Memory-friendly load order: ``cute_llama_p`` runs in fp16 (~13 GB on
    # a 7B Llama-2 backbone), the instruct variants run in 4-bit (~5 GB).
    # Slurm 2786 OOM'd in ``caching_allocator_warmup`` when cute_llama_p
    # loaded *last* — after three load/del/empty_cache cycles the 24 GB
    # MIG had no contiguous 13 GB block for the warmup's one-shot
    # ``torch.empty(byte_count // 2)``. Loading cute_llama_p first while
    # the allocator is fragment-free fixes this; the JSON / markdown still
    # iterate ``variant_specs`` in the authored display order below.
    def _load_priority(spec: dict) -> int:
        return 0 if spec["model"] == "cute_llama_p" else 1

    load_order = sorted(variant_specs, key=_load_priority)

    rows_by_variant: dict[str, list[dict]] = {}
    for spec in load_order:
        label = spec["label"]
        print(
            f"\n[qual] Loading {label} "
            f"(model={spec['model']}, adapter={spec['adapter']}) …"
        )
        model, tok = load_eval_model(spec["model"], adapter_path=spec["adapter"])
        rows = _score_variant(
            model,
            tok,
            label,
            en_list,
            ug_list,
            args.indices,
            spec["prompt_style"],
            en2ug_exemplars,
            ug2en_exemplars,
        )
        rows_by_variant[label] = rows
        del model
        del tok
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_rows: list[dict] = []
    for spec in variant_specs:
        all_rows.extend(rows_by_variant.get(spec["label"], []))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flores_split": "devtest",
        "indices": args.indices,
        "ft_adapters": {
            variant: str(adapter.resolve())
            for variant, adapter, _ in adapter_checks
            if variant in args.variants
        },
        "variant_specs": [
            {**spec, "adapter": str(spec["adapter"]) if spec["adapter"] else None}
            for spec in variant_specs
        ],
        "rows": all_rows,
    }

    out_json = args.out_json or (
        REPO_ROOT / "results" / "reports" / "qualitative_examples.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[qual] Wrote {out_json}")

    out_md = args.out_md or (
        REPO_ROOT / "results" / "reports" / "qualitative_examples.md"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _emit_markdown(report, out_md)
    print(f"[qual] Wrote {out_md}")

    print("\n[qual] Per-variant mean chrF:")
    for spec in variant_specs:
        label = spec["label"]
        variant_rows = [r for r in all_rows if r["variant"] == label]
        if not variant_rows:
            continue
        en2ug_scores = [
            r["en2ug"]["chrf"]
            for r in variant_rows
            if r["en2ug"]["chrf"] is not None
        ]
        ug2en_scores = [
            r["ug2en"]["chrf"]
            for r in variant_rows
            if r["ug2en"]["chrf"] is not None
        ]
        en2ug_mean = (
            round(sum(en2ug_scores) / len(en2ug_scores), 2)
            if en2ug_scores
            else None
        )
        ug2en_mean = (
            round(sum(ug2en_scores) / len(ug2en_scores), 2)
            if ug2en_scores
            else None
        )
        print(f"  {label}: EN→UG={en2ug_mean}  UG→EN={ug2en_mean}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
