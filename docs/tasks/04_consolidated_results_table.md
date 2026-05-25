# Task 04 — Consolidated results table across experiments 0 / 1 / 2

> **Status:** not started.
> **Depends on:** Tasks 01 (`cute_llama_p` numbers), 02 (WCM accuracy
> backfill), 03 (UG→EN decoding fix). All three must have produced
> final artifacts before this task runs.
> **Blocks:** Task 05 (analysis), Task 06 (final report).
> **Estimated wall-clock:** ~2 h of scripting + manual review.

## Goal

Produce **the single canonical comparison table** the final report and
analysis sections both reference. The numbers come from artifacts on
disk — no model inference happens in this task. Output goes into two
places: a fresh dated section in `docs/PROJECT_RESULTS.md` (the
append-only audit log) *and* a machine-readable JSON the final report
can pull from.

## Deliverables

1. `scripts/aggregate_results.py` — a small read-only script that:
   - Scans `results/run_*/experiment_*/artifacts/eval_summary.json`
   - Selects the latest artifact for each `(variant, benchmark)` pair
     (latest = most recent `run_status.json` `finished_at`, fall back to
     directory mtime). Honour an optional `--runs` CLI flag accepting
     a comma-separated list of run ids when we want to freeze a
     specific snapshot for the report.
   - Emits `results/reports/consolidated_results.json` with the shape:

     ```json
     {
       "generated_at": "2026-MM-DDTHH:MM:SSZ",
       "source_runs": {
         "qwen_zeroshot":  "20260524_020432",
         "llama_zeroshot": "20260524_020432",
         "qwen_finetuned": "<post-task03 run id>",
         "cute_llama_p":   "<task01 run id>"
       },
       "variants": {
         "qwen_zeroshot":  { "flores": {...}, "wcm": {...}, "perplexity": {...} },
         "llama_zeroshot": { ... },
         "qwen_finetuned": { ... },
         "cute_llama_p":   { ... }
       }
     }
     ```
   - Also emits `results/reports/consolidated_results.md` containing the
     markdown table below, ready to be copy-pasted into the report and
     into `PROJECT_RESULTS.md`.
2. A new section in `docs/PROJECT_RESULTS.md` headed
   `## YYYY-MM-DD — consolidated_results_v1 (composite)` containing the
   table and a short summary of which run id provided which row (since
   the table aggregates across runs — that source-run mapping is the
   "what, when, why" the file's header insists on).
3. The aggregator is wired into `pytest` minimally: a smoke test that
   asserts `scripts/aggregate_results.py --runs 20260524_020432
   --dry-run` exits 0 and parses the file (no real assertions on
   numbers — those are not stable across re-runs).

## Implementation plan

### Step 1 — table schema

The canonical table for the report:

| Variant            | Prompt style              | FLORES EN→UG chrF | EN→UG BLEU | FLORES UG→EN chrF | UG→EN BLEU | WCM-v2 Uyghur acc. | C4 EN PPL |
|--------------------|---------------------------|-------------------|-----------|-------------------|-----------|--------------------|-----------|
| Qwen2.5-7B zero-shot   | chat template (instruct)  | … | … | … | … | … | … |
| LLaMA-3.1-8B zero-shot | chat template (instruct)  | … | … | … | … | … | … |
| **Qwen2.5-7B Mix-20 FT** | chat template (instruct)  | … | … | … | … | … | … |
| CUTE-Llama-P       | 3-shot continuation (base LM) | … | … | … | … | … | … |

Bold for our system. `n=1012` per FLORES direction, `n=300` for WCM,
`n=1000` for C4 PPL — note these in the caption, not the cells.

### Step 2 — write `scripts/aggregate_results.py`

Rough structure (uses only stdlib + `pathlib`; no HF / torch):

```python
#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from datetime import datetime, timezone

VARIANTS = ("qwen_zeroshot", "llama_zeroshot", "qwen_finetuned",
            "cute_llama_p")
RESULTS_ROOT = Path("results")
REPORTS_DIR = Path("results/reports")

def _summary_paths(runs=None):
    """Yield (run_id, exp_dir, summary_dict) for every eval_summary.json."""
    for run_dir in sorted(RESULTS_ROOT.glob("run_*")):
        if runs and run_dir.name.removeprefix("run_") not in runs:
            continue
        for exp_dir in run_dir.glob("experiment_*"):
            p = exp_dir / "artifacts" / "eval_summary.json"
            if p.is_file():
                yield run_dir.name, exp_dir, json.loads(p.read_text())

def _pick_latest(per_variant):
    """Per variant, keep the row from the run with the latest
    run_status.json `finished_at`, falling back to dir mtime."""
    ...

def _emit_markdown(table):
    """Render the table from STEP 1 above."""
    ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                    help="Comma-separated run ids to restrict aggregation to.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute the result but do not write files.")
    args = ap.parse_args()
    runs = set(args.runs.split(",")) if args.runs else None
    rows = collect(runs)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    (REPORTS_DIR / "consolidated_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    (REPORTS_DIR / "consolidated_results.md").write_text(_emit_markdown(rows))
```

Edge cases to handle:

- A variant whose latest `eval_summary.json` block has `wcm.status ==
  "ERROR"`: do *not* silently drop it — emit `n/a (ERROR)` in the
  markdown and set the JSON to `{"status": "ERROR", "error": "…"}`. The
  whole point of this script is to make a partial result auditable.
- Multiple runs with the same variant: pick the latest `finished_at`
  recorded in `run_status.json`, then fall back to directory mtime if
  status JSON is missing.
- The `eval_summary.json` from `run_20260524_020432` lives inside
  `experiment_1/`, even though it contains rows for `qwen_zeroshot`
  and `llama_zeroshot` (legacy combined run before experiment 0 was
  split out). Either accept that for the first invocation, or rerun
  experiment 0 first so the data is split cleanly.

### Step 3 — record the snapshot in PROJECT_RESULTS.md

Append (per the file's append-only convention) a new section:

```
## YYYY-MM-DD — consolidated_results_v1 (composite)

**Source runs.**
- qwen_zeroshot, llama_zeroshot: run_<id> (Slurm <jobid>)
- qwen_finetuned: run_<id> (Slurm <jobid>) — post-Task-03 decoding fix
- cute_llama_p: run_<id> (Slurm <jobid>) — Task-01 baseline run

**Table.** <paste markdown from results/reports/consolidated_results.md>

**Analysis.** <one-paragraph headline finding, referencing
docs/tasks/05_results_analysis.md for the long-form discussion>
```

## Validation / success criteria

1. `python3 scripts/aggregate_results.py` runs to completion and writes
   both `consolidated_results.json` and `consolidated_results.md` under
   `results/reports/`.
2. Every cell in the markdown table is filled with a numeric value or
   `n/a (ERROR)` — no `null`, `None`, or empty strings.
3. `consolidated_results.json.source_runs` lists exactly four run ids
   (one per variant) and each id corresponds to a real directory under
   `results/`.
4. The Markdown table is byte-identical between
   `consolidated_results.md` and the appended `PROJECT_RESULTS.md`
   section (the script is the single source of truth).
5. `pytest tests/` still passes; the smoke test for the new script lives
   in `tests/test_aggregate_results.py` and uses a tiny fixture run
   directory in `tests/fixtures/` so it does not depend on the real
   results layout.

## References

- Per-run artifact layout: `docs/PROJECT.md` §"Per-run Artifacts" and
  `utils/io.py`.
- Per-row source: `eval_summary.json` shape — see
  `results/run_20260524_020432/experiment_1/artifacts/eval_summary.json`
  for an example, and `shared/evaluation.py::run_eval` for the writer.
- Append-only convention for `PROJECT_RESULTS.md`: header at the top
  of that file.
