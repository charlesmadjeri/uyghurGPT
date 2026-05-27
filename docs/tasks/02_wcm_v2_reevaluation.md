# Task 02 — WCM-v2 re-evaluation across variants

> **Status:** running (3 of 4 variants done — `qwen_finetuned` 21.00 %
> via constrained-LL on Slurm 2744; `qwen_zeroshot` 6.33 % and
> `llama_zeroshot` 3.00 % on Slurm 2749 / `run_20260526_223852`;
> `cute_llama_p` cell still blocked on Task 01 — Slurm 2745 / 2748 /
> 2750 have all stalled before any FLORES progress, see
> `PROJECT_RESULTS.md` §1 "2026-05-26 / 27 — Slurm 2748 + 2750").
> **Depends on:** none for the qwen / llama re-eval; Task 01 must land
> first to also cover `cute_llama_p`.
> **Blocks:** Task 04 (consolidated results table), Task 05 (analysis),
> Task 06 (final report).
> **Estimated wall-clock:** ~30 min per variant on the 24 GB MIG slice
> (300 rows × ~1 s/row generation under constrained-LL scoring). The
> current re-eval re-runs the full exp-0 pipeline (FLORES + WCM + C4)
> rather than WCM-only because the `--eval-only` knob below was never
> implemented; budget the full exp-0 wall (~6 h) for the zero-shot
> WCM backfill.

## Goal

Two bugs corrupted the original `run_20260524_020432` WCM numbers:

1. **Loader bug (now fixed in code).** `eval_summary.json` showed
   `"wcm": {"status": "ERROR", "error": "Unrecognized WCM-v2 schema:
   ['text']"}` for every variant. `shared/evaluation.py::_load_wcm_dataset`
   now downloads `minority/ug.txt` via `hf_hub_download` and parses
   `text\tlabel` rows (`PROJECT_REFINEMENT.md` §12).
2. **Scoring bug (now fixed in code).** Once the loader was fixed, the
   2026-05-26 backfill (Slurm 2714 / 2715) reported `qwen_zs` 6.33 %,
   `llama_zs` 0.67 %, `qwen_ft` 7.33 % — all **below random** (16.7 %).
   Root cause: `_classify_uyghur` did free-form generation + substring
   match on the candidate label set; the models almost never produced a
   verbatim label and the prediction collapsed. The fix swaps it for
   **constrained log-likelihood scoring** over the 6 label strings
   (`PROJECT_REFINEMENT.md` §13). Slurm 2744 confirmed `qwen_finetuned`
   moves from 7.33 % → **21.00 %** under the fix; the zero-shot
   variants still need to be re-scored under the new path.

The minimum-results comparison table *requires* the WCM accuracy column
for every variant under a **single consistent scoring protocol**
(constrained-LL).

## Deliverables

1. WCM-v2 Uyghur accuracy populated under **constrained-LL scoring**
   for each of:
   - `qwen_zeroshot` — **done** (6.33 %, 19 / 300, Slurm 2749 /
     `run_20260526_223852`). Identical count to the May-26 free-form
     2714 backfill — both protocols pick the same 19 rows for Qwen.
   - `llama_zeroshot` — **done** (3.00 %, 9 / 300, Slurm 2749 /
     `run_20260526_223852`). ×4.5 over the 0.67 % free-form 2714 cell.
   - `qwen_finetuned` — **done** (21.00 %, 63 / 300, Slurm 2744 on
     `run_20260524_020432`'s `final/` adapter).
   - `cute_llama_p` — still blocked on Task 01. Three exp-2
     submissions (Slurm 2745 / 2748 / 2750) have all stalled before
     any FLORES progress dots; see `PROJECT_RESULTS.md` §1 entry
     "2026-05-26 / 27 — Slurm 2748 + 2750" for the investigation
     handoff and `TODO.md` for the next-action checklist.
2. The `wcm` block of each variant's `eval_summary.json` reports
   `accuracy`, `correct`, `total = 300`, `text_column = "text"`,
   `label_column = "label"`, and the constrained-LL marker (no
   free-form `prediction` strings). Stand-alone `wcm_<variant>.json`
   side-cars are optional — the canonical artifact is `eval_summary.json`.
3. `PROJECT_RESULTS.md` updated in the **same commit** as each artifact
   pull: append a dated bullet to §1 *Change log* with the Slurm job id
   and the WCM delta, and overwrite the `qwen_zeroshot` /
   `llama_zeroshot` / `cute_llama_p` WCM cells in §2 *Final results —
   core experiments* (and the matching cells in §2's "Sources for
   populated cells" sub-table). The legacy "append a sub-bullet under
   the 2026-05-24 section" convention is superseded by the §1 + §2
   layout — do not use it.

## Implementation plan

### Step 1 — sanity-check the loader locally

The fix is already in code. Before burning Slurm time, verify the loader
returns 300 rows with the expected schema:

```bash
python3 - <<'EOF'
from pathlib import Path
from shared.evaluation import _load_wcm_dataset, _wcm_columns
ds, repo = _load_wcm_dataset(None)
print(repo, len(ds), ds.column_names, _wcm_columns(ds))
print(ds[0])
EOF
```

Expected: `hfl/wcm-v2:minority/ug.txt 300 ['text', 'label'] ('text',
'label')` and a sample row that prints Uyghur Arabic-script text + a
short label string. If any of that is wrong, the loader has regressed
and must be fixed before any re-eval is submitted.

### Step 2 — re-eval on the cluster (full exp-0 / exp-1, no WCM-only knob)

The `--eval-only` knob originally planned here was never implemented;
the realised workflow re-runs the **full** eval pipeline (FLORES +
WCM + C4) on a fresh run id and keeps the WCM column. The FLORES /
C4 numbers are byte-identical re-runs of the existing cells under
deterministic decoding (confirmed on Slurm 2744 vs 2650 for
`qwen_finetuned`) so the only **new** information is the WCM column;
the cost is a full exp-0 wall (~6 h) instead of the originally
hoped-for ~1 h.

- `qwen_zeroshot` + `llama_zeroshot` — Slurm 2749 /
  `run_20260526_223852` (`--experiment 0 --mode eval --new-run`,
  `--time 6:00:00` from `push.py`'s default).
- `qwen_finetuned` — **done** on Slurm 2744 via a full exp-1 re-eval
  resuming `run_20260524_020432`'s adapter (`--experiment 1 --mode eval
  --run-id 20260524_020432`).
- `cute_llama_p` — folded into Task 01's main eval run (Slurm 2750 /
  `run_20260526_222254`, `--experiment 2 --mode eval --new-run --time
  1-00:00:00`). No separate WCM-only submission.

#### Deferred: a `--eval-only` knob

Originally planned here. Not implemented and not required for the core
results table (the full-pipeline re-runs are deterministic and the
delta vs. the existing cells is auditable via the per-metric source
sub-table in `PROJECT_RESULTS.md` §2). Keep as a future stretch item
if a re-eval ever needs to skip FLORES generation for cost reasons —
sketch was:

```python
parser.add_argument(
    "--eval-only",
    default=None,
    choices=["flores", "wcm", "ppl"],
    help="Restrict --mode eval to a single benchmark (skip the others).",
)
```
threaded through each `Experiment{N}Config.from_namespace` and gated
inside `shared/evaluation.run_eval`. Until that lands, use the full
exp-0 / exp-1 / exp-2 re-runs above.

### Step 3 — submit (the commands actually used)

```bash
# qwen + llama zero-shot, full exp-0 re-eval (~6 h wall)
python3 scripts/push.py --server ju-compute-server \
  --experiment 0 --mode eval --new-run

# qwen_finetuned, full exp-1 eval resuming the May-24 adapter (~5h24m
# observed on Slurm 2744; already done — left here for traceability)
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --run-id 20260524_020432 --time 6:00:00

# cute_llama_p, full exp-2 eval (24 h wall — see Task 01)
python3 scripts/push.py --server ju-compute-server \
  --experiment 2 --mode eval --new-run --time 1-00:00:00
```

Pull with `python3 scripts/check.py --server ju-compute-server --pull`.

### Step 4 — update `PROJECT_RESULTS.md`

In the **same commit** as each artifact pull, do both of:

1. Append a dated bullet to `PROJECT_RESULTS.md` §1 *Change log* with
   the Slurm job id, the new run id, and the WCM delta (e.g.
   `qwen_zs.wcm: 6.33 % (free-form, Slurm 2714) → X.XX % (constrained-LL,
   Slurm 2749, run_20260526_223852)`).
2. Overwrite the affected WCM cell(s) in §2 *Final results — core
   experiments* and the matching row(s) of §2's "Sources for populated
   cells" sub-table.

Do **not** sub-bullet under the existing `2026-05-24` section, and do
**not** use the legacy per-run template at the bottom of the file —
the §1 / §2 layout is the single source of truth.

## Validation / success criteria

1. Every variant's `eval_summary.json::wcm` block contains numeric
   `accuracy`, `correct`, `total = 300`, `text_column = "text"`,
   `label_column = "label"`. No `"status": "ERROR"`. **No free-form
   `prediction` strings** — the constrained-LL path always returns a
   legal label id.
2. `wcm.accuracy` is **at or above the random baseline (16.7 %)** for
   every variant. Slurm 2744's 21.00 % on `qwen_finetuned` is the
   reference; values still below random after the fix indicate a
   prompt-side bug, not a scoring-side bug, and should be debugged
   before reporting.
3. `pytest tests/` still passes. New protocol coverage already lives in
   `tests/test_evaluation_cute_llama_p.py::test_build_wcm_base_lm_prompt`
   and `::test_classify_uyghur_base_lm_constrained_ll` — extend with
   a chat-template constrained-LL test if Step 3's zero-shot re-eval
   surfaces a regression.
4. The WCM column of `PROJECT_RESULTS.md` §2 *Final results — core
   experiments* contains four constrained-LL numbers (no `pending`,
   no mixed-protocol rows) once Slurm 2749 + Slurm 2750 land.

## References

- Loader fix (bug 1): `shared/evaluation.py::_load_wcm_dataset` +
  `_parse_wcm_tab_file` — see `PROJECT_REFINEMENT.md` §12.
- Scoring fix (bug 2): `shared/evaluation.py::_classify_uyghur`
  constrained log-likelihood path — see `PROJECT_REFINEMENT.md` §13
  and the Slurm 2744 entry in `PROJECT_RESULTS.md` §1.
- Append-only logging convention (current §1 + §2 layout): header of
  `PROJECT_RESULTS.md`.
