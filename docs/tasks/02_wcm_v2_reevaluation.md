# Task 02 — WCM-v2 re-evaluation across variants

> **Status:** waiting for results (except for cute-llama-p that need first to ).
> **Depends on:** none for the qwen / llama re-eval; Task 01 must land
> first to also cover `cute_llama_p`.
> **Blocks:** Task 04 (consolidated results table), Task 05 (analysis),
> Task 06 (final report).
> **Estimated wall-clock:** ~30 min per variant on the 24 GB MIG slice
> (300 rows × ~1 s/row generation; total < 2 h for all four variants on
> separate Slurm jobs).

## Goal

`results/run_20260524_020432/experiment_1/artifacts/eval_summary.json`
shows `"wcm": {"status": "ERROR", "error": "Unrecognized WCM-v2 schema:
['text']"}` for every variant (`qwen_zeroshot`, `llama_zeroshot`,
`qwen_finetuned`). The loader has since been fixed
(`shared/evaluation.py::_load_wcm_dataset` now downloads
`minority/ug.txt` via `hf_hub_download` and parses `text\tlabel` rows —
see `PROJECT_REFINEMENT.md` §12), but no run has produced WCM accuracy
numbers with the fix in place yet. The minimum-results comparison table
*requires* the WCM accuracy column for every variant.

## Deliverables

1. WCM-v2 Uyghur accuracy populated in each of:
   - `qwen_zeroshot`
   - `llama_zeroshot`
   - `qwen_finetuned` (run 20260524_020432's `final/` adapter, or whichever
     adapter is current after Task 03 if a re-train happened)
   - `cute_llama_p` (delivered by Task 01)
2. A single `wcm_<variant>.json` artifact per variant under
   `results/run_<id>/experiment_<N>/artifacts/`, *and* an updated `wcm`
   block in the corresponding `eval_summary.json`.
3. A short note in `PROJECT_RESULTS.md` (sub-bullet inside the
   `2026-05-24 — run_20260524_020432` section) recording the back-filled
   WCM values, exactly per the convention at the top of that file ("If a
   re-evaluation supersedes a number for a given run … the new value is
   added inside the same section with a dated sub-bullet").

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

### Step 2 — minimal eval-only re-run on the cluster

For each of the three existing variants, submit an **eval-only** Slurm
run that touches only WCM. The cheapest path is to add a CLI knob that
restricts the eval harness to a single benchmark, then submit:

- `--experiment 0 --mode eval --eval-only wcm` for `qwen_zeroshot` +
  `llama_zeroshot`
- `--experiment 1 --mode eval --run-id 20260524_020432 --eval-only wcm`
  for `qwen_finetuned` (resume the existing run so the adapter is
  already on disk)
- `--experiment 2 --mode eval --eval-only wcm` for `cute_llama_p` (after
  Task 01 lands; can be folded into Task 01's main eval run instead of
  a separate job)

#### Implementation of `--eval-only`

Add a CLI flag in `main.py`:

```python
parser.add_argument(
    "--eval-only",
    default=None,
    choices=["flores", "wcm", "ppl"],
    help="Restrict --mode eval to a single benchmark (skip the others). "
         "Use to backfill a previously failed sub-eval without re-running "
         "FLORES generation, which is the expensive one.",
)
```

In `shared/evaluation.py::run_eval`, when `getattr(cfg, "eval_only", None)`
is set, skip the benchmarks not requested. Pull the value through
`Experiment{0,1,2}Config.from_namespace`. Default behaviour (no
`--eval-only`) is unchanged.

This is a 10-line patch and is reusable by every future "one benchmark
broke, re-run that benchmark only" situation.

### Step 3 — submit and pull

```bash
# qwen + llama zero-shot WCM only (~1 h total)
python3 scripts/push.py --server ju-compute-server \
  --experiment 0 --mode eval --eval-only wcm --new-run --time 2:00:00

# qwen_finetuned WCM only, reusing the existing run's adapter
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --eval-only wcm \
  --run-id 20260524_020432 --time 2:00:00

# cute_llama_p WCM is part of Task 01's main eval run; if it failed,
# re-submit just that benchmark:
python3 scripts/push.py --server ju-compute-server \
  --experiment 2 --mode eval --eval-only wcm \
  --run-id <task01_run_id> --time 2:00:00
```

Pull with `python3 scripts/check.py --server ju-compute-server --pull`.

### Step 4 — update `PROJECT_RESULTS.md`

Inside the existing `## 2026-05-24 — run_20260524_020432` section, add
under "External benchmarks" a dated sub-bullet:

```
- 2026-MM-DD WCM backfill: qwen_zs = X.X% (n=300), llama_zs = X.X%,
  qwen_ft = X.X% (run <new_run_id>). Loader: hfl/wcm-v2 → minority/ug.txt
  via hf_hub_download (PROJECT_REFINEMENT.md §12).
```

If a brand-new run was submitted just for the qwen_finetuned WCM
re-eval (rather than resuming 20260524_020432), add a new "WCM-only"
section for that run at the bottom of `PROJECT_RESULTS.md` with the
template at the file's footer.

## Validation / success criteria

1. Every `wcm_<variant>.json` artifact contains numeric `accuracy`,
   `correct`, `total = 300`, `text_column = "text"`, `label_column =
   "label"`. No `"status": "ERROR"`.
2. `wcm.accuracy` is non-zero for at least `qwen_zeroshot` and
   `qwen_finetuned` (zero would mean the model output never matches any
   label; if that happens, debug the prompt before reporting).
3. `pytest tests/` still passes (`tests/test_data_split.py` and
   `tests/test_config.py` are unaffected by this change; the
   `--eval-only` flag should not need a new test, but adding one to
   `tests/test_config.py` that checks `Experiment0Config` carries the
   field through `from_namespace` is welcome).
4. The `wcm` column of Task 04's consolidated results table can be
   filled in completely (no `n/a (ERROR)` rows).

## References

- Loader fix: `shared/evaluation.py::_load_wcm_dataset` (lines 149–186)
  + `_parse_wcm_tab_file` (lines 149–163), introduced after run
  20260524_020432.
- Originally captured failure mode: `docs/PROJECT_RESULTS.md`
  "2026-05-24" section, "WCM-v2 missing" bullet.
- Append-only logging convention: header of `PROJECT_RESULTS.md`.
