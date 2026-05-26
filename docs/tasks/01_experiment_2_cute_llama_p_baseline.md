# Task 01 — Experiment 2: CUTE-Llama-P few-shot baseline

> **Status:** running (code shipped; cluster results pending — Slurm
> `2745` timed out at 6 h on `[eval] 50/1012` EN→UG, no artifacts.
> Resubmitted as Slurm `2750` / `run_20260526_222254` with 24 h walltime;
> waiting on it to land).
> **Depends on:** none (eval-only; preflight check 5 already PASS on the
> current 24 GB MIG slice — see `results/preflight/check5.json`).
> **Blocks:** Tasks 04 (consolidated results table), 05 (analysis), 06
> (final report).
> **Estimated wall-clock:** **24 h** on the 24 GB MIG slice. The
> original ~3–4 h estimate was wrong — fp16 7B + eager attention +
> `repetition_penalty` on FLORES devtest 1012 × 2 directions runs at
> ~30 s/sentence (Slurm 2745 telemetry). Budget 1 day, not 6 h.

## Goal

Run the **published comparison baseline** — CUTE-Llama-P (Zhuang & Sun,
COLING 2025; Llama2-7B + ~155 K vocab expansion + continued pretraining on
CUTE-P) — through the *same* external benchmarks our fine-tuned Qwen is
evaluated on. The paper only publishes ZH→UG numbers, so EN↔UG must be
produced by us.

This experiment is **eval-only** — it never trains, never preprocesses, and
does not depend on any artifact from experiment 0 or 1.

## Why it is "main", not "bonus"

CUTE-Llama-P is the paper-paired baseline that motivates the entire
research framing ("QLoRA instruction tuning on a native-multilingual model
vs. continued pretraining + vocabulary surgery"). Without these numbers in
the report we cannot answer the actual research question.

## Deliverables

1. `experiments/experiment_2/` package mirroring `experiments/experiment_0/`:
   - `experiments/experiment_2/__init__.py` re-exporting `run` from `run.py`
   - `experiments/experiment_2/config.py` defining `Experiment2Config`
     (eval-only, no training params, `eval_variants=("cute_llama_p",)`)
   - `experiments/experiment_2/run.py` orchestration (coerces non-`eval`
     modes to `eval`, like experiment 0)
2. New `cute_llama_p` evaluation variant wired through
   `shared/evaluation.py`:
   - `ALL_EVAL_VARIANTS` extended to include `"cute_llama_p"`
   - `_variant_specs` adds a spec with `model="cute_llama_p"` and
     `adapter=None`
   - `load_eval_model` accepts `"cute_llama_p"`, loads
     `CMLI-NLP/CUTE-Llama` (`subfolder="CUTE-Llama-Parallel"`,
     `trust_remote_code=True`, `dtype=torch.float16`,
     `attn_implementation="eager"`) — **no** 4-bit quantization
     (preflight check 5 documented that NF4 produced degenerate output on
     this vocab-expanded base; fp16 is the validated path)
   - `MODEL_IDS` in `shared/models.py` gains `"cute_llama_p":
     "CMLI-NLP/CUTE-Llama"` (and a sibling helper for the subfolder if
     needed)
3. A base-LM-appropriate prompt path for FLORES translation: CUTE-Llama-P
   is **not** an instruct model and has no chat template. Reuse the few-shot
   continuation prompt from `shared/preflight.py::_build_fewshot_prompt`
   (FLORES dev exemplars `k=3`, evaluated on `devtest`). Implement this as a
   `generate_translation_fewshot(model, tokenizer, source, src_lang,
   tgt_lang, exemplars)` in `shared/evaluation.py` and route `cute_llama_p`
   through it from `eval_flores`.
4. `main.py` accepts `--experiment 2`; `scripts/push.py` accepts
   `--experiment 2` and picks a sensible `--time` default (start with
   `6:00:00` like experiment 0; tighten after the first observed wall).
5. One Slurm run on `ju-compute-server` producing
   `results/run_<id>/experiment_2/artifacts/eval_summary.json` with the
   `cute_llama_p` row populated for FLORES EN→UG / UG→EN, WCM-v2, and C4
   PPL.
6. `docs/PROJECT_RESULTS.md` updated in **the same commit** that pulls
   the artifacts: append a dated bullet to §1 *Change log* with the
   `cute_llama_p` deltas, and populate the `cute_llama_p` row of §2
   *Final results — core experiments* (plus the matching cell in §2's
   "Sources for populated cells" sub-table). Do **not** add a new
   per-run section — the legacy template at the bottom of that file is
   superseded by the §1 + §2 layout.

## Implementation plan

### Step 1 — extend `shared/models.py`

Add CUTE-Llama-P to `MODEL_IDS` and a helper for the HF subfolder:

```python
MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "cute_llama_p": "CMLI-NLP/CUTE-Llama",
}

CUTE_LLAMA_P_SUBFOLDER = "CUTE-Llama-Parallel"
```

`load_tokenizer` and any helper that calls `AutoTokenizer.from_pretrained`
must pass `subfolder=CUTE_LLAMA_P_SUBFOLDER, trust_remote_code=True` when
`choice == "cute_llama_p"`. The cleanest way is a small helper
`_from_pretrained_kwargs(choice)` returning the extra kwargs and used by
both the tokenizer and model loaders.

### Step 2 — extend `shared/evaluation.py`

- In `load_eval_model`, branch on `model_choice == "cute_llama_p"` and load
  the base in **fp16, eager attention, no `bnb_config`** (preflight check
  5's validated path — see `shared/preflight.py::check5_cute_llama`,
  lines 548–566). Skip the `PeftModel.from_pretrained` branch entirely
  (no adapter).
- Implement `generate_translation_fewshot(model, tokenizer, source,
  src_lang, tgt_lang, exemplars)` reusing the prompt format used by
  `shared/preflight.py::_build_fewshot_prompt` (`"{src_lang}: {x}\n{tgt_lang}:
  {y}\n\n"` × k, then `"{src_lang}: {source}\n{tgt_lang}:"`).
  Generation: `do_sample=False`, `max_new_tokens=256`, stop on the next
  `"\n{src_lang}:"` (post-process) so the model does not run away into a
  fourth example.
- Hoist FLORES `dev` exemplar loading into a helper that returns
  `(en_dev, ug_dev)` (k=3 by default) so the variant spec can pass them
  in. Re-use `_load_flores_fewshot` if it is already importable; otherwise
  copy it into `shared/evaluation.py`.
- In `eval_flores`, branch on `spec["model"]` to call
  `generate_translation_fewshot` when it is `"cute_llama_p"` and the
  existing `generate_translation` otherwise.
- In `eval_wcm`, swap the chat-template-based `_classify_uyghur` for a
  base-LM-friendly prompt when the model is `cute_llama_p` (few-shot pattern
  `"Uyghur: {text}\nLabel: {label}\n\n"` × 3 with labels picked round-robin
  from the label set, then `"Uyghur: {test_text}\nLabel:"`). PPL on C4 is
  unchanged.
- `_variant_specs` learns `"cute_llama_p"`:

  ```python
  if "cute_llama_p" in wanted:
      specs.append({"label": "cute_llama_p", "model": "cute_llama_p",
                    "adapter": None})
  ```
  and `ALL_EVAL_VARIANTS` is extended accordingly.

### Step 3 — `experiments/experiment_2/`

Mirror `experiments/experiment_0/` literally:

```python
# experiments/experiment_2/config.py
@dataclass
class Experiment2Config:
    experiment_id: int = 2
    model: str = "cute_llama_p"
    mix: int = 0
    epochs: int = 0
    sample_count: int | None = None
    results_root: str = "results"
    flores_max_samples: int | None = None
    wcm_max_samples: int | None = None
    ppl_max_samples: int = 1000
    eval_variants: tuple[str, ...] = field(default_factory=lambda: ("cute_llama_p",))
    @classmethod
    def from_namespace(cls, args): ...     # copy from Experiment0Config
    def to_dict(self): ...
    @property
    def model_label(self): return "cute_llama_p"
```

```python
# experiments/experiment_2/run.py
def run(args):
    cfg = Experiment2Config.from_namespace(args)
    run_id = io.resolve_run_id(args.run_id, cfg.results_root)
    root = io.ensure_run_layout(cfg.results_root, run_id, cfg.experiment_id)
    io.write_run_config(root, {"experiment": cfg.experiment_id, **cfg.to_dict(), "run_id": run_id})
    mode = getattr(args, "mode", "eval")
    if mode in ("preprocess", "train"):
        print(f"[exp2] mode={mode!r} is a no-op for the CUTE-Llama-P baseline; running eval instead.")
        mode = "eval"
    io.write_run_status(root, "started", {"mode": mode})
    if mode in ("eval", "all"):
        stage("Experiment 2 — CUTE-Llama-P few-shot baseline")
        from shared import evaluation
        evaluation.run_eval(cfg, root)
```

### Step 4 — wire the CLI

- `main.py::run_experiment` adds a branch for `args.experiment == 2`
  importing `experiments.experiment_2`.
- `scripts/push.py` accepts `--experiment 2` (the `--experiment` argparse
  default of `1` is already there, just make sure the experiment-aware
  `--time` picker handles `2`). **The current default of `6:00:00` is
  too short** — Slurm 2745 timed out at 6 h on `[eval] 50/1012`. Pass
  `--time 1-00:00:00` explicitly, or raise the default in `push.py`
  before the next submission. Calibrate downwards once a full run lands.

### Step 5 — run it

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 2 --mode eval --new-run --time 1-00:00:00
```

Monitor with `python3 scripts/check.py --server ju-compute-server`. Pull
results with `python3 scripts/check.py --server ju-compute-server --pull`.

## Validation / success criteria

1. `results/run_<id>/experiment_2/artifacts/eval_summary.json` exists and
   contains a single `cute_llama_p` block with `flores.en2ug`,
   `flores.ug2en`, `wcm` and `perplexity` populated (no `"status":
   "ERROR"` on any sub-key).
2. `cute_llama_p.flores.en2ug.chrF ≥ 10` (preflight check 5 confirms
   Uyghur generation works; if chrF is < 5 the prompt is degenerate, not
   the model — re-check stop conditions).
3. `cute_llama_p.flores.ug2en.chrF ≤ cute_llama_p.flores.en2ug.chrF + 25`
   (CUTE-Llama-P was trained Chinese→Uyghur direction and is not expected
   to be strong on UG→EN; this is just a sanity bound).
4. `PROJECT_RESULTS.md` is updated per Deliverable 6: a dated bullet
   in §1 *Change log* with the Slurm job id, and the `cute_llama_p` row
   of §2 *Final results — core experiments* (plus the §2 source-cell
   sub-table) is no longer `pending`. Both edits land in the same
   commit as the artifact pull.
5. Tests still pass: `pytest tests/` (the data-split contract suite is
   unaffected; this task only touches the eval path).

## References

- Existing implementation we are mirroring: `experiments/experiment_0/run.py`
  and `experiments/experiment_0/config.py`.
- Validated few-shot prompt + load recipe: `shared/preflight.py`
  §`check5_cute_llama` (lines 513–635), including the explicit fp16 +
  no-quantization choice (lines 544–562) and the FLORES dev exemplar
  loader (`_load_flores_fewshot`).
- Baseline scope and prompting protocol: `docs/PROJECT.md`
  §CUTE-Llama-P Baseline.
- Wall-clock budget rationale: `docs/PROJECT_RESULTS.md` §1 (Slurm
  2745, timed out at 6 h on `[eval] 50/1012` EN→UG) — fp16 7B base LM
  generation is **not** comparable to the quantized exp-0 wall. Budget
  24 h until the first full run calibrates the number.
