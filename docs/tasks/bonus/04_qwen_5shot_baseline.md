# Bonus B4 — Experiment 5: Qwen2.5-7B 5-shot ICL baseline

> **Status:** not started.
> **Depends on:** main path stable (Tasks 01–04). Has no dependency
> on the fine-tune itself; can be run any time.
> **Estimated wall-clock:** ~4 h on the 24 GB MIG slice (FLORES devtest
> 1012 × 2 directions; ICL prompts are ~5× longer than the zero-shot
> chat prompt, so generation latency is a bit higher than experiment 0).

## Goal

Add a **5-shot in-context-learning** baseline using the same
Qwen2.5-7B-Instruct base model, with **no weight updates**. This
isolates "what does fine-tuning buy us *over and above* what the same
model can already do given five FLORES dev exemplars?"

This is the *optional* baseline listed in `docs/PROJECT.md`
§Evaluation Plan ("Qwen2.5-7B-Instruct, 5-shot — in-context examples
without weight updates"). It is a small additional row in the canonical
results table; it does not change the headline framing.

## Deliverables

1. `experiments/experiment_5/` package mirroring `experiments/experiment_0/`
   (eval-only). `Experiment5Config.eval_variants = ("qwen_5shot",)`.
2. `shared/evaluation.py` gains:
   - `"qwen_5shot"` in `ALL_EVAL_VARIANTS`.
   - In `_variant_specs`, when `"qwen_5shot"` is requested, append
     `{"label": "qwen_5shot", "model": "qwen", "adapter": None,
       "few_shot_k": 5}`.
   - The `eval_flores` / `eval_wcm` paths branch on the spec's
     `few_shot_k` (if present) to build a few-shot prompt for the
     chat model. Reuse the Task-01 helper
     `generate_translation_fewshot` for FLORES; for WCM, mirror it
     into `_classify_uyghur_fewshot`.
3. One Slurm run producing
   `results/run_<id>/experiment_5/artifacts/eval_summary.json` with
   the `qwen_5shot` row populated.
4. The aggregator (Task 04) extended to recognise `qwen_5shot` as a
   fifth canonical variant — emit an extra row in
   `consolidated_results.md` between `qwen_zeroshot` and
   `qwen_finetuned`.
5. New `PROJECT_RESULTS.md` section dated to the run.

## Implementation plan

### Step 1 — few-shot prompt for the chat model

The 5 FLORES dev exemplars must be packaged into the *user* message
of a single chat turn (not the system message, not multiple turns —
the model handles a single user prompt with embedded exemplars more
reliably). Sketch:

```python
def _five_shot_user(source, exemplars, src_lang, tgt_lang):
    body = "\n".join(
        f"{src_lang}: {ex_src}\n{tgt_lang}: {ex_tgt}" for ex_src, ex_tgt in exemplars
    )
    return (
        f"Translate the following {src_lang} sentence to {tgt_lang}.\n"
        f"Here are some examples:\n\n{body}\n\n"
        f"{src_lang}: {source}\n{tgt_lang}:"
    )
```

Wrap this into the standard `messages = [{"role": "system", ...},
{"role": "user", "content": _five_shot_user(...)}]` and reuse
`tokenizer.apply_chat_template(... add_generation_prompt=True)`.

### Step 2 — exemplar selection

For FLORES, take the **first five sentences of the `dev` split** (not
`devtest`, to avoid leakage — `devtest` is the test set). Use the same
5 exemplars for every test sentence. Record the FLORES `id`s in the
`eval_summary.json` for auditability.

For WCM, pick 5 rows from the same `minority/ug.txt` file but
stratified across the label set (round-robin one per label).

### Step 3 — wire CLI + run

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 5 --mode eval --new-run --time 8:00:00
```

### Step 4 — analysis

The interesting comparison is **three-way**: zero-shot vs 5-shot vs
fine-tune, on the *same model*. If 5-shot already closes most of the
gap to the fine-tune, the LoRA contribution is small (and that is a
legitimate, publishable finding). The report's §Analysis should
include this comparison if this task lands.

## Validation / success criteria

1. `eval_summary.json` has a `qwen_5shot` block populated for FLORES
   EN→UG / UG→EN, WCM, and C4 PPL (PPL is identical to
   `qwen_zeroshot` by construction — same model, same eval — but
   re-compute it as a sanity check).
2. `qwen_5shot.flores.en2ug.chrF` > `qwen_zeroshot.flores.en2ug.chrF`
   (5 exemplars should help non-trivially; if not, the prompt is
   broken — debug before reporting).
3. Aggregator (Task 04) successfully includes the new variant in the
   canonical table.
4. `pytest tests/` still passes.

## References

- Optional baseline listing: `docs/PROJECT.md` §Evaluation Plan ("Qwen
  2.5-7B-Instruct, 5-shot — in-context examples without weight
  updates").
- Few-shot helpers introduced by Task 01:
  `shared/evaluation.py::generate_translation_fewshot`.
- Variant wiring pattern: `shared/evaluation.py::_variant_specs`
  (lines 294–325).
