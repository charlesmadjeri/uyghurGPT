# Task 03 — Fix the UG→EN decoding regression on the fine-tuned model

> **Status:** not started.
> **Depends on:** none. Can run in parallel with Tasks 01 / 02.
> **Blocks:** Task 04 (consolidated results table — the current
> `qwen_finetuned` UG→EN chrF of 9.38 is almost certainly understated and
> should not go into the report unchanged), Task 05 (analysis), Task 06
> (final report).
> **Estimated wall-clock:** ~30 min local debug + ~2 h on the cluster for
> a UG→EN-only re-eval (1012 FLORES sentences × 1 direction).

## Goal

`run_20260524_020432` produced **chrF 30.29 (qwen_zs) → 9.38 (qwen_ft)**
on FLORES UG→EN — a 21-point regression on the easier direction (UG→EN
should be easier than EN→UG because the model only needs to *generate*
English).

`docs/PROJECT_RESULTS.md` (2026-05-24 §Analysis, bullet 2) flags the
likely cause as **prompt template / stop-token mismatch on UG→EN**, not
catastrophic forgetting — C4 PPL only moved from 16.59 → 16.17 (~0.4),
which is far too small to explain a 21-chrF translation collapse. This
task confirms that diagnosis and fixes it before the numbers are frozen
into the report.

## Hypotheses (rank-ordered)

1. **Generation hangover.** The adapter has learned to keep
   `<|im_start|>assistant\n` open and continue past the natural stop —
   the EN output gets contaminated by a trailing Uyghur sentence, FLAN
   filler, or a second response. This degrades chrF dramatically while
   leaving the first few English tokens intact.
2. **EOS / stop-token mismatch.** `generate_translation` uses
   `eos_token_id=tokenizer.eos_token_id`. Qwen has both `<|endoftext|>`
   (151643) and `<|im_end|>` (151645); if the model emits `<|im_end|>`
   before the EOS, generation continues until `max_new_tokens=256`,
   appending garbage.
3. **System prompt direction confusion.** The system prompt
   `"Translate the {src_lang} input to {tgt_lang}"` is identical in
   wording for both directions and the FT model may anchor on the more
   common training direction (EN→UG appears identically often, so this
   is unlikely to be the dominant cause but worth eliminating).
4. **Decoder length mismatch.** `max_new_tokens=256` is fine for
   sentence-level FLORES — not the cause.

## Deliverables

1. A short investigation note (paste outputs from 10–20 representative
   UG→EN FLORES sentences and the fine-tuned model's full raw
   generations + decoded outputs, including the special tokens, into the
   PR description / commit message for this task).
2. The chosen fix landed in `shared/evaluation.py::generate_translation`,
   with the existing zero-shot numbers reproduced (must remain
   essentially unchanged within ±0.5 chrF for `qwen_zeroshot` and
   `llama_zeroshot`, otherwise the fix is broken).
3. A `qwen_finetuned`-only UG→EN FLORES re-eval Slurm run, with the new
   chrF / BLEU written to `results/run_<id>/experiment_1/artifacts/
   flores_qwen_finetuned.json` and the corresponding `eval_summary.json`
   updated.
4. A `PROJECT_RESULTS.md` sub-bullet in the 2026-05-24 section logging
   the back-fill (per the file's append-only convention), OR a new
   dated section if a brand-new run id was used.

## Implementation plan

### Step 1 — reproduce locally on a tiny sample

`shared/evaluation.py::generate_translation` is small and self-contained.
Reproduce the failure on 20 FLORES UG→EN sentences locally with the
saved adapter from `results/run_20260524_020432/experiment_1/checkpoints/
qwen_mix20/final`:

```python
from shared.evaluation import load_eval_model, load_flores_pairs, \
    generate_translation
en, ug = load_flores_pairs(max_samples=20)
m, t = load_eval_model("qwen", adapter_path=Path(
    "results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final"
))
for i, u in enumerate(ug):
    print(repr(generate_translation(m, t, u, "Uyghur", "English")))
```

Note any outputs that:
- end with a stray `<|im_end|>`, `<|im_start|>`, or `<|endoftext|>`
- contain a second turn (`\nassistant\n…`)
- continue into Uyghur after a brief English fragment
- run on to `max_new_tokens=256` (visible as long outputs)

This pinpoints which hypothesis is dominant.

### Step 2 — apply the fix

**Most likely fix (covers hypotheses 1 + 2):** pass a tuple of stop ids
to `generate`:

```python
stop_ids = [tokenizer.eos_token_id]
for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is not None and tid >= 0:
        stop_ids.append(tid)
out = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=stop_ids,            # transformers ≥ 4.45 accepts a list
)
```

And after decoding, hard-trim at the first occurrence of any chat marker
that survived:

```python
text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
for marker in ("\nassistant", "<|im_end|>", "<|im_start|>", "user\n",
               "system\n"):
    idx = text.find(marker)
    if idx >= 0:
        text = text[:idx].rstrip()
return text
```

`skip_special_tokens=True` *should* already do this, but the
post-decode trim catches the literal-string variants the adapter may
have learned to emit as normal text. **Crucially**, this fix is
direction-agnostic — it makes the existing `qwen_zeroshot` and
`llama_zeroshot` numbers a strictly-better baseline (or unchanged) and
must not regress them. The validation step below enforces that.

### Step 3 — local sanity vs cluster re-eval

After the fix, re-run the local 20-sentence loop from Step 1 — outputs
should be clean English sentences ending in normal punctuation, not chat
markup.

Then submit a UG→EN-only FLORES re-eval on the cluster. The cheapest
path is to extend the `--eval-only` flag from Task 02 with a sub-option
for direction, or just temporarily hard-code `eval_flores` to only run
UG→EN for this specific re-eval and revert. The former is preferred:

```python
parser.add_argument(
    "--flores-direction",
    default=None,
    choices=["en2ug", "ug2en"],
    help="Restrict FLORES eval to one direction (re-eval helper).",
)
```

Threaded through `Experiment1Config` and `eval_flores`.

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --eval-only flores --flores-direction ug2en \
  --run-id 20260524_020432 --time 4:00:00
```

### Step 4 — validate and log

The fix is good if, on the SAME run:

- `qwen_zeroshot` UG→EN chrF stays within ±0.5 of 30.29 (sanity)
- `llama_zeroshot` UG→EN chrF stays within ±0.5 of 4.71 (sanity)
- `qwen_finetuned` UG→EN chrF moves *up*, ideally to ≥ 28 (within
  ~2 chrF of zero-shot Qwen, since UG→EN is the easier direction and
  the fine-tune should not regress it once decoding is honest)

If the fine-tune still under-performs zero-shot on UG→EN after the
decoding fix, that is a real result (the Mix-20 ratio over-fits the
generate-Uyghur direction at the cost of generate-English fluency) and
goes into the analysis (Task 05) instead of being engineered away.

Append a sub-bullet to the 2026-05-24 section of `PROJECT_RESULTS.md`
recording the new numbers and the fix. If the fine-tune still
under-performs zero-shot meaningfully, also flag this as the headline
finding for Task 05.

## Validation / success criteria

1. Local reproduction (Step 1) shows the exact failure mode in the
   outputs (chat markers leaking, runaway generation, or second-turn
   contamination). The investigation note records *which*.
2. After the fix, the zero-shot UG→EN numbers reproduce within ±0.5
   chrF — this guarantees the fix is purely additive.
3. The fine-tuned UG→EN chrF improves by at least +10 chrF over the
   pre-fix 9.38, OR the analysis concludes the regression is genuine
   (and the report carries that explanation rather than burying the
   bad number).
4. No regression in EN→UG chrF / BLEU or in C4 PPL — those values are
   identical to the previous run (no re-eval needed for them; they
   share no code with the fix).

## References

- Anomaly first flagged: `docs/PROJECT_RESULTS.md` 2026-05-24 §Analysis,
  bullet 2.
- Code to patch: `shared/evaluation.py::generate_translation`
  (lines 75–99).
- Adapter to re-eval against:
  `results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final`.
- Reusable `--eval-only` flag introduced by Task 02 (this task adds the
  matching `--flores-direction` flag).
