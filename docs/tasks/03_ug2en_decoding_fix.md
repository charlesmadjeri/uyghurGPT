# Task 03 — Fix the UG→EN decoding regression on the fine-tuned model

> **Status:** done (decoding fix shipped + validated; the leak
> hypothesis was **falsified** — see "Outcome" below). One optional
> follow-up — the per-sentence failure-mode diagnostic
> (`scripts/debug_ug2en.py`) — is tracked in `TODO.md`, not here.
> **Depends on:** none. Can run in parallel with Tasks 01 / 02.
> **Blocks:** nothing in the critical path. Task 04 (consolidated
> results table) consumes the post-fix `qwen_finetuned` UG→EN cell
> from `PROJECT_RESULTS.md` §2, but that cell is already populated
> (Slurm 2744, chrF 9.385 — unchanged after the fix).
> **Estimated wall-clock:** ~30 min local debug + ~6 h on the cluster
> for a full exp-1 re-eval (the `--eval-only flores --flores-direction
> ug2en` knob below was never implemented; the realised path was a
> full `--experiment 1 --mode eval --run-id 20260524_020432`).

## Outcome (added 2026-05-26 after Slurm 2744)

- **Decoding fix shipped** in `shared/evaluation.py::generate_translation`
  (stop-token list passed to `model.generate` + post-decode trim at
  chat markers `<|im_end|>`, `<|im_start|>`, `<|endoftext|>`,
  `\nassistant`, `user\n`, `system\n`). Direction-agnostic; covers
  hypotheses 1 + 2 below.
- **Zero-shot sanity check holds.** `qwen_zeroshot` and `llama_zeroshot`
  reproduce their May-24 chrF / BLEU **byte-identically** after the
  fix — the change is strictly additive.
- **Leak hypothesis FALSIFIED on the FT adapter.** Slurm 2744 re-eval
  of `qwen_finetuned` on `run_20260524_020432`'s adapter returned
  `chrF = 9.385`, `BLEU = 0.1387` — *byte-identical* to the May-24
  pre-fix numbers. The chat-marker fix had **zero measurable effect**
  on the regression, so the "stop-token / template-leak" diagnosis in
  `PROJECT_RESULTS.md` 2026-05-24 §Analysis bullet 2 was wrong. See
  `PROJECT_REFINEMENT.md` §13 "Empirical update (2026-05-26 re-eval,
  Slurm 2744)" for the falsification record.
- **Real cause: genuine Mix-20 over-fitting on the generate-English
  direction.** C4 EN PPL barely moved (16.59 → 16.17), so this is
  task-shaped, not catastrophic-forgetting-shaped. Reported as the
  headline finding in `PROJECT_RESULTS.md` §2 *Analysis* and feeds
  Task 05 (results analysis).
- **Optional follow-up** — per-sentence failure-mode classification
  via `scripts/debug_ug2en.py` (wrong-language / garbled-EN /
  template-leak / ok-EN buckets). Not required to close this task;
  tracked in `TODO.md` under "UG→EN failure-mode diagnostic on the
  compute server".

## Goal (original, retained for context)

`run_20260524_020432` produced **chrF 30.29 (qwen_zs) → 9.38 (qwen_ft)**
on FLORES UG→EN — a 21-point regression on the easier direction (UG→EN
should be easier than EN→UG because the model only needs to *generate*
English).

`docs/PROJECT_RESULTS.md` (2026-05-24 §Analysis, bullet 2) originally
flagged the likely cause as **prompt template / stop-token mismatch on
UG→EN**, not catastrophic forgetting — C4 PPL only moved from
16.59 → 16.17 (~0.4), which is far too small to explain a 21-chrF
translation collapse. This task was opened to confirm that diagnosis
and fix it before the numbers were frozen into the report. The fix
was shipped; the diagnosis was wrong (see "Outcome" above).

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

## Deliverables (status)

1. **Done** — decoding fix in `shared/evaluation.py::generate_translation`
   (stop-token list + post-decode chat-marker trim). Zero-shot
   reproduction within ±0.0 chrF for both `qwen_zeroshot` and
   `llama_zeroshot` (byte-identical, not just within ±0.5).
2. **Done** — full exp-1 re-eval on Slurm 2744 against
   `run_20260524_020432`'s `final/` adapter; `eval_summary.json` and
   the per-direction FLORES artifact carry the post-fix numbers. The
   numbers happen to be byte-identical to the May-24 pre-fix cells —
   the fix is a no-op for this adapter, which is the diagnostic
   result (see "Outcome" above), not a deliverable failure.
3. **Done** — `PROJECT_RESULTS.md` §1 "2026-05-26 — Slurm 2744" entry
   logs the byte-identical FLORES result and the leak-hypothesis
   falsification; §2 *Final results* keeps the 9.385 cell, now
   annotated as "genuine Mix-20 regression, not a decoding artifact".
   No legacy "sub-bullet under 2026-05-24" was added — superseded by
   the §1 + §2 layout.
4. **Deferred** — per-sentence investigation note (10–20 raw
   UG→EN outputs with chat-marker visibility). Not needed for the
   leak hypothesis (already falsified at the aggregate level) but
   useful for Task 05 analysis and for picking a Mix-50 retrain vs.
   prompt-anchoring next step. Implementation already in
   `scripts/debug_ug2en.py`; run instructions in `TODO.md`.

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

### Step 3 — cluster re-eval (the path actually used)

The `--eval-only flores --flores-direction ug2en` knob originally
sketched here was **never implemented**. The realised path was a
**full** exp-1 eval resuming the May-24 adapter — same wall as a
normal exp-1 eval (~5h24m on Slurm 2744). FLORES EN→UG, WCM, and
C4 PPL are re-computed alongside UG→EN; that is wasted compute but
deterministic, so the only new information against the May-24 cells
is whatever the decoding fix changes (which turned out to be
nothing for FLORES — see "Outcome").

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --run-id 20260524_020432 --time 6:00:00
```

If a future repro needs a UG→EN-only re-eval to save the FLORES
EN→UG generation cost, implement the deferred `--eval-only flores
--flores-direction ug2en` flags here and in Task 02 (sketches kept
in those task files).

### Step 4 — validate and log (closed)

Realised pass-criteria on Slurm 2744:

- `qwen_zeroshot` UG→EN chrF: 30.29 → 30.29 (byte-identical) — sanity
  holds with margin to spare.
- `llama_zeroshot` UG→EN chrF: 4.71 → 4.71 (byte-identical) — sanity
  holds.
- `qwen_finetuned` UG→EN chrF: 9.385 → **9.385** (byte-identical) —
  the fix is a no-op on this adapter. The regression is real, not a
  decoding artifact. This is now the headline finding for Task 05.

Logged in `PROJECT_RESULTS.md` §1 under "2026-05-26 — Slurm 2744
post-fix re-eval" and falsifies the leak hypothesis recorded in
`PROJECT_REFINEMENT.md` §13.

## Validation / success criteria (status)

1. **Partially met / deferred.** Aggregate-level falsification is in
   place via Slurm 2744 (byte-identical pre/post-fix chrF); the
   per-sentence investigation note from `scripts/debug_ug2en.py` is
   deferred to `TODO.md`. The aggregate result is sufficient to close
   this task because the leak hypothesis is falsified independently
   of the bucketed analysis.
2. **Met.** Zero-shot UG→EN numbers reproduced byte-identically (well
   inside ±0.5 chrF) — the fix is strictly additive.
3. **Path B taken.** The fine-tuned UG→EN chrF did **not** improve;
   the analysis concludes the regression is genuine Mix-20
   over-fitting on the generate-English direction. Carried as the
   headline finding for Task 05, not engineered away.
4. **Met.** EN→UG chrF / BLEU and C4 PPL are byte-identical to the
   May-24 cells (Slurm 2744). The decoding fix touched only
   `generate_translation`; the FLORES EN→UG generation, WCM scoring,
   and C4 PPL paths are unchanged.

## References

- Anomaly first flagged: `docs/PROJECT_RESULTS.md` 2026-05-24 §Analysis,
  bullet 2.
- Decoding fix: `shared/evaluation.py::generate_translation`
  (commit `da8e8d8`).
- Falsification record: `PROJECT_REFINEMENT.md` §13 "Empirical update
  (2026-05-26 re-eval, Slurm 2744)".
- Re-eval artifacts: `results/run_20260524_020432/experiment_1/
  artifacts/eval_summary.json` (Slurm 2744 overwrite).
- Adapter re-eval'd:
  `results/run_20260524_020432/experiment_1/checkpoints/qwen_mix20/final`.
- Per-sentence diagnostic (deferred): `scripts/debug_ug2en.py` +
  `TODO.md` "UG→EN failure-mode diagnostic on the compute server".
