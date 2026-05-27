# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

## Re-eval `qwen_finetuned` after UG→EN repetition controls (priority)

Code shipped: `generate_translation` applies `repetition_penalty=1.15` +
`no_repeat_ngram_size=4` when `tgt_lang == "English"` only (Slurm 2766
showed 12/20 FT sentences stuck in a `"The 2 1 1 1 …"` greedy loop).

**Push + submit:**

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/

python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --run-id 20260524_020432 --time 6:00:00
```

**Pull + log** (same commit): update `docs/PROJECT_RESULTS.md` §1 delta +
§2 `qwen_finetuned` FLORES cells if UG→EN (or EN→UG) moves.

**Sanity gate:** `qwen_zeroshot` UG→EN chrF must stay within ±0.5 of **30.10**
(`run_20260526_223852`). If EN→UG chrF drops >0.5 vs **14.18**, narrow the
penalty to UG→EN-only (already the case) and investigate EN→UG separately.

Mechanism write-up: `PROJECT_REFINEMENT.md` §14. Diagnostic log:
`results/debug/slurm_ug2en_2766.out`.

## Done (remove when read)

- ~~Slurm 2766 `debug_ug2en`~~ — mechanism report in `PROJECT_RESULTS.md`
  §1 + `PROJECT_REFINEMENT.md` §14.
- ~~Training-data audit (option 4)~~ — balanced `ug2en`/`en2ug`; not a
  coverage bug.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 table~~ — Slurm 2750 / 2749.
