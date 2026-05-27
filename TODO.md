# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

## Sanity-gate re-run: zero-shot UG→EN under the new decoder

Slurm 2768 was variant-scoped to `qwen_finetuned`. The direction-
conditional `repetition_penalty=1.15` / `no_repeat_ngram_size=4` apply
to **any** model when `tgt_lang == "English"`, so the zero-shot UG→EN
chrF could in principle move. Slurm 2766 showed 0× greedy collapse on
`qwen_zeroshot` (20 / 20 source-anchored), so a near no-op is expected
— but unconfirmed.

**Push + submit** (experiment-0 re-run is cheapest, ~1 h):

```bash
rsync -avz --progress \
  --exclude=results/ --exclude=results.archive/ --exclude=__pycache__/ \
  --exclude='*.pyc' --exclude=.git/ --exclude=.venv/ --exclude='*.ipynb' \
  --exclude=docs/papers/ --exclude=dataset/ --exclude=models/ \
  --exclude=checkpoints/ \
  ./ ju-compute-server:~/uyghurGPT/

python3 scripts/push.py --server ju-compute-server \
  --experiment 0 --mode eval --time 2:00:00
```

**Gate:** `qwen_zeroshot` UG→EN chrF must stay within ±0.5 of **30.10**
(`run_20260526_223852`, Slurm 2749). If it moves >0.5, narrow the
penalty further or scope it to fine-tuned variants only.

**Pull + log** (same commit): update `docs/PROJECT_RESULTS.md` §1 + §2
zero-shot rows if anything moves; otherwise log a one-line `=`
confirmation entry.

## Done (remove when read)

- ~~Slurm 2768 `qwen_finetuned` UG→EN re-eval~~ — `PROJECT_RESULTS.md`
  §1 + §2 updated; `PROJECT_REFINEMENT.md` §14 has the pre/post table.
- ~~Slurm 2766 `debug_ug2en`~~ — mechanism report in `PROJECT_RESULTS.md`
  §1 + `PROJECT_REFINEMENT.md` §14.
- ~~Training-data audit (option 4)~~ — balanced `ug2en`/`en2ug`; not a
  coverage bug.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 table~~ — Slurm 2750 / 2749.
