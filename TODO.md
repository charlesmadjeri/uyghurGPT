# TODO

Short-lived, actionable items. Pop entries as they land; delete this file
when empty.

## A1 — Beam search UG→EN (eval-only)

`shared/evaluation.py::_chat_generate_extra_kwargs` now reads
`UYGHUR_UG2EN_NUM_BEAMS` at call time. Default = `1` (no beams; Slurm 2768
numbers reproduce). `scripts/push.py` accepts `--env KEY=VAL` to forward
exports into the Slurm wrap.

**Push + submit** (`qwen_finetuned` only, ~30 min on `priority`):

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 1 --mode eval --run-id 20260524_020432 \
  --time 4:00:00 \
  --env UYGHUR_UG2EN_NUM_BEAMS=4
```

**Pull + log** (`PROJECT_RESULTS.md` §1 + §2 row):

- Compare against Slurm 2768 (UG→EN chrF 16.8079, EN→UG 14.1762).
- Document delta. EN→UG should remain byte-identical (gate works).

## A2 — Chat-style few-shot UG→EN k=3 (diagnostic, FT adapter)

`scripts/debug_ug2en.py` accepts `--fewshot-k`. Uses chat-template
multi-turn (system / user / assistant pairs) with exemplars from FLORES
**dev** (disjoint from devtest). Run on the FT adapter and compare to
zero-shot:

```bash
ssh ju-compute-server 'cd ~/uyghurGPT && \
  $HOME/micromamba/envs/uyghur_env/bin/python scripts/debug_ug2en.py \
    --fewshot-k 3 -n 50 --compare-zeroshot \
    --out results/debug/ug2en_fewshot_$(date -u +%Y%m%d_%H%M%S).json'
```

Pull the JSON; record mean chrF + failure-mode buckets in
`PROJECT_RESULTS.md` §1 alongside Slurm 2766 (n=20 zero-shot UG→EN
greedy baseline).

## Decision rule for B1 + B2 (only run if A2 says we should)

Reference points after Slurm 2768:

- `qwen_finetuned` UG→EN chrF on FLORES devtest 1012 = **16.81**
- `qwen_zeroshot` UG→EN chrF on FLORES devtest 1012 = **30.10**
- residual gap to zero-shot = **−13.29 chrF**

Apply this rule to the A2 n=50 result:

| A2 `qwen_finetuned` mean chrF | Interpretation | Next step |
|-------------------------------|----------------|-----------|
| **≥ 27** (within ~3 chrF of zero-shot) | Adapter still knows UG→EN; chat prompt is anchoring on FLAN-shaped completions. | **Stop**. Cheaper fix is at eval prompt / few-shot, not retraining. Write up. |
| **≥ 22 and < 27** (clear lift but residual gap > 3 chrF) | Mixed signal — circuit partly intact but adapter has shifted. | Run **B2 alone** (direction-stratified `eval_loss` checkpoint selection on a *new* retrain). |
| **< 22** (no or marginal lift over Slurm 2768's 16.81) | Adapter genuinely lost UG→EN; in-context examples don't rescue it. | Run **B1 + B2** combined: `ug2en` row weight 2× + per-direction `eval_loss` early stopping. |

Cross-check: A2 zero-shot mean chrF must be within ±2 chrF of the
n=20 Slurm 2766 baseline (`qwen_zeroshot` mean chrF 30.33) to trust
the A2 numbers.

## Sanity-gate (carried — cheap to fold into A1 push)

`UYGHUR_UG2EN_NUM_BEAMS=4` also applies to zero-shot at eval time, so
either (a) include `--experiment 0` in A1 as a second push, or (b) trust
the Slurm 2766 zero-shot 0 % collapse and skip. **Recommend (a)** —
1 h cost, settles the gate from §2768's open item:

```bash
python3 scripts/push.py --server ju-compute-server \
  --experiment 0 --mode eval --time 2:00:00 \
  --env UYGHUR_UG2EN_NUM_BEAMS=4
```

`qwen_zeroshot` UG→EN chrF must stay within ±0.5 of **30.10**
(Slurm 2749). If it moves >0.5, narrow the beam path to fine-tuned-only
(e.g. an additional env var or restrict via adapter presence).

## Done (remove when read)

- ~~Slurm 2768 `qwen_finetuned` UG→EN re-eval~~ — `PROJECT_RESULTS.md`
  §1 + §2 updated; `PROJECT_REFINEMENT.md` §14 has the pre/post table.
- ~~Slurm 2766 `debug_ug2en`~~ — mechanism report in `PROJECT_RESULTS.md`
  §1 + `PROJECT_REFINEMENT.md` §14.
- ~~Training-data audit (option 4)~~ — balanced `ug2en`/`en2ug`; not a
  coverage bug.
- ~~CUTE-Llama-P / Tasks 01–02 / core §2 table~~ — Slurm 2750 / 2749.
