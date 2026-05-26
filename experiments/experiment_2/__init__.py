"""Experiment 2 — CUTE-Llama-P few-shot baseline (eval-only).

Loads ``CMLI-NLP/CUTE-Llama`` (subfolder ``CUTE-Llama-Parallel``) in fp16
and evaluates it on FLORES-200 EN↔UG + WCM-v2 Uyghur + C4 PPL using the
base-LM continuation prompt validated by ``shared/preflight.py`` check 5.
See ``docs/PROJECT.md`` §CUTE-Llama-P Baseline.
"""
