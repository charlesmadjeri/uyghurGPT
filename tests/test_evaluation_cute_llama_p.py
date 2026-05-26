"""Unit tests for the CUTE-Llama-P (base-LM) evaluation path.

Covers the helpers added for ``--experiment 2`` (docs/tasks/01_experiment_2_cute_llama_p_baseline.md):

- ``shared.models.model_load_kwargs`` resolves ``subfolder`` + ``trust_remote_code``
  for the ``cute_llama_p`` choice without leaking those kwargs to other models.
- ``ALL_EVAL_VARIANTS`` includes ``"cute_llama_p"``.
- ``build_fewshot_translation_prompt`` formats the few-shot continuation
  prompt with exemplars and ends on a ``{tgt_lang}:`` cue.
- ``_trim_at_fewshot_boundary`` trims at the *earliest* exemplar boundary.
- ``build_wcm_base_lm_prompt`` produces a flat string (no chat template)
  ending in ``Label:``.
- ``_classify_uyghur(prompt_style="base_lm")`` picks the argmax label
  via log-likelihood scoring on a stub model.

No model load, no GPU; all helpers are pure-Python / pure-torch ops.
"""

from __future__ import annotations

import pytest


def test_model_load_kwargs_for_cute_llama_p_includes_subfolder_and_trust_remote_code():
    from shared.models import model_load_kwargs

    kwargs = model_load_kwargs("cute_llama_p")
    assert kwargs["subfolder"] == "CUTE-Llama-Parallel"
    assert kwargs["trust_remote_code"] is True


def test_model_load_kwargs_empty_for_instruct_models():
    from shared.models import model_load_kwargs

    assert model_load_kwargs("qwen") == {}
    assert model_load_kwargs("llama") == {}


def test_model_load_kwargs_returns_independent_copies():
    """Mutating the returned dict must not poison the next caller."""
    from shared.models import model_load_kwargs

    first = model_load_kwargs("cute_llama_p")
    first["subfolder"] = "MUTATED"
    second = model_load_kwargs("cute_llama_p")
    assert second["subfolder"] == "CUTE-Llama-Parallel"


def test_cute_llama_p_is_a_registered_eval_variant():
    from shared.evaluation import ALL_EVAL_VARIANTS

    assert "cute_llama_p" in ALL_EVAL_VARIANTS


def test_experiment2_config_defaults():
    from experiments.experiment_2.config import Experiment2Config

    cfg = Experiment2Config()
    assert cfg.experiment_id == 2
    assert cfg.model == "cute_llama_p"
    assert cfg.eval_variants == ("cute_llama_p",)
    assert cfg.epochs == 0
    assert cfg.model_label == "cute_llama_p"


def test_experiment2_config_to_dict_is_json_safe():
    import json

    from experiments.experiment_2.config import Experiment2Config

    payload = Experiment2Config().to_dict()
    json.dumps(payload)


def test_build_fewshot_translation_prompt_includes_exemplars_and_label_cue():
    from shared.evaluation import build_fewshot_translation_prompt

    exemplars = [
        ("Hello.", "ھەللو."),
        ("Goodbye.", "خوش."),
    ]
    prompt = build_fewshot_translation_prompt(
        source="How are you?",
        src_lang="English",
        tgt_lang="Uyghur",
        exemplars=exemplars,
    )
    for ex_src, ex_tgt in exemplars:
        assert ex_src in prompt
        assert ex_tgt in prompt
    assert "How are you?" in prompt
    # The prompt must end on the {tgt_lang}: cue so the model continues
    # with the answer rather than the next prompted source line.
    assert prompt.rstrip().endswith("Uyghur:")


def test_build_fewshot_translation_prompt_truncates_long_exemplars():
    from shared.evaluation import build_fewshot_translation_prompt

    long_src = "a" * 1000
    prompt = build_fewshot_translation_prompt(
        source="test",
        src_lang="English",
        tgt_lang="Uyghur",
        exemplars=[(long_src, "ref")],
    )
    # 400-char cap per ``build_fewshot_translation_prompt``.
    assert "a" * 401 not in prompt
    assert "a" * 400 in prompt


def test_build_fewshot_translation_prompt_zero_exemplars_still_works():
    """Edge case: ``k=0`` produces a clean continuation-style prompt with
    no preceding exemplars but still terminates on the ``{tgt_lang}:`` cue."""
    from shared.evaluation import build_fewshot_translation_prompt

    prompt = build_fewshot_translation_prompt(
        source="Hello.", src_lang="English", tgt_lang="Uyghur", exemplars=[]
    )
    assert prompt == "English: Hello.\nUyghur:"


def test_trim_at_fewshot_boundary_stops_at_english_marker():
    from shared.evaluation import _trim_at_fewshot_boundary

    polluted = "translation here.\nEnglish: next prompted source"
    assert _trim_at_fewshot_boundary(polluted) == "translation here."


def test_trim_at_fewshot_boundary_stops_at_uyghur_marker():
    from shared.evaluation import _trim_at_fewshot_boundary

    polluted = "the answer\nUyghur: another exemplar"
    assert _trim_at_fewshot_boundary(polluted) == "the answer"


def test_trim_at_fewshot_boundary_picks_earliest():
    from shared.evaluation import _trim_at_fewshot_boundary

    polluted = "first\n\nthen\nEnglish: next"
    assert _trim_at_fewshot_boundary(polluted) == "first"


def test_trim_at_fewshot_boundary_passthrough_when_clean():
    from shared.evaluation import _trim_at_fewshot_boundary

    clean = "A complete translation."
    assert _trim_at_fewshot_boundary(clean) == clean


def test_trim_at_fewshot_boundary_is_idempotent():
    from shared.evaluation import _trim_at_fewshot_boundary

    once = _trim_at_fewshot_boundary("output\n\ngarbage")
    twice = _trim_at_fewshot_boundary(once)
    assert once == twice == "output"


def test_build_wcm_base_lm_prompt_is_flat_string_ending_in_label_cue():
    from shared.evaluation import build_wcm_base_lm_prompt

    prompt = build_wcm_base_lm_prompt(
        text="ئۇيغۇر تېكستى", labels=["1", "3", "4"]
    )
    assert isinstance(prompt, str)
    for label in ("1", "3", "4"):
        assert label in prompt
    assert "ئۇيغۇر تېكستى" in prompt
    assert prompt.rstrip().endswith("Label:")
    # The base-LM prompt must NOT contain any chat-role marker — a
    # base LM has no <|im_start|> / <|eot_id|> / etc.
    for marker in ("<|im_start|>", "<|im_end|>", "<|eot_id|>"):
        assert marker not in prompt


def test_build_wcm_base_lm_prompt_includes_exemplars():
    from shared.evaluation import build_wcm_base_lm_prompt

    exemplars = [("uy 1", "3"), ("uy 2", "1")]
    prompt = build_wcm_base_lm_prompt(
        text="target", labels=["1", "3", "4"], exemplars=exemplars
    )
    for ex_text, ex_label in exemplars:
        assert ex_text in prompt
        assert f"Label: {ex_label}" in prompt
    assert prompt.rstrip().endswith("Label:")


def _make_score_stub(positional_logits):
    """Stub model with controllable next-token logits (copy of the helper
    used in ``tests/test_evaluation_wcm.py``)."""
    import torch

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _StubModel:
        device = torch.device("cpu")

        def __call__(self, input_ids):
            T = input_ids.shape[1]
            vocab = 32
            logits = torch.full((1, T, vocab), -1e9)
            for (pos, tok), val in positional_logits.items():
                logits[0, pos, tok] = val
            return _Out(logits)

    return _StubModel()


def _make_base_lm_stub_tokenizer(label_to_tok):
    """Tokenizer stub with **no** ``apply_chat_template``: callers that
    accidentally route a base-LM variant through the chat path will raise
    ``AttributeError`` — exactly the regression we want to catch."""
    import torch

    class _Out(dict):
        def __init__(self, ids):
            super().__init__(input_ids=ids)
            self.input_ids = ids

    class _StubTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token_id = 1
        model_max_length = 2048

        def __call__(
            self,
            text,
            return_tensors="pt",
            truncation=False,
            max_length=None,
            add_special_tokens=True,
        ):
            # Map the (long) base-LM prompt to a fixed 2-token id sequence
            # so the score stub above can target position 1 deterministically.
            if text in label_to_tok:
                value = label_to_tok[text]
                ids = [value] if isinstance(value, list) else [[value]]
            else:
                ids = [[1, 2]]
            return _Out(torch.tensor(ids))

    return _StubTokenizer()


def test_classify_uyghur_base_lm_picks_argmax_label():
    """End-to-end on the ``prompt_style="base_lm"`` path."""
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    labels = ["1", "3", "4"]
    label_to_tok = {"1": 11, "3": 13, "4": 14}
    tokenizer = _make_base_lm_stub_tokenizer(label_to_tok)
    # Position 1 is the prediction position for the first label token.
    model = _make_score_stub({
        (1, 11): 0.5,
        (1, 13): 4.0,
        (1, 14): 1.5,
    })

    pred = _classify_uyghur(
        model, tokenizer, text="anything", labels=labels, prompt_style="base_lm"
    )
    assert pred == "3"


def test_classify_uyghur_rejects_unknown_prompt_style():
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    tokenizer = _make_base_lm_stub_tokenizer({"x": 5})
    model = _make_score_stub({})
    with pytest.raises(ValueError):
        _classify_uyghur(
            model, tokenizer, text="t", labels=["x"], prompt_style="bogus"
        )
