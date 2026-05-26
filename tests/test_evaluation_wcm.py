"""Unit tests for the WCM-v2 Uyghur classification helpers.

Guard against the regression we shipped in run_20260525_143722 (Slurm 2714)
where ``_classify_uyghur`` used free-form generation + substring matching
and produced below-chance accuracy on a heavily imbalanced label set
(see ``docs/PROJECT_RESULTS.md`` 2026-05-26 §Analysis).

These tests never download a model: they exercise the helpers with stub
torch modules whose logits we control, so the suite stays in the same
~2 s budget as the rest of ``tests/``.
"""

from __future__ import annotations

import math

import pytest


def test_wcm_messages_includes_labels_text_and_label_cue():
    from shared.evaluation import _wcm_messages

    msgs = _wcm_messages(text="ئۇيغۇر تېكستى", labels=["1", "3", "4"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    user = msgs[-1]["content"]
    for label in ("1", "3", "4"):
        assert label in user, f"label {label!r} missing from user message"
    assert "ئۇيغۇر تېكستى" in user
    assert user.rstrip().endswith("Label:"), (
        "user message must end on a 'Label:' cue so the next predicted token "
        "is the label being scored"
    )


def test_wcm_messages_with_exemplars_keeps_label_cue_at_end():
    from shared.evaluation import _wcm_messages

    exemplars = [("uy text one", "3"), ("uy text two", "1")]
    msgs = _wcm_messages(text="target", labels=["1", "3", "4"], exemplars=exemplars)
    user = msgs[-1]["content"]
    for ex_text, ex_label in exemplars:
        assert ex_text in user
        assert f"Label: {ex_label}" in user
    assert "Text: target" in user
    assert user.rstrip().endswith("Label:")


def _make_score_stub(positional_logits):
    """Build a stub model whose forward returns the given fixed logits.

    ``positional_logits`` maps ``(time_index, token_id) -> logit_value`` for
    the *prediction* position (i.e. the index whose argmax over vocab gives
    the next token). All other entries default to ``-1e9`` so softmax sends
    them to ~0 probability.
    """
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


def test_score_label_logprob_matches_log_softmax():
    """Joint log-likelihood == log_softmax over the prediction position."""
    pytest.importorskip("torch")
    import torch

    from shared.evaluation import _score_label_logprob

    prompt_ids = torch.tensor([[1, 2]])
    label_ids = torch.tensor([[7]])
    model = _make_score_stub({(1, 7): 2.0, (1, 8): 0.5})

    score = _score_label_logprob(model, prompt_ids, label_ids)
    expected = math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(0.5)))
    assert abs(score - expected) < 1e-4


def test_score_label_logprob_picks_higher_logit():
    """A label with higher pre-softmax logit gets the higher log-likelihood."""
    pytest.importorskip("torch")
    import torch

    from shared.evaluation import _score_label_logprob

    prompt_ids = torch.tensor([[1, 2]])
    model = _make_score_stub({(1, 7): 5.0, (1, 8): 0.0})
    score_high = _score_label_logprob(model, prompt_ids, torch.tensor([[7]]))
    score_low = _score_label_logprob(model, prompt_ids, torch.tensor([[8]]))
    assert score_high > score_low


def test_score_label_logprob_handles_multi_token_labels():
    """Two-token labels are scored as the sum of per-token log probs."""
    pytest.importorskip("torch")
    import torch

    from shared.evaluation import _score_label_logprob

    prompt_ids = torch.tensor([[1, 2]])
    # Label [21, 22]: position 1 predicts token 21, position 2 predicts 22.
    # Make both positions equally confident; total = 2 * log_softmax peak.
    model = _make_score_stub({
        (1, 21): 4.0, (1, 22): -1.0,
        (2, 22): 4.0, (2, 21): -1.0,
    })
    score = _score_label_logprob(model, prompt_ids, torch.tensor([[21, 22]]))
    per_token = math.log(math.exp(4.0) / (math.exp(4.0) + math.exp(-1.0)))
    assert abs(score - 2 * per_token) < 1e-4


def _make_stub_tokenizer(label_to_tok):
    """Stub tokenizer that ignores text content and returns deterministic ids.

    - ``apply_chat_template`` always returns the literal string ``"PROMPT"``.
    - ``tokenizer("PROMPT", ...)`` returns the prompt ids ``[[1, 2]]``.
    - ``tokenizer(label, add_special_tokens=False, ...)`` returns
      ``[[label_to_tok[label]]]`` (or a list for multi-token labels).
    """
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

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is True
            return "PROMPT"

        def __call__(
            self,
            text,
            return_tensors="pt",
            truncation=False,
            max_length=None,
            add_special_tokens=True,
        ):
            if text == "PROMPT":
                ids = [[1, 2]]
            elif text in label_to_tok:
                value = label_to_tok[text]
                ids = [value] if isinstance(value, list) else [[value]]
            else:
                ids = [[5]]
            return _Out(torch.tensor(ids))

    return _StubTokenizer()


def test_classify_uyghur_returns_argmax_label_from_candidate_set():
    """Picks the label whose tokens have the highest joint log-likelihood."""
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    labels = ["1", "3", "4"]
    label_to_tok = {"1": 11, "3": 13, "4": 14}
    tokenizer = _make_stub_tokenizer(label_to_tok)
    # prompt ids = [[1, 2]] -> the position predicting the first label token
    # is index 1 (predicts token at position 2).
    model = _make_score_stub({
        (1, 11): 1.0,
        (1, 13): 5.0,
        (1, 14): 0.0,
    })

    pred = _classify_uyghur(model, tokenizer, text="anything", labels=labels)
    assert pred == "3"


def test_classify_uyghur_always_returns_a_label_even_under_low_confidence():
    """Even with all logits near zero, the returned value is a candidate."""
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    labels = ["1", "3", "4"]
    label_to_tok = {"1": 11, "3": 13, "4": 14}
    tokenizer = _make_stub_tokenizer(label_to_tok)
    # All three candidate token ids get the same (low) logit at the
    # prediction position. Argmax is deterministic but unambiguous: the
    # first label scanned (insertion order) wins on tie-break, which is
    # exactly the documented behaviour (best_label initialized to labels[0]
    # and `>` comparison never overwrites on equality).
    model = _make_score_stub({
        (1, 11): 1.0,
        (1, 13): 1.0,
        (1, 14): 1.0,
    })

    pred = _classify_uyghur(model, tokenizer, text="anything", labels=labels)
    assert pred in labels


def test_classify_uyghur_handles_multi_token_label_without_crash():
    """Labels that tokenize to >1 BPE tokens must still score and return."""
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    labels = ["AB", "X"]
    # "AB" -> two tokens [21, 22]; "X" -> single token [29]. Token ids must
    # stay inside the stub's 32-vocab budget (see _make_score_stub).
    label_to_tok = {"AB": [21, 22], "X": 29}
    tokenizer = _make_stub_tokenizer(label_to_tok)
    # Strongly prefer "X": single-token, very high logit at the predicting position.
    model = _make_score_stub({
        (1, 29): 20.0,
        (1, 21): -5.0,
        (2, 22): -5.0,
    })

    pred = _classify_uyghur(model, tokenizer, text="anything", labels=labels)
    assert pred == "X"


def test_classify_uyghur_rejects_empty_label_set():
    pytest.importorskip("torch")

    from shared.evaluation import _classify_uyghur

    tokenizer = _make_stub_tokenizer({})
    model = _make_score_stub({})
    with pytest.raises(ValueError):
        _classify_uyghur(model, tokenizer, text="anything", labels=[])
