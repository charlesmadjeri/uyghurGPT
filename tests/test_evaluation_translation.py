"""Unit tests for the translation decoding helpers in ``shared.evaluation``.

Guard against the UG→EN regression diagnosed in
``docs/tasks/03_ug2en_decoding_fix.md`` (Mix-20 fine-tune emitting literal
chat markers like ``<|im_end|>`` or a second ``\\nassistant\\n`` turn header
after the natural stop, collapsing chrF 30.29 → 9.38). These tests pin two
guarantees:

1. ``_stop_token_ids`` includes EOS plus every chat marker the tokenizer
   actually knows about (Qwen ChatML + LLaMA-3.1 header tokens), so
   ``model.generate`` halts as soon as the model emits one.
2. ``_clean_translation_output`` hard-trims those markers from the decoded
   hypothesis, so even literal-string variants the adapter learned to
   emit cannot pollute chrF / BLEU.

Both helpers are pure Python / pure string ops; no model load, no GPU.
"""

from __future__ import annotations


def test_stop_token_ids_includes_eos_first():
    from shared.evaluation import _stop_token_ids

    class _Tok:
        eos_token_id = 42

        def convert_tokens_to_ids(self, _marker):  # no chat markers known
            return None

    ids = _stop_token_ids(_Tok())
    assert ids[0] == 42, "EOS must always lead the stop-id list when present"
    assert len(ids) == 1


def test_stop_token_ids_adds_known_chat_markers():
    from shared.evaluation import _stop_token_ids

    known = {
        "<|im_end|>": 151645,
        "<|endoftext|>": 151643,
    }

    class _Tok:
        eos_token_id = 151643

        def convert_tokens_to_ids(self, marker):
            return known.get(marker, None)

    ids = _stop_token_ids(_Tok())
    assert 151645 in ids
    assert 151643 in ids
    assert ids[0] == 151643, "EOS still leads"
    assert len(set(ids)) == len(ids), "stop-id list must be deduplicated"


def test_stop_token_ids_skips_missing_markers():
    """``convert_tokens_to_ids`` returns ``unk_token_id`` (or None) for
    markers a tokenizer does not know. These must not be appended to the
    stop list — otherwise ``unk_token_id`` becomes a stop trigger and
    cripples generation."""
    from shared.evaluation import _stop_token_ids

    class _Tok:
        eos_token_id = 7

        def convert_tokens_to_ids(self, _marker):
            return -1  # missing-marker sentinel

    assert _stop_token_ids(_Tok()) == [7]


def test_stop_token_ids_handles_missing_eos():
    from shared.evaluation import _stop_token_ids

    class _Tok:
        eos_token_id = None

        def convert_tokens_to_ids(self, marker):
            return 99 if marker == "<|im_end|>" else None

    assert _stop_token_ids(_Tok()) == [99]


def test_clean_translation_output_trims_literal_im_end():
    from shared.evaluation import _clean_translation_output

    polluted = "Hello world.<|im_end|>"
    assert _clean_translation_output(polluted) == "Hello world."


def test_clean_translation_output_trims_second_assistant_turn():
    """Adapter learned to keep talking after the natural stop."""
    from shared.evaluation import _clean_translation_output

    polluted = (
        "The cat sat on the mat.\nassistant\nThe dog sat on the rug.\n"
    )
    assert _clean_translation_output(polluted) == "The cat sat on the mat."


def test_clean_translation_output_picks_earliest_marker():
    from shared.evaluation import _clean_translation_output

    polluted = "first<|im_end|>middle\nassistant\nlast"
    assert _clean_translation_output(polluted) == "first"


def test_clean_translation_output_strips_surrounding_whitespace():
    from shared.evaluation import _clean_translation_output

    assert _clean_translation_output("   Hello.   ") == "Hello."


def test_clean_translation_output_passthrough_when_clean():
    from shared.evaluation import _clean_translation_output

    clean = "A clean translation hypothesis."
    assert _clean_translation_output(clean) == clean


def test_clean_translation_output_is_idempotent():
    from shared.evaluation import _clean_translation_output

    once = _clean_translation_output("A.<|im_end|>B")
    twice = _clean_translation_output(once)
    assert once == twice == "A."


def test_clean_translation_output_handles_llama_eot():
    from shared.evaluation import _clean_translation_output

    polluted = "Translation here.<|eot_id|>extra garbage"
    assert _clean_translation_output(polluted) == "Translation here."


def test_chat_generate_extra_kwargs_english_gets_repetition_controls():
    from shared.evaluation import (
        _UG2EN_NO_REPEAT_NGRAM_SIZE,
        _UG2EN_REPETITION_PENALTY,
        _chat_generate_extra_kwargs,
    )

    extras = _chat_generate_extra_kwargs("English")
    assert extras == {
        "repetition_penalty": _UG2EN_REPETITION_PENALTY,
        "no_repeat_ngram_size": _UG2EN_NO_REPEAT_NGRAM_SIZE,
    }


def test_chat_generate_extra_kwargs_uyghur_target_is_empty():
    from shared.evaluation import _chat_generate_extra_kwargs

    assert _chat_generate_extra_kwargs("Uyghur") == {}
    assert _chat_generate_extra_kwargs("uyghur") == {}


def test_chat_generate_extra_kwargs_beams_off_by_default(monkeypatch):
    """Default behaviour must match Slurm 2768 — no beams unless opted in."""
    from shared.evaluation import _UG2EN_NUM_BEAMS_ENV, _chat_generate_extra_kwargs

    monkeypatch.delenv(_UG2EN_NUM_BEAMS_ENV, raising=False)
    extras = _chat_generate_extra_kwargs("English")
    assert "num_beams" not in extras
    assert "early_stopping" not in extras


def test_chat_generate_extra_kwargs_beams_enable_via_env(monkeypatch):
    """``UYGHUR_UG2EN_NUM_BEAMS=4`` adds num_beams=4 + early_stopping."""
    from shared.evaluation import (
        _UG2EN_NUM_BEAMS_ENV,
        _UG2EN_NO_REPEAT_NGRAM_SIZE,
        _UG2EN_REPETITION_PENALTY,
        _chat_generate_extra_kwargs,
    )

    monkeypatch.setenv(_UG2EN_NUM_BEAMS_ENV, "4")
    extras = _chat_generate_extra_kwargs("English")
    assert extras == {
        "repetition_penalty": _UG2EN_REPETITION_PENALTY,
        "no_repeat_ngram_size": _UG2EN_NO_REPEAT_NGRAM_SIZE,
        "num_beams": 4,
        "early_stopping": True,
    }


def test_chat_generate_extra_kwargs_beams_ignored_for_uyghur(monkeypatch):
    """Env var must not leak into EN→UG (Uyghur-script generation)."""
    from shared.evaluation import _UG2EN_NUM_BEAMS_ENV, _chat_generate_extra_kwargs

    monkeypatch.setenv(_UG2EN_NUM_BEAMS_ENV, "4")
    assert _chat_generate_extra_kwargs("Uyghur") == {}


def test_chat_generate_extra_kwargs_beams_bad_value_falls_back(monkeypatch):
    """Garbage env-var values are silently coerced to ``num_beams=1`` (no beams)."""
    from shared.evaluation import _UG2EN_NUM_BEAMS_ENV, _chat_generate_extra_kwargs

    monkeypatch.setenv(_UG2EN_NUM_BEAMS_ENV, "not-a-number")
    extras = _chat_generate_extra_kwargs("English")
    assert "num_beams" not in extras


def test_build_chat_fewshot_messages_structure():
    from shared.evaluation import _build_chat_fewshot_messages

    exemplars = [("ug1", "en1"), ("ug2", "en2"), ("ug3", "en3")]
    msgs = _build_chat_fewshot_messages("ug_target", "Uyghur", "English", exemplars)

    assert msgs[0]["role"] == "system"
    assert "Uyghur" in msgs[0]["content"]
    assert "English" in msgs[0]["content"]
    assert len(msgs) == 1 + 2 * len(exemplars) + 1

    for i, (ex_src, ex_tgt) in enumerate(exemplars):
        u = msgs[1 + 2 * i]
        a = msgs[1 + 2 * i + 1]
        assert u == {"role": "user", "content": ex_src}
        assert a == {"role": "assistant", "content": ex_tgt}

    assert msgs[-1] == {"role": "user", "content": "ug_target"}


def test_build_chat_fewshot_messages_zero_exemplars_matches_zero_shot_shape():
    """k=0 falls back to system + single user turn — same shape as ``generate_translation``."""
    from shared.evaluation import _build_chat_fewshot_messages

    msgs = _build_chat_fewshot_messages("src", "Uyghur", "English", [])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "src"}
