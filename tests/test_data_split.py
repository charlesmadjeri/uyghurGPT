"""Contract tests for the train/test split policy.

These tests must run without any HuggingFace downloads, so we monkey-patch
the heavy loaders out and stub the tokenizer. They lock in two
deep-learning hygiene invariants:

1. CUTE-P is split at parallel-pair level (no en2ug/ug2en leakage).
2. The split is reproducible from the configured seed.
"""

from __future__ import annotations

import pytest


def test_flan_count_for_mix_basic():
    from shared.data import _flan_count_for_mix

    assert _flan_count_for_mix(100, 0) == 0
    assert _flan_count_for_mix(80, 20) == 20
    assert _flan_count_for_mix(50, 50) == 50


def test_flan_count_for_mix_rejects_100():
    from shared.data import _flan_count_for_mix

    with pytest.raises(ValueError):
        _flan_count_for_mix(100, 100)


def test_pair_split_disjoint_and_sized():
    from shared.data import _split_pair_indices

    train, test = _split_pair_indices(1000, 0.05, seed=42)
    assert len(train) + len(test) == 1000
    assert len(test) == 50
    assert set(train).isdisjoint(set(test))
    # No duplicates within a split.
    assert len(set(train)) == len(train)
    assert len(set(test)) == len(test)


def test_pair_split_reproducible_from_seed():
    from shared.data import _split_pair_indices

    a = _split_pair_indices(1000, 0.05, seed=42)
    b = _split_pair_indices(1000, 0.05, seed=42)
    c = _split_pair_indices(1000, 0.05, seed=43)
    assert a == b
    assert a != c


def test_pair_split_zero_pct():
    from shared.data import _split_pair_indices

    train, test = _split_pair_indices(100, 0.0, seed=42)
    assert train == list(range(100))
    assert test == []


def test_pair_split_too_few_pairs():
    from shared.data import _split_pair_indices

    train, test = _split_pair_indices(5, 0.2, seed=42)
    assert train == [0, 1, 2, 3, 4]
    assert test == []


def test_no_pair_leakage_after_bidirectional_expansion(monkeypatch):
    """The critical invariant: a CUTE-P pair appears in either train OR test, never both."""
    from shared import data

    n = 200
    en = [f"en-{i}" for i in range(n)]
    ug = [f"ug-{i}" for i in range(n)]
    monkeypatch.setattr(data, "load_cute_parallel_lines", lambda *a, **k: (en, ug))
    monkeypatch.setattr(data, "load_flan_en_only", lambda *a, **k: [])

    class Cfg:
        model = "qwen"
        mix = 0
        sample_count = None
        flan_seed = 42
        flan_subset_size = 0
        test_split_pct = 0.1

    ds_dict = data.build_training_dataset(Cfg())

    def pair_ids(split):
        """Recover the pair index from the EN side, which appears as user
        message in en2ug and as assistant message in ug2en."""
        ids = set()
        for row in split:
            for msg in row["messages"]:
                content = msg["content"]
                if content.startswith("en-"):
                    ids.add(int(content.split("-")[1]))
                    break
        return ids

    train_ids = pair_ids(ds_dict["train"])
    test_ids = pair_ids(ds_dict["test"])
    assert train_ids.isdisjoint(test_ids), (
        f"pair leakage: {len(train_ids & test_ids)} pair(s) in both splits"
    )
    assert train_ids | test_ids == set(range(n))


def test_each_pair_appears_in_both_directions(monkeypatch):
    """Bidirectional expansion: every kept pair produces exactly one en2ug AND one ug2en."""
    from shared import data
    from collections import Counter

    n = 50
    en = [f"en-{i}" for i in range(n)]
    ug = [f"ug-{i}" for i in range(n)]
    monkeypatch.setattr(data, "load_cute_parallel_lines", lambda *a, **k: (en, ug))
    monkeypatch.setattr(data, "load_flan_en_only", lambda *a, **k: [])

    class Cfg:
        model = "qwen"
        mix = 0
        sample_count = None
        flan_seed = 42
        flan_subset_size = 0
        test_split_pct = 0.0

    ds_dict = data.build_training_dataset(Cfg())
    tasks = Counter(row["task"] for row in ds_dict["train"])
    assert tasks["en2ug"] == n
    assert tasks["ug2en"] == n
