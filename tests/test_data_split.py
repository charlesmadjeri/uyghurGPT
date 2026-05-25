"""Contract tests for the train/test split policy.

These tests must run without any HuggingFace downloads, so we monkey-patch
the heavy loaders out and stub the tokenizer. They lock in two
deep-learning hygiene invariants:

1. CUTE-P is split at parallel-pair level (no en2ug/ug2en leakage).
2. The split is reproducible from the configured seed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_hf_cache(tmp_path, monkeypatch):
    """Keep `Dataset.from_generator` from writing to the shared HF cache.

    The generator builder writes a fingerprinted arrow file + a filelock
    under ``HF_DATASETS_CACHE``; in CI sandboxes (or shared user caches)
    that directory can be read-only. Per-test tmp dirs side-step both.
    """
    cache = tmp_path / "hf_datasets_cache"
    cache.mkdir()
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))


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


def test_streaming_path_no_pair_leakage(tmp_path, monkeypatch):
    """Streaming-from-disk preprocess path also enforces pair-level split.

    Production runs hit ``_stream_cute_rows`` (not the in-memory
    generator). This locks in the same no-leakage invariant on that
    path so a future refactor cannot regress it silently.
    """
    from shared import data

    n = 200
    en_path = tmp_path / "en.txt"
    ug_path = tmp_path / "uy.txt"
    en_path.write_text("\n".join(f"en-{i}" for i in range(n)) + "\n")
    ug_path.write_text("\n".join(f"ug-{i}" for i in range(n)) + "\n")

    monkeypatch.setattr(
        data, "_local_cute_paths", lambda: (en_path, ug_path)
    )
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
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(range(n))
    # Bidirectional expansion preserved through streaming.
    assert len(ds_dict["train"]) == 2 * len(train_ids)
    assert len(ds_dict["test"]) == 2 * len(test_ids)


def test_streaming_skips_overlong_lines(tmp_path, monkeypatch):
    """Outlier CUTE-P lines are dropped before tokenization."""
    from shared import data

    n = 20
    en_lines = [f"en-{i}" for i in range(n)]
    ug_lines = [f"ug-{i}" for i in range(n)]
    en_lines[5] = "x" * 10_000
    en_path = tmp_path / "en.txt"
    ug_path = tmp_path / "uy.txt"
    en_path.write_text("\n".join(en_lines) + "\n")
    ug_path.write_text("\n".join(ug_lines) + "\n")

    monkeypatch.setattr(data, "_local_cute_paths", lambda: (en_path, ug_path))
    monkeypatch.setattr(data, "load_flan_en_only", lambda *a, **k: [])

    class Cfg:
        model = "qwen"
        mix = 0
        sample_count = None
        flan_seed = 42
        flan_subset_size = 0
        test_split_pct = 0.0
        max_seq_length = 512

    ds_dict = data.build_training_dataset(Cfg())
    ids = set()
    for row in ds_dict["train"]:
        for msg in row["messages"]:
            if msg["content"].startswith("en-"):
                ids.add(int(msg["content"].split("-")[1]))
                break
    assert 5 not in ids
    assert len(ds_dict["train"]) == 2 * (n - 1)


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
