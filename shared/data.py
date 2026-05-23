"""CUTE-P + FLAN loading and instruction formatting for fine-tuning.

Data shape on disk (`artifacts/preprocessed_dataset/`):
    DatasetDict(
        train=Dataset(features={"messages": list[dict], "task": str}),
        test =Dataset(features={"messages": list[dict], "task": str}),  # optional
    )

The conversational ``messages`` schema is the modern TRL format and unlocks
``SFTConfig(assistant_only_loss=True)`` for native prompt-token masking (no
custom collator required). ``shared/training.py`` falls back to templating
on the fly if the installed TRL build doesn't support it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # heavy ML deps are imported lazily inside functions
    from datasets import DatasetDict

CUTE_DATASETS_REPO = "CMLI-NLP/CUTE-Datasets"
CUTE_EN_HUB_FILE = "parallel-corpus/en.txt"
CUTE_UG_HUB_FILE = "parallel-corpus/uy.txt"
FLAN_REPO = "Muennighoff/flan"

DEFAULT_LOCAL_EN = os.path.expanduser("~/uyghurGPT/dataset/en.txt")
DEFAULT_LOCAL_UG = os.path.expanduser("~/uyghurGPT/dataset/uy.txt")

# Per message field, before chat templating (~4 chars/token is conservative).
_CHARS_PER_TOKEN_ESTIMATE = 4


def max_line_chars(max_seq_length: int) -> int:
    return max_seq_length * _CHARS_PER_TOKEN_ESTIMATE


def _line_within_budget(text: str, max_chars: int) -> bool:
    return len(text) <= max_chars


def messages_within_char_budget(example: dict, max_chars: int) -> bool:
    """True when every message ``content`` fits the preprocess char cap."""
    return all(
        len(m.get("content", "")) <= max_chars for m in example.get("messages", [])
    )


def _ensure_cute_local(en_path: str, ug_path: str) -> None:
    """Download CUTE-P EN+UG parallel files from the Hub to local paths.

    One-time per machine: ~10.9 GB total. Reuses HF cache (HF_HOME) so a
    crashed/cancelled download resumes on retry. Atomic copy into the
    target paths so a partial run never leaves a half-written file
    that ``load_cute_parallel_lines`` would silently use.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    pairs = (
        (CUTE_EN_HUB_FILE, en_path),
        (CUTE_UG_HUB_FILE, ug_path),
    )
    for hub_file, dst in pairs:
        if Path(dst).is_file() and Path(dst).stat().st_size > 0:
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] Downloading {CUTE_DATASETS_REPO}:{hub_file} -> {dst}")
        cached = hf_hub_download(
            repo_id=CUTE_DATASETS_REPO,
            filename=hub_file,
            repo_type="dataset",
            token=token,
        )
        tmp = Path(dst).with_suffix(Path(dst).suffix + ".part")
        shutil.copyfile(cached, tmp)
        tmp.replace(dst)
        size_gb = Path(dst).stat().st_size / 1e9
        print(f"[data] Wrote {dst} ({size_gb:.2f} GB)")


def load_cute_parallel_lines(
    sample_count: int | None = None,
    en_path: str | None = None,
    ug_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load aligned EN/UG lines from local files, fetching from Hub on first use.

    If either local file is missing, both CUTE-P parallel files are
    downloaded from ``CMLI-NLP/CUTE-Datasets`` and persisted under
    ``~/uyghurGPT/dataset/`` (one-time ~10.9 GB), so subsequent
    preprocess runs hit disk directly and stay memory-bounded.
    """
    en_path = en_path or DEFAULT_LOCAL_EN
    ug_path = ug_path or DEFAULT_LOCAL_UG

    if not (Path(en_path).is_file() and Path(ug_path).is_file()):
        _ensure_cute_local(en_path, ug_path)

    en_lines = Path(en_path).read_text(encoding="utf-8").splitlines()
    ug_lines = Path(ug_path).read_text(encoding="utf-8").splitlines()

    n = min(len(en_lines), len(ug_lines))
    en_lines, ug_lines = en_lines[:n], ug_lines[:n]
    if sample_count is not None:
        n = min(sample_count, n)
        en_lines, ug_lines = en_lines[:n], ug_lines[:n]

    print(f"[data] CUTE-P pairs={n} source=local path={en_path}")
    return en_lines, ug_lines


def _flan_count_for_mix(n_parallel: int, mix_pct: int) -> int:
    if mix_pct <= 0:
        return 0
    if mix_pct >= 100:
        raise ValueError("mix must be < 100 for parallel + FLAN blending")
    return int(n_parallel * mix_pct / (100 - mix_pct))


def load_flan_en_only(count: int, seed: int, max_pool: int) -> list[dict]:
    """Sample EN-only FLAN examples for catastrophic-forgetting control.

    Uses streaming to avoid downloading the entire 2.5 GB FLAN train split
    (and to dodge HF NonMatchingSplitsSizesError when only some shards are
    cached). Reservoir-style sampling over the first ``max_pool`` rows.
    """
    if count <= 0:
        return []
    from datasets import load_dataset  # heavy import deferred until needed

    pool = min(max_pool, max(count, 1))
    print(f"[data] Streaming FLAN pool={pool}, sampling {count} (seed={seed}) ...")
    ds = load_dataset(FLAN_REPO, split="train", streaming=True)

    rng = random.Random(seed)
    indices = set(rng.sample(range(pool), k=min(count, pool)))

    rows: list[dict] = []
    for i, row in enumerate(ds):
        if i >= pool:
            break
        if i not in indices:
            continue
        user = (row.get("inputs") or row.get("input") or "").strip()
        assistant = (row.get("targets") or row.get("output") or "").strip()
        if not user or not assistant:
            continue
        rows.append({"EN": user, "UG": assistant, "task": "flan_en"})
    print(f"[data] FLAN usable rows={len(rows)}")
    return rows


def _translation_messages(
    source_text: str, target_text: str, source_lang: str, target_lang: str
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful bilingual assistant. "
                f"Translate the {source_lang} input to {target_lang}."
            ),
        },
        {"role": "user", "content": source_text},
        {"role": "assistant", "content": target_text},
    ]


def _flan_messages(user_text: str, assistant_text: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful English assistant."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


# Kept for backwards compatibility / notebook use; not used in the pipeline.
def format_translation_example(tokenizer, source_text, target_text, source_lang, target_lang) -> str:
    return tokenizer.apply_chat_template(
        _translation_messages(source_text, target_text, source_lang, target_lang),
        tokenize=False,
    )


def format_flan_example(tokenizer, user_text, assistant_text) -> str:
    return tokenizer.apply_chat_template(
        _flan_messages(user_text, assistant_text), tokenize=False
    )


def _new_dataset_from_list(rows: list[dict]):
    from datasets import Dataset  # deferred heavy import

    return Dataset.from_list(rows)


def _new_dataset_dict(**splits):
    from datasets import DatasetDict  # deferred heavy import

    return DatasetDict(**splits)


def _split_pair_indices(
    n_pairs: int, test_pct: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split parallel-pair indices [0, n_pairs) into (train, test).

    Splitting at PAIR level (before bidirectional expansion) guarantees
    that the EN→UG and UG→EN halves of the same sentence pair always
    live in the same split. A row-level shuffle would put one direction
    in train and the other in test, underestimating eval_loss because
    the model has already seen the source/target tokens.
    """
    if test_pct <= 0.0 or n_pairs < 10:
        return list(range(n_pairs)), []
    rng = random.Random(seed)
    order = list(range(n_pairs))
    rng.shuffle(order)
    n_test = max(1, int(round(n_pairs * test_pct)))
    train_idx = sorted(order[n_test:])
    test_idx = sorted(order[:n_test])
    return train_idx, test_idx


def _count_lines(path: "Path") -> int:
    """Count newlines in *path*; cache result in ``<path>.lines`` so the
    full ~12 GB sweep happens only once per machine."""
    cache = Path(str(path) + ".lines")
    if cache.is_file():
        try:
            return int(cache.read_text().strip())
        except Exception:
            pass
    n = 0
    with open(path, "rb", buffering=1 << 20) as f:
        for _ in f:
            n += 1
    try:
        cache.write_text(str(n))
    except OSError:
        pass
    return n


def _local_cute_paths() -> tuple["Path", "Path"] | None:
    """Return local CUTE-P paths, downloading on first use; ``None`` if
    no local files exist and the download is unavailable (e.g. in tests
    without ``HF_TOKEN``).
    """
    en_path = Path(DEFAULT_LOCAL_EN)
    ug_path = Path(DEFAULT_LOCAL_UG)
    if not (en_path.is_file() and ug_path.is_file()):
        try:
            _ensure_cute_local(str(en_path), str(ug_path))
        except Exception as e:  # noqa: BLE001 — fall back is intentional
            print(
                f"[data] WARN: streaming source unavailable "
                f"({type(e).__name__}: {e}); using in-memory loader"
            )
            return None
    if en_path.is_file() and ug_path.is_file():
        return en_path, ug_path
    return None


def _stream_cute_rows(
    en_path,
    ug_path,
    kept_indices,
    sample_count=None,
    max_line_chars=2048,
    _skip_stats=None,
):
    """Stream bidirectional CUTE-P rows directly from disk (RAM-bounded).

    For the full ~934k-pair corpus, holding both lines lists in Python
    already costs ~12 GB; expanding to row dicts costs another
    ~30-50 GB. Streaming line-by-line keeps preprocess peak RAM in the
    low tens of MB regardless of corpus size.

    Pairs whose EN or UG field exceeds ``max_line_chars`` are dropped
  (CUTE-P has rare document-scale outliers that would otherwise trigger
    tokenizer warnings and waste tokenization time).
    """
    kept = set(kept_indices)
    with open(en_path, "r", encoding="utf-8") as fe, open(
        ug_path, "r", encoding="utf-8"
    ) as fu:
        for i, (en, ug) in enumerate(zip(fe, fu)):
            if sample_count is not None and i >= sample_count:
                break
            if i not in kept:
                continue
            en = en.rstrip("\n")
            ug = ug.rstrip("\n")
            if not (
                _line_within_budget(en, max_line_chars)
                and _line_within_budget(ug, max_line_chars)
            ):
                if _skip_stats is not None:
                    _skip_stats["pairs"] = _skip_stats.get("pairs", 0) + 1
                continue
            yield {
                "messages": _translation_messages(en, ug, "English", "Uyghur"),
                "task": "en2ug",
            }
            yield {
                "messages": _translation_messages(ug, en, "Uyghur", "English"),
                "task": "ug2en",
            }


def _cute_row_iter(
    en_lines,
    ug_lines,
    indices,
    max_line_chars=2048,
    _skip_stats=None,
):
    """In-memory bidirectional row generator (test fallback only).

    Production hits ``_stream_cute_rows`` instead; this path runs when
    local files are absent (tests with monkeypatched
    ``load_cute_parallel_lines``).
    """
    for i in indices:
        en, ug = en_lines[i], ug_lines[i]
        if not (
            _line_within_budget(en, max_line_chars)
            and _line_within_budget(ug, max_line_chars)
        ):
            if _skip_stats is not None:
                _skip_stats["pairs"] = _skip_stats.get("pairs", 0) + 1
            continue
        yield {
            "messages": _translation_messages(en, ug, "English", "Uyghur"),
            "task": "en2ug",
        }
        yield {
            "messages": _translation_messages(ug, en, "Uyghur", "English"),
            "task": "ug2en",
        }


def _flan_row_iter(flan_items, max_line_chars=2048, _skip_stats=None):
    for it in flan_items:
        en, ug = it["EN"], it["UG"]
        if not (
            _line_within_budget(en, max_line_chars)
            and _line_within_budget(ug, max_line_chars)
        ):
            if _skip_stats is not None:
                _skip_stats["flan"] = _skip_stats.get("flan", 0) + 1
            continue
        yield {
            "messages": _flan_messages(en, ug),
            "task": "flan_en",
        }


def _build_dataset_from_gen(generator, gen_kwargs: dict):
    """Stream rows into an on-disk Arrow Dataset (writer batch = 1k).

    ``Dataset.from_generator`` auto-shards on **list**-typed gen_kwargs
    for multi-process generation, which corrupts our shared line lists
    (it would slice them and pass a partial copy to each shard). Cast
    every list to a tuple so the builder treats them as opaque inputs
    and runs the generator exactly once, in-process.
    """
    from datasets import Dataset

    safe_kwargs = {
        k: tuple(v) if isinstance(v, list) else v for k, v in gen_kwargs.items()
    }
    return Dataset.from_generator(generator, gen_kwargs=safe_kwargs)


def build_training_dataset(cfg) -> "DatasetDict":
    """Build Mix-{N} instruction dataset (bidirectional CUTE-P + FLAN EN-only).

    Returns a DatasetDict with `train` and (optional) `test` splits:
      - `train`: used to update model weights during fine-tuning.
      - `test`:  held-out from the same mix, used at eval_steps intervals
                 to produce an `eval/loss` curve in TensorBoard. This is
                 the in-distribution overfit detector.

    Split policy (see `_split_pair_indices`):
      - CUTE-P split is at parallel-pair level BEFORE bidirectional
        expansion, so no source/target sentence ends up on both sides.
      - FLAN samples are row-level (no pair structure); they get an
        independent same-percentage split.

    Memory policy:
      - When local CUTE-P files exist (``_local_cute_paths``), rows are
        **streamed from disk** line-by-line and written into Arrow in
        1k-row batches; preprocess peak RAM stays under ~1 GB
        regardless of corpus size.
      - When they don't (test fixture, no local files), we fall back
        to the in-memory generator over Python lists provided by
        ``load_cute_parallel_lines`` — only safe for small samples.

    The external/final evaluation (FLORES+ devtest, WCM-v2, C4 PPL) is
    performed by ``shared/evaluation.py`` and is unrelated to this split.
    """
    from datasets import concatenate_datasets

    seed = cfg.flan_seed
    test_pct = float(getattr(cfg, "test_split_pct", 0.0) or 0.0)
    line_char_cap = max_line_chars(getattr(cfg, "max_seq_length", 512))
    skip_stats: dict[str, int] = {}

    paths = _local_cute_paths()
    if paths is not None:
        en_path, ug_path = paths
        n_en = _count_lines(en_path)
        n_ug = _count_lines(ug_path)
        n_pairs = min(n_en, n_ug)
        if cfg.sample_count is not None:
            n_pairs = min(n_pairs, cfg.sample_count)
        print(
            f"[data] CUTE-P pairs={n_pairs} source=local-stream "
            f"path={en_path}"
        )
        train_idx, test_idx = _split_pair_indices(n_pairs, test_pct, seed)
        print(
            f"[data] Building Arrow splits: pairs train={len(train_idx)} "
            f"test={len(test_idx)} (test_pct={test_pct:.2%})"
        )
        train_ds = _build_dataset_from_gen(
            _stream_cute_rows,
            {
                "en_path": str(en_path),
                "ug_path": str(ug_path),
                "kept_indices": train_idx,
                "sample_count": cfg.sample_count,
                "max_line_chars": line_char_cap,
                "_skip_stats": skip_stats,
            },
        )
        print(f"[data] CUTE train rows materialised: {len(train_ds)}")
        test_ds = None
        if test_idx:
            test_ds = _build_dataset_from_gen(
                _stream_cute_rows,
                {
                    "en_path": str(en_path),
                    "ug_path": str(ug_path),
                    "kept_indices": test_idx,
                    "sample_count": cfg.sample_count,
                    "max_line_chars": line_char_cap,
                    "_skip_stats": skip_stats,
                },
            )
            print(f"[data] CUTE test rows materialised: {len(test_ds)}")
    else:
        en_lines, ug_lines = load_cute_parallel_lines(cfg.sample_count)
        train_idx, test_idx = _split_pair_indices(len(en_lines), test_pct, seed)
        print(
            f"[data] Building Arrow splits (in-memory): pairs "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"(test_pct={test_pct:.2%})"
        )
        train_ds = _build_dataset_from_gen(
            _cute_row_iter,
            {
                "en_lines": en_lines,
                "ug_lines": ug_lines,
                "indices": train_idx,
                "max_line_chars": line_char_cap,
                "_skip_stats": skip_stats,
            },
        )
        test_ds = None
        if test_idx:
            test_ds = _build_dataset_from_gen(
                _cute_row_iter,
                {
                    "en_lines": en_lines,
                    "ug_lines": ug_lines,
                    "indices": test_idx,
                    "max_line_chars": line_char_cap,
                    "_skip_stats": skip_stats,
                },
            )
        del en_lines, ug_lines

    # FLAN scales on the TRAINING pair count, so the mix ratio is preserved
    # after holding out the test pairs.
    flan_n = _flan_count_for_mix(len(train_idx), cfg.mix)
    flan_items = load_flan_en_only(flan_n, seed, cfg.flan_subset_size)

    if flan_items:
        flan_ds = _build_dataset_from_gen(
            _flan_row_iter,
            {
                "flan_items": flan_items,
                "max_line_chars": line_char_cap,
                "_skip_stats": skip_stats,
            },
        )
        if test_pct > 0.0 and len(flan_ds) >= 10:
            flan_split = flan_ds.train_test_split(test_size=test_pct, seed=seed + 1)
            train_ds = concatenate_datasets([train_ds, flan_split["train"]])
            test_ds = (
                concatenate_datasets([test_ds, flan_split["test"]])
                if test_ds is not None
                else flan_split["test"]
            )
        else:
            train_ds = concatenate_datasets([train_ds, flan_ds])
        del flan_items, flan_ds

    if skip_stats:
        parts = []
        if skip_stats.get("pairs"):
            parts.append(f"{skip_stats['pairs']} CUTE-P pair(s)")
        if skip_stats.get("flan"):
            parts.append(f"{skip_stats['flan']} FLAN row(s)")
        print(
            f"[data] Skipped {' and '.join(parts)} over {line_char_cap} chars/field "
            f"(max_seq_length={getattr(cfg, 'max_seq_length', 512)})"
        )

    train_ds = train_ds.shuffle(seed=seed)
    if test_ds is None:
        print(f"[data] Training examples={len(train_ds)} (mix={cfg.mix}%, no test split)")
        return _new_dataset_dict(train=train_ds)

    test_ds = test_ds.shuffle(seed=seed + 1)
    print(
        f"[data] Mix={cfg.mix}% pairs(train={len(train_idx)}, test={len(test_idx)}, "
        f"test_pct={test_pct:.2%}) -> rows train={len(train_ds)}, test={len(test_ds)} "
        f"(seed={seed})"
    )
    return _new_dataset_dict(train=train_ds, test=test_ds)


def preprocess_and_save(cfg, run_root: "Path"):
    from utils.io import preprocessed_dataset_dir, write_run_status

    ds_dict = build_training_dataset(cfg)
    out = preprocessed_dataset_dir(run_root)
    ds_dict.save_to_disk(str(out))
    extra = {
        "num_train": len(ds_dict["train"]),
        "num_test": len(ds_dict.get("test", [])),
        "path": str(out),
    }
    write_run_status(run_root, "preprocessed", extra)
    print(f"[data] Saved preprocessed dataset -> {out} ({extra})")
    return out


def load_preprocessed(run_root: "Path") -> "DatasetDict":
    """Load the preprocessed splits as a DatasetDict (always normalised).

    Backward-compatible with the older single-Dataset layout: such a
    directory is loaded as `{"train": <ds>}` with no `test` split.
    """
    from datasets import DatasetDict, load_from_disk  # deferred heavy import

    from utils.io import preprocessed_dataset_dir

    path = preprocessed_dataset_dir(run_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed data at {path}. Run --mode preprocess first."
        )
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return loaded
    print(f"[data] WARN: legacy single-Dataset preprocess at {path}; no test split available.")
    return DatasetDict({"train": loaded})
