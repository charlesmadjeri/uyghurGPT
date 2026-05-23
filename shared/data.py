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
HF_EN_PATH = f"datasets/{CUTE_DATASETS_REPO}/parallel-corpus/en.txt"
HF_UG_PATH = f"datasets/{CUTE_DATASETS_REPO}/parallel-corpus/uy.txt"
FLAN_REPO = "Muennighoff/flan"

DEFAULT_LOCAL_EN = os.path.expanduser("~/uyghurGPT/dataset/en.txt")
DEFAULT_LOCAL_UG = os.path.expanduser("~/uyghurGPT/dataset/uy.txt")


def _read_n_lines_hf(fs, path: str, n: int) -> list[str]:
    lines = []
    with fs.open(path, "r", encoding="utf-8") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def load_cute_parallel_lines(
    sample_count: int | None = None,
    en_path: str | None = None,
    ug_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load aligned EN/UG lines from local files or HuggingFace Hub."""
    en_path = en_path or DEFAULT_LOCAL_EN
    ug_path = ug_path or DEFAULT_LOCAL_UG

    if Path(en_path).is_file() and Path(ug_path).is_file():
        en_lines = Path(en_path).read_text(encoding="utf-8").splitlines()
        ug_lines = Path(ug_path).read_text(encoding="utf-8").splitlines()
        source = "local"
    else:
        from huggingface_hub import HfFileSystem

        token = os.environ.get("HF_TOKEN")
        fs = HfFileSystem(token=token)
        n = sample_count or 10_000
        en_lines = _read_n_lines_hf(fs, HF_EN_PATH, n)
        ug_lines = _read_n_lines_hf(fs, HF_UG_PATH, n)
        source = "huggingface"

    n = min(len(en_lines), len(ug_lines))
    en_lines, ug_lines = en_lines[:n], ug_lines[:n]
    if sample_count is not None:
        n = min(sample_count, n)
        en_lines, ug_lines = en_lines[:n], ug_lines[:n]

    print(f"[data] CUTE-P pairs={n} source={source}")
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

    The external/final evaluation (FLORES+ devtest, WCM-v2, C4 PPL) is
    performed by `shared/evaluation.py` and is unrelated to this split.
    """
    en_lines, ug_lines = load_cute_parallel_lines(cfg.sample_count)
    seed = cfg.flan_seed
    test_pct = float(getattr(cfg, "test_split_pct", 0.0) or 0.0)

    train_idx, test_idx = _split_pair_indices(len(en_lines), test_pct, seed)

    def _expand(indices: list[int]) -> list[dict]:
        out: list[dict] = []
        for i in indices:
            en, ug = en_lines[i], ug_lines[i]
            out.append(
                {
                    "messages": _translation_messages(en, ug, "English", "Uyghur"),
                    "task": "en2ug",
                }
            )
            out.append(
                {
                    "messages": _translation_messages(ug, en, "Uyghur", "English"),
                    "task": "ug2en",
                }
            )
        return out

    train_rows = _expand(train_idx)
    test_rows = _expand(test_idx)

    # FLAN: scale based on the TRAINING pair count, not the full corpus,
    # so the mix ratio remains accurate after holding out the test pairs.
    flan_n = _flan_count_for_mix(len(train_idx), cfg.mix)
    flan_items = load_flan_en_only(flan_n, seed, cfg.flan_subset_size)
    flan_rows = [
        {
            "messages": _flan_messages(it["EN"], it["UG"]),
            "task": "flan_en",
        }
        for it in flan_items
    ]
    if test_pct > 0.0 and len(flan_rows) >= 10:
        rng = random.Random(seed + 1)
        rng.shuffle(flan_rows)
        n_flan_test = max(1, int(round(len(flan_rows) * test_pct)))
        test_rows.extend(flan_rows[:n_flan_test])
        train_rows.extend(flan_rows[n_flan_test:])
    else:
        train_rows.extend(flan_rows)

    train_ds = _new_dataset_from_list(train_rows).shuffle(seed=seed)
    if not test_rows:
        print(f"[data] Training examples={len(train_ds)} (mix={cfg.mix}%, no test split)")
        return _new_dataset_dict(train=train_ds)

    test_ds = _new_dataset_from_list(test_rows).shuffle(seed=seed + 1)
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
