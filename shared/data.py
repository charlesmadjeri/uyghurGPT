"""CUTE-P + FLAN loading and instruction formatting for fine-tuning."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_dataset

from shared.models import load_tokenizer

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
    """Sample EN-only FLAN examples for catastrophic-forgetting control."""
    if count <= 0:
        return []
    pool = min(max_pool, max(count, 1))
    print(f"[data] Loading FLAN pool={pool}, sampling {count} (seed={seed}) ...")
    ds = load_dataset(FLAN_REPO, split=f"train[:{pool}]")
    ds = ds.shuffle(seed=seed).select(range(min(count, len(ds))))

    rows = []
    for row in ds:
        user = (row.get("inputs") or row.get("input") or "").strip()
        assistant = (row.get("targets") or row.get("output") or "").strip()
        if not user or not assistant:
            continue
        rows.append({"EN": user, "UG": assistant, "task": "flan_en"})
    print(f"[data] FLAN usable rows={len(rows)}")
    return rows


def format_translation_example(
    tokenizer,
    source_text: str,
    target_text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    system = (
        "You are a helpful bilingual assistant. "
        f"Translate the {source_lang} input to {target_lang}."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": source_text},
        {"role": "assistant", "content": target_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_flan_example(tokenizer, user_text: str, assistant_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful English assistant."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_training_dataset(cfg):
    """Build Mix-{N} instruction dataset (bidirectional CUTE-P + FLAN EN-only)."""
    tokenizer = load_tokenizer(cfg.model)
    en_lines, ug_lines = load_cute_parallel_lines(cfg.sample_count)

    rows: list[dict] = []
    for en, ug in zip(en_lines, ug_lines):
        rows.append(
            {
                "text": format_translation_example(
                    tokenizer, en, ug, "English", "Uyghur"
                ),
                "task": "en2ug",
            }
        )
        rows.append(
            {
                "text": format_translation_example(
                    tokenizer, ug, en, "Uyghur", "English"
                ),
                "task": "ug2en",
            }
        )

    flan_n = _flan_count_for_mix(len(en_lines), cfg.mix)
    for item in load_flan_en_only(flan_n, cfg.flan_seed, cfg.flan_subset_size):
        rows.append(
            {
                "text": format_flan_example(tokenizer, item["EN"], item["UG"]),
                "task": "flan_en",
            }
        )

    ds = Dataset.from_list(rows).shuffle(seed=cfg.flan_seed)
    print(f"[data] Training examples={len(ds)} (mix={cfg.mix}%)")
    return ds


def preprocess_and_save(cfg, run_root: Path):
    from utils.io import preprocessed_dataset_dir, write_run_status

    ds = build_training_dataset(cfg)
    out = preprocessed_dataset_dir(run_root)
    ds.save_to_disk(str(out))
    write_run_status(run_root, "preprocessed", {"num_examples": len(ds), "path": str(out)})
    print(f"[data] Saved preprocessed dataset -> {out}")
    return out


def load_preprocessed(run_root: Path):
    from datasets import load_from_disk

    from utils.io import preprocessed_dataset_dir

    path = preprocessed_dataset_dir(run_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed data at {path}. Run --mode preprocess first."
        )
    return load_from_disk(str(path))
