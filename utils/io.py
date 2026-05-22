"""Run directory layout and artifact I/O (docs/PROJECT.md §Per-run Artifacts)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir(results_root: str | Path, run_id: str) -> Path:
    return Path(results_root) / f"run_{run_id}"


def ensure_run_layout(results_root: str | Path, run_id: str) -> Path:
    root = run_dir(results_root, run_id)
    for sub in ("artifacts", "checkpoints", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_run_config(root: Path, config: dict[str, Any]) -> Path:
    path = root / "artifacts" / "run_config.json"
    write_json(path, config)
    return path


def write_run_status(root: Path, status: str, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    path = root / "artifacts" / "run_status.json"
    write_json(path, payload)
    return path


def write_eval_artifact(root: Path, benchmark: str, payload: dict[str, Any]) -> Path:
    path = root / "artifacts" / f"eval_{benchmark}.json"
    write_json(path, payload)
    return path


def preprocessed_dataset_dir(root: Path) -> Path:
    return root / "artifacts" / "preprocessed_dataset"


def checkpoint_dir(root: Path, model_label: str) -> Path:
    return root / "checkpoints" / model_label


def resolve_run_id(explicit: str | None, results_root: str | Path) -> str:
    if explicit:
        return explicit
    return new_run_id()
