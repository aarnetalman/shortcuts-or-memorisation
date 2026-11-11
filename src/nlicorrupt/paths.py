"""Filesystem layout. Every location comes from an environment variable.

NLICORRUPT_DATA   root for corrupted data, run outputs and results (default: ./work)
NLICORRUPT_ORIG   checkout of the original 2021/2022 data repos (default: $NLICORRUPT_DATA/originals)
NLTK_DATA         NLTK resources (default: $NLICORRUPT_DATA/nltk_data)
HF_HOME           Hugging Face cache (Hugging Face default if unset)
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("NLICORRUPT_DATA", "work")).expanduser()


def originals_root() -> Path:
    return Path(os.environ.get("NLICORRUPT_ORIG", str(data_root() / "originals"))).expanduser()


def nltk_root() -> Path:
    return Path(os.environ.get("NLTK_DATA", str(data_root() / "nltk_data"))).expanduser()


def tagged_dir() -> Path:
    return data_root() / "tagged"


def corrupted_dir() -> Path:
    return data_root() / "corrupted"


def runs_dir() -> Path:
    return data_root() / "runs"


def results_dir() -> Path:
    return data_root() / "results"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
