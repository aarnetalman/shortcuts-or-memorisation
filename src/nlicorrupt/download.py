"""Fetch everything a compute job needs, so jobs can run with the hub offline.

Needs internet. Later runs can then set HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import logging
import subprocess

from . import paths
from .corrupt import ensure_nltk_data
from .tasks import ANLI_TASKS, FEVER_NLI_REV, GLUE_TASKS, SNLI_REV, get_task

log = logging.getLogger(__name__)

ORIGINAL_REPOS = {
    "nli-data-sanity-check": "https://github.com/Helsinki-NLP/nli-data-sanity-check.git",
    "nlu-dataset-diagnostics": "https://github.com/Helsinki-NLP/nlu-dataset-diagnostics.git",
}


def download_datasets(tasks=GLUE_TASKS + ANLI_TASKS, anli_mix: bool = True) -> None:
    from datasets import load_dataset

    for name in tasks:
        spec = get_task(name)
        log.info("dataset %s", name)
        load_dataset(spec.hf_path, spec.hf_config, revision=spec.revision)
    if anli_mix:
        load_dataset("stanfordnlp/snli", revision=SNLI_REV)
        load_dataset("pietrolesci/nli_fever", revision=FEVER_NLI_REV)


def download_models(model_ids: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for m in model_ids:
        from .config import split_revision

        m, rev = split_revision(m)
        log.info("model %s", m)
        snapshot_download(
            m,
            revision=rev,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "tokenizer*"],
        )


def download_originals() -> None:
    root = paths.ensure(paths.originals_root())
    for name, url in ORIGINAL_REPOS.items():
        dest = root / name
        if dest.exists():
            log.info("original repo %s present", name)
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def download_all(models: list[str] | None = None, originals: bool = True) -> None:
    ensure_nltk_data(download=True)
    download_datasets()
    if originals:
        download_originals()
    if models:
        download_models(models)
