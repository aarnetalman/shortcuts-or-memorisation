"""Tag splits once, then write every corrupted version to disk.

Layout under $NLICORRUPT_DATA:
  tagged/<task>/<split>.jsonl.gz
  corrupted/<style>/<task>/<split>/<corruption>.jsonl.gz

Records in corrupted files: {"id", "label", <text keys>}. Records dropped by a
corruption (2021 style, empty field) are left out. Files are written once;
rerunning skips existing files unless overwrite=True.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from . import paths
from .corrupt import apply, ensure_nltk_data, tag_record
from .data import load_anli_mix, load_split, read_jsonl, write_jsonl
from .tasks import get_task

log = logging.getLogger(__name__)

PAIR_KEYS = ("premise", "hypothesis")


def task_keys(task: str) -> tuple[str, ...]:
    return PAIR_KEYS if task == "anli_mix" else get_task(task).keys


def load_raw(task: str, split: str) -> list[dict]:
    if task == "anli_mix":
        if split != "train":
            raise ValueError("anli_mix only has a train split")
        return load_anli_mix()
    return load_split(task, split)


def tagged_path(task: str, split: str) -> Path:
    return paths.tagged_dir() / task / f"{split}.jsonl.gz"


def corrupted_path(task: str, split: str, corruption: str, style: str) -> Path:
    return paths.corrupted_dir() / style / task / split / f"{corruption}.jsonl.gz"


def _init_worker():
    ensure_nltk_data()


def _workers(n: int | None) -> int:
    if n:
        return n
    return os.cpu_count() or 1


def tag_split(task: str, split: str, workers: int | None = None, overwrite: bool = False) -> Path:
    out = tagged_path(task, split)
    if out.exists() and not overwrite:
        log.info("tagged %s/%s exists", task, split)
        return out
    ensure_nltk_data()
    raw = load_raw(task, split)
    if task == "anli_mix":
        # Upsampled duplicates share ids; tag each distinct record once.
        seen, uniq = set(), []
        for r in raw:
            if r["id"] not in seen:
                seen.add(r["id"])
                uniq.append(r)
        raw = uniq
    keys = task_keys(task)
    n = _workers(workers)
    log.info("tagging %s/%s: %d records with %d workers", task, split, len(raw), n)
    fn = partial(tag_record, keys=keys)
    if n > 1:
        with Pool(n, initializer=_init_worker) as pool:
            tagged = list(pool.imap(fn, raw, chunksize=256))
    else:
        tagged = [fn(r) for r in raw]
    write_jsonl(out, tagged)
    return out


def materialize(
    task: str,
    split: str,
    corruptions: list[str],
    style: str = "2022",
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Path]:
    todo = {c: corrupted_path(task, split, c, style) for c in corruptions}
    todo = {c: p for c, p in todo.items() if overwrite or not p.exists()}
    if not todo:
        return {c: corrupted_path(task, split, c, style) for c in corruptions}
    tagged = list(read_jsonl(tag_split(task, split)))
    ensure_nltk_data()
    keys = task_keys(task)
    for c, p in todo.items():
        recs = []
        for t in tagged:
            out = apply(c, t, style=style, seed=seed)
            if out is None:
                continue
            rec = {"id": t["id"], "label": t["label"]}
            rec.update({k: out[k] for k in keys})
            recs.append(rec)
        write_jsonl(p, recs)
        log.info("%s %s/%s %s: %d of %d records", style, task, split, c, len(recs), len(tagged))
    return {c: corrupted_path(task, split, c, style) for c in corruptions}


def load_corrupted(task: str, split: str, corruption: str, style: str = "2022") -> list[dict]:
    p = corrupted_path(task, split, corruption, style)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run: nlicorrupt prepare --task {task} --split {split} "
            f"--corruptions {corruption} --style {style}"
        )
    return list(read_jsonl(p))


def load_anli_mix_training(corruption: str, style: str = "2022") -> list[dict]:
    """The ANLI training mixture with upsampling applied to a corrupted file."""
    from .tasks import ANLI_MIX

    base = load_corrupted("anli_mix", "train", corruption, style)
    out = []
    for r in base:
        src = r["id"].split(":", 1)[0]
        out.extend([r] * ANLI_MIX.get(src, 1))
    return out
