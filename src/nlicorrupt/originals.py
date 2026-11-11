"""Read the corrupted datasets released with the 2021 and 2022 papers.

`nlicorrupt download` clones both repos into $NLICORRUPT_ORIG. They are used to
replicate the published numbers on exactly the published data.

Corruption names: 'released-drop-<POS>' (e.g. released-drop-NOUN).
Available:
  2021 (nli-data-sanity-check): MNLI train and dev (validation_matched), 8 classes;
        ANLI test r1-r3, 8 classes; ANLI train r1-r3, NOUN only.
  2022 (nlu-dataset-diagnostics): all 8 GLUE tasks, train and dev, NOUN only.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from . import paths
from .tasks import get_task

PREFIX = "released-"
_NLI = {"entailment": 0, "neutral": 1, "contradiction": 2, "e": 0, "n": 1, "c": 2}


def is_released(corruption: str) -> bool:
    return corruption.startswith(PREFIX)


def _pos(corruption: str) -> str:
    m = re.fullmatch(r"released-drop-([A-Z]+)", corruption)
    if not m:
        raise ValueError(f"released files only contain single-class drops, not {corruption!r}")
    return m.group(1)


def _repo(name: str) -> Path:
    p = paths.originals_root() / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `nlicorrupt download` (clones the original repos)")
    return p


def load_released(task: str, split: str, corruption: str, style: str) -> list[dict]:
    pos = _pos(corruption)
    spec = get_task(task)
    if style == "2021":
        return _load_2021(task, split, pos)
    if style == "2022":
        return _load_2022(spec, split, pos)
    raise ValueError(f"unknown style {style!r}")


def _load_2021(task: str, split: str, pos: str) -> list[dict]:
    root = _repo("nli-data-sanity-check") / "data"
    if task == "mnli":
        sub = {"train": "Corrupt-Train", "validation_matched": "Corrupt-Test"}.get(split)
        if sub is None:
            raise ValueError(f"2021 MNLI files cover train and validation_matched, not {split}")
        out = []
        csv.field_size_limit(sys.maxsize)
        with open(root / "MNLI" / sub / f"MNLI-{pos}.tsv", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)
            for row in reader:
                idx, prem, hyp, lab = row[0], row[1], row[2], row[3]
                out.append({"id": f"released2021:mnli:{split}:{idx}", "label": _NLI[lab], "premise": prem, "hypothesis": hyp})
        return out
    m = re.fullmatch(r"anli_r([123])", task)
    if m:
        r = m.group(1)
        if split == f"test_r{r}":
            f = root / "ANLI" / "Corrupt-Test" / f"r{r}" / f"test-{pos.lower()}.jsonl"
        elif split == f"train_r{r}" and pos == "NOUN":
            f = root / "ANLI" / "Corrupt-Train" / f"r{r}" / "train-noun.jsonl"
        else:
            raise ValueError(f"no released 2021 file for {task}/{split}/{pos}")
        out = []
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            out.append({"id": f"{task}:{split}:{d['uid']}", "label": _NLI[d["label"]],
                        "premise": d["premise"], "hypothesis": d["hypothesis"]})
        return out
    raise ValueError(f"no released 2021 files for task {task}")


def _load_2022(spec, split: str, pos: str) -> list[dict]:
    if pos != "NOUN":
        raise ValueError("the 2022 repo only releases NOUN-corrupted files")
    root = _repo("nlu-dataset-diagnostics") / "data"
    kind = "train" if split == spec.train_split else "dev" if split == spec.eval_split else None
    if kind is None:
        raise ValueError(f"2022 files cover {spec.train_split} and {spec.eval_split}, not {split}")
    out = []
    csv.field_size_limit(sys.maxsize)
    with open(root / f"{kind}_{spec.name}-NOUN.csv", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            lab = float(row["label"]) if spec.is_regression else int(row["label"])
            rec = {"id": f"{spec.name}:{split}:{row.get('') or i}", "label": lab}
            for k in spec.keys:
                rec[k] = row[k] or ""
            out.append(rec)
    return out


def get_records(task: str, split: str, corruption: str, style: str = "2022") -> list[dict]:
    """Corrupted records from our pipeline, or from the released files for 'released-*'."""
    if is_released(corruption):
        return load_released(task, split, corruption, style)
    from .prepare import load_anli_mix_training, load_corrupted

    if corruption == "alldrop":
        # Talman et al. (2021) MNLI-AllDrop: the original training set plus all
        # eight single-class Corrupt-Train sets.
        from .corrupt import CORRUPTION_CLASSES

        out = list(get_records(task, split, "clean", style))
        for c in CORRUPTION_CLASSES:
            out += get_records(task, split, f"drop-{c}", style)
        return out
    if task == "anli_mix":
        return load_anli_mix_training(corruption, style)
    return load_corrupted(task, split, corruption, style)
