"""Uncontaminated control items, kept outside this repository until publication.

$NLICORRUPT_PRIVATE points to a clone of the private control-set repository.
Expected file: $NLICORRUPT_PRIVATE/fresh/nli.jsonl, one item per line:

  {"id": "f0001", "premise": "...", "hypothesis": "...", "label": "entailment",
   "source": "fresh",            # fresh: newly written; twin: rewrite of a benchmark item
   "twin_of": null,               # for twins: id of the original, e.g. "mnli:validation_matched:123"
   "author": "AT", "written": "2026-11-15"}

Labels: entailment, neutral, contradiction. `nlicorrupt validate-fresh` checks a file.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from .tasks import NLI_LABELS, TaskSpec

REQUIRED = ("id", "premise", "hypothesis", "label", "source")
SOURCES = ("fresh", "twin")


def private_root() -> Path:
    p = os.environ.get("NLICORRUPT_PRIVATE")
    if not p:
        raise FileNotFoundError(
            "NLICORRUPT_PRIVATE is not set. Clone the private control-set repo and point "
            "NLICORRUPT_PRIVATE at it."
        )
    return Path(p).expanduser()


def validate(items: list[dict]) -> list[str]:
    errors = []
    ids = Counter(i.get("id") for i in items)
    for n, it in enumerate(items, 1):
        missing = [k for k in REQUIRED if not it.get(k)]
        if missing:
            errors.append(f"line {n}: missing {missing}")
        if it.get("label") not in NLI_LABELS:
            errors.append(f"line {n}: label must be one of {NLI_LABELS}, got {it.get('label')!r}")
        if it.get("source") not in SOURCES:
            errors.append(f"line {n}: source must be one of {SOURCES}")
        if it.get("source") == "twin" and not it.get("twin_of"):
            errors.append(f"line {n}: twin items need twin_of")
        if ids[it.get("id")] > 1:
            errors.append(f"line {n}: duplicate id {it.get('id')!r}")
    return errors


def read_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_local(spec: TaskSpec, split: str) -> list[dict]:
    if split != spec.eval_split:
        raise ValueError(f"{spec.name} only has the {spec.eval_split!r} split")
    path = private_root() / spec.hf_path.removeprefix("local:")
    items = read_items(path)
    errors = validate(items)
    if errors:
        raise ValueError(f"{path} is invalid:\n" + "\n".join(errors[:20]))
    return [
        {"id": f"{spec.name}:{split}:{it['id']}", "label": NLI_LABELS.index(it["label"]),
         "premise": it["premise"], "hypothesis": it["hypothesis"],
         "source": it["source"], "twin_of": it.get("twin_of")}
        for it in items
    ]
