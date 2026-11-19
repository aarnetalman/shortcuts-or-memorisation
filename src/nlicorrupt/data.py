"""Loading task splits as plain records, and JSONL(.gz) helpers.

A record is a dict: {"id": str, "label": int | float, <text keys>: str}.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .tasks import ANLI_MIX, FEVER_NLI_REV, SNLI_REV, TaskSpec, get_task


def open_text(path: Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file with the same compression, then rename atomically.
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    n = 0
    with open_text(tmp, "wt") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open_text(Path(path)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_hf(path: str, config: str | None, split: str, revision: str):
    from datasets import load_dataset

    return load_dataset(path, config, split=split, revision=revision)


def load_split(task: str | TaskSpec, split: str) -> list[dict]:
    """Load a split of a registered task as records, in the dataset's own order."""
    spec = get_task(task) if isinstance(task, str) else task
    if spec.name == "anli_mix":
        raise ValueError("use load_anli_mix() for the ANLI training mixture")
    if spec.hf_path.startswith("local:"):
        from .fresh import load_local

        return load_local(spec, split)
    ds = _load_hf(spec.hf_path, spec.hf_config, split, spec.revision)
    out = []
    for i, row in enumerate(ds):
        rid = row.get("idx", row.get("uid", i))
        rec = {"id": f"{spec.name}:{split}:{rid}", "label": row["label"]}
        for k in spec.keys:
            rec[k] = row[k]
        out.append(rec)
    return out


def load_anli_mix(weights: dict[str, int] | None = None) -> list[dict]:
    """Training data of Nie et al. (2020): SNLI + MNLI + FEVER-NLI + ANLI R1-R3.

    ANLI rounds are repeated according to `weights` (default 10/20/10). Records use
    premise/hypothesis keys. In the hub copy of FEVER-NLI the claim is stored as
    'premise' and the evidence as 'hypothesis'; we swap them so the evidence is
    the premise, as in Nie et al.
    """
    weights = dict(ANLI_MIX if weights is None else weights)
    out: list[dict] = []

    def add(records, w):
        for _ in range(w):
            out.extend(records)

    if weights.get("snli"):
        ds = _load_hf("stanfordnlp/snli", None, "train", SNLI_REV)
        recs = [
            {"id": f"snli:train:{i}", "label": r["label"], "premise": r["premise"], "hypothesis": r["hypothesis"]}
            for i, r in enumerate(ds)
            if r["label"] in (0, 1, 2)
        ]
        add(recs, weights["snli"])
    if weights.get("mnli"):
        add(load_split("mnli", "train"), weights["mnli"])
    if weights.get("fever_nli"):
        ds = _load_hf("pietrolesci/nli_fever", None, "train", FEVER_NLI_REV)
        recs = [
            {"id": f"fever_nli:train:{r['fid']}", "label": r["label"], "premise": r["hypothesis"], "hypothesis": r["premise"]}
            for r in ds
        ]
        add(recs, weights["fever_nli"])
    for r in (1, 2, 3):
        w = weights.get(f"anli_r{r}", 0)
        if w:
            add(load_split(f"anli_r{r}", f"train_r{r}"), w)
    return out
