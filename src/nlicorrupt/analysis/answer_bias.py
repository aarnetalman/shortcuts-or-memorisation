"""Scores that separate "the model lost the information" from "the model changed its answer bias".

Instruction-tuned LMs often answer "no" / "neutral" to almost every garbled input, so a drop in
accuracy can mean a shifted decision threshold rather than lost information. From the per-item
answer scores saved by `logprob` runs, for each run and corruption:

  accuracy            argmax of the answer log-probabilities (as in the main tables)
  centered_accuracy   argmax after subtracting, per corruption, each answer's mean score over
                      all items: removes a constant preference for one answer without using any
                      gold labels (in the spirit of contextual calibration, Zhao et al. 2021)
  auc                 ranking quality, threshold-free: binary tasks use the score margin of
                      label 1 over label 0; 3-way tasks the macro one-vs-rest AUC of the softmax
  top_answer_rate     share of items given the most frequent predicted answer (100 = always the same)

`prompt` runs (hosted APIs) have scores only when the request asked for logprobs (the
answer token's top log-probabilities); otherwise only accuracy and top_answer_rate.
STS-B is skipped: its prediction is already an expected score and its Pearson metric is
threshold-free.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..data import read_jsonl
from ..tasks import TASKS

COLUMNS = ["family", "model", "task", "style", "condition", "corruption", "n",
           "accuracy", "centered_accuracy", "auc", "top_answer_rate"]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def item_metrics(labels: np.ndarray, scores: np.ndarray | None, preds: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    out = {"n": len(labels), "accuracy": 100.0 * float(np.mean(preds == labels))}
    counts = Counter(preds.tolist())
    out["top_answer_rate"] = 100.0 * max(counts.values()) / len(preds) if len(preds) else np.nan
    out["centered_accuracy"] = np.nan
    out["auc"] = np.nan
    if scores is None:
        return out
    # Rows without scores (NaN; e.g. an API answer blocked by a content filter) count as
    # wrong for centred accuracy and are left out of the AUC.
    ok = ~np.isnan(scores).any(axis=1)
    centered = scores - np.nanmean(scores[ok], axis=0, keepdims=True)
    cpred = np.where(ok, np.nan_to_num(centered, nan=-np.inf).argmax(axis=1), -1)
    out["centered_accuracy"] = 100.0 * float(np.mean(cpred == labels))
    k = scores.shape[1]
    labels, scores = labels[ok], scores[ok]
    present = np.unique(labels)
    try:
        if k == 2 and len(present) == 2:
            out["auc"] = 100.0 * float(roc_auc_score(labels, scores[:, 1] - scores[:, 0]))
        elif k > 2 and len(present) == k:
            out["auc"] = 100.0 * float(roc_auc_score(labels, _softmax(scores), multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out


def _rows_for_dir(family: str, model: str, task: str, style: str, condition: str, run: Path) -> list[dict]:
    spec = TASKS.get(task)
    if spec is None or spec.is_regression:
        return []
    rows = []
    for f in sorted(run.glob("*.jsonl.gz")):
        recs = list(read_jsonl(f))
        if not recs:
            continue
        labels = np.array([int(r["label"]) for r in recs])
        preds = np.array([-1 if r.get("pred") is None else int(r["pred"]) for r in recs])
        scores = None
        if any("scores" in r for r in recs):
            k = len(next(r["scores"] for r in recs if "scores" in r))
            scores = np.array([r.get("scores") or [np.nan] * k for r in recs], dtype=float)
        m = item_metrics(labels, scores, preds)
        rows.append({"family": family, "model": model, "task": task, "style": style, "condition": condition,
                     "corruption": f.name.removesuffix(".jsonl.gz"), **m})
    return rows


def collect(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or paths.runs_dir())
    rows: list[dict] = []
    for run in sorted(root.glob("logprob/*/*/*/*")):
        if run.is_dir():
            rows += _rows_for_dir("logprob", run.parents[2].name, run.parents[1].name, run.parent.name, run.name, run)
    for run in sorted(root.glob("prompt/*/*/*/*/reasoning-*")):
        if run.is_dir():
            rows += _rows_for_dir("prompt", run.parents[3].name, run.parents[2].name, run.parents[1].name,
                                  f"{run.parent.name}/{run.name}", run)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    key = ["family", "model", "task", "style", "condition"]
    clean = df[df.corruption == "clean"].set_index(key)[["accuracy", "centered_accuracy", "auc"]]
    clean.columns = [f"{c}_clean" for c in clean.columns]
    df = df.join(clean, on=key)
    for c in ("accuracy", "centered_accuracy", "auc"):
        df[f"delta_{c}"] = df[c] - df[f"{c}_clean"]
    return df.drop(columns=[c for c in df.columns if c.endswith("_clean")])
