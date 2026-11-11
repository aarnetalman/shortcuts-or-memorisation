"""Metrics. Yes/no: accuracy, macro F1, centred accuracy, AUC, majority share.
Factoid: strict accuracy, lenient accuracy, MRR. List: mean precision, recall, F1.
Retained share R = 100 (s_corrupt - c) / (s_clean - c).
"""
import re
import string
from typing import Dict, List, Optional, Sequence

import numpy as np

ARTICLES = {"a", "an", "the"}


def normalise(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(w for w in s.split() if w not in ARTICLES)
    return re.sub(r"\s+", " ", s).strip()


# yes/no -----------------------------------------------------------------

def yesno_accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    return float(np.mean([p == g for p, g in zip(preds, golds)]))


def macro_f1(preds: Sequence[str], golds: Sequence[str]) -> float:
    f1s = []
    for c in ("yes", "no"):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def centred_preds(logp_yes: Sequence[float], logp_no: Sequence[float]) -> List[str]:
    """Subtract the per-answer mean score over the set before the argmax.
    Removes a constant answer preference without using gold labels."""
    ly = np.array(logp_yes, dtype=float)
    ln = np.array(logp_no, dtype=float)
    d = (ly - ly.mean()) - (ln - ln.mean())
    return ["yes" if x > 0 else "no" for x in d]


def auc(scores: Sequence[float], golds: Sequence[str]) -> float:
    """AUC for score = logp_yes - logp_no with yes as the positive class."""
    s = np.array(scores, dtype=float)
    y = np.array([g == "yes" for g in golds])
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        idx = s == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def majority_share(preds: Sequence[str]) -> float:
    vals, counts = np.unique(np.array(preds), return_counts=True)
    return float(counts.max() / counts.sum())


def retained(s_corrupt: float, s_clean: float, chance: float) -> Optional[float]:
    if s_clean - chance <= 0:
        return None
    return 100.0 * (s_corrupt - chance) / (s_clean - chance)


# factoid ------------------------------------------------------------------

def factoid_scores(pred: List[str], gold: List[List[str]]) -> Dict[str, float]:
    gold_norm = {normalise(s) for syn in gold for s in syn}
    pn = [normalise(p) for p in pred[:5]]
    strict = float(bool(pn) and pn[0] in gold_norm)
    rank = next((i + 1 for i, p in enumerate(pn) if p in gold_norm), None)
    return {"strict": strict, "lenient": float(rank is not None),
            "rr": 1.0 / rank if rank else 0.0}


# list ---------------------------------------------------------------------

def list_scores(pred: List[str], gold: List[List[str]]) -> Dict[str, float]:
    gold_sets = [{normalise(s) for s in syn} for syn in gold]
    pn = [normalise(p) for p in pred]
    hit = [any(p in gs for gs in gold_sets) for p in pn]
    tp = sum(hit)
    covered = sum(any(p in gs for p in pn) for gs in gold_sets)
    prec = tp / len(pn) if pn else 0.0
    rec = covered / len(gold_sets) if gold_sets else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


# bootstrap ----------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0):
    v = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
