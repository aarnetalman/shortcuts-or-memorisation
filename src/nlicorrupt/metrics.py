"""Task metrics as in the original papers, plus bootstrap confidence intervals.

Implemented with numpy/scipy/sklearn so results do not depend on the version of
a metrics hub. Values are on a 0-100 scale, as in the papers' tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from .tasks import TaskSpec


def _acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def _f1(y, p):
    return float(f1_score(y, p, average="binary", pos_label=1, zero_division=0))


def _mcc(y, p):
    if len(set(p)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(matthews_corrcoef(y, p))


def _pearson(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(y, p)[0])


def _spearman(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(y, p)[0])


METRICS = {"accuracy": _acc, "f1": _f1, "mcc": _mcc, "pearson": _pearson, "spearman": _spearman}


def metric_names(spec: TaskSpec) -> list[str]:
    return [spec.metric, *spec.extra_metrics]


def compute(spec: TaskSpec, labels: Sequence, preds: Sequence) -> dict[str, float]:
    """All metrics for a task, scaled to 0-100. Unparsed predictions (None) count as wrong."""
    labels = list(labels)
    preds = list(preds)
    out: dict[str, float] = {}
    if spec.is_regression:
        y = np.asarray(labels, dtype=float)
        p = np.asarray([np.nan if v is None else v for v in preds], dtype=float)
        p = np.where(np.isnan(p), np.nanmean(p) if np.isfinite(p).any() else 0.0, p)
    else:
        y = np.asarray(labels, dtype=int)
        p = np.asarray([-1 if v is None else v for v in preds], dtype=int)
    for m in metric_names(spec):
        out[m] = 100.0 * METRICS[m](y, p)
    out["n"] = len(labels)
    return out


def bootstrap_ci(
    spec: TaskSpec,
    labels: Sequence,
    preds: Sequence,
    metric: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    metric = metric or spec.metric
    labels = list(labels)
    preds = list(preds)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(compute(spec, [labels[i] for i in idx], [preds[i] for i in idx])[metric])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_delta_ci(
    spec: TaskSpec,
    labels: Sequence,
    preds_a: Sequence,
    preds_b: Sequence,
    metric: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Δ = metric(b) − metric(a) on the same items, with a paired bootstrap 95% CI."""
    metric = metric or spec.metric
    labels, preds_a, preds_b = list(labels), list(preds_a), list(preds_b)
    delta = compute(spec, labels, preds_b)[metric] - compute(spec, labels, preds_a)[metric]
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = [labels[i] for i in idx]
        vals.append(
            compute(spec, ys, [preds_b[i] for i in idx])[metric] - compute(spec, ys, [preds_a[i] for i in idx])[metric]
        )
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(delta), float(lo), float(hi)
