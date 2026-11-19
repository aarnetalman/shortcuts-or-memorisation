"""Linear probes on frozen LM hidden states.

The in-model analogue of the 2022 configurations: a probe trained on clean
training items and tested on corrupted evaluation items (Corrupt-Test), trained
on corrupted items and tested on clean ones (Corrupt-Train), or both corrupted.

extract: last-token hidden states of the zero-shot prompt (without the answer)
  -> runs/hidden/<model>/<task>/<style>/<split>/<corruption>.npz  (X [n, layers, d] float16, labels, ids)
probe:   logistic regression (ridge for STS-B) per layer
  -> runs/probe/<model>/<task>/<style>/train-<corruption>/metrics.json
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import paths
from .config import model_id
from .metrics import compute
from .originals import get_records
from .runinfo import manifest, write_json
from .tasks import get_task

log = logging.getLogger(__name__)


def hidden_path(model: str, task: str, style: str, split: str, corruption: str, n: int | None = None) -> Path:
    """n: size of a stratified evaluation subsample, recorded in the file name."""
    suffix = f".n{n}" if n else ""
    return paths.runs_dir() / "hidden" / model.replace("/", "__") / task / style / split / f"{corruption}{suffix}.npz"


def subsample_ids(task: str, split: str, n: int, seed: int = 0, style: str = "2022") -> set[str]:
    """Ids of a fixed label-stratified subsample of a split, drawn from its clean version.

    The same items are used for every corruption and every system.
    """
    spec = get_task(task)
    clean = get_records(task, split, "clean", style)
    return {r["id"] for r in stratified_subsample(clean, n, seed, spec.is_regression)}


def stratified_subsample(records: list[dict], n: int | None, seed: int, regression: bool) -> list[dict]:
    if not n or len(records) <= n:
        return records
    rng = np.random.default_rng(seed)
    if regression:
        idx = rng.choice(len(records), n, replace=False)
        return [records[i] for i in sorted(idx)]
    by = {}
    for i, r in enumerate(records):
        by.setdefault(int(r["label"]), []).append(i)
    keep = []
    for lab, idxs in sorted(by.items()):
        k = round(n * len(idxs) / len(records))
        keep += list(rng.choice(idxs, min(k, len(idxs)), replace=False))
    return [records[i] for i in sorted(keep)]


def extract(
    model: str,
    task: str,
    split: str,
    corruptions: list[str],
    style: str = "2022",
    max_examples: int | None = None,
    seed: int = 0,
    batch_size: int = 8,
    fmt: str = "auto",
    layers: list[int] | None = None,
    lm=None,
    tag_n: bool = False,
) -> list[Path]:
    """tag_n: put the subsample size in the file name (used for evaluation subsamples)."""
    from .lm import build_prompt, default_layers, hidden_states, load_lm

    spec = get_task(task)
    outs = []
    for corr in corruptions:
        p = hidden_path(model, task, style, split, corr, max_examples if tag_n else None)
        outs.append(p)
        if p.exists():
            continue
        lm = lm or load_lm(model_id(model), fmt=fmt)
        layers = layers or default_layers(lm)
        recs = get_records(task, split, corr, style)
        # Same items for every corruption: subsample by id from the clean split.
        if max_examples:
            keep = subsample_ids(task, split, max_examples, seed, style)
            recs = [r for r in recs if r["id"] in keep]
        prompts = [build_prompt(lm, task, r) for r in recs]
        X = hidden_states(lm, prompts, layers, batch_size=batch_size)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, X=X, labels=np.array([r["label"] for r in recs], dtype=float),
            ids=np.array([r["id"] for r in recs]), layers=np.array(layers),
        )
        log.info("hidden %s %s/%s %s: %s", model, task, split, corr, X.shape)
    return outs


def _load(p: Path):
    d = np.load(p, allow_pickle=False)
    return d["X"].astype(np.float32), d["labels"], list(d["ids"]), list(d["layers"])


def train_and_eval(
    model: str,
    task: str,
    train_corruption: str,
    eval_corruptions: list[str],
    style: str = "2022",
    C: float = 1.0,
    seed: int = 0,
    eval_n: int | None = None,
) -> Path:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    spec = get_task(task)
    Xtr, ytr, _, layers = _load(hidden_path(model, task, style, spec.train_split, train_corruption))
    evals = {c: _load(hidden_path(model, task, style, spec.eval_split, c, eval_n)) for c in eval_corruptions}
    metrics: dict[str, dict] = {}
    for li, layer in enumerate(layers):
        if spec.is_regression:
            clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0 / C))
            clf.fit(Xtr[:, li], ytr)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
            clf.fit(Xtr[:, li], ytr.astype(int))
        for c, (Xe, ye, _, _) in evals.items():
            pred = clf.predict(Xe[:, li])
            pred = pred.astype(float).tolist() if spec.is_regression else pred.astype(int).tolist()
            golds = ye.tolist() if spec.is_regression else ye.astype(int).tolist()
            metrics[f"layer{layer}/{c}"] = compute(spec, golds, pred)
    out = paths.runs_dir() / "probe" / model.replace("/", "__") / task / style / f"train-{train_corruption}"
    write_json(out / "metrics.json", metrics)
    write_json(out / "manifest.json", manifest({
        "model": model, "task": task, "train_corruption": train_corruption,
        "eval_corruptions": eval_corruptions, "style": style, "C": C, "seed": seed, "eval_items": eval_n,
        "train_examples": int(len(ytr)), "layers": [int(x) for x in layers]}))
    return out
