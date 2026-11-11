"""Do corruptions hurt less on items a shortcut can solve?

Following Gururangan et al. (2018), a hypothesis-only classifier (bag of word uni- and
bigrams, logistic regression) is trained on the task's training hypotheses and applied
to the evaluation items. Items it labels correctly are 'easy' (the hypothesis alone
carries a usable cue), the rest 'hard'. If a model's robustness to corruption comes from
such cues, its corruption drop should be smaller on easy items than on hard ones.

The comparison uses the same machinery as the seen/unseen analysis: Δ (corrupted −
clean) per group with a paired bootstrap CI, and the easy − hard gap with an unpaired
bootstrap CI, on answer-bias-corrected predictions where per-item scores exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..data import load_split, read_jsonl, write_jsonl
from ..tasks import TASKS, get_task

log = logging.getLogger(__name__)

DEFAULT_CORRUPTIONS = ["drop-NOUN", "drop-VERB", "shuffle-n1", "keep-NOUN+VERB", "hyp-only"]


def shortcut_file(task: str, root: Path | None = None) -> Path:
    return Path(root or paths.runs_dir()) / "analysis" / "shortcuts" / f"{task}.jsonl.gz"


def hypothesis_only(task: str, seed: int = 0, train=None, evaluation=None) -> list[dict]:
    """Per evaluation item: the hypothesis-only prediction and whether it is right."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    spec = get_task(task)
    if not spec.is_pair or spec.is_regression:
        raise ValueError(f"{task}: needs a sentence-pair classification task")
    h = spec.keys[-1]
    train = train if train is not None else load_split(task, spec.train_split)
    evaluation = evaluation if evaluation is not None else load_split(task, spec.eval_split)
    train = [r for r in train if int(r["label"]) >= 0]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    x = vec.fit_transform([r[h] or "" for r in train])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x, [int(r["label"]) for r in train])
    pred = clf.predict(vec.transform([r[h] or "" for r in evaluation]))
    return [{"id": r["id"], "label": int(r["label"]), "hyp_pred": int(p), "easy": int(p) == int(r["label"])}
            for r, p in zip(evaluation, pred)]


def build(tasks: list[str], force: bool = False, root: Path | None = None) -> dict[str, float]:
    """Train and apply the hypothesis-only classifier per task; returns its accuracy (%)."""
    acc = {}
    for task in tasks:
        f = shortcut_file(task, root)
        if force or not f.exists():
            rows = hypothesis_only(task)
            write_jsonl(f, rows)
        rows = list(read_jsonl(f))
        acc[task] = 100.0 * sum(r["easy"] for r in rows) / len(rows)
        log.info("%s: hypothesis-only accuracy %.1f%% (%d items)", task, acc[task], len(rows))
    return acc


def _runs_for(task: str, runs: Path):
    """(family, model, condition, run dir) for zero-shot runs that saved predictions."""
    for run in sorted(runs.glob(f"logprob/*/{task}/*/k0")):
        yield "logprob", run.parents[2].name, "k0", run
    for run in sorted(runs.glob(f"prompt/*/{task}/*/k0/reasoning-*")):
        yield "prompt", run.parents[3].name, f"k0/{run.name}", run


def table(tasks: list[str], corruptions: list[str] = DEFAULT_CORRUPTIONS, n_boot: int = 500,
          root: Path | None = None) -> pd.DataFrame:
    from .tables import group_delta

    runs = Path(root or paths.runs_dir())
    out = []
    for task in tasks:
        f = shortcut_file(task, runs)
        if not f.exists() or TASKS.get(task) is None:
            continue
        flags = {r["id"]: ("easy" if r["easy"] else "hard") for r in read_jsonl(f)}
        for family, model, cond, run in _runs_for(task, runs):
            clean = run / "clean.jsonl.gz"
            have = [c for c in corruptions if (run / f"{c}.jsonl.gz").exists()]
            if not clean.exists() or not have:
                continue
            # Centre only if the clean file and every corrupted file have answer scores.
            centered = all(any(r.get("scores") is not None for r in read_jsonl(run / f"{c}.jsonl.gz"))
                           for c in ["clean", *have])
            rows = group_delta(run, task, flags, have, centered=centered, n_boot=n_boot, labels=("easy", "hard"))
            out += [{"family": family, "model": model, "condition": cond, **r} for r in rows]
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).rename(columns={"seen": "group"})
    return df[["family", "model", "task", "condition", "centered", "corruption", "group", "n", "clean",
               "delta", "lo", "hi"]]
