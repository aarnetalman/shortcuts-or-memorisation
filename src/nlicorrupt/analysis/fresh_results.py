"""Scores-only summary of the fresh control set, safe to export.

The fresh items stay private (they are a control only while no model has seen them), so
per-item predictions are never exported. This module reduces them to aggregate tables:

  fresh_summary.csv    per model, condition (style or generator) and perturbation: n,
                       accuracy, centred accuracy (where answer scores exist), and the
                       share of above-chance performance retained (R)
  fresh_shortcuts.csv  easy minus hard drop per model, style and perturbation, where easy
                       items are those the MNLI hypothesis-only classifier labels correctly

Conditions are the item's style (mnli, cuefree, natural-revision, human), its generator,
and 'all'. Item metadata comes from $NLICORRUPT_PRIVATE/fresh/nli.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..data import read_jsonl
from ..tasks import NLI_LABELS
from .tables import _centered_preds, group_delta

TASK = "fresh_nli"
CORE = ["drop-NOUN", "drop-VERB", "shuffle-n1", "keep-NOUN+VERB", "hyp-only"]


def item_meta(path: Path | None = None) -> dict[str, dict]:
    from ..fresh import private_root

    path = path or private_root() / "fresh" / "nli.jsonl"
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            out[f"{TASK}:test:{it['id']}"] = it
    return out


def runs(root: Path | None = None):
    root = Path(root or paths.runs_dir())
    for run in sorted(root.glob(f"logprob/*/{TASK}/*/k0")):
        yield "logprob", run.parents[2].name, run
    for run in sorted(root.glob(f"prompt/*/{TASK}/*/k0/reasoning-*")):
        yield "prompt", run.parents[3].name + ("" if run.name == "reasoning-off" else f"/{run.name}"), run


def _scored(recs: dict) -> bool:
    return any(r.get("scores") is not None for r in recs.values())


def summary(meta: dict, root: Path | None = None) -> pd.DataFrame:
    rows = []
    for family, model, run in runs(root):
        files = {f.name.removesuffix(".jsonl.gz"): {r["id"]: r for r in read_jsonl(f)}
                 for f in sorted(run.glob("*.jsonl.gz"))}
        if "clean" not in files:
            continue
        scored = all(_scored(v) for v in files.values())
        for corr, recs in files.items():
            cen = _centered_preds(recs) if scored else None
            groups = {"all": list(recs)}
            for i in recs:
                m = meta.get(i)
                if m:
                    groups.setdefault(f"style={m['style']}", []).append(i)
                    if m.get("origin") == "generated":
                        groups.setdefault(f"generator={m.get('author')}", []).append(i)
            for cond, ids in groups.items():
                labels = np.array([int(recs[i]["label"]) for i in ids])
                pred = np.array([-1 if recs[i].get("pred") is None else int(recs[i]["pred"]) for i in ids])
                row = {"family": family, "model": model, "condition": cond, "corruption": corr, "n": len(ids),
                       "accuracy": 100.0 * float(np.mean(pred == labels)) if ids else np.nan,
                       "centered_accuracy": (100.0 * float(np.mean(np.array([cen[i] for i in ids]) == labels))
                                             if cen and ids else np.nan)}
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    chance = 100.0 / len(NLI_LABELS)
    key = ["family", "model", "condition"]
    clean = df[df.corruption == "clean"].set_index(key)[["accuracy", "centered_accuracy"]]
    df = df.join(clean, on=key, rsuffix="_clean")
    measure = np.where(df.centered_accuracy.notna(), df.centered_accuracy, df.accuracy)
    base = np.where(df.centered_accuracy.notna(), df.centered_accuracy_clean, df.accuracy_clean)
    df["retained"] = 100.0 * (measure - chance) / (base - chance)
    df["retained_from"] = np.where(df.centered_accuracy.notna(), "centred", "raw")
    return df.drop(columns=["accuracy_clean", "centered_accuracy_clean"])


def shortcuts(meta: dict, root: Path | None = None, corruptions=CORE, n_boot: int = 500) -> pd.DataFrame:
    from .shortcuts import hypothesis_only

    ev = [{"id": i, "label": NLI_LABELS.index(m["label"]), "premise": m["premise"], "hypothesis": m["hypothesis"]}
          for i, m in meta.items()]
    easy = {r["id"]: ("easy" if r["easy"] else "hard") for r in hypothesis_only("mnli", evaluation=ev)}
    styles = sorted({m["style"] for m in meta.values()})
    rows = []
    for family, model, run in runs(root):
        have = [c for c in corruptions if (run / f"{c}.jsonl.gz").exists()]
        if not (run / "clean.jsonl.gz").exists() or not have:
            continue
        scored = all(_scored({r["id"]: r for r in read_jsonl(run / f"{c}.jsonl.gz")}) for c in ["clean", *have])
        for style in ["all", *styles]:
            flags = {i: e for i, e in easy.items() if style == "all" or meta[i]["style"] == style}
            for r in group_delta(run, TASK, flags, have, centered=scored, n_boot=n_boot, labels=("easy", "hard")):
                rows.append({"family": family, "model": model, "style": style, **r})
    return pd.DataFrame(rows).rename(columns={"seen": "group"}) if rows else pd.DataFrame()


def write(out_dir: Path | None = None, n_boot: int = 500) -> list[Path]:
    out_dir = Path(out_dir or paths.results_dir() / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = item_meta()
    outs = []
    for name, table in (("fresh_summary.csv", summary(meta)), ("fresh_shortcuts.csv", shortcuts(meta, n_boot=n_boot))):
        p = out_dir / name  # file names must not contain the private marker 'fresh_nli'
        table.round(2).to_csv(p, index=False)
        outs.append(p)
    return outs
