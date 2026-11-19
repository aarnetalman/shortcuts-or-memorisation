"""Walk $NLICORRUPT_DATA/runs and gather every metric into one long table.

Columns: family, model, task, style, condition, seed, split, corruption, metric, value, n

family     finetune | evaluate | logprob | prompt | probe
condition  finetune: train-<corruption>; evaluate: pretrained; logprob: k0 or k5-<exemplars>;
           prompt: <k>/reasoning-<off|on>; probe: train-<corruption>/layer<N>
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths

COLUMNS = ["family", "model", "task", "style", "condition", "seed", "split", "corruption", "metric", "value", "n"]


def _rows_from_metrics(metrics: dict, base: dict, label_parser) -> list[dict]:
    rows = []
    for label, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        parsed = label_parser(label)
        if parsed is None:
            continue
        n = vals.get("n")
        for m, v in vals.items():
            if m in ("n", "missing", "unparsed") or v is None:
                continue
            rows.append({**base, **parsed, "metric": m, "value": float(v), "n": n})
    return rows


def _task_split_corr(label: str):
    parts = label.split("/")
    if len(parts) != 3:
        return None
    return {"task": parts[0], "split": parts[1], "corruption": parts[2]}


def collect(root: Path | None = None) -> list[dict]:
    root = Path(root or paths.runs_dir())
    rows: list[dict] = []

    # finetune/<model>/<task[@train_task]>/<style>/train-<c>/seed<s>/metrics.json
    for f in sorted(root.glob("finetune/*/*/*/train-*/seed*/metrics.json")):
        seed_dir, cond, style, task_dir, model = f.parent, f.parent.parent, f.parents[2], f.parents[3], f.parents[4]
        train_task = task_dir.name.partition("@")[2]
        base = {"family": "finetune", "model": model.name, "style": style.name,
                "condition": cond.name + (f"@{train_task}" if train_task else ""),
                "seed": int(seed_dir.name.removeprefix("seed"))}
        rows += _rows_from_metrics(json.loads(f.read_text()), base, _task_split_corr)

    # finetune runs re-evaluated with evaluate-model write into the same metrics.json (covered above);
    # ready-made classifiers: evaluate/<model>/<style>/metrics.json
    for f in sorted(root.glob("evaluate/*/*/metrics.json")):
        base = {"family": "evaluate", "model": f.parents[1].name, "style": f.parent.name,
                "condition": "pretrained", "seed": None}
        rows += _rows_from_metrics(json.loads(f.read_text()), base, _task_split_corr)

    # logprob/<model>/<task>/<style>/<k>/metrics.json  (keys: corruption)
    for f in sorted(root.glob("logprob/*/*/*/*/metrics.json")):
        k, style, task, model = f.parent, f.parents[1], f.parents[2], f.parents[3]
        base = {"family": "logprob", "model": model.name, "task": task.name, "style": style.name,
                "condition": k.name, "seed": None, "split": "eval"}
        rows += _rows_from_metrics(json.loads(f.read_text()), base, lambda c: {"corruption": c})

    # prompt/<model>/<task>/<style>/<k>/reasoning-<x>/metrics.json
    for f in sorted(root.glob("prompt/*/*/*/*/reasoning-*/metrics.json")):
        r, k, style, task, model = f.parent, f.parents[1], f.parents[2], f.parents[3], f.parents[4]
        base = {"family": "prompt", "model": model.name, "task": task.name, "style": style.name,
                "condition": f"{k.name}/{r.name}", "seed": None, "split": "eval"}
        rows += _rows_from_metrics(json.loads(f.read_text()), base, lambda c: {"corruption": c})

    # probe/<model>/<task>/<style>/train-<c>/metrics.json  (keys: layer<N>/<corruption>)
    for f in sorted(root.glob("probe/*/*/*/train-*/metrics.json")):
        tc, style, task, model = f.parent, f.parents[1], f.parents[2], f.parents[3]
        base = {"family": "probe", "model": model.name, "task": task.name, "style": style.name,
                "seed": None, "split": "eval"}

        def parse(label, tc=tc.name):
            layer, _, corr = label.partition("/")
            return {"condition": f"{tc}/{layer}", "corruption": corr} if corr else None

        rows += _rows_from_metrics(json.loads(f.read_text()), base, parse)
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLUMNS})
    return path
