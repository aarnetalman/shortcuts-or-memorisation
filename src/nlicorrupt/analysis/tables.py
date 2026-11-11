"""Δ tables from the collected long table (pandas).

Δ is always corrupted minus clean on the task's primary metric, so a drop is negative.

corrupt_test      same system, evaluated on corrupted vs clean data
                  (finetune: trained on clean; logprob/prompt/probe/evaluate: as is)
corrupt_train     finetune only: trained on corrupted vs clean data, evaluated on clean data
train_and_test    finetune only: trained and evaluated on corrupted data vs all clean
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import load_yaml
from ..data import read_jsonl
from ..metrics import paired_delta_ci
from ..tasks import TASKS, get_task

KEY = ["family", "model", "task", "style", "condition", "seed", "split"]


def primary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only each task's primary metric on its main evaluation split."""
    prim = {t: s.metric for t, s in TASKS.items()}
    evals = {t: s.eval_split for t, s in TASKS.items()}
    keep = df["metric"] == df["task"].map(prim)
    keep &= (df["split"] == "eval") | (df["split"] == df["task"].map(evals))
    return df[keep].copy()


def corrupt_test(df: pd.DataFrame) -> pd.DataFrame:
    d = primary(df)
    d["seed"] = d["seed"].fillna(-1)
    clean = d[d["corruption"] == "clean"][KEY + ["value"]].rename(columns={"value": "clean"})
    out = d[d["corruption"] != "clean"].merge(clean, on=KEY, how="inner")
    out["delta"] = out["value"] - out["clean"]
    out = out[~((out["family"] == "finetune") & ~out["condition"].str.startswith("train-clean"))]
    return out


def corrupt_train(df: pd.DataFrame) -> pd.DataFrame:
    d = primary(df[df["family"] == "finetune"])
    d["train_corruption"] = d["condition"].str.extract(r"^train-([^@]+)")[0]
    d["train_task"] = d["condition"].str.extract(r"@(.+)$")[0].fillna("")
    k = ["model", "task", "style", "seed", "train_task"]
    base = d[(d["train_corruption"] == "clean") & (d["corruption"] == "clean")][k + ["value"]]
    base = base.rename(columns={"value": "clean"})
    rows = d[d["train_corruption"] != "clean"].merge(base, on=k)
    ct = rows[rows["corruption"] == "clean"].assign(config="corrupt_train")
    tt = rows[rows["corruption"] == rows["train_corruption"]].assign(config="train_and_test")
    out = pd.concat([ct, tt])
    out["delta"] = out["value"] - out["clean"]
    return out


def summarise(deltas: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = deltas.groupby(by, dropna=False)
    return g.agg(value=("value", "mean"), clean=("clean", "mean"), delta=("delta", "mean"),
                 delta_sd=("delta", "std"), seeds=("delta", "size")).reset_index()


def reproduction(df: pd.DataFrame) -> pd.DataFrame:
    """Our MNLI/MRPC/... numbers next to the published ones (2022 RoBERTa, 2021 BERT)."""
    t = load_yaml("paper_targets.yaml")
    ct = summarise(corrupt_test(df), ["family", "model", "task", "style", "corruption"])
    tr = summarise(corrupt_train(df), ["model", "task", "style", "train_corruption", "config"])
    rows = []
    for year, model in (("2022", "roberta-base"), ("2021", "bert-base-uncased")):
        tgt = t[year]
        for task, clean in tgt["clean"].items():
            mine = ct[(ct.model == model) & (ct.task == task) & (ct["style"] == year)]
            if not mine.empty:
                rows.append({"paper": year, "model": model, "task": task, "config": "clean",
                             "corruption": "-", "published": clean, "ours": float(mine["clean"].iloc[0])})
            for pos in ("NOUN", "VERB"):
                if year == "2022":
                    pub = {c: tgt[pos][c].get(task) for c in ("corrupt_test", "corrupt_train", "train_and_test")}
                else:
                    trip = tgt.get(task, {}).get(pos)
                    pub = dict(zip(("corrupt_train", "corrupt_test", "train_and_test"), trip)) if trip else {}
                for name in ("drop", "released-drop"):
                    corr = f"{name}-{pos}"
                    sel = mine[mine.corruption == corr]
                    if pub.get("corrupt_test") is not None and not sel.empty:
                        rows.append({"paper": year, "model": model, "task": task, "config": "corrupt_test",
                                     "corruption": corr, "published": pub["corrupt_test"], "ours": float(sel["value"].iloc[0])})
                    for cfg in ("corrupt_train", "train_and_test"):
                        sel2 = tr[(tr.model == model) & (tr.task == task) & (tr["style"] == year)
                                  & (tr.train_corruption == corr) & (tr.config == cfg)]
                        if pub.get(cfg) is not None and not sel2.empty:
                            rows.append({"paper": year, "model": model, "task": task, "config": cfg,
                                         "corruption": corr, "published": pub[cfg], "ours": float(sel2["value"].iloc[0])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["diff"] = out["ours"] - out["published"]
    return out


def benchmark_vs_fresh(ct: pd.DataFrame, benchmarks=("mnli", "anli_r1", "anli_r2", "anli_r3"), fresh="fresh_nli"):
    """Mean Δ per system on familiar NLI benchmarks vs the fresh control items."""
    s = summarise(ct, ["family", "model", "style", "condition", "task", "corruption"])
    b = s[s.task.isin(benchmarks)].groupby(["family", "model", "style", "condition", "corruption"])["delta"].mean()
    f = s[s.task == fresh].set_index(["family", "model", "style", "condition", "corruption"])["delta"]
    out = pd.concat({"delta_benchmark": b, "delta_fresh": f}, axis=1).dropna().reset_index()
    out["gap"] = out["delta_benchmark"] - out["delta_fresh"]
    return out


def _centered_preds(recs: dict) -> dict:
    """Argmax after subtracting each answer's mean score over all items (see answer_bias)."""
    import numpy as np

    ids = list(recs)
    scored = [i for i in ids if recs[i].get("scores") is not None]
    if not scored:
        raise ValueError("no item has answer scores; use centered=False")
    # Items without scores (e.g. an API answer blocked by a content filter) are counted
    # as wrong (-1), as in answer_bias; the answer means come from the scored items.
    scores = np.array([recs[i]["scores"] for i in scored], dtype=float)
    pred = (scores - scores.mean(axis=0, keepdims=True)).argmax(axis=1)
    out = {i: -1 for i in ids}
    out.update({i: int(p) for i, p in zip(scored, pred)})
    return out


def group_delta(pred_dir: Path, task: str, flags: dict, corruptions: list[str], centered: bool = False,
                n_boot: int = 1000, seed: int = 0, labels: tuple = (True, False)) -> list[dict]:
    """Δ (corrupted − clean) separately for two groups of items, with 95% CIs.

    flags maps item id -> group (labels[0] or labels[1]; other values are ignored).
    `centered` scores the answer-bias-corrected prediction (needs per-item scores).
    For each corruption: one row per group (paired bootstrap CI), then a 'gap' row,
    Δ(labels[0]) − Δ(labels[1]), with an unpaired bootstrap CI.
    """
    import numpy as np

    from ..metrics import compute

    spec = get_task(task)
    clean = {r["id"]: r for r in read_jsonl(Path(pred_dir) / "clean.jsonl.gz")}
    pc = _centered_preds(clean) if centered else {i: r["pred"] for i, r in clean.items()}
    metric = spec.metric
    rng = np.random.default_rng(seed)
    rows = []
    for corr in corruptions:
        cor = {r["id"]: r for r in read_jsonl(Path(pred_dir) / f"{corr}.jsonl.gz")}
        px = _centered_preds(cor) if centered else {i: r["pred"] for i, r in cor.items()}
        groups = {}
        for flag in labels:
            ids = [i for i in cor if i in clean and flags.get(i) == flag and type(flags.get(i)) is type(flag)]
            if len(ids) < 10:
                continue
            groups[flag] = ids
            ys = [clean[i]["label"] for i in ids]
            d, lo, hi = paired_delta_ci(spec, ys, [pc[i] for i in ids], [px[i] for i in ids], n_boot=n_boot)
            rows.append({"task": task, "centered": centered, "corruption": corr, "seen": flag, "n": len(ids),
                         "clean": compute(spec, ys, [pc[i] for i in ids])[metric], "delta": d, "lo": lo, "hi": hi})
        if len(groups) == 2:
            def delta(ids):
                ys = [clean[i]["label"] for i in ids]
                return compute(spec, ys, [px[i] for i in ids])[metric] - compute(spec, ys, [pc[i] for i in ids])[metric]

            a, b = groups[labels[0]], groups[labels[1]]
            boot = [delta(list(rng.choice(a, len(a)))) - delta(list(rng.choice(b, len(b)))) for _ in range(n_boot)]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append({"task": task, "centered": centered, "corruption": corr, "seen": "gap",
                         "n": len(a) + len(b), "clean": np.nan, "delta": delta(a) - delta(b),
                         "lo": float(lo), "hi": float(hi)})
    return rows


def seen_unseen(pred_dir: Path, task: str, seen_file: Path, corruptions: list[str], by: str = "any",
                centered: bool = False, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Δ (corrupted − clean) on items found vs not found in pre-training data, with 95% CIs.

    pred_dir holds <corruption>.jsonl.gz prediction files (logprob or prompt runs);
    seen_file comes from `nlicorrupt contamination-infinigram`. `by` is a field name
    (e.g. premise, hypothesis) or 'any'. `centered` scores the answer-bias-corrected
    prediction (logprob runs only), which keeps a shifted answer preference from
    masquerading as a difference between the groups. A third row per corruption,
    seen='gap', is Δ(found) − Δ(not found) with an unpaired bootstrap CI.
    """
    from ..contamination.infinigram import item_flags

    cfg = load_yaml("contamination.yaml")["infinigram"]
    flags = {}
    for r in read_jsonl(seen_file):
        flags[r["id"]] = item_flags(r, cfg["n"], cfg["min_words_full_match"])[by] if "fields" in r else (
            r["seen"] if by == "any" else None)
    rows = group_delta(pred_dir, task, flags, corruptions, centered=centered, n_boot=n_boot, seed=seed)
    return pd.DataFrame([{**r, "by": by} for r in rows])


def seen_unseen_all(index: str, corruptions: list[str], n_boot: int = 500, runs: Path | None = None) -> pd.DataFrame:
    """seen_unseen for every zero-shot logprob run of a task that has infini-gram counts,
    split by each text field and by 'any', on answer-bias-corrected predictions."""
    from .. import paths

    runs = Path(runs or paths.runs_dir())
    out = []
    for seen_file in sorted((runs / "contamination" / "infinigram" / index).glob("*.jsonl.gz")):
        task = seen_file.name.removesuffix(".jsonl.gz")
        spec = TASKS.get(task)
        if spec is None or spec.is_regression:
            continue
        for run in sorted(runs.glob(f"logprob/*/{task}/*/k0")):
            have = [c for c in corruptions if (run / f"{c}.jsonl.gz").exists()]
            if not (run / "clean.jsonl.gz").exists() or not have:
                continue
            for by in [*spec.keys, "any"]:
                t = seen_unseen(run, task, seen_file, have, by=by, centered=True, n_boot=n_boot)
                out.append(t.assign(model=run.parents[2].name, style=run.parent.name, index=index))
    if not out:
        return pd.DataFrame()
    cols = ["index", "model", "task", "style", "by", "centered", "corruption", "seen", "n", "clean", "delta", "lo", "hi"]
    return pd.concat(out)[cols]


def write_all(long_csv: Path, out_dir: Path) -> dict[str, Path]:
    df = pd.read_csv(long_csv, dtype={"style": str, "condition": str})
    out_dir.mkdir(parents=True, exist_ok=True)
    ct = corrupt_test(df)
    files = {}
    tables = {
        "corrupt_test": summarise(ct, ["family", "model", "task", "style", "condition", "corruption"]),
        "corrupt_train": summarise(corrupt_train(df), ["model", "task", "style", "train_task", "train_corruption", "config"])
        if (df["family"] == "finetune").any() else pd.DataFrame(),
        "reproduction": reproduction(df) if (df["family"] == "finetune").any() else pd.DataFrame(),
        "benchmark_vs_fresh": benchmark_vs_fresh(ct),
    }
    from .answer_bias import collect as answer_bias

    tables["answer_bias"] = answer_bias()
    for name, t in tables.items():
        p = out_dir / f"{name}.csv"
        t.round(2).to_csv(p, index=False)
        files[name] = p
    (out_dir / "README.txt").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=2))
    return files

