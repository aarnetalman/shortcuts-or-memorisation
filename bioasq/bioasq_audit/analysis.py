"""Produce the tables of the audit from the result files.

Table 1  cue network vs majority vs hypothesis-only LR, per yes/no set
Table 2  per model and condition: clean score, R per corruption (yes/no, centred where logprobs exist)
Table 3  easy minus hard drop per model and corruption (cue-net split), bootstrap CIs
Table 4  factoid and list: noctx, gold, corrupted; evidence contribution and R relative to it

Result files are results/{model}__{cond}__{corruption}.jsonl from the runners.
"""
import argparse
import csv
import re
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import metrics as M


def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def parse_name(path: str):
    base = os.path.basename(path)[:-6]
    model, cond, corr = base.split("__")
    return model, cond, corr


def yesno_table(results_dir: str, cue: Dict[str, bool], out_dir: str):
    rows, splits = [], []
    files = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    by_model = defaultdict(dict)
    for p in files:
        model, cond, corr = parse_name(p)
        recs = [r for r in read_jsonl(p) if r["type"] == "yesno"]
        if not recs:
            continue
        key = "noctx" if cond == "noctx" else corr
        by_model[model][key] = recs

    for model, runs in by_model.items():
        if "clean" not in runs:
            continue
        clean = runs["clean"]
        golds = [r["gold"] for r in clean]
        has_lp = all(r.get("logp_yes") is not None for r in clean)

        def scores(recs):
            g = [r["gold"] for r in recs]
            raw = [r["pred"] for r in recs]
            acc = M.yesno_accuracy(raw, g)
            f1 = M.macro_f1(raw, g)
            if has_lp and all(r.get("logp_yes") is not None for r in recs):
                cen = M.centred_preds([r["logp_yes"] for r in recs], [r["logp_no"] for r in recs])
                cacc = M.yesno_accuracy(cen, g)
                a = M.auc([r["logp_yes"] - r["logp_no"] for r in recs], g)
            else:
                cen, cacc, a = raw, None, None
            return acc, f1, cacc, a, M.majority_share(raw), cen

        c_acc, c_f1, c_cacc, c_auc, c_maj, c_cen = scores(clean)
        for key, recs in runs.items():
            acc, f1, cacc, a, maj, cen = scores(recs)
            base = c_cacc if cacc is not None else c_acc
            cur = cacc if cacc is not None else acc
            rows.append({"model": model, "corruption": key, "n": len(recs), "acc": round(acc, 3),
                         "macro_f1": round(f1, 3), "centred_acc": None if cacc is None else round(cacc, 3),
                         "auc": None if a is None else round(a, 3), "majority_share": round(maj, 3),
                         "R_acc": None if key == "clean" else round(M.retained(cur, base, 0.5) or float("nan"), 1)})
            # easy/hard split on the cue net, drop in (centred) accuracy relative to clean, per item
            if key == "clean":
                continue
            clean_by_id = {r["id"]: ok for r, ok in zip(clean, [p == g for p, g in zip(c_cen, golds)])}
            cur_by_id = {r["id"]: ok for r, ok in zip(recs, [p == g for p, g in zip(cen, [r["gold"] for r in recs])])}
            easy_drop, hard_drop = [], []
            for i, ok_clean in clean_by_id.items():
                if i not in cur_by_id or i not in cue:
                    continue
                d = float(ok_clean) - float(cur_by_id[i])
                (easy_drop if cue[i] else hard_drop).append(d)
            if easy_drop and hard_drop:
                rng = np.random.default_rng(0)
                e, h = np.array(easy_drop), np.array(hard_drop)
                diffs = [h[rng.integers(0, len(h), len(h))].mean() - e[rng.integers(0, len(e), len(e))].mean()
                         for _ in range(500)]
                splits.append({"model": model, "corruption": key, "n_easy": len(e), "n_hard": len(h),
                               "easy_drop_pts": round(100 * e.mean(), 1), "hard_drop_pts": round(100 * h.mean(), 1),
                               "hard_minus_easy": round(100 * (h.mean() - e.mean()), 1),
                               "ci_low": round(100 * np.percentile(diffs, 2.5), 1),
                               "ci_high": round(100 * np.percentile(diffs, 97.5), 1)})
    write_csv(os.path.join(out_dir, "table2_yesno_retained.csv"), rows)
    write_csv(os.path.join(out_dir, "table3_easy_hard.csv"), splits)


def factoid_list_table(results_dir: str, out_dir: str):
    rows = []
    by_model = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        model, cond, corr = parse_name(p)
        key = "noctx" if cond == "noctx" else corr
        recs = read_jsonl(p)
        for qtype in ("factoid", "list"):
            sub = [r for r in recs if r["type"] == qtype]
            if sub:
                by_model[(model, qtype)][key] = sub
    for (model, qtype), runs in by_model.items():
        def mean_score(recs):
            if qtype == "factoid":
                return float(np.mean([M.factoid_scores(r["pred"], r["gold"])["strict"] for r in recs]))
            return float(np.mean([M.list_scores(r["pred"], r["gold"])["f1"] for r in recs]))
        gold = mean_score(runs["clean"]) if "clean" in runs else None
        floor = mean_score(runs["noctx"]) if "noctx" in runs else None
        for key, recs in runs.items():
            s = mean_score(recs)
            r_ev = None
            if gold is not None and floor is not None and key not in ("clean", "noctx") and gold > floor:
                r_ev = round(100 * (s - floor) / (gold - floor), 1)
            rows.append({"model": model, "type": qtype, "corruption": key, "n": len(recs),
                         "score": round(s, 3), "metric": "strict_acc" if qtype == "factoid" else "mean_f1",
                         "evidence_gain": None if gold is None or floor is None else round(gold - floor, 3),
                         "R_vs_evidence": r_ev})
    write_csv(os.path.join(out_dir, "table4_factoid_list.csv"), rows)


def cue_table(cue_path: str, items_path: str, out_dir: str):
    """Cue net vs majority vs hypothesis-only LR, per source set. LR is trained on
    the training-set questions and evaluated on every other set (and by 5-fold on
    the training set itself)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    cue = read_jsonl(cue_path)
    items = {it["id"]: it for it in read_jsonl(items_path) if it["type"] == "yesno"}
    by_src = defaultdict(list)
    for r in cue:
        by_src[re.sub(r"^(\d+B)\d_golden\.json$", r"\1_golden", r["source"])].append(r)
    by_src["gold_all"] = [r for k, v in by_src.items() if k.endswith("_golden") for r in v]
    train_src = next((s for s in by_src if s.startswith("training")), None)
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
    lr_preds = {}
    if train_src:
        tr = by_src[train_src]
        X = vec.fit_transform([items[r["id"]]["question"] for r in tr])
        y = [r["gold"] for r in tr]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        cv = cross_val_predict(clf, X, y, cv=5)
        lr_preds.update({r["id"]: p for r, p in zip(tr, cv)})
        clf.fit(X, y)
        for src, recs in by_src.items():
            if src == train_src:
                continue
            Xs = vec.transform([items[r["id"]]["question"] for r in recs])
            lr_preds.update({r["id"]: p for r, p in zip(recs, clf.predict(Xs))})
    rows = []
    for src, recs in sorted(by_src.items()):
        g = [r["gold"] for r in recs]
        maj = max(set(g), key=g.count)
        cue_p = [r["pred"] for r in recs]
        m, lo, hi = M.bootstrap_ci([p == gg for p, gg in zip(cue_p, g)])
        row = {"set": src, "n": len(recs), "share_yes": round(g.count("yes") / len(g), 3),
               "majority_acc": round(g.count(maj) / len(g), 3),
               "cue_acc": round(m, 3), "cue_ci": f"[{lo:.3f}, {hi:.3f}]",
               "cue_macro_f1": round(M.macro_f1(cue_p, g), 3)}
        if lr_preds:
            lp = [lr_preds.get(r["id"]) for r in recs]
            row["lr_acc"] = round(M.yesno_accuracy(lp, g), 3)
            row["lr_macro_f1"] = round(M.macro_f1(lp, g), 3)
            row["cue_lr_agreement"] = round(float(np.mean([a == b for a, b in zip(cue_p, lp)])), 3)
        rows.append(row)
    write_csv(os.path.join(out_dir, "table1_cue_net.csv"), rows)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        print("no rows for", path)
        return
    keys = list({k: None for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cue", required=True, help="results/cue_preds.jsonl")
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cue = {r["id"]: r["cue_easy"] for r in read_jsonl(args.cue)}
    cue_table(args.cue, args.items, args.out)
    yesno_table(args.results, cue, args.out)
    factoid_list_table(args.results, args.out)


if __name__ == "__main__":
    main()
