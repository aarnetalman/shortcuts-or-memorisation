"""Share kept without evidence, per BioASQ year (train, 12B, 13B, 14B), with bootstrap 95% CIs.
factoid: strict_noctx/strict_clean, list: F1_noctx/F1_clean, yes/no: (AUC_noctx-.5)/(AUC_clean-.5).
Writes results/tables/year_split.csv (aggregate only). 1000 resamples, seed 0."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioasq_audit.metrics import factoid_scores, list_scores

def rankdata(a):
    a = np.atleast_2d(a); out = np.empty(a.shape)
    for i, row in enumerate(a):
        o = np.argsort(row, kind="mergesort"); sr = row[o]; first = np.r_[True, sr[1:] != sr[:-1]]
        grp = np.cumsum(first) - 1; st = np.flatnonzero(first); en = np.r_[st[1:], len(sr)]
        out[i, o] = ((st + en + 1) / 2.0)[grp]
    return out

def aucv(s, y):
    r = rankdata(s); p = y.sum(1); n = (~y).sum(1)
    return ((r * y).sum(1) - p * (p + 1) / 2) / (p * n)

yr = {}
for l in open("data/items.jsonl"):
    it = json.loads(l); s = it["source"]; yr[it["id"]] = "train" if s.startswith("training") else s[:3]

def load(m, c, k):
    return {(r["id"], r["type"]): r for r in map(json.loads, open(f"results/runs/{m}__{c}__{k}.jsonl"))}

rng = np.random.default_rng(0); B = 1000
out = open("results/tables/year_split.csv", "w"); out.write("model,qtype,group,n,share,ci_low,ci_high\n")
for m in ["gpt-4o", "DeepSeek-V4-Pro", "Llama-3.3-70B-Instruct"]:
    C, N = load(m, "gold", "clean"), load(m, "noctx", "noctx")
    for qt in ["yesno", "factoid", "list"]:
        ks = [k for k in C if k[1] == qt and k in N and (qt != "yesno" or all(
            d[k].get("logp_yes") is not None and d[k].get("logp_no") is not None for d in (C, N)))]
        line = []
        for g in ["train", "12B", "13B", "14B"]:
            ix = [k for k in ks if yr[k[0]] == g]; n = len(ix); bi = rng.integers(0, n, (B, n))
            if qt == "yesno":
                y = np.array([C[k]["gold"] == "yes" for k in ix])
                sc = np.array([C[k]["logp_yes"] - C[k]["logp_no"] for k in ix])
                sn = np.array([N[k]["logp_yes"] - N[k]["logp_no"] for k in ix])
                pt = ((aucv(sn[None], y[None]) - .5) / (aucv(sc[None], y[None]) - .5))[0]
                bs = (aucv(sn[bi], y[bi]) - .5) / (aucv(sc[bi], y[bi]) - .5)
            else:
                fn, key = (factoid_scores, "strict") if qt == "factoid" else (list_scores, "f1")
                vc = np.array([fn(C[k]["pred"], C[k]["gold"])[key] for k in ix])
                vn = np.array([fn(N[k]["pred"], N[k]["gold"])[key] for k in ix])
                pt = vn.mean() / vc.mean(); bs = vn[bi].mean(1) / vc[bi].mean(1)
            bs = bs[np.isfinite(bs)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            out.write(f"{m},{qt},{g},{n},{pt:.4f},{lo:.4f},{hi:.4f}\n")
            line.append(f"{g} {100*pt:4.0f} [{100*lo:3.0f},{100*hi:3.0f}] n={n}")
        print(f"{m[:15]:15s} {qt:7s} | " + " | ".join(line))
