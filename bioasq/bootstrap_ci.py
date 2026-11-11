"""Bootstrap intervals for the BioASQ train/gold tables (hosted models).

Measures, all paired over the same resampled items:
  yes/no  acc share  = (acc_noctx - majority) / (acc_clean - majority)
  yes/no  AUC share  = (AUC_noctx - 0.5) / (AUC_clean - 0.5)
  factoid share      = strict_noctx / strict_clean
  list share         = F1_noctx / F1_clean
  yes/no evidence-relative R per perturbation = (AUC_pert - AUC_noctx) / (AUC_clean - AUC_noctx)
For each: train, gold, and gold minus train (independent resampling of the two groups).
1000 resamples, seed 0, percentile 95% intervals.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioasq_audit.metrics import factoid_scores, list_scores

B, SEED = 1000, 0
MODELS = ["gpt-4o", "DeepSeek-V4-Pro", "Llama-3.3-70B-Instruct"]
PERTS = ["drop-ADJ", "drop-DET", "drop-NOUN", "drop-VERB", "keep-NOUN+VERB", "shuffle-n1", "shuffle-n3"]
RUNS = "results/runs"

group = {}
for l in open("data/items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    group[it["id"]] = "train" if it["source"].startswith("training") else "gold"

def load(m, cond, corr):
    p = f"{RUNS}/{m}__{cond}__{corr}.jsonl"
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[(r["id"], r["type"])] = r
    return out

def rankdata(a, axis=1):
    """Average ranks (ties averaged) along axis 1, numpy only."""
    a = np.atleast_2d(a)
    out = np.empty(a.shape, float)
    for i, row in enumerate(a):
        o = np.argsort(row, kind="mergesort")
        sr = row[o]
        first = np.r_[True, sr[1:] != sr[:-1]]
        grp = np.cumsum(first) - 1
        starts = np.flatnonzero(first)
        ends = np.r_[starts[1:], len(sr)]
        avg = (starts + ends + 1) / 2.0
        out[i, o] = avg[grp]
    return out

def auc_vec(s, y):
    """s: (B, n) scores, y: (B, n) bool. Mann-Whitney AUC per row, ties averaged."""
    r = rankdata(s, axis=1)
    npos = y.sum(1); nneg = (~y).sum(1)
    return ((r * y).sum(1) - npos * (npos + 1) / 2) / (npos * nneg)

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile(x, 2.5), np.percentile(x, 97.5)

rng = np.random.default_rng(SEED)
rows = []
def emit(model, measure, grp, point, boot):
    lo, hi = ci(boot)
    rows.append((model, measure, grp, point, lo, hi))

for m in MODELS:
    clean, noctx = load(m, "gold", "clean"), load(m, "noctx", "noctx")
    perts = {p: load(m, "gold", p) for p in PERTS}
    # ---------- yes/no ----------
    keys = [k for k in clean if k[1] == "yesno" and k in noctx
            and all(clean[k].get(f) is not None and noctx[k].get(f) is not None for f in ("logp_yes", "logp_no"))
            and all(k in perts[p] and perts[p][k].get("logp_yes") is not None and perts[p][k].get("logp_no") is not None for p in PERTS)]
    gold_y = np.array([clean[k]["gold"] == "yes" for k in keys])
    acc_c = np.array([clean[k]["pred"] == clean[k]["gold"] for k in keys], float)
    acc_n = np.array([noctx[k]["pred"] == noctx[k]["gold"] for k in keys], float)
    sc = np.array([clean[k]["logp_yes"] - clean[k]["logp_no"] for k in keys])
    sn = np.array([noctx[k]["logp_yes"] - noctx[k]["logp_no"] for k in keys])
    sp = {p: np.array([perts[p][k]["logp_yes"] - perts[p][k]["logp_no"] for k in keys]) for p in PERTS}
    g = np.array([group[k[0]] for k in keys])
    idx = {grp: np.where(g == grp)[0] for grp in ("train", "gold")}
    idx["all"] = np.arange(len(keys))
    boots = {}
    for grp, ix in idx.items():
        bi = ix[rng.integers(0, len(ix), (B, len(ix)))]
        maj = lambda yy: np.maximum(yy.mean(-1), 1 - yy.mean(-1))
        def acc_share(ii):
            yy = gold_y[ii]; mj = maj(yy)
            return (acc_n[ii].mean(-1) - mj) / (acc_c[ii].mean(-1) - mj)
        def auc_share(ii):
            yy = gold_y[ii]
            ac = auc_vec(sc[ii], yy); an = auc_vec(sn[ii], yy)
            return (an - .5) / (ac - .5)
        pt = idx[grp][None, :]
        boots[(grp, "acc")] = acc_share(bi); emit(m, "yesno_acc_share", grp, float(acc_share(pt)[0]), boots[(grp, "acc")])
        boots[(grp, "auc")] = auc_share(bi); emit(m, "yesno_auc_share", grp, float(auc_share(pt)[0]), boots[(grp, "auc")])
        if grp == "all":
            yy = gold_y[bi]; ac = auc_vec(sc[bi], yy); an = auc_vec(sn[bi], yy)
            yp = gold_y[pt]; acp = auc_vec(sc[pt], yp); anp = auc_vec(sn[pt], yp)
            for p in PERTS:
                ap = auc_vec(sp[p][bi], yy); app = auc_vec(sp[p][pt], yp)
                emit(m, f"yesno_evidence_R_{p}", "all", float(((app - anp) / (acp - anp))[0]), (ap - an) / (ac - an))
    for meas in ("acc", "auc"):
        emit(m, f"yesno_{meas}_share", "gold-train",
             float(rows[[i for i, r in enumerate(rows) if r[0] == m and r[1] == f"yesno_{meas}_share" and r[2] == "gold"][0]][3]
                   - rows[[i for i, r in enumerate(rows) if r[0] == m and r[1] == f"yesno_{meas}_share" and r[2] == "train"][0]][3]),
             boots[("gold", meas)] - boots[("train", meas)])
    # ---------- factoid / list ----------
    for qt, fn, key in (("factoid", factoid_scores, "strict"), ("list", list_scores, "f1")):
        ks = [k for k in clean if k[1] == qt and k in noctx]
        vc = np.array([fn(clean[k]["pred"], clean[k]["gold"])[key] for k in ks], float)
        vn = np.array([fn(noctx[k]["pred"], noctx[k]["gold"])[key] for k in ks], float)
        gq = np.array([group[k[0]] for k in ks])
        bb = {}
        for grp in ("train", "gold"):
            ix = np.where(gq == grp)[0]
            bi = ix[rng.integers(0, len(ix), (B, len(ix)))]
            bb[grp] = vn[bi].mean(1) / vc[bi].mean(1)
            emit(m, f"{qt}_share", grp, float(vn[ix].mean() / vc[ix].mean()), bb[grp])
        emit(m, f"{qt}_share", "gold-train",
             float(vn[gq == "gold"].mean() / vc[gq == "gold"].mean() - vn[gq == "train"].mean() / vc[gq == "train"].mean()),
             bb["gold"] - bb["train"])

os.makedirs("results/tables", exist_ok=True)
with open("results/tables/bootstrap_ci.csv", "w") as f:
    f.write("model,measure,group,point,ci_low,ci_high\n")
    for r in rows:
        f.write("%s,%s,%s,%.4f,%.4f,%.4f\n" % r)
for r in rows:
    print("%-24s %-32s %-10s %6.1f  [%6.1f, %6.1f]" % (r[0], r[1], r[2], 100 * r[3], 100 * r[4], 100 * r[5]))
