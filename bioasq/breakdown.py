import json, os, sys
sys.path.insert(0, ".")
from bioasq_audit.metrics import factoid_scores, list_scores

MODELS = ["gpt-4o", "DeepSeek-V4-Pro", "Llama-3.3-70B-Instruct"]
CONDS = [("noctx", "noctx"), ("gold", "clean"), ("gold", "drop-NOUN"), ("gold", "keep-NOUN+VERB"), ("gold", "shuffle-n1")]

items = {}
for l in open("data/items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    it["group"] = "train" if it["source"].startswith("training") else "gold"
    items[it["id"]] = it

def load(model, cond, corr):
    p = "results/runs/%s__%s__%s.jsonl" % (model, cond, corr)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def pct(x):
    return "%5.1f" % (100 * x)

print("== logprobs stored for yes/no ==")
for m in MODELS:
    recs = [r for r in load(m, "gold", "clean") if r["type"] == "yesno"]
    print(m, sum(r.get("logp_yes") is not None for r in recs), "/", len(recs))

print("\n== yes/no: acc (share 'yes') on train | gold | acc on gold=yes | gold=no ==")
for m in MODELS:
    for cond, corr in CONDS:
        recs = [r for r in load(m, cond, corr) if r["type"] == "yesno" and r["id"] in items]
        if not recs:
            continue
        out = []
        for f in (lambda r: items[r["id"]]["group"] == "train", lambda r: items[r["id"]]["group"] == "gold",
                  lambda r: r["gold"] == "yes", lambda r: r["gold"] == "no"):
            sub = [r for r in recs if f(r)]
            acc = sum(r["pred"] == r["gold"] for r in sub) / max(len(sub), 1)
            yes = sum(r["pred"] == "yes" for r in sub) / max(len(sub), 1)
            out.append((acc, yes))
        print("%-24s %-15s train %s (%s) | gold %s (%s) | yes %s | no %s" % (m, corr, pct(out[0][0]), pct(out[0][1]), pct(out[1][0]), pct(out[1][1]), pct(out[2][0]), pct(out[3][0])))

print("\n== factoid strict acc / list F1: train | gold ==")
for m in MODELS:
    for qtype in ("factoid", "list"):
        for cond, corr in CONDS:
            recs = [r for r in load(m, cond, corr) if r["type"] == qtype and r["id"] in items]
            if not recs:
                continue
            vals = []
            for g in ("train", "gold"):
                sub = [r for r in recs if items[r["id"]]["group"] == g]
                if qtype == "factoid":
                    s = [factoid_scores(r["pred"], r["gold"])["strict"] for r in sub]
                else:
                    s = [list_scores(r["pred"], r["gold"])["f1"] for r in sub]
                vals.append(sum(s) / max(len(s), 1))
            print("%-24s %-8s %-15s train %s | gold %s" % (m, qtype, corr, pct(vals[0]), pct(vals[1])))
