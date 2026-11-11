import json, os, sys
sys.path.insert(0, ".")
from bioasq_audit.metrics import auc

MODELS = ["gpt-4o", "DeepSeek-V4-Pro", "Llama-3.3-70B-Instruct"]
CONDS = [("noctx","noctx"),("gold","clean"),("gold","drop-ADJ"),("gold","drop-DET"),("gold","drop-NOUN"),("gold","drop-VERB"),("gold","keep-NOUN+VERB"),("gold","shuffle-n1"),("gold","shuffle-n3")]
items = {}
for l in open("data/items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    items[it["id"]] = "train" if it["source"].startswith("training") else "gold"

print("AUC on yes/no (0.5 = chance, 1 = perfect), all | train | gold, and R = (AUC-0.5)/(AUC_clean-0.5)")
for m in MODELS:
    base = {}
    for cond, corr in CONDS:
        p = "results/runs/%s__%s__%s.jsonl" % (m, cond, corr)
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8")]
        recs = [r for r in recs if r["type"] == "yesno" and r.get("logp_yes") is not None and r.get("logp_no") is not None]
        row = {}
        for g in ("all", "train", "gold"):
            sub = [r for r in recs if g == "all" or items.get(r["id"]) == g]
            row[g] = auc([r["logp_yes"] - r["logp_no"] for r in sub], [r["gold"] for r in sub])
        if corr == "clean":
            base = row
        rs = []
        for g in ("all", "train", "gold"):
            if base and corr != "clean" and base[g] > 0.5:
                rs.append("%5.1f" % (100 * (row[g] - 0.5) / (base[g] - 0.5)))
            else:
                rs.append("    -")
        print("%-24s %-15s AUC %.3f | %.3f | %.3f   R %s | %s | %s" % (m, corr, row["all"], row["train"], row["gold"], rs[0], rs[1], rs[2]))
