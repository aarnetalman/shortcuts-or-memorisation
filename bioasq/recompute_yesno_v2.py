"""Recompute the hosted yes/no AUC figures from the corrected yes/no rerun (results/runs_yesno_v2/).

The first hosted run (5 Nov) collapsed the "Yes"/"yes" token variants. runs_yesno_v2 (8 Nov) is the yes/no-only
rerun with the variants handled and raw top-5 kept. This script:
  1. builds results/runs_merged_v2/ (gitignored): hosted files = runs/ with every yes/no record replaced by v2,
     open-model files = runs/ unchanged;
  2. rewrites results/tables/auc_all6.txt for all six models from the merged runs (same format as before);
  3. reruns bootstrap_ci.py on the merged runs for the hosted models and replaces their rows in
     results/tables/bootstrap_ci_all6.csv. Open-model rows are untouched.
Previous versions are kept as *_firstrun.* next to them."""
import json, os, glob, io, contextlib, csv
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
HOSTED = ["gpt-4o", "DeepSeek-V4-Pro", "Llama-3.3-70B-Instruct"]
OPEN = ["Qwen_Qwen3.8-27B", "allenai_Olmo-3-1025-7B", "allenai_Olmo-3-1125-32B"]
MERGED = "results/runs_merged_v2"; os.makedirs(MERGED, exist_ok=True)
for f in glob.glob("results/runs/*.jsonl"):
    name = os.path.basename(f); m = name.split("__")[0]
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    if m in HOSTED:
        v2 = {r["id"]: r for r in map(json.loads, open(f"results/runs_yesno_v2/{name}", encoding="utf-8"))}
        recs = [v2[r["id"]] if r["type"] == "yesno" else r for r in recs]
    with open(f"{MERGED}/{name}", "w", encoding="utf-8") as o:
        for r in recs: o.write(json.dumps(r, ensure_ascii=False) + "\n")

# 2. AUC table, same layout as auc.py
from bioasq_audit.metrics import auc
CONDS = [("noctx","noctx"),("gold","clean"),("gold","drop-ADJ"),("gold","drop-DET"),("gold","drop-NOUN"),("gold","drop-VERB"),("gold","keep-NOUN+VERB"),("gold","shuffle-n1"),("gold","shuffle-n3")]
items = {}
for l in open("data/items.jsonl", encoding="utf-8"):
    it = json.loads(l); items[it["id"]] = "train" if it["source"].startswith("training") else "gold"
lines = ["AUC on yes/no (0.5 = chance, 1 = perfect), all | train | gold, and R = (AUC-0.5)/(AUC_clean-0.5)",
         "(hosted models: corrected yes/no rerun, results/runs_yesno_v2; open models unchanged)"]
for m in HOSTED + OPEN:
    base = {}
    for cond, corr in CONDS:
        recs = [json.loads(l) for l in open(f"{MERGED}/{m}__{cond}__{corr}.jsonl", encoding="utf-8")]
        recs = [r for r in recs if r["type"] == "yesno" and r.get("logp_yes") is not None and r.get("logp_no") is not None]
        row = {g: auc([r["logp_yes"] - r["logp_no"] for r in recs if g == "all" or items.get(r["id"]) == g],
                      [r["gold"] for r in recs if g == "all" or items.get(r["id"]) == g]) for g in ("all", "train", "gold")}
        if corr == "clean": base = row
        rs = ["%5.1f" % (100 * (row[g] - .5) / (base[g] - .5)) if base and corr != "clean" and base[g] > .5 else "    -" for g in ("all", "train", "gold")]
        lines.append("%-24s %-15s AUC %.3f | %.3f | %.3f   R %s | %s | %s" % (m, corr, row["all"], row["train"], row["gold"], *rs))
open("results/tables/auc_all6.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:20]))

# 3. bootstrap rows for the hosted models
src = open("bootstrap_ci.py", encoding="utf-8").read()
src = src.replace('RUNS = "results/runs"', f'RUNS = "{MERGED}"').replace('"results/tables/bootstrap_ci.csv"', '"results/tables/_bootstrap_hosted_v2.csv"')
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "bootstrap_ci.py", "exec"), {"__name__": "__main__", "__file__": os.path.join(HERE, "bootstrap_ci.py")})
new = {(r["model"], r["measure"], r["group"]): r for r in csv.DictReader(open("results/tables/_bootstrap_hosted_v2.csv"))}
rows = list(csv.DictReader(open("results/tables/bootstrap_ci_all6_firstrun.csv")))
changed = 0
for r in rows:
    k = (r["model"], r["measure"], r["group"])
    if r["model"] in HOSTED and r["measure"].startswith("yesno") and k in new:
        if new[k]["point"] != r["point"]: changed += 1
        r.update(new[k])
with open("results/tables/bootstrap_ci_all6.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "measure", "group", "point", "ci_low", "ci_high"]); w.writeheader(); w.writerows(rows)
print("bootstrap rows updated:", changed)
