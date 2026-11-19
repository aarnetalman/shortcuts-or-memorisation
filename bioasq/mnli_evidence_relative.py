"""MNLI like-for-like comparison: share of what the premise adds, kept under perturbation.
R_hyp = (score_pert - score_hyp-only) / (score_clean - score_hyp-only), from results/paper/answer_bias.csv.
Also the paper's chance-based R for reference. Centred accuracy and AUC, k0 conditions."""
import csv
rows = [r for r in csv.DictReader(open("../results/paper/answer_bias.csv")) if r["task"] == "mnli"]
PERTS = ["drop-NOUN", "drop-VERB", "keep-NOUN+VERB", "shuffle-n1"]
MODELS = [("qwen3.8-27b", "k0"), ("olmo3-32b", "k0"), ("olmo3-7b", "k0"), ("muse-glimmer-30b", "k0"),
          ("azure-gpt-4o", "k0/reasoning-off"), ("azure-deepseek-v4-pro", "k0/reasoning-off")]
f = lambda x: float(x) if x not in ("", None) else None
out = open("results/tables/mnli_evidence_relative.csv", "w")
out.write("model,metric,clean,hyp_only," + ",".join(f"R_chance_{p}" for p in PERTS) + "," + ",".join(f"R_hyp_{p}" for p in PERTS) + "\n")
for m, c in MODELS:
    d = {r["corruption"]: r for r in rows if r["model"] == m and r["condition"] == c}
    for met, ch in (("centered_accuracy", 100 / 3), ("auc", 50.0)):
        cl, hy = f(d["clean"][met]), f(d["hyp-only"][met])
        rc = [100 * (f(d[p][met]) - ch) / (cl - ch) for p in PERTS]
        rh = [100 * (f(d[p][met]) - hy) / (cl - hy) for p in PERTS]
        out.write(f"{m},{met},{cl},{hy}," + ",".join(f"{x:.1f}" for x in rc + rh) + "\n")
        print(f"{m:22s} {met:17s} R_chance " + " ".join(f"{x:4.0f}" for x in rc) + " | R_hyp " + " ".join(f"{x:4.0f}" for x in rh))
