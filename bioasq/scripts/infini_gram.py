"""BioASQ contamination check with the paper's infini-gram client (rate-limited, SQLite-cached, resumable).

Same method and index as the NLI part: index v4_olmo-mix-1124_llama (OLMo 2's data mix, because the
OLMo 3 index was not usable), so counts are a LOWER BOUND for OLMo 3. A field is 'found' if one of two
evenly spaced 13-grams occurs, or the whole field occurs and has at least 13 words
(configs/contamination.yaml). Client: src/nlicorrupt/contamination/infinigram.py, at most 1 request/s.

Fields: the question (written by BioASQ experts, the analogue of an NLI hypothesis: a hit is evidence
the benchmark itself was in the data) and up to --snippets snippets per item (PubMed text, the analogue
of an ANLI premise: hits mostly reflect the public source text).

Per-item output goes to bioasq/data/contamination/ (gitignored, BioASQ licence). The aggregate summary
goes to results/tables/contamination_summary.csv.
Budget: questions alone are about 10k requests (~3 h at 1/s), each snippet per item adds about 13k.
Usage:  python scripts/infini_gram.py --limit 20     (smoke test)
        python scripts/infini_gram.py --snippets 2   (full run, resumable)
"""
import argparse, csv, json, logging, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from nlicorrupt.config import load_yaml  # noqa: E402
from nlicorrupt.contamination.infinigram import InfiniGram, ngram_queries, field_found  # noqa: E402
from pathlib import Path  # noqa: E402

INDEX = "v4_olmo-mix-1124_llama"
ap = argparse.ArgumentParser()
ap.add_argument("--items", default=os.path.join(HERE, "..", "data", "items.jsonl"))
ap.add_argument("--snippets", type=int, default=0, help="snippets per item to query (0 = questions only)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cache", default=None, help="SQLite cache path (default: bioasq/data/contamination/); must be on a local disk")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = load_yaml("contamination.yaml")["infinigram"]
data_dir = Path(HERE).parent / "data" / "contamination"; data_dir.mkdir(parents=True, exist_ok=True)
client = InfiniGram(cfg["endpoint"], INDEX, cfg["sleep_seconds"], cache=Path(a.cache) if a.cache else data_dir / "infinigram_cache.sqlite")

def check(text):
    q = ngram_queries(text or "", cfg["n"], cfg["max_ngrams_per_field"], cfg["full_max_words"])
    f = {"full": client.count(q["full"]) if q["full"] else None, "ngrams": [client.count(g) for g in q["ngrams"]],
         "words": len((text or "").split())}
    return field_found(f, cfg["n"], cfg["min_words_full_match"])

items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
if a.limit: items = items[:a.limit]
out_path = data_dir / f"bioasq_{INDEX}_s{a.snippets}.jsonl"
rows = []
with open(out_path, "w", encoding="utf-8") as out:
    for i, it in enumerate(items):
        sn = it["snippets"][: a.snippets]
        row = {"id": it["id"], "type": it["type"], "source": it["source"],
               "question_found": check(it["question"]),
               "snippets_found": sum(check(s) for s in sn), "snippets_checked": len(sn)}
        rows.append(row); out.write(json.dumps(row) + "\n")
        if i % 200 == 0: logging.info("%d/%d items", i, len(items))

agg = defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    g = "train" if r["source"].startswith("training") else r["source"][:3]
    for key in ((r["type"], g), ("all", g)):
        a_ = agg[key]; a_[0] += 1; a_[1] += r["question_found"]; a_[2] += r["snippets_found"] > 0; a_[3] += r["snippets_checked"] > 0
summ = os.path.join(HERE, "..", "results", "tables", "contamination_summary.csv")
with open(summ, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["index", "qtype", "group", "n_items", "question_found_pct", "any_snippet_found_pct"])
    for (t, g), (n, qf, sf, sc) in sorted(agg.items()):
        w.writerow([INDEX, t, g, n, round(100 * qf / n, 1), round(100 * sf / sc, 1) if sc else ""])
print("per-item:", out_path); print("summary:", summ)
