"""Token and cost estimate for the hosted runs before launching them.
Usage: python scripts/cost_estimate.py data/corrupted/clean.jsonl --in-price 2.5 --out-price 10
Prices in USD per million tokens, check the current ones first."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse, json
from bioasq_audit.prompts import build
ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("--in-price", type=float, required=True); ap.add_argument("--out-price", type=float, required=True)
ap.add_argument("--conditions", type=int, default=9); ap.add_argument("--factoid-cap", type=int, default=1000); ap.add_argument("--list-cap", type=int, default=500)
a = ap.parse_args()
items = [json.loads(l) for l in open(a.items)]
caps = {"yesno": 10**9, "factoid": a.factoid_cap, "list": a.list_cap}
seen = {"yesno": 0, "factoid": 0, "list": 0}
tok_in = tok_out = 0
for it in items:
    if seen[it["type"]] >= caps[it["type"]]: continue
    seen[it["type"]] += 1
    tok_in += len(build(it, "gold").split()) * 1.3
    tok_out += {"yesno": 2, "factoid": 20, "list": 60}[it["type"]]
calls = sum(seen.values()) * a.conditions
cost = (tok_in * a.conditions * a.in_price + tok_out * a.conditions * a.out_price) / 1e6
print(f"items {seen}, calls {calls:,}, input tokens {tok_in*a.conditions/1e6:.1f}M, est. cost ${cost:,.0f} per hosted model")
