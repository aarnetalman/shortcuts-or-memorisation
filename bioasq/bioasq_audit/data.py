"""Flatten a BioASQ Task b JSON file into one JSONL item per question.

Keeps yesno, factoid and list questions. Summary questions are dropped.
Gold snippets are kept as a list of strings in document order.
"""
import argparse
import json
from typing import Dict, Iterable, List


def load_bioasq(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    qs = raw["questions"] if isinstance(raw, dict) else raw
    return qs


def normalise_exact(qtype: str, exact) -> object:
    if qtype == "yesno":
        return str(exact).strip().lower()
    if qtype == "factoid":
        # list of synonym lists in recent versions, flat list in older ones
        if exact and isinstance(exact[0], list):
            return [[s.strip() for s in syn] for syn in exact]
        return [[s.strip() for s in exact]]
    if qtype == "list":
        if exact and isinstance(exact[0], list):
            return [[s.strip() for s in syn] for syn in exact]
        return [[s.strip()] for s in exact]
    return exact


def flatten(questions: Iterable[Dict], source: str) -> Iterable[Dict]:
    for q in questions:
        qtype = q.get("type")
        if qtype not in {"yesno", "factoid", "list"}:
            continue
        snippets = [s["text"].strip() for s in q.get("snippets", []) if s.get("text")]
        yield {
            "id": q["id"],
            "source": source,
            "type": qtype,
            "question": q["body"].strip(),
            "snippets": snippets,
            "gold": normalise_exact(qtype, q.get("exact_answer")),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+", help="BioASQ JSON file(s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in args.inp:
            for item in flatten(load_bioasq(path), source=path.rsplit("/", 1)[-1]):
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} items to {args.out}")


if __name__ == "__main__":
    main()
