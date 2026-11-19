"""Question-only cue features for BioASQ yes/no questions.

Every feature is computed from the question text alone. The feature set is
fixed before any data is seen and must not be extended by looking at errors
on BioASQ items. New features need a citation or a stated construction
hypothesis in weights/yesno_v0.json.
"""
import argparse
import json
import re
from typing import Dict, List

NEGATION = {"not", "no", "never", "lack", "lacks", "lacking", "absence", "absent", "without",
            "unable", "fail", "fails", "failed", "n't", "non", "none", "neither", "nor"}
UNIVERSAL = {"only", "all", "always", "every", "exclusively", "solely", "entirely", "completely",
             "universally", "invariably", "any"}
POSSIBILITY = {"can", "could", "may", "might", "possible", "possibly", "able", "potential",
               "potentially", "feasible"}
ASSOCIATION = {"associated", "association", "involved", "involvement", "role", "linked", "link",
               "implicated", "related", "relate", "contribute", "contributes", "regulate",
               "regulates", "regulated", "interact", "interacts", "interaction", "expressed",
               "expression", "affect", "affects", "influence", "influences", "modulate",
               "modulates", "mediate", "mediates", "correlate", "correlated"}
IDENTITY = {"same", "identical", "equivalent", "interchangeable", "synonym", "synonymous",
            "synonyms", "interchangeably", "equal"}
COMPARISON = {"than", "more", "less", "higher", "lower", "better", "worse", "greater", "smaller",
              "larger", "superior", "inferior", "faster", "slower"}
APPROVAL = {"approved", "approval", "fda", "ema", "licensed", "marketed", "indicated"}
CONTRADICTION = {"contradict", "contradictory", "contradicts", "disprove", "refute", "refuted",
                 "controversial", "debated", "questionable", "disputed", "myth"}
EFFICACY = {"effective", "efficacy", "efficacious", "beneficial", "benefit", "benefits",
            "improve", "improves", "protective", "protect", "protects"}
PREVENT = {"prevent", "prevents", "prevention", "cure", "cures", "eliminate", "eliminates",
           "eradicate"}

FEATURE_NAMES: List[str] = [
    "bias", "neg", "universal", "possibility", "association", "identity", "comparison",
    "has_number", "approval", "contradiction", "efficacy", "prevent", "len_gt_15", "starts_is_are",
    "starts_does_do", "starts_can", "starts_has_have", "two_entities",
]


def tokens(q: str) -> List[str]:
    return re.findall(r"[a-z0-9']+|n't", q.lower())


def features(question: str) -> Dict[str, float]:
    t = tokens(question)
    ts = set(t)
    first = t[0] if t else ""
    caps = re.findall(r"\b[A-Z][A-Za-z0-9-]{1,}\b", question[1:])  # capitalised tokens after the first char
    return {
        "bias": 1.0,
        "neg": float(sum(1 for w in t if w in NEGATION)),
        "universal": float(bool(ts & UNIVERSAL)),
        "possibility": float(bool(ts & POSSIBILITY)),
        "association": float(bool(ts & ASSOCIATION)),
        "identity": float(bool(ts & IDENTITY)),
        "comparison": float(bool(ts & COMPARISON)),
        "has_number": float(bool(re.search(r"\b\d+(\.\d+)?%?\b", question))),
        "approval": float(bool(ts & APPROVAL)),
        "contradiction": float(bool(ts & CONTRADICTION)),
        "efficacy": float(bool(ts & EFFICACY)),
        "prevent": float(bool(ts & PREVENT)),
        "len_gt_15": float(len(t) > 15),
        "starts_is_are": float(first in {"is", "are", "was", "were"}),
        "starts_does_do": float(first in {"does", "do", "did"}),
        "starts_can": float(first in {"can", "could", "may", "might"}),
        "starts_has_have": float(first in {"has", "have", "had"}),
        "two_entities": float(len(caps) >= 2),
    }


def vector(question: str) -> List[float]:
    f = features(question)
    return [f[n] for n in FEATURE_NAMES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.inp, encoding="utf-8"):
            it = json.loads(line)
            if it["type"] != "yesno":
                continue
            out.write(json.dumps({"id": it["id"], "source": it["source"], "gold": it["gold"],
                                  "features": features(it["question"])}) + "\n")
            n += 1
    print(f"wrote features for {n} yes/no items")


if __name__ == "__main__":
    main()
