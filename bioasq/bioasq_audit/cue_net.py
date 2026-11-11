"""Apply a hand-written linear cue classifier to question-only features."""
import argparse
import json
from typing import Dict


def load_weights(path: str) -> Dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["weights"]


def score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in features.items())


def predict(features: Dict[str, float], weights: Dict[str, float]) -> str:
    return "yes" if score(features, weights) > 0 else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = load_weights(args.weights)
    n = correct = 0
    by_source = {}
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.features, encoding="utf-8"):
            it = json.loads(line)
            s = score(it["features"], w)
            pred = "yes" if s > 0 else "no"
            ok = pred == it["gold"]
            out.write(json.dumps({"id": it["id"], "source": it["source"], "gold": it["gold"],
                                  "pred": pred, "score": round(s, 3), "cue_easy": ok}) + "\n")
            n += 1
            correct += ok
            d = by_source.setdefault(it["source"], [0, 0])
            d[0] += 1
            d[1] += ok
    print(f"cue net accuracy: {correct / max(n, 1):.3f} on {n} items")
    for src, (tot, c) in sorted(by_source.items()):
        print(f"  {src}: {c / tot:.3f} ({tot})")


if __name__ == "__main__":
    main()
