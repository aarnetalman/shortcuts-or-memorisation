"""Input corruptions following Talman et al. (2021, 2022).

drop-X        remove all words of universal POS class X
keep-X+Y      keep only words of the listed classes
shuffle-nk    split into consecutive k-grams and shuffle their order
noctx         remove the snippets entirely (hypothesis-only analogue)

By default the corruption is applied to the snippets only and the question
stays clean, so that the question-only cue split is not confounded.
`--target question` applies it to the question instead.
"""
import argparse
import json
import os
import random
from typing import Callable, Dict, List

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("taggers/universal_tagset")
except LookupError:
    nltk.download("universal_tagset", quiet=True)

_detok = TreebankWordDetokenizer()

ALL_CORRUPTIONS = [
    "clean", "noctx",
    "drop-NOUN", "drop-VERB", "drop-ADJ", "drop-ADV", "drop-DET", "drop-PRON", "drop-ADP",
    "keep-NOUN+VERB", "keep-NOUN", "shuffle-n1", "shuffle-n3",
]


def tag(text: str) -> List[tuple]:
    toks = nltk.word_tokenize(text)
    return nltk.pos_tag(toks, tagset="universal")


def drop(text: str, classes: set) -> str:
    return _detok.detokenize([w for w, t in tag(text) if t not in classes])


def keep(text: str, classes: set) -> str:
    return _detok.detokenize([w for w, t in tag(text) if t in classes])


def shuffle_ngrams(text: str, k: int, rng: random.Random) -> str:
    toks = nltk.word_tokenize(text)
    grams = [toks[i:i + k] for i in range(0, len(toks), k)]
    rng.shuffle(grams)
    return _detok.detokenize([w for g in grams for w in g])


def make_fn(name: str, seed: int = 0) -> Callable[[str], str]:
    rng = random.Random(seed)
    if name == "clean":
        return lambda s: s
    if name.startswith("drop-"):
        cls = set(name[5:].split("+"))
        return lambda s: drop(s, cls)
    if name.startswith("keep-"):
        cls = set(name[5:].split("+"))
        return lambda s: keep(s, cls)
    if name.startswith("shuffle-n"):
        k = int(name[len("shuffle-n"):])
        return lambda s: shuffle_ngrams(s, k, rng)
    raise ValueError(name)


def corrupt_item(item: Dict, name: str, fn: Callable, target: str) -> Dict:
    out = dict(item)
    out["corruption"] = name
    out["target"] = target
    if name == "noctx":
        out["snippets"] = []
        return out
    if target == "snippets":
        out["snippets"] = [fn(s) for s in item["snippets"]]
    else:
        out["question"] = fn(item["question"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="output directory, one JSONL per corruption")
    ap.add_argument("--corruptions", default="all")
    ap.add_argument("--target", choices=["snippets", "question"], default="snippets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = ALL_CORRUPTIONS if args.corruptions == "all" else args.corruptions.split(",")
    os.makedirs(args.out, exist_ok=True)
    items = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    for name in names:
        fn = make_fn(name, args.seed) if name != "noctx" else None
        suffix = "" if args.target == "snippets" else "-q"
        path = os.path.join(args.out, f"{name}{suffix}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(corrupt_item(it, name, fn, args.target), ensure_ascii=False) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
