"""Corruption transformations.

All transformations work on tagged records produced by tagging.tag_record and
return a dict {key: corrupted text}, or None when the record is dropped.

Styles
------
"2022"  Talman et al. (2022): rejoin with TreebankWordDetokenizer, keep records
        whose text becomes empty. Used for all new experiments.
"2021"  Talman et al. (2021): rejoin with single spaces, drop records where any
        text field becomes empty. Used only to replicate 2021 numbers.

Names
-----
clean                    original text, untouched
drop-NOUN                remove one word class (any of CORRUPTION_CLASSES)
drop-NOUN+PRON           remove several classes
keep-NOUN+VERB           keep only the listed classes (2021 inverse sets)
shuffle-n1|n2|n3         shuffle n-grams within each text field (Pham et al. 2020)
randdrop-NOUN            remove as many random tokens as drop-NOUN would (length control)
hyp-only                 empty the first text field (premise-free baseline)
swap                     swap the two text fields
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache

from .tagging import TAGS

STYLES = ("2021", "2022")


@lru_cache(maxsize=1)
def _detok():
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    return TreebankWordDetokenizer()


def join(tokens: list[str], style: str) -> str:
    if style == "2021":
        return " ".join(tokens)
    if style == "2022":
        return _detok().detokenize(tokens)
    raise ValueError(f"unknown style {style!r}")


def item_rng(seed: int, record_id: str, key: str, name: str) -> random.Random:
    """Deterministic per-item RNG, independent of processing order."""
    return random.Random(f"{seed}|{record_id}|{key}|{name}")


def ngram_shuffle(tokens: list[str], n: int, rng: random.Random, tries: int = 10) -> list[str]:
    """Split tokens into consecutive n-grams and shuffle their order.

    Follows Pham et al. (2020): the shuffled order must differ from the original
    whenever that is possible (more than one distinct n-gram).
    """
    chunks = [tokens[i : i + n] for i in range(0, len(tokens), n)]
    if len({tuple(c) for c in chunks}) < 2:
        return list(tokens)
    for _ in range(tries):
        order = list(range(len(chunks)))
        rng.shuffle(order)
        if [chunks[i] for i in order] != chunks:
            break
    return [t for i in order for t in chunks[i]]


@dataclass(frozen=True)
class Corruption:
    name: str
    kind: str
    classes: tuple[str, ...] = ()
    n: int = 0

    def needs_pair(self) -> bool:
        return self.kind in ("hyp-only", "swap")


_CLASS_RE = r"[A-Z.]+(?:\+[A-Z.]+)*"


def parse(name: str) -> Corruption:
    if name in ("clean", "hyp-only", "swap"):
        return Corruption(name, name)
    m = re.fullmatch(rf"(drop|keep|randdrop)-({_CLASS_RE})", name)
    if m:
        classes = tuple(m.group(2).split("+"))
        bad = [c for c in classes if c not in TAGS]
        if bad:
            raise ValueError(f"unknown POS class(es) {bad} in {name!r}; universal tags: {TAGS}")
        if m.group(1) == "randdrop" and len(classes) != 1:
            raise ValueError("randdrop takes exactly one class")
        return Corruption(name, m.group(1), classes)
    m = re.fullmatch(r"shuffle-n([1-9])", name)
    if m:
        return Corruption(name, "shuffle", n=int(m.group(1)))
    raise ValueError(f"unknown corruption {name!r}")


def apply(name: str | Corruption, tagged: dict, style: str = "2022", seed: int = 0) -> dict | None:
    """Apply a corruption to one tagged record. Returns {key: text} or None (dropped)."""
    c = parse(name) if isinstance(name, str) else name
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    keys = list(tagged["text"])
    if c.kind == "clean":
        return dict(tagged["text"])
    if c.needs_pair() and len(keys) != 2:
        raise ValueError(f"{c.name} needs a sentence-pair task")
    if c.kind == "hyp-only":
        return {keys[0]: "", keys[1]: tagged["text"][keys[1]]}
    if c.kind == "swap":
        return {keys[0]: tagged["text"][keys[1]], keys[1]: tagged["text"][keys[0]]}

    out = {}
    for k in keys:
        pairs = tagged["tagged"][k]
        if c.kind == "drop":
            toks = [w for w, t in pairs if t not in c.classes]
        elif c.kind == "keep":
            toks = [w for w, t in pairs if t in c.classes]
        elif c.kind == "randdrop":
            k_remove = sum(1 for _, t in pairs if t == c.classes[0])
            rng = item_rng(seed, tagged["id"], k, c.name)
            remove = set(rng.sample(range(len(pairs)), k_remove)) if k_remove else set()
            toks = [w for i, (w, _) in enumerate(pairs) if i not in remove]
        elif c.kind == "shuffle":
            rng = item_rng(seed, tagged["id"], k, c.name)
            toks = ngram_shuffle([w for w, _ in pairs], c.n, rng)
        else:  # pragma: no cover
            raise AssertionError(c.kind)
        out[k] = join(toks, style)

    if style == "2021" and any(not v.strip() for v in out.values()):
        return None
    return out


def standard_eval_corruptions(pair: bool) -> list[str]:
    """The default set of test-time corruptions used for every task."""
    from .tagging import CORRUPTION_CLASSES

    names = ["clean"] + [f"drop-{c}" for c in CORRUPTION_CLASSES]
    names += ["shuffle-n1", "shuffle-n2", "shuffle-n3"]
    names += KEEP_SETS_2021 + ["drop-NOUN+PRON"]
    if pair:
        names += ["hyp-only", "swap"]
    return names


# Inverse sets from Talman et al. (2021), Table 2.
KEEP_SETS_2021 = [
    "keep-NOUN+PRON+VERB",
    "keep-NOUN+ADV+VERB",
    "keep-NOUN+VERB",
    "keep-NOUN+VERB+ADJ",
    "keep-NOUN+VERB+ADV+ADJ",
]
