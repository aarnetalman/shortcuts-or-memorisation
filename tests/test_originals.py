"""Agreement of our corruption code with the files released with the 2021/2022 papers.

Fixtures hold the first rows of the original corrupted dev sets with their clean
source text. Exact string agreement is not expected: NLTK's tokeniser quote handling
and the detokeniser changed after 2022 (and the 2022 MRPC text used escaped quotes).
We check that the same words are removed, and record the agreement rates.
"""

import json
import re

import pytest

from nlicorrupt.corrupt import apply, tag_record
from tests.conftest import FIXTURES

_NORM = re.compile(r"\s+|``|''|\"|\\")


def norm(s: str) -> str:
    return _NORM.sub("", s)


def load(name):
    return [json.loads(l) for l in open(FIXTURES / name, encoding="utf-8")]


@pytest.mark.parametrize(
    "fixture,keys,min_exact,min_words",
    [
        ("orig2022_mrpc_dev_NOUN.jsonl", ("sentence1", "sentence2"), 0.5, 0.85),
        ("orig2022_sst2_dev_NOUN.jsonl", ("sentence",), 0.85, 0.95),
    ],
)
def test_2022_noun_removal_matches_released_files(needs_nltk, fixture, keys, min_exact, min_words):
    rows = load(fixture)
    exact = words = 0
    for i, r in enumerate(rows):
        t = tag_record({"id": str(i), "label": 0, **r["clean"]}, keys)
        out = apply("drop-NOUN", t, style="2022")
        exact += all(out[k] == r["corrupted"][k] for k in keys)
        words += all(norm(out[k]) == norm(r["corrupted"][k]) for k in keys)
    assert exact / len(rows) >= min_exact
    assert words / len(rows) >= min_words


def test_2021_mnli_noun_matches_released_file(needs_nltk):
    rows = load("orig2021_mnli_dev_NOUN.jsonl")
    clean = [r for r in rows if r["kind"] == "clean"]
    released = [r for r in rows if r["kind"] == "corrupted"]
    ours = []
    for i, r in enumerate(clean):
        t = tag_record({"id": str(i), "label": r["label"], "premise": r["premise"], "hypothesis": r["hypothesis"]},
                       ("premise", "hypothesis"))
        out = apply("drop-NOUN", t, style="2021")
        if out is not None:
            ours.append(out)
    n = len(released)
    agree = sum(norm(o["premise"]) == norm(r["premise"]) and norm(o["hypothesis"]) == norm(r["hypothesis"])
                for o, r in zip(ours[:n], released))
    assert agree / n >= 0.95
