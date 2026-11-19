import random

import pytest

from nlicorrupt.corrupt import apply, ngram_shuffle, parse, standard_eval_corruptions
from tests.conftest import tagged_record

PREM = [("The", "DET"), ("old", "ADJ"), ("man", "NOUN"), ("sold", "VERB"), ("two", "NUM"), ("boats", "NOUN"), (".", ".")]
HYP = [("He", "PRON"), ("sold", "VERB"), ("nothing", "NOUN"), (".", ".")]


def rec():
    return tagged_record("mnli:x:1", 2, premise=PREM, hypothesis=HYP)


def test_parse_names():
    assert parse("drop-NOUN").classes == ("NOUN",)
    assert parse("keep-NOUN+VERB").kind == "keep"
    assert parse("shuffle-n2").n == 2
    for bad in ["drop-FOO", "shuffle-n0", "nonsense", "randdrop-NOUN+VERB"]:
        with pytest.raises(ValueError):
            parse(bad)


def test_drop_and_keep_space_style():
    out = apply("drop-NOUN", rec(), style="2021")
    assert out == {"premise": "The old sold two .", "hypothesis": "He sold ."}
    out = apply("keep-NOUN+VERB", rec(), style="2021")
    assert out == {"premise": "man sold boats", "hypothesis": "sold nothing"}
    out = apply("drop-NOUN+PRON", rec(), style="2021")
    assert out["hypothesis"] == "sold ."


def test_detok_style_reattaches_punctuation():
    out = apply("drop-NOUN", rec(), style="2022")
    assert out == {"premise": "The old sold two.", "hypothesis": "He sold."}


def test_2021_drops_records_with_empty_field_2022_keeps_them():
    r = tagged_record("x", 0, premise=[("Dogs", "NOUN")], hypothesis=[("Cats", "NOUN"), ("run", "VERB")])
    assert apply("drop-NOUN", r, style="2021") is None
    assert apply("drop-NOUN", r, style="2022") == {"premise": "", "hypothesis": "run"}


def test_clean_hyp_only_swap():
    r = rec()
    assert apply("clean", r) == r["text"]
    assert apply("hyp-only", r) == {"premise": "", "hypothesis": r["text"]["hypothesis"]}
    assert apply("swap", r) == {"premise": r["text"]["hypothesis"], "hypothesis": r["text"]["premise"]}
    single = tagged_record("s", 1, sentence=HYP)
    with pytest.raises(ValueError):
        apply("swap", single)


def test_shuffle_is_deterministic_and_changes_order():
    a = apply("shuffle-n1", rec(), seed=0)
    b = apply("shuffle-n1", rec(), seed=0)
    c = apply("shuffle-n1", rec(), seed=1)
    assert a == b
    assert a != c or a["premise"] != rec()["text"]["premise"]
    assert sorted(a["premise"].replace(".", " .").split()) == sorted(w for w, _ in PREM)


def test_ngram_shuffle_keeps_ngrams_together():
    toks = list("abcdefgh")
    out = ngram_shuffle(toks, 2, random.Random(3))
    assert out != toks
    pairs = {tuple(out[i : i + 2]) for i in range(0, 8, 2)}
    assert pairs == {("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")}
    assert ngram_shuffle(["x", "x"], 1, random.Random(0)) == ["x", "x"]


def test_randdrop_removes_as_many_tokens_as_class():
    out = apply("randdrop-NOUN", rec(), style="2021", seed=5)
    assert len(out["premise"].split()) == len(PREM) - 2
    assert len(out["hypothesis"].split()) == len(HYP) - 1
    assert out == apply("randdrop-NOUN", rec(), style="2021", seed=5)


def test_standard_sets():
    pair = standard_eval_corruptions(pair=True)
    single = standard_eval_corruptions(pair=False)
    assert pair[0] == "clean" and "hyp-only" in pair and "hyp-only" not in single
    assert len(set(pair)) == len(pair)
    for name in pair:
        parse(name)
