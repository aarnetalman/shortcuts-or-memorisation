import json

import numpy as np
import pytest

from nlicorrupt.cli import main
from nlicorrupt.contamination import infinigram
from nlicorrupt.contamination.local import guess_prompt, mask_target, min_k_plus_plus
from nlicorrupt.data import load_split, read_jsonl

ITEMS = [
    {"id": "f1", "premise": "The ferry left late.", "hypothesis": "The ferry was delayed.", "label": "entailment",
     "source": "fresh", "author": "AT", "written": "2026-11-15"},
    {"id": "f2", "premise": "Nobody came.", "hypothesis": "Everyone came.", "label": "contradiction",
     "source": "twin", "twin_of": "mnli:validation_matched:7"},
]


@pytest.fixture
def private(tmp_path, monkeypatch):
    root = tmp_path / "private"
    (root / "fresh").mkdir(parents=True)
    with open(root / "fresh" / "nli.jsonl", "w") as f:
        for it in ITEMS:
            f.write(json.dumps(it) + "\n")
    monkeypatch.setenv("NLICORRUPT_PRIVATE", str(root))
    return root


def test_fresh_items_load_as_a_task(private):
    recs = load_split("fresh_nli", "test")
    assert [r["label"] for r in recs] == [0, 2]
    assert recs[1]["id"] == "fresh_nli:test:f2" and recs[1]["twin_of"] == "mnli:validation_matched:7"
    assert main(["validate-fresh"]) == 0


def test_invalid_fresh_file_is_rejected(private):
    with open(private / "fresh" / "nli.jsonl", "a") as f:
        f.write(json.dumps({"id": "f1", "premise": "x", "hypothesis": "y", "label": "maybe", "source": "twin"}) + "\n")
    assert main(["validate-fresh"]) == 1
    with pytest.raises(ValueError, match="invalid"):
        load_split("fresh_nli", "test")


def test_fresh_needs_private_root(monkeypatch):
    monkeypatch.delenv("NLICORRUPT_PRIVATE", raising=False)
    with pytest.raises(FileNotFoundError, match="NLICORRUPT_PRIVATE"):
        load_split("fresh_nli", "test")


def test_ngram_queries():
    text = " ".join(f"w{i}" for i in range(30))
    q = infinigram.ngram_queries(text, n=13, max_ngrams=4, full_max_words=50)
    assert q["full"] == text and len(q["ngrams"]) == 4
    assert q["ngrams"][0].startswith("w0 ") and q["ngrams"][-1].endswith("w29")
    assert infinigram.ngram_queries("short text", 13, 4, 50)["ngrams"] == []
    assert infinigram.ngram_queries(text, 13, 4, 10)["full"] is None


def test_infinigram_run_with_fake_api(monkeypatch, tmp_path):
    calls = []

    def post(payload):
        calls.append(payload)
        return {"count": 3 if "cat" in payload["query"] else 0}

    def recs(task, split, corruption, style="2022"):
        return [{"id": f"rte:validation:{i}", "label": 0, "sentence1": "a cat sat" if i == 0 else "a dog ran",
                 "sentence2": "fine"} for i in range(3)]

    monkeypatch.setattr(infinigram, "get_records", recs)
    client = infinigram.InfiniGram("http://unused", "test-index", sleep=0, cache=tmp_path / "c.sqlite", post=post)
    out = infinigram.run(["rte"], index="test-index", client=client)[0]
    rows = list(read_jsonl(out))
    assert [r["seen"] for r in rows] == [True, False, False]
    # "a cat sat" is found, but is too short for a whole-field match to count
    assert [r["found"] for r in rows] == [{"sentence1": False, "sentence2": False}] * 3
    assert rows[0]["fields"]["sentence1"]["words"] == 3
    assert calls[0] == {"index": "test-index", "query_type": "count", "query": "a cat sat"}
    n = len(calls)
    infinigram.run(["rte"], index="test-index", client=client)
    assert len(calls) == n  # cached


def test_mask_target_and_guess_prompt():
    masked, w = mask_target("The committee postponed the decision until spring.")
    assert w == "committee" and "[MASK]" in masked and "committee" not in masked
    assert mask_target("it is so") is None
    p, target = guess_prompt("mnli", {"premise": "A man sleeps.", "hypothesis": "Somebody is sleeping outdoors."})
    # "sleeping" shares its stem with "sleeps" in the premise, so it is never chosen.
    assert target == "Somebody" and "MultiNLI" in p and "Premise: A man sleeps." in p


def test_mask_target_avoids_words_the_model_can_copy():
    # "committee" occurs in the premise, so the longest remaining word is masked.
    masked, w = mask_target("The committee postponed lunch.", "The committee met.")
    assert w == "postponed" and "committee" in masked
    assert mask_target("Dogs bark at dogs.", "")[1] == "bark"  # repeated words are never masked
    assert mask_target("The weather changed.", "The weather changed quickly.") is None


def test_min_k_plus_plus():
    lp = np.array([-1.0, -5.0, -2.0, -0.5, -3.0])
    mu = np.full(5, -2.0)
    sd = np.ones(5)
    assert min_k_plus_plus(lp, mu, sd, k=0.4) == pytest.approx(np.mean([-3.0, -1.0]))


@pytest.mark.network
def test_local_contamination_on_tiny_model(monkeypatch):
    pytest.importorskip("torch")
    from nlicorrupt.contamination import local
    from nlicorrupt.lm import load_lm

    lm = load_lm("hf-internal-testing/tiny-random-LlamaForCausalLM", fmt="text", device="cpu")

    def recs(task, split, corruption, style="2022"):
        return [{"id": f"rte:validation:{i}", "label": 0, "sentence1": "The weather was terrible today.",
                 "sentence2": f"Yesterday everything looked wonderful {i}."} for i in range(4)]

    monkeypatch.setattr(local, "get_records", recs)
    g = local.run_guess("tiny", ["rte"], batch_size=2, lm=lm)
    assert g["rte"]["n"] == 4 and 0 <= g["rte"]["exact_match"] <= 100
    m = local.run_mink("tiny", ["rte"], batch_size=2, lm=lm)
    assert m["rte"]["n"] == 4 and np.isfinite(m["rte"]["mean"])


def test_infinigram_stops_on_403_and_waits_on_429(monkeypatch, tmp_path):
    import io
    import urllib.error

    sleeps = []
    monkeypatch.setattr(infinigram.time, "sleep", lambda s: sleeps.append(s))
    responses = [urllib.error.HTTPError("u", 429, "", {"Retry-After": "90"}, io.BytesIO(b"")),
                 io.BytesIO(b'{"count": 7}')]

    class Resp:
        def __init__(self, b): self.b = b
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.b.getvalue()

    def fake_urlopen(req, timeout):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return Resp(r)

    monkeypatch.setattr(infinigram.urllib.request, "urlopen", fake_urlopen)
    c = infinigram.InfiniGram("http://x", "idx", sleep=0, cache=tmp_path / "c.sqlite")
    assert c.count("some query") == 7 and sleeps == [90.0]

    responses[:] = [urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b""))]
    with pytest.raises(infinigram.InfiniGramRefused):
        c.count("another query")


def test_field_found_ignores_short_whole_field_matches():
    f = infinigram.field_found
    assert not f({"full": 495, "ngrams": [], "words": 4})              # "The man is asleep."
    assert f({"full": 2, "ngrams": [0, 0], "words": 20})                # long field, whole match
    assert f({"full": None, "ngrams": [0, 1], "words": 60})             # a 13-gram match
    assert f({"full": 1, "ngrams": [0]})                                # old file: has 13-grams, so >= 13 words
    assert not f({"full": 1, "ngrams": []})                             # old file, short field
    row = {"fields": {"premise": {"full": None, "ngrams": [3]}, "hypothesis": {"full": 9, "ngrams": []}}}
    assert infinigram.item_flags(row) == {"premise": True, "hypothesis": False, "any": True}
