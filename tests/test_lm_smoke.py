"""Log-prob scoring and probing on a tiny random causal LM (downloads ~1 MB)."""

import json
import math

import numpy as np
import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def lm():
    pytest.importorskip("torch")
    from nlicorrupt.lm import load_lm

    return load_lm(TINY, fmt="text", device="cpu")


def records(task, split, corruption, style):
    n = 40 if split.startswith("train") else 9
    return [
        {"id": f"{task}:{split}:{i}", "label": i % 3,
         "premise": f"the cat {i} sat" if corruption == "clean" else f"the {i} sat",
         "hypothesis": "a cat sat"}
        for i in range(n)
    ]


def test_full_mode_matches_manual_computation(lm):
    import torch

    from nlicorrupt import lm as lmmod

    prompts = ["Premise: a\nAnswer:", "Premise: a much longer premise here\nAnswer:"]
    conts = lmmod.answer_continuations(lm, "mnli")
    scores, mode = lmmod.score_prompts(lm, "mnli", prompts, batch_size=2)
    # Force full mode and compare with an unbatched computation.
    orig = lmmod.scoring_mode
    lmmod.scoring_mode = lambda c: "full"
    try:
        full, m2 = lmmod.score_prompts(lm, "mnli", prompts, batch_size=3)
    finally:
        lmmod.scoring_mode = orig
    assert m2 == "full"
    for pi, p in enumerate(prompts):
        ids = lm.tokenizer(p).input_ids
        for j, c in enumerate(conts):
            seq = torch.tensor([ids + c])
            with torch.no_grad():
                logp = torch.log_softmax(lm.model(seq).logits.float(), -1)[0]
            manual = sum(float(logp[len(ids) - 1 + k, t]) for k, t in enumerate(c))
            assert full[pi][j] == pytest.approx(manual, abs=1e-3)
            if mode == "first-token":
                manual1 = float(logp[len(ids) - 1, c[0]])
                assert scores[pi][j] == pytest.approx(manual1, abs=1e-3)


def test_logprob_run_and_resume(lm, monkeypatch):
    from nlicorrupt import logprob

    monkeypatch.setattr(logprob, "get_records", records)
    cfg = logprob.LogprobConfig(model=TINY, task="mnli", corruptions=["clean", "drop-NOUN"], shots=2,
                                exemplars="drop-NOUN", batch_size=4)
    out = logprob.run(cfg, lm=lm)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics) == {"clean", "drop-NOUN"} and metrics["clean"]["n"] == 9
    man = json.loads((out / "manifest.json").read_text())
    assert len(man["exemplar_ids"]) == 2 and "Answer:" in man["example_prompt"]
    assert out.name == "k2-drop-NOUN"
    assert logprob.run(cfg, lm=lm) == out  # nothing left to do


def test_stsb_prediction_is_expected_score():
    from nlicorrupt.logprob import predict_from_scores

    s = [math.log(p) for p in (0.0001, 0.0001, 0.0001, 0.5, 0.5, 0.0001)]
    assert predict_from_scores("stsb", s) == pytest.approx(3.5, abs=0.01)
    assert predict_from_scores("mnli", [-3.0, -1.0, -2.0]) == 1


def test_extract_and_probe(lm, monkeypatch):
    from nlicorrupt import probe

    monkeypatch.setattr(probe, "get_records", records)
    paths = probe.extract(TINY, "mnli", "train", ["clean", "drop-NOUN"], max_examples=30, layers=[1, 2], lm=lm)
    probe.extract(TINY, "mnli", "validation_matched", ["clean", "drop-NOUN"], layers=[1, 2], lm=lm)
    d = np.load(paths[0])
    assert d["X"].shape[:2] == (30, 2) and d["X"].dtype == np.float16
    out = probe.train_and_eval(TINY, "mnli", "clean", ["clean", "drop-NOUN"])
    m = json.loads((out / "metrics.json").read_text())
    assert set(m) == {"layer1/clean", "layer1/drop-NOUN", "layer2/clean", "layer2/drop-NOUN"}


def test_eval_subsample_is_stratified_tagged_and_shared(lm, monkeypatch):
    from nlicorrupt import logprob, probe

    monkeypatch.setattr(logprob, "get_records", records)
    monkeypatch.setattr(probe, "get_records", records)
    cfg = logprob.LogprobConfig(model=TINY, task="mnli", corruptions=["clean", "drop-NOUN"], eval_items=6)
    out = logprob.run(cfg, lm=lm)
    assert out.name == "k0-n6"
    a = {r["id"] for r in logprob.load_predictions(out, "clean")}
    b = {r["id"] for r in logprob.load_predictions(out, "drop-NOUN")}
    assert a == b and len(a) == 6

    probe.extract(TINY, "mnli", "train", ["clean"], max_examples=30, layers=[1], lm=lm)
    p = probe.extract(TINY, "mnli", "validation_matched", ["clean", "drop-NOUN"], max_examples=6,
                      layers=[1], lm=lm, tag_n=True)
    assert p[0].name == "clean.n6.npz"
    assert set(np.load(p[0])["ids"]) == a
    m = json.loads((probe.train_and_eval(TINY, "mnli", "clean", ["clean", "drop-NOUN"], eval_n=6)
                    / "metrics.json").read_text())
    assert m["layer1/clean"]["n"] == 6
