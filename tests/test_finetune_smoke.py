"""End-to-end fine-tuning on a tiny random model (downloads ~1 MB from the hub)."""

import json

import pytest

from nlicorrupt import finetune
from nlicorrupt.data import read_jsonl

TINY = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"


def fake_records(task, split, corruption, style):
    n = 24 if split.startswith("train") else 10
    words = "the cat sat on a mat" if corruption == "clean" else "the sat on a"
    return [
        {"id": f"{task}:{split}:{i}", "label": i % 3, "premise": f"{words} {i}", "hypothesis": f"{words}"}
        for i in range(n)
    ]


@pytest.mark.network
def test_finetune_writes_metrics_predictions_and_skips_when_done(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setattr(finetune, "get_records", fake_records)
    cfg = finetune.FinetuneConfig(
        model=TINY, task="mnli", train_corruption="drop-NOUN",
        eval_corruptions=["clean", "drop-NOUN"], epochs=1, batch_size=8, bf16=False,
        out_dir=str(tmp_path / "run"),
    )
    out = finetune.run(cfg)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics) == {
        "mnli/validation_matched/clean",
        "mnli/validation_matched/drop-NOUN",
        "mnli/validation_mismatched/clean",
    }
    assert 0 <= metrics["mnli/validation_matched/clean"]["accuracy"] <= 100
    preds = list(read_jsonl(out / "predictions" / "mnli__validation_matched__drop-NOUN.jsonl.gz"))
    assert len(preds) == 10 and {"id", "label", "pred"} <= set(preds[0])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["args"]["train_corruption"] == "drop-NOUN"
    assert not (out / "model").exists()  # auto: only clean-trained models are saved

    mtime = (out / "metrics.json").stat().st_mtime_ns
    finetune.run(cfg)
    assert (out / "metrics.json").stat().st_mtime_ns == mtime


@pytest.mark.network
def test_evaluate_model_accepts_a_pinned_revision(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setattr(finetune, "get_records", fake_records)
    m = finetune.evaluate_model(TINY + "@main", tmp_path / "eval", ["mnli"], ["clean", "drop-NOUN"])
    assert set(m) == {"mnli/validation_matched/clean", "mnli/validation_matched/drop-NOUN"}
