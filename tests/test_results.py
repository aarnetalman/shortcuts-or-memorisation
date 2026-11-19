import json

import pandas as pd
import pytest

from nlicorrupt import paths
from nlicorrupt.analysis.collect import collect
from nlicorrupt.analysis.tables import corrupt_test, corrupt_train, reproduction, seen_unseen
from nlicorrupt.cli import main
from nlicorrupt.data import write_jsonl


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def runs():
    r = paths.runs_dir()
    base = r / "finetune" / "roberta-base" / "mnli" / "2022"
    for seed, off in ((13, 0.0), (42, 1.0)):
        put(base / "train-clean" / f"seed{seed}" / "metrics.json", {
            "mnli/validation_matched/clean": {"accuracy": 88.0 + off, "n": 10},
            "mnli/validation_matched/drop-NOUN": {"accuracy": 73.0 + off, "n": 10},
            "mnli/validation_mismatched/clean": {"accuracy": 87.0, "n": 10}})
        put(base / "train-drop-NOUN" / f"seed{seed}" / "metrics.json", {
            "mnli/validation_matched/clean": {"accuracy": 85.0 + off, "n": 10},
            "mnli/validation_matched/drop-NOUN": {"accuracy": 78.0 + off, "n": 10}})
    put(r / "logprob" / "qwen3.8-27b" / "rte" / "2022" / "k0" / "metrics.json", {
        "clean": {"accuracy": 90.0, "n": 10}, "drop-VERB": {"accuracy": 80.0, "n": 10}})
    put(r / "logprob" / "qwen3.8-27b" / "fresh_nli" / "2022" / "k0" / "metrics.json", {
        "clean": {"accuracy": 80.0, "n": 10}, "drop-VERB": {"accuracy": 50.0, "n": 10}})
    put(r / "prompt" / "kimi-k3" / "cola" / "2022" / "k0" / "reasoning-off" / "metrics.json", {
        "clean": {"mcc": 60.0, "n": 10, "unparsed": 0, "missing": 0}, "drop-NOUN": {"mcc": 20.0, "n": 10}})
    put(r / "probe" / "olmo3-7b" / "sst2" / "2022" / "train-clean" / "metrics.json", {
        "layer8/clean": {"accuracy": 90.0, "n": 5}, "layer8/drop-ADJ": {"accuracy": 70.0, "n": 5}})
    return r


def test_collect_and_deltas(runs):
    df = pd.DataFrame(collect())
    assert set(df.family) == {"finetune", "logprob", "prompt", "probe"}
    ct = corrupt_test(df)
    ft = ct[ct.family == "finetune"]
    assert sorted(ft.delta) == [-15.0, -15.0]
    assert set(ct[ct.family == "probe"].condition) == {"train-clean/layer8"}
    assert float(ct[ct.family == "prompt"].delta.iloc[0]) == -40.0
    tr = corrupt_train(df)
    assert sorted(tr[tr.config == "corrupt_train"].delta) == [-3.0, -3.0]
    assert sorted(tr[tr.config == "train_and_test"].delta) == [-10.0, -10.0]
    rep = reproduction(df)
    row = rep[(rep.config == "corrupt_test") & (rep.corruption == "drop-NOUN")].iloc[0]
    assert row.published == 72.85 and row.ours == pytest.approx(73.5)


def test_results_command_writes_tables_and_figures(runs):
    assert main(["results"]) == 0
    out = paths.results_dir()
    assert (out / "all_results.csv").exists()
    bvf = pd.read_csv(out / "tables" / "benchmark_vs_fresh.csv")
    assert bvf.empty  # no shared benchmark task for this model in the fixture
    figs = list((out / "figures").glob("*.png"))
    assert len(figs) == 4 and all(f.with_suffix(".pdf").exists() for f in figs)


def test_seen_unseen(tmp_path):
    ids = [f"mnli:x:{i}" for i in range(40)]
    write_jsonl(tmp_path / "clean.jsonl.gz", [{"id": i, "label": 0, "pred": 0} for i in ids])
    write_jsonl(tmp_path / "drop-NOUN.jsonl.gz",
                [{"id": i, "label": 0, "pred": 0 if n < 20 else 1} for n, i in enumerate(ids)])
    write_jsonl(tmp_path / "seen.jsonl.gz", [{"id": i, "seen": n < 20} for n, i in enumerate(ids)])
    t = seen_unseen(tmp_path, "mnli", tmp_path / "seen.jsonl.gz", ["drop-NOUN"], n_boot=100)
    assert t.set_index("seen").loc[True, "delta"] == 0.0
    assert t.set_index("seen").loc[False, "delta"] == -100.0
    assert t.set_index("seen").loc["gap", "delta"] == 100.0


def test_seen_unseen_by_field_and_centered(tmp_path):
    ids = [f"mnli:x:{i}" for i in range(40)]
    # Clean: correct. Corrupted: every score shifted towards answer 1 (pure answer bias),
    # plus a real loss of information on items 20-39 (scores swapped).
    clean = [{"id": i, "label": n % 2, "pred": n % 2, "scores": [1.0, 0.0] if n % 2 == 0 else [0.0, 1.0]}
             for n, i in enumerate(ids)]
    cor = []
    for n, r in enumerate(clean):
        s0, s1 = r["scores"]
        if n >= 20:
            s0, s1 = s1, s0
        cor.append({**r, "scores": [s0, s1 + 5.0], "pred": 1})
    write_jsonl(tmp_path / "clean.jsonl.gz", clean)
    write_jsonl(tmp_path / "drop-NOUN.jsonl.gz", cor)
    # premise found for items 0-19 (13-gram hit); hypothesis never found
    write_jsonl(tmp_path / "seen.jsonl.gz", [
        {"id": i, "fields": {"premise": {"full": None, "ngrams": [int(n < 20)], "words": 30},
                             "hypothesis": {"full": 0, "ngrams": [], "words": 5}}} for n, i in enumerate(ids)])
    raw = seen_unseen(tmp_path, "mnli", tmp_path / "seen.jsonl.gz", ["drop-NOUN"], by="premise", n_boot=50)
    cen = seen_unseen(tmp_path, "mnli", tmp_path / "seen.jsonl.gz", ["drop-NOUN"], by="premise",
                      centered=True, n_boot=50)
    raw, cen = raw.set_index("seen"), cen.set_index("seen")
    assert raw.loc[True, "delta"] == -50.0          # answer bias alone halves raw accuracy
    assert cen.loc[True, "delta"] == 0.0            # ... and centring removes it
    assert cen.loc[False, "delta"] == -100.0        # the real loss stays
    assert cen.loc["gap", "delta"] == 100.0
    hyp = seen_unseen(tmp_path, "mnli", tmp_path / "seen.jsonl.gz", ["drop-NOUN"], by="hypothesis", n_boot=10)
    assert list(hyp["seen"]) == [False]             # nobody in the found group, so no gap row


def test_reproduction_table_survives_csv_roundtrip(runs):
    from nlicorrupt.analysis.collect import write_csv
    from nlicorrupt.analysis.tables import write_all

    csv = write_csv(collect(), paths.results_dir() / "all_results.csv")
    files = write_all(csv, paths.results_dir() / "tables")
    rep = pd.read_csv(files["reproduction"])
    assert not rep.empty and 72.85 in set(rep["published"])


def test_answer_bias_separates_threshold_shift_from_lost_information():
    import numpy as np

    from nlicorrupt.analysis.answer_bias import collect

    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 50)
    signal = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.3, 100)

    def write(corr, margin_shift, noise_only=False):
        m = rng.normal(0, 1, 100) if noise_only else signal + margin_shift
        recs = [{"id": f"rte:v:{i}", "label": int(y), "scores": [0.0, float(x)], "pred": int(x > 0)}
                for i, (y, x) in enumerate(zip(labels, m))]
        write_jsonl(paths.runs_dir() / "logprob" / "m" / "rte" / "2022" / "k0" / f"{corr}.jsonl.gz", recs)

    write("clean", 0.0)
    write("drop-NOUN", -3.0)         # same information, everything pushed towards "no"
    write("shuffle-n1", 0.0, True)   # information gone
    t = collect().set_index("corruption")
    assert t.loc["clean", "accuracy"] > 95
    assert t.loc["drop-NOUN", "accuracy"] < 60 and t.loc["drop-NOUN", "top_answer_rate"] > 90
    assert t.loc["drop-NOUN", "centered_accuracy"] > 95 and t.loc["drop-NOUN", "auc"] > 99
    assert t.loc["shuffle-n1", "auc"] < 70 and t.loc["shuffle-n1", "delta_auc"] < -25


def test_heatmap_file_names_keep_dots_in_model_names(tmp_path):
    from nlicorrupt.analysis.plots import heatmap

    t = pd.DataFrame([{"task": "rte", "corruption": "drop-NOUN", "delta": -3.0},
                      {"task": "cola", "corruption": "drop-VERB", "delta": -9.0}])
    out = heatmap(t, "x", tmp_path / "logprob__qwen3.8-27b__2022__k0")
    assert out.name == "logprob__qwen3.8-27b__2022__k0.png" and out.exists()
    assert (tmp_path / "logprob__qwen3.8-27b__2022__k0.pdf").exists()
