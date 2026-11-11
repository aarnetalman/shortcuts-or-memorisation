"""Hypothesis-only shortcut split (offline)."""

from nlicorrupt.analysis import shortcuts
from nlicorrupt.data import read_jsonl, write_jsonl


def _rec(i, hyp, label):
    return {"id": f"mnli:x:{i}", "label": label, "premise": "a man is outside", "hypothesis": hyp}


def test_hypothesis_only_finds_a_negation_cue():
    train = [_rec(i, "nobody is outside" if i % 2 else "a person is outside", 2 if i % 2 else 0) for i in range(60)]
    ev = [_rec(100, "nobody is here", 2), _rec(101, "a person is here", 0), _rec(102, "nobody is here", 0)]
    rows = shortcuts.hypothesis_only("mnli", train=train, evaluation=ev)
    assert [r["easy"] for r in rows] == [True, True, False]


def test_table_compares_easy_and_hard_items(tmp_path):
    ids = [f"mnli:validation_matched:{i}" for i in range(40)]
    write_jsonl(shortcuts.shortcut_file("mnli", tmp_path),
                [{"id": i, "label": 0, "hyp_pred": 0, "easy": n < 20} for n, i in enumerate(ids)])
    run = tmp_path / "logprob" / "m" / "mnli" / "2022" / "k0"
    write_jsonl(run / "clean.jsonl.gz", [{"id": i, "label": 0, "pred": 0} for i in ids])
    # corruption hurts only the hard items
    write_jsonl(run / "drop-NOUN.jsonl.gz", [{"id": i, "label": 0, "pred": 0 if n < 20 else 1} for n, i in enumerate(ids)])
    t = shortcuts.table(["mnli"], ["drop-NOUN"], n_boot=50, root=tmp_path).set_index("group")
    assert t.loc["easy", "delta"] == 0.0 and t.loc["hard", "delta"] == -100.0 and t.loc["gap", "delta"] == 100.0
    assert not bool(t.loc["easy", "centered"])  # no scores in these files
    assert len(list(read_jsonl(shortcuts.shortcut_file("mnli", tmp_path)))) == 40


def test_table_handles_items_without_scores(tmp_path):
    ids = [f"mnli:validation_matched:{i}" for i in range(40)]
    write_jsonl(shortcuts.shortcut_file("mnli", tmp_path),
                [{"id": i, "label": 0, "hyp_pred": 0, "easy": n < 20} for n, i in enumerate(ids)])
    run = tmp_path / "prompt" / "api-model" / "mnli" / "2022" / "k0" / "reasoning-off"
    rec = lambda i, n: {"id": i, "label": 0, "pred": 0, **({} if n == 5 else {"scores": [0.0, -1.0, -2.0]})}
    write_jsonl(run / "clean.jsonl.gz", [rec(i, n) for n, i in enumerate(ids)])  # item 5 was filtered
    write_jsonl(run / "drop-NOUN.jsonl.gz", [rec(i, n) for n, i in enumerate(ids)])
    t = shortcuts.table(["mnli"], ["drop-NOUN"], n_boot=20, root=tmp_path).set_index("group")
    assert bool(t.loc["easy", "centered"]) and t.loc["easy", "clean"] == 95.0  # 19 of 20, item 5 wrong
    # a run without any scores (e.g. Mistral) falls back to raw predictions
    run2 = tmp_path / "prompt" / "raw-model" / "mnli" / "2022" / "k0" / "reasoning-off"
    for f in ("clean", "drop-NOUN"):
        write_jsonl(run2 / f"{f}.jsonl.gz", [{"id": i, "label": 0, "pred": 0} for i in ids])
    t2 = shortcuts.table(["mnli"], ["drop-NOUN"], n_boot=20, root=tmp_path)
    assert not t2[t2.model == "raw-model"].centered.any()
