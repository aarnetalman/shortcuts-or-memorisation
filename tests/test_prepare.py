from nlicorrupt import prepare
from nlicorrupt.data import read_jsonl, write_jsonl

RAW = [
    {"id": "rte:validation:0", "label": 0, "sentence1": "The cat sat on the mat.", "sentence2": "A cat sat."},
    {"id": "rte:validation:1", "label": 1, "sentence1": "Dogs bark.", "sentence2": "Cats."},
]


def test_jsonl_roundtrip_gz(tmp_path):
    p = tmp_path / "a" / "x.jsonl.gz"
    assert write_jsonl(p, RAW) == 2
    assert list(read_jsonl(p)) == RAW
    assert not list(tmp_path.glob("a/*.tmp*"))


def test_tag_and_materialize(monkeypatch, needs_nltk):
    monkeypatch.setattr(prepare, "load_raw", lambda task, split: [dict(r) for r in RAW])
    tp = prepare.tag_split("rte", "validation", workers=1)
    tagged = list(read_jsonl(tp))
    assert tagged[0]["tagged"]["sentence1"][1] == ["cat", "NOUN"]

    paths21 = prepare.materialize("rte", "validation", ["clean", "drop-NOUN"], style="2021")
    paths22 = prepare.materialize("rte", "validation", ["clean", "drop-NOUN"], style="2022")
    clean = list(read_jsonl(paths22["clean"]))
    assert clean == RAW
    d22 = prepare.load_corrupted("rte", "validation", "drop-NOUN", "2022")
    d21 = prepare.load_corrupted("rte", "validation", "drop-NOUN", "2021")
    assert len(d22) == 2 and d22[1]["sentence2"] == "."
    assert len(d21) == 2  # "." survives, so nothing is empty
    assert paths21["drop-NOUN"].exists()

    # Rerunning does not rewrite existing files.
    mtime = paths22["drop-NOUN"].stat().st_mtime_ns
    prepare.materialize("rte", "validation", ["drop-NOUN"], style="2022")
    assert paths22["drop-NOUN"].stat().st_mtime_ns == mtime


def test_load_corrupted_missing_file_explains_how_to_make_it():
    import pytest

    with pytest.raises(FileNotFoundError, match="nlicorrupt prepare"):
        prepare.load_corrupted("rte", "validation", "drop-VERB")


def test_nltk_data_dir_is_created_private(tmp_path, monkeypatch):
    import os

    from nlicorrupt.corrupt import tagging

    root = tmp_path / "nltk_private"
    monkeypatch.setenv("NLTK_DATA", str(root))
    monkeypatch.setattr("nltk.download", lambda *a, **k: True)
    old = os.umask(0o002)
    try:
        tagging.ensure_nltk_data(download=True)
        assert os.stat(root).st_mode & 0o022 == 0
        os.chmod(root, 0o775)  # an existing group-writable directory is fixed too
        tagging.ensure_nltk_data(download=True)
        assert os.stat(root).st_mode & 0o022 == 0
    finally:
        os.umask(old)
