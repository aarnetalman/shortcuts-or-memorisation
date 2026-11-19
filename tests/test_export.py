import json
import subprocess

import numpy as np
import pytest

from nlicorrupt import export as ex
from nlicorrupt import paths
from nlicorrupt.cli import main
from nlicorrupt.data import write_jsonl


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git clone with a bare 'origin', standing in for the user's clone."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    r = tmp_path / "clone"
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    for k, v in (("user.name", "Test"), ("user.email", "test@example.org")):
        subprocess.run(["git", "-C", str(r), "config", k, v], check=True)
    (r / "README.md").write_text("x\n")
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(r), "remote", "add", "origin", str(origin)], check=True)
    monkeypatch.setattr(ex, "repo_root", lambda: r)
    return r, origin


@pytest.fixture
def runs():
    rd = paths.runs_dir()
    for task in ("mnli", "fresh_nli"):
        base = rd / "logprob" / "olmo3-32b" / task / "2022" / "k0"
        put(base / "metrics.json", {"clean": {"accuracy": 80.0, "n": 4}, "drop-NOUN": {"accuracy": 60.0, "n": 4}})
        put(base / "manifest.json", {"example_prompt": f"secret {task} item text"})
        write_jsonl(base / "clean.jsonl.gz", [{"id": f"{task}:x:1", "label": 0, "pred": 0}])
    hid = rd / "hidden" / "olmo3-32b" / "mnli" / "2022" / "validation_matched"
    hid.mkdir(parents=True)
    np.savez(hid / "clean.npz", X=np.zeros((2, 2)))
    ft = rd / "finetune" / "roberta-base" / "mnli" / "2022" / "train-clean" / "seed13"
    put(ft / "metrics.json", {"mnli/validation_matched/clean": {"accuracy": 88.0, "n": 4},
                              "mnli/validation_matched/drop-NOUN": {"accuracy": 73.0, "n": 4}})
    (ft / "model").mkdir()
    (ft / "model" / "model.safetensors").write_bytes(b"0" * 100)
    return rd


def test_export_copies_summaries_and_never_private_or_heavy_files(repo, runs):
    r, _ = repo
    rep = ex.export()
    out = r / "results"
    assert (out / "all_results.csv").exists() and (out / "tables" / "corrupt_test.csv").exists()
    assert list((out / "figures").glob("*.png"))
    assert (out / "runs" / "logprob" / "olmo3-32b" / "mnli" / "2022" / "k0" / "manifest.json").exists()
    exported = [str(p) for p in out.rglob("*") if p.is_file()]
    assert not [p for p in exported if "fresh_nli" in p.split("results", 1)[1]]
    assert not [p for p in exported if p.endswith((".npz", ".safetensors"))]
    assert not (out / "predictions").exists()  # only on request
    assert all("secret fresh_nli" not in open(p, errors="ignore").read() for p in exported if p.endswith(".json"))
    assert len(rep.skipped_private) >= 2
    meta = json.loads((out / "EXPORT.json").read_text())
    assert meta["files"] == len(rep.copied)


def test_requested_predictions_and_size_limit(repo, runs):
    r, _ = repo
    ex.export(predictions=["logprob/olmo3-*"])
    pred = r / "results" / "predictions" / "logprob" / "olmo3-32b"
    assert (pred / "mnli" / "2022" / "k0" / "clean.jsonl.gz").exists()
    assert not (pred / "fresh_nli").exists()
    with pytest.raises(ValueError, match="above the"):
        ex.export(out=r / "results2", max_mb=1e-6)
    assert not (r / "results2").exists()


def test_push_commits_only_results_on_a_new_branch(repo, runs):
    r, origin = repo
    (r / "unrelated.txt").write_text("not for the commit\n")
    before = subprocess.run(["git", "-C", str(r), "branch", "--show-current"], capture_output=True, text=True).stdout
    assert main(["export", "--push", "--branch", "results-test"]) == 0
    after = subprocess.run(["git", "-C", str(r), "branch", "--show-current"], capture_output=True, text=True).stdout
    assert after == before  # back on the original branch, so `git pull` keeps working
    files = subprocess.run(["git", "-C", str(origin), "ls-tree", "-r", "--name-only", "results-test"],
                           capture_output=True, text=True, check=True).stdout.split()
    assert "results/all_results.csv" in files and "unrelated.txt" not in files
    assert not [f for f in files if "fresh_nli" in f]
    author = subprocess.run(["git", "-C", str(origin), "log", "-1", "--format=%an <%ae>", "results-test"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert author == "Test <test@example.org>"


def test_no_runs_is_a_clear_error(repo):
    assert main(["export"]) == 1


def test_push_without_git_identity_fails_clearly_and_leaves_branch_alone(repo, runs, monkeypatch, capsys):
    r, _ = repo
    for k in ("user.name", "user.email"):
        subprocess.run(["git", "-C", str(r), "config", "--unset", k], check=True)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    before = subprocess.run(["git", "-C", str(r), "branch", "--show-current"], capture_output=True, text=True).stdout
    assert main(["export", "--push", "--branch", "results-x"]) == 1
    assert "git user.name is not set" in capsys.readouterr().err
    after = subprocess.run(["git", "-C", str(r), "branch", "--show-current"], capture_output=True, text=True).stdout
    assert before == after
