"""Package results for sharing through the git repository.

Copies into <repo>/results/:
  all_results.csv, tables/, figures/     built by `nlicorrupt results`
  runs/**/metrics.json, manifest*.json   per-run metrics and provenance (small)
  predictions/...                        per-item prediction files, only for --predictions globs
  logs/                                  job logs, only with --logs
  EXPORT.json                            what was exported, when, from which commit

Never exported, whatever is asked:
  - anything belonging to the private control set: any path with a 'fresh_nli' part
    (per-item files, manifests with example prompts, contamination outputs)
  - hidden states (*.npz), model weights, the API cache and cost log, NLTK/HF caches
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .runinfo import git_sha

PRIVATE_MARKERS = ("fresh_nli",)
NEVER_SUFFIXES = (".npz", ".safetensors", ".bin", ".pt", ".sqlite", ".sqlite3")
NEVER_DIRS = ("model", "hidden", "api")
SMALL_FILES = ("metrics.json", "manifest.json", "manifest_eval.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_private(rel: Path) -> bool:
    return any(m in part for part in rel.parts for m in PRIVATE_MARKERS)


def is_excluded(rel: Path) -> bool:
    return (
        is_private(rel)
        or rel.suffix in NEVER_SUFFIXES
        or any(part in NEVER_DIRS for part in rel.parts[:-1])
    )


def _matches(rel: Path, pattern: str) -> bool:
    """A glob relative to runs/ selects matching files and everything below matching folders."""
    pattern = pattern.strip("/")
    return fnmatch.fnmatch(str(rel), pattern) or any(
        fnmatch.fnmatch(str(parent), pattern) for parent in rel.parents if str(parent) != "."
    )


@dataclass
class ExportReport:
    out: Path
    copied: list[str] = field(default_factory=list)
    skipped_private: list[str] = field(default_factory=list)
    total_bytes: int = 0


def _copy(src: Path, dst: Path, report: ExportReport, rel_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    report.copied.append(rel_name)
    report.total_bytes += src.stat().st_size


def export(
    out: Path | None = None,
    predictions: list[str] = (),
    logs: bool = False,
    max_mb: float = 50.0,
    rebuild: bool = True,
) -> ExportReport:
    out = Path(out or repo_root() / "results")
    runs = paths.runs_dir()
    res = paths.results_dir()

    if rebuild:
        from .analysis.collect import collect, write_csv
        from .analysis.tables import write_all

        rows = collect()
        if not rows:
            raise FileNotFoundError(f"no runs found under {runs}")
        long_csv = write_csv(rows, res / "all_results.csv")
        files = write_all(long_csv, res / "tables")
        try:
            import pandas as pd

            from .analysis.plots import all_heatmaps

            all_heatmaps(pd.read_csv(files["corrupt_test"]), res / "figures")
        except ImportError:
            pass

    # Plan the copy first, so nothing is written if the size limit is exceeded.
    plan: list[tuple[Path, Path, str]] = []
    skipped: list[str] = []

    def consider(src: Path, rel: Path, dst: Path):
        if is_excluded(rel):
            if is_private(rel):
                skipped.append(str(rel))
            return
        plan.append((src, dst, str(Path(dst).relative_to(out))))

    for name in ("all_results.csv",):
        if (res / name).exists():
            consider(res / name, Path(name), out / name)
    for sub in ("tables", "figures"):
        for f in sorted((res / sub).rglob("*")) if (res / sub).exists() else []:
            if f.is_file():
                rel = f.relative_to(res)
                consider(f, rel, out / rel)

    if runs.exists():
        for f in sorted(runs.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(runs)
            if f.name in SMALL_FILES:
                consider(f, rel, out / "runs" / rel)
            elif any(_matches(rel, pat) for pat in predictions):
                consider(f, rel, out / "predictions" / rel)

    if logs:
        logdir = repo_root() / "logs"
        for f in sorted(logdir.glob("*.out")) if logdir.exists() else []:
            consider(f, Path("logs") / f.name, out / "logs" / f.name)

    total = sum(src.stat().st_size for src, _, _ in plan)
    if total > max_mb * 1e6:
        biggest = sorted(plan, key=lambda t: -t[0].stat().st_size)[:10]
        lines = "\n".join(f"  {s.stat().st_size / 1e6:8.1f} MB  {r}" for s, _, r in biggest)
        raise ValueError(
            f"export would be {total / 1e6:.1f} MB, above the {max_mb:g} MB limit. Largest files:\n{lines}\n"
            "Narrow --predictions, or raise --max-mb."
        )

    report = ExportReport(out=out, skipped_private=skipped)
    for src, dst, rel in plan:
        _copy(src, dst, report, rel)
    meta = {
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit": git_sha(),
        "data_root": str(paths.data_root()),
        "files": len(report.copied),
        "megabytes": round(report.total_bytes / 1e6, 2),
        "predictions": list(predictions),
        "logs": logs,
        "private_items_skipped": len(skipped),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "EXPORT.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return report


def push(out: Path, branch: str | None = None, message: str | None = None) -> str:
    """Commit the export on a new branch and push it, then switch back to the branch the
    clone was on. Uses the clone's own git identity.

    Switching back matters: otherwise the clone stays on the results branch, `git pull`
    reports "Already up to date" against that branch, and code merged to main never
    arrives (the installed package runs from this checkout).
    """
    repo = repo_root()
    branch = branch or "results-" + time.strftime("%Y-%m-%d-%H%M")

    def git(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip()
            hint = ""
            if "tell me who you are" in msg or "user.email" in msg:
                hint = ("\nSet your git identity first:\n  git config --global user.name \"Your Name\"\n"
                        "  git config --global user.email you@example.org")
            raise RuntimeError(f"git {' '.join(a)} failed:\n{msg}{hint}")
        return r.stdout

    rel = Path(out).resolve().relative_to(repo.resolve())
    for key in ("user.name", "user.email"):
        if subprocess.run(["git", "-C", str(repo), "config", key], capture_output=True).returncode != 0:
            raise RuntimeError(f"git {key} is not set. Set it first:\n"
                               f"  git config --global {key} \"...\"")
    start = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if start == "HEAD":  # detached
        start = git("rev-parse", "HEAD").strip()
    git("checkout", "-b", branch)
    try:
        git("add", "--", str(rel))
        git("commit", "-m", message or "Add exported results", "--", str(rel))
        git("push", "-u", "origin", branch)
    finally:
        git("checkout", start)
    return branch
