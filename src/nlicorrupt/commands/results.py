"""Commands: results (tables and figures), export (share results through the repo)."""

from __future__ import annotations

from pathlib import Path


def cmd_results(args):
    import pandas as pd

    from .. import paths
    from ..analysis.collect import collect, write_csv
    from ..analysis.plots import all_heatmaps
    from ..analysis.tables import write_all

    out = Path(args.out) if args.out else paths.results_dir()
    rows = collect()
    if not rows:
        print(f"no results under {paths.runs_dir()}")
        return 1
    long_csv = write_csv(rows, out / "all_results.csv")
    files = write_all(long_csv, out / "tables")
    print(f"{len(rows)} metric rows -> {long_csv}")
    for name, p in files.items():
        print(f"table {name}: {p}")
    if not args.no_plots:
        figs = all_heatmaps(pd.read_csv(files["corrupt_test"]), out / "figures")
        print(f"{len(figs)} figures -> {out / 'figures'}")
    return 0


def cmd_seen_unseen(args):
    from .. import paths
    from ..analysis.tables import seen_unseen_all

    corruptions = [c for c in args.corruptions.split(",") if c]
    t = seen_unseen_all(args.index, corruptions, n_boot=args.n_boot)
    if t.empty:
        print(f"no logprob runs with infini-gram counts for {args.index} under {paths.runs_dir()}")
        return 1
    out = Path(args.out) if args.out else paths.results_dir() / "tables" / "seen_unseen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.round(2).to_csv(out, index=False)
    gaps = t[t.seen.astype(str) == "gap"]
    excl = gaps[(gaps.lo > 0) | (gaps.hi < 0)]
    print(f"{len(t)} rows -> {out}; {len(excl)} of {len(gaps)} seen-minus-unseen gaps have a 95% CI excluding 0")
    return 0


def cmd_shortcuts(args):
    from .. import paths
    from ..analysis import shortcuts

    tasks = [t for t in args.tasks.split(",") if t]
    acc = shortcuts.build(tasks, force=args.retrain)
    for t, a in acc.items():
        print(f"{t}: hypothesis-only classifier accuracy {a:.1f}%")
    t = shortcuts.table(tasks, [c for c in args.corruptions.split(",") if c], n_boot=args.n_boot)
    if t.empty:
        print(f"no zero-shot runs with predictions for {args.tasks} under {paths.runs_dir()}")
        return 1
    out = Path(args.out) if args.out else paths.results_dir() / "tables" / "shortcuts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.round(2).to_csv(out, index=False)
    gaps = t[t.group.astype(str) == "gap"]
    excl = gaps[(gaps.lo > 0) | (gaps.hi < 0)]
    print(f"{len(t)} rows -> {out}; {len(excl)} of {len(gaps)} easy-minus-hard gaps have a 95% CI excluding 0")
    return 0


def cmd_export(args):
    import sys

    from ..export import export, push

    try:
        rep = export(out=args.out, predictions=[p for p in args.predictions.split(",") if p],
                     logs=args.logs, max_mb=args.max_mb, rebuild=not args.no_rebuild)
    except (FileNotFoundError, ValueError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1
    print(f"exported {len(rep.copied)} files ({rep.total_bytes / 1e6:.1f} MB) -> {rep.out}")
    if rep.skipped_private:
        print(f"left out {len(rep.skipped_private)} files belonging to the private control set")
    if args.push:
        try:
            branch = push(rep.out, args.branch, args.message)
        except RuntimeError as e:
            print(f"push failed: {e}", file=sys.stderr)
            return 1
        print(f"pushed branch {branch}; share its name to have the results analysed")
    else:
        print("review the files, then rerun with --push (or commit results/ yourself)")
    return 0


def add_parser(sub):
    p = sub.add_parser("results", help="collect runs into tables (CSV) and Δ heatmaps")
    p.add_argument("--out", default=None, help="default: $NLICORRUPT_DATA/results")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("seen-unseen", help="corruption Δ on items found vs not found by infini-gram, per field")
    p.add_argument("--index", required=True, help="infini-gram index, e.g. v4_olmo-mix-1124_llama")
    p.add_argument("--corruptions", default="drop-NOUN,drop-VERB,shuffle-n1,hyp-only,keep-NOUN+VERB")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--out", default=None, help="default: $NLICORRUPT_DATA/results/tables/seen_unseen.csv")
    p.set_defaults(func=cmd_seen_unseen)

    p = sub.add_parser("shortcuts", help="corruption Δ on items a hypothesis-only classifier gets right vs wrong")
    p.add_argument("--tasks", default="mnli,anli_r1,anli_r2,anli_r3,rte")
    p.add_argument("--corruptions", default="drop-NOUN,drop-VERB,shuffle-n1,keep-NOUN+VERB,hyp-only")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--retrain", action="store_true", help="retrain the classifier even if its output exists")
    p.add_argument("--out", default=None, help="default: $NLICORRUPT_DATA/results/tables/shortcuts.csv")
    p.set_defaults(func=cmd_shortcuts)

    p = sub.add_parser("export", help="copy results into the repo's results/ folder for sharing (optionally push)")
    p.add_argument("--out", default=None, help="default: <repo>/results")
    p.add_argument("--predictions", default="",
                   help="per-item prediction folders to include, globs relative to runs/, comma separated "
                        "(e.g. 'logprob/olmo3-*,contamination/infinigram')")
    p.add_argument("--logs", action="store_true", help="include job logs from logs/*.out")
    p.add_argument("--max-mb", type=float, default=50.0, help="refuse to export more than this")
    p.add_argument("--no-rebuild", action="store_true", help="do not rerun `nlicorrupt results` first")
    p.add_argument("--push", action="store_true", help="commit results/ on a new branch and push it")
    p.add_argument("--branch", default=None, help="branch name for --push (default: results-<date>-<time>)")
    p.add_argument("--message", default=None, help="commit message for --push")
    p.set_defaults(func=cmd_export)
