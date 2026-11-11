"""Commands: fresh-premises, fresh-generate, fresh-natural, fresh-sheets, fresh-merge, fresh-artefacts.

Files, all under $NLICORRUPT_PRIVATE/fresh/:
  premises.jsonl               fresh-premises
  generated.jsonl              fresh-generate (candidate items)
  natural.jsonl                fresh-natural (naturally occurring pairs, no intended label)
  human/write_*.csv            writing templates for hand-written items (fresh-sheets --template)
  candidates.jsonl             generated + hand-written candidates (fresh-sheets)
  validation/validate_<A>.csv  one sheet per annotator (fresh-sheets)
  nli.jsonl                    validated items (fresh-merge): the fresh_nli task
  validation/report.json       agreement report (fresh-merge)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .train import _list


def _dir() -> Path:
    from ..fresh import private_root

    return private_root() / "fresh"


def cmd_premises(args):
    from ..config import load_yaml
    from ..freshgen import premises

    cfg = load_yaml("fresh.yaml")["premises"]
    if args.since:
        cfg["since"] = args.since
    if args.sources:
        cfg["sources"] = _list(args.sources)
    if args.per_source:
        cfg["per_source"] = args.per_source
    ig = None
    if cfg.get("infinigram_check") and not args.no_infinigram:
        from ..contamination.infinigram import InfiniGram

        icfg = load_yaml("contamination.yaml")["infinigram"]
        index = args.index or icfg.get("index")
        if not index:
            print("set --index (e.g. v4_olmo-mix-1124_llama) or use --no-infinigram", file=sys.stderr)
            return 1
        ig = InfiniGram(icfg["endpoint"], index, icfg["sleep_seconds"])
    try:
        ps = premises.fetch(cfg, infinigram=ig)
    except ValueError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    out = premises.write(ps, _dir() / "premises.jsonl")
    print(f"{len(ps)} premises -> {out}  {dict(Counter(p['source'] for p in ps))}")
    return 0


def cmd_generate(args):
    from ..freshgen import generate as g
    from ..prompting.cache import ResponseCache
    from ..prompting.runner import SpendNotConfirmed

    cfg = g.config()
    ps = g.read_jsonl(_dir() / "premises.jsonl")
    if args.limit:
        ps = ps[: args.limit]
    styles = _list(args.styles) if args.styles else cfg["styles"]
    examples = g.mnli_examples(cfg["mnli_examples"]) if "mnli" in styles else []
    reqs = g.build_requests(ps, styles, _list(args.models), examples, cfg["temperature"], cfg["max_tokens"])
    p = g.plan(reqs, ResponseCache())
    print(json.dumps({k: v for k, v in p.items() if k != "todo"}, indent=2))
    if p["to_send"] and not args.confirm_spend:
        print("\nDRY RUN: nothing was sent. Paid calls need explicit approval; then re-run with "
              "--confirm-spend --max-usd <budget>.", file=sys.stderr)
        return 0
    if p["to_send"] and sys.stdin.isatty():
        typed = input(f"Send {p['to_send']} requests with a budget of ${args.max_usd}? "
                      "Type the budget to confirm: ").strip()
        if typed not in (str(args.max_usd), f"{args.max_usd:g}"):
            print("not confirmed; nothing was sent", file=sys.stderr)
            return 1
    try:
        res = g.execute(reqs, args.confirm_spend, args.max_usd)
    except SpendNotConfirmed as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    out = g.write_jsonl(res["items"], _dir() / "generated.jsonl")
    print(f"{len(res['items'])} candidate items -> {out}; {res['unparsed']} unparsed responses, "
          f"{res['missing']} not yet generated")
    return 0


def cmd_natural(args):
    from ..config import load_yaml
    from ..freshgen import generate as g
    from ..freshgen import natural

    cfg = load_yaml("fresh.yaml")["premises"]
    if args.since:
        cfg["since"] = args.since
    try:
        items = natural.build(cfg, _list(args.kinds), args.per_kind, max_pages=args.max_pages)
    except ValueError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    out = g.write_jsonl(items, _dir() / "natural.jsonl")
    print(f"{len(items)} natural pairs -> {out}  {dict(Counter(i['style'] for i in items))}")
    return 0


def cmd_sheets(args):
    import random

    from ..config import load_yaml
    from ..freshgen import generate as g
    from ..freshgen import sheets as s

    d = _dir()
    ps = g.read_jsonl(d / "premises.jsonl")
    if args.template:
        rng = random.Random(f"template|{args.seed}")
        picked = rng.sample(ps, min(args.template, len(ps)))
        out = s.write_template(picked, d / "human" / f"write_{args.author or 'ALL'}.csv")
        print(f"writing template with {len(picked)} premises -> {out}")
        return 0
    cands = []
    for f in ("generated.jsonl", "natural.jsonl"):
        if (d / f).exists():
            cands += g.read_jsonl(d / f)
    by_pid = {p["pid"]: p for p in ps}
    for t in sorted((d / "human").glob("write_*.csv")) if (d / "human").exists() else []:
        cands += s.read_template(t, by_pid)
    if not cands:
        print(f"no candidates: run fresh-generate or fresh-natural, or fill a template in {d / 'human'}",
              file=sys.stderr)
        return 1
    if args.premises:
        cands = s.sample_premises(cands, args.premises, seed=args.seed)
    g.write_jsonl(cands, d / "candidates.jsonl")
    annotators = _list(args.annotators) if args.annotators else load_yaml("fresh.yaml")["validation"]["annotators"]
    vdir = d / "validation"
    existing = [p for p in (vdir / f"validate_{a}.csv" for a in annotators) if p.exists()]
    if existing and not args.overwrite:
        print(f"sheets already exist ({', '.join(p.name for p in existing)}); filled labels would be lost. "
              "Use --overwrite to replace them.", file=sys.stderr)
        return 1
    paths = s.write_sheets(cands, annotators, vdir, seed=args.seed)
    print(f"{len(cands)} candidates ({dict(Counter(c['style'] for c in cands))}) -> " + ", ".join(map(str, paths)))
    return 0


def cmd_merge(args):
    from ..fresh import validate
    from ..freshgen import generate as g
    from ..freshgen import sheets as s

    d = _dir()
    cands = g.read_jsonl(d / "candidates.jsonl")
    labels_file = Path(args.labels) if args.labels else d / "validation" / "labels.csv"
    if labels_file.exists():  # the annotation tool's export: all raters in one file
        sheets, rater_stats = s.read_labels_file(labels_file)
    else:
        sheets = {p.stem.removeprefix("validate_"): s.read_sheet(p)
                  for p in sorted((d / "validation").glob("validate_*.csv"))}
        rater_stats = {"raters": len(sheets)}
    kept, rep = s.merge(cands, sheets, min_labels=args.min_labels)
    rep["raters"] = rater_stats
    errors = validate(kept)
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        return 1
    g.write_jsonl(kept, d / "nli.jsonl")
    (d / "validation" / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"{len(kept)} validated items -> {d / 'nli.jsonl'}")
    return 0


def cmd_artefacts(args):
    """Hypothesis-only cues in the fresh items, measured with the MNLI classifier."""
    from ..analysis.shortcuts import hypothesis_only
    from ..freshgen import generate as g
    from ..tasks import NLI_LABELS

    path = Path(args.file) if args.file else _dir() / "candidates.jsonl"
    # natural pairs only have a label after validation: run on nli.jsonl to include them
    items = [it for it in g.read_jsonl(path) if it.get("label")]
    ev = [{"id": it["id"], "label": NLI_LABELS.index(it["label"]), "premise": it["premise"],
           "hypothesis": it["hypothesis"]} for it in items]
    res = {r["id"]: r for r in hypothesis_only("mnli", evaluation=ev)}
    neg = ("not", "no", "never", "nobody", "nothing", "none", "n't")
    out = {}
    for style in sorted({it["style"] for it in items}):
        its = [it for it in items if it["style"] == style]
        row = {"n": len(its), "hyp_only_accuracy": round(100.0 * sum(res[it["id"]]["easy"] for it in its) / len(its), 1)}
        for lab in NLI_LABELS:
            hs = [it["hypothesis"].lower() for it in its if it["label"] == lab]
            if hs:
                row[f"{lab}_words"] = round(sum(len(h.split()) for h in hs) / len(hs), 1)
                row[f"{lab}_negation_%"] = round(100.0 * sum(any(w in h.replace("n't", " n't ").split() for w in neg)
                                                             for h in hs) / len(hs), 1)
        out[style] = row
    print(json.dumps(out, indent=2))
    print("For reference, the same classifier reaches about 54% on MNLI (chance 33%).")
    return 0


def cmd_results(args):
    from ..analysis.fresh_results import write

    for p in write(n_boot=args.n_boot):
        print(f"wrote {p}")
    print("These tables hold aggregate scores only and are included by export.sh.")
    return 0


def add_parser(sub):
    p = sub.add_parser("fresh-premises", help="fetch fresh premises from Wikinews and new Wikipedia articles")
    p.add_argument("--since", default=None, help="override premises.since in configs/fresh.yaml (YYYY-MM-DD)")
    p.add_argument("--sources", default=None, help="wikinews,wikipedia")
    p.add_argument("--per-source", type=int, default=None)
    p.add_argument("--index", default=None, help="infini-gram index for the freshness check")
    p.add_argument("--no-infinigram", action="store_true", help="skip the corpus check")
    p.set_defaults(func=cmd_premises)

    p = sub.add_parser("fresh-generate", help="generate hypotheses for the fresh premises (dry run by default)")
    p.add_argument("--models", required=True, help="generator aliases in configs/providers.yaml, comma separated")
    p.add_argument("--styles", default=None, help="mnli,cuefree (default from configs/fresh.yaml)")
    p.add_argument("--limit", type=int, default=None, help="only the first N premises (e.g. a pilot)")
    p.add_argument("--confirm-spend", action="store_true")
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("fresh-natural", help="naturally occurring pairs: Wikinews headlines, Wikipedia revisions")
    p.add_argument("--kinds", default="headline,revision")
    p.add_argument("--per-kind", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=600,
                   help="new Wikipedia pages to check for revision pairs (about 1 in 18 yields a pair)")
    p.add_argument("--since", default=None, help="override premises.since in configs/fresh.yaml")
    p.set_defaults(func=cmd_natural)

    p = sub.add_parser("fresh-sheets", help="validation sheets per annotator, or a writing template")
    p.add_argument("--annotators", default=None, help="e.g. AT,SC (default from configs/fresh.yaml)")
    p.add_argument("--template", type=int, default=None, help="write a template with N premises for hand-written items")
    p.add_argument("--author", default=None, help="initials for the template file name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="replace existing validation sheets")
    p.add_argument("--premises", type=int, default=None,
                   help="validate only N generated premises (all styles and labels of each, balanced "
                        "across generators and sources); hand-written and natural items are always kept")
    p.set_defaults(func=cmd_sheets)

    p = sub.add_parser("fresh-merge", help="merge filled validation sheets into the fresh item file")
    p.add_argument("--labels", default=None,
                   help="annotation-tool export (one row per label); default: fresh/validation/labels.csv "
                        "if present, else the validate_<A>.csv sheets")
    p.add_argument("--min-labels", type=int, default=2, help="labels an item needs to be kept")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("fresh-results", help="scores-only summary of all runs on the fresh set (exportable)")
    p.add_argument("--n-boot", type=int, default=500)
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("fresh-artefacts", help="hypothesis-only cues in fresh items, by style")
    p.add_argument("--file", default=None, help="default: $NLICORRUPT_PRIVATE/fresh/candidates.jsonl")
    p.set_defaults(func=cmd_artefacts)
