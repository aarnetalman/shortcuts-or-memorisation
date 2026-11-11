"""Commands: validate-fresh, contamination-infinigram, contamination-guess, contamination-mink."""

from __future__ import annotations

import sys
from pathlib import Path

from .train import _list


def cmd_validate_fresh(args):
    from ..fresh import private_root, read_items, validate

    path = Path(args.file) if args.file else private_root() / "fresh" / "nli.jsonl"
    items = read_items(path)
    errors = validate(items)
    for e in errors:
        print(e, file=sys.stderr)
    from collections import Counter

    print(f"{path}: {len(items)} items, labels {dict(Counter(i.get('label') for i in items))}, "
          f"sources {dict(Counter(i.get('source') for i in items))}")
    return 1 if errors else 0


def cmd_infinigram(args):
    from ..contamination.infinigram import InfiniGramRefused, run

    try:
        for p in run(_list(args.tasks), index=args.index):
            print(p)
    except InfiniGramRefused as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1


def cmd_guess(args):
    from ..contamination.local import run_guess

    print(run_guess(args.model, _list(args.tasks), batch_size=args.batch_size, fmt=args.format))


def cmd_mink(args):
    from ..contamination.local import run_mink

    print(run_mink(args.model, _list(args.tasks), batch_size=args.batch_size))


def add_parser(sub):
    p = sub.add_parser("validate-fresh", help="check the private fresh control-set file")
    p.add_argument("--file", default=None, help="default: $NLICORRUPT_PRIVATE/fresh/nli.jsonl")
    p.set_defaults(func=cmd_validate_fresh)

    p = sub.add_parser("contamination-infinigram", help="n-gram counts in a pre-training corpus")
    p.add_argument("--tasks", required=True)
    p.add_argument("--index", default=None, help="default: configs/contamination.yaml")
    p.set_defaults(func=cmd_infinigram)

    p = sub.add_parser("contamination-guess", help="TS-guessing with a local model")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--format", default="auto", choices=["auto", "text", "chat"])
    p.set_defaults(func=cmd_guess)

    p = sub.add_parser("contamination-mink", help="Min-K%%++ membership scores with a local model")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.set_defaults(func=cmd_mink)
