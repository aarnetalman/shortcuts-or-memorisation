"""Commands: prompt (hosted APIs, dry-run by default), api-costs."""

from __future__ import annotations

import json
import sys

from .train import _list


def cmd_prompt(args):
    from ..prompting.runner import DEFAULT_SUBSAMPLE, PromptConfig, SpendNotConfirmed, execute, plan

    cfg = PromptConfig(
        model=args.model,
        tasks=_list(args.tasks),
        corruptions=_list(args.corruptions) if args.corruptions != "standard" else None,
        style=args.style,
        shots=args.shots,
        exemplars=args.exemplars,
        seed=args.seed,
        reasoning=args.reasoning,
        max_tokens=args.max_tokens,
        subsample={} if args.no_subsample else dict(DEFAULT_SUBSAMPLE),
        max_examples=args.max_examples,
        confirm_spend=args.confirm_spend,
        max_usd=args.max_usd,
        concurrency=args.concurrency,
    )
    summary = plan(cfg)["summary"]
    print(json.dumps(summary, indent=2))
    if summary["to_send"] and not args.confirm_spend:
        print("\nDRY RUN: nothing was sent. Paid calls need explicit approval; then re-run with "
              "--confirm-spend --max-usd <budget>.", file=sys.stderr)
        return 0
    if summary["to_send"] and sys.stdin.isatty():
        typed = input(f"Send {summary['to_send']} requests to {cfg.model} with a budget of "
                      f"${args.max_usd}? Type the budget to confirm: ").strip()
        if typed != str(args.max_usd) and typed != f"{args.max_usd:g}":
            print("not confirmed; nothing was sent", file=sys.stderr)
            return 1
    try:
        metrics = execute(cfg)
    except SpendNotConfirmed as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_api_costs(args):
    from ..prompting.cache import CostLog

    log = CostLog()
    totals: dict[str, float] = {}
    if log.path.exists():
        for line in open(log.path, encoding="utf-8"):
            e = json.loads(line)
            totals[e.get("run", "?")] = totals.get(e.get("run", "?"), 0.0) + float(e.get("cost_usd", 0))
    for run, usd in sorted(totals.items()):
        print(f"{usd:10.4f}  {run}")
    print(f"{sum(totals.values()):10.4f}  TOTAL ({log.path})")


def add_parser(sub):
    p = sub.add_parser("prompt", help="prompt a hosted model (dry run unless --confirm-spend)")
    p.add_argument("--model", required=True, help="alias in configs/providers.yaml")
    p.add_argument("--tasks", required=True, help="comma separated")
    p.add_argument("--corruptions", default="standard")
    p.add_argument("--style", default="2022", choices=["2021", "2022"])
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--exemplars", default="clean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning", default="off", choices=["off", "on"])
    p.add_argument("--max-tokens", type=int, default=None, help="default 16 (reasoning off) / 4096 (on)")
    p.add_argument("--no-subsample", action="store_true", help="full QQP/MNLI dev sets instead of 2,000 items")
    p.add_argument("--max-examples", type=int, default=None, help="cap per corruption, e.g. for a pilot")
    p.add_argument("--confirm-spend", action="store_true", help="actually send uncached requests (paid)")
    p.add_argument("--max-usd", type=float, default=None, help="hard budget for this run")
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("api-costs", help="sum logged API spending per run")
    p.set_defaults(func=cmd_api_costs)
