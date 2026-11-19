"""Command line interface: `nlicorrupt <command> ...`.

Each command lives in its own module and registers itself through add_parser().
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

# module name -> command(s) it registers
COMMAND_MODULES = [
    "nlicorrupt.commands.setup",
    "nlicorrupt.commands.grid",
    "nlicorrupt.commands.train",
    "nlicorrupt.commands.lm",
    "nlicorrupt.commands.prompt",
    "nlicorrupt.commands.control",
    "nlicorrupt.commands.fresh",
    "nlicorrupt.commands.results",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nlicorrupt", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for mod in COMMAND_MODULES:
        importlib.import_module(mod).add_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
