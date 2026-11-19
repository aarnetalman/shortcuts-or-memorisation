"""Command: grid — expand an experiment YAML into one command per line.

The output file is what `scripts/run_grid.sh` runs, one line per run.
Each command skips itself if its run is already complete, so a grid can be
resubmitted after failures.

YAML format:

  blocks:
    - name: roberta-2022            # optional
      command: finetune             # any nlicorrupt command
      matrix:                       # cartesian product
        task: [mnli, mrpc]
        seed: [13, 42, 87]
      args:                         # fixed options; lists are joined with commas
        model: roberta-base
        eval_corruptions: [clean, drop-NOUN]
        no_bf16: false              # booleans become flags when true
      time: "02:00:00"              # optional: only used to name the output file
"""

from __future__ import annotations

import itertools
import shlex
from pathlib import Path


def _opt(key: str, value) -> list[str]:
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    return [flag, str(value)]


def expand(cfg: dict, only: list[str] | None = None) -> list[str]:
    lines: list[str] = []
    for i, block in enumerate(cfg.get("blocks", [])):
        name = block.get("name", f"block{i}")
        if only and name not in only:
            continue
        matrix = block.get("matrix", {})
        keys = list(matrix)
        for combo in itertools.product(*(matrix[k] for k in keys)):
            opts = dict(block.get("args", {}))
            opts.update(dict(zip(keys, combo)))
            argv = ["nlicorrupt", block["command"]]
            for k, v in opts.items():
                argv += _opt(k, v)
            lines.append(shlex.join(argv))
    return lines


def cmd_grid(args):
    from ..config import load_yaml

    cfg = load_yaml(args.config)
    only = [b for b in args.blocks.split(",") if b] or None
    lines = expand(cfg, only)
    out = Path(args.out) if args.out else Path("grids") / (Path(args.config).stem + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} commands -> {out}")


def add_parser(sub):
    p = sub.add_parser("grid", help="expand an experiment YAML into a grid file, one command per line")
    p.add_argument("config", help="path, or name under configs/, e.g. grids/reproduce.yaml")
    p.add_argument("--out", default=None, help="default: grids/<config name>.txt")
    p.add_argument("--blocks", default="", help="only these block names, comma separated")
    p.set_defaults(func=cmd_grid)
