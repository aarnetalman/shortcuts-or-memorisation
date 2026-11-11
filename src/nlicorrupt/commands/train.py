"""Commands: finetune, evaluate-model."""

from __future__ import annotations

from pathlib import Path


def _list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def expand_corruptions(spec: str, pair: bool) -> list[str]:
    from ..corrupt import standard_eval_corruptions

    out: list[str] = []
    for c in _list(spec):
        out += standard_eval_corruptions(pair) if c == "standard" else [c]
    return list(dict.fromkeys(out))


def cmd_finetune(args):
    from ..finetune import FinetuneConfig, run
    from ..tasks import get_task

    pair = get_task(args.task).is_pair
    cfg = FinetuneConfig(
        model=args.model,
        task=args.task,
        train_corruption=args.train_corruption,
        eval_corruptions=expand_corruptions(args.eval_corruptions, pair),
        style=args.style,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_len=args.max_len,
        eval_batch_size=args.eval_batch_size,
        bf16=not args.no_bf16,
        save_model=args.save_model,
        train_task=args.train_task,
        extra_eval_tasks=_list(args.extra_eval_tasks),
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        out_dir=args.out_dir,
    )
    print(run(cfg))


def cmd_evaluate_model(args):
    from .. import paths
    from ..finetune import evaluate_model
    from ..tasks import get_task

    tasks = _list(args.tasks)
    corrs = expand_corruptions(args.corruptions, get_task(tasks[0]).is_pair)
    out = Path(args.out_dir) if args.out_dir else (
        Path(args.model) if Path(args.model).is_dir()
        else paths.runs_dir() / "evaluate" / args.model.replace("/", "__") / args.style)
    print(evaluate_model(args.model, out, tasks, corrs, style=args.style, max_len=args.max_len))


def add_parser(sub):
    p = sub.add_parser("finetune", help="fine-tune an encoder and evaluate on clean and corrupted sets")
    p.add_argument("--model", required=True, help="alias from configs/models.yaml or hub id")
    p.add_argument("--task", required=True)
    p.add_argument("--train-corruption", default="clean")
    p.add_argument("--eval-corruptions", default="standard", help="comma list; 'standard' expands to the full set")
    p.add_argument("--style", default="2022", choices=["2021", "2022"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--save-model", default="auto", choices=["auto", "always", "never"])
    p.add_argument("--train-task", default=None, help="train on another task's data, e.g. anli_mix")
    p.add_argument("--extra-eval-tasks", default="", help="also evaluate on these tasks, e.g. anli_r2,anli_r3")
    p.add_argument("--max-train-examples", type=int, default=None, help="smoke tests only")
    p.add_argument("--max-eval-examples", type=int, default=None, help="smoke tests only")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("evaluate-model", help="evaluate a saved run or a ready-made classifier on corrupted sets")
    p.add_argument("--model", required=True, help="finetune run directory, or alias / hub id of a classifier")
    p.add_argument("--tasks", required=True, help="comma separated, e.g. anli_r1,anli_r2,anli_r3")
    p.add_argument("--corruptions", default="standard")
    p.add_argument("--style", default="2022", choices=["2021", "2022"])
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--out-dir", default=None, help="default: the run directory, or runs/evaluate/<model>/<style>")
    p.set_defaults(func=cmd_evaluate_model)
