"""Commands: logprob, extract, probe (local open models)."""

from __future__ import annotations

from .train import _list, expand_corruptions


def cmd_logprob(args):
    from ..logprob import LogprobConfig, run
    from ..tasks import get_task

    for task in _list(args.task):
        cfg = LogprobConfig(
            model=args.model,
            task=task,
            corruptions=expand_corruptions(args.corruptions, get_task(task).is_pair),
            style=args.style,
            shots=args.shots,
            exemplars=args.exemplars,
            seed=args.seed,
            fmt=args.format,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            eval_items=args.eval_items,
        )
        print(run(cfg))


def cmd_extract(args):
    from ..probe import extract
    from ..tasks import get_task

    for task in _list(args.task):
        spec = get_task(task)
        corrs = expand_corruptions(args.corruptions, spec.is_pair)
        train_corrs = expand_corruptions(args.train_corruptions, spec.is_pair)
        extract(args.model, task, spec.train_split, train_corrs, args.style, args.max_train,
                args.seed, args.batch_size, args.format)
        extract(args.model, task, spec.eval_split, corrs, args.style, args.max_eval,
                args.seed, args.batch_size, args.format, tag_n=True)


def cmd_probe(args):
    from ..probe import train_and_eval
    from ..tasks import get_task

    for task in _list(args.task):
        spec = get_task(task)
        for tc in expand_corruptions(args.train_corruptions, spec.is_pair):
            print(train_and_eval(args.model, task, tc, expand_corruptions(args.corruptions, spec.is_pair),
                                 args.style, args.C, args.seed, eval_n=args.max_eval))


def _common(p):
    p.add_argument("--model", required=True, help="alias from configs/models.yaml or hub id")
    p.add_argument("--task", required=True, help="task(s), comma separated")
    p.add_argument("--corruptions", default="standard", help="evaluation corruptions; 'standard' = full set")
    p.add_argument("--style", default="2022", choices=["2021", "2022"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", default="auto", choices=["auto", "text", "chat"],
                   help="prompt format; auto = chat template if the tokenizer has one")
    p.add_argument("--batch-size", type=int, default=8)


def add_parser(sub):
    p = sub.add_parser("logprob", help="score answers by log-probability with a local LM")
    _common(p)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--exemplars", default="clean", help="corruption applied to k-shot exemplars")
    p.add_argument("--max-examples", type=int, default=None, help="smoke tests only")
    p.add_argument("--eval-items", type=int, default=None,
                   help="fixed label-stratified subsample of the evaluation set (e.g. 2000 for QQP/MNLI)")
    p.set_defaults(func=cmd_logprob)

    p = sub.add_parser("extract", help="save last-token hidden states for probing")
    _common(p)
    p.add_argument("--train-corruptions", default="clean,drop-NOUN,drop-VERB")
    p.add_argument("--max-train", type=int, default=20000, help="stratified training subsample per task")
    p.add_argument("--max-eval", type=int, default=None,
                   help="fixed label-stratified evaluation subsample (recorded in the file names)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("probe", help="train linear probes on extracted hidden states")
    _common(p)
    p.add_argument("--train-corruptions", default="clean,drop-NOUN,drop-VERB")
    p.add_argument("--C", type=float, default=1.0, help="inverse regularisation strength")
    p.add_argument("--max-eval", type=int, default=None, help="must match the value used for extract")
    p.set_defaults(func=cmd_probe)
