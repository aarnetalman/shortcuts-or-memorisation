"""Commands: download, prepare, corruptions."""

from __future__ import annotations

from ..tasks import ANLI_TASKS, GLUE_TASKS, get_task


def _split_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def cmd_download(args):
    from ..config import model_group, model_id
    from ..download import download_all

    models: list[str] = []
    for g in _split_list(args.model_groups):
        models += list(model_group(g).values())
    models += [model_id(m) for m in _split_list(args.models)]
    download_all(models=models, originals=not args.no_originals)
    print("download complete")


def default_splits(task: str) -> list[str]:
    if task == "anli_mix":
        return ["train"]
    spec = get_task(task)
    if spec.exemplar_task:  # evaluation-only control sets
        return [spec.eval_split]
    return [spec.train_split, spec.eval_split, *spec.extra_eval_splits] + (
        [spec.dev_split] if spec.dev_split != spec.eval_split else []
    )


def train_corruptions() -> list[str]:
    """Corrupt-Train sets: the eight single-class drops plus the 2021 extra sets."""
    from ..corrupt import CORRUPTION_CLASSES, KEEP_SETS_2021

    return ["clean"] + [f"drop-{c}" for c in CORRUPTION_CLASSES] + ["drop-NOUN+PRON"] + KEEP_SETS_2021


def cmd_prepare(args):
    from ..corrupt import standard_eval_corruptions
    from ..prepare import materialize, tag_split, task_keys

    tasks: list[str] = []
    for t in _split_list(args.task):
        tasks += list(GLUE_TASKS + ANLI_TASKS) if t == "all" else [t]
    for task in tasks:
        splits = _split_list(args.split) if args.split else default_splits(task)
        pair = len(task_keys(task)) == 2
        for split in splits:
            tag_split(task, split, workers=args.workers)
            if args.tag_only:
                continue
            if args.corruptions:
                corrs = _split_list(args.corruptions)
            elif split.startswith("train"):
                corrs = train_corruptions()
            else:
                corrs = standard_eval_corruptions(pair)
            if not pair:
                corrs = [c for c in corrs if c not in ("hyp-only", "swap")]
            for style in _split_list(args.style):
                materialize(task, split, corrs, style=style, seed=args.seed, overwrite=args.overwrite)
    print("prepare complete")


def cmd_corruptions(args):
    from ..corrupt import standard_eval_corruptions

    for c in standard_eval_corruptions(pair=True):
        print(c)


def add_parser(sub):
    p = sub.add_parser("download", help="fetch datasets, NLTK data, original repos and models")
    p.add_argument("--model-groups", default="", help="groups from configs/models.yaml, e.g. encoders,open_models")
    p.add_argument("--models", default="", help="extra model aliases or hub ids, comma separated")
    p.add_argument("--no-originals", action="store_true", help="skip cloning the 2021/2022 data repos")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="tag splits and write corrupted versions")
    p.add_argument("--task", default="all", help="task name(s), comma separated, 'anli_mix', or 'all'")
    p.add_argument("--split", default="", help="split(s); default: train + eval (+ dev) splits")
    p.add_argument("--corruptions", default="", help="default: the standard set (nlicorrupt corruptions)")
    p.add_argument("--style", default="2022", help="2022, 2021 or 2021,2022")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None, help="default: all cores")
    p.add_argument("--tag-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("corruptions", help="list the standard corruption names")
    p.set_defaults(func=cmd_corruptions)
