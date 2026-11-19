"""Fine-tune an encoder on (possibly corrupted) training data and evaluate it on
clean and corrupted evaluation sets in one run.

One training run covers three configurations of Talman et al. (2022): with a
clean training set, the corrupted evaluations give Corrupt-Test; with a
corrupted training set, the clean evaluation gives Corrupt-Train and the
corrupted evaluations give Corrupt-Train-and-Test.

Output: $NLICORRUPT_DATA/runs/finetune/<model>/<task>/<style>/train-<corruption>/seed<seed>/
  metrics.json            {eval_set: {metric: value}}
  predictions/<set>.jsonl.gz   id, label, prediction
  manifest.json           arguments, code version, packages, hardware
  model/                  final weights (see --save-model)
  DONE                    written last; a run with DONE is skipped
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .config import model_id, split_revision
from .data import write_jsonl
from .metrics import compute
from .originals import get_records
from .runinfo import manifest, write_json
from .tasks import get_task

log = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    model: str
    task: str
    train_corruption: str = "clean"
    eval_corruptions: list[str] = field(default_factory=lambda: ["clean"])
    style: str = "2022"
    seed: int = 42
    epochs: float = 3.0
    batch_size: int = 32
    lr: float = 2e-5
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_len: int = 128
    eval_batch_size: int = 256
    bf16: bool = True
    save_model: str = "auto"  # auto: only when trained on clean data; always; never
    train_task: str | None = None  # e.g. anli_mix for the ANLI experiments
    extra_eval_tasks: list[str] = field(default_factory=list)  # e.g. anli_r2, anli_r3
    max_train_examples: int | None = None  # for smoke tests only
    max_eval_examples: int | None = None
    out_dir: str | None = None

    def run_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        name = self.model.replace("/", "__")
        task = self.task if not self.train_task else f"{self.task}@{self.train_task}"
        return paths.runs_dir() / "finetune" / name / task / self.style / f"train-{self.train_corruption}" / f"seed{self.seed}"


def eval_sets(cfg: FinetuneConfig) -> list[tuple[str, str, str, str]]:
    """(label, task, split, corruption) for every evaluation set of the run.

    Labels look like 'mnli/validation_matched/drop-NOUN'.
    """
    sets = []
    for task in [cfg.task, *cfg.extra_eval_tasks]:
        spec = get_task(task)
        sets += [(f"{task}/{spec.eval_split}/{c}", task, spec.eval_split, c) for c in cfg.eval_corruptions]
        for s in spec.extra_eval_splits:
            sets.append((f"{task}/{s}/clean", task, s, "clean"))
    return sets


def _encode(tokenizer, records, keys, max_len, is_regression):
    from datasets import Dataset

    texts = [[r[k] or "" for r in records] for k in keys]
    enc = tokenizer(*texts, truncation=True, max_length=max_len)
    enc["labels"] = [float(r["label"]) if is_regression else int(r["label"]) for r in records]
    return Dataset.from_dict(dict(enc))


def run(cfg: FinetuneConfig) -> Path:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    out = cfg.run_dir()
    if (out / "DONE").exists():
        log.info("already done: %s", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    spec = get_task(cfg.task)
    hub_id = model_id(cfg.model)
    set_seed(cfg.seed)

    train_task = cfg.train_task or cfg.task
    train_split = "train" if train_task == "anli_mix" else get_task(train_task).train_split
    train = get_records(train_task, train_split, cfg.train_corruption, cfg.style)
    if cfg.max_train_examples:
        train = train[: cfg.max_train_examples]
    log.info("train %s/%s %s (%s): %d examples", train_task, train_split, cfg.train_corruption, cfg.style, len(train))

    repo, rev = split_revision(hub_id)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo,
        revision=rev,
        num_labels=spec.num_labels,
        problem_type="regression" if spec.is_regression else "single_label_classification",
        ignore_mismatched_sizes=True,  # a fresh head when the checkpoint has one of another size
    )
    train_ds = _encode(tok, train, spec.keys, cfg.max_len, spec.is_regression)

    steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
    warmup = int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs)
    use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scratch = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / f"nlicorrupt-{os.getpid()}"
    targs = TrainingArguments(
        output_dir=str(scratch),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup,
        lr_scheduler_type="linear",
        seed=cfg.seed,
        data_seed=cfg.seed,
        bf16=use_bf16,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=min(8, os.cpu_count() or 2),
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=8),
        processing_class=tok,
    )
    train_result = trainer.train()

    metrics: dict[str, dict] = {}
    for label, etask, split, corr in eval_sets(cfg):
        espec = get_task(etask)
        recs = get_records(etask, split, corr, cfg.style)
        if cfg.max_eval_examples:
            recs = recs[: cfg.max_eval_examples]
        ds = _encode(tok, recs, espec.keys, cfg.max_len, espec.is_regression)
        logits = trainer.predict(ds).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.squeeze(-1).astype(float).tolist() if espec.is_regression else np.argmax(logits, -1).tolist()
        golds = [r["label"] for r in recs]
        metrics[label] = compute(espec, golds, preds)
        write_jsonl(
            out / "predictions" / f"{label.replace('/', '__')}.jsonl.gz",
            ({"id": r["id"], "label": r["label"], "pred": p} for r, p in zip(recs, preds)),
        )
        log.info("%s: %s", label, metrics[label])

    save = cfg.save_model == "always" or (cfg.save_model == "auto" and cfg.train_corruption == "clean")
    if save:
        trainer.save_model(str(out / "model"))
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "manifest.json",
        manifest(asdict(cfg), {"hub_id": hub_id, "train_examples": len(train), "train_metrics": train_result.metrics}),
    )
    (out / "DONE").write_text("ok\n")
    return out


def evaluate_model(
    model: str,
    out_dir: Path,
    tasks: list[str],
    corruptions: list[str],
    style: str = "2022",
    max_len: int = 128,
    batch_size: int = 256,
) -> dict:
    """Evaluate a classifier on clean/corrupted sets without training.

    `model` is a run directory with a saved model/ (from `finetune`), or a hub id or
    alias of a ready-made classifier (e.g. Nie et al.'s ANLI model). The model's
    labels must follow the task's label order, which holds for both.
    """
    import json

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    if (Path(model) / "model").exists():
        src, rev = str(Path(model) / "model"), None
    else:
        src, rev = split_revision(model_id(model))
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    clf = AutoModelForSequenceClassification.from_pretrained(src, revision=rev).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf.to(dev)
    collate = DataCollatorWithPadding(tok)
    out_dir = Path(out_dir)
    path = out_dir / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    for task in tasks:
        spec = get_task(task)
        for corr in corruptions:
            label = f"{task}/{spec.eval_split}/{corr}"
            if label in metrics:
                continue
            recs = get_records(task, spec.eval_split, corr, style)
            ds = _encode(tok, recs, spec.keys, max_len, spec.is_regression).remove_columns("labels")
            outs = []
            with torch.inference_mode():
                for i in range(0, len(ds), batch_size):
                    batch = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
                    outs.append(clf(**{k: v.to(dev) for k, v in batch.items()}).logits.float().cpu().numpy())
            logits = np.concatenate(outs)
            preds = logits.squeeze(-1).tolist() if spec.is_regression else np.argmax(logits, -1).tolist()
            metrics[label] = compute(spec, [r["label"] for r in recs], preds)
            write_jsonl(
                out_dir / "predictions" / f"{label.replace('/', '__')}.jsonl.gz",
                ({"id": r["id"], "label": r["label"], "pred": p} for r, p in zip(recs, preds)),
            )
            write_json(path, metrics)
            log.info("%s: %s", label, metrics[label])
    write_json(out_dir / "manifest_eval.json", manifest(
        {"model": str(model), "tasks": tasks, "corruptions": corruptions, "style": style, "max_len": max_len}))
    return metrics
