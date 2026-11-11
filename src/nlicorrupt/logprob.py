"""Zero- and few-shot evaluation of local LMs by answer log-probability.

Output: $NLICORRUPT_DATA/runs/logprob/<model>/<task>/<style>/k<shots>-<exemplars>/
  <corruption>.jsonl.gz   per item: id, label, scores (one per answer), pred
  metrics.json            {corruption: metrics}, updated after each corruption
  manifest.json
Each corruption file is written once; a rerun skips finished corruptions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .config import model_id
from .data import read_jsonl, write_jsonl
from .metrics import compute
from .originals import get_records
from .prompts import select_exemplars
from .runinfo import manifest, write_json
from .tasks import get_task

log = logging.getLogger(__name__)


@dataclass
class LogprobConfig:
    model: str
    task: str
    corruptions: list[str] = field(default_factory=lambda: ["clean"])
    style: str = "2022"
    shots: int = 0
    exemplars: str = "clean"  # corruption applied to the k-shot exemplars
    seed: int = 0
    fmt: str = "auto"
    batch_size: int = 8
    max_examples: int | None = None  # smoke tests only (first n items)
    eval_items: int | None = None  # fixed label-stratified subsample of the evaluation set
    split: str | None = None  # default: the task's evaluation split

    def run_dir(self) -> Path:
        name = self.model.replace("/", "__")
        k = f"k{self.shots}-{self.exemplars}" if self.shots else "k0"
        if self.eval_items:
            k += f"-n{self.eval_items}"
        return paths.runs_dir() / "logprob" / name / self.task / self.style / k


def predict_from_scores(task: str, scores: list[float]):
    if get_task(task).is_regression:
        m = max(scores)
        ps = [math.exp(s - m) for s in scores]
        z = sum(ps)
        return sum(i * p / z for i, p in enumerate(ps))  # expected score 0-5
    return max(range(len(scores)), key=lambda i: scores[i])


def run(cfg: LogprobConfig, lm=None) -> Path:
    from .lm import build_prompt, load_lm, score_prompts

    spec = get_task(cfg.task)
    split = cfg.split or spec.eval_split
    out = cfg.run_dir()
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfg.corruptions if not (out / f"{c}.jsonl.gz").exists()]
    if not todo:
        log.info("all corruptions done: %s", out)
        return out

    lm = lm or load_lm(model_id(cfg.model), fmt=cfg.fmt)
    exemplars = []
    if cfg.shots:
        ex_task = spec.exemplar_task or cfg.task
        pool = get_records(ex_task, get_task(ex_task).train_split, cfg.exemplars, cfg.style)
        exemplars = select_exemplars(cfg.task, pool, cfg.shots, cfg.seed)

    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    mode = None
    keep = None
    if cfg.eval_items:
        from .probe import subsample_ids

        keep = subsample_ids(cfg.task, split, cfg.eval_items, 0, cfg.style)
    for corr in todo:
        recs = get_records(cfg.task, split, corr, cfg.style)
        if keep is not None:
            recs = [r for r in recs if r["id"] in keep]
        if cfg.max_examples:
            recs = recs[: cfg.max_examples]
        prompts = [build_prompt(lm, cfg.task, r, exemplars) for r in recs]
        scores, mode = score_prompts(lm, cfg.task, prompts, batch_size=cfg.batch_size)
        preds = [predict_from_scores(cfg.task, s) for s in scores]
        write_jsonl(
            out / f"{corr}.jsonl.gz",
            ({"id": r["id"], "label": r["label"], "scores": s, "pred": p} for r, s, p in zip(recs, scores, preds)),
        )
        metrics[corr] = compute(spec, [r["label"] for r in recs], preds)
        write_json(metrics_path, metrics)
        log.info("%s %s %s: %s", cfg.model, cfg.task, corr, metrics[corr])

    write_json(
        out / "manifest.json",
        manifest(asdict(cfg), {"hub_id": lm.hub_id, "format": lm.fmt, "scoring_mode": mode,
                               "example_prompt": prompts[0] if prompts else None,
                               "exemplar_ids": [e["id"] for e in exemplars]}),
    )
    return out


def load_predictions(run_dir: Path, corruption: str) -> list[dict]:
    return list(read_jsonl(Path(run_dir) / f"{corruption}.jsonl.gz"))
