"""Model-based contamination indicators for local LMs: TS-guessing and Min-K%++.

Both run on clean evaluation items of the benchmarks and on the fresh control
items, which no model can have seen; the fresh items give the reference level.
"""

from __future__ import annotations

import logging
import re
import string

import numpy as np

from .. import paths
from ..config import load_yaml, model_id
from ..data import write_jsonl
from ..originals import get_records
from ..runinfo import write_json
from ..tasks import get_task
from .infinigram import item_ids_for

log = logging.getLogger(__name__)

_STOP = set("""a an the and or but if then than that this these those there here of in on at to for from by with
about into over under after before between through during without within is are was were be been being have has
had do does did not no yes it its they them their he she his her we our you your i me my what which who whom whose
when where why how all any both each few more most other some such only own same so too very can will just should
could would may might must shall also""".split())

# v2: masked words never occur in the other fields of the item (see mask_target).
GUESS_DIR = "guess-v2"

DATASET_NAMES = {
    "cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "qqp": "QQP", "stsb": "STS-B", "mnli": "MultiNLI",
    "qnli": "QNLI", "rte": "RTE", "anli_r1": "ANLI", "anli_r2": "ANLI", "anli_r3": "ANLI", "fresh_nli": "MultiNLI",
}


def _stem(w: str) -> str:
    """Crude stem for overlap checks: lowercase, first five letters."""
    return w.lower()[:5]


def mask_target(text: str, context: str = "") -> tuple[str, str] | None:
    """Mask the longest content word (first if tied) that cannot simply be copied.

    A candidate is skipped if it occurs more than once in `text`, or if it or a word
    sharing its first five letters occurs in `context` (the other fields of the item,
    which the model sees). Otherwise a pair task would mostly test copying from the
    premise, not memorisation. Returns (masked text, word) or None.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    seen_ctx = {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z'-]+", context)}
    counts: dict[str, int] = {}
    for w in words:
        counts[w.lower()] = counts.get(w.lower(), 0) + 1
    cands = [w for w in words
             if w.lower() not in _STOP and len(w) >= 4
             and counts[w.lower()] == 1 and _stem(w) not in seen_ctx]
    if not cands:
        return None
    target = max(cands, key=len)
    return re.sub(rf"\b{re.escape(target)}\b", "[MASK]", text, count=1), target


def guess_prompt(task: str, record: dict) -> tuple[str, str] | None:
    spec = get_task(task)
    last = spec.keys[-1]
    context = " ".join(record[k] or "" for k in spec.keys[:-1])
    m = mask_target(record[last] or "", context)
    if m is None:
        return None
    masked, target = m
    ctx = "".join(f"{k.capitalize()}: {record[k]}\n" for k in spec.keys[:-1])
    prompt = (f"The following item is from the {DATASET_NAMES.get(task, task)} dataset. One word in the last "
              f"line has been replaced by [MASK]. Reply with the missing word only.\n{ctx}"
              f"{last.capitalize()}: {masked}")
    return prompt, target


def _norm(w: str) -> str:
    return w.strip().strip(string.punctuation + "\"'“”‘’").lower()


def _records(task: str, subsample: dict) -> list[dict]:
    spec = get_task(task)
    keep = item_ids_for(task, subsample)
    return [r for r in get_records(task, spec.eval_split, "clean") if keep is None or r["id"] in keep]


def run_guess(model: str, tasks: list[str], batch_size: int = 8, fmt: str = "auto", lm=None) -> dict:
    from ..lm import generate, load_lm, wrap_prompt

    cfg = load_yaml("contamination.yaml")["guess"]
    lm = lm or load_lm(model_id(model), fmt=fmt)
    results = {}
    for task in tasks:
        items = []
        for r in _records(task, cfg.get("subsample", {})):
            gp = guess_prompt(task, r)
            if gp:
                items.append((r, *gp))
        outs = generate(lm, [wrap_prompt(lm, p) for _, p, _ in items], cfg["max_new_tokens"], batch_size)
        rows = []
        for (r, p, target), o in zip(items, outs):
            first = o.split()[0] if o.split() else ""
            rows.append({"id": r["id"], "target": target, "output": o, "match": _norm(first) == _norm(target)})
        out = paths.runs_dir() / "contamination" / GUESS_DIR / model.replace("/", "__") / f"{task}.jsonl.gz"
        write_jsonl(out, rows)
        results[task] = {"exact_match": 100.0 * np.mean([x["match"] for x in rows]) if rows else None, "n": len(rows)}
        log.info("guess %s %s: %s", model, task, results[task])
    write_json(paths.runs_dir() / "contamination" / GUESS_DIR / model.replace("/", "__") / "metrics.json", results)
    return results


def min_k_plus_plus(token_logp: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: float) -> float:
    z = (token_logp - mu) / np.maximum(sigma, 1e-6)
    n = max(1, int(round(k * len(z))))
    return float(np.mean(np.sort(z)[:n]))


def run_mink(model: str, tasks: list[str], batch_size: int = 4, lm=None) -> dict:
    from ..lm import token_stats, load_lm

    cfg = load_yaml("contamination.yaml")["mink"]
    lm = lm or load_lm(model_id(model), fmt="text")
    results = {}
    for task in tasks:
        spec = get_task(task)
        recs = _records(task, cfg.get("subsample", {}))
        texts = [" ".join(r[k] or "" for k in spec.keys) for r in recs]
        stats = token_stats(lm, texts, batch_size)
        rows = [{"id": r["id"], "mink_pp": min_k_plus_plus(lp, mu, sd, cfg["k"])}
                for r, (lp, mu, sd) in zip(recs, stats)]
        out = paths.runs_dir() / "contamination" / "mink" / model.replace("/", "__") / f"{task}.jsonl.gz"
        write_jsonl(out, rows)
        results[task] = {"mean": float(np.mean([x["mink_pp"] for x in rows])), "n": len(rows)}
        log.info("min-k%%++ %s %s: %s", model, task, results[task])
    write_json(paths.runs_dir() / "contamination" / "mink" / model.replace("/", "__") / "metrics.json", results)
    return results
