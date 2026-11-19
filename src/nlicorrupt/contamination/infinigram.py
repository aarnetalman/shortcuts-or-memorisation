"""Exact n-gram counts of evaluation items in a pre-training corpus via the infini-gram API.

Each text field is 'found' if one of its sampled 13-grams occurs in the corpus, or if the
whole field occurs and is at least `min_words_full_match` words long: a short field such
as "The man is asleep." occurs in any large corpus and says nothing about the benchmark.

Fields are reported separately because they mean different things. A found premise
usually reflects the benchmark's public source text (Wikipedia for ANLI R1/R2). A found
hypothesis is stronger evidence that the benchmark itself was in the data, since
hypotheses were written by annotators. `seen` (any field, any match) is kept for
compatibility. Responses are cached in SQLite, so reruns are free and an
interrupted run resumes. Needs internet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from .. import paths
from ..config import load_yaml
from ..data import write_jsonl
from ..originals import get_records
from ..tasks import get_task

log = logging.getLogger(__name__)


def ngram_queries(text: str, n: int, max_ngrams: int, full_max_words: int) -> dict:
    words = text.split()
    out = {"full": " ".join(words) if 0 < len(words) <= full_max_words else None, "ngrams": []}
    if len(words) >= n:
        starts = list(range(0, len(words) - n + 1))
        if len(starts) > max_ngrams:
            step = (len(starts) - 1) / (max_ngrams - 1) if max_ngrams > 1 else 0
            starts = sorted({round(i * step) for i in range(max_ngrams)})
        out["ngrams"] = [" ".join(words[s : s + n]) for s in starts]
    return out


def field_found(f: dict, n: int = 13, min_words: int = 13) -> bool:
    """Whether one field's counts show it in the corpus (see module docstring).

    Older output files have no word count; a field with 13-grams has at least n words.
    """
    if any(c > 0 for c in f.get("ngrams", [])):
        return True
    words = f.get("words", n if f.get("ngrams") else 0)
    return (f.get("full") or 0) > 0 and words >= min_words


def item_flags(row: dict, n: int = 13, min_words: int = 13) -> dict:
    """{field: found, ...} for one output row, plus 'any' (any field found)."""
    flags = {k: field_found(f, n, min_words) for k, f in row["fields"].items()}
    flags["any"] = any(flags.values())
    return flags


class InfiniGramRefused(RuntimeError):
    pass


class InfiniGram:
    def __init__(self, endpoint: str, index: str, sleep: float = 0.2, cache: Path | None = None, post=None):
        self.endpoint, self.index, self.sleep = endpoint, index, sleep
        path = cache or paths.data_root() / "contamination" / "infinigram_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (key TEXT PRIMARY KEY, idx TEXT, query TEXT, count INTEGER)")
        self._post = post or self._http_post

    def _http_post(self, payload: dict) -> dict:
        """POST with polite retries.

        429 (rate limited): wait as long as the server asks (Retry-After), at least 60 s.
        403 (forbidden): stop. The server has refused this client, usually after too many
        requests; hammering on keeps it blocked. Counts already received stay cached.
        """
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        delay = 5.0
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    raise InfiniGramRefused(
                        "infini-gram refused the request (HTTP 403). This usually means the client sent "
                        "too many requests and is blocked for a while. Wait (an hour or more) before "
                        "rerunning; counts received so far are cached. For large jobs, ask the "
                        "infini-gram maintainers for bulk access."
                    ) from e
                if attempt == 7:
                    raise
                wait = delay
                if e.code == 429:
                    try:
                        wait = max(60.0, float(e.headers.get("Retry-After", 0)))
                    except (TypeError, ValueError):
                        wait = 60.0
                log.warning("infini-gram HTTP %s; waiting %.0f s", e.code, wait)
                time.sleep(wait)
                delay = min(delay * 2, 300.0)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 7:
                    raise
                log.warning("infini-gram request failed (%s); retrying in %.0f s", e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 300.0)
        raise RuntimeError("unreachable")

    def count(self, query: str) -> int:
        key = hashlib.sha256(f"{self.index}|{query}".encode()).hexdigest()
        row = self.db.execute("SELECT count FROM counts WHERE key = ?", (key,)).fetchone()
        if row:
            return row[0]
        resp = self._post({"index": self.index, "query_type": "count", "query": query})
        if "error" in resp:
            raise RuntimeError(f"infini-gram error for {query[:60]!r}: {resp['error']}")
        c = int(resp["count"])
        self.db.execute("INSERT OR REPLACE INTO counts VALUES (?, ?, ?, ?)", (key, self.index, query, c))
        self.db.commit()
        if self.sleep:
            time.sleep(self.sleep)
        return c


def item_ids_for(task: str, subsample: dict, seed: int = 0) -> set[str] | None:
    if task not in subsample:
        return None
    from ..probe import stratified_subsample

    spec = get_task(task)
    recs = get_records(task, spec.eval_split, "clean")
    return {r["id"] for r in stratified_subsample(recs, subsample[task], seed, spec.is_regression)}


def run(tasks: list[str], index: str | None = None, client: InfiniGram | None = None) -> list[Path]:
    cfg = load_yaml("contamination.yaml")["infinigram"]
    index = index or cfg["index"]
    if not index:
        raise ValueError("set infinigram.index in configs/contamination.yaml or pass --index")
    client = client or InfiniGram(cfg["endpoint"], index, cfg["sleep_seconds"])
    outs = []
    for task in tasks:
        spec = get_task(task)
        keep = item_ids_for(task, cfg.get("subsample", {}))
        recs = [r for r in get_records(task, spec.eval_split, "clean") if keep is None or r["id"] in keep]
        rows = []
        for i, r in enumerate(recs):
            fields = {}
            for k in spec.keys:
                q = ngram_queries(r[k] or "", cfg["n"], cfg["max_ngrams_per_field"], cfg["full_max_words"])
                fields[k] = {
                    "full": client.count(q["full"]) if q["full"] else None,
                    "ngrams": [client.count(g) for g in q["ngrams"]],
                    "words": len((r[k] or "").split()),
                }
            seen = any((f["full"] or 0) > 0 or any(c > 0 for c in f["ngrams"]) for f in fields.values())
            row = {"id": r["id"], "label": r["label"], "fields": fields, "seen": seen}
            row["found"] = {k: v for k, v in item_flags(row, cfg["n"], cfg["min_words_full_match"]).items()
                            if k != "any"}
            rows.append(row)
            if i % 200 == 0:
                log.info("%s: %d/%d items", task, i, len(recs))
        out = paths.runs_dir() / "contamination" / "infinigram" / index / f"{task}.jsonl.gz"
        write_jsonl(out, rows)
        found = ", ".join(f"{k} {sum(r['found'][k] for r in rows)}" for k in spec.keys)
        log.info("%s: %d items in %s; fields found: %s", task, len(rows), index, found)
        outs.append(out)
    return outs
