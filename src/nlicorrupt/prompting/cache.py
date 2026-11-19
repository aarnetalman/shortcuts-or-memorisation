"""Response cache (SQLite) and cost log (JSONL)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .. import paths


def request_key(provider: str, body: dict) -> str:
    blob = json.dumps({"provider": provider, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or paths.data_root() / "api" / "cache.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Shared across worker threads; the runner serialises writes with a lock.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, provider TEXT, model TEXT, "
            "request TEXT, response TEXT, usage TEXT, cost_usd REAL, created REAL)"
        )
        self.db.commit()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def has(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone() is not None

    def put(self, key: str, provider: str, body: dict, response: dict, cost: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (key, provider, body.get("model"), json.dumps(body, ensure_ascii=False),
             json.dumps(response, ensure_ascii=False), json.dumps(response.get("usage")), cost, time.time()),
        )
        self.db.commit()


class CostLog:
    """Append-only log of every paid call: time, model, tokens, cost, run label."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or paths.data_root() / "api" / "costs.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, **entry) -> None:
        entry.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def total(self, run: str | None = None) -> float:
        if not self.path.exists():
            return 0.0
        s = 0.0
        for line in open(self.path, encoding="utf-8"):
            e = json.loads(line)
            if run is None or e.get("run") == run:
                s += float(e.get("cost_usd", 0.0))
        return s
