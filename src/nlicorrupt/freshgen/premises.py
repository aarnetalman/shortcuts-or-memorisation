"""Premises from text published after a cut-off date (needs internet).

Sources (both openly licensed):
  wikinews   articles published on English Wikinews (CC BY 2.5), by publication date
  wikipedia  articles created on English Wikipedia (CC BY-SA 4.0), by creation date

A new Wikipedia page can reuse older text (splits, translations), so the date alone does
not guarantee fresh text; with infinigram_check, premises with any 13-gram found in the
pre-training corpus index are dropped as well.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import load_yaml

log = logging.getLogger(__name__)

APIS = {"wikinews": "https://en.wikinews.org/w/api.php", "wikipedia": "https://en.wikipedia.org/w/api.php",
        "currentevents": "https://en.wikipedia.org/w/api.php"}
LICENCES = {"wikinews": "CC BY 2.5", "wikipedia": "CC BY-SA 4.0", "currentevents": "CC BY-SA 4.0",
            "globalvoices": "CC BY 3.0"}
# Wikinews was closed (read-only since May 2026); news text now comes from Wikipedia's
# daily Portal:Current events pages (premises) and Global Voices (headline pairs).

# Wikinews articles start with a dateline such as "Monday, October 5, 2026" and end with
# boilerplate; neither makes a premise.
_DATELINE = re.compile(r"^\s*(?:(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s+)?[A-Z][a-z]+ \d{1,2}, \d{4}\s*")
_BAD = re.compile(r"https?://|www\.|\bmay refer to\b|\bhave an opinion\b|\bsources?\b\s*$|[{}<>|=]|\.\.\.", re.I)


class Wiki:
    """MediaWiki API client. Every response is cached in SQLite, so an interrupted
    fresh-premises or fresh-natural run resumes without fetching anything twice.
    cache=False turns the cache off (tests)."""

    def __init__(self, user_agent: str, sleep: float = 0.5, get=None, cache: Path | bool | None = None):
        if "TO FILL" in user_agent:
            raise ValueError("set premises.user_agent in configs/fresh.yaml to include a contact first")
        self.ua, self.sleep = user_agent, sleep
        self._raw_get = get or self._http_get
        self.db = None
        if cache is not False:
            from .. import paths

            path = Path(cache) if cache else paths.data_root() / "fresh" / "wiki_cache.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path)
            self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT)")
        self.requests = self.cached = 0

    def _get(self, api: str, params: dict) -> dict:
        key = hashlib.sha256(json.dumps([api, params], sort_keys=True).encode()).hexdigest()
        if self.db is not None:
            row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if row:
                self.cached += 1
                return json.loads(row[0])
        out = self._raw_get(api, params)
        self.requests += 1
        if self.db is not None:
            self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(out)))
            self.db.commit()
        if (self.requests + self.cached) % 100 == 0:
            log.info("wiki: %d requests sent, %d from cache", self.requests, self.cached)
        return out

    def _http_get(self, api: str, params: dict) -> dict:
        q = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
        req = urllib.request.Request(f"{api}?{q}", headers={"User-Agent": self.ua})
        delay = 5.0
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read().decode())
                time.sleep(self.sleep)
                return out
            except Exception as e:  # noqa: BLE001 - network errors of all kinds: retry
                if attempt == 4:
                    raise
                log.warning("wiki request failed (%s); retrying in %.0f s", e, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def wikinews_titles(self, since: str, until: str, limit: int) -> list[tuple[str, str]]:
        """(title, date) of articles added to Category:Published between the dates, newest first."""
        out, cont = [], {}
        while len(out) < limit:
            r = self._get(APIS["wikinews"], {"action": "query", "list": "categorymembers",
                                             "cmtitle": "Category:Published", "cmnamespace": 0,
                                             "cmsort": "timestamp", "cmdir": "desc", "cmprop": "title|timestamp",
                                             "cmstart": until + "T23:59:59Z", "cmend": since + "T00:00:00Z",
                                             "cmlimit": 500, **cont})
            out += [(m["title"], m["timestamp"][:10]) for m in r["query"]["categorymembers"]]
            if "continue" not in r:
                break
            cont = r["continue"]
        return out[:limit]

    def wikipedia_titles(self, since: str, until: str, limit: int) -> list[tuple[str, str]]:
        """(title, date) of main-namespace pages created between the dates, newest first."""
        out, cont = [], {}
        while len(out) < limit:
            r = self._get(APIS["wikipedia"], {"action": "query", "list": "logevents", "letype": "create",
                                              "lenamespace": 0, "ledir": "older", "leprop": "title|timestamp",
                                              "lestart": until + "T23:59:59Z", "leend": since + "T00:00:00Z",
                                              "lelimit": 500, **cont})
            out += [(e["title"], e["timestamp"][:10]) for e in r["query"]["logevents"]]
            if "continue" not in r:
                break
            cont = r["continue"]
        return out[:limit]

    def texts(self, source: str, titles: list[str]) -> dict[str, str]:
        """Plain text of existing, non-redirect pages: the full article for Wikinews, the
        lead section for Wikipedia (the extracts API returns several only for leads)."""
        out = {}
        batch = 1 if source == "wikinews" else 20
        for i in range(0, len(titles), batch):
            params = {"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
                      "titles": "|".join(titles[i:i + batch])}
            if source == "wikipedia":
                params.update(exintro=1, exlimit=batch)
            r = self._get(APIS[source], params)
            for p in r.get("query", {}).get("pages", []):
                if p.get("missing") or p.get("redirect") or not p.get("extract"):
                    continue
                out[p["title"]] = p["extract"]
        return out


_SOURCE_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def current_events_text(wikitext: str) -> str:
    """The event sentences of a Portal:Current events day page: its bullet items, without
    markup and without the trailing news-source links."""
    from .natural import wikitext_to_text

    lines = []
    for ln in wikitext.split("\n"):
        if not ln.lstrip().startswith("*"):
            continue
        t = wikitext_to_text(ln.lstrip("* ").strip())
        while _SOURCE_PAREN.search(t):
            t = _SOURCE_PAREN.sub("", t)
        if t:
            lines.append(t.strip())
    return "\n".join(lines)


def current_events_pages(wiki: "Wiki", since: str, until: str) -> list[tuple[str, str, str]]:
    """(title, date, text) of the daily Portal:Current events pages between the dates."""
    d0, d1 = dt.date.fromisoformat(since), dt.date.fromisoformat(until)
    days = [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    titles = {f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}": d.isoformat() for d in days}
    out, names = [], list(titles)
    for i in range(0, len(names), 50):
        r = wiki._get(APIS["currentevents"], {"action": "query", "prop": "revisions", "rvprop": "content",
                                              "rvslots": "main", "titles": "|".join(names[i:i + 50])})
        for p in r.get("query", {}).get("pages", []):
            if p.get("missing") or not p.get("revisions"):
                continue
            text = current_events_text(p["revisions"][0]["slots"]["main"].get("content", ""))
            if text:
                out.append((p["title"], titles.get(p["title"]), text))
    return out


def candidate_sentences(text: str, min_words: int, max_words: int) -> list[str]:
    from nltk.tokenize import sent_tokenize

    from ..corrupt.tagging import ensure_nltk_data

    ensure_nltk_data()
    text = _DATELINE.sub("", text, count=1)
    out = []
    for para in text.split("\n"):
        for s in sent_tokenize(para.strip()):
            s = " ".join(s.split())
            n = len(s.split())
            if not (min_words <= n <= max_words) or _BAD.search(s):
                continue
            if not s[0].isupper() or s[-1] not in ".!?" or s.count('"') % 2 or s.count("(") != s.count(")"):
                continue
            if sum(c.isdigit() for c in s) > len(s) * 0.15:
                continue
            out.append(s)
    return out


def fetch(cfg: dict | None = None, wiki: Wiki | None = None, infinigram=None) -> list[dict]:
    cfg = cfg or load_yaml("fresh.yaml")["premises"]
    until = cfg.get("until") or dt.date.today().isoformat()
    wiki = wiki or Wiki(cfg["user_agent"], cfg.get("sleep_seconds", 0.5))
    rng = random.Random(cfg.get("seed", 0))
    out, seen = [], set()
    for source in cfg["sources"]:
        if source == "currentevents":
            pages = current_events_pages(wiki, cfg["since"], until)
            dates = {t: d for t, d, _ in pages}
            texts = {t: x for t, _, x in pages}
        else:
            lister = wiki.wikinews_titles if source == "wikinews" else wiki.wikipedia_titles
            # Over-fetch titles: many new pages are stubs, lists or yield no usable sentence.
            titles = lister(cfg["since"], until, cfg["per_source"] * 8)
            dates = dict(titles)
            texts = wiki.texts(source, [t for t, _ in titles])
        kept = 0
        for title in rng.sample(sorted(texts), len(texts)):
            sents = candidate_sentences(texts[title], cfg["min_words"], cfg["max_words"])
            rng.shuffle(sents)
            for s in sents[: cfg["per_article"]]:
                if s.lower() in seen:
                    continue
                if infinigram is not None and _found(infinigram, s):
                    log.info("dropped (found in corpus): %s", s[:80])
                    continue
                seen.add(s.lower())
                out.append({"pid": f"{source[:2]}{len(out):04d}", "premise": s, "source": source,
                            "title": title, "date": dates.get(title),
                            "url": f"https://en.{'wikipedia' if source == 'currentevents' else source}.org/wiki/"
                                   f"{urllib.parse.quote(title.replace(' ', '_'))}",
                            "licence": LICENCES[source]})
                kept += 1
            if kept >= cfg["per_source"]:
                break
        log.info("%s: %d premises from %d articles", source, kept, len(texts))
    return out


def _found(client, sentence: str) -> bool:
    from ..contamination.infinigram import ngram_queries

    q = ngram_queries(sentence, 13, 2, 0)
    return any(client.count(g) > 0 for g in q["ngrams"])


def write(premises: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in premises:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path
