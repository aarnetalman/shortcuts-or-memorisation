"""Naturally occurring sentence pairs: nobody wrote the hypothesis for NLI.

  headline  Global Voices: the first sentence of the article (premise) and its headline
            (hypothesis). Mostly entailment, some neutral. (Wikinews, the original source,
            closed in May 2026.)
  revision  Wikipedia: a sentence in the current version of a recently created article
            (premise) and the closest sentence in the article's first version
            (hypothesis), when an edit changed a number, date or name. Often contradiction
            or entailment, sometimes neutral.

These pairs have no intended label: annotators label them from scratch and an item is kept
when they agree (see sheets.merge).
"""

from __future__ import annotations

import difflib
import logging
import random
import re
import urllib.parse

from .premises import APIS, LICENCES, Wiki, candidate_sentences

log = logging.getLogger(__name__)


# --- wikitext to plain text (enough for article prose; tables and infoboxes are dropped) ---

_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG = re.compile(r"<[^>]+>")
_FILE = re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.I)
_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_EXT = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
_QUOTES = re.compile(r"'{2,}")


def _drop_templates(text: str) -> str:
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{{", i) or text.startswith("{|", i):
            depth += 1
            i += 2
        elif depth and (text.startswith("}}", i) or text.startswith("|}", i)):
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def wikitext_to_text(wt: str) -> str:
    import html

    t = _COMMENT.sub("", wt)
    t = _REF.sub("", t)
    t = _drop_templates(t)
    t = _FILE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _EXT.sub(r"\1", t)
    t = _TAG.sub("", t)
    t = _QUOTES.sub("", t)
    t = html.unescape(t).replace("\xa0", " ")
    lines = [ln for ln in t.split("\n") if ln.strip() and not ln.lstrip().startswith(("=", "*", "#", "|", "!", ":"))]
    return "\n".join(lines)


# --- pairing ---

_FACT = re.compile(r"\d|\b[A-Z][a-z]+")


def _norm_tokens(s: str) -> list[str]:
    import html

    s = html.unescape(s).replace("\xa0", " ").lower()
    return re.findall(r"[a-z0-9]+", s)


def trivial_revision(a: str, b: str) -> bool:
    """True if two sentences differ only in case, punctuation, spacing, HTML entities or
    word order: such pairs are trivial entailments and say nothing about inference."""
    return sorted(_norm_tokens(a)) == sorted(_norm_tokens(b))


def changed_fact(a: str, b: str) -> bool:
    """True if the token difference between a and b involves a number or a capitalised word."""
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=ta, b=tb)
    diff = [w for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal" for w in ta[i1:i2] + tb[j1:j2]]
    return bool(diff) and any(_FACT.search(w) for w in diff)


def revision_pairs_from(old_text: str, new_text: str, min_words: int, max_words: int,
                        lo: float = 0.6, hi: float = 0.97) -> list[tuple[str, str]]:
    """(new sentence, old sentence) pairs that are similar but not identical, with a changed fact."""
    old = candidate_sentences(old_text, min_words, max_words)
    new = candidate_sentences(new_text, min_words, max_words)
    out = []
    for n in new:
        best, score = None, 0.0
        for o in old:
            r = difflib.SequenceMatcher(a=o.split(), b=n.split()).ratio()
            if r > score:
                best, score = o, r
        if best is not None and lo <= score <= hi and changed_fact(best, n) and not trivial_revision(best, n):
            out.append((n, best))
    return out


GLOBAL_VOICES = "https://globalvoices.org/wp-json/wp/v2/posts"


def _html_text(h: str) -> str:
    import html

    t = html.unescape(re.sub(r"<[^>]+>", " ", h or ""))
    return " ".join(t.replace("[…]", " ").replace("[&hellip;]", " ").split())


def headline_pairs(wiki: Wiki, cfg: dict, until: str, limit: int, rng: random.Random) -> list[dict]:
    """Global Voices (CC BY 3.0): the headline (hypothesis) and the first sentence of the
    article's excerpt (premise), for posts published between the dates (WordPress API)."""
    posts, page = [], 1
    while len(posts) < limit * 4:
        try:
            r = wiki._get(GLOBAL_VOICES, {"after": cfg["since"] + "T00:00:00", "before": until + "T23:59:59",
                                          "per_page": 100, "page": page, "_fields": "title,excerpt,link,date"})
        except Exception as e:  # noqa: BLE001 - past the last page the API answers 400
            log.info("Global Voices: stopped at page %d (%s)", page, e)
            break
        if not isinstance(r, list) or not r:
            break
        posts += r
        if len(r) < 100:
            break
        page += 1
    out = []
    for p in rng.sample(posts, len(posts)):
        title = _html_text((p.get("title") or {}).get("rendered", ""))
        if len(title.split()) < 4:
            continue
        sents = candidate_sentences(_html_text((p.get("excerpt") or {}).get("rendered", "")),
                                    cfg["min_words"], cfg["max_words"])
        if not sents or sents[0].rstrip(".").lower() == title.lower():
            continue
        out.append({"premise": sents[0], "hypothesis": title, "kind": "headline", "title": title,
                    "date": (p.get("date") or "")[:10], "source_text": "globalvoices", "url": p.get("link"),
                    "licence": LICENCES["globalvoices"]})
        if len(out) >= limit:
            break
    return out


def wiki_revisions(wiki: Wiki, title: str) -> tuple[str, str] | None:
    """Wikitext of the first and the current revision of a page, if it has been edited."""
    def one(direction):
        r = wiki._get(APIS["wikipedia"], {"action": "query", "prop": "revisions", "titles": title,
                                          "rvprop": "ids|content", "rvslots": "main", "rvlimit": 1,
                                          "rvdir": direction})
        pages = r.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("revisions"):
            return None
        rev = pages[0]["revisions"][0]
        return rev["revid"], rev["slots"]["main"].get("content", "")

    first, last = one("newer"), one("older")
    if not first or not last or first[0] == last[0]:
        return None
    return first[1], last[1]


def revision_pairs(wiki: Wiki, cfg: dict, until: str, limit: int, rng: random.Random,
                   max_pages: int = 600) -> list[dict]:
    titles = wiki.wikipedia_titles(cfg["since"], until, max_pages)
    out = []
    for n, (title, created) in enumerate(rng.sample(titles, len(titles)), 1):
        if n % 50 == 0:
            log.info("revision pairs: %d pages checked, %d pairs", n, len(out))
        revs = wiki_revisions(wiki, title)
        if revs is None:
            continue
        pairs = revision_pairs_from(wikitext_to_text(revs[0]), wikitext_to_text(revs[1]),
                                    cfg["min_words"], cfg["max_words"])
        for prem, hyp in pairs[: cfg.get("per_article", 2)]:
            out.append({"premise": prem, "hypothesis": hyp, "kind": "revision", "title": title, "date": created,
                        "source_text": "wikipedia",
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                        "licence": LICENCES["wikipedia"]})
        if len(out) >= limit:
            break
    return out[:limit]


def build(cfg: dict, kinds: list[str], per_kind: int, wiki: Wiki | None = None, until: str | None = None,
          max_pages: int = 600) -> list[dict]:
    import datetime as dt

    from .generate import opaque_id

    until = until or cfg.get("until") or dt.date.today().isoformat()
    wiki = wiki or Wiki(cfg["user_agent"], cfg.get("sleep_seconds", 0.5))
    rng = random.Random(f"natural|{cfg.get('seed', 0)}")
    pairs = []
    if "headline" in kinds:
        pairs += headline_pairs(wiki, cfg, until, per_kind, rng)
    if "revision" in kinds:
        pairs += revision_pairs(wiki, cfg, until, per_kind, rng, max_pages=max_pages)
    items = []
    for p in pairs:
        items.append({"id": opaque_id("n", p["kind"], p["premise"], p["hypothesis"]), "premise": p["premise"],
                      "hypothesis": p["hypothesis"], "label": None, "origin": "natural",
                      "style": f"natural-{p['kind']}", "generator": None, "pid": None,
                      "source_text": p["source_text"], "url": p["url"], "date": p["date"]})
    log.info("natural pairs: %s", {k: sum(i["style"] == f"natural-{k}" for i in items) for k in kinds})
    return items
