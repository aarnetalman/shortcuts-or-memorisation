"""Human validation of the fresh items.

Validation sheets (one CSV per annotator, opened in any spreadsheet program): the same
items in a different random order for each annotator, with the intended label hidden, so
the labels are given blind. Columns to fill in:
  label    e, n or c (entailment, neutral, contradiction); leave empty to skip
  problem  anything that makes the item unusable (ungrammatical, needs outside knowledge,
           ambiguous, offensive); any text here excludes the item
  comment  free text

Hand-written items: a writing template with one premise per row and empty entailment,
neutral and contradiction columns. Filled templates are added to the candidates, so every
hand-written item is validated by the other annotators like the generated ones.

An item is kept if every annotator gave the intended label and nobody flagged a problem.
Naturally occurring pairs (natural.py) have no intended label: they are kept if all
annotators give the same label, which becomes the item's label.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from datetime import date
from pathlib import Path

from ..tasks import NLI_LABELS

SHEET_COLUMNS = ["item", "premise", "hypothesis", "label", "problem", "comment"]
TEMPLATE_COLUMNS = ["pid", "premise", "entailment", "neutral", "contradiction", "author"]
_SHORT = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def normalise_label(x: str | None) -> str | None:
    x = (x or "").strip().lower()
    if not x:
        return None
    if x in NLI_LABELS:
        return x
    return _SHORT.get(x[0])


def sample_premises(items: list[dict], n_premises: int, seed: int = 0) -> list[dict]:
    """Keep all items of n_premises generated premises (every style and label of each, so
    the style comparison is paired), balanced across generators and source texts.
    Hand-written and natural items are always kept."""
    gen = [it for it in items if it.get("origin") == "generated"]
    rest = [it for it in items if it.get("origin") != "generated"]
    strata: dict[tuple, list[str]] = {}
    for it in gen:
        key = (it.get("generator"), it.get("source_text"))
        if it["pid"] not in strata.setdefault(key, []):
            strata[key].append(it["pid"])
    rng = random.Random(f"sample|{seed}")
    for pids in strata.values():
        rng.shuffle(pids)
    picked, queues = [], [list(v) for _, v in sorted(strata.items(), key=lambda kv: str(kv[0]))]
    while len(picked) < n_premises and any(queues):
        for q in queues:  # round-robin over the strata
            if q and len(picked) < n_premises:
                picked.append(q.pop())
    keep = set(picked)
    return [it for it in gen if it["pid"] in keep] + rest


def write_sheets(items: list[dict], annotators: list[str], out_dir: Path, seed: int = 0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in annotators:
        rows = list(items)
        random.Random(f"sheet|{a}|{seed}").shuffle(rows)
        p = out_dir / f"validate_{a}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(SHEET_COLUMNS)
            for it in rows:
                w.writerow([it["id"], it["premise"], it["hypothesis"], "", "", ""])
        paths.append(p)
    return paths


def write_template(premises: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEMPLATE_COLUMNS)
        for p in premises:
            w.writerow([p["pid"], p["premise"], "", "", "", ""])
    return path


def read_template(path: Path, premises: dict[str, dict]) -> list[dict]:
    """Filled writing template -> candidate items (rows with all three hypotheses)."""
    items = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hyps = {lab: " ".join((row.get(lab) or "").split()) for lab in NLI_LABELS}
            if not all(hyps.values()):
                continue
            from .generate import opaque_id

            p = premises.get(row["pid"], {})
            author = (row.get("author") or "").strip() or None
            for lab in NLI_LABELS:
                items.append({"id": opaque_id("h", row["pid"], lab, author), "premise": row["premise"], "hypothesis": hyps[lab],
                              "label": lab, "origin": "human", "style": "human", "generator": None,
                              "author": author, "pid": row["pid"],
                              "source_text": p.get("source"), "url": p.get("url"), "date": p.get("date")})
    return items


def read_sheet(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["item"]: {"label": normalise_label(row.get("label")),
                              "problem": (row.get("problem") or "").strip(),
                              "comment": (row.get("comment") or "").strip()}
                for row in csv.DictReader(f)}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, labels=list(NLI_LABELS)))


def read_labels_file(path: Path) -> tuple[dict[str, dict[str, dict]], dict]:
    """The annotation tool's export (one row per label: item, rater_id, label, problem,
    seconds, ...) as annotator -> item -> {label, problem}, plus summary rater statistics.
    Personal details (name, age, language) are only summarised, never copied."""
    import statistics

    sheets: dict[str, dict[str, dict]] = {}
    secs: list[float] = []
    native: Counter = Counter()
    levels: Counter = Counter()
    problem_types: Counter = Counter()
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            r = (row.get("rater_id") or "").strip()
            if not r or not row.get("item"):
                continue
            if r not in sheets:
                lang = " ".join((row.get("native_lang") or "").lower().split())
                level = (row.get("english_level") or "").strip().lower()
                native["english" in lang or level == "native speaker"] += 1
                levels[level or "unknown"] += 1
            for part in (row.get("problem") or "").split("; "):
                if part.strip() in ("ambiguous", "ungrammatical", "needs outside knowledge"):
                    problem_types[part.strip()] += 1
                elif part.strip():
                    problem_types["free text"] += 1
            sheets.setdefault(r, {})[row["item"]] = {"label": normalise_label(row.get("label")),
                                                     "problem": (row.get("problem") or "").strip()}
            try:
                secs.append(float(row.get("seconds") or ""))
            except ValueError:
                pass
    per = sorted(len(v) for v in sheets.values())
    stats = {"raters": len(sheets), "labels": sum(per), "items_per_rater_median": statistics.median(per) if per else 0,
             "items_per_rater_range": [per[0], per[-1]] if per else [], "native_english_raters": native[True],
             "seconds_per_item_median": round(statistics.median(secs), 1) if secs else None,
             "english_levels": dict(levels), "problem_types": dict(problem_types)}
    return sheets, stats


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Nominal Krippendorff's alpha over items labelled by different subsets of raters."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    o: Counter = Counter()
    for u in units:
        m = len(u)
        c = Counter(u)
        for a in c:
            for b in c:
                o[(a, b)] += c[a] * (c[b] - (a == b)) / (m - 1)
    n_c: Counter = Counter()
    for (a, _), v in o.items():
        n_c[a] += v
    n = sum(n_c.values())
    d_o = sum(v for (a, b), v in o.items() if a != b) / n
    d_e = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n * (n - 1))
    return None if d_e == 0 else round(1 - d_o / d_e, 3)


def merge(candidates: list[dict], sheets: dict[str, dict[str, dict]], min_labels: int = 2) -> tuple[list[dict], dict]:
    """Kept items in the fresh-set format, and an agreement report.

    sheets: annotator -> item id -> {label, problem, comment}. Annotators may label different
    subsets of items: an item is judged on the labels it received, and needs at least
    min_labels of them (fewer only if fewer eligible annotators exist). Hand-written items
    are not validated by their own author.
    """
    from .natural import trivial_revision

    annotators = sorted(sheets)
    kept, report_rows = [], []
    trivial = [it for it in candidates if it.get("style") == "natural-revision"
               and trivial_revision(it["premise"], it["hypothesis"])]
    drop = {it["id"] for it in trivial}
    candidates = [it for it in candidates if it["id"] not in drop]
    for it in candidates:
        eligible = [a for a in annotators if a != it.get("author")]
        votes = {a: sheets[a][it["id"]] for a in eligible if it["id"] in sheets[a]}
        labels = {a: v.get("label") for a, v in votes.items() if v.get("label")}
        problems = [v["problem"] for v in votes.values() if v.get("problem")]
        complete = len(labels) >= max(1, min(min_labels, len(eligible)))
        if it.get("label"):
            final = it["label"] if all(lab == it["label"] for lab in labels.values()) else None
        else:  # natural pair: the annotators' unanimous label
            final = next(iter(labels.values())) if len(set(labels.values())) == 1 else None
        ok = complete and not problems and final is not None
        report_rows.append({"item": it, "labels": labels, "complete": complete, "problem": bool(problems), "ok": ok,
                            "final": final})
        if ok:
            kept.append({"id": it["id"], "premise": it["premise"], "hypothesis": it["hypothesis"],
                         "label": final, "source": "fresh", "twin_of": None,
                         "author": it.get("author") or it.get("generator"), "written": date.today().isoformat(),
                         "origin": it["origin"], "style": it["style"], "pid": it.get("pid"),
                         "url": it.get("url"), "validated_by": sorted(labels)})
    rep = agreement(report_rows, annotators)
    rep["trivial_revision_pairs_dropped"] = len(drop)
    return kept, rep


def agreement(rows: list[dict], annotators: list[str]) -> dict:
    done = [r for r in rows if r["complete"]]
    rep: dict = {"candidates": len(rows), "fully_labelled": len(done), "kept": sum(r["ok"] for r in rows),
                 "flagged": sum(r["problem"] for r in rows)}
    intended = [r for r in done if r["item"].get("label")]
    if intended:
        rep["agree_with_intended"] = round(100.0 * sum(
            all(lab == r["item"]["label"] for lab in r["labels"].values()) for r in intended) / len(intended), 1)
    natural = [r for r in done if not r["item"].get("label")]
    if natural:
        rep["natural_unanimous"] = round(100.0 * sum(len(set(r["labels"].values())) == 1 for r in natural)
                                         / len(natural), 1)
    rep["krippendorff_alpha"] = krippendorff_alpha([list(r["labels"].values()) for r in rows])
    rep["labels_per_item"] = dict(Counter(len(r["labels"]) for r in rows))
    if len(annotators) == 2:
        a, b = annotators[:2]
        pairs = [(r["labels"][a], r["labels"][b]) for r in done if a in r["labels"] and b in r["labels"]]
        if pairs:
            rep[f"kappa_{a}_{b}"] = round(cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]), 3)
    by_style = Counter(r["item"]["style"] for r in rows if r["ok"])
    rep["kept_by_style"] = dict(by_style)
    rep["kept_by_label"] = dict(Counter(r["final"] for r in rows if r["ok"]))
    rep["kept_by_style_and_label"] = {f"{st}/{lab}": n for (st, lab), n in sorted(
        Counter((r["item"]["style"], r["final"]) for r in rows if r["ok"]).items())}
    return rep
