"""Fresh control-set builder (offline: fake wiki API, fake generator)."""

import json

import pytest

from nlicorrupt.freshgen import generate as g
from nlicorrupt.freshgen import premises as pr
from nlicorrupt.freshgen import sheets as s
from nlicorrupt.prompting.cache import CostLog, ResponseCache

ARTICLE = ("Monday, October 5, 2026 The city council of Riverton approved a new budget for public libraries "
           "after a long debate on Tuesday evening.\n"
           "Have an opinion on this story? Share it!\n"
           "Officials said the extra funding would allow three branches to stay open on weekends next year.\n"
           "See www.example.org for details.")


def fake_get(api, params):
    if params.get("list") == "categorymembers":
        return {"query": {"categorymembers": [{"title": "Riverton budget", "timestamp": "2026-10-05T10:00:00Z"},
                                              {"title": "Old redirect", "timestamp": "2026-10-04T10:00:00Z"}]}}
    if params.get("list") == "logevents":
        return {"query": {"logevents": []}}
    pages = []
    for t in params["titles"].split("|"):
        if t == "Old redirect":
            pages.append({"title": t, "redirect": True, "extract": "Something else entirely."})
        else:
            pages.append({"title": t, "extract": ARTICLE})
    return {"query": {"pages": pages}}


def test_candidate_sentences_strip_dateline_and_boilerplate(needs_nltk):
    sents = pr.candidate_sentences(ARTICLE, 12, 50)
    assert sents[0].startswith("The city council of Riverton")
    assert not any("opinion" in x or "www." in x for x in sents)
    assert len(sents) == 2


def test_fetch_respects_redirects_per_article_cap_and_corpus_check(needs_nltk):
    cfg = {"since": "2026-09-01", "until": "2026-10-31", "sources": ["wikinews"], "per_source": 10,
           "per_article": 1, "min_words": 12, "max_words": 50, "seed": 0}
    wiki = pr.Wiki("test-agent (contact: test)", sleep=0, get=fake_get, cache=False)
    ps = pr.fetch(cfg, wiki=wiki)
    assert len(ps) == 1 and ps[0]["title"] == "Riverton budget" and ps[0]["licence"] == "CC BY 2.5"

    class Found:  # every 13-gram "found": the premise is dropped
        def count(self, q):
            return 1

    assert pr.fetch(cfg, wiki=wiki, infinigram=Found()) == []


def test_user_agent_must_be_filled():
    with pytest.raises(ValueError, match="user_agent"):
        pr.Wiki("nlicorrupt-fresh/0.1 (contact: TO FILL)")


def test_parse_generator_output():
    out = g.parse('<think>hmm</think>Here: {"entailment": "A council voted.", "neutral": "It rained.", '
                  '"contradiction": "The council rejected the budget."}')
    assert out == {"entailment": "A council voted.", "neutral": "It rained.",
                   "contradiction": "The council rejected the budget."}
    assert g.parse('{"entailment": "x", "neutral": ""}') is None
    assert g.parse("no json here") is None


def test_styles_differ_and_mnli_style_shows_examples():
    ex = [{"premise": "A man sleeps.", "hypothesis": "A person rests.", "label_name": "entailment"}]
    m = g.messages("P.", "mnli", ex)[0]["content"]
    c = g.messages("P.", "cuefree", ex)[0]["content"]
    assert "A person rests." in m and "A person rests." not in c and "negation" in c


class FakeClient:
    def __init__(self):
        self.calls = 0

    def chat(self, body):
        self.calls += 1
        text = json.dumps({"entailment": "A council voted.", "neutral": "The vote was close.",
                           "contradiction": "The council cut the library budget."})
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 300, "completion_tokens": 40}}


def test_generate_dry_run_then_confirmed(tmp_path, monkeypatch):
    m = {"provider": "mock", "name": "gen-model", "usd_per_m_input": 1.0, "usd_per_m_output": 2.0}
    monkeypatch.setattr(g, "model_entry", lambda alias: (m, {"base_url": "http://unused", "api_key_env": "X"}))
    ps = [{"pid": f"wn{i:04d}", "premise": f"Premise number {i} is long enough.", "source": "wikinews"}
          for i in range(3)]
    reqs = g.build_requests(ps, ["mnli", "cuefree"], ["gen-a"], examples=[])
    cache, costs = ResponseCache(tmp_path / "c.sqlite"), CostLog(tmp_path / "costs.jsonl")
    with pytest.raises(g.SpendNotConfirmed):
        g.execute(reqs, False, None, cache=cache, costs=costs)
    client = FakeClient()
    res = g.execute(reqs, True, 1.0, cache=cache, costs=costs, clients={"gen-a": client})
    assert client.calls == 6 and len(res["items"]) == 18
    assert {i["label"] for i in res["items"]} == {"entailment", "neutral", "contradiction"}
    assert costs.total() == pytest.approx(6 * (300 * 1.0 + 40 * 2.0) / 1e6)
    g.execute(reqs, True, 1.0, cache=cache, costs=costs, clients={"gen-a": client})
    assert client.calls == 6  # cached


def _cands():
    return [{"id": g.opaque_id("g", "mnli", f"p{i}", lab, "gen-a"), "premise": "P.", "hypothesis": f"H{i}{k}",
             "label": lab, "origin": "generated", "style": "mnli", "generator": "gen-a", "pid": f"p{i}"}
            for i in range(4) for k, lab in enumerate(("entailment", "neutral", "contradiction"))]


def test_sheets_are_blind_and_shuffled_per_annotator(tmp_path):
    paths = s.write_sheets(_cands(), ["AT", "SC"], tmp_path)
    a, b = (p.read_text() for p in paths)
    body = a.split("\n", 1)[1]
    assert not any(w in body for w in ("entailment", "neutral", "contradiction", "mnli"))  # nothing gives it away
    assert a != b and sorted(a.splitlines()[1:]) == sorted(b.splitlines()[1:])


def test_merge_keeps_unanimous_intended_labels_and_reports_agreement():
    cands = _cands()
    sheets = {"AT": {}, "SC": {}}
    for c in cands:
        for a in sheets:
            sheets[a][c["id"]] = {"label": c["label"], "problem": "", "comment": ""}
    sheets["SC"][cands[0]["id"]]["label"] = "neutral"           # disagreement: dropped
    sheets["AT"][cands[1]["id"]]["problem"] = "ambiguous"       # flagged: dropped
    del sheets["SC"][cands[2]["id"]]                            # not labelled: dropped
    kept, rep = s.merge(cands, sheets)
    assert len(kept) == 9 and rep["kept"] == 9 and rep["flagged"] == 1
    assert rep["fully_labelled"] == 11 and 0 < rep["kappa_AT_SC"] < 1
    from nlicorrupt.fresh import validate

    assert validate(kept) == []


def test_hand_written_items_are_not_validated_by_their_author(tmp_path):
    t = s.write_template([{"pid": "wn0001", "premise": "P."}], tmp_path / "write_AT.csv")
    rows = t.read_text().splitlines()
    rows[1] = 'wn0001,P.,"E h.","N h.","C h.",AT'
    t.write_text("\n".join(rows) + "\n")
    items = s.read_template(t, {"wn0001": {"source": "wikinews"}})
    assert len(items) == 3 and all(i["author"] == "AT" and i["origin"] == "human" for i in items)
    sheets = {"AT": {}, "SC": {i["id"]: {"label": i["label"], "problem": ""} for i in items}}
    kept, _ = s.merge(items, sheets)
    assert len(kept) == 3 and all(k["validated_by"] == ["SC"] for k in kept)


# --- natural pairs ---

from nlicorrupt.freshgen import natural as nat  # noqa: E402


def test_wikitext_to_text_keeps_prose_only():
    wt = ("{{Infobox person|name=X}}'''Jane Doe''' is a [[Finland|Finnish]] chemist<ref>Source.</ref> "
          "born in [[Turku]].\n== Career ==\n* list item\n[[File:x.jpg|thumb]]She works at the university.")
    t = nat.wikitext_to_text(wt)
    assert t.splitlines() == ["Jane Doe is a Finnish chemist born in Turku.", "She works at the university."]


def test_changed_fact():
    assert nat.changed_fact("The bridge opened in 2021 after delays.", "The bridge opened in 2023 after delays.")
    assert not nat.changed_fact("The bridge opened after long delays.", "The bridge opened after lengthy delays.")


def test_revision_pairs_from(needs_nltk):
    old = "The new bridge over the river was opened in March 2026 by the mayor of the town."
    new = "The new bridge over the river was opened in May 2026 by the mayor of the town."
    assert nat.revision_pairs_from(old, new, 12, 50) == [(new, old)]
    assert nat.revision_pairs_from(old, old, 12, 50) == []  # unchanged


def test_headline_pairs_from_global_voices(needs_nltk):
    def get(api, params):
        assert "globalvoices" in api
        return [{"title": {"rendered": "Riverton council approves library budget"},
                 "excerpt": {"rendered": "<p>The city council of Riverton approved a new budget for public libraries "
                                         "after a long debate on Tuesday evening. More [&hellip;]</p>"},
                 "link": "https://globalvoices.org/x", "date": "2026-10-05T10:00:00"},
                {"title": {"rendered": "Short"}, "excerpt": {"rendered": "x"}, "link": "", "date": ""}]

    wiki = pr.Wiki("test-agent (contact: test)", sleep=0, get=get, cache=False)
    cfg = {"since": "2026-09-01", "min_words": 12, "max_words": 50, "seed": 0, "user_agent": "x"}
    items = nat.build(cfg, ["headline"], 5, wiki=wiki, until="2026-10-31")
    assert len(items) == 1 and items[0]["hypothesis"] == "Riverton council approves library budget"
    assert items[0]["premise"].startswith("The city council of Riverton") and items[0]["label"] is None
    assert items[0]["style"] == "natural-headline"


def test_current_events_text():
    wt = ("{{Current events header|2026|9|1}}\n'''Armed conflicts'''\n* [[Some war]]\n"
          "** Rebels capture the town of [[Exampleville]] after two days of fighting near the border. "
          "([https://example.org/a Reuters])\n")
    assert pr.current_events_text(wt).splitlines() == [
        "Some war", "Rebels capture the town of Exampleville after two days of fighting near the border."]


def test_merge_natural_pairs_keeps_unanimous_labels():
    items = [{"id": f"n{i}", "premise": "P.", "hypothesis": f"H{i}", "label": None, "origin": "natural",
              "style": "natural-headline"} for i in range(3)]
    sheets = {"AT": {"n0": {"label": "entailment", "problem": ""}, "n1": {"label": "neutral", "problem": ""},
                     "n2": {"label": "entailment", "problem": ""}},
              "SC": {"n0": {"label": "entailment", "problem": ""}, "n1": {"label": "entailment", "problem": ""},
                     "n2": {"label": "entailment", "problem": "needs outside knowledge"}}}
    kept, rep = s.merge(items, sheets)
    assert [k["id"] for k in kept] == ["n0"] and kept[0]["label"] == "entailment"
    assert rep["natural_unanimous"] == pytest.approx(66.7) and "agree_with_intended" not in rep


def test_reasoning_generator_uses_its_reasoning_parameters():
    reqs = g.build_requests([{"pid": "wn0001", "premise": "P.", "source": "wikinews"}], ["cuefree"],
                            ["fresh-generator-a", "fresh-generator-b"], examples=[])
    llama = reqs[0].body
    assert llama["model"].startswith("Llama-4-Maverick") and llama["max_tokens"] == 400
    reqs = g.build_requests([{"pid": f"wn{i}", "premise": "P.", "source": "wikinews"} for i in range(2)],
                            ["cuefree"], ["fresh-generator-a", "fresh-generator-b"], examples=[])
    gpt = reqs[1].body  # generators alternate over premises
    assert gpt["model"] == "gpt-5.5" and "temperature" not in gpt and "max_tokens" not in gpt
    assert gpt["max_completion_tokens"] == 4000 and gpt["reasoning_effort"] == "low"


def test_wiki_responses_are_cached(tmp_path):
    calls = []

    def get(api, params):
        calls.append(params)
        return {"query": {"pages": [{"title": "A", "extract": "Text."}]}}

    w = pr.Wiki("test-agent (contact: test)", sleep=0, get=get, cache=tmp_path / "w.sqlite")
    assert w.texts("wikinews", ["A"]) == {"A": "Text."}
    w2 = pr.Wiki("test-agent (contact: test)", sleep=0, get=get, cache=tmp_path / "w.sqlite")  # a rerun
    assert w2.texts("wikinews", ["A"]) == {"A": "Text."} and len(calls) == 1 and w2.cached == 1


def test_sample_premises_keeps_whole_premises_balanced():
    items = [{"id": f"{g}{src}{i}{st}{lab}", "origin": "generated", "generator": g, "source_text": src,
              "pid": f"{g}{src}{i}", "style": st, "label": lab}
             for g in ("a", "b") for src in ("wiki", "news") for i in range(10)
             for st in ("mnli", "cuefree") for lab in ("entailment", "neutral", "contradiction")]
    items += [{"id": "n1", "origin": "natural", "style": "natural-revision", "label": None}]
    out = s.sample_premises(items, 8)
    pids = {it["pid"] for it in out if it["origin"] == "generated"}
    assert len(pids) == 8 and len(out) == 8 * 6 + 1
    assert {(it["generator"], it["source_text"]) for it in out if it["origin"] == "generated"} == {
        ("a", "wiki"), ("a", "news"), ("b", "wiki"), ("b", "news")}


def test_trivial_revisions_are_recognised_and_dropped_at_merge():
    assert nat.trivial_revision("He has a bachelor's degree.", "He has a Bachelor's degree.")
    assert nat.trivial_revision("A 15&nbsp;km walk.", "A 15 km walk.")
    assert nat.trivial_revision("She ran against X in 2026.", "In 2026 she ran against X.")
    assert not nat.trivial_revision("It opened in 2021.", "It opened in 2023.")
    items = [{"id": "n1", "premise": "A b c.", "hypothesis": "a B c.", "label": None, "origin": "natural",
              "style": "natural-revision"}]
    sheets = {"AT": {"n1": {"label": "entailment", "problem": ""}}, "SC": {"n1": {"label": "entailment", "problem": ""}}}
    kept, rep = s.merge(items, sheets)
    assert kept == [] and rep["trivial_revision_pairs_dropped"] == 1


def test_tool_export_with_many_raters(tmp_path):
    cands = _cands()[:6]
    f = tmp_path / "labels.csv"
    rows = ["item,premise,hypothesis,rater_id,name,age,native_lang,english_level,label,problem,seconds,time"]
    for k, c in enumerate(cands):
        raters = ["r1", "r2"] if k < 4 else ["r3"]          # the last two items get one label only
        for r in raters:
            lab = c["label"][0] if not (k == 0 and r == "r2") else "n"   # one disagreement
            rows.append(f"{c['id']},P.,H.,{r},Name,30,english,native,{lab},,12,t")
    f.write_text("\n".join(rows) + "\n")
    sheets, stats = s.read_labels_file(f)
    assert stats["raters"] == 3 and stats["labels"] == 10 and "Name" not in str(stats)
    kept, rep = s.merge(cands, sheets)
    assert len(kept) == 3                       # items 1-3; item 0 disagrees; items 4-5 have one label
    assert rep["labels_per_item"] == {2: 4, 1: 2} and rep["krippendorff_alpha"] < 1


def test_krippendorff_alpha_perfect_and_chance():
    assert s.krippendorff_alpha([["e", "e"], ["n", "n"], ["c", "c"]]) == 1.0
    assert s.krippendorff_alpha([["e", "n"], ["n", "e"]]) < 0


def test_fresh_results_summary_and_privacy(tmp_path, monkeypatch):
    from nlicorrupt.analysis import fresh_results as fr
    from nlicorrupt.data import write_jsonl
    from nlicorrupt.export import is_private
    from pathlib import Path

    meta = {f"fresh_nli:test:g{i}": {"id": f"g{i}", "premise": "P.", "hypothesis": "H.", "label": "entailment",
                                      "style": "mnli" if i < 10 else "cuefree", "origin": "generated",
                                      "author": "fresh-generator-a"} for i in range(20)}
    run = tmp_path / "logprob" / "m" / "fresh_nli" / "2022" / "k0"
    write_jsonl(run / "clean.jsonl.gz", [{"id": i, "label": 0, "pred": 0, "scores": [0.0, -1.0, -2.0]} for i in meta])
    write_jsonl(run / "drop-NOUN.jsonl.gz", [{"id": i, "label": 0, "pred": 0 if n < 10 else 1,
                                              "scores": [0.0, -1.0, -2.0] if n < 10 else [-2.0, 0.0, -1.0]}
                                             for n, i in enumerate(meta)])
    df = fr.summary(meta, root=tmp_path)
    row = df[(df.condition == "style=cuefree") & (df.corruption == "drop-NOUN")].iloc[0]
    assert row["centered_accuracy"] < 100 and row["retained_from"] == "centred"
    assert set(df.condition) == {"all", "style=mnli", "style=cuefree", "generator=fresh-generator-a"}
    assert not is_private(Path("tables/fresh_summary.csv")) and not is_private(Path("tables/fresh_shortcuts.csv"))
    assert "P." not in df.to_csv()  # no item text in the summary
