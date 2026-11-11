"""Generated hypotheses for the fresh premises: one entailed, one neutral and one
contradicting hypothesis per premise, from a generator model that is not evaluated.

Two styles:
  mnli     MNLI's annotation instructions with a few MNLI training pairs as examples, so
           the hypotheses carry MNLI-like regularities (the shortcuts survive, the items
           are new)
  cuefree  the same task, with instructions that rule out the best-known hypothesis-only
           cues (negation for contradiction, vagueness for neutral, copying for entailment)

Like `nlicorrupt prompt`, this is a dry run unless spending is confirmed with a budget;
every response is cached and every paid call logged.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import load_yaml
from ..prompting.cache import CostLog, ResponseCache, request_key
from ..prompting.client import APIError, ChatClient, response_text
from ..prompting.runner import SpendNotConfirmed, billed_tokens, cost_usd, estimate_tokens, model_entry, out_tokens
from ..tasks import NLI_LABELS

log = logging.getLogger(__name__)

TASK = (
    "You write items for a natural language inference dataset. You are given a premise, a "
    "sentence from a recent news or encyclopedia article. Write three hypotheses, each a "
    "single sentence:\n"
    "- entailment: definitely true if the premise is true\n"
    "- neutral: might be true or false; the premise does not settle it\n"
    "- contradiction: definitely false if the premise is true\n"
)
MNLI_STYLE = (
    "Write the hypotheses the way crowd workers did for the MultiNLI dataset: short, natural "
    "sentences about the situation the premise describes. Examples:\n"
)
CUEFREE_STYLE = (
    "Avoid the patterns that let a reader guess the label from the hypothesis alone:\n"
    "- do not use negation words (not, no, never, nobody, nothing, none) in any hypothesis; "
    "a contradiction must state a different fact, not deny the premise\n"
    "- the neutral hypothesis must be as specific as the others, not vague or hedged\n"
    "- the entailment must not copy a long span of the premise; paraphrase or infer\n"
    "- keep the three hypotheses about the same length and about the same topic\n"
)
FORMAT = ('Answer with JSON only, in this form: {"entailment": "...", "neutral": "...", '
          '"contradiction": "..."}')
STYLES = ("mnli", "cuefree")


@dataclass
class GenRequest:
    premise: dict
    style: str
    model: str
    body: dict
    key: str


def mnli_examples(k: int, seed: int = 0, pool: list[dict] | None = None) -> list[dict]:
    """k MNLI training pairs per label (premise, hypothesis, label name)."""
    if pool is None:
        from ..data import load_split

        pool = load_split("mnli", "train")
    rng = random.Random(f"fresh-examples|{seed}")
    out = []
    for lab in range(3):
        cands = [r for r in pool if int(r["label"]) == lab and 5 <= len(r["premise"].split()) <= 40]
        out += [{**r, "label_name": NLI_LABELS[lab]} for r in rng.sample(cands, k)]
    rng.shuffle(out)
    return out


def messages(premise: str, style: str, examples: list[dict]) -> list[dict]:
    if style == "mnli":
        ex = "\n".join(f"Premise: {e['premise']}\n{e['label_name']}: {e['hypothesis']}" for e in examples)
        guide = MNLI_STYLE + ex + "\n"
    elif style == "cuefree":
        guide = CUEFREE_STYLE
    else:
        raise ValueError(f"unknown style {style!r}; use one of {STYLES}")
    return [{"role": "system", "content": TASK + guide + FORMAT},
            {"role": "user", "content": f"Premise: {premise}"}]


def build_requests(premises: list[dict], styles: list[str], models: list[str], examples: list[dict],
                   temperature: float = 0.7, max_tokens: int = 400) -> list[GenRequest]:
    """Each premise gets every style; generators alternate over premises."""
    reqs = []
    for i, p in enumerate(premises):
        model = models[i % len(models)]
        m, _ = model_entry(model)
        # A model that always reasons uses its reasoning parameters (e.g. no temperature,
        # max_completion_tokens instead of max_tokens), as in the hosted-model runs.
        params = dict(m.get("params_on" if m.get("reasoning") == "always" else "params_off") or {})
        for style in styles:
            body = {"model": m["name"] or f"<unset:{model}>", "messages": messages(p["premise"], style, examples),
                    "temperature": temperature, "max_tokens": max_tokens}
            for k, v in params.items():
                if v is None:
                    body.pop(k, None)
                else:
                    body[k] = v
            reqs.append(GenRequest(p, style, model, body, request_key(m["provider"], body)))
    return reqs


_JSON = re.compile(r"\{.*\}", re.S)


def parse(text: str | None) -> dict | None:
    """The three hypotheses from a response, or None if any is missing."""
    if not text:
        return None
    m = _JSON.search(re.sub(r"<think>.*?</think>", "", text, flags=re.S))
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    hyps = {lab: " ".join(str(d.get(lab, "")).split()) for lab in NLI_LABELS}
    return hyps if all(hyps.values()) else None


def plan(reqs: list[GenRequest], cache: ResponseCache) -> dict:
    todo = [r for r in reqs if not cache.has(r.key)]
    by_model: dict[str, list[GenRequest]] = {}
    for r in todo:
        by_model.setdefault(r.model, []).append(r)
    est = {}
    for model, rs in by_model.items():
        m, _ = model_entry(model)
        priced = m.get("usd_per_m_input") is not None and m.get("usd_per_m_output") is not None
        tin = sum(estimate_tokens(r.body) for r in rs)
        est[model] = {"to_send": len(rs), "est_input_tokens": tin,
                      "est_usd_expected": cost_usd(m, tin, 120 * len(rs)) if priced else None,
                      "est_usd_upper_bound": cost_usd(m, tin, sum(out_tokens(r.body, 400) for r in rs)) if priced else None}
    return {"requests": len(reqs), "cached": len(reqs) - len(todo), "to_send": len(todo), "by_model": est,
            "todo": todo}


def execute(reqs: list[GenRequest], confirm_spend: bool, max_usd: float | None, cache: ResponseCache | None = None,
            costs: CostLog | None = None, clients: dict | None = None) -> dict:
    import os

    cache = cache or ResponseCache()
    costs = costs or CostLog()
    p = plan(reqs, cache)
    if p["todo"]:
        if not confirm_spend:
            raise SpendNotConfirmed("dry run: nothing was sent. Re-run with --confirm-spend --max-usd N after approval.")
        if max_usd is None:
            raise SpendNotConfirmed("--confirm-spend needs --max-usd")
        spent = 0.0
        clients = clients or {}
        for r in p["todo"]:
            m, prov = model_entry(r.model)
            if m.get("usd_per_m_input") is None or not m.get("name") or not prov.get("base_url"):
                raise SpendNotConfirmed(f"set name, prices and provider base_url for {r.model} in configs/providers.yaml")
            worst = cost_usd(m, estimate_tokens(r.body) * 2, out_tokens(r.body, 400))
            if spent + worst > max_usd:
                log.warning("budget of $%.2f reached; stopped early (spent $%.4f)", max_usd, spent)
                break
            if r.model not in clients:
                key = os.environ.get(prov["api_key_env"])
                if not key:
                    raise SpendNotConfirmed(f"environment variable {prov['api_key_env']} is not set")
                clients[r.model] = ChatClient(prov["base_url"], key, auth=prov.get("auth", "bearer"),
                                              api_version=prov.get("api_version"),
                                              deployment_in_path=bool(prov.get("deployment_in_path")),
                                              timeout=float(m.get("timeout") or prov.get("timeout") or 120.0))
            try:
                resp = clients[r.model].chat(r.body)
            except APIError as e:
                if not e.transient:
                    raise
                log.warning("request failed, skipped (%s)", e)
                continue
            u = resp.get("usage") or {}
            tin, tout = billed_tokens(u, estimate_tokens(r.body), out_tokens(r.body, 400))  # incl. hidden reasoning
            c = cost_usd(m, tin, tout)
            spent += c
            cache.put(r.key, m["provider"], r.body, resp, c)
            costs.add(run=f"fresh-generate/{r.style}", model=r.model, provider_model=m["name"], task="fresh",
                      corruption=r.style, id=r.premise["pid"], input_tokens=tin, output_tokens=tout, cost_usd=c)
        log.info("spent $%.4f this run", spent)
    return candidates(reqs, cache)


def candidates(reqs: list[GenRequest], cache: ResponseCache) -> dict:
    """Cached responses as candidate items: three per premise and style."""
    items, failed, missing = [], 0, 0
    for r in reqs:
        resp = cache.get(r.key)
        if resp is None:
            missing += 1
            continue
        hyps = parse(response_text(resp))
        if hyps is None:
            failed += 1
            continue
        p = r.premise
        for lab in NLI_LABELS:
            items.append({"id": opaque_id("g", r.style, p["pid"], lab, r.model), "premise": p["premise"], "hypothesis": hyps[lab],
                          "label": lab, "origin": "generated", "style": r.style, "generator": r.model,
                          "pid": p["pid"], "source_text": p["source"], "url": p.get("url"), "date": p.get("date")})
    return {"items": items, "unparsed": failed, "missing": missing}


def opaque_id(prefix: str, *parts) -> str:
    """An item id that does not reveal the intended label (annotators see the ids)."""
    import hashlib

    return prefix + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def config() -> dict:
    return load_yaml("fresh.yaml")["generate"]
