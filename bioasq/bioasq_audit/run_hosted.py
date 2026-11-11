"""Run a hosted model through an OpenAI-compatible chat endpoint (Azure AI
Foundry, OpenAI, DeepSeek, Mistral all expose one).

Set OPENAI_BASE_URL and OPENAI_API_KEY (or AZURE_* equivalents) in the
environment. Temperature 0. Requests top-5 logprobs on the first generated
token for yes/no where the endpoint supports it, so that centred accuracy can
be computed as for the open models. Same output format as run_open.py.

Resumable: items already in the output file are skipped.
"""
import argparse
import json
import math
import os
import time
from typing import Dict, Optional

from openai import OpenAI

from .prompts import build
import re


def parse_lines(text, limit):
    lines = []
    for l in text.strip().split("\n"):
        l = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip()
        if l:
            lines.append(l)
        if len(lines) >= limit:
            break
    return lines


def call(client: OpenAI, model: str, prompt: str, max_tokens: int, logprobs: bool) -> Dict:
    for attempt in range(6):
        try:
            kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}],
                          temperature=0, max_tokens=max_tokens)
            if logprobs:
                kwargs.update(logprobs=True, top_logprobs=5)
            r = client.chat.completions.create(**kwargs)
            choice = r.choices[0]
            out = {"text": choice.message.content or ""}
            if logprobs and choice.logprobs and choice.logprobs.content:
                top = choice.logprobs.content[0].top_logprobs
                out["top"] = {t.token.strip().lower(): t.logprob for t in top}
            return out
        except Exception as e:  # rate limits, transient errors
            wait = 2 ** attempt
            print(f"retry {attempt} after {wait}s: {e}")
            time.sleep(wait)
    return {"text": "", "error": "failed"}


def yesno_from(resp: Dict) -> Dict:
    text = resp["text"].strip().lower()
    pred = "yes" if text.startswith("yes") else "no" if text.startswith("no") else text[:10]
    top = resp.get("top", {})
    ly: Optional[float] = top.get("yes")
    ln: Optional[float] = top.get("no")
    # a missing answer in the top-5 gets a floor below the lowest listed logprob
    if top and (ly is None or ln is None):
        floor = min(top.values()) - math.log(5)
        ly = floor if ly is None else ly
        ln = floor if ln is None else ln
    return {"pred": pred, "logp_yes": ly, "logp_no": ln}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--cond", choices=["gold", "noctx"], default="gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="yesno,factoid,list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-logprobs", action="store_true")
    args = ap.parse_args()

    client = OpenAI()
    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    types = set(args.types.split(","))
    items = [it for it in items if it["type"] in types]
    if args.limit:
        items = items[:args.limit]
    corruption = items[0].get("corruption", "clean") if items else "clean"
    if args.cond == "noctx":
        corruption = "noctx"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model}__{args.cond}__{corruption}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}

    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            if it["id"] in done:
                continue
            prompt = build(it, args.cond)
            if it["type"] == "yesno":
                resp = call(client, args.model, prompt, 3, not args.no_logprobs)
                rec = {"id": it["id"], "type": "yesno", "gold": it["gold"], "raw": resp["text"]}
                rec.update(yesno_from(resp))
            else:
                limit, ntok = (5, 48) if it["type"] == "factoid" else (30, 160)
                resp = call(client, args.model, prompt, ntok, False)
                rec = {"id": it["id"], "type": it["type"], "gold": it["gold"],
                       "pred": parse_lines(resp["text"], limit), "raw": resp["text"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print("wrote", path)


if __name__ == "__main__":
    main()
