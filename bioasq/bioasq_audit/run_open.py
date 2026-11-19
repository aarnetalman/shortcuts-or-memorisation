"""Run an open-weight model on a corrupted items file.

Yes/no: score by the log-probability of " yes" versus " no" as the next token
after the prompt (first subword if multi-token). Both scores are stored so
that centred accuracy and AUC can be computed later.
Factoid and list: greedy decoding, parsed one answer per line.

Runs on the Azure container. Swap to vLLM for throughput if needed; the
output format must stay the same.

Output: results/{model_tag}__{condition}__{corruption}.jsonl with one line per item:
{id, type, gold, pred, logp_yes, logp_no, raw}
"""
import argparse
import json
import os
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import YESNO_ANSWERS, build


def first_token_ids(tok, answers: List[str]) -> List[int]:
    ids = []
    for a in answers:
        enc = tok.encode(" " + a, add_special_tokens=False)
        ids.append(enc[0])
    return ids


@torch.no_grad()
def score_yesno(model, tok, prompts: List[str], ans_ids: List[int], device) -> List[List[float]]:
    out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        out.append([logp[i].item() for i in ans_ids])
    return out


@torch.no_grad()
def generate(model, tok, prompts: List[str], device, max_new_tokens: int) -> List[str]:
    out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        out.append(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def parse_lines(text: str, limit: int) -> List[str]:
    lines = []
    for l in text.strip().split("\n"):
        l = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip()
        if l:
            lines.append(l)
        if len(lines) >= limit:
            break
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True, help="one corrupted JSONL file")
    ap.add_argument("--cond", choices=["gold", "noctx"], default="gold")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--types", default="yesno,factoid,list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 device_map="auto")
    model.eval()
    ans_ids = first_token_ids(tok, YESNO_ANSWERS)

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    types = set(args.types.split(","))
    items = [it for it in items if it["type"] in types]
    if args.limit:
        items = items[:args.limit]
    corruption = items[0].get("corruption", "clean") if items else "clean"
    if args.cond == "noctx":
        corruption = "noctx"
    tag = args.model.replace("/", "_")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{tag}__{args.cond}__{corruption}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        yn = [it for it in items if it["type"] == "yesno"]
        if yn:
            prompts = [build(it, args.cond) for it in yn]
            scores = score_yesno(model, tok, prompts, ans_ids, device)
            for it, (ly, ln) in zip(yn, scores):
                f.write(json.dumps({"id": it["id"], "type": "yesno", "gold": it["gold"],
                                    "pred": "yes" if ly > ln else "no",
                                    "logp_yes": ly, "logp_no": ln}) + "\n")
        for qtype, limit, ntok in (("factoid", 5, 48), ("list", 30, 160)):
            sub = [it for it in items if it["type"] == qtype]
            if not sub:
                continue
            prompts = [build(it, args.cond) for it in sub]
            raws = generate(model, tok, prompts, device, ntok)
            for it, raw in zip(sub, raws):
                f.write(json.dumps({"id": it["id"], "type": qtype, "gold": it["gold"],
                                    "pred": parse_lines(raw, limit), "raw": raw}, ensure_ascii=False) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
