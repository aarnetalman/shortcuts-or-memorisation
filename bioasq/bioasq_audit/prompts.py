"""Prompt templates. One template per question type, same across models.

The snippets block is omitted under the noctx condition. Templates end in a
one-word or short-answer instruction so that open models can be scored by
answer log-probability (yes/no) or greedy decoding (factoid, list).
"""
from typing import Dict, List

MAX_SNIPPETS = 10

YESNO = (
    "You are answering a biomedical question.{ctx}\n"
    "Question: {q}\n"
    "Answer with one word: yes or no.\nAnswer:"
)

FACTOID = (
    "You are answering a biomedical question.{ctx}\n"
    "Question: {q}\n"
    "Give up to five candidate answers as short expressions (an entity name, a number, or similar), "
    "most likely first, one per line, with no explanation.\nAnswers:"
)

LIST = (
    "You are answering a biomedical question.{ctx}\n"
    "Question: {q}\n"
    "Give the list of answers as short expressions, one per line, with no explanation.\nAnswers:"
)


def ctx_block(snippets: List[str]) -> str:
    if not snippets:
        return ""
    lines = "\n".join(f"- {s}" for s in snippets[:MAX_SNIPPETS])
    return f"\nUse the following evidence:\n{lines}"


def build(item: Dict, condition: str) -> str:
    snippets = [] if condition == "noctx" else item["snippets"]
    ctx = ctx_block(snippets)
    tpl = {"yesno": YESNO, "factoid": FACTOID, "list": LIST}[item["type"]]
    return tpl.format(ctx=ctx, q=item["question"])


YESNO_ANSWERS = ["yes", "no"]
