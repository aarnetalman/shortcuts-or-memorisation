"""Prompt templates and label words, shared by log-probability scoring, probing
and the hosted-API prompting harness, so every system sees the same input.

Each task has an instruction ending in "Answer:" and one answer word per label.
STS-B asks for an integer 0-5.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .tasks import get_task


@dataclass(frozen=True)
class Template:
    body: str  # uses the task's text keys as format fields
    answers: tuple[str, ...]  # answer word for label id i; for STS-B the scores "0".."5"


_NLI = Template(
    "Premise: {premise}\nHypothesis: {hypothesis}\n"
    "Does the premise entail the hypothesis, contradict it, or neither?\n"
    "Answer with one word: entailment, neutral, or contradiction.",
    ("entailment", "neutral", "contradiction"),
)

TEMPLATES: dict[str, Template] = {
    "cola": Template(
        "Is the following sentence grammatically acceptable in English?\nSentence: {sentence}\n"
        "Answer with one word: yes or no.",
        ("no", "yes"),
    ),
    "sst2": Template(
        "What is the sentiment of the following sentence?\nSentence: {sentence}\n"
        "Answer with one word: positive or negative.",
        ("negative", "positive"),
    ),
    "mrpc": Template(
        "Do the following two sentences mean the same thing?\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n"
        "Answer with one word: yes or no.",
        ("no", "yes"),
    ),
    "qqp": Template(
        "Are the following two questions asking the same thing?\nQuestion 1: {question1}\nQuestion 2: {question2}\n"
        "Answer with one word: yes or no.",
        ("no", "yes"),
    ),
    "stsb": Template(
        "How similar in meaning are the following two sentences, on a scale from 0 (completely different) "
        "to 5 (same meaning)?\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n"
        "Answer with a single whole number from 0 to 5.",
        ("0", "1", "2", "3", "4", "5"),
    ),
    "mnli": _NLI,
    "anli_r1": _NLI,
    "anli_r2": _NLI,
    "anli_r3": _NLI,
    "fresh_nli": _NLI,
    "qnli": Template(
        "Question: {question}\nSentence: {sentence}\n"
        "Does the sentence contain the answer to the question?\nAnswer with one word: yes or no.",
        ("yes", "no"),  # label 0 = entailment, 1 = not_entailment
    ),
    "rte": Template(
        "Premise: {sentence1}\nHypothesis: {sentence2}\n"
        "Does the premise entail the hypothesis?\nAnswer with one word: yes or no.",
        ("yes", "no"),  # label 0 = entailment, 1 = not_entailment
    ),
}

ANSWER_CUE = "Answer:"


def template(task: str) -> Template:
    return TEMPLATES[task]


def render(task: str, record: dict) -> str:
    spec = get_task(task)
    return template(task).body.format(**{k: record.get(k) or "" for k in spec.keys})


def gold_answer(task: str, record: dict) -> str:
    t = template(task)
    if get_task(task).is_regression:
        return str(int(round(float(record["label"]))))
    return t.answers[int(record["label"])]


def select_exemplars(task: str, pool: list[dict], k: int, seed: int = 0) -> list[dict]:
    """k exemplars from the training pool, cycling through labels for balance."""
    if k <= 0:
        return []
    rng = random.Random(f"exemplars|{task}|{seed}")
    spec = get_task(task)
    if spec.is_regression:
        return rng.sample(pool, k)
    by_label: dict[int, list[dict]] = {}
    for r in pool:
        by_label.setdefault(int(r["label"]), []).append(r)
    labels = sorted(by_label)
    out = []
    for i in range(k):
        out.append(rng.choice(by_label[labels[i % len(labels)]]))
    rng.shuffle(out)
    return out


def text_prompt(task: str, record: dict, exemplars: list[dict] = ()) -> str:
    """Plain-text prompt for base models; the answer follows 'Answer:' after a space."""
    parts = [f"{render(task, e)}\n{ANSWER_CUE} {gold_answer(task, e)}" for e in exemplars]
    parts.append(f"{render(task, record)}\n{ANSWER_CUE}")
    return "\n\n".join(parts)


def chat_messages(task: str, record: dict, exemplars: list[dict] = ()) -> list[dict]:
    """Chat messages for instruction-tuned models and hosted APIs."""
    msgs = []
    for e in exemplars:
        msgs.append({"role": "user", "content": render(task, e)})
        msgs.append({"role": "assistant", "content": gold_answer(task, e)})
    msgs.append({"role": "user", "content": render(task, record)})
    return msgs


_WORD = re.compile(r"[a-z]+")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_answer(task: str, text: str | None):
    """Map a free-text answer to a label id (or a float for STS-B). None if unparseable.

    Takes the first answer word that appears; reasoning traces should be removed first.
    """
    if not text:
        return None
    spec = get_task(task)
    t = text.strip().lower()
    if spec.is_regression:
        m = _NUM.search(t)
        if not m:
            return None
        return min(5.0, max(0.0, float(m.group(0))))
    answers = template(task).answers
    for w in _WORD.findall(t):
        if w in answers:
            return answers.index(w)
    return None
