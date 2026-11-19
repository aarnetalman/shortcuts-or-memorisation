import pytest

from nlicorrupt.prompts import (
    TEMPLATES,
    chat_messages,
    gold_answer,
    parse_answer,
    render,
    select_exemplars,
    text_prompt,
)
from nlicorrupt.tasks import TASKS, get_task


def test_every_task_has_a_template_with_one_answer_per_label():
    for name, spec in TASKS.items():
        t = TEMPLATES[name]
        assert len(t.answers) == (6 if spec.is_regression else spec.num_labels)
        assert len(set(t.answers)) == len(t.answers)
        rec = {k: "x" for k in spec.keys}
        assert "{" not in render(name, rec)


def test_text_and_chat_prompts_with_exemplars():
    ex = [{"premise": "A man sleeps.", "hypothesis": "A man is awake.", "label": 2}]
    rec = {"premise": "", "hypothesis": "Someone runs.", "label": 0}
    p = text_prompt("mnli", rec, ex)
    assert p.endswith("Answer:") and "Answer: contradiction" in p and "Premise: \n" in p
    msgs = chat_messages("mnli", rec, ex)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "contradiction"
    assert gold_answer("stsb", {"label": 3.6}) == "4"


@pytest.mark.parametrize(
    "task,text,expected",
    [
        ("mnli", "Neutral.", 1),
        ("mnli", "**Contradiction** because...", 2),
        ("rte", "Yes, it does.", 0),
        ("qnli", "no", 1),
        ("cola", "I am not sure", None),
        ("stsb", "I'd say 3.5", 3.5),
        ("stsb", "7", 5.0),
        ("sst2", "", None),
    ],
)
def test_parse_answer(task, text, expected):
    assert parse_answer(task, text) == expected


def test_exemplars_are_balanced_and_deterministic():
    pool = [{"id": str(i), "label": i % 3} for i in range(30)]
    a = select_exemplars("mnli", pool, 6, seed=1)
    assert a == select_exemplars("mnli", pool, 6, seed=1)
    assert sorted(e["label"] for e in a) == [0, 0, 1, 1, 2, 2]
    assert select_exemplars("mnli", pool, 0) == []
    assert get_task("mnli").num_labels == 3
