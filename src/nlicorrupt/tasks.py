"""Task registry: datasets, text fields, labels, splits and metrics.

Splits and metrics follow Talman et al. (2021, 2022): GLUE validation sets are
the evaluation sets (test labels are not public), MNLI uses validation_matched,
ANLI uses the public test sets with dev for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Dataset revisions on the Hugging Face hub, pinned for reproducibility.
GLUE_REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
ANLI_REV = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
SNLI_REV = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
FEVER_NLI_REV = "1eddac63112eee1fdf1966e0bca27a5ff248c772"

NLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_path: str
    hf_config: str | None
    revision: str
    keys: tuple[str, ...]
    labels: tuple[str, ...]  # empty for regression
    train_split: str
    eval_split: str  # the split the papers report on
    dev_split: str  # split for model selection / k-shot exemplars
    metric: str  # primary metric, as in the original papers
    extra_metrics: tuple[str, ...] = ()
    extra_eval_splits: tuple[str, ...] = field(default_factory=tuple)
    exemplar_task: str | None = None  # where k-shot exemplars come from, if not this task

    @property
    def is_regression(self) -> bool:
        return not self.labels

    @property
    def num_labels(self) -> int:
        return 1 if self.is_regression else len(self.labels)

    @property
    def is_pair(self) -> bool:
        return len(self.keys) == 2


def _glue(name, keys, labels, metric, extra=(), eval_split="validation", extra_eval=()):
    return TaskSpec(
        name=name,
        hf_path="nyu-mll/glue",
        hf_config=name,
        revision=GLUE_REV,
        keys=keys,
        labels=labels,
        train_split="train",
        eval_split=eval_split,
        dev_split=eval_split,
        metric=metric,
        extra_metrics=extra,
        extra_eval_splits=extra_eval,
    )


def _anli(r: int) -> TaskSpec:
    return TaskSpec(
        name=f"anli_r{r}",
        hf_path="facebook/anli",
        hf_config=None,
        revision=ANLI_REV,
        keys=("premise", "hypothesis"),
        labels=NLI_LABELS,
        train_split=f"train_r{r}",
        eval_split=f"test_r{r}",
        dev_split=f"dev_r{r}",
        metric="accuracy",
    )


TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in [
        _glue("cola", ("sentence",), ("unacceptable", "acceptable"), "mcc"),
        _glue("sst2", ("sentence",), ("negative", "positive"), "accuracy"),
        _glue("mrpc", ("sentence1", "sentence2"), ("not_equivalent", "equivalent"), "accuracy", ("f1",)),
        _glue("qqp", ("question1", "question2"), ("not_duplicate", "duplicate"), "accuracy", ("f1",)),
        _glue("stsb", ("sentence1", "sentence2"), (), "pearson", ("spearman",)),
        _glue(
            "mnli",
            ("premise", "hypothesis"),
            NLI_LABELS,
            "accuracy",
            eval_split="validation_matched",
            extra_eval=("validation_mismatched",),
        ),
        _glue("qnli", ("question", "sentence"), ("entailment", "not_entailment"), "accuracy"),
        _glue("rte", ("sentence1", "sentence2"), ("entailment", "not_entailment"), "accuracy"),
        _anli(1),
        _anli(2),
        _anli(3),
        # Uncontaminated control set: items written for this study, kept in a private
        # repo until publication ($NLICORRUPT_PRIVATE/fresh/nli.jsonl, see fresh.py).
        TaskSpec(
            name="fresh_nli",
            hf_path="local:fresh/nli.jsonl",
            hf_config=None,
            revision="",
            keys=("premise", "hypothesis"),
            labels=NLI_LABELS,
            train_split="train",  # unused: exemplars come from MNLI
            eval_split="test",
            dev_split="test",
            metric="accuracy",
            exemplar_task="mnli",
        ),
    ]
}

GLUE_TASKS = ("cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte")
ANLI_TASKS = ("anli_r1", "anli_r2", "anli_r3")
CONTROL_TASKS = ("fresh_nli",)

# Training mixture of Nie et al. (2020), used by the 2021 ANLI experiments:
# SNLI + MNLI + FEVER-NLI + ANLI R1-R3, with ANLI rounds upsampled 10/20/10.
ANLI_MIX = {
    "snli": 1,
    "mnli": 1,
    "fever_nli": 1,
    "anli_r1": 10,
    "anli_r2": 20,
    "anli_r3": 10,
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as e:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(TASKS)}") from e
