import pytest

from nlicorrupt.cli import build_parser
from nlicorrupt.config import model_id
from nlicorrupt.tasks import ANLI_TASKS, CONTROL_TASKS, GLUE_TASKS, TASKS, get_task


def test_registry_matches_papers():
    assert set(GLUE_TASKS) | set(ANLI_TASKS) | set(CONTROL_TASKS) == set(TASKS)
    assert get_task("mnli").eval_split == "validation_matched"
    assert get_task("anli_r3").eval_split == "test_r3"
    assert get_task("cola").metric == "mcc"
    assert get_task("stsb").is_regression and get_task("stsb").num_labels == 1
    with pytest.raises(KeyError):
        get_task("wnli")


def test_model_aliases():
    assert model_id("roberta-base") == "FacebookAI/roberta-base"
    assert model_id("org/unknown") == "org/unknown"


def test_cli_parses_prepare():
    args = build_parser().parse_args(["prepare", "--task", "rte", "--style", "2021,2022"])
    assert args.task == "rte" and args.style == "2021,2022"


def test_split_revision():
    from nlicorrupt.config import split_revision

    assert split_revision("org/model@refs/pr/1") == ("org/model", "refs/pr/1")
    assert split_revision("org/model") == ("org/model", None)
    assert split_revision("/scratch/p/run@x/model") == ("/scratch/p/run@x/model", None)
    assert split_revision(model_id("roberta-large-anli"))[1] == "refs/pr/1"
