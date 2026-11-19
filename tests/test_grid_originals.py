import shlex

import pytest

from nlicorrupt.cli import build_parser
from nlicorrupt.commands.grid import expand
from nlicorrupt.config import load_yaml
from nlicorrupt.originals import get_records, load_released


def test_grid_expansion_parses_back():
    cfg = {
        "blocks": [
            {
                "name": "a",
                "command": "finetune",
                "matrix": {"task": ["mrpc", "rte"], "seed": [1, 2]},
                "args": {"model": "roberta-base", "eval_corruptions": ["clean", "drop-NOUN"], "no_bf16": True},
            }
        ]
    }
    lines = expand(cfg)
    assert len(lines) == 4
    args = build_parser().parse_args(shlex.split(lines[0])[1:])
    assert args.task == "mrpc" and args.seed == 1 and args.no_bf16
    assert args.eval_corruptions == "clean,drop-NOUN"


def test_shipped_grids_expand_and_parse():
    for name in ["grids/reproduce.yaml", "grids/encoders.yaml", "grids/anli.yaml", "grids/open_models.yaml", "grids/contamination.yaml"]:
        lines = expand(load_yaml(name))
        assert lines
        for line in lines:
            build_parser().parse_args(shlex.split(line)[1:])


@pytest.fixture
def fake_originals(tmp_path, monkeypatch):
    root = tmp_path / "orig"
    monkeypatch.setenv("NLICORRUPT_ORIG", str(root))
    m = root / "nli-data-sanity-check" / "data" / "MNLI" / "Corrupt-Test"
    m.mkdir(parents=True)
    (m / "MNLI-NOUN.tsv").write_text(
        "index\tsentence1\tsentence2\tgold_label\n0\tThe new are nice\treally likes the\tneutral\n"
        "1\tdo n't \" know\tI like\tcontradiction\n"
    )
    a = root / "nli-data-sanity-check" / "data" / "ANLI" / "Corrupt-Test" / "r1"
    a.mkdir(parents=True)
    (a / "test-verb.jsonl").write_text('{"uid": "u1", "premise": "p", "hypothesis": "h", "label": "e", "reason": ""}\n')
    g = root / "nlu-dataset-diagnostics" / "data"
    g.mkdir(parents=True)
    (g / "dev_mnli-NOUN.csv").write_text(",hypothesis,premise,label\n0,h0,p0,2\n1,,p1,0\n")
    (g / "dev_stsb-NOUN.csv").write_text(",sentence1,sentence2,label\n0,A,B,5.0\n")
    return root


def test_released_loaders(fake_originals):
    r = load_released("mnli", "validation_matched", "released-drop-NOUN", "2021")
    assert [x["label"] for x in r] == [1, 2] and r[1]["premise"] == 'do n\'t " know'
    r = load_released("anli_r1", "test_r1", "released-drop-VERB", "2021")
    assert r[0] == {"id": "anli_r1:test_r1:u1", "label": 0, "premise": "p", "hypothesis": "h"}
    r = get_records("mnli", "validation_matched", "released-drop-NOUN", "2022")
    assert r[0]["premise"] == "p0" and r[0]["hypothesis"] == "h0" and r[1]["hypothesis"] == ""
    assert r[0]["id"] == "mnli:validation_matched:0"
    assert load_released("stsb", "validation", "released-drop-NOUN", "2022")[0]["label"] == 5.0
    with pytest.raises(ValueError):
        load_released("mnli", "validation_matched", "released-drop-VERB", "2022")
    with pytest.raises(ValueError):
        load_released("mnli", "validation_matched", "released-keep-NOUN", "2021")
