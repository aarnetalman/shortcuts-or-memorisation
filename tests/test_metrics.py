import pytest

from nlicorrupt.metrics import bootstrap_ci, compute, paired_delta_ci
from nlicorrupt.tasks import get_task


def test_classification_metrics():
    mrpc = get_task("mrpc")
    m = compute(mrpc, [1, 0, 1, 1], [1, 0, 0, 1])
    assert m["accuracy"] == pytest.approx(75.0)
    assert m["f1"] == pytest.approx(100 * 2 * 2 / (2 * 2 + 1))
    assert m["n"] == 4


def test_mcc_and_unparsed_predictions():
    cola = get_task("cola")
    assert compute(cola, [0, 1, 0, 1], [0, 1, 0, 1])["mcc"] == pytest.approx(100.0)
    assert compute(cola, [0, 1, 0, 1], [1, 1, 1, 1])["mcc"] == 0.0
    mnli = get_task("mnli")
    assert compute(mnli, [0, 1, 2], [0, None, 2])["accuracy"] == pytest.approx(200 / 3)


def test_regression_metrics():
    stsb = get_task("stsb")
    m = compute(stsb, [0.0, 1.0, 2.0, 3.0], [0.1, 1.1, 1.9, 3.2])
    assert m["pearson"] > 99 and m["spearman"] == pytest.approx(100.0)


def test_bootstrap_and_paired_delta():
    rte = get_task("rte")
    y = [0, 1] * 50
    a = list(y)
    b = [1 - v if i % 4 == 0 else v for i, v in enumerate(y)]
    lo, hi = bootstrap_ci(rte, y, b, n_boot=200)
    assert lo <= 75.0 <= hi
    d, dlo, dhi = paired_delta_ci(rte, y, a, b, n_boot=200)
    assert d == pytest.approx(-25.0) and dlo <= d <= dhi < 0
