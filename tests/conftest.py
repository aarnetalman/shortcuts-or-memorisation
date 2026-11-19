import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    """Every test writes under its own temporary NLICORRUPT_DATA."""
    monkeypatch.setenv("NLICORRUPT_DATA", str(tmp_path / "work"))
    # Keep NLTK data where the user put it, if anywhere.
    if "NLTK_DATA" in os.environ:
        monkeypatch.setenv("NLTK_DATA", os.environ["NLTK_DATA"])
    yield


def nltk_ready() -> bool:
    try:
        from nlicorrupt.corrupt import tag

        tag("The cat sat.")
        return True
    except LookupError:
        return False


@pytest.fixture
def needs_nltk():
    if not nltk_ready():
        pytest.skip("NLTK resources missing: run `nlicorrupt download` or set NLTK_DATA")


def tagged_record(rid="r1", label=0, **fields):
    """Hand-tagged record for tests that should not depend on the NLTK tagger."""
    return {
        "id": rid,
        "label": label,
        "text": {k: " ".join(w for w, _ in v) for k, v in fields.items()},
        "tagged": {k: [list(p) for p in v] for k, v in fields.items()},
    }
