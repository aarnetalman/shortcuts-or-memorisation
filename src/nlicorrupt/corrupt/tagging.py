"""Tokenisation and universal POS tagging, as in Talman et al. (2021, 2022).

Both papers use nltk.word_tokenize and nltk.pos_tag(..., tagset='universal') with
the averaged perceptron tagger. NLTK >= 3.9 ships the tagger as
'averaged_perceptron_tagger_eng'. Tagging results agree with the 2021/2022 files
on the large majority of items (see tests/test_originals.py); remaining
differences come from changes in NLTK's quote handling and detokeniser.
"""

from __future__ import annotations

import os
from functools import lru_cache

NLTK_RESOURCES = ("punkt_tab", "averaged_perceptron_tagger_eng", "universal_tagset")

TAGS = ("ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB", ".", "X")
# The eight classes removed in both papers.
CORRUPTION_CLASSES = ("ADJ", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "VERB")


def ensure_nltk_data(download: bool = False) -> None:
    """Point NLTK at $NLTK_DATA and optionally download the resources."""
    import nltk

    from ..paths import nltk_root

    root = str(nltk_root())
    if root not in nltk.data.path:
        nltk.data.path.insert(0, root)
    if download:
        # NLTK >= 3.10 only installs into (and reads from) a data directory that no
        # other account can write to. A default umask such as 0002 would make
        # a new directory group-writable, so create it private and fix existing ones.
        os.makedirs(root, mode=0o700, exist_ok=True)
        mode = os.stat(root).st_mode
        if mode & 0o022:
            os.chmod(root, mode & ~0o022)
        for r in NLTK_RESOURCES:
            if not nltk.download(r, download_dir=root, quiet=True):
                raise RuntimeError(
                    f"could not download NLTK resource {r!r} into {root}. NLTK >= 3.10 refuses "
                    "data directories that other users can write to: check `ls -ld` on it and its "
                    "subfolders (chmod go-w), or set NLTK_DATA to a private directory such as "
                    "~/nltk_data. Behind an HTTP proxy it also needs NLTK_ALLOW_PROXIED_URLOPEN=1."
                )


@lru_cache(maxsize=1)
def _tools():
    import nltk

    ensure_nltk_data()
    return nltk.word_tokenize, nltk.pos_tag


def tag(text: str) -> list[tuple[str, str]]:
    """Return [(token, universal_tag), ...] for one text."""
    word_tokenize, pos_tag = _tools()
    if not text or not text.strip():
        return []
    return [(w, t) for w, t in pos_tag(word_tokenize(text), tagset="universal")]


def tag_record(record: dict, keys: tuple[str, ...]) -> dict:
    """Tag the text fields of a record. Output keeps id/label and adds 'tagged'."""
    return {
        "id": record["id"],
        "label": record["label"],
        "text": {k: record[k] for k in keys},
        "tagged": {k: [list(p) for p in tag(record[k])] for k in keys},
    }
