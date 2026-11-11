"""Twenty handwritten yes/no questions, each isolating one cue. These are not
BioASQ items and the expected label is the SHORTCUT label the cue net should
produce, not necessarily the true answer. This checks that the weights
implement the rules. It is not a tuning set.
"""
import os

import pytest

from bioasq_audit.cue_net import load_weights, predict
from bioasq_audit.features import features

W = load_weights(os.path.join(os.path.dirname(__file__), "..", "weights", "yesno_v0.json"))

# Known property of v0: universal (-0.9) plus association (+0.5) plus bias (0.6)
# sums to +0.2, so "expressed only in X" is still called yes. Recorded, not tuned.
CASES = [
    # no cue: bias alone -> yes
    ("Is protein X expressed in the liver?", "yes"),
    ("Does drug Y bind receptor Z?", "yes"),
    # association -> yes
    ("Is gene A associated with disease B?", "yes"),
    ("Is enzyme C involved in the metabolism of D?", "yes"),
    # possibility -> yes
    ("Can compound E cause liver toxicity?", "yes"),
    ("Could infection with virus F lead to encephalitis?", "yes"),
    # approval -> yes
    ("Is drug G approved by the FDA for migraine?", "yes"),
    # universal -> no
    ("Is protein H found only in neurons?", "no"),
    ("Are all cases of disease I caused by mutation J?", "no"),
    ("Is drug K always effective in children?", "no"),
    # negation -> no (one negation plus bias still above zero; two push below)
    ("Is it true that drug L does not cross the blood-brain barrier and lacks metabolites?", "no"),
    # identity -> no
    ("Are M and N the same protein?", "no"),
    ("Are the terms O and P used interchangeably?", "no"),
    # contradiction -> yes alone (bias 0.6 - 0.4), with a number -> no
    ("Is the function of Q disputed?", "yes"),
    ("Is the function of Q disputed in 3 trials?", "no"),
    # number -> no only when combined with another negative cue
    ("Does the S complex have 7 subunits?", "yes"),
    ("Is the T gene located only on chromosome 21?", "no"),
    # comparison alone -> yes (weak cue), with universal -> no
    ("Is U more potent than V?", "yes"),
    ("Is U always more potent than V?", "no"),
    # prevent + number -> no
    ("Does vaccine W prevent 100% of infections?", "no"),
]


@pytest.mark.parametrize("q,expected", CASES)
def test_cue(q, expected):
    assert predict(features(q), W) == expected, (q, features(q))
