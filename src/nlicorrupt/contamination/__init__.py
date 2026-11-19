"""Contamination checks.

infinigram: exact n-gram counts of evaluation items in an open pre-training corpus
            (OLMo's data) through the infini-gram API -> items labelled seen/unseen.
guess:      TS-guessing (Deng et al., 2024): can the model fill in a masked word of a
            benchmark item it should not be able to guess?
mink:       Min-K%++ (Zhang et al., 2024) membership scores from a local model.

None of these proves contamination on its own; they are reported as indicators,
and the seen/unseen split from OLMo's open data is the main contrast.
"""
