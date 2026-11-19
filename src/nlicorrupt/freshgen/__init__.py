"""Building the fresh control set: premises from recent text, generated hypotheses,
human validation.

  premises  fetch sentences from Wikinews and newly created Wikipedia articles published
            after a cut-off date, optionally dropping any found in a pre-training corpus
  generate  ask a generator model (one not evaluated in the study) for an entailed, a
            neutral and a contradicting hypothesis per premise, in one of two styles
  natural   naturally occurring pairs (Wikinews headline and lead; Wikipedia sentences
            before and after an edit that changed a fact), labelled from scratch
  sheets    per-annotator validation sheets (labels hidden), and a template for
            hand-written items
  merge     read the filled sheets, report agreement, write the final item file

Everything is written under $NLICORRUPT_PRIVATE/fresh/, outside this repository.
"""
