# Prompt for regenerating the cue-network weights with a second frontier model

Log: model name and version, date, full prompt and full reply, in weights/CHANGELOG.md. Do not show the model any BioASQ item or any result.

---

You are helping with an audit of the BioASQ yes/no question-answering benchmark. We want a linear classifier over question-only features that predicts "yes" or "no" without ever seeing the data. The features below are computed from the question text alone. Your task is to write the weight for each feature and a bias, from your knowledge of how biomedical experts write yes/no questions and from the literature on annotation artefacts in yes/no and NLI datasets (Gururangan et al. 2018, Poliak et al. 2018, Clark et al. 2019 BoolQ, the BioASQ overview papers). Decision rule: predict yes if the weighted sum is greater than zero.

For each weight give one sentence of justification and say whether it rests on a published finding or on a hypothesis about question construction. Do not ask for data and do not ask to iterate. Output JSON with keys "weights" (feature name to number) and "justification" (feature name to sentence).

Features:
- bias: constant 1
- neg: count of negation words (not, no, never, lack, absence, without, unable, fail)
- universal: 1 if the question contains only, all, always, every, exclusively, solely, entirely, any
- possibility: 1 if it contains can, could, may, might, possible, able, potential
- association: 1 if it contains associated, involved, role, linked, implicated, related, regulate, interact, expressed, affect, mediate, correlate
- identity: 1 if it contains same, identical, equivalent, interchangeable, synonym
- comparison: 1 if it contains than, more, less, higher, lower, better, worse, greater
- has_number: 1 if it contains a numeral
- approval: 1 if it contains approved, FDA, EMA, licensed, marketed, indicated
- contradiction: 1 if it contains contradict, disprove, refute, controversial, debated, disputed
- efficacy: 1 if it contains effective, efficacy, beneficial, improve, protective
- prevent: 1 if it contains prevent, cure, eliminate, eradicate
- len_gt_15: 1 if the question has more than 15 tokens
- starts_is_are, starts_does_do, starts_can, starts_has_have: 1 for the respective first word
- two_entities: 1 if the question contains at least two capitalised tokens after the first
