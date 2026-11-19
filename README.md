# Shortcuts or Memorisation?

Code and aggregate results for *Shortcuts or Memorisation? Revisiting Data Corruption in
Natural Language Understanding with Large Language Models* (Aarne Talman and Stergios
Chatzikyriakidis), [arXiv:2611.16843](https://arxiv.org/abs/2611.16843).

The paper revisits [Talman et al. (2021)](https://aclanthology.org/2021.nodalida-main.28/) and
[Talman et al. (2022)](https://aclanthology.org/2022.starsem-1.20/): do models still score well
on NLI and GLUE tasks when the input is perturbed, for example when all nouns are removed or
the words are shuffled? And if they do, is it memorisation or dataset shortcuts?

## Contents

| Folder | What it holds |
|---|---|
| `src/nlicorrupt/` | perturbations, fine-tuning, LLM scoring, hosted models, contamination and shortcut analyses |
| `bioasq/` | the BioASQ experiments (see `bioasq/README.md`) |
| `results/paper/` | aggregate results behind the paper's tables and figures |
| `configs/` | models, experiment grids and API providers |
| `scripts/run_grid.sh` | runs an experiment grid |

The fresh control items are not included yet; they will be released separately.

## Installation

```bash
git clone https://github.com/aarnetalman/shortcuts-or-memorisation.git
cd shortcuts-or-memorisation
# install PyTorch for your platform first: https://pytorch.org/get-started/locally/
make install
make test

nlicorrupt download                                # datasets, NLTK data, original 2021/2022 data
nlicorrupt download --model-groups open_models     # open LLM weights
nlicorrupt prepare --task all,anli_mix --style 2021,2022   # write all perturbed versions
```

Data, runs and results go to `$NLICORRUPT_DATA` (default `./work`). Each run needs one GPU;
the 27–32B models need about 96 GB in bf16.

## Running experiments

Each experiment is a grid in `configs/grids/`, which expands into one command per line:

```bash
nlicorrupt grid grids/reproduce.yaml          # also: encoders, anli, open_models, contamination
bash scripts/run_grid.sh grids/reproduce.txt  # or lines FIRST LAST, to split across GPUs
```

Finished runs are skipped, so a grid can simply be run again.

**Hosted models.** Configure the endpoints and models in `configs/providers.yaml` and
`configs/models.yaml`. `nlicorrupt prompt` only estimates the cost unless you add
`--confirm-spend` and a budget:

```bash
nlicorrupt prompt --model azure-gpt-4o --tasks rte,mnli                            # cost estimate
nlicorrupt prompt --model azure-gpt-4o --tasks rte,mnli --confirm-spend --max-usd 20
```

**Contamination and shortcuts.**

```bash
nlicorrupt contamination-infinigram --tasks mnli,rte,anli_r1,anli_r2,anli_r3   # n-gram counts
nlicorrupt seen-unseen --index v4_olmo-mix-1124_llama                          # found vs not found
nlicorrupt shortcuts --tasks mnli,anli_r1,anli_r2,anli_r3,rte                   # hypothesis-only split
```

**Results.** `nlicorrupt results` collects all runs into tables and figures in
`$NLICORRUPT_DATA/results/`; `notebooks/results.ipynb` explores them.

## Reproducing the paper

Run these steps in order after the installation above. Each step names the part of the
paper it reproduces. `results/paper/` holds our aggregate numbers to compare against.

1. **Earlier encoder results** (Appendix A):
   ```bash
   nlicorrupt grid grids/reproduce.yaml --blocks mrpc-2022,mnli-2022,mnli-2021
   bash scripts/run_grid.sh grids/reproduce.txt
   ```
2. **Fine-tuned encoders and the ANLI reference model** (Table 1, Appendix):
   ```bash
   nlicorrupt grid grids/encoders.yaml --blocks glue-small,glue-large,mnli-2021
   bash scripts/run_grid.sh grids/encoders.txt
   nlicorrupt grid grids/anli.yaml --blocks reference
   bash scripts/run_grid.sh grids/anli.txt
   ```
3. **Open LLMs, zero-shot and five-shot** (Table 1, Section 4, Appendix):
   ```bash
   nlicorrupt grid grids/open_models.yaml --blocks logprob-zero-shot,logprob-five-shot
   bash scripts/run_grid.sh grids/open_models.txt
   ```
4. **Hosted models** (Table 1, Section 4.4). This step makes paid calls. Run each command
   without `--confirm-spend` first to see the cost estimate. We used the core set of ten
   perturbations; MNLI and QQP use a fixed 2,000-item sample:
   ```bash
   CORE=clean,drop-NOUN,drop-VERB,drop-ADJ,drop-DET,drop-PRON,shuffle-n1,shuffle-n3,keep-NOUN+VERB,hyp-only
   for m in azure-gpt-4o azure-deepseek-v4-pro azure-mistral-large-3; do
     nlicorrupt prompt --model $m --tasks mnli,rte,anli_r1,anli_r2,anli_r3 --corruptions $CORE \
       --reasoning off --confirm-spend --max-usd 30
   done
   # reasoning model: five perturbations, first 400 items per task
   nlicorrupt prompt --model azure-grok-4.7 --tasks mnli,rte,anli_r1,anli_r2,anli_r3 \
     --corruptions clean,drop-NOUN,drop-VERB,shuffle-n1,hyp-only --max-examples 400 \
     --reasoning on --confirm-spend --max-usd 40
   ```
5. **Contamination indicators and the shortcut split** (Section 5.1–5.3):
   ```bash
   nlicorrupt grid grids/contamination.yaml && bash scripts/run_grid.sh grids/contamination.txt
   nlicorrupt contamination-infinigram --tasks mnli,rte,anli_r1,anli_r2,anli_r3
   nlicorrupt seen-unseen --index v4_olmo-mix-1124_llama
   nlicorrupt shortcuts --tasks mnli,anli_r1,anli_r2,anli_r3,rte
   ```
6. **Tables and figures:**
   ```bash
   nlicorrupt results        # writes $NLICORRUPT_DATA/results/
   ```
7. **BioASQ** (Section 5.5, Appendix): follow `bioasq/README.md`.

The fresh-item experiments (Section 5.4) need the fresh items, which will be released
separately. Once you have them, set `NLICORRUPT_PRIVATE` to their folder, run
`nlicorrupt prepare --task fresh_nli`, and score them with
`nlicorrupt logprob --model <open model> --task fresh_nli` and
`nlicorrupt prompt --model <hosted model> --tasks fresh_nli`; `nlicorrupt fresh-results` then
summarises them.

Exact numbers can differ slightly from ours: across GPUs and library versions for the open
models, and because hosted providers update the models behind their endpoints.

## Perturbations

| Name | Meaning |
|---|---|
| `drop-X` | remove all words of a word class, e.g. `drop-NOUN` |
| `keep-X+Y` | keep only the listed classes, e.g. `keep-NOUN+VERB` |
| `shuffle-n1`, `-n2`, `-n3` | shuffle words or n-grams within each text |
| `randdrop-NOUN` | remove as many random words as `drop-NOUN` would |
| `hyp-only`, `swap` | remove the premise; swap premise and hypothesis |

`nlicorrupt corruptions` lists them all. The `2022` style is used for all new experiments,
the `2021` style only to reproduce the 2021 numbers.

## Citation

```bibtex
@misc{talman-chatzikyriakidis-2026-shortcuts,
  title         = {Shortcuts or Memorisation? Revisiting Data Corruption in Natural Language Understanding with Large Language Models},
  author        = {Talman, Aarne and Chatzikyriakidis, Stergios},
  year          = {2026},
  eprint        = {2611.16843},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.16843}
}
```

The earlier papers this work revisits:

```bibtex
@inproceedings{talman-etal-2021-nli,
  title     = {{NLI} Data Sanity Check: Assessing the Effect of Data Corruption on Model Performance},
  author    = {Talman, Aarne and Apidianaki, Marianna and Chatzikyriakidis, Stergios and Tiedemann, J{\"o}rg},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  year      = {2021},
  pages     = {276--287},
  url       = {https://aclanthology.org/2021.nodalida-main.28/}
}

@inproceedings{talman-etal-2022-data,
  title     = {How Does Data Corruption Affect Natural Language Understanding Models? A Study on {GLUE} datasets},
  author    = {Talman, Aarne and Apidianaki, Marianna and Chatzikyriakidis, Stergios and Tiedemann, J{\"o}rg},
  booktitle = {Proceedings of the 11th Joint Conference on Lexical and Computational Semantics},
  year      = {2022},
  pages     = {226--233},
  doi       = {10.18653/v1/2022.starsem-1.20},
  url       = {https://aclanthology.org/2022.starsem-1.20/}
}
```

## Licence

MIT.
