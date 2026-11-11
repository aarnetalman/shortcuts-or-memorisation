# Shortcuts or Memorisation?

Code and aggregate results for the paper *Shortcuts or Memorisation? Revisiting Data
Corruption in Natural Language Understanding with Large Language Models* (Aarne Talman and
Stergios Chatzikyriakidis). The paper revisits Talman et al. (2021, NoDaLiDa) and Talman
et al. (2022, *SEM) with current models: do models still score well on NLI and GLUE tasks
when the input is perturbed, and is that because of memorisation or of dataset shortcuts?

- `src/nlicorrupt/`: perturbations, fine-tuning, LLM scoring, hosted-model prompting,
  contamination checks, shortcut analysis and the fresh control set pipeline.
- `bioasq/`: the BioASQ yes/no, factoid and list experiments (see `bioasq/README.md`).
- `results/paper/`: the aggregate results behind the paper's tables and figures.
- `configs/`, `scripts/`: model, grid and provider configurations, and the grid runner.

The fresh control items are not included; they will be released separately.

All paths come from environment variables; nothing is hardcoded. The experiments need
one GPU with enough memory for the model at hand (the 27–32B open models were run in
bf16 on 96 GB GPUs); the hosted-model and analysis steps need no GPU.

## Installation

```bash
git clone https://github.com/aarnetalman/shortcuts-or-memorisation.git
cd shortcuts-or-memorisation
# Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
make install                 # pinned requirements and the nlicorrupt package
make test                    # offline tests

# Datasets, NLTK data, model weights and the original 2021/2022 data (needs internet)
nlicorrupt download
nlicorrupt download --model-groups open_models     # weights of the open LLMs

# Tag every split once and write all perturbed versions (2021 and 2022 styles)
nlicorrupt prepare --task all,anli_mix --style 2021,2022
```

### Where things go

| Variable | Default | Contents |
|---|---|---|
| `NLICORRUPT_DATA` | `./work` | tagged and perturbed data, runs, results |
| `NLICORRUPT_ORIG` | `$NLICORRUPT_DATA/originals` | the original 2021/2022 data repositories |
| `NLTK_DATA` | `$NLICORRUPT_DATA/nltk_data` | tokeniser and tagger |
| `HF_HOME` | Hugging Face default | datasets and model weights |
| `NLICORRUPT_PRIVATE` | unset | clone of the fresh control items (not public yet) |

## Experiments

Every experiment is a grid: a YAML file under `configs/grids/` that expands into one
`nlicorrupt` command per line. Each line is an independent run on one GPU, and completed
runs (with a `DONE` marker) are skipped, so a grid can simply be run again.

```bash
nlicorrupt grid grids/<name>.yaml            # writes grids/<name>.txt
bash scripts/run_grid.sh grids/<name>.txt    # runs the lines one after another
```

`scripts/run_grid.sh grids/<name>.txt FIRST LAST` runs only lines FIRST to LAST, so a grid
can be split across GPUs or submitted to any job scheduler, one block of lines per job.

### Reproducing the published numbers

```bash
nlicorrupt grid grids/reproduce.yaml --blocks mrpc-2022,mnli-2022,mnli-2021
bash scripts/run_grid.sh grids/reproduce.txt
```

Targets (MNLI-matched, nouns removed): 2022 RoBERTa-base 72.85 (Corrupt-Test) and
85.64 (Corrupt-Train); 2021 BERT-base-uncased 69.80 and 80.72. Each run evaluates
both our regenerated perturbation and the released file (`released-drop-NOUN`).
Results are in `$NLICORRUPT_DATA/runs/finetune/<model>/<task>/<style>/train-<corruption>/seed<seed>/metrics.json`.

To evaluate a saved clean model on more perturbations later:
`nlicorrupt evaluate-model --model <run dir> --tasks mnli --corruptions standard`.

### Encoders, ANLI and open models

```bash
# Encoders: 4 models x 8 GLUE tasks x (clean + 8 Corrupt-Train sets) x 3 seeds,
# each evaluated on every perturbation; plus the 2021 BERT/MNLI sets incl. AllDrop.
nlicorrupt grid grids/encoders.yaml --blocks glue-small,glue-large,mnli-2021
bash scripts/run_grid.sh grids/encoders.txt

# ANLI (2021): Nie et al.'s released model as clean reference, and RoBERTa-large runs
nlicorrupt grid grids/anli.yaml --blocks reference,train
bash scripts/run_grid.sh grids/anli.txt

# Open models (Qwen3.8-27B, Muse Glimmer 30B, OLMo 3), bf16
nlicorrupt grid grids/open_models.yaml --blocks logprob-zero-shot
bash scripts/run_grid.sh grids/open_models.txt
```

Five-shot runs and probes use the fixed 2,000-item label-stratified sample of QQP and
MNLI (`--eval-items` / `--max-eval`), the same items the paid APIs see; all other tasks,
and all zero-shot runs, use the full evaluation sets. Subsampled runs are stored
separately (`k5-clean-n2000/`, `*.n2000.npz`).

**How open models are scored.** Every system gets the same prompt per task
(`src/nlicorrupt/prompts.py`). Instruction-tuned models get it through their chat
template, with reasoning off (`enable_thinking=False` for Qwen; Glimmer's direct
answer channel with a pinned date and `reasoning_strength`, see `lm.py`). Base models
get plain text ending in `Answer:`. The prediction is the answer word with the highest
log-probability (STS-B: the expected score over 0-5). Results go to
`runs/logprob/<model>/<task>/<style>/k<shots>[-<exemplars>]/`, with per-item scores.

**Probes.** `extract` saves last-token hidden states (layers at 1/4, 1/2, 3/4 and the
top) for a 20k stratified training subsample and all evaluation sets; `probe` trains
logistic-regression probes (ridge for STS-B) on clean or corrupted training states and
tests them on every corruption.

### Hosted models (paid APIs)

**No paid call is made without explicit approval.** `nlicorrupt prompt` is a dry run by
default: it builds every request, checks the cache and prints the number of requests and
an estimated cost, without touching the network. Real calls need `--confirm-spend` and a
hard `--max-usd` budget; in a terminal you also have to type the budget to confirm. The
run stops before the budget would be exceeded.

```bash
export AZURE_AI_API_KEY=$(cat ~/.azure_ai_key)   # never commit keys
# 1. Connection check: one item, one corruption (a fraction of a cent)
nlicorrupt prompt --model azure-mistral-large-3 --tasks rte --corruptions clean --max-examples 1 \
    --confirm-spend --max-usd 0.05
# 2. Dry run: cost estimate only
nlicorrupt prompt --model azure-gpt-4o --tasks rte,mnli,anli_r1,anli_r2,anli_r3 --reasoning off
# 3. After approval only:
nlicorrupt prompt --model azure-gpt-4o --tasks rte,mnli --reasoning off --confirm-spend --max-usd 20
nlicorrupt api-costs                     # spending per run, from the cost log
```

We ran the hosted models on Azure AI Foundry (`configs/providers.yaml`: GPT deployments
through the Azure OpenAI API, DeepSeek, Mistral and Grok through Azure AI model inference;
one key for both); set `base_url` to your own resource, or add another OpenAI-compatible
provider. Prices in the config are unverified third-party
figures: compare the cost log with the Azure cost view after the first runs. Models
marked `reasoning: always` (Grok 4.7, GPT-5.5) refuse `--reasoning off`; their hidden
reasoning tokens are billed as output, so run them on subsets.

These runs are network-bound and need no GPU. QQP and MNLI use the fixed 2,000-item label-stratified subsample
(`--no-subsample` for full sets). Each request is cached in
`$NLICORRUPT_DATA/api/cache.sqlite` with the full prompt and raw response, each paid
call is logged in `$NLICORRUPT_DATA/api/costs.jsonl`, and results go to
`runs/prompt/<model>/<task>/<style>/k<shots>/reasoning-<off|on>/` (raw response, parsed
label, token usage per item; unparseable answers count as wrong and are reported).

### Fresh items and contamination

**Fresh items** live in a separate private repository until publication.
Set `NLICORRUPT_PRIVATE` to a clone of it. The file
`$NLICORRUPT_PRIVATE/fresh/nli.jsonl` holds one item per line (format in
`src/nlicorrupt/fresh.py`; `nlicorrupt validate-fresh` checks it). The items become the
task `fresh_nli`, which goes through the same corruptions, prompts and metrics as the
benchmarks (k-shot exemplars come from MNLI):

```bash
nlicorrupt validate-fresh
nlicorrupt prepare --task fresh_nli
nlicorrupt logprob --model qwen3.8-27b --task fresh_nli       # or through a grid / prompt
nlicorrupt evaluate-model --model <MNLI finetune run dir> --tasks fresh_nli
```

**Contamination indicators.**

```bash
# n-gram counts in OLMo's pre-training data (infini-gram API, needs internet).
# Set infinigram.index in configs/contamination.yaml first.
nlicorrupt contamination-infinigram --tasks mnli,rte,anli_r1,anli_r2,anli_r3,fresh_nli
# Corruption drops on found vs not-found items, per field (needs the logprob runs)
nlicorrupt seen-unseen --index v4_olmo-mix-1124_llama
# Shortcut split: corruption drops on items a hypothesis-only classifier gets right vs wrong
nlicorrupt shortcuts --tasks mnli,anli_r1,anli_r2,anli_r3,rte
# TS-guessing and Min-K%++ with the local open models
nlicorrupt grid grids/contamination.yaml
bash scripts/run_grid.sh grids/contamination.txt
```

The infini-gram check records, for each text field, whether it occurs in OLMo's training
data (a sampled 13-gram, or the whole field if it has at least 13 words; short fields
such as "The man is asleep." occur anywhere). Fields are kept apart: a found premise
usually reflects the benchmark's public source text (Wikipedia for ANLI R1/R2), while a
found hypothesis points to the benchmark itself, since hypotheses were written by
annotators. `seen-unseen` compares corruption drops between found and not-found items
for each field, on answer-bias-corrected predictions, and writes
`results/tables/seen_unseen.csv` (the `gap` rows are found minus not found, with 95% CIs).
**Fresh control items.** Premises come from text published after the models' training
cut-off; a generator model that is not evaluated in the study writes the hypotheses; people
validate every item blind. All files go to `$NLICORRUPT_PRIVATE/fresh/`, never into this
repository.

```bash
# 0. Set premises.user_agent (a contact) and premises.since in configs/fresh.yaml, and the
#    generator deployment(s) as fresh-generator-a/-b in configs/providers.yaml.
# 1. Premises from Wikipedia current-events pages and new articles, minus any found by infini-gram
nlicorrupt fresh-premises --index v4_olmo-mix-1124_llama
# 2. Hypotheses in two styles (mnli, cuefree); dry run first, then with a budget
nlicorrupt fresh-generate --models fresh-generator-a,fresh-generator-b
nlicorrupt fresh-generate --models fresh-generator-a,fresh-generator-b --confirm-spend --max-usd 10
# 2b. Naturally occurring pairs, labelled from scratch: Global Voices headline + lead,
#     and Wikipedia sentences before and after an edit that changed a fact
nlicorrupt fresh-natural --kinds headline,revision --per-kind 100
# 3. Optional hand-written items: a template per author, filled in a spreadsheet
nlicorrupt fresh-sheets --template 30 --author AT
# 4. Blind validation sheets, one per annotator (labels hidden, order shuffled)
nlicorrupt fresh-sheets --annotators AT,SC
nlicorrupt fresh-artefacts            # hypothesis-only cues per style, vs ~54% on MNLI
# 5. After the sheets are filled: keep items every annotator labels as intended
#    (natural pairs: items all annotators give the same label)
nlicorrupt fresh-merge                # writes fresh/nli.jsonl, the fresh_nli task
```

Fragments-style synthetic items are not included yet; their role in this paper is
still open.

### Results

```bash
nlicorrupt results          # -> $NLICORRUPT_DATA/results/
```

This collects every run into `all_results.csv` (one row per metric) and writes:

| File | Content |
|---|---|
| `tables/corrupt_test.csv` | Δ (corrupted − clean) per system, task and corruption; mean and sd over seeds |
| `tables/corrupt_train.csv` | encoders: Corrupt-Train and Corrupt-Train-and-Test Δ |
| `tables/reproduction.csv` | our numbers next to the published 2021/2022 ones (`configs/paper_targets.yaml`) |
| `tables/benchmark_vs_fresh.csv` | mean Δ on MNLI/ANLI vs on the fresh items, per system |
| `tables/answer_bias.csv` | open models: accuracy, accuracy after removing a constant answer preference, threshold-free AUC and the share of the most frequent answer, per corruption (see `analysis/answer_bias.py`) |
| `figures/*.png`, `*.pdf` | Δ heatmaps (tasks × corruptions), one per system |

`notebooks/results.ipynb` explores these tables, including the seen/unseen comparison
for OLMo.

## Perturbations

`nlicorrupt corruptions` lists the standard set. Names:

| Name | Meaning | Source |
|---|---|---|
| `clean` | original text | |
| `drop-NOUN` (ADJ, ADV, CONJ, DET, NOUN, NUM, PRON, VERB) | remove all words of a universal POS class | 2021, 2022 |
| `drop-NOUN+PRON` | remove several classes | 2021 |
| `keep-NOUN+VERB` etc. | keep only the listed classes | 2021 |
| `shuffle-n1`, `-n2`, `-n3` | shuffle n-grams within each text, premise and hypothesis independently | 2021 (Pham et al. 2020) |
| `randdrop-NOUN` | remove as many random tokens as `drop-NOUN` would | new length control |
| `hyp-only`, `swap` | empty the premise; swap the two texts | recommended in 2021 |

Two styles: `2022` (detokenised, empty texts kept; all new experiments) and `2021`
(space-joined, pairs with an empty side dropped; only to replicate 2021 numbers).

**Agreement with the released files.** With NLTK 3.10, our `drop-NOUN` output matches
the released 2021 MNLI Corrupt-Test file in row count (9,769) and exactly on 98.4% of
rows (99.1% ignoring spacing and quote style). For the 2022 files, the same words are
removed for 85–100% of items depending on the task (CoLA 100%, SST-2 99%, RTE and STS-B 96%,
MRPC 85%); the differences come from changes in NLTK's quote handling and detokeniser since
2022, and for MRPC from escaped quotes in the dataset version used then. Replication runs can use
the released files directly (`nlicorrupt download` clones them).

## Tests

```bash
make test        # offline tests
make test-all    # also tests that download from the Hugging Face hub
```
