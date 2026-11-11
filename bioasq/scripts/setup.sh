#!/usr/bin/env bash
# One-time setup on a GPU machine. Run from the repo's bioasq/ directory.
set -e
pip install -r requirements.txt --quiet
pip install vllm --quiet || echo "vllm install failed, run_open.py falls back to transformers"
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt_tab','averaged_perceptron_tagger_eng','universal_tagset']]"
mkdir -p data results/runs
[ -f data/training11b.json ] || curl -sL -o data/training11b.json "https://zenodo.org/api/records/7655130/files/training11b.json/content"
echo "fc1fe03831b69157c82a746337c00712  data/training11b.json" | md5sum -c -
# gold test sets (12b, 13b, 14b) go in data/gold/ by hand, from the participants area
ls data/gold/*.json 2>/dev/null && python -m bioasq_audit.data --in data/training11b.json data/gold/*.json --out data/items.jsonl \
  || python -m bioasq_audit.data --in data/training11b.json --out data/items.jsonl
python -m bioasq_audit.corruptions --in data/items.jsonl --out data/corrupted/ --corruptions all
echo "setup done"
