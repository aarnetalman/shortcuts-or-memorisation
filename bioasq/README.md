# bioasq_audit

Shortcut and corruption audit of BioASQ Task b (yes/no, factoid, list).

```
pip install -r requirements.txt
pytest tests/                      # cue net sanity check, no data needed
python -m bioasq_audit.data --in data/training11b.json --out data/items.jsonl
python -m bioasq_audit.features --in data/items.jsonl --out results/features.jsonl
python -m bioasq_audit.cue_net --features results/features.jsonl --weights weights/yesno_v0.json --out results/cue_preds.jsonl
python -m bioasq_audit.corruptions --in data/items.jsonl --out data/corrupted/
python -m bioasq_audit.run_open --model allenai/OLMo-3-7B --items data/corrupted/clean.jsonl --cond noctx --out results/runs/
python -m bioasq_audit.run_open --model allenai/OLMo-3-7B --items data/corrupted/drop-NOUN.jsonl --cond gold --out results/runs/
python -m bioasq_audit.analysis --results results/runs/ --cue results/cue_preds.jsonl --items data/items.jsonl --out results/tables/
```
