#!/usr/bin/env bash
# End-to-end smoke test on synthetic BioASQ-format items (no models). Run from repo root.
set -e
python -m bioasq_audit.data --in tests/smoke_bioasq.json --out /tmp/items.jsonl
python -m bioasq_audit.features --in /tmp/items.jsonl --out /tmp/features.jsonl
python -m bioasq_audit.cue_net --features /tmp/features.jsonl --weights weights/yesno_v0.json --out /tmp/cue_preds.jsonl
python -m bioasq_audit.corruptions --in /tmp/items.jsonl --out /tmp/corrupted/ --corruptions clean,noctx,drop-NOUN,keep-NOUN+VERB,shuffle-n1
echo "smoke ok"
