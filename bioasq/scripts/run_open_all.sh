#!/usr/bin/env bash
# Tier 1: open models, all conditions. Usage: bash scripts/run_open_all.sh [LIMIT]
# LIMIT=50 for a dry run. Logs in results/logs/. Resume-safe: skips finished files.
set -e
LIMIT=${1:-0}
MODELS=("allenai/OLMo-3-7B" "allenai/OLMo-3-32B" "Qwen/Qwen3.8-27B")   # check exact hub ids before running
CORR=(clean drop-NOUN drop-VERB drop-ADJ drop-DET shuffle-n1 shuffle-n3 keep-NOUN+VERB)
mkdir -p results/logs
for M in "${MODELS[@]}"; do
  TAG=${M//\//_}
  # noctx first: the parametric-memory floor
  OUT="results/runs/${TAG}__noctx__noctx.jsonl"
  [ -s "$OUT" ] || python -m bioasq_audit.run_open --model "$M" --items data/corrupted/clean.jsonl --cond noctx --out results/runs/ --limit $LIMIT 2>&1 | tee -a results/logs/${TAG}.log
  for C in "${CORR[@]}"; do
    OUT="results/runs/${TAG}__gold__${C}.jsonl"
    [ -s "$OUT" ] || python -m bioasq_audit.run_open --model "$M" --items "data/corrupted/${C}.jsonl" --cond gold --out results/runs/ --limit $LIMIT 2>&1 | tee -a results/logs/${TAG}.log
  done
done
python -m bioasq_audit.analysis --results results/runs/ --cue results/cue_preds.jsonl --items data/items.jsonl --out results/tables/
echo "tier 1 done"
