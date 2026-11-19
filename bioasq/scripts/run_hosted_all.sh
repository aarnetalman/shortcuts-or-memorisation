#!/usr/bin/env bash
# Tier 2: hosted models through an OpenAI-compatible endpoint. Usage: bash scripts/run_hosted_all.sh MODEL [LIMIT]
# Needs OPENAI_BASE_URL and OPENAI_API_KEY in the environment (Azure AI Foundry endpoint and key).
# Start with gpt-4o only. Run DeepSeek-V4-Pro and Mistral Large 3 after the GPT-4o result is in.
set -e
M=${1:?model name on the endpoint, e.g. gpt-4o}
LIMIT=${2:-0}
CORR=(clean drop-NOUN drop-VERB drop-ADJ drop-DET shuffle-n1 shuffle-n3 keep-NOUN+VERB)
python -m bioasq_audit.run_hosted --model "$M" --items data/corrupted/clean.jsonl --cond noctx --out results/runs/ --limit $LIMIT
for C in "${CORR[@]}"; do
  python -m bioasq_audit.run_hosted --model "$M" --items "data/corrupted/${C}.jsonl" --cond gold --out results/runs/ --limit $LIMIT
done
echo "tier 2 done for $M"
