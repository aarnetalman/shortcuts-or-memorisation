"""Prompting hosted models through OpenAI-compatible chat endpoints.

Safety: nothing is sent to a paid API unless the caller passes confirm_spend=True
and a max_usd budget (CLI: --confirm-spend --max-usd). Every request and response
is cached, so reruns cost nothing, and every paid call is logged with its cost.
"""
