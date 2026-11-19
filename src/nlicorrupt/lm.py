"""Local causal language models: loading, prompt formatting, label scoring and
hidden-state extraction with Hugging Face transformers.

Scoring
-------
Each label has an answer word (prompts.py). The model's score for a label is the
log-probability of its answer tokens after the prompt. When the answer words of
a task start with different tokens (the usual case), only the first token is
scored and one forward pass over the prompt suffices ("first-token" mode);
otherwise every prompt+answer sequence is scored in full ("full" mode). The mode
is recorded with the results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .prompts import chat_messages, template, text_prompt

log = logging.getLogger(__name__)


@dataclass
class LoadedLM:
    model: object
    tokenizer: object
    fmt: str  # "text" or "chat"
    hub_id: str
    device: str


def load_lm(hub_id: str, fmt: str = "auto", dtype: str = "bfloat16", device: str | None = None) -> LoadedLM:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
    from .config import split_revision

    repo, rev = split_revision(hub_id)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch_dtype, device_map=device)
    except (ValueError, KeyError):
        # Multimodal checkpoints (e.g. *ForConditionalGeneration) used text-only.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch_dtype, device_map=device)
    model.eval()
    if fmt == "auto":
        fmt = "chat" if getattr(tok, "chat_template", None) else "text"
    log.info("loaded %s (%s format) on %s", hub_id, fmt, device)
    return LoadedLM(model, tok, fmt, hub_id, device)


# Extra variables for chat templates. Unused variables are ignored by Jinja.
# - enable_thinking=False: Qwen3.x answers directly (empty <think></think>).
# - reasoning_strength / current_date: Muse Glimmer's template otherwise defaults to
#   "high" reasoning and inserts today's date, which would change prompts between runs.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "reasoning_strength": "low", "current_date": "2026-11-01"}

_SENTINEL = "XQZANSWERSENTINEL"


def _render_chat(tok, msgs, gen: bool) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **CHAT_TEMPLATE_KWARGS)


def answer_prefix(tok) -> str:
    """Text a chat template puts between the generation prompt and the answer itself.

    Found by rendering a conversation whose assistant turn is a sentinel. Empty for
    most templates; for Muse Glimmer it selects the direct-to-user channel, so the
    scored answer is not preceded by a reasoning channel.
    """
    cached = getattr(tok, "_nlicorrupt_answer_prefix", None)
    if cached is not None:
        return cached
    user = [{"role": "user", "content": "Q"}]
    gen = _render_chat(tok, user, True)
    full = _render_chat(tok, user + [{"role": "assistant", "content": _SENTINEL}], False)
    prefix = ""
    if full.startswith(gen):
        i = full.find(_SENTINEL, len(gen))
        if i >= 0:
            prefix = full[len(gen) : i]
    tok._nlicorrupt_answer_prefix = prefix
    return prefix


def build_prompt(lm: LoadedLM, task: str, record: dict, exemplars: list[dict] = ()) -> str:
    if lm.fmt == "text":
        return text_prompt(task, record, exemplars)
    msgs = chat_messages(task, record, exemplars)
    return _render_chat(lm.tokenizer, msgs, True) + answer_prefix(lm.tokenizer)


def answer_continuations(lm: LoadedLM, task: str) -> list[list[int]]:
    """Token ids of each answer word as it follows the prompt."""
    words = template(task).answers
    prefix = " " if lm.fmt == "text" else ""
    return [lm.tokenizer(prefix + w, add_special_tokens=False).input_ids for w in words]


def scoring_mode(conts: list[list[int]]) -> str:
    firsts = [c[0] for c in conts]
    return "first-token" if len(set(firsts)) == len(firsts) else "full"


def _encode(lm: LoadedLM, prompts: list[str]):
    add_special = lm.fmt == "text"  # chat templates already contain BOS etc.
    return [lm.tokenizer(p, add_special_tokens=add_special).input_ids for p in prompts]


def _pad_left(seqs: list[list[int]], pad_id: int):
    import torch

    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s) :] = torch.tensor(s)
        mask[i, n - len(s) :] = 1
    return ids, mask


def _forward(lm: LoadedLM, ids, mask, keep: int, hidden: bool = False):
    import torch

    kwargs = {"input_ids": ids.to(lm.device), "attention_mask": mask.to(lm.device)}
    if hidden:
        kwargs["output_hidden_states"] = True
    with torch.inference_mode():
        try:
            return lm.model(**kwargs, logits_to_keep=keep)
        except TypeError:
            return lm.model(**kwargs)


def score_prompts(lm: LoadedLM, task: str, prompts: list[str], batch_size: int = 8) -> tuple[list[list[float]], str]:
    """Log-probability of each answer for each prompt. Returns (scores, mode)."""
    import torch

    conts = answer_continuations(lm, task)
    mode = scoring_mode(conts)
    enc = _encode(lm, prompts)
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
    scores: list[list[float] | None] = [None] * len(enc)
    pad = lm.tokenizer.pad_token_id

    if mode == "first-token":
        first = torch.tensor([c[0] for c in conts])
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            ids, mask = _pad_left([enc[i] for i in idx], pad)
            logits = _forward(lm, ids, mask, keep=1).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)[:, first.to(logits.device)].cpu().tolist()
            for i, row in zip(idx, lp):
                scores[i] = row
        return scores, mode

    # Full mode: score prompt+answer sequences; answers end the sequence (left padding).
    flat = [(i, j) for i in order for j in range(len(conts))]
    per_batch = max(1, batch_size)
    for b in range(0, len(flat), per_batch):
        chunk = flat[b : b + per_batch]
        seqs = [enc[i] + conts[j] for i, j in chunk]
        keep = max(len(conts[j]) for _, j in chunk) + 1
        ids, mask = _pad_left(seqs, pad)
        logits = _forward(lm, ids, mask, keep=keep).logits.float()
        logp = torch.log_softmax(logits, -1)
        T = logp.shape[1]
        for row, (i, j) in enumerate(chunk):
            c = conts[j]
            # token c[k] sits at position (len-len(c)+k); its logit is one step earlier.
            pos = [T - 1 - len(c) + k for k in range(len(c))]
            s = sum(float(logp[row, p, t]) for p, t in zip(pos, c))
            if scores[i] is None:
                scores[i] = [0.0] * len(conts)
            scores[i][j] = s
    return scores, mode


def hidden_states(lm: LoadedLM, prompts: list[str], layers: list[int], batch_size: int = 8):
    """Last-token hidden state of each prompt at the given layers -> array [n, len(layers), d]."""
    import numpy as np

    enc = _encode(lm, prompts)
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
    out = [None] * len(enc)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        ids, mask = _pad_left([enc[i] for i in idx], lm.tokenizer.pad_token_id)
        hs = _forward(lm, ids, mask, keep=1, hidden=True).hidden_states
        stack = np.stack([hs[layer][:, -1, :].float().cpu().numpy() for layer in layers], axis=1)
        for k, i in enumerate(idx):
            out[i] = stack[k].astype(np.float16)
    return np.stack(out)


def default_layers(lm: LoadedLM) -> list[int]:
    cfg = lm.model.config
    n = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")
    return sorted({max(1, round(n * f)) for f in (0.25, 0.5, 0.75)} | {n})


def wrap_prompt(lm: LoadedLM, content: str) -> str:
    """A free-form single-turn prompt in the model's format."""
    if lm.fmt == "text":
        return f"{content}\nAnswer:"
    msgs = [{"role": "user", "content": content}]
    return _render_chat(lm.tokenizer, msgs, True) + answer_prefix(lm.tokenizer)


def generate(lm: LoadedLM, prompts: list[str], max_new_tokens: int = 8, batch_size: int = 8) -> list[str]:
    """Greedy continuations of each prompt (decoded, special tokens removed)."""
    import torch

    enc = _encode(lm, prompts)
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
    out: list[str | None] = [None] * len(enc)
    pad = lm.tokenizer.pad_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        ids, mask = _pad_left([enc[i] for i in idx], pad)
        with torch.inference_mode():
            gen = lm.model.generate(
                input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
            )
        for k, i in enumerate(idx):
            out[i] = lm.tokenizer.decode(gen[k, ids.shape[1]:], skip_special_tokens=True).strip()
    return out


def token_stats(lm: LoadedLM, texts: list[str], batch_size: int = 4):
    """Per text: (log p of each token, mean and std of log p under the model's own
    next-token distribution) for tokens 2..n, as needed by Min-K%++."""
    import numpy as np
    import torch

    enc = [lm.tokenizer(t).input_ids for t in texts]
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
    out = [None] * len(enc)
    pad = lm.tokenizer.pad_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        ids, mask = _pad_left([enc[i] for i in idx], pad)
        logits = _forward(lm, ids, mask, keep=0).logits.float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        mu = (p * logp).sum(-1)
        sigma = ((p * logp.pow(2)).sum(-1) - mu.pow(2)).clamp_min(0).sqrt()
        T = ids.shape[1]
        for k, i in enumerate(idx):
            n = len(enc[i])
            start = T - n  # first real token
            tgt = ids[k, start + 1 :].to(logp.device)
            pos = torch.arange(start, T - 1, device=logp.device)
            lp = logp[k, pos, tgt].cpu().numpy()
            out[i] = (lp, mu[k, pos].cpu().numpy(), sigma[k, pos].cpu().numpy())
    return out
