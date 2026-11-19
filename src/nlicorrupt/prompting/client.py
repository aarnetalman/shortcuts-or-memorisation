"""Minimal OpenAI-compatible chat client (standard library only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


class APIError(RuntimeError):
    """A request that failed. transient=True means it failed only after retries on errors
    that usually pass (timeouts, rate limits, server errors): the run can skip the item
    and a rerun will send it again. Other errors (bad request, authentication) stop the run."""

    def __init__(self, msg: str, transient: bool = False):
        super().__init__(msg)
        self.transient = transient


@dataclass
class ChatClient:
    base_url: str
    api_key: str
    timeout: float = 120.0
    max_retries: int = 6
    auth: str = "bearer"  # "bearer" (Authorization: Bearer <key>) or "api-key" (Azure: api-key: <key>)
    api_version: str | None = None  # Azure: sent as ?api-version=...
    deployment_in_path: bool = False  # Azure OpenAI: /openai/deployments/<model>/chat/completions

    def url_for(self, body: dict) -> str:
        base = self.base_url.rstrip("/")
        if self.deployment_in_path:
            url = f"{base}/openai/deployments/{urllib.parse.quote(body['model'], safe='')}/chat/completions"
        else:
            url = base + "/chat/completions"
        if self.api_version:
            url += "?" + urllib.parse.urlencode({"api-version": self.api_version})
        return url

    def chat(self, body: dict) -> dict:
        url = self.url_for(body)
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.auth == "api-key":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                if e.code == 400 and "content_filter" in raw:
                    return filtered_response(raw)
                detail = raw[:500]
                retryable = e.code in (408, 409, 429) or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise APIError(f"HTTP {e.code}: {detail}", transient=retryable) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise APIError(str(e), transient=True) from e
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise APIError("unreachable")


def filtered_response(raw: str) -> dict:
    """A content-filter refusal (Azure returns HTTP 400 when the prompt or the answer is
    blocked) as an ordinary response with no answer, so the item is recorded, cached and
    counted instead of aborting the run. Benchmark text sometimes trips these filters."""
    try:
        body = json.loads(raw)
    except ValueError:
        body = {}
    usage = body.get("usage") or {}
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                         "finish_reason": "content_filter"}],
            "usage": usage, "content_filter": True, "filter_detail": raw[:1000]}


def is_filtered(resp: dict) -> bool:
    if resp.get("content_filter"):
        return True
    return any(c.get("finish_reason") == "content_filter" for c in resp.get("choices") or [])


def response_text(resp: dict) -> str | None:
    """Final answer text; reasoning traces returned in separate fields are ignored."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    content = msg.get("content")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def answer_scores(resp: dict, answers: tuple[str, ...]) -> list[float] | None:
    """Per-answer log-probabilities from the first answer token's top_logprobs, if the
    response has them (request with logprobs: true, top_logprobs: N).

    A candidate token counts for an answer if, stripped and lower-cased, it is a prefix of
    exactly one answer word ("Yes", " yes", "ent" -> entailment); probabilities of several
    such tokens are summed. An answer missing from the top list gets the lowest returned
    log-probability, an upper bound on its true value. None without logprobs.
    """
    import math

    try:
        content = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    first = next((c for c in content or [] if (c.get("token") or "").strip()), None)
    if first is None or not first.get("top_logprobs"):
        return None
    tops = first["top_logprobs"]
    probs = [0.0] * len(answers)
    found = [False] * len(answers)
    for t in tops:
        tok = (t.get("token") or "").strip().lower()
        if not tok:
            continue
        hits = [i for i, a in enumerate(answers) if a.startswith(tok)]
        if len(hits) == 1:
            probs[hits[0]] += math.exp(t["logprob"])
            found[hits[0]] = True
    if not any(found):
        return None
    floor = min(t["logprob"] for t in tops)
    return [math.log(p) if f else floor for p, f in zip(probs, found)]
