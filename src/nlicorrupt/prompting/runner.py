"""Plan, cost-check and run prompting experiments on hosted models.

Without confirm_spend the runner never touches the network: it builds every
request, checks the cache, estimates the cost of the uncached ones and stops.
With confirm_spend it needs max_usd and known prices, reserves the worst-case
cost of each request before sending it, and stops before the budget would be
exceeded. Outputs are always written from the cache, so a partial run can be
completed later at no extra cost for the part already done.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import paths
from ..config import load_yaml
from ..data import write_jsonl
from ..metrics import compute
from ..originals import get_records
from ..prompts import chat_messages, parse_answer, select_exemplars, template
from ..runinfo import manifest, write_json
from ..tasks import get_task
from .cache import CostLog, ResponseCache, request_key
from .client import APIError, ChatClient, answer_scores, is_filtered, response_text

log = logging.getLogger(__name__)

# paid APIs see a fixed, label-stratified 2,000-item subsample of these.
DEFAULT_SUBSAMPLE = {"qqp": 2000, "mnli": 2000}
_THINK = re.compile(r"<think>.*?</think>", re.S)


class SpendNotConfirmed(RuntimeError):
    pass


@dataclass
class PromptConfig:
    model: str
    tasks: list[str]
    corruptions: list[str] | None = None  # None: the standard set per task
    style: str = "2022"
    shots: int = 0
    exemplars: str = "clean"
    seed: int = 0
    reasoning: str = "off"  # off: main grid; on: capped sub-study (D8b)
    max_tokens: int | None = None  # default 16 (off) / 4096 (on)
    subsample: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SUBSAMPLE))
    max_examples: int | None = None  # smoke tests
    confirm_spend: bool = False
    max_usd: float | None = None
    concurrency: int = 4
    run_label: str | None = None

    def out_tokens(self) -> int:
        return self.max_tokens or (16 if self.reasoning == "off" else 4096)

    def label(self) -> str:
        return self.run_label or f"{self.model}/{self.style}/k{self.shots}-{self.exemplars}/reasoning-{self.reasoning}"


@dataclass
class Request:
    task: str
    corruption: str
    record: dict
    body: dict
    key: str


def model_entry(alias: str) -> tuple[dict, dict]:
    cfg = load_yaml("providers.yaml")
    try:
        m = cfg["models"][alias]
    except KeyError as e:
        raise KeyError(f"model {alias!r} not in configs/providers.yaml") from e
    return m, cfg["providers"][m["provider"]]


def _subsample_ids(task: str, n: int, seed: int, style: str) -> set[str]:
    from ..probe import stratified_subsample

    spec = get_task(task)
    clean = get_records(task, spec.eval_split, "clean", style)
    return {r["id"] for r in stratified_subsample(clean, n, seed, spec.is_regression)}


def build_requests(cfg: PromptConfig) -> list[Request]:
    from ..corrupt import standard_eval_corruptions

    m, prov = model_entry(cfg.model)
    if m.get("reasoning") == "always" and cfg.reasoning == "off":
        raise ValueError(f"{cfg.model} always reasons (its reasoning cannot be switched off); "
                         "run it with --reasoning on")
    params = dict(m.get("params_off" if cfg.reasoning == "off" else "params_on") or {})
    reqs = []
    for task in cfg.tasks:
        spec = get_task(task)
        corrs = cfg.corruptions or standard_eval_corruptions(spec.is_pair)
        if not spec.is_pair:
            corrs = [c for c in corrs if c not in ("hyp-only", "swap")]
        keep = _subsample_ids(task, cfg.subsample[task], cfg.seed, cfg.style) if task in cfg.subsample else None
        exemplars = []
        if cfg.shots:
            ex_task = spec.exemplar_task or task
            pool = get_records(ex_task, get_task(ex_task).train_split, cfg.exemplars, cfg.style)
            exemplars = select_exemplars(task, pool, cfg.shots, cfg.seed)
        for corr in corrs:
            recs = get_records(task, spec.eval_split, corr, cfg.style)
            if keep is not None:
                recs = [r for r in recs if r["id"] in keep]
            if cfg.max_examples:
                recs = recs[: cfg.max_examples]
            for r in recs:
                body = {
                    "model": m["name"] or f"<unset:{cfg.model}>",
                    "messages": chat_messages(task, r, exemplars),
                    "temperature": 0,
                    "max_tokens": cfg.out_tokens(),
                    "seed": cfg.seed,
                }
                for k, v in params.items():
                    if v is None:
                        body.pop(k, None)
                    else:
                        body[k] = v
                reqs.append(Request(task, corr, r, body, request_key(m["provider"], body)))
    return reqs


def out_tokens(body: dict, default: int) -> int:
    """The request's output-token limit: model params may replace max_tokens with
    max_completion_tokens (reasoning models), which then bounds the cost."""
    return int(body.get("max_completion_tokens") or body.get("max_tokens") or default)


def billed_tokens(usage: dict, est_in: int, est_out: int) -> tuple[int, int]:
    """(input, output) tokens to bill from a response's usage block.

    Providers differ on hidden reasoning tokens: OpenAI counts them inside
    completion_tokens, xAI (grok) reports them only in completion_tokens_details and
    total_tokens. Both are billed as output, so output = the larger of
    completion_tokens and total_tokens - prompt_tokens.
    """
    in_tok = int(usage.get("prompt_tokens", est_in))
    out_tok = int(usage.get("completion_tokens", est_out))
    if usage.get("total_tokens") is not None:
        out_tok = max(out_tok, int(usage["total_tokens"]) - in_tok)
    return in_tok, out_tok


def estimate_tokens(body: dict) -> int:
    chars = sum(len(msg["content"]) for msg in body["messages"])
    return chars // 4 + 8 * len(body["messages"]) + 8


def cost_usd(m: dict, in_tok: int, out_tok: int) -> float:
    return (in_tok * m["usd_per_m_input"] + out_tok * m["usd_per_m_output"]) / 1e6


def plan(cfg: PromptConfig, cache: ResponseCache | None = None) -> dict:
    m, _ = model_entry(cfg.model)
    cache = cache or ResponseCache()
    reqs = build_requests(cfg)
    todo = [r for r in reqs if not cache.has(r.key)]
    in_tok = sum(estimate_tokens(r.body) for r in todo)
    max_out = sum(out_tokens(r.body, cfg.out_tokens()) for r in todo)
    priced = m.get("usd_per_m_input") is not None and m.get("usd_per_m_output") is not None
    summary = {
        "model": cfg.model,
        "provider_model": m.get("name"),
        "requests": len(reqs),
        "cached": len(reqs) - len(todo),
        "to_send": len(todo),
        "est_input_tokens": in_tok,
        "max_output_tokens": max_out,
        "est_usd_expected": cost_usd(m, in_tok, len(todo) * 5) if priced and cfg.reasoning == "off" else None,
        "est_usd_upper_bound": cost_usd(m, in_tok, max_out) if priced else None,
        "prices_known": priced,
    }
    return {"summary": summary, "requests": reqs, "todo": todo}


def execute(cfg: PromptConfig, client: ChatClient | None = None, cache: ResponseCache | None = None,
            costs: CostLog | None = None) -> dict:
    cache = cache or ResponseCache()
    p = plan(cfg, cache)
    s = p["summary"]
    if p["todo"]:
        if not cfg.confirm_spend:
            raise SpendNotConfirmed(
                f"{s['to_send']} uncached requests would be sent (upper bound ${s['est_usd_upper_bound']}). "
                "Nothing was sent. Re-run with --confirm-spend --max-usd N after approval."
            )
        if cfg.max_usd is None:
            raise SpendNotConfirmed("--confirm-spend needs --max-usd")
        if not s["prices_known"]:
            raise SpendNotConfirmed(f"prices for {cfg.model} are not set in configs/providers.yaml")
        m, prov = model_entry(cfg.model)
        if not m.get("name") or not prov.get("base_url"):
            raise SpendNotConfirmed(f"provider model name or base_url for {cfg.model} is not set")
        if client is None:
            key = os.environ.get(prov["api_key_env"])
            if not key:
                raise SpendNotConfirmed(f"environment variable {prov['api_key_env']} is not set")
            client = ChatClient(prov["base_url"], key, auth=prov.get("auth", "bearer"),
                                timeout=float(m.get("timeout") or prov.get("timeout") or 120.0),
                                api_version=prov.get("api_version"),
                                deployment_in_path=bool(prov.get("deployment_in_path")))
        _send(cfg, p["todo"], m, client, cache, costs or CostLog())
    return write_outputs(cfg, p["requests"], cache)


MAX_FAILURES_IN_A_ROW = 20  # transient failures in a row before a run gives up


def _send(cfg, todo, m, client, cache, costs):
    lock = threading.Lock()
    state = {"committed": 0.0, "reserved": 0.0, "stopped": False, "failed": 0, "in_a_row": 0}

    def one(req: Request):
        worst = cost_usd(m, estimate_tokens(req.body) * 2, out_tokens(req.body, cfg.out_tokens()))  # 2x input margin
        with lock:
            if state["stopped"] or state["committed"] + state["reserved"] + worst > cfg.max_usd:
                state["stopped"] = True
                return
            state["reserved"] += worst
        try:
            resp = client.chat(req.body)
        except APIError as e:
            with lock:
                state["reserved"] -= worst
                if not e.transient:
                    raise
                # Skip the item (not cached, so a rerun sends it again) unless the
                # provider looks down: then stop instead of burning through the queue.
                state["failed"] += 1
                state["in_a_row"] += 1
                log.warning("request failed, skipped (%s): %s %s %s", e, req.task, req.corruption, req.record["id"])
                if state["in_a_row"] >= MAX_FAILURES_IN_A_ROW:
                    state["stopped"] = True
                    raise APIError(f"{state['in_a_row']} requests in a row failed; last error: {e}") from e
            return
        except Exception:
            with lock:
                state["reserved"] -= worst
            raise
        usage = resp.get("usage") or {}
        in_tok, out_tok = billed_tokens(usage, estimate_tokens(req.body), out_tokens(req.body, cfg.out_tokens()))
        c = cost_usd(m, in_tok, out_tok)
        with lock:
            state["in_a_row"] = 0
            cache.put(req.key, m["provider"], req.body, resp, c)
            state["reserved"] -= worst
            state["committed"] += c
            costs.add(run=cfg.label(), model=cfg.model, provider_model=m["name"], task=req.task,
                      corruption=req.corruption, id=req.record["id"], input_tokens=in_tok,
                      output_tokens=out_tok, cost_usd=c)

    if cfg.concurrency <= 1:
        for r in todo:
            one(r)
    else:
        # Cache writes are serialised through the lock.
        with ThreadPoolExecutor(cfg.concurrency) as ex:
            list(ex.map(one, todo))
    if state["failed"]:
        log.warning("%d requests failed and were skipped; rerun the same command to send them again",
                    state["failed"])
    if state["stopped"]:
        log.warning("budget of $%.2f reached; stopped early (spent $%.4f)", cfg.max_usd, state["committed"])
    log.info("spent $%.4f this run", state["committed"])


def run_dir(cfg: PromptConfig, task: str) -> Path:
    return (paths.runs_dir() / "prompt" / cfg.model / task / cfg.style /
            (f"k{cfg.shots}-{cfg.exemplars}" if cfg.shots else "k0") / f"reasoning-{cfg.reasoning}")


def write_outputs(cfg: PromptConfig, reqs: list[Request], cache: ResponseCache) -> dict:
    groups: dict[tuple[str, str], list[Request]] = {}
    for r in reqs:
        groups.setdefault((r.task, r.corruption), []).append(r)
    all_metrics: dict[str, dict] = {}
    for (task, corr), rs in groups.items():
        spec = get_task(task)
        rows, golds, preds, missing, filtered = [], [], [], 0, 0
        for r in rs:
            resp = cache.get(r.key)
            if resp is None:
                missing += 1
                continue
            text = response_text(resp)
            pred = parse_answer(task, _THINK.sub("", text or ""))
            blocked = is_filtered(resp)
            filtered += blocked
            row = {"id": r.record["id"], "label": r.record["label"], "response": text,
                   "pred": pred, "usage": resp.get("usage"), "filtered": blocked}
            if not spec.is_regression:
                sc = answer_scores(resp, template(task).answers)
                if sc is not None:
                    row["scores"] = sc
            rows.append(row)
            golds.append(r.record["label"])
            preds.append(pred)
        out = run_dir(cfg, task)
        mpath = out / "metrics.json"
        metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
        n_missing = sum("scores" not in x for x in rows) if any("scores" in x for x in rows) else None
        if rows:
            write_jsonl(out / f"{corr}.jsonl.gz", rows)
            metrics[corr] = compute(spec, golds, preds)
            metrics[corr]["unparsed"] = sum(p is None for p in preds)  # includes filtered
            metrics[corr]["filtered"] = filtered
            if n_missing is not None:
                metrics[corr]["no_scores"] = n_missing
        metrics.setdefault(corr, {})["missing"] = missing
        write_json(mpath, metrics)
        write_json(out / "manifest.json", manifest(asdict(cfg) | {"confirm_spend": cfg.confirm_spend},
                                                   {"example_request": rs[0].body}))
        all_metrics[f"{task}/{corr}"] = metrics[corr]
    return all_metrics
