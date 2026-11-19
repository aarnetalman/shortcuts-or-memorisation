"""Prompting harness against a local mock server. No external network is used."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nlicorrupt.cli import main
from nlicorrupt.data import read_jsonl
from nlicorrupt.prompting import runner
from nlicorrupt.prompting.cache import CostLog, ResponseCache
from nlicorrupt.prompting.client import ChatClient


class MockHandler(BaseHTTPRequestHandler):
    calls = []
    headers_seen = []
    filter_word = None  # prompts containing it get Azure's content-filter HTTP 400
    fail_word = None  # prompts containing it get HTTP 503 (a transient failure)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockHandler.calls.append(body)
        MockHandler.headers_seen.append(dict(self.headers))
        prompt = body["messages"][-1]["content"]
        if MockHandler.fail_word and MockHandler.fail_word in prompt:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if MockHandler.filter_word and MockHandler.filter_word in prompt:
            data = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                            "finish_reason": "content_filter"}],
                               "usage": {"prompt_tokens": 40, "total_tokens": 40}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        answer = "<think>maybe neutral</think>Contradiction." if "cat" in prompt else "entailment"
        resp = {"choices": [{"message": {"role": "assistant", "content": answer}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 3}}
        if body.get("logprobs"):  # the right answer leads, but "Ent" always gets some mass
            first = "Contr" if "cat" in prompt else "ent"
            resp["choices"][0]["logprobs"] = {"content": [{"token": first, "top_logprobs": [
                {"token": first, "logprob": -0.2}, {"token": " Ent", "logprob": -1.5},
                {"token": "maybe", "logprob": -3.0}]}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    MockHandler.calls = []
    MockHandler.headers_seen = []
    MockHandler.filter_word = None
    MockHandler.fail_word = None
    httpd = HTTPServer(("127.0.0.1", 0), MockHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield ChatClient(f"http://127.0.0.1:{httpd.server_port}/v1", "test-key", max_retries=0)
    httpd.shutdown()


def fake_records(task, split, corruption, style):
    from nlicorrupt.tasks import get_task

    k1, k2 = get_task(task).keys
    n = 20 if split.startswith("train") else 10
    return [{"id": f"{task}:{split}:{i}", "label": 2 if i % 2 else 0,
             k1: ("a cat" if i % 2 else "a dog") + f" number {i}" + (" sat" if corruption == "clean" else ""),
             k2: "an animal"} for i in range(n)]


@pytest.fixture(autouse=True)
def _fake_data(monkeypatch):
    monkeypatch.setattr(runner, "get_records", fake_records)


def cfg(**kw):
    base = dict(model="mock-model", tasks=["rte"], corruptions=["clean", "drop-NOUN"], subsample={})
    base.update(kw)
    return runner.PromptConfig(**base)


def test_dry_run_sends_nothing(server):
    p = runner.plan(cfg())
    assert p["summary"]["to_send"] == 20 and p["summary"]["est_usd_upper_bound"] > 0
    with pytest.raises(runner.SpendNotConfirmed):
        runner.execute(cfg(), client=server)
    with pytest.raises(runner.SpendNotConfirmed, match="max-usd"):
        runner.execute(cfg(confirm_spend=True), client=server)
    assert MockHandler.calls == []


def test_unpriced_model_is_refused(server):
    with pytest.raises(runner.SpendNotConfirmed, match="prices"):
        runner.execute(cfg(model="kimi-k3", confirm_spend=True, max_usd=1.0), client=server)
    assert MockHandler.calls == []


def test_confirmed_run_caches_logs_and_scores(server):
    c = cfg(tasks=["mnli"], confirm_spend=True, max_usd=1.0, concurrency=2)
    metrics = runner.execute(c, client=server)
    assert len(MockHandler.calls) == 20
    body = MockHandler.calls[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 16 and body["model"] == "mock-model"
    # "cat" items are label 2 and get "Contradiction" (after stripping <think>), others label 0 / "entailment".
    assert metrics["mnli/clean"]["accuracy"] == 100.0 and metrics["mnli/clean"]["unparsed"] == 0
    assert CostLog().total() == pytest.approx(20 * (50 * 1.0 + 3 * 2.0) / 1e6)

    # Everything is cached: a rerun without confirmation sends nothing and still writes outputs.
    again = runner.execute(cfg(tasks=["mnli"]), client=server)
    assert len(MockHandler.calls) == 20 and again == metrics


def test_budget_stops_early(server):
    one_call = runner.cost_usd(runner.model_entry("mock-model")[0], 200, 16)
    c = cfg(confirm_spend=True, max_usd=one_call * 3.5, concurrency=1)
    metrics = runner.execute(c, client=server)
    assert 1 <= len(MockHandler.calls) < 20
    assert sum(m.get("missing", 0) for m in metrics.values()) == 20 - len(MockHandler.calls)


def test_subsample_and_reasoning_params(monkeypatch):
    c = cfg(tasks=["rte"], subsample={"rte": 4}, reasoning="on")
    reqs = runner.build_requests(c)
    assert len({r.record["id"] for r in reqs}) == 4 and len(reqs) == 8
    assert reqs[0].body["max_tokens"] == 4096
    assert len({r.key for r in reqs}) == len(reqs)


def test_cli_prompt_is_dry_run_by_default(capsys):
    assert main(["prompt", "--model", "mock-model", "--tasks", "rte", "--corruptions", "clean"]) == 0
    out = capsys.readouterr()
    assert '"to_send": 10' in out.out and "DRY RUN" in out.err
    assert ResponseCache().db.execute("select count(*) from responses").fetchone()[0] == 0


def test_azure_key_goes_in_api_key_header(server):
    server.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert MockHandler.headers_seen[-1].get("Authorization") == "Bearer test-key"
    azure = ChatClient(server.base_url, "az-key", max_retries=0, auth="api-key")
    azure.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    h = {k.lower(): v for k, v in MockHandler.headers_seen[-1].items()}  # header names are case-insensitive
    assert h.get("api-key") == "az-key" and "authorization" not in h


def test_azure_urls():
    body = {"model": "gpt-4o", "messages": []}
    oai = ChatClient("https://r.openai.azure.com", "k", api_version="2025-04-01-preview", deployment_in_path=True)
    assert oai.url_for(body) == ("https://r.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
                                 "?api-version=2025-04-01-preview")
    inf = ChatClient("https://r.services.ai.azure.com/models", "k", api_version="2024-05-01-preview")
    assert inf.url_for(body) == "https://r.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
    assert ChatClient("https://x/v1", "k").url_for(body) == "https://x/v1/chat/completions"


def test_azure_models_are_configured():
    for alias in ("azure-gpt-4o", "azure-deepseek-v4-pro", "azure-mistral-large-3", "azure-grok-4.7"):
        m, prov = runner.model_entry(alias)
        assert m["name"] and prov["base_url"].startswith("https://")
        assert prov["api_key_env"] == "AZURE_AI_API_KEY" and prov["api_version"]


def test_always_reasoning_model_refuses_reasoning_off_and_budgets_its_tokens():
    with pytest.raises(ValueError, match="--reasoning on"):
        runner.build_requests(cfg(model="azure-grok-4.7"))
    reqs = runner.build_requests(cfg(model="azure-grok-4.7", reasoning="on"))
    b = reqs[0].body
    assert "max_tokens" not in b and "temperature" not in b and b["max_completion_tokens"] == 2048
    assert "seed" not in b
    s = runner.plan(cfg(model="azure-grok-4.7", reasoning="on"))["summary"]
    assert s["max_output_tokens"] == 2048 * s["to_send"] and s["est_usd_expected"] is None


def test_hidden_reasoning_tokens_are_billed():
    grok = {"prompt_tokens": 62, "completion_tokens": 1, "total_tokens": 362,
            "completion_tokens_details": {"reasoning_tokens": 299}}
    assert runner.billed_tokens(grok, 80, 16) == (62, 300)
    gpt = {"prompt_tokens": 67, "completion_tokens": 1, "total_tokens": 68}
    assert runner.billed_tokens(gpt, 80, 16) == (67, 1)
    o_series = {"prompt_tokens": 50, "completion_tokens": 400, "total_tokens": 450,
                "completion_tokens_details": {"reasoning_tokens": 399}}
    assert runner.billed_tokens(o_series, 80, 16) == (50, 400)  # already included: not counted twice
    assert runner.billed_tokens({}, 80, 16) == (80, 16)


def test_content_filter_refusal_is_recorded_not_fatal(server):
    MockHandler.filter_word = "number 3"
    metrics = runner.execute(cfg(tasks=["mnli"], corruptions=["clean"], confirm_spend=True, max_usd=1.0,
                                 concurrency=1), client=server)
    m = metrics["mnli/clean"]
    assert m["filtered"] == 1 and m["unparsed"] == 1 and m["accuracy"] == 90.0
    n = len(MockHandler.calls)
    runner.execute(cfg(tasks=["mnli"], corruptions=["clean"]), client=server)
    assert len(MockHandler.calls) == n  # the refusal is cached, not re-sent


def test_answer_scores_from_top_logprobs():
    import math

    from nlicorrupt.prompting.client import answer_scores

    resp = {"choices": [{"logprobs": {"content": [{"token": "\n", "top_logprobs": []}, {"token": "Yes",
            "top_logprobs": [{"token": "Yes", "logprob": -0.1}, {"token": " yes", "logprob": -2.0},
                             {"token": "No", "logprob": -2.5}, {"token": "Maybe", "logprob": -6.0}]}]}}]}
    sc = answer_scores(resp, ("yes", "no"))
    assert sc[0] == pytest.approx(math.log(math.exp(-0.1) + math.exp(-2.0))) and sc[1] == -2.5
    nli = {"choices": [{"logprobs": {"content": [{"token": "neutral", "top_logprobs": [
           {"token": "neutral", "logprob": -0.3}, {"token": "e", "logprob": -4.0}]}]}}]}
    # "e" is a prefix of entailment only; contradiction is missing and gets the floor
    assert answer_scores(nli, ("entailment", "neutral", "contradiction")) == [-4.0, -0.3, -4.0]
    assert answer_scores({"choices": [{"message": {}}]}, ("yes", "no")) is None


def test_logprob_run_writes_scores_for_answer_bias(server, tmp_path):
    from nlicorrupt.analysis.answer_bias import collect

    MockHandler.filter_word = "number 3"
    runner.execute(cfg(model="mock-model-logprobs", tasks=["mnli"], corruptions=["clean"], confirm_spend=True,
                       max_usd=1.0, concurrency=1), client=server)
    assert MockHandler.calls[0]["logprobs"] is True and MockHandler.calls[0]["top_logprobs"] == 5
    out = runner.run_dir(cfg(model="mock-model-logprobs", tasks=["mnli"]), "mnli")
    rows = list(read_jsonl(out / "clean.jsonl.gz"))
    assert sum("scores" in r for r in rows) == 9  # the filtered item has none
    assert json.loads((out / "metrics.json").read_text())["clean"]["no_scores"] == 1
    ab = collect()
    row = ab[(ab.model == "mock-model-logprobs") & (ab.corruption == "clean")].iloc[0]
    assert row["centered_accuracy"] == 90.0  # the filtered item counts as wrong
    # (no 3-way AUC: the fake items use only two of the three labels)


def test_transient_failure_skips_the_item_and_a_rerun_sends_it(server):
    MockHandler.fail_word = "number 3"
    c = cfg(tasks=["mnli"], corruptions=["clean"], confirm_spend=True, max_usd=1.0, concurrency=1)
    m = runner.execute(c, client=server)["mnli/clean"]
    assert m["missing"] == 1 and m["n"] == 9  # the run finished; one item is left for later
    MockHandler.fail_word = None
    m = runner.execute(c, client=server)["mnli/clean"]
    assert m["missing"] == 0 and m["n"] == 10


def test_many_failures_in_a_row_stop_the_run(server, monkeypatch):
    from nlicorrupt.prompting.client import APIError

    monkeypatch.setattr(runner, "MAX_FAILURES_IN_A_ROW", 3)
    MockHandler.fail_word = "number"  # every item fails
    with pytest.raises(APIError, match="in a row"):
        runner.execute(cfg(tasks=["mnli"], corruptions=["clean"], confirm_spend=True, max_usd=1.0,
                           concurrency=1), client=server)
