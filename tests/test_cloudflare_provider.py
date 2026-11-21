"""Local-only Cloudflare Workers AI provider tests (gitignored until experiment integration)."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "NSGA-III"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from providers.cloudflare import LLAMA, MISTRAL
from providers.cloudflare_pool import CloudflarePool
from providers.settings import create_decision_client, load_experiment_settings, resolve_model_profile
from run_experiment import commands

ACCOUNT_A = "a" * 32
ACCOUNT_B = "b" * 32
NO_TARGET = {
    "solution_id": None, "region_id": None, "boundary_id": None,
    "relevant_constraint_ids": [],
}
PROFILES = {
    LLAMA: "cloudflare_llama",
    MISTRAL: "cloudflare_mistral",
}


def model_settings(model):
    return resolve_model_profile(PROFILES[model])


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc).timestamp()
        self.on_sleep = None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()


def add_pair(path, slot, account, token):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"CLOUDFLARE_ACCOUNT_ID_{slot:03d}={account}\n")
        stream.write(f"CLOUDFLARE_API_TOKEN_{slot:03d}={token}\n")


@pytest.fixture
def setup(tmp_path):
    env = tmp_path / ".env"
    usage = tmp_path / "usage.json"
    env.write_text("# Gemini entries are ignored by the Cloudflare pool.\nGEMINI_API_KEY_001=example\n",
                   encoding="utf-8")
    add_pair(env, 1, ACCOUNT_A, "fake-token-a")
    clock = Clock()
    return env, usage, clock


def client(setup, model, handler, *, pace=0):
    env, usage, clock = setup
    pool = CloudflarePool(env, usage, clock=clock.time, monotonic=clock.time, sleep=clock.sleep)
    profile = {**model_settings(model), "min_request_interval_seconds": pace}
    return create_decision_client(profile, free_tier_confirmed=True, pool=pool,
                                  transport=httpx.MockTransport(handler))


def ok(answer=NO_TARGET):
    return httpx.Response(200, json={"success": True, "result": {"response": answer, "usage": {
        "prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26,
    }}})


def quota():
    return httpx.Response(429, json={"success": False, "errors": [{"code": 3036}]})


def read_state(setup):
    return json.loads(setup[1].read_bytes())


def test_local_cloudflare_profiles_integrate_with_runner():
    exp = load_experiment_settings()
    for name, expected_model in (("cloudflare_llama", LLAMA), ("cloudflare_mistral", MISTRAL)):
        resolved = resolve_model_profile(name, experiment_settings=exp)
        assert resolved["provider"] == "cloudflare" and resolved["model"] == expected_model
        cmd = commands(exp, pipeline="single", seed=17, instances=["wbg-fal10"],
                       model=name, output=None, confirm_free_tier=True)[0]
        assert cmd[cmd.index("--provider") + 1] == "cloudflare"
        assert cmd[cmd.index("--model") + 1] == expected_model
        assert "--thinking-level" not in cmd


def test_shared_account_quota_and_model_formats(setup):
    env, usage, _ = setup
    add_pair(env, 2, ACCOUNT_A, "fake-token-duplicate-account")
    add_pair(env, 3, ACCOUNT_B, "fake-token-b")
    llama_requests = []
    mistral_requests = []

    def llama_handler(request):
        llama_requests.append(request)
        return quota() if len(llama_requests) == 1 else ok()

    def mistral_handler(request):
        mistral_requests.append(request)
        return ok(json.dumps(NO_TARGET))

    llama = client(setup, LLAMA, llama_handler)
    mistral = client(setup, MISTRAL, mistral_handler)
    try:
        action, record = llama.request("inspector", {"state": []})
        assert action == NO_TARGET
        assert record["api_requests"] == 1
        assert record["http_requests_by_key_slot"] == {"1": 1, "3": 1}
        assert record["total_tokens"] == 26
        assert [request.headers["Authorization"] for request in llama_requests] == [
            "Bearer fake-token-a", "Bearer fake-token-b"]
        assert llama_requests[0].content == llama_requests[1].content
        llama_body = json.loads(llama_requests[0].content)
        assert llama_body["response_format"]["type"] == "json_schema"
        assert llama_body["max_tokens"] == 4096
        assert llama_requests[0].url.path.endswith("/ai/run/" + LLAMA)
        assert mistral.request("inspector", {"state": []})[0] == NO_TARGET
        assert mistral_requests[0].headers["Authorization"] == "Bearer fake-token-b"
        assert f"/{ACCOUNT_B}/ai/run/{MISTRAL}" in mistral_requests[0].url.path
        mistral_body = json.loads(mistral_requests[0].content)
        assert "response_format" not in mistral_body
        assert "exactly one JSON object" in mistral_body["messages"][0]["content"]
    finally:
        llama.close()
        mistral.close()
    assert "fake-token" not in usage.read_text()
    assert ACCOUNT_A not in usage.read_text()


def test_transient_retry_rotates_after_three_and_reuses_after_cooldown(setup):
    env, _, _ = setup
    add_pair(env, 2, ACCOUNT_B, "fake-token-b")
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            raise httpx.ReadTimeout("synthetic error")
        if len(seen) <= 3:
            return httpx.Response(429, json={"errors": [{"code": 3040}]})
        if len(seen) == 4:
            return quota()
        return ok()

    adapter = client(setup, LLAMA, handler)
    try:
        action, record = adapter.request("inspector", {})
    finally:
        adapter.close()
    assert action == NO_TARGET
    assert seen == ["Bearer fake-token-a"] * 3 + ["Bearer fake-token-b", "Bearer fake-token-a"]
    assert record["latency_seconds"] == 63.0
    assert record["http_requests_by_key_slot"] == {"1": 4, "2": 1}
    state = read_state(setup)
    slots = state["days"][state["quota_day"]]["slots"]
    assert slots["1"]["consecutive_errors"] == 0
    assert "cooldown_until" not in slots["1"]


@pytest.mark.parametrize("append", [False, True])
def test_daily_pause_continues_same_request_after_append_or_utc_reset(setup, append):
    env, _, clock = setup
    if not append:
        clock.now = datetime(2026, 11, 21, 23, 59, 55, tzinfo=timezone.utc).timestamp()
    seen = []
    observed_pause = []

    def handler(request):
        seen.append((request.url.path, request.content))
        return quota() if len(seen) == 1 else ok()

    def on_sleep():
        if read_state(setup).get("waiting") != "daily_quota":
            return
        observed_pause.append(True)
        if append:
            with env.open("a", encoding="utf-8") as stream:
                if len(observed_pause) == 1:
                    stream.write(f"CLOUDFLARE_ACCOUNT_ID_002={ACCOUNT_B}\n")
                    return
                stream.write("CLOUDFLARE_API_TOKEN_002=fake-token-b\n")
        clock.on_sleep = None

    clock.on_sleep = on_sleep
    adapter = client(setup, LLAMA, handler)
    try:
        action, record = adapter.request("inspector", {}, remaining_seconds=60)
    finally:
        adapter.close()
    assert action == NO_TARGET
    assert len(observed_pause) == (2 if append else 1)
    assert seen[0][1] == seen[1][1]
    assert f"/{ACCOUNT_B if append else ACCOUNT_A}/" in seen[1][0]
    assert record["quota_wait_seconds"] == (30 if append else 4)
    assert read_state(setup)["quota_day"] == ("2026-11-21" if append else "2026-11-22")


def test_quota_wait_respects_deadline_and_billing_error_fails_closed(setup):
    adapter = client(setup, LLAMA, lambda request: quota())
    try:
        action, record = adapter.request("inspector", {}, remaining_seconds=3)
    finally:
        adapter.close()
    assert action is None and record["status"] == "api_deadline_skipped"
    assert record["http_requests_by_key_slot"] == {"1": 1}

    _, _, clock = setup
    clock.now += 86400
    adapter = client(setup, MISTRAL, lambda request: httpx.Response(
        403, json={"errors": [{"code": 5035, "message": "fake-token-a"}]}))
    try:
        action, record = adapter.request("inspector", {})
        assert action is None and record["status"] == "api_billing_required"
        assert "fake-token-a" not in str(record)
        assert adapter.request("inspector", {})[1]["api_requests"] == 0
    finally:
        adapter.close()


def test_utc_rollover_starts_at_first_slot_even_if_second_was_healthy(setup):
    env, _, clock = setup
    add_pair(env, 2, ACCOUNT_B, "fake-token-b")
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return quota() if len(seen) == 1 else ok()

    adapter = client(setup, LLAMA, handler)
    try:
        assert adapter.request("inspector", {})[0] == NO_TARGET
        assert read_state(setup)["next_key_slot"] == 2
        clock.now += 86400
        assert adapter.request("inspector", {})[0] == NO_TARGET
        assert seen == ["Bearer fake-token-a", "Bearer fake-token-b", "Bearer fake-token-a"]
    finally:
        adapter.close()


def test_pre_unification_cloudflare_ledger_keeps_same_day_quota(setup):
    env, usage, _ = setup
    add_pair(env, 2, ACCOUNT_B, "fake-token-b")
    usage.write_text(json.dumps({
        "utc_day": "2026-11-21", "next_key_slot": 1,
        "ordered_account_digests": [hashlib.sha256(account.encode()).hexdigest()
                                    for account in (ACCOUNT_A, ACCOUNT_B)],
        "days": {"2026-11-21": {"accounts": {"1": {"daily_exhausted": True}}, "slots": {}}},
    }), encoding="utf-8")
    keys = []

    def send(request):
        keys.append(request.headers["Authorization"])
        return ok()

    adapter = client(setup, LLAMA, send)
    try:
        assert adapter.request("inspector", {})[0] == NO_TARGET
    finally:
        adapter.close()
    assert keys == ["Bearer fake-token-b"]
    assert read_state(setup)["quota_day"] == "2026-11-21"


def test_gemini_thinking_env_does_not_break_cloudflare_and_preflight_validates_pairs(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from tools import gemini_preflight

    exp = load_experiment_settings()
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "high")
    monkeypatch.delenv("LLM_THINKING_LEVEL", raising=False)
    monkeypatch.setattr(_sys, "argv", ["run_experiment.py", "--pipeline", "single",
                                       "--instance", "wbg-fal10", "--model", "cloudflare_llama", "--dry-run"])
    import run_experiment
    assert run_experiment.main() == 0
    out = capsys.readouterr().out
    assert "--provider cloudflare" in out and "--thinking-level" not in out

    env_path = tmp_path / ".env"
    env_path.write_text("CLOUDFLARE_API_TOKEN_001=fake-only-token\n", encoding="utf-8")
    monkeypatch.setattr("providers.cloudflare_pool.ENV_PATH", env_path)
    monkeypatch.setattr(_sys, "argv", ["gemini_preflight.py", "--model", "cloudflare_llama"])
    gemini_preflight.main()
    bad_report = json.loads(capsys.readouterr().out)
    assert bad_report["credentials_configured"] is False
    assert bad_report["credential_slots"] == 0
    assert bad_report["credential_error"] == "ValueError"

    env_path.write_text(f"CLOUDFLARE_ACCOUNT_ID_001={ACCOUNT_A}\nCLOUDFLARE_API_TOKEN_001=fake-ok-token\n",
                        encoding="utf-8")
    gemini_preflight.main()
    ok_report = json.loads(capsys.readouterr().out)
    assert ok_report["credentials_configured"] is True
    assert ok_report["credential_slots"] == 1
    assert ok_report["thinking_level"] is None
    assert "credential_error" not in ok_report
    assert "fake-ok-token" not in json.dumps(ok_report)

    monkeypatch.setattr(_sys, "argv", ["gemini_preflight.py", "--model", "cloudflare_mistral",
                                       "--thinking-level", "low", "--live", "--confirm-free-tier"])
    with pytest.raises(SystemExit) as exc:
        gemini_preflight.main()
    assert exc.value.code == 2
