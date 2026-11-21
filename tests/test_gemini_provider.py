"""Gemini provider, centralized settings, and multi-key quota pool tests."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "NSGA-III"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from providers.credential_pool import pool_digest
from providers.gemini_pool import FLASH_LITE, HYDROGEN, GeminiPool, read_gemini_keys
from providers.settings import (
    available_model_profiles,
    create_decision_client,
    load_experiment_settings,
    resolve_model_profile,
)

NO_TARGET = {
    "solution_id": None, "region_id": None, "boundary_id": None,
    "relevant_constraint_ids": [],
}
PROFILES = {
    HYDROGEN: "gemini_hydrogen",
    FLASH_LITE: "gemini_flash_lite",
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


def gemini_ok(*, finish_reason="STOP", tokens=21):
    return httpx.Response(200, json={
        "modelVersion": "example-version",
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": "intermediate", "thought": True}, {"text": json.dumps(NO_TARGET)}]},
        }],
        "usageMetadata": {"promptTokenCount": tokens, "candidatesTokenCount": 5, "thoughtsTokenCount": 3},
    })


def gemini_client(model, root, clock, handler, *, level=None, limits=None, legacy_path=None):
    profile = {**model_settings(model), "min_request_interval_seconds": 0}
    if limits is not None:
        profile["quota"] = limits
    pool = GeminiPool(model, profile["quota"], env_path=root / ".env", usage_path=root / f"{model}.json",
                      legacy_path=legacy_path, clock=clock.time, monotonic=clock.time, sleep=clock.sleep)
    return create_decision_client(profile, thinking_level=level, free_tier_confirmed=True,
                                  pool=pool, transport=httpx.MockTransport(handler))


def gemini_env(path):
    path.write_text("GEMINI_API_KEY_001=fake-google-a\nGEMINI_API_KEY_002=fake-google-b\n", encoding="utf-8")


def test_centralized_settings_resolve_gemini_profiles():
    exp = load_experiment_settings()
    assert len(exp["instances"]) == 12
    assert exp["llm"]["model_profile"] == "gemini_hydrogen"
    profiles = available_model_profiles()
    assert {"gemini_hydrogen", "gemini_flash_lite"} <= set(profiles)
    by_name = resolve_model_profile("gemini_flash_lite", experiment_settings=exp)
    by_id = resolve_model_profile("gemini-3.5-flash-lite", experiment_settings=exp)
    assert by_name["model"] == FLASH_LITE == by_id["model"]
    assert by_name["max_output_tokens"] == exp["llm"]["max_output_tokens"] == 4096


def test_env_var_fallback_when_default_env_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-single-env-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("providers.gemini_pool.ENV_PATH", tmp_path / "missing.env")
    creds = read_gemini_keys(tmp_path / "missing.env")
    assert [c.token for c in creds] == ["fake-single-env-key"]


@pytest.mark.parametrize("model", [FLASH_LITE, HYDROGEN])
@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_gemini_profiles_build_request_with_selected_thinking_level(tmp_path, model, level):
    gemini_env(tmp_path / ".env")
    seen = []

    def send(request):
        seen.append(request)
        return gemini_ok()

    adapter = gemini_client(model, tmp_path, Clock(), send, level=level)
    try:
        assert adapter.metadata()["thinking_level"] == level
        action, record = adapter.request("inspector", {"examples": []})
        assert action == NO_TARGET
        assert record["http_requests_by_key_slot"] == {"1": 1}
        assert record["thinking_tokens"] == 3
    finally:
        adapter.close()
    request, = seen
    assert request.url.path == f"/v1beta/models/{model}:generateContent"
    assert "key" not in request.url.params
    assert request.headers["x-goog-api-key"] == "fake-google-a"
    params = json.loads(request.content)["generationConfig"]
    assert params["thinkingConfig"] == {"thinkingLevel": level.upper(), "includeThoughts": False}
    assert params["maxOutputTokens"] == 4096
    assert params["responseMimeType"] == "application/json"
    assert "boundary_id" in params["responseJsonSchema"]["required"]


def test_gemini_daily_429_rotates_without_using_another_model_quota(tmp_path):
    gemini_env(tmp_path / ".env")
    clock = Clock()
    requests = []

    def flash(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"details": [{
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
            }]}})
        return gemini_ok()

    adapter = gemini_client(FLASH_LITE, tmp_path, clock, flash)
    try:
        assert adapter.request("inspector", {"same": "packet"})[0] == NO_TARGET
    finally:
        adapter.close()
    assert [r.headers["x-goog-api-key"] for r in requests] == ["fake-google-a", "fake-google-b"]
    assert requests[0].content == requests[1].content
    state = json.loads((tmp_path / f"{FLASH_LITE}.json").read_bytes())
    assert state["days"][state["quota_day"]]["accounts"]["1"]["daily_exhausted"]
    assert "fake-google-a" not in json.dumps(state)
    hydrogen_seen = []

    def hydrogen(request):
        hydrogen_seen.append(request.headers["x-goog-api-key"])
        return gemini_ok()

    adapter = gemini_client(HYDROGEN, tmp_path, clock, hydrogen)
    try:
        assert adapter.request("inspector", {})[0] == NO_TARGET
    finally:
        adapter.close()
    assert hydrogen_seen == ["fake-google-a"]


@pytest.mark.parametrize("limits", [{"minute_requests": 1}, {"minute_input_tokens": 21}])
def test_gemini_minute_limits_keep_current_slot(tmp_path, limits):
    gemini_env(tmp_path / ".env")
    clock = Clock()
    seen = []

    def send(request):
        seen.append(request.headers["x-goog-api-key"])
        return gemini_ok()

    adapter = gemini_client(FLASH_LITE, tmp_path, clock, send, limits=limits)
    try:
        assert adapter.request("inspector", {})[0] == NO_TARGET
        action, record = adapter.request("inspector", {})
        assert action == NO_TARGET
        assert record["pacing_wait_seconds"] == 60.0
    finally:
        adapter.close()
    assert seen == ["fake-google-a"] * 2


def test_verified_hydrogen_counts_pause_then_reset_at_pacific_midnight(tmp_path):
    gemini_env(tmp_path / ".env")
    clock = Clock()
    clock.now = datetime(2026, 11, 21, 7, 59, 55, tzinfo=timezone.utc).timestamp()
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({
        "key_count": 2, "key_order_sha256": pool_digest(["fake-google-a", "fake-google-b"]),
        "days": {"2026-11-20": {"1": {"legacy_sdk_calls": 20},
                                "2": {"provider_daily_exhausted": True}}},
    }), encoding="utf-8")
    before = legacy.read_bytes()
    seen = []

    def send(request):
        seen.append(request.headers["x-goog-api-key"])
        return gemini_ok()

    adapter = gemini_client(HYDROGEN, tmp_path, clock, send, legacy_path=legacy)
    try:
        action, record = adapter.request("inspector", {}, remaining_seconds=10)
        assert action == NO_TARGET
        assert record["quota_wait_seconds"] == 5.0
    finally:
        adapter.close()
    assert seen == ["fake-google-a"] and legacy.read_bytes() == before


def test_gemini_truncated_json_and_invalid_thinking_level(tmp_path):
    gemini_env(tmp_path / ".env")
    with pytest.raises(ValueError, match="thinking_level"):
        gemini_client(FLASH_LITE, tmp_path, Clock(), lambda request: gemini_ok(), level="minimal")
    adapter = gemini_client(FLASH_LITE, tmp_path, Clock(), lambda request: gemini_ok(finish_reason="MAX_TOKENS"))
    try:
        action, record = adapter.request("inspector", {})
        assert action is None and record["status"] == "incomplete_response"
    finally:
        adapter.close()
