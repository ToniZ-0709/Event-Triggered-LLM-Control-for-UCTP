"""Gemini credentials and per-model project quota policy."""

from functools import partial
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import httpx

from .credential_pool import (ROOT, ENV_PATH, Credential, CredentialPool, Quota,
                              numbered_entries, pool_digest, response_object)

HYDROGEN = "gemini-4-hydrogen"
FLASH_LITE = "gemini-3.5-flash-lite"
GEMINI_MODELS = (HYDROGEN, FLASH_LITE)
PACIFIC = ZoneInfo("America/Los_Angeles")
KEY_NAME = "GEMINI_API_KEY"
HYDROGEN_LEGACY_USAGE = ROOT / "ExpRun" / "gemini_daily_usage.json"


def read_gemini_keys(path: Path) -> list[Credential]:
    if not path.is_file() and path == ENV_PATH:
        gemini_key = os.environ.get(KEY_NAME)
        google_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_key and google_key and gemini_key != google_key:
            raise RuntimeError(
                "GEMINI_API_KEY and GOOGLE_API_KEY differ; keep one key to avoid using the wrong project's billing tier."
            )
        token = (gemini_key or google_key or "").strip()
        if not token:
            raise RuntimeError("Configure GEMINI_API_KEY_001 in .env or set GEMINI_API_KEY locally.")
        if not re.fullmatch(r"[A-Za-z0-9._~-]+", token):
            raise ValueError("Expected a valid Gemini API key in environment.")
        return [Credential(token=token)]
    credentials = []
    for slot, entry in enumerate(numbered_entries(path, KEY_NAME), 1):
        token = entry.get(KEY_NAME, "")
        if not re.fullmatch(r"[A-Za-z0-9._~-]+", token):
            raise ValueError(f"Expected a Gemini API key for slot {slot}.")
        credentials.append(Credential(token=token))
    if len(set(credentials)) != len(credentials):
        raise ValueError("Gemini .env contains duplicate API keys.")
    return credentials


def has_daily_quota_error(response: httpx.Response) -> bool:
    error = response_object(response).get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).lower()
    if re.search(r"per[ _-]?day|daily quota|requests\s*/\s*day|per 24 hours", message):
        return True
    details = error.get("details")
    for detail in details if isinstance(details, list) else ():
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations")
        for violation in violations if isinstance(violations, list) else ():
            if not isinstance(violation, dict):
                continue
            for name in ("quotaId", "quotaMetric", "description"):
                text = re.sub(r"[ _-]+", "", str(violation.get(name, "")).lower())
                if "perday" in text or "daily" in text:
                    return True
    return False


def gemini_input_tokens(response: httpx.Response) -> int | None:
    usage = response_object(response).get("usageMetadata")
    reported = usage.get("promptTokenCount") if isinstance(usage, dict) else None
    return reported if type(reported) is int and reported >= 0 else None


def legacy_hydrogen_requests(path: Path, daily_limit: int | None,
                    credentials: list[Credential], day: str) -> dict[str, int]:
    if not path.is_file():
        return {}
    state = json.loads(path.read_bytes())
    today = state["days"].get(day)
    if not today:
        return {}
    count = state["key_count"]
    if len(credentials) < count or pool_digest([item.token for item in credentials[:count]]) != state["key_order_sha256"]:
        raise ValueError("Gemini slots do not match the Hydrogen usage ledger for this Pacific day.")
    totals = {}
    for slot, entry in today.items():
        requests = sum(entry.get(name, 0) for name in (
            "http_requests", "legacy_sdk_calls", "dashboard_rpd_at_resume"))
        if entry.get("provider_daily_exhausted") and daily_limit is not None:
            requests = max(requests, daily_limit)
        totals[slot] = requests
    return totals


class GeminiPool(CredentialPool):
    """Use the same keys for both models, with separate quota ledgers."""

    def __init__(self, model: str, limits: dict, *, env_path: Path | None = None,
                 usage_path: Path | None = None, legacy_path: Path | None = None, **kwargs):
        if model not in GEMINI_MODELS:
            raise ValueError("Unsupported Gemini model.")
        resolved_env = ENV_PATH if env_path is None else env_path
        quota = Quota(timezone=PACIFIC, **limits)
        path = usage_path or ROOT / "tmp" / "gemini" / model / "usage.json"
        if model == HYDROGEN and legacy_path is None and resolved_env == ENV_PATH and usage_path is None:
            legacy_path = HYDROGEN_LEGACY_USAGE
        prior = None
        if model == HYDROGEN and legacy_path is not None:
            prior = partial(legacy_hydrogen_requests, legacy_path, quota.daily_requests)
        super().__init__(resolved_env, path, read_credentials=read_gemini_keys, quota=quota,
                         daily_quota_error=has_daily_quota_error, input_tokens=gemini_input_tokens,
                         prior_requests=prior, **kwargs)
