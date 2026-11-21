"""Check local LLM setup (Gemini and Cloudflare); only --live sends one bounded API call."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.settings import (  # noqa: E402
    create_decision_client,
    load_experiment_settings,
    resolve_model_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send one synthetic no-op packet, not a GA run")
    parser.add_argument("--confirm-free-tier", action="store_true")
    parser.add_argument("--model", default=None, help="Model profile name, model ID, or profile JSON path")
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()

    exp_settings = load_experiment_settings()
    try:
        profile = resolve_model_profile(args.model, experiment_settings=exp_settings)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        http_client_version = version("httpx")
    except PackageNotFoundError:
        http_client_version = None

    provider = profile["provider"]
    explicit_thinking = (
        args.thinking_level
        or os.environ.get("LLM_THINKING_LEVEL")
        or (os.environ.get("GEMINI_THINKING_LEVEL") if provider == "gemini" else None)
    )
    chosen_thinking = explicit_thinking or profile.get("thinking_level")
    timeout = float(args.timeout_seconds if args.timeout_seconds is not None
                    else profile.get("api_timeout_seconds", 60.0))
    free_tier_confirmed = (
        args.confirm_free_tier
        or os.environ.get(f"{provider.upper()}_FREE_TIER_CONFIRMED") == "1"
    )
    credentials_configured = False
    credential_slots = 0
    credential_error = None
    try:
        if provider == "gemini":
            from providers import gemini_pool
            creds = gemini_pool.read_gemini_keys(gemini_pool.ENV_PATH)
        elif provider == "cloudflare":
            from providers import cloudflare_pool
            creds = cloudflare_pool.read_accounts(cloudflare_pool.ENV_PATH)
        else:
            creds = []
        credentials_configured = bool(creds)
        credential_slots = len(creds)
    except (OSError, ValueError, RuntimeError) as exc:
        credential_error = type(exc).__name__

    report = {
        "model_profile": profile["profile_name"],
        "provider": provider,
        "model": profile["model"],
        "thinking_level": chosen_thinking,
        "timeout_seconds": timeout,
        "max_output_tokens": profile.get("max_output_tokens", 4096),
        "httpx": http_client_version,
        "credentials_configured": credentials_configured,
        "credential_slots": credential_slots,
        **({"credential_error": credential_error} if credential_error else {}),
        "free_tier_confirmation": free_tier_confirmed,
        "billing_checked_programmatically": False,
        "network_requests": 0,
    }
    if not args.live:
        report["status"] = "offline_check_only"
        print(json.dumps(report, indent=2))
        return

    try:
        client = create_decision_client(
            profile,
            thinking_level=explicit_thinking or None,
            free_tier_confirmed=args.confirm_free_tier,
            timeout_seconds=timeout,
            max_output_tokens=profile.get("max_output_tokens", 4096),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    try:
        decision, record = client.request(
            "single_agent",
            {
                "instance": "connection-check-no-real-data",
                "global": {"generation": 5},
                "solutions": {},
                "candidate_regions": [],
                "recent_outcomes": [],
                "allowed_actions": {"execution_budget": [1, 1], "review_interval": [3, 20]},
            },
            timeout,
        )
        report.update(
            status=record["status"],
            network_requests=record["api_requests"],
            decision=decision,
            record=record,
        )
        print(json.dumps(report, indent=2))
        if record["status"] != "ok" or decision["operator_id"] != "no_op":
            raise SystemExit(3)
    finally:
        client.close()


if __name__ == "__main__":
    main()
