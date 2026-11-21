"""Check local LLM setup via centralized settings; only --live sends one bounded API call."""

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
    chosen_thinking = (
        args.thinking_level
        or os.environ.get("LLM_THINKING_LEVEL")
        or os.environ.get("GEMINI_THINKING_LEVEL")
        or profile.get("thinking_level")
    )
    timeout = float(args.timeout_seconds if args.timeout_seconds is not None
                    else profile.get("api_timeout_seconds", 60.0))
    free_tier_confirmed = (
        args.confirm_free_tier
        or os.environ.get(f"{provider.upper()}_FREE_TIER_CONFIRMED") == "1"
    )
    env_file = ROOT / ".env"
    has_env_creds = False
    if env_file.is_file():
        prefix = "GEMINI_API_KEY_" if provider == "gemini" else f"{provider.upper()}_API_"
        has_env_creds = any(
            line.strip().startswith(prefix) and "=" in line and not line.strip().startswith("#")
            for line in env_file.read_text(encoding="utf-8-sig").splitlines()
        )
    has_var_creds = (
        bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        if provider == "gemini"
        else False
    )

    report = {
        "model_profile": profile["profile_name"],
        "provider": provider,
        "model": profile["model"],
        "thinking_level": chosen_thinking,
        "timeout_seconds": timeout,
        "max_output_tokens": profile.get("max_output_tokens", 4096),
        "httpx": http_client_version,
        "credentials_configured": has_env_creds or has_var_creds,
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
            thinking_level=args.thinking_level or os.environ.get("LLM_THINKING_LEVEL") or os.environ.get("GEMINI_THINKING_LEVEL") or None,
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
