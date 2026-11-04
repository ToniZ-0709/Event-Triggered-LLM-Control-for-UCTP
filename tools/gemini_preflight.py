"""Check local Gemini setup; only --live sends exactly one bounded API call."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"NSGA-III"))
from itc2019.llm import DEFAULT_MODEL, FREE_TIER_MODELS, GeminiDecisionClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send one synthetic no-op packet, not a GA run")
    parser.add_argument("--confirm-free-tier", action="store_true")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL",DEFAULT_MODEL))
    args = parser.parse_args()
    try:
        sdk = version("google-genai")
    except PackageNotFoundError:
        sdk = None
    report = {"provider":"gemini", "model":args.model,"sdk":sdk,
              "key_present":bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
              "model_in_free_tier_allowlist":args.model in FREE_TIER_MODELS,
              "free_tier_confirmation":args.confirm_free_tier or os.environ.get("GEMINI_FREE_TIER_CONFIRMED")=="1",
              "billing_checked_programmatically":False,"network_requests":0}
    if not args.live:
        report["status"] = "offline_check_only"
        print(json.dumps(report,indent=2))
        return
    try:
        client = GeminiDecisionClient(args.model,15,1024,free_tier_confirmed=args.confirm_free_tier)
    except (ValueError,RuntimeError) as exc:
        parser.error(str(exc))
    try:
        decision, record = client.request("single_agent", {
            "schema_version":1,"instance":"connection-check-no-real-data",
            "global":{"generation":5},"candidate_regions":[],"recent_outcomes":[],
            "allowed_actions":{"operators":["no_op"],"execution_budget":[1,1],"review_interval":[3,20]}},15)
        report.update(status=record["status"],network_requests=record["api_requests"],
                      decision=decision,record=record)
        print(json.dumps(report,indent=2))
        if record["status"] != "ok" or decision["operator_id"] != "no_op":
            raise SystemExit(3)
    finally:
        client.close()


if __name__ == "__main__":
    main()
