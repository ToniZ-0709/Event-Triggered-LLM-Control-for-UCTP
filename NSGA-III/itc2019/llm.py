"""Gemini Developer API adapter; no OpenAI or paid-provider fallback.

The caller must confirm the project is Free Tier in AI Studio. An API key alone
does not expose its billing tier, so this is an explicit human confirmation, not
a technical guarantee that Google cannot bill a misconfigured project.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .operators import OPERATORS

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Text models with a published Free Tier when this integration was prepared.
# No latest/preview aliases or silent fallback to a different model/provider.
FREE_TIER_MODELS = (DEFAULT_MODEL, "gemini-3.6-flash", "gemini-3.8-flash")


def _object(properties: dict[str, dict]) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_NULLABLE_ID = {"type": ["string", "null"]}
SCHEMAS = {
    "single_agent": _object({
        "solution_id": _NULLABLE_ID, "region_id": _NULLABLE_ID,
        "boundary_id": _NULLABLE_ID,
        "operator_id": {"type": "string", "enum": ["no_op", *OPERATORS]},
        "execution_budget": {"type": "integer"},
        "next_review_interval": {"type": "integer"},
    }),
    "inspector": _object({
        "solution_id": _NULLABLE_ID, "region_id": _NULLABLE_ID,
        "boundary_id": _NULLABLE_ID,
        "relevant_constraint_ids": {"type": "array", "items": {"type": "string"}},
    }),
    "planner": _object({
        "operator_id": {"type": "string", "enum": ["no_op", *OPERATORS]},
        "execution_budget": {"type": "integer"},
        "next_review_interval": {"type": "integer"},
    }),
}

PROMPTS = {
    "single_agent": (
        "You control one bounded NSGA-III improvement step for ITC 2019. "
        "Select WHERE from candidate_regions[].target, HOW/WHEN from action_limits. "
        "Choose an operator in the target region's available_actions.operators. "
        "Use boundary_context joined by region_id for rules and fixed blockers. "
        "Diagnostics are bounded local checks, not full scores or executable placements. "
        "guided_move ranks one-gene proposals; distribution_repair targets its published required rule. "
        "blocker_chain may move only classes in its separately published expanded target. "
        "Use only listed IDs and integer ranges. Return no_op with null IDs and budget 0 "
        "if no suitable action exists. Never output code or a timetable. "
        "The solver validates and executes your decision."
    ),
    "inspector": (
        "You are the Inspector for ITC 2019 NSGA-III. Choose only WHERE: one "
        "candidate region's target.solution_id, target.region_id, and target.boundary_id. "
        "Use boundary_context constraints.ids for relevant_constraint_ids. "
        "Read diagnostics as bounded local evidence, with unknown full/student effects. "
        "Return null IDs if no target is appropriate. Do not choose an operator or schedule."
    ),
    "planner": (
        "You are the Planner for ITC 2019 NSGA-III. The Inspector's target is fixed. "
        "Choose only HOW and WHEN from action_limits; the operator must "
        "appear in the target region's available_actions.operators. "
        "guided_move ranks one-gene changes, distribution_repair targets its published required rule, "
        "and destroy_repair jointly repairs the active region. Local checks do not guarantee full improvement. "
        "blocker_chain is available only on an explicitly expanded target. "
        "Return no_op with budget 0 if no suitable operator exists. "
        "Never output code or a timetable."
    ),
}


def validate_shape(role: str, value: Any) -> bool:
    """Validate JSON types/keys; the solver separately validates stateful IDs."""
    schema = SCHEMAS[role]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return False
    for key, spec in schema["properties"].items():
        item = value[key]
        kind = spec["type"]
        if isinstance(kind, list):
            if item is not None and not isinstance(item, str):
                return False
        elif kind == "integer" and type(item) is not int:
            return False
        elif kind == "string" and not isinstance(item, str):
            return False
        elif kind == "array" and (not isinstance(item, list) or
                                  any(not isinstance(x, str) for x in item)):
            return False
        if "enum" in spec and item not in spec["enum"]:
            return False
    return True


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _enum(value) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


class GeminiDecisionClient:
    provider = "gemini"

    def __init__(self, model: str, timeout_seconds: float,
                 max_output_tokens: int, *, free_tier_confirmed: bool = False,
                 min_request_interval_seconds: float = 10.0,
                 max_input_characters: int = 60_000) -> None:
        if not (free_tier_confirmed or os.environ.get("GEMINI_FREE_TIER_CONFIRMED") == "1"):
            raise RuntimeError("Check that this key's project is Free Tier in AI Studio, then use "
                               "--confirm-free-tier or GEMINI_FREE_TIER_CONFIRMED=1. "
                               "Do not enable billing for this test.")
        if model not in FREE_TIER_MODELS:
            raise ValueError("This free-tier-only adapter permits: " + ", ".join(FREE_TIER_MODELS))
        if (timeout_seconds <= 0 or not 1 <= max_output_tokens <= 4096 or
                not 0 <= min_request_interval_seconds <= 60 or max_input_characters < 1):
            raise ValueError("Invalid Gemini timeout, token, pacing or input limits")
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY locally before running a controller; "
                               "never put the key in a CLI argument or commit it")
        if (os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY") and
                os.environ["GEMINI_API_KEY"] != os.environ["GOOGLE_API_KEY"]):
            raise RuntimeError("GEMINI_API_KEY and GOOGLE_API_KEY differ; keep one key to avoid "
                               "using the wrong project's billing tier")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the controller requirements (google-genai)") from exc
        self.types = types
        self.client = genai.Client(api_key=key, vertexai=False,
            http_options=types.HttpOptions(api_version="v1beta",
                base_url="https://generativelanguage.googleapis.com",
                timeout=max(1, int(timeout_seconds*1000)),
                retry_options=types.HttpRetryOptions(attempts=1)))
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_input_characters = max_input_characters
        self._last_request_started: float | None = None
        self.blocked_reason: str | None = None

    def metadata(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "api": "gemini_developer_generate_content", "free_tier_only": True,
                "api_base_url": "https://generativelanguage.googleapis.com",
                "billing_tier_verification": "user_confirmed_not_queryable_from_key",
                "min_request_interval_seconds": self.min_request_interval_seconds,
                "max_input_characters": self.max_input_characters,
                "thinking_level": "low", "sdk_attempts_per_call": 1,
                "automatic_continuation": False, "automatic_function_calling": False,
                "quota_policy": "stop_api_and_mark_incomplete_no_provider_fallback"}

    def close(self) -> None:
        self.client.close()

    def request(self, role: str, payload: dict[str, Any],
                remaining_seconds: float | None = None) -> tuple[dict | None, dict]:
        prompt = PROMPTS[role]
        schema = SCHEMAS[role]
        started = time.perf_counter()
        record: dict[str, Any] = {
            "provider": self.provider, "role": role, "model_requested": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "input_tokens": None, "output_tokens": None, "response_id": None,
            "thinking_tokens": None, "total_tokens": None, "cached_input_tokens": None,
            "api_requests": 0, "pacing_wait_seconds": 0.0,
        }
        try:
            if self.blocked_reason:
                record["status"] = self.blocked_reason
                return None, record
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            record["input_characters"] = len(body)+len(prompt)
            if record["input_characters"] > self.max_input_characters:
                record["status"] = "api_input_limit"
                return None, record
            wait = (0.0 if self._last_request_started is None else max(0.0,
                    self.min_request_interval_seconds-(started-self._last_request_started)))
            if remaining_seconds is not None and remaining_seconds <= wait+0.1:
                record["status"] = "api_deadline_skipped"
                return None, record
            if wait:
                time.sleep(wait)
                record["pacing_wait_seconds"] = round(wait, 3)
            remaining = (None if remaining_seconds is None else
                         remaining_seconds-(time.perf_counter()-started))
            if remaining is not None and remaining <= 0.1:
                record["status"] = "api_deadline_skipped"
                return None, record
            call_timeout = self.timeout_seconds if remaining is None else min(self.timeout_seconds,remaining)
            record["request_timeout_seconds"] = round(call_timeout,3)
            config = self.types.GenerateContentConfig(
                system_instruction=prompt, response_mime_type="application/json",
                response_json_schema=schema, candidate_count=1,
                max_output_tokens=self.max_output_tokens,
                automatic_continuation=False,
                automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=self.types.ThinkingConfig(thinking_level="low",include_thoughts=False),
                http_options=self.types.HttpOptions(timeout=max(1,int(call_timeout*1000)),
                    retry_options=self.types.HttpRetryOptions(attempts=1)))
            self._last_request_started = time.perf_counter()
            record["api_requests"] = 1
            response = self.client.models.generate_content(model=self.model,contents=body,config=config)
            record["response_id"] = getattr(response,"response_id",None)
            record["model_returned"] = getattr(response,"model_version",None)
            usage = getattr(response,"usage_metadata",None)
            if usage is not None:
                for dest,src in (("input_tokens","prompt_token_count"),
                                 ("output_tokens","candidates_token_count"),
                                 ("thinking_tokens","thoughts_token_count"),
                                 ("total_tokens","total_token_count"),
                                 ("cached_input_tokens","cached_content_token_count")):
                    record[dest] = getattr(usage,src,None)
            candidates = getattr(response,"candidates",None) or []
            reason = _enum(getattr(candidates[0],"finish_reason",None)) if candidates else None
            record["finish_reason"] = reason
            record["response_status"] = reason
            feedback = getattr(response,"prompt_feedback",None)
            block_reason = _enum(getattr(feedback,"block_reason",None))
            if (block_reason not in (None,"BLOCK_REASON_UNSPECIFIED","BLOCKED_REASON_UNSPECIFIED") or
                    reason in {"SAFETY","BLOCKLIST","PROHIBITED_CONTENT","RECITATION"}):
                record["status"] = "model_refusal"
                return None, record
            if reason == "MAX_TOKENS":
                record["status"] = "incomplete_response"
                record["incomplete_reason"] = "max_output_tokens_including_thinking"
                return None, record
            if reason != "STOP":
                record["status"] = "incomplete_response"
                record["incomplete_reason"] = reason or "no_candidates"
                return None, record
            text = getattr(response,"text",None)
            if not text:
                record["status"] = "empty_response"
                return None, record
            try:
                decision = json.loads(text,object_pairs_hook=_json_pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
            except (ValueError,TypeError):
                record["status"] = "invalid_json"
                return None, record
            if not validate_shape(role,decision):
                record["status"] = "invalid_schema"
                return None, record
            record["status"] = "ok"
            record["raw_decision"] = decision
            return decision, record
        except Exception as exc:
            status_code = getattr(exc,"code",None) or getattr(exc,"status_code",None)
            record["http_status"] = status_code
            record["status"] = ("api_quota_exhausted" if status_code == 429 else
                                "api_billing_required" if status_code == 402 else
                                "api_configuration_error" if status_code in (400,401,403,404) else
                                "api_timeout" if "timeout" in type(exc).__name__.lower() else
                                "api_service_error" if isinstance(status_code,int) and status_code >= 500 else
                                "api_error")
            if record["status"] in {"api_quota_exhausted","api_billing_required","api_configuration_error"}:
                self.blocked_reason = record["status"]
            record["error_type"] = type(exc).__name__
            message = str(exc)
            for name in ("GEMINI_API_KEY","GOOGLE_API_KEY"):
                key = os.environ.get(name)
                if key:
                    message = message.replace(key,"[REDACTED]")
            record["error"] = message[:300]
            return None, record
        finally:
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
