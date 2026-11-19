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

DEFAULT_MODEL = "gemini-4-hydrogen"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
VALID_THINKING_LEVELS = ("minimal", "low", "medium", "high")
RETRY_ATTEMPTS = 3
# Retry transient timeouts and 5xx errors only; keep 429 fail-closed per quota policy.
TRANSIENT_RETRY_STATUS_CODES = (408, 500, 502, 503, 504)
# Text models with a published Free Tier when this integration was prepared.
# No latest/preview aliases or silent fallback to a different model/provider.
FREE_TIER_MODELS = (DEFAULT_MODEL, "gemini-3.5-flash-lite")


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

_STATE_GUIDANCE = (
    "Look up each candidate region's solution_id in solutions to inspect its schedule hard "
    "violations (hard), internal soft total (total), optional unlocalized_penalties, and class "
    "assignments (classes), and compare hard then total against global.incumbent_hard and "
    "global.incumbent_internal_total. A large regional penalty alone is weak evidence if its "
    "source is far behind the incumbent. Work on a different source only if its repair "
    "opportunity plausibly closes the gap or offers a distinct useful placement. "
    "constraint_details lists active violations touching the region, ordered with hard and in-region "
    "violations first; if it is empty, regional soft cost comes from unary current_time_penalty or "
    "current_room_penalty in solutions[solution_id].classes. Use required to separate hard from "
    "soft constraints. "
    "region_violation_units covers violating groups touching the region; when violations extend "
    "outside the region, current_violations also reports the constraint's full violation total. "
    "Read params (when present) in the constraint type's order (e.g. MinGap's first parameter is "
    "the minimum gap in time slots). In violation_groups, in_region_class_ids lists violating "
    "classes inside the region and outside_region_class_ids lists fixed violating classes outside; "
    "an empty outside_region_class_ids means all violating classes can change together, whereas when "
    "non-empty inspect solutions[solution_id].classes (time_options, room_options) to see if "
    "changing in-region classes can resolve the violation. cross_boundary indicates whether any "
    "part of the constraint lies outside the region; group lists can be incomplete. Check "
    "boundary_id, overlapping class_ids, source_hard, source_total, best_trial_hard, and "
    "best_trial_total in recent_outcomes to distinguish clusters that are hard-stuck from those "
    "close to improving. Revisit a failed cluster only if the state changed or a different operator "
    "fits better. "
)
_ACCEPTANCE_GUIDANCE = (
    "For source hard > 0, a trial can reach population selection by reducing hard violations "
    "or by lowering total at the same hard count. For source hard = 0, a trial must stay "
    "feasible and improve at least one weighted objective (time, room, distribution, student). "
    "A feasible trade-off can reach selection even if total increases. NSGA-III decides "
    "whether to keep the trial. Use these rules to judge whether a target is worth the evaluations. "
)
_BUDGET_GUIDANCE = (
    "Region scoring, operator trials and any recheck share the entire search evaluation budget "
    "(1,800 in the Phase 2 campaign; use global.evaluation_budget and remaining_evaluations "
    "for this run). Even no_op consumes a review and region scoring has already spent evaluations. "
    "Use reviews_used and max_reviews to preserve review opportunities and leave evaluations "
    "for NSGA-III generations. "
)
_ACTION_GUIDANCE = (
    "Use an operator in the target's available_operators and respect allowed_actions ranges. "
    "All operators act only inside the selected region and charge full scoring on trial evaluations. "
    "Prefer the smallest change supported by the current violations. "
    "move ranks time-and-room placements for one in-region class by hard conflicts then soft costs, "
    "including estimated student conflicts (typically 2 to 5 evaluations). Use it when one class "
    "can resolve a boundary conflict or reduce a time, room, or student penalty. "
    "swap ranks room-only, time-only, and full exchanges between compatible in-region classes "
    "(typically 3 to 6 evaluations). Use it when an exchange between two classes can address the "
    "conflict. "
    "destroy_repair runs bounded CP-SAT over hard conflicts (and soft distributions when feasible) "
    "followed by a ranked sequential rebuild (typical budget 6 to 10 evaluations); use it for coupled "
    "multi-class conflicts inside the region, though it can stall when violations depend mainly on "
    "outside_region classes. "
    "Check recent_outcomes before choosing: if the same operator recently returned "
    "no_improving_candidate on a similar source or region, switch to a different operator, region, "
    "or no_op instead of repeating the same call. "
    "For no_op, set execution_budget=0 and next_review_interval to 10-20 within the allowed range, "
    "or the nearest allowed value. Choose 3-5 only when the current state shows a concrete "
    "opportunity that needs an early follow-up. "
    "next_review_interval sets the scheduled review. Stagnation or low diversity can trigger "
    "an earlier review once the minimum review interval has elapsed. "
)

PROMPTS = {
    "single_agent": (
        "You control one bounded NSGA-III improvement step for ITC 2019. "
        "Select WHERE from candidate_regions, HOW from allowed_actions, and WHEN. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE + _ACTION_GUIDANCE
        + "Use only listed IDs. For no_op, return null solution_id, region_id and boundary_id. "
        "Never output code or a timetable. "
        "The solver validates and executes your decision."
    ),
    "inspector": (
        "You are the Inspector for ITC 2019 NSGA-III. Choose only WHERE: one "
        "candidate region's solution_id, region_id, and boundary_id. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE
        + "Check recent_outcomes first: if an overlapping cluster already returned "
        "no_improving_candidate on a similar source state, avoid repeating the same target when another "
        "candidate_region or solution_id offers a distinct repair opportunity, or return null target IDs "
        "and an empty list if no target is promising. "
        "Return only a short list of relevant_constraint_ids from the selected region's "
        "constraint_details that explains why this region is worth repairing (or an empty list "
        "if the region is selected for class time/room penalties). "
        "Do not choose an operator or schedule."
    ),
    "planner": (
        "You are the Planner for ITC 2019 NSGA-III. The Inspector's target is fixed. "
        "Choose only HOW and WHEN. Use the diagnosis IDs to look up the target's constraints. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE + _ACTION_GUIDANCE
        + "Never output code or a timetable."
    ),
}


def _encode_payload(payload: dict[str, Any], max_characters: int) -> tuple[str, int]:
    def encode(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    body = encode(payload)
    if len(body) <= max_characters:
        return body, 0

    state = payload.get("state", payload)
    regions = state.get("candidate_regions") if isinstance(state, dict) else None
    if not regions:
        return body, 0
    diagnosis = payload.get("diagnosis", {})
    diagnosed_ids = set(diagnosis.get("relevant_constraint_ids", ()))
    target_keys = ("solution_id", "region_id", "boundary_id")
    ranked_details = []
    for region in regions:
        is_target = diagnosed_ids and all(region.get(key) == diagnosis.get(key) for key in target_keys)
        required_ids = diagnosed_ids if is_target else set()
        def _detail_units(item: dict[str, Any]) -> int:
            if "region_violation_units" in item:
                return int(item["region_violation_units"])
            return int(item.get("current_violations", 0))

        details = sorted(region.get("constraint_details", ()), key=lambda item: (
            item["constraint_id"] not in required_ids,
            not bool(_detail_units(item)),
            not item["required"],
            item["cross_boundary"],
            -_detail_units(item),
        ))
        keep = sum(item["constraint_id"] in required_ids for item in details)
        ranked_details.append((region, details, keep))

    def limited(limit: int) -> tuple[str, int]:
        public_regions = []
        omitted = 0
        for region, details, keep in ranked_details:
            count = max(limit, keep)
            dropped = max(0, len(details) - count)
            if dropped:
                region = {**region, "constraint_details": details[:count],
                          "constraint_details_omitted": dropped}
            public_regions.append(region)
            omitted += dropped
        public_state = {**state, "candidate_regions": public_regions}
        public_payload = public_state if state is payload else {**payload, "state": public_state}
        return encode(public_payload), omitted

    # Fit as many details per region as possible without changing IDs or the response schema.
    best = limited(0)
    low, high = 1, max(len(details) for _, details, _ in ranked_details)
    while low <= high:
        limit = (low + high) // 2
        reduced = limited(limit)
        if len(reduced[0]) <= max_characters:
            best = reduced
            low = limit + 1
        else:
            high = limit - 1
    return best


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

    def __init__(self, model: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, *, free_tier_confirmed: bool = False,
                 min_request_interval_seconds: float = 10.0,
                 max_input_characters: int = 60_000,
                 thinking_level: str = DEFAULT_THINKING_LEVEL) -> None:
        if not (free_tier_confirmed or os.environ.get("GEMINI_FREE_TIER_CONFIRMED") == "1"):
            raise RuntimeError("Check that this key's project is Free Tier in AI Studio, then use "
                               "--confirm-free-tier or GEMINI_FREE_TIER_CONFIRMED=1. "
                               "Do not enable billing for this test.")
        if model not in FREE_TIER_MODELS:
            raise ValueError("This free-tier-only adapter permits: " + ", ".join(FREE_TIER_MODELS))
        if thinking_level not in VALID_THINKING_LEVELS:
            raise ValueError("thinking_level must be one of: " + ", ".join(VALID_THINKING_LEVELS))
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
        self.retry_options = types.HttpRetryOptions(
            attempts=RETRY_ATTEMPTS,
            http_status_codes=list(TRANSIENT_RETRY_STATUS_CODES),
        )
        self.client = genai.Client(api_key=key, vertexai=False,
            http_options=types.HttpOptions(api_version="v1beta",
                base_url="https://generativelanguage.googleapis.com",
                timeout=max(1, int(timeout_seconds*1000)),
                retry_options=self.retry_options))
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_input_characters = max_input_characters
        self.thinking_level = thinking_level
        self._last_request_started: float | None = None
        self.blocked_reason: str | None = None

    def metadata(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "api": "gemini_developer_generate_content", "free_tier_only": True,
                "api_base_url": "https://generativelanguage.googleapis.com",
                "billing_tier_verification": "user_confirmed_not_queryable_from_key",
                "min_request_interval_seconds": self.min_request_interval_seconds,
                "max_input_characters": self.max_input_characters,
                "thinking_level": self.thinking_level, "sdk_attempts_per_call": RETRY_ATTEMPTS,
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
            body, omitted_details = _encode_payload(payload, self.max_input_characters - len(prompt))
            if omitted_details:
                record["constraint_details_omitted"] = omitted_details
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
                thinking_config=self.types.ThinkingConfig(
                    thinking_level=self.thinking_level, include_thoughts=False),
                http_options=self.types.HttpOptions(timeout=max(1,int(call_timeout*1000)),
                    retry_options=self.retry_options))
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
