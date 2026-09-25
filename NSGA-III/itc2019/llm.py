"""OpenAI Responses API adapter for bounded, structured controller decisions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .operators import OPERATORS


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
        "Select WHERE from candidate_regions, HOW from allowed_actions, and WHEN. "
        "Choose an operator listed in the selected region's available_operators. "
        "Use only listed IDs and integer ranges. Return no_op with null IDs and budget 0 "
        "if no suitable action exists. Never output code or a timetable. "
        "The solver validates and executes your decision."
    ),
    "inspector": (
        "You are the Inspector for ITC 2019 NSGA-III. Choose only WHERE: one "
        "candidate region's solution_id, region_id, and boundary_id. "
        "Return null IDs if no target is appropriate. Do not choose an operator or schedule."
    ),
    "planner": (
        "You are the Planner for ITC 2019 NSGA-III. The Inspector's target is fixed. "
        "Choose only HOW and WHEN from allowed_actions and ranges; the operator must "
        "appear in the target region's available_operators. "
        "Return no_op with budget 0 if no suitable operator exists. "
        "Never output code or a timetable."
    ),
}


class OpenAIDecisionClient:
    def __init__(self, model: str, timeout_seconds: float,
                 max_output_tokens: int) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running a controller pipeline")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the controller requirements to use the OpenAI API") from exc
        self.client = OpenAI(timeout=timeout_seconds, max_retries=0)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    def request(self, role: str, payload: dict[str, Any],
                remaining_seconds: float | None = None) -> tuple[dict | None, dict]:
        prompt = PROMPTS[role]
        schema = SCHEMAS[role]
        started = time.perf_counter()
        record: dict[str, Any] = {
            "role": role, "model_requested": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "input_tokens": None, "output_tokens": None, "response_id": None,
        }
        try:
            call_timeout = (self.timeout_seconds if remaining_seconds is None else
                            min(self.timeout_seconds, max(0.1, remaining_seconds)))
            response = self.client.with_options(timeout=call_timeout).responses.create(
                model=self.model,
                input=[{"role": "system", "content": prompt},
                       {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                text={"format": {"type": "json_schema", "name": role,
                                 "schema": schema, "strict": True}},
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
            record["response_id"] = response.id
            record["model_returned"] = response.model
            record["response_status"] = response.status
            if response.usage is not None:
                record["input_tokens"] = response.usage.input_tokens
                record["output_tokens"] = response.usage.output_tokens
            if response.status != "completed":
                details = getattr(response, "incomplete_details", None)
                record["status"] = "incomplete_response"
                record["incomplete_reason"] = getattr(details, "reason", None)
                return None, record
            if any(getattr(content, "type", None) == "refusal"
                   for item in response.output
                   for content in (getattr(item, "content", None) or ())):
                record["status"] = "model_refusal"
                return None, record
            if not response.output_text:
                record["status"] = "empty_response"
                return None, record
            try:
                decision = json.loads(response.output_text)
            except json.JSONDecodeError:
                record["status"] = "invalid_json"
                return None, record
            if not isinstance(decision, dict):
                record["status"] = "invalid_json_object"
                return None, record
            record["status"] = "ok"
            record["raw_decision"] = decision
            return decision, record
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            record["http_status"] = status_code
            record["status"] = ("api_configuration_error" if status_code in (400, 401, 403, 404)
                                else "api_error")
            record["error_type"] = type(exc).__name__
            message = str(exc)
            key = os.environ.get("OPENAI_API_KEY")
            record["error"] = message.replace(key, "[REDACTED]")[:300] if key else message[:300]
            return None, record
        finally:
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
