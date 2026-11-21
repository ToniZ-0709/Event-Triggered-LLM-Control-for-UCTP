"""Shared controller request, deadline, validation and tracing."""

from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import httpx


def _ensure_solver_import_path() -> None:
    solver_root = str(Path(__file__).resolve().parents[1] / "NSGA-III")
    if solver_root not in sys.path:
        sys.path.insert(0, solver_root)


class DecisionClient:
    def __init__(self, model, timeout_seconds, max_output_tokens, *, free_tier_confirmed,
                 min_request_interval_seconds, max_input_characters, load_pool, transport):
        if not (free_tier_confirmed or os.environ.get(f"{self.provider.upper()}_FREE_TIER_CONFIRMED") == "1"):
            raise RuntimeError(f"Confirm Free Tier before using {self.provider}.")
        if (not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or
                type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 4096 or
                not 0 <= min_request_interval_seconds <= 60 or max_input_characters < 1):
            raise ValueError("Invalid API timeout, output limit, pacing or input limit.")
        self.pool = load_pool()
        self.client = httpx.Client(transport=transport, follow_redirects=False)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_input_characters = max_input_characters
        self.blocked_reason: str | None = None

    def metadata(self) -> dict:
        return {
            "provider": self.provider, "model": self.model, "api": self.api,
            "api_base_url": self.api_base_url, "free_tier_only": True,
            "billing_tier_verification": "user_confirmed", "response_mode": self.response_mode,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "max_input_characters": self.max_input_characters,
            "automatic_continuation": False, "automatic_function_calling": False,
            "quota_policy": "wait_in_same_request_until_credential_is_available",
            "daily_reset_timezone": str(self.pool.quota.timezone),
        }

    def close(self):
        self.client.close()

    def system_prompt(self, prompt, schema):
        return prompt

    def request(self, role: str, payload: dict[str, Any],
                remaining_seconds: float | None = None) -> tuple[dict | None, dict]:
        _ensure_solver_import_path()
        from itc2019.llm import PROMPTS, SCHEMAS, _encode_payload, _json_pairs, validate_shape

        started = self.pool.monotonic()
        deadline = None if remaining_seconds is None else started + remaining_seconds
        schema = SCHEMAS[role]
        prompt = self.system_prompt(PROMPTS[role], schema)
        record: dict[str, Any] = {
            "provider": self.provider, "role": role, "model_requested": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
            "thinking_tokens": None, "cached_input_tokens": None, "response_id": None,
            "api_requests": 0, "http_requests_by_key_slot": {},
            "pacing_wait_seconds": 0.0, "quota_wait_seconds": 0.0,
        }
        try:
            if self.blocked_reason:
                record["status"] = self.blocked_reason
                return None, record
            body, omitted = _encode_payload(payload, self.max_input_characters - len(prompt))
            record["input_characters"] = len(prompt) + len(body)
            if omitted:
                record["constraint_details_omitted"] = omitted
            if record["input_characters"] > self.max_input_characters:
                record["status"] = "api_input_limit"
                return None, record

            request = self.request_body(prompt, body, schema)
            response = self.pool.send(
                partial(self.send_once, request), timeout_seconds=self.timeout_seconds,
                min_interval=self.min_request_interval_seconds, deadline=deadline, record=record)
            if response is None:
                record["status"] = "api_deadline_skipped"
                return None, record
            record["http_status"] = response.status_code
            answer = self.response_answer(response, record)
            if "status" in record:
                return None, record
            if answer is None or answer == "":
                record["status"] = "empty_response"
                return None, record
            if isinstance(answer, str):
                try:
                    answer = json.loads(
                        answer,
                        object_pairs_hook=_json_pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"Non-finite JSON constant: {token}")
                        ),
                    )
                except ValueError:
                    record["status"] = "invalid_json"
                    return None, record
            if not validate_shape(role, answer):
                record["status"] = "invalid_schema"
                return None, record
            record.update(status="ok", raw_decision=answer)
            return answer, record
        except (OSError, ValueError, httpx.HTTPError) as exc:
            # HTTP exceptions can contain secrets or URLs. Record only their type.
            record.update(status="api_error", error_type=type(exc).__name__)
            return None, record
        finally:
            record["latency_seconds"] = round(self.pool.monotonic() - started, 3)


def record_usage(record: dict, usage, fields: dict[str, str]):
    if isinstance(usage, dict):
        for target, source in fields.items():
            value = usage.get(source)
            if type(value) is int and value >= 0:
                record[target] = value
