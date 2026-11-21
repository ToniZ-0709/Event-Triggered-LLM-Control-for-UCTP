"""Gemini Developer API adapter using the shared credential pool."""

import json
from pathlib import Path
import sys

import httpx

from .decision_client import DecisionClient, record_usage
from .gemini_pool import GEMINI_MODELS, GeminiPool

API_BASE_URL = "https://generativelanguage.googleapis.com"
THINKING_LEVELS = ("low", "medium", "high")


def _ensure_solver_import_path() -> None:
    solver_root = str(Path(__file__).resolve().parents[1] / "NSGA-III")
    if solver_root not in sys.path:
        sys.path.insert(0, solver_root)


class GeminiDecisionClient(DecisionClient):
    provider = "gemini"
    api = "gemini_developer_generate_content"
    api_base_url = API_BASE_URL
    response_mode = "json_schema"

    def __init__(self, model: str, timeout_seconds: float = 60.0,
                 max_output_tokens: int = 4096, *, thinking_level: str, quota: dict,
                 free_tier_confirmed: bool = False, min_request_interval_seconds: float = 10.0,
                 max_input_characters: int = 60_000, pool: GeminiPool | None = None,
                 transport: httpx.BaseTransport | None = None):
        if model not in GEMINI_MODELS:
            raise ValueError("Gemini model must be one of: " + ", ".join(GEMINI_MODELS))
        if thinking_level not in THINKING_LEVELS:
            raise ValueError("Gemini thinking_level must be one of: " + ", ".join(THINKING_LEVELS))
        self.thinking_level = thinking_level
        super().__init__(model, timeout_seconds, max_output_tokens, free_tier_confirmed=free_tier_confirmed,
                         min_request_interval_seconds=min_request_interval_seconds,
                         max_input_characters=max_input_characters,
                         load_pool=lambda: GeminiPool(model, quota) if pool is None else pool,
                         transport=transport)

    def metadata(self):
        quota = self.pool.quota
        return {
            **super().metadata(),
            "thinking_level": self.thinking_level,
            "quota": {"daily_requests": quota.daily_requests, "minute_requests": quota.minute_requests,
                      "minute_input_tokens": quota.minute_input_tokens},
        }

    def request_body(self, prompt, body, schema):
        return {
            "systemInstruction": {"parts": [{"text": prompt}]},
            "contents": [{"role": "user", "parts": [{"text": body}]}],
            "generationConfig": {
                "responseMimeType": "application/json", "responseJsonSchema": schema,
                "candidateCount": 1, "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": self.thinking_level.upper(), "includeThoughts": False},
            },
        }

    def send_once(self, request, credential, timeout):
        return self.client.post(f"{API_BASE_URL}/v1beta/models/{self.model}:generateContent",
                                headers={"x-goog-api-key": credential.token}, json=request, timeout=timeout)

    def response_answer(self, response, record):
        _ensure_solver_import_path()
        from itc2019.llm import _json_pairs

        record["thinking_level"] = self.thinking_level
        if not response.is_success:
            self.blocked_reason = ("api_billing_required" if response.status_code == 402
                                   else "api_configuration_error")
            record["status"] = self.blocked_reason
            return None

        document = json.loads(response.content, object_pairs_hook=_json_pairs)
        if not isinstance(document, dict):
            record["status"] = "api_error"
            return None
        record["response_id"] = document.get("responseId")
        record["model_returned"] = document.get("modelVersion")
        record_usage(record, document.get("usageMetadata"), {
            "input_tokens": "promptTokenCount", "output_tokens": "candidatesTokenCount",
            "thinking_tokens": "thoughtsTokenCount", "total_tokens": "totalTokenCount",
            "cached_input_tokens": "cachedContentTokenCount",
        })
        candidates = document.get("candidates") or []
        candidate = (candidates[0] if isinstance(candidates, list) and candidates
                     and isinstance(candidates[0], dict) else {})
        reason = candidate.get("finishReason")
        record.update(finish_reason=reason, response_status=reason)
        feedback = document.get("promptFeedback")
        block = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if (block not in (None, "BLOCK_REASON_UNSPECIFIED", "BLOCKED_REASON_UNSPECIFIED") or
                reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION")):
            record["status"] = "model_refusal"
            return None
        if reason != "STOP":
            record["status"] = "incomplete_response"
            record["incomplete_reason"] = (
                "max_output_tokens_including_thinking" if reason == "MAX_TOKENS" else reason or "no_candidates")
            return None
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            return None
        return "".join(part["text"] for part in parts if isinstance(part, dict)
                       and not part.get("thought") and isinstance(part.get("text"), str))
