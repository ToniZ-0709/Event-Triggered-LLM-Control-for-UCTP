"""Cloudflare Workers AI adapter for the controller decision interface."""

import json

import httpx

from .cloudflare_pool import CloudflarePool, error_codes
from .decision_client import DecisionClient, record_usage

API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
LLAMA = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
MISTRAL = "@cf/mistralai/mistral-small-3.1-24b-instruct"
RESPONSE_MODES = {LLAMA: "json_schema", MISTRAL: "prompt_json"}


class CloudflareDecisionClient(DecisionClient):
    provider = "cloudflare"
    api = "workers_ai_run"
    api_base_url = API_BASE_URL

    def __init__(self, model: str, timeout_seconds: float = 60.0,
                 max_output_tokens: int = 4096, *, free_tier_confirmed: bool = False,
                 min_request_interval_seconds: float = 10.0, max_input_characters: int = 60_000,
                 pool: CloudflarePool | None = None, transport: httpx.BaseTransport | None = None):
        if model not in RESPONSE_MODES:
            raise ValueError("Cloudflare model must be one of: " + ", ".join(RESPONSE_MODES))
        self.response_mode = RESPONSE_MODES[model]
        super().__init__(model, timeout_seconds, max_output_tokens, free_tier_confirmed=free_tier_confirmed,
                         min_request_interval_seconds=min_request_interval_seconds,
                         max_input_characters=max_input_characters,
                         load_pool=CloudflarePool if pool is None else lambda: pool, transport=transport)

    def system_prompt(self, prompt, schema):
        if self.model == MISTRAL:
            encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"))
            prompt += (" Return exactly one JSON object matching this schema, without Markdown "
                       "or extra text: " + encoded)
        return prompt

    def request_body(self, prompt, body, schema):
        request = {
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": body}],
            "max_tokens": self.max_output_tokens, "stream": False,
        }
        if self.model == LLAMA:
            request["response_format"] = {"type": "json_schema", "json_schema": schema}
        return request

    def send_once(self, request, credential, timeout):
        return self.client.post(f"{API_BASE_URL}/{credential.account_id}/ai/run/{self.model}",
                                headers={"Authorization": f"Bearer {credential.token}"},
                                json=request, timeout=timeout)

    def response_answer(self, response, record):
        from itc2019.llm import _json_pairs

        if not response.is_success:
            codes = error_codes(response)
            if codes:
                record["provider_error_codes"] = sorted(codes)
            if self.model == LLAMA and "json mode couldn't be met" in response.text.lower():
                record["status"] = "invalid_schema"
            else:
                self.blocked_reason = ("api_billing_required" if response.status_code == 402 or 5035 in codes
                                       else "api_configuration_error")
                record["status"] = self.blocked_reason
            return None

        envelope = json.loads(response.content, object_pairs_hook=_json_pairs)
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(result, dict) or envelope.get("success") is False:
            record["status"] = "api_error"
            return None
        record_usage(record, result.get("usage"), {
            "input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        })
        return result.get("response")
