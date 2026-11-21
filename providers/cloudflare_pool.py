"""Cloudflare credentials and account quota policy."""

from pathlib import Path
import re

import httpx

from .credential_pool import (ROOT, ENV_PATH, Credential, CredentialPool, Quota,
                              numbered_entries, response_object)

USAGE_PATH = ROOT / "tmp" / "cloudflare" / "usage.json"
DAILY_QUOTA_CODE = 3036
ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
API_TOKEN = "CLOUDFLARE_API_TOKEN"


def read_accounts(path: Path) -> list[Credential]:
    accounts = []
    for slot, values in enumerate(numbered_entries(path, ACCOUNT_ID, API_TOKEN), 1):
        account_id = values.get(ACCOUNT_ID, "").lower()
        token = values.get(API_TOKEN, "")
        if not re.fullmatch(r"[a-f0-9]{32}", account_id) or not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token):
            raise ValueError(f"Expected a Cloudflare Account ID and API token for slot {slot}.")
        accounts.append(Credential(token=token, account_id=account_id))
    if len(set(accounts)) != len(accounts):
        raise ValueError("Cloudflare .env contains a duplicate credential pair.")
    return accounts


def error_codes(response: httpx.Response) -> set[int]:
    errors = response_object(response).get("errors")
    codes = set()
    if isinstance(errors, list):
        for error in errors:
            code = str(error.get("code", "")) if isinstance(error, dict) else ""
            if re.fullmatch(r"[0-9]{1,8}", code):
                codes.add(int(code))
    return codes


class CloudflarePool(CredentialPool):
    """Share one account cursor and daily quota for Llama and Mistral."""

    def __init__(self, env_path: Path = ENV_PATH, usage_path: Path = USAGE_PATH, **kwargs):
        super().__init__(env_path, usage_path, read_credentials=read_accounts, quota=Quota(),
                         daily_quota_error=lambda response: DAILY_QUOTA_CODE in error_codes(response),
                         input_tokens=lambda response: None, **kwargs)
