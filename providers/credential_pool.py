"""Credential rotation, request pacing, and quota ledgers for LLM providers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
MAX_KEY_ERRORS = 3
COOLDOWN_SECONDS = 60.0
POLL_SECONDS = 15.0
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True, repr=False)
class Credential:
    token: str
    account_id: str = ""


@dataclass(frozen=True)
class Quota:
    timezone: tzinfo = timezone.utc
    daily_requests: int | None = None
    minute_requests: int | None = None
    minute_input_tokens: int | None = None

    def __post_init__(self):
        for limit in (self.daily_requests, self.minute_requests, self.minute_input_tokens):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Quota limits must be positive integers.")


def numbered_entries(path: Path, *names: str) -> list[dict[str, str]]:
    prefixes = tuple(name + "_" for name in names)
    pattern = re.compile(rf"({'|'.join(names)})_([0-9]{{3,}})")
    slots: dict[int, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("#")[0].partition("=")
        name = name.strip()
        if not name.startswith(prefixes):
            continue
        match = pattern.fullmatch(name)
        if match is None or not separator:
            raise ValueError("Expected numbered credential entries in .env.")
        kind, number = match.groups()
        slot = int(number)
        values = slots.setdefault(slot, {})
        if kind in values or number != f"{slot:03d}":
            raise ValueError(f"Invalid or duplicate credential entry for slot {slot}.")
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values[kind] = value

    if not slots or sorted(slots) != list(range(1, len(slots) + 1)):
        raise ValueError("Number credentials consecutively from 001 in .env.")
    return [slots[slot] for slot in range(1, len(slots) + 1)]


@contextmanager
def _try_lock(path: Path):
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            yield False
        else:
            yield True
        # Closing the handle releases the lock, including when a process exits.


def _save(path: Path, state: dict) -> None:
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def response_object(response: httpx.Response) -> dict:
    try:
        document = response.json()
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


class CredentialPool:
    """One ordered credential pool with a ledger for the provider's quota scope."""

    def __init__(self, env_path: Path, usage_path: Path, *, read_credentials, quota: Quota,
                 daily_quota_error, input_tokens, prior_requests=None,
                 clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        self.env_path = env_path
        self.usage_path = usage_path
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep
        self.read_credentials = read_credentials
        self.quota = quota
        self.daily_quota_error = daily_quota_error
        self.input_tokens = input_tokens
        self.prior_requests = prior_requests
        self.accounts = read_credentials(env_path)

    def _state(self, accounts: list[Credential], day: str) -> tuple[dict, dict, list[str]]:
        path = self.usage_path
        state = json.loads(path.read_bytes()) if path.exists() else {"days": {}}
        identities = [item.account_id or item.token for item in accounts]
        saved = state.pop("ordered_account_digests", None)
        if saved is not None:
            fingerprints = [hashlib.sha256(item.encode()).hexdigest() for item in identities[:len(saved)]]
            if fingerprints != saved:
                raise ValueError("Credential order changed. Append new slots after the existing slots.")
        count = state.get("credential_count", 0)
        if count and (len(identities) < count or
                      pool_digest(identities[:count]) != state["credential_order_sha256"]):
            raise ValueError("Credential order changed. Append new slots after the existing slots.")
        state.update(credential_count=len(identities), credential_order_sha256=pool_digest(identities))
        first_slots: dict[str, str] = {}
        account_slots = [first_slots.setdefault(identity, str(slot))
                         for slot, identity in enumerate(identities, 1)]
        old_day = state.pop("utc_day", None)
        if state.get("quota_day", old_day) != day:
            state["next_key_slot"] = 1
        state["quota_day"] = day
        today = state["days"].setdefault(day, {"accounts": {}, "slots": {}})
        return state, today, account_slots

    def _select(self, state: dict, today: dict, account_slots: list[str], now: float,
                next_reset: float, prior: dict[str, int],
                min_interval: float) -> tuple[int | None, float, str]:
        count = len(account_slots)
        start = state["next_key_slot"]
        cooldown_wait = None
        last_starts = state.get("last_started_by_account_slot", {})
        for offset in range(count):
            slot = (start - 1 + offset) % count + 1
            account = account_slots[slot - 1]
            usage = today["accounts"].get(account, {})
            requests = usage.get("http_requests", 0) + prior.get(account, 0)
            if usage.get("daily_exhausted") or (
                self.quota.daily_requests is not None and requests >= self.quota.daily_requests
            ):
                continue
            entry = today["slots"].get(str(slot), {})
            cooldown = entry.get("cooldown_until", 0.0)
            paced = max(last_starts.get(account, 0.0) + min_interval,
                        entry.get("retry_not_before", 0.0),
                        self._minute_ready(state, account, now))
            if cooldown > now:
                ready = max(cooldown, paced)
                cooldown_wait = ready if cooldown_wait is None else min(cooldown_wait, ready)
                continue
            if paced > now:
                return None, paced - now, "pacing"
            return slot, 0.0, ""
        if cooldown_wait is not None:
            return None, cooldown_wait - now, "cooldown"
        return None, next_reset - now, "daily_quota"

    def _minute_ready(self, state: dict, account: str, now: float) -> float:
        recent_by_slot = state.setdefault("recent_requests_by_account_slot", {})
        recent = [event for event in recent_by_slot.get(account, ()) if event[0] > now - 60]
        recent_by_slot[account] = recent
        ready = 0.0
        limit = self.quota.minute_requests
        if limit is not None and len(recent) >= limit:
            ready = recent[-limit][0] + 60
        limit = self.quota.minute_input_tokens
        if limit is not None:
            tokens = sum(event[1] or 0 for event in recent)
            for started, amount in recent:
                if tokens < limit:
                    break
                ready = max(ready, started + 60)
                tokens -= amount or 0
        return ready

    def send(self, send_once, *, timeout_seconds: float, min_interval: float, deadline: float | None,
             record: dict) -> httpx.Response | None:
        path = self.usage_path
        path.parent.mkdir(parents=True, exist_ok=True)
        while deadline is None or deadline - self.monotonic() > 0.1:
            wait, reason = 0.1, "pool_busy"
            with _try_lock(path.with_suffix(".lock")) as locked:
                if locked:
                    try:
                        accounts = self.read_credentials(self.env_path)
                    except (OSError, ValueError):
                        # An editor can expose an incomplete pair while saving an append.
                        accounts = self.accounts
                    now = self.clock()
                    local_day = datetime.fromtimestamp(now, self.quota.timezone).date()
                    day = local_day.isoformat()
                    next_reset = datetime.combine(local_day + timedelta(days=1), datetime.min.time(),
                                                  self.quota.timezone).timestamp()
                    state, today, account_slots = self._state(accounts, day)
                    self.accounts = accounts
                    prior = self.prior_requests(accounts, day) if self.prior_requests else {}
                    slot, wait, reason = self._select(
                        state, today, account_slots, now, next_reset, prior, min_interval)
                    if slot is not None:
                        remaining = timeout_seconds if deadline is None else deadline - self.monotonic()
                        if remaining <= 0.1:
                            return None
                        response, wait = self._send(
                            send_once, accounts[slot - 1], slot, account_slots[slot - 1],
                            now, min(timeout_seconds, remaining), state, today, record)
                        if response is not None:
                            return response
                        reason = "retry"
                    elif (state.get("waiting") != reason or
                          state.get("waiting_quota_day") != day):
                        state.update(waiting=reason, waiting_quota_day=day)
                        _save(path, state)
            wait = min(POLL_SECONDS, max(0.05, wait))
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - self.monotonic()))
            if reason == "pacing":
                record["pacing_wait_seconds"] += wait
            elif reason == "daily_quota":
                record["quota_wait_seconds"] += wait
            self.sleep(wait)
        return None

    def _send(self, send_once, account: Credential, slot: int, account_slot: str,
              now: float, timeout: float, state: dict, today: dict,
              record: dict) -> tuple[httpx.Response | None, float]:
        usage = today["accounts"].setdefault(account_slot, {
            "http_requests": 0, "http_ok": 0, "http_errors": 0,
        })
        entry = today["slots"].setdefault(str(slot), {"consecutive_errors": 0})
        state.pop("waiting", None)
        state.pop("waiting_quota_day", None)
        state.pop("waiting_utc_day", None)  # Pre-unification Cloudflare ledgers.
        state.setdefault("last_started_by_account_slot", {})[account_slot] = now
        state["next_key_slot"] = slot
        usage["http_requests"] += 1
        recent_by_account = state["recent_requests_by_account_slot"]
        event = [now, None]
        recent_by_account[account_slot].append(event)
        _save(self.usage_path, state)

        record["api_requests"] = 1  # One logical controller request, like Gemini.
        per_slot = record["http_requests_by_key_slot"]
        per_slot[str(slot)] = per_slot.get(str(slot), 0) + 1
        try:
            response = send_once(account, timeout)
        except httpx.TransportError:
            response = None
        status = response.status_code if response is not None else None
        # Keep the next slot number so a newly appended account is considered before wrapping.
        next_slot = slot + 1
        delay = 0.0
        if response is not None and response.is_success:
            usage["http_ok"] += 1
            event[1] = self.input_tokens(response)
            entry.clear()
            entry["consecutive_errors"] = 0
        else:
            usage["http_errors"] += 1
            if status == 429 and self.daily_quota_error(response):
                usage["daily_exhausted"] = True
                entry["consecutive_errors"] = 0
                state["next_key_slot"] = next_slot
                response = None
                delay = 1.0
            elif status is None or status in _TRANSIENT_STATUSES:
                errors = entry["consecutive_errors"] + 1
                delay = float(2 ** (errors - 1))
                retry_at = self.clock() + delay
                retry_after = response.headers.get("Retry-After") if response is not None else None
                if retry_after and retry_after.isascii() and retry_after.isdecimal():
                    retry_at = max(retry_at, self.clock() + int(retry_after))
                entry["retry_not_before"] = retry_at
                if errors >= MAX_KEY_ERRORS:
                    entry.update(consecutive_errors=0, cooldown_until=self.clock() + COOLDOWN_SECONDS)
                    state["next_key_slot"] = next_slot
                    delay = 1.0
                else:
                    entry["consecutive_errors"] = errors
                response = None
        _save(self.usage_path, state)
        return response, delay


def pool_digest(identities: list[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()
