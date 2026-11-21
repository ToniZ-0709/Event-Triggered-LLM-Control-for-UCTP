"""Load centralized experiment settings and construct controller clients."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .decision_client import DecisionClient

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = ROOT / "settings"
DEFAULT_SETTINGS_PATH = SETTINGS_DIR / "experiment.json"
MODELS_DIR = SETTINGS_DIR / "models"


def load_experiment_settings(path: Path | None = None) -> dict[str, Any]:
    """Load the centralized experiment configuration JSON."""
    target = DEFAULT_SETTINGS_PATH if path is None else path
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported experiment settings schema in {target}")
    return data


def available_model_profiles(models_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return configured model profiles keyed by profile stem."""
    directory = MODELS_DIR if models_dir is None else models_dir
    if not directory.is_dir():
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for file_path in sorted(directory.glob("*.json")):
        profiles[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))
    return profiles


def resolve_model_profile(
    selector: str | None = None,
    *,
    experiment_settings: dict[str, Any] | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a profile name, model ID, or profile JSON path into merged LLM settings."""
    exp = experiment_settings if experiment_settings is not None else load_experiment_settings()
    llm_defaults = {
        key: value
        for key, value in exp.get("llm", {}).items()
        if key in {
            "api_timeout_seconds",
            "max_output_tokens",
            "min_request_interval_seconds",
            "max_input_characters",
        }
    }
    chosen = (
        selector
        or os.environ.get("LLM_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or exp.get("llm", {}).get("model_profile")
        or exp.get("llm", {}).get("model")
        or "gemini_hydrogen"
    )
    candidate_path = Path(chosen)
    profile_name: str
    raw_profile: dict[str, Any]
    if chosen.endswith(".json") and candidate_path.is_file():
        profile_name = candidate_path.stem
        raw_profile = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        profiles = available_model_profiles(models_dir)
        if chosen in profiles:
            profile_name = chosen
            raw_profile = profiles[chosen]
        else:
            match = next(
                ((name, data) for name, data in profiles.items() if data.get("model") == chosen),
                None,
            )
            if match is None:
                available = ", ".join(sorted(profiles)) or "(none)"
                raise ValueError(f"Unknown model or profile: {chosen!r}. Available profiles: {available}")
            profile_name, raw_profile = match
    merged = {**llm_defaults, **raw_profile, "profile_name": profile_name}
    return merged


def _client_class_for_provider(provider: Any) -> type[DecisionClient]:
    """Dynamically resolve an installed provider's DecisionClient subclass."""
    if not isinstance(provider, str) or not provider.isidentifier():
        raise ValueError("Unsupported model provider.")
    try:
        module = importlib.import_module(f".{provider}", __package__)
    except ModuleNotFoundError as exc:
        raise ValueError("Unsupported model provider.") from exc
    for candidate in vars(module).values():
        if (
            isinstance(candidate, type)
            and issubclass(candidate, DecisionClient)
            and candidate is not DecisionClient
            and getattr(candidate, "provider", None) == provider
        ):
            return candidate
    raise ValueError("Unsupported model provider.")


def create_decision_client(
    settings: dict,
    *,
    thinking_level: str | None = None,
    free_tier_confirmed: bool = False,
    timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
    pool=None,
    transport=None,
):
    if settings.get("free_tier_only") is not True:
        raise ValueError("The model profile must specify free_tier_only.")
    resolved_timeout = settings.get("api_timeout_seconds", 60.0) if timeout_seconds is None else timeout_seconds
    resolved_max_tokens = settings.get("max_output_tokens", 4096) if max_output_tokens is None else max_output_tokens
    provider = settings.get("provider")
    model = settings.get("model")
    client_cls = _client_class_for_provider(provider)
    params = inspect.signature(client_cls.__init__).parameters
    options: dict[str, Any] = {
        "timeout_seconds": resolved_timeout,
        "max_output_tokens": resolved_max_tokens,
        "free_tier_confirmed": free_tier_confirmed,
        "min_request_interval_seconds": settings.get("min_request_interval_seconds", 10.0),
        "max_input_characters": settings.get("max_input_characters", 60_000),
        "pool": pool,
        "transport": transport,
    }
    if "thinking_level" in params:
        options["thinking_level"] = settings.get("thinking_level") if thinking_level is None else thinking_level
    elif thinking_level is not None or settings.get("thinking_level") is not None:
        raise ValueError(f"Provider {provider!r} does not accept thinking_level.")
    if "quota" in params:
        options["quota"] = settings["quota"]
    return client_cls(model, **options)
