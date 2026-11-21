"""Run ITC 2019 experiment jobs with an isolated solver budget and logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.settings import DEFAULT_SETTINGS_PATH, load_experiment_settings, resolve_model_profile

ENTRY = {"baseline": ROOT / "NSGA-III/main.py",
         "random": ROOT / "Random Controller/main.py",
         "greedy": ROOT / "Greedy Controller/main.py",
         "single": ROOT / "Single-Agent Controller/main.py",
         "multi": ROOT / "Multi-Agent Controller/main.py"}
LLM_PIPELINES = {"single", "multi"}
SOLVER_ONLY_KEYS = {"seed"}
LLM_SETTING_KEYS = {
    "model_profile",
    "model",
    "provider",
    "thinking_level",
    "free_tier_only",
    "min_request_interval_seconds",
    "max_input_characters",
    "api_timeout_seconds",
    "max_output_tokens",
}
CONTROLLER_SHARED_SKIP = {"recent_outcomes"}
API_CONTROLLER_FLAGS = {"api_timeout_seconds", "max_output_tokens", "max_api_calls", "thinking_level"}


def commands(profile: dict, *, pipeline: str, seed: int,
             instances: list[str], model: str | None, output: Path | None,
             confirm_free_tier: bool = False,
             thinking_level: str | None = None) -> list[list[str]]:
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported experiment profile schema")
    selected = dict.fromkeys(instances or profile["instances"])
    result = []
    for instance in selected:
        if instance not in profile["instances"]:
            raise ValueError(f"Instance not configured in profile: {instance}")
        command = [sys.executable, str(ENTRY[pipeline]), "--instance", instance, "--seed", str(seed)]
        local = profile["instances"][instance]
        solver_and_limits = {key: value for key, value in local.items()
                             if not key.startswith("repair_") and key not in SOLVER_ONLY_KEYS}
        base_solver = {key: value for key, value in profile["solver"].items()
                       if key not in SOLVER_ONLY_KEYS}
        for key, value in {**base_solver, **solver_and_limits}.items():
            if value is not None:
                command.extend(("--" + key.replace("_", "-"), str(value)))
        local_repair = {key.removeprefix("repair_"): value for key, value in local.items()
                        if key.startswith("repair_")}
        for key, value in {**profile.get("repair", {}), **local_repair}.items():
            command.extend(("--repair-" + key.replace("_", "-"), str(value)))
        llm = profile.get("llm", {})
        controller = dict(profile["controller"])
        if pipeline != "baseline":
            if pipeline in LLM_PIPELINES:
                controller["max_api_calls"] = controller["max_reviews"] * (2 if pipeline == "multi" else 1)
                for shared_key in ("api_timeout_seconds", "max_output_tokens"):
                    if shared_key not in controller and shared_key in llm:
                        controller[shared_key] = llm[shared_key]
            for key, value in controller.items():
                if key in CONTROLLER_SHARED_SKIP:
                    continue
                if pipeline not in LLM_PIPELINES and key in API_CONTROLLER_FLAGS:
                    continue
                command.extend(("--" + key.replace("_", "-"), str(value)))
        if pipeline in LLM_PIPELINES:
            try:
                resolved = resolve_model_profile(model, experiment_settings=profile)
                provider = resolved["provider"]
                model_arg = resolved["model"]
                default_thinking = resolved.get("thinking_level")
            except ValueError:
                if model is None:
                    raise
                provider = llm.get("provider", "gemini")
                model_arg = model
                default_thinking = llm.get("thinking_level")
            command.extend(("--provider", provider, "--model", model_arg))
            chosen_thinking = thinking_level or (None if model and provider != "gemini" else default_thinking) or controller.get("thinking_level")
            if chosen_thinking:
                command.extend(("--thinking-level", chosen_thinking))
            command.extend(("--api-min-interval-seconds",
                            str(llm.get("min_request_interval_seconds", 10.0))))
            command.extend(("--api-max-input-characters",
                            str(llm.get("max_input_characters", 60000))))
            if confirm_free_tier:
                command.append("--confirm-free-tier")
        if output:
            command.extend(("--output", str(output.resolve() / pipeline)))
        result.append(command)
    return result


def show_progress(log: BinaryIO, label: str, *, finished: bool) -> None:
    while True:
        start = log.tell()
        line = log.readline()
        if not line:
            return
        if not finished and not line.endswith(b"\n"):
            log.seek(start)
            return
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        print(f"[{label}] {text}", file=sys.stderr, flush=True)


def run_parallel(work: list[tuple[str, list[str]]], *, workers: int, log_root: Path,
                 live: bool = False) -> int:
    """Fill worker slots while allowing at most one active LLM job in this batch."""
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=log_root.resolve()))
    print(f"Worker logs: {log_dir}", flush=True)
    pending = list(enumerate(work, 1))
    running: dict[subprocess.Popen, tuple[str, str, BinaryIO | None]] = {}
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"} if live else None
    failure = 0
    try:
        while running or (pending and not failure):
            for process, (_, label, log) in list(running.items()):
                code = process.poll()
                if log is not None:
                    show_progress(log, label, finished=code is not None)
                if code is not None:
                    del running[process]
                    if log is not None:
                        log.close()
                    print(f"[exit {code}] {label}", flush=True)
                    if code and not failure:
                        failure = code

            while not failure and len(running) < workers:
                api_busy = any(method in LLM_PIPELINES for method, _, _ in running.values())
                # Give a waiting LLM job the next free slot so offline jobs
                # can overlap its API waits.
                index = None if api_busy else next(
                    (i for i, (_, (method, _)) in enumerate(pending) if method in LLM_PIPELINES),
                    None)
                if index is None:
                    index = next((i for i, (_, (method, _)) in enumerate(pending)
                                  if method not in LLM_PIPELINES), None)
                if index is None:
                    break
                ordinal, (method, command) = pending.pop(index)
                instance = command[command.index("--instance") + 1]
                label = f"{method}/{instance}"
                job_dir = log_dir / f"{ordinal:03d}-{method}"
                job_dir.mkdir()
                progress_log = job_dir / "stderr.log"
                with (job_dir / "stdout.log").open("wb") as stdout, progress_log.open("wb") as stderr:
                    log = progress_log.open("rb") if live else None
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, env=child_env,
                                                   stdout=stdout, stderr=stderr)
                    except BaseException:
                        if log is not None:
                            log.close()
                        raise
                    running[process] = method, label, log
                print(f"[start] {label} -> {progress_log}", flush=True)
            if running:
                time.sleep(0.1)
    finally:
        for process in running:
            if process.poll() is None:
                process.terminate()
        for process, (_, _, log) in running.items():
            process.wait()
            if log is not None:
                log.close()
    return failure


def main() -> int:
    defaults = load_experiment_settings()
    default_seed = int(defaults.get("solver", {}).get("seed", 17))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--pipeline", choices=ENTRY, action="append",
                        help="Pipeline to run (repeatable, default: baseline)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Maximum simultaneous jobs (default: 1, at most one LLM job)")
    parser.add_argument("--live", action="store_true",
                        help="Show each parallel job's progress in this terminal as well as its log")
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--model", help="Model profile name, model ID, or profile JSON path")
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"))
    parser.add_argument("--confirm-free-tier", action="store_true",
                        help="Pass only after checking the API credentials are on Free Tier")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        profile = load_experiment_settings(args.profile)
        requested_model = args.model or os.environ.get("LLM_MODEL") or os.environ.get("GEMINI_MODEL")
        requested_thinking = (args.thinking_level
                              or os.environ.get("LLM_THINKING_LEVEL")
                              or os.environ.get("GEMINI_THINKING_LEVEL")
                              or None)
        pipelines = list(dict.fromkeys(args.pipeline or ["baseline"]))
        if any(method in LLM_PIPELINES for method in pipelines):
            resolved_model = resolve_model_profile(requested_model, experiment_settings=profile)
            if requested_thinking is not None and resolved_model.get("provider") != "gemini":
                raise ValueError(f"Model profile {resolved_model['profile_name']!r} does not accept --thinking-level")
        work = [
            (method, command)
            for method in pipelines
            for command in commands(profile, pipeline=method, seed=args.seed,
                                    instances=args.instance,
                                    model=requested_model,
                                    output=args.output, confirm_free_tier=args.confirm_free_tier,
                                    thinking_level=requested_thinking)
        ]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        if args.workers > 1 and work and not args.dry_run:
            output = args.output or ROOT / "ExperimentResult"
            return run_parallel(work, workers=args.workers, log_root=output / "logs",
                                live=args.live)
        for _, command in work:
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                result = subprocess.run(command, cwd=ROOT)
                if result.returncode:
                    return result.returncode
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
