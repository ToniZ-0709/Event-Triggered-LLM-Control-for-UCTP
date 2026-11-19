"""Finish the LLM cells of an existing Stage 2A batch with live logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "phase2.json"
INSTANCES = ("bet-spr18", "muni-fsps-spr17", "iku-spr18")
OFFLINE_METHODS = ("baseline", "random", "greedy")
METHODS = ("single", "multi")
SEED = 17
MAX_OUTPUT_TOKENS = 4096
MODEL = "gemini-3.5-flash-lite"
KEY_SLOT = 1

sys.path.insert(0, str(ROOT))
import run_pilot


def completed_runs(profile: dict, output: Path, method: str) -> dict[str, dict]:
    found = {}
    for manifest in (output / method).glob("run_*/run_config.json"):
        config = json.loads(manifest.read_bytes())
        settings = config["settings"]
        row, = json.loads(manifest.with_name("summary.json").read_bytes())
        instance = row["instance"]
        expected = {k: v for k, v in profile["solver"].items() if k != "time_limit_seconds"}
        expected["seed"] = SEED
        if (config["completion_status"] != "completed" or instance not in INSTANCES or
                instance in found or row["evaluations"] != expected["evaluations"] or
                any(settings["config"].get(k) != v for k, v in expected.items()) or
                settings["time_limit_seconds"] is not None or
                any(settings["repair"].get(k) != v for k, v in profile["repair"].items())):
            raise ValueError(f"Existing {method}/{instance} run does not match Stage 2A.")
        if method in METHODS:
            controller = settings["controller"]
            if (controller["max_output_tokens"] != MAX_OUTPUT_TOKENS or
                    controller["model"] != profile["llm"]["model"] or
                    not row["controller"]["comparison_eligible"]):
                raise ValueError(f"Existing {method}/{instance} run does not match the new LLM settings.")
        elif row.get("controller", {}).get("api_calls", 0):
            raise ValueError(f"Expected zero API calls for {method}/{instance}.")
        found[instance] = settings
    return found


def verify_workspace(completed: dict[str, dict[str, dict]]) -> None:
    for method, runs in completed.items():
        for instance, settings in runs.items():
            if any(
                hashlib.sha256((ROOT / "NSGA-III" / name).read_bytes()).hexdigest() != digest
                for name, digest in settings["solver_sources_sha256"].items()
            ):
                raise ValueError(f"Solver sources differ from the existing {method}/{instance} run.")
            dataset, = settings["instances"]
            dataset_path = ROOT / "Dataset_ITC2019" / f"{instance}.xml"
            if (dataset["file"] != dataset_path.name or
                    hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset["sha256"]):
                raise ValueError(f"Dataset differs from the existing {method}/{instance} run.")


def stage_jobs(profile: dict, output: Path, *, confirm_free_tier: bool,
               track_http: bool) -> list[tuple[str, list[str]]]:
    completed = {method: completed_runs(profile, output, method)
                 for method in (*OFFLINE_METHODS, *METHODS)}
    for method in OFFLINE_METHODS:
        if set(completed[method]) != set(INSTANCES):
            raise ValueError(f"The batch is missing completed {method} results for Stage 2A.")
    jobs = []
    for method in METHODS:
        instances = [instance for instance in INSTANCES if instance not in completed[method]]
        if not instances:
            continue
        commands = run_pilot.commands(
            profile, pipeline=method, seed=SEED, instances=instances, model=None,
            output=output, confirm_free_tier=confirm_free_tier)
        for instance, command in zip(instances, commands, strict=True):
            if track_http:
                usage_path = output / "http_usage" / f"{method}-{instance}.json"
                command = [command[0], "-B", __file__, "--flash-lite-worker",
                           str(usage_path), *command[1:]]
            jobs.append((method, command))
    if jobs:
        verify_workspace(completed)
    return jobs


def campaign_modules():
    directory = ROOT / "ExpRun"
    if not (directory / "runner.py").is_file() or not (directory / "flash_lite_quota.py").is_file():
        raise RuntimeError("Stage 2A needs the local ExpRun runner and Flash-Lite quota tracker.")
    sys.path.insert(0, str(directory))
    import flash_lite_quota
    import runner
    return runner, flash_lite_quota


def flash_lite_worker(usage_path: Path, command: list[str]) -> None:
    import httpx

    runner, quota = campaign_modules()
    keys = runner.read_keys()
    worker = Path(command[0]).resolve(strict=True)
    if worker not in {run_pilot.ENTRY[method].resolve() for method in METHODS}:
        raise RuntimeError("Unexpected Stage 2A worker.")
    if quota.MODEL != MODEL:
        raise RuntimeError("Flash-Lite quota tracker does not match the configured model.")
    attempt_id = f"stage2a/{usage_path.parents[1].name}/{usage_path.stem}"
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = keys[KEY_SLOT - 1]
    usage = quota.FlashLiteUsage()
    original = httpx.Client.send
    requests = 0

    def tracked_send(client, request, *args, **kwargs):
        nonlocal requests
        if request.url.host != quota.HOST:
            return original(client, request, *args, **kwargs)
        if (request.url.scheme, request.url.path, request.method) != (
                "https", quota.GENERATE_PATH, "POST"):
            raise RuntimeError("Unexpected Gemini endpoint in a Stage 2A Flash-Lite worker.")
        requests += 1
        try:
            response = usage.send(original, client, request, *args, **kwargs)
        except httpx.TransportError as exc:
            print(f"[Flash-Lite HTTP] {type(exc).__name__}", file=sys.stderr, flush=True)
            raise
        if response.is_error:
            print(f"[Flash-Lite HTTP] status={response.status_code}", file=sys.stderr, flush=True)
        return response

    httpx.Client.send = tracked_send
    sys.argv = command
    sys.path.insert(0, str(worker.parent))
    try:
        runpy.run_path(str(worker), run_name="__main__")
    finally:
        httpx.Client.send = original
        runner.atomic_json(usage_path, {
            "attempt_id": attempt_id, "model": MODEL,
            "http_requests_counted": requests,
            "http_requests_by_key_slot": {str(KEY_SLOT): requests} if requests else {},
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Existing timestamped folder under tmp/stage2a")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-free-tier", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}-\d{6}-\d{6}", args.batch):
        parser.error("--batch must be a timestamped Stage 2A folder name")
    if not 1 <= args.workers <= 15:
        parser.error("--workers must be between 1 and 15")
    if not args.dry_run and not args.confirm_free_tier:
        parser.error("Confirm the API projects are on the Gemini Free Tier, then pass -ConfirmFreeTier.")

    try:
        output = ROOT / "tmp" / "stage2a" / args.batch
        profile = json.loads(PROFILE.read_bytes())
        solver = profile["solver"]
        if (solver["evaluations"] != 1800 or solver["generations"] != 200 or
                solver.get("time_limit_seconds") is not None or solver["population"] != 20 or
                solver["partitions"] != 3 or any(profile["instances"][i] for i in INSTANCES) or
                profile["llm"]["provider"] != "gemini" or
                profile["llm"]["thinking_level"] != "low" or not profile["llm"]["free_tier_only"] or
                profile["llm"]["min_request_interval_seconds"] != 10 or
                profile["controller"]["api_timeout_seconds"] != 60 or
                profile["controller"]["max_reviews"] != 12 or
                profile["repair"]["neighborhood_seconds"] != 8 or
                profile["repair"]["neighborhood_deterministic_limit"] != 0.5):
            raise ValueError("profiles/phase2.json no longer matches the Stage 2A protocol.")
        profile["controller"]["max_output_tokens"] = MAX_OUTPUT_TOKENS
        profile["llm"]["model"] = MODEL
        if args.dry_run:
            jobs = stage_jobs(profile, output, confirm_free_tier=args.confirm_free_tier, track_http=False)
            if not jobs:
                print("Stage 2A is complete. No LLM jobs remain.", flush=True)
            for _, command in jobs:
                print(subprocess.list2cmdline(command), flush=True)
            return 0

        runner, _ = campaign_modules()
        lock = runner.acquire_lock()
        if lock is None:
            raise RuntimeError("Another ExpRun batch holds the shared campaign lock.")
        with lock:
            jobs = stage_jobs(profile, output, confirm_free_tier=True, track_http=True)
            if not jobs:
                print("Stage 2A is complete. No LLM jobs remain.", flush=True)
                return 0
            runner.read_keys()
            print(f"Stage 2A: {len(jobs)} jobs, {MODEL}, seed {SEED}, "
                  f"{MAX_OUTPUT_TOKENS} output tokens", flush=True)
            print(f"Results: {output}", flush=True)
            return run_pilot.run_parallel(jobs, workers=args.workers,
                                          log_root=output / "pilot_logs", live=True)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Stage 2A setup failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    if sys.argv[1:2] == ["--flash-lite-worker"]:
        flash_lite_worker(Path(sys.argv[2]), sys.argv[3:])
    else:
        raise SystemExit(main())
