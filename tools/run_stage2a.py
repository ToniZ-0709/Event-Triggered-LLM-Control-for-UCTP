"""Run the Stage 2A diagnostic slice with live per-job logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "phase2.json"
INSTANCES = ("bet-spr18", "muni-fsps-spr17", "iku-spr18")
METHODS = ("baseline", "random", "greedy", "single", "multi")
SEED = 17

sys.path.insert(0, str(ROOT))
import run_pilot


def stage_jobs(profile: dict, output: Path, *, confirm_free_tier: bool,
               wrap_hydrogen: bool) -> list[tuple[str, list[str]]]:
    jobs = []
    for method in METHODS:
        commands = run_pilot.commands(
            profile, pipeline=method, seed=SEED, instances=list(INSTANCES), model=None,
            output=output, confirm_free_tier=confirm_free_tier)
        for instance, command in zip(INSTANCES, commands, strict=True):
            if wrap_hydrogen and method in run_pilot.LLM_PIPELINES:
                usage_path = output / "http_usage" / f"{method}-{instance}.json"
                command = [command[0], "-B", __file__, "--hydrogen-worker",
                           str(usage_path), *command[1:]]
            jobs.append((method, command))
    return jobs


def campaign_modules():
    directory = ROOT / "ExpRun"
    if not (directory / "runner.py").is_file() or not (directory / "hydrogen_quota.py").is_file():
        raise RuntimeError("Stage 2A needs the local ExpRun runner and Hydrogen quota tracker.")
    sys.path.insert(0, str(directory))
    import hydrogen_quota
    import runner
    return runner, hydrogen_quota


def hydrogen_worker(usage_path: Path, command: list[str]) -> None:
    runner, quota = campaign_modules()
    keys = runner.read_keys()
    attempt_id = f"stage2a/{usage_path.parents[1].name}/{usage_path.stem}"
    slot = (int(quota.read_usage().get("next_key_slot") or 1) - 1) % len(keys) + 1
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = keys[slot - 1]
    os.environ["PHASE2_START_KEY_SLOT"] = str(slot)
    os.environ["PHASE2_ATTEMPT_ID"] = attempt_id
    sys.argv = [quota.__file__, *command]
    try:
        quota.main()
    finally:
        usage = quota.read_usage().get("current_attempt") or {}
        if usage.get("id") == attempt_id:
            counts = usage["requests_by_slot"]
            runner.atomic_json(usage_path, {
                "attempt_id": attempt_id, "http_requests_by_key_slot": counts,
                "http_requests_counted": sum(counts.values()),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-free-tier", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 15:
        parser.error("--workers must be between 1 and 15")
    if not args.dry_run and not args.confirm_free_tier:
        parser.error("Confirm the API projects are on the Gemini Free Tier, then pass -ConfirmFreeTier.")

    try:
        profile = json.loads(PROFILE.read_bytes())
        solver = profile["solver"]
        if (solver["evaluations"] != 1800 or solver["generations"] != 200 or
                solver.get("time_limit_seconds") is not None or solver["population"] != 20 or
                solver["partitions"] != 3 or any(profile["instances"][i] for i in INSTANCES) or
                profile["llm"]["provider"] != "gemini" or
                profile["llm"]["model"] != "gemini-4-hydrogen" or
                profile["llm"]["thinking_level"] != "low" or not profile["llm"]["free_tier_only"] or
                profile["llm"]["min_request_interval_seconds"] != 10 or
                profile["controller"]["api_timeout_seconds"] != 60 or
                profile["controller"]["max_reviews"] != 12 or
                profile["repair"]["neighborhood_seconds"] != 8 or
                profile["repair"]["neighborhood_deterministic_limit"] != 0.5):
            raise ValueError("profiles/phase2.json no longer matches the Stage 2A protocol.")
        if args.dry_run:
            preview = ROOT / "tmp" / "stage2a" / "preview"
            for _, command in stage_jobs(profile, preview, confirm_free_tier=args.confirm_free_tier,
                                         wrap_hydrogen=False):
                print(subprocess.list2cmdline(command), flush=True)
            return 0

        runner, quota = campaign_modules()
        lock = runner.acquire_lock()
        if lock is None:
            raise RuntimeError("Another ExpRun batch holds the shared campaign lock.")
        with lock:
            keys = runner.read_keys()
            usage = quota.read_usage()
            recorded = usage["key_count"]
            if len(keys) < recorded or quota.pool_digest(keys[:recorded]) != usage["key_order_sha256"]:
                raise RuntimeError("The .env key order does not match the Hydrogen quota ledger.")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            output = ROOT / "tmp" / "stage2a" / stamp
            output.mkdir(parents=True, exist_ok=False)
            jobs = stage_jobs(profile, output, confirm_free_tier=True, wrap_hydrogen=True)
            print(f"Stage 2A: {len(jobs)} jobs, seed {SEED}, {args.workers} worker(s)", flush=True)
            print(f"Results: {output}", flush=True)
            return run_pilot.run_parallel(jobs, workers=args.workers,
                                          log_root=output / "pilot_logs", live=True)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Stage 2A setup failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    if sys.argv[1:2] == ["--hydrogen-worker"]:
        hydrogen_worker(Path(sys.argv[2]), sys.argv[3:])
    else:
        raise SystemExit(main())
