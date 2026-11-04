"""Resume the frozen expanded pilot sequentially, with an auditable retry policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pilot import commands
from tools.summarize_paired_pilot import CORE_FILES, METHOD_ROOTS, audit_attempts, digest, discover
from tools.summarize_expanded_pilot import DEFAULT_REPORT, DEFAULT_STATE_ROOT, completed, save_report

METHODS = ("baseline", "single", "multi")
NEW_INSTANCES = ("pu-cs-fal07", "lums-spr18", "tg-spr18")
TRANSIENT = {"api_service_error", "api_network_error", "api_timeout"}
STATE_ROOT = DEFAULT_STATE_ROOT


def unpublished_failures(state_root: Path) -> list[dict]:
    path = state_root / "storage_failures.json"
    return json.loads(path.read_text(encoding="utf-8"))["failures"] if path.is_file() else []


def continuation_allowed(attempts: list[dict], unpublished: list[dict]) -> bool:
    # The one known export failure may be retried only after explicit human approval.
    # It counts toward the two-attempt limit; it never becomes a model-quality result.
    if unpublished:
        return (len(unpublished) == 1 and not attempts and
                unpublished[0].get("retry_authorized") is True)
    return retry_allowed(attempts)


def next_attempt_number(state_root: Path, seed: int, instance: str, method: str,
                        attempts: list[dict], unpublished: list[dict]) -> int:
    used = [len(attempts), *(record["attempt"] for record in unpublished)]
    for path in state_root.glob(f"s{seed}_{instance}_{method}_attempt*.log"):
        used.append(int(path.stem.rsplit("attempt", 1)[1]))
    return max(used, default=0) + 1


def compact_output(method: str, seed: int, number: int, tag: str = "x") -> Path:
    # The run fingerprint already distinguishes instances; do not repeat long names.
    if not re.fullmatch(r"[a-z0-9]{1,8}", tag):
        raise ValueError("Output tag must be 1-8 lowercase letters or digits")
    return ROOT / METHOD_ROOTS[method] / "results_itc2019" / tag / f"{seed}-{number}"


def check_output_paths(profile: dict, tag: str = "x") -> int:
    longest = 0
    for seed, instance, method in jobs(profile):
        config = {**profile["solver"], **profile["instances"][instance]}
        run_id = (f"run_n1_p{config['population']}_g{config['generations']}"
                  f"_r{config['partitions']}_s{seed}_" + "0" * 16)
        # Include both staging and the longer publication-backup spelling.
        for suffix in (".tmp-" + "0" * 8, ".previous-" + "0" * 32):
            predicted = (compact_output(method, seed, 99, tag).resolve() / method /
                         ("." + run_id + suffix) / instance / "official_validation.json")
            longest = max(longest, len(str(predicted)))
    if longest > 255:
        raise ValueError(f"Output path preflight failed ({longest} characters); shorten the workspace/output root before any API call")
    return longest


def jobs(profile: dict) -> list[tuple[int, str, str]]:
    original = [name for name in profile["instances"] if name not in NEW_INSTANCES]
    additional = [name for name in NEW_INSTANCES if name in profile["instances"]]
    return ([(17, name, method) for name in additional for method in METHODS] +
            [(17, name, method) for name in original for method in METHODS] +
            [(42, name, method) for name in profile["instances"] for method in METHODS])


def retry_allowed(attempts: list[dict]) -> bool:
    """At most two whole-run attempts; never auto-retry quota/auth or unknown failures."""
    if not attempts:
        return True
    failures = {key for attempt in attempts for key, count in attempt["api_failures"].items() if count}
    return len(attempts) < 2 and bool(failures) and failures <= TRANSIENT


def human_recovery_allowed(state_root: Path, seed: int, instance: str, method: str,
                           number: int, attempts: list[dict], unpublished: list[dict],
                           profile_sha256: str) -> bool:
    """Permit an exact human recovery, retaining any explicitly recorded user stops."""
    if number < 3 or len(attempts) + len(unpublished) != number - 1:
        return False
    stopped_numbers = []
    for stopped in unpublished:
        stopped_number = stopped.get("attempt")
        log = stopped.get("log")
        if (type(stopped_number) is not int or
                stopped.get("status") != "user_interrupted_unpublished_attempt" or
                (stopped.get("seed"), stopped.get("instance"), stopped.get("method")) !=
                (seed, instance, method) or
                stopped.get("comparison_eligible") is not False or
                stopped.get("retry_authorized") is not False or
                not isinstance(log, str) or not log):
            return False
        log_path = Path(log)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        expected = state_root / f"s{seed}_{instance}_{method}_attempt{stopped_number}.log"
        if log_path.resolve() != expected.resolve() or not log_path.is_file():
            return False
        stopped_numbers.append(stopped_number)
    if sorted(stopped_numbers) != list(range(len(attempts) + 1, number)):
        return False
    for attempt in attempts:
        failures = {key for key, count in attempt["api_failures"].items() if count}
        if (attempt.get("completion_status") != "incomplete_api_failure" or
                not failures or not failures <= TRANSIENT):
            return False
    path = state_root / "human_recoveries.json"
    if not path.is_file():
        return False
    previous = sorted(attempt["folder"] for attempt in attempts)
    if len(set(previous)) != len(previous):
        return False
    return any(record.get("authorized") is True and
               (record.get("seed"), record.get("instance"), record.get("method"),
                record.get("attempt")) == (seed, instance, method, number) and
               record.get("profile_sha256") == profile_sha256 and
               sorted(record.get("previous_attempt_folders", [])) == previous and
               record.get("previous_unpublished_attempts", []) == unpublished
               for record in json.loads(path.read_text(encoding="utf-8"))["recoveries"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/expanded_pilot.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--output-tag", default="x", help="Short results namespace; keep campaigns separate")
    parser.add_argument("--confirm-free-tier", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile["llm"]["model"] != "gemini-3.5-flash-lite":
        parser.error("This campaign is pinned to gemini-3.5-flash-lite")
    if not args.dry_run and (not args.confirm_free_tier or not os.environ.get("GEMINI_API_KEY")):
        parser.error("Set GEMINI_API_KEY locally and confirm the previously verified Free Tier")
    longest = check_output_paths(profile, args.output_tag)
    print(json.dumps({"status": "output_path_preflight", "longest_predicted_path": longest}), flush=True)
    frozen = {str(ROOT / "NSGA-III" / name): digest(ROOT / "NSGA-III" / name) for name in CORE_FILES}
    for folder in ("Single-Agent Controller", "Multi-Agent Controller"):
        for name in ("main.py", "controller.py"):
            frozen[str(ROOT / folder / name)] = digest(ROOT / folder / name)
    frozen[str(ROOT / "NSGA-III/itc2019/llm.py")] = digest(ROOT / "NSGA-III/itc2019/llm.py")
    frozen[str(args.profile.resolve())] = digest(args.profile)
    state_root = args.state_root.resolve()
    no_time_cap = any(settings.get("time_limit_seconds") is None
                      for settings in profile["instances"].values())
    if no_time_cap and state_root == STATE_ROOT.resolve():
        parser.error("Use a separate --state-root for a no-time-cap follow-up")
    if no_time_cap and args.report.resolve() == DEFAULT_REPORT.resolve():
        parser.error("Use a separate --report for a no-time-cap follow-up")

    def publish_report() -> dict:
        return save_report(args.profile, [17, 42], args.report,
                           state_root=state_root, allow_no_time_limit=no_time_cap)

    if not args.dry_run:
        state_root.mkdir(parents=True, exist_ok=True)
        frozen_path = state_root / "frozen_inputs.json"
        if frozen_path.is_file():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
                raise RuntimeError("Frozen solver/controller/profile differs from the original campaign; refusing to resume")
        else:
            frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    for index, (seed, instance, method) in enumerate(jobs(profile), 1):
        if any(digest(Path(path)) != expected for path, expected in frozen.items()):
            raise RuntimeError("Frozen solver/controller/profile changed; campaign stopped")
        folder = discover(profile, instance, seed, method)
        if completed(folder, method):
            print(json.dumps({"job": index, "status": "reuse_completed", "seed": seed,
                              "instance": instance, "method": method}), flush=True)
            continue
        if args.dry_run:
            print(json.dumps({"job": index, "status": "pending", "seed": seed,
                              "instance": instance, "method": method}), flush=True)
            continue
        while True:
            attempts = [record for record in audit_attempts(profile, seed, (method,))
                        if record["instance"] == instance]
            unpublished = [record for record in unpublished_failures(state_root)
                           if (record["seed"], record["instance"], record["method"]) == (seed, instance, method)]
            number = next_attempt_number(state_root, seed, instance, method, attempts, unpublished)
            human_recovery = human_recovery_allowed(
                state_root, seed, instance, method, number, attempts, unpublished,
                frozen[str(args.profile.resolve())])
            if not continuation_allowed(attempts, unpublished) and not human_recovery:
                publish_report()
                raise RuntimeError(f"Retry policy stopped {seed}/{instance}/{method}; inspect retained attempts")
            if attempts:
                print(json.dumps({"status": "transient_retry_wait", "seconds": 30,
                                  "seed": seed, "instance": instance, "method": method}), flush=True)
                time.sleep(30)
            output = compact_output(method, seed, number, args.output_tag)
            log = state_root / f"s{seed}_{instance}_{method}_attempt{number}.log"
            command = commands(profile, pipeline=method, seed=seed, instances=[instance],
                               model=profile["llm"]["model"], output=output,
                               confirm_free_tier=args.confirm_free_tier)[0]
            print(json.dumps({"job": index, "status": "start", "seed": seed, "instance": instance,
                              "method": method, "attempt": number, "log": str(log),
                              "human_authorized_recovery": human_recovery}), flush=True)
            (state_root / "current_job.json").write_text(json.dumps({
                "seed": seed, "instance": instance, "method": method, "attempt": number,
                "log": str(log), "output": str(output),
                "human_authorized_recovery": human_recovery}, indent=2) + "\n", encoding="utf-8")
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            report = publish_report()
            folder = discover(profile, instance, seed, method)
            print(json.dumps({"job": index, "status": "finished", "exit_code": result.returncode,
                              "completed_cells": report["completed_cells"],
                              "completed_triples": report["completed_triples"]}), flush=True)
            if completed(folder, method):
                break
            attempts_after = [record for record in audit_attempts(profile, seed, (method,))
                              if record["instance"] == instance]
            if len(attempts_after) == len(attempts):
                failures = unpublished_failures(state_root)
                failures.append({"instance": instance, "seed": seed, "method": method,
                    "attempt": number, "status": "unpublished_execution_failure",
                    "log": str(log.relative_to(ROOT)), "exit_code": result.returncode,
                    "comparison_eligible": False, "retry_authorized": False,
                    "api_calls": None, "reported_native_tokens": None,
                    "usage_note": "Execution ended without a published result; API usage may be unknown. Inspect the retained log before any restart."})
                (state_root / "storage_failures.json").write_text(
                    json.dumps({"failures": failures}, indent=2) + "\n", encoding="utf-8")
                (state_root / "current_job.json").write_text(json.dumps({
                    "status": "stopped_unpublished_failure", "seed": seed, "instance": instance,
                    "method": method, "attempt": number, "log": str(log)}) + "\n", encoding="utf-8")
                publish_report()
                raise RuntimeError(f"No committed result for {seed}/{instance}/{method}; inspect {log}")
    if not args.dry_run:
        report = publish_report()
        (state_root / "current_job.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
        print(json.dumps({"status": "campaign_complete", "completed_cells": report["completed_cells"],
                          "completed_triples": report["completed_triples"], "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
