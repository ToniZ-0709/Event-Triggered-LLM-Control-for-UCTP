"""Collect matched baseline/single pilot runs without rerunning or altering them."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NSGA-III"))
from compare_pipelines import compare
from itc2019.nsga3 import Config
from itc2019.repair_policy import RepairSettings

CORE_FILES = ("main.py", "itc2019/instance.py", "itc2019/nsga3.py", "itc2019/scoring.py",
              "itc2019/controller.py", "itc2019/regions.py", "itc2019/operators.py",
              "itc2019/room_search.py", "itc2019/neighborhood.py", "itc2019/repair_policy.py",
              "itc2019/construction.py", "itc2019/sectioning.py",
              "itc2019/diagnostics.py", "itc2019/state_packet.py")
METHOD_ROOTS = {"baseline": "NSGA-III", "single": "Single-Agent Controller",
                "multi": "Multi-Agent Controller"}
PIPELINE_NAMES = {"single": "NSGA-III + Single-Agent",
                  "multi": "NSGA-III + Inspector-Planner"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def behavior(folder: Path, instance: str) -> dict:
    trace = folder / instance / "controller_trace.jsonl"
    if not trace.is_file():
        return {"trace_available": False}
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [event for event in events if event["validation"] in ("valid_action", "valid_no_op")]
    gains = hard_gains = feasible_soft_gains = infeasible_soft_gains = 0
    before_hard = []
    for event in events:
        before = event.get("incumbent_before")
        if before is None:
            old = (event.get("state") or {}).get("global", {})
            before = {"hard": old.get("incumbent_hard"), "internal_total": old.get("incumbent_internal_total")}
        after = event["incumbent_after"]
        if before.get("hard") is None or before.get("internal_total") is None:
            before_hard.append(None)
            continue
        before_hard.append(before["hard"])
        gains += (after["hard"], after["internal_total"]) < (
            before["hard"], before["internal_total"])
        hard_gains += after["hard"] < before["hard"]
        feasible_soft_gains += (before["hard"] == after["hard"] == 0
                               and after["internal_total"] < before["internal_total"])
        infeasible_soft_gains += (before["hard"] == after["hard"] > 0
                                 and after["internal_total"] < before["internal_total"])
    return {"trace_available": True,
        "review_generations": [event["generation"] for event in events],
        "reviews_before_feasibility": sum(h is not None and h > 0 for h in before_hard),
        "reviews_after_feasibility": sum(h == 0 for h in before_hard),
        "reviews_with_unknown_source_quality": sum(h is None for h in before_hard),
        "operator_counts_valid": dict(Counter(event["action"]["operator_id"] for event in valid)),
        "next_review_interval_counts_valid": dict(Counter(str(event["action"]["next_review_interval"]) for event in valid)),
        "execution_budget_counts_valid": dict(Counter(str(event["action"]["execution_budget"]) for event in valid)),
        "trigger_counts": dict(Counter(reason for event in events for reason in event["trigger"])),
        "validation_counts": dict(Counter(event["validation"] for event in events)),
        "outcome_counts": dict(Counter(event["outcome"]["status"] for event in events if event.get("outcome"))),
        "operator_evaluations": sum(event["outcome"]["evaluations"] -
            (event["outcome"].get("scoring_rechecks", 0) if event.get("metadata", {}).get("schema_version", 1) >= 2 else 0)
            for event in events if event.get("outcome")),
        "interventions_improving_incumbent": gains,
        "interventions_reducing_incumbent_hard": hard_gains,
        "interventions_improving_feasible_soft": feasible_soft_gains,
        "interventions_lowering_infeasible_soft_only": infeasible_soft_gains,
        "interventions_improving_primary_progress": hard_gains + feasible_soft_gains}


def matching_runs(profile: dict, instance: str, seed: int, method: str) -> list[Path]:
    local = profile["instances"][instance]
    config_fields = set(Config.__dataclass_fields__)
    expected_config = asdict(Config(**{key: value for key, value in
        {**profile["solver"], **local, "seed": seed}.items() if key in config_fields}))
    overrides = {key.removeprefix("repair_"): value for key, value in local.items()
                 if key.startswith("repair_")}
    expected_repair = asdict(RepairSettings(**{**profile["repair"], **overrides}))
    expected_sources = {key: digest(ROOT / "NSGA-III" / key) for key in CORE_FILES}
    expected_input = [{"file": instance + ".xml",
                       "sha256": digest(ROOT / "Dataset_ITC2019" / (instance + ".xml"))}]
    folder = ROOT / METHOD_ROOTS[method]
    matches = []
    for path in sorted((folder / "results_itc2019").rglob("run_config.json")):
        if not path.parent.name.startswith("run_"):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        settings = manifest["settings"]
        if (settings.get("config") != expected_config or settings.get("repair") != expected_repair
                or settings.get("instances") != expected_input
                or settings.get("solver_sources_sha256") != expected_sources
                or settings.get("time_limit_seconds") != local["time_limit_seconds"]
                or settings.get("stop_on_feasible") is not False):
            continue
        if method == "baseline":
            if "pipeline" in settings:
                continue
        else:
            if settings.get("pipeline", {}).get("name") != PIPELINE_NAMES[method]:
                continue
            controller = settings.get("controller", {})
            if (controller.get("model") != profile["llm"]["model"]
                    or controller.get("max_api_calls") != profile["controller"]["max_reviews"] * (2 if method == "multi" else 1)
                    or any(controller.get(key) != value for key, value in profile["controller"].items())):
                continue
            sources = settings.get("controller_sources_sha256", {})
            if not sources:
                continue
            current = True
            for name, expected in sources.items():
                source = (ROOT / name.replace("\\", "/")).resolve()
                if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or digest(source) != expected:
                    current = False
                    break
            if not current:
                continue
        matches.append(path.parent)
    return matches


def discover(profile: dict, instance: str, seed: int, method: str) -> Path | None:
    matches = matching_runs(profile, instance, seed, method)
    completed = []
    for folder in matches:
        manifest = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (manifest.get("completion_status") == "completed" and
                (method == "baseline" or all(row.get("controller", {}).get("comparison_eligible") for row in rows))):
            completed.append(folder)
    if len(completed) > 1:
        raise ValueError(f"Multiple completed {method} runs for {instance}; do not select by score: {completed}")
    if completed:
        return completed[0]
    # With no completed attempt, compare() rejects the representative failure;
    # audit_attempts() still retains every incomplete attempt.
    return matches[0] if matches else None


def audit_attempts(profile: dict, seed: int, methods=("baseline", "single")) -> list[dict]:
    attempts = []
    for instance in profile["instances"]:
        for method in methods:
            for folder in matching_runs(profile, instance, seed, method):
                manifest = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
                rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                for row in rows:
                    controller = row.get("controller", {})
                    attempts.append({"instance": instance, "method": method, "folder": str(folder),
                        "completion_status": manifest.get("completion_status"),
                        "controller_status": controller.get("completion_status"),
                        "evaluations": row["evaluations"], "runtime_seconds": row["runtime_seconds"],
                        "hard": row["score"]["hard"], "internal_total": row["score"]["total"],
                        "api_calls": controller.get("api_calls", 0),
                        "reported_native_tokens": controller.get("total_native_tokens", 0),
                        "api_failures": controller.get("api_failures", {})})
    return attempts


def summarize(profile: dict, seed: int) -> dict:
    pairs, missing, excluded = [], [], []
    for instance in profile["instances"]:
        folders = {method: discover(profile, instance, seed, method) for method in ("baseline", "single")}
        if any(folder is None for folder in folders.values()):
            missing.append({"instance": instance, "missing_methods":
                            [method for method, folder in folders.items() if folder is None]})
            continue
        try:
            comparison = compare(folders)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            excluded.append({"instance": instance, "reason": str(exc),
                             "runs": {key: str(value) for key, value in folders.items()}})
            continue
        row = comparison["instances"][0]
        base, single = row["methods"]["baseline"], row["methods"]["single"]
        row["runs"] = comparison["runs"]
        row["single_behavior"] = behavior(folders["single"], instance)
        row["controller_evaluations"] = (single["region_scoring_calls"] + single["operator_scoring_rechecks"]
                                          + row["single_behavior"].get("operator_evaluations", 0))
        row["budget_matched"] = bool(base["evaluation_cap_reached"] and single["evaluation_cap_reached"])
        row["soft_change_percent_if_both_feasible"] = (
            100 * (single["internal_total"] - base["internal_total"]) / base["internal_total"]
            if base["feasible_internal"] and single["feasible_internal"] and base["internal_total"] else None)
        if single["hard"] != base["hard"]:
            row["quality_outcome"] = "single_better" if single["hard"] < base["hard"] else "baseline_better"
        elif base["feasible_internal"] and single["feasible_internal"]:
            row["quality_outcome"] = ("tie" if single["internal_total"] == base["internal_total"]
                else "single_better" if single["internal_total"] < base["internal_total"] else "baseline_better")
        else:
            row["quality_outcome"] = "infeasible_hard_tie"
        pairs.append(row)
    matched = [row for row in pairs if row["budget_matched"]]
    attempts = audit_attempts(profile, seed)
    behavior_totals = {}
    for key in ("operator_counts_valid", "next_review_interval_counts_valid", "execution_budget_counts_valid",
                "trigger_counts", "validation_counts", "outcome_counts"):
        counter = Counter()
        for row in pairs:
            counter.update(row["single_behavior"].get(key, {}))
        behavior_totals[key] = dict(counter)
    for key in ("reviews_before_feasibility", "reviews_after_feasibility", "interventions_improving_incumbent",
                "interventions_reducing_incumbent_hard", "interventions_improving_feasible_soft",
                "interventions_lowering_infeasible_soft_only", "interventions_improving_primary_progress"):
        behavior_totals[key] = sum(row["single_behavior"].get(key, 0) for row in pairs)
    return {"experiment": "baseline_vs_single_pilot", "seed": seed,
        "model": profile["llm"]["model"], "score_source": "internal_evaluator",
        "execution": "sequential_local_jobs; one previously completed single run reused",
        "budget_note": "Shared caps, not fixed equal runtime. Quality win counts below exclude evaluation-censored pairs. Do not average raw penalties across different instances.",
        "expected_instances": list(profile["instances"]), "paired_instances": len(pairs),
        "evaluation_matched_pairs": len(matched),
        "quality_counts_matched_only": {value: sum(row["quality_outcome"] == value for row in matched)
            for value in ("single_better", "baseline_better", "tie", "infeasible_hard_tie")},
        "feasible_instances": {method: sum(row["methods"][method]["feasible_internal"] for row in pairs)
                               for method in ("baseline", "single")},
        "single_totals_completed_pairs": {key: sum(row["methods"]["single"][key] for row in pairs)
            for key in ("api_calls", "total_native_tokens", "api_latency_seconds", "interventions", "selected_interventions", "fallback_no_op")},
        "behavior_totals_completed_pairs": behavior_totals,
        "all_attempts": attempts,
        "all_attempt_totals": {"api_calls": sum(row["api_calls"] for row in attempts),
            "reported_native_tokens": sum(row["reported_native_tokens"] for row in attempts),
            "incomplete_runs": sum(row["completion_status"] != "completed" for row in attempts)},
        "retry_policy": "Keep failed attempts; use the sole completed attempt, never choose the best score. API failures can have unreported token usage.",
        "instances": pairs, "missing": missing, "excluded": excluded,
        "limitations": ["One seed per instance; descriptive only, not a significance test.",
                        "Official validation has not been performed for these runs.",
                        "A local intervention gain alone does not establish a gain over baseline."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/pilot.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = summarize(profile, args.seed)
    report["profile_sha256"] = digest(args.profile)
    report["profile_snapshot"] = profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "paired_instances": report["paired_instances"],
                     "missing": report["missing"], "excluded": report["excluded"]}, ensure_ascii=False), flush=True)
    if (report["missing"] or report["excluded"]) and not args.allow_partial:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
