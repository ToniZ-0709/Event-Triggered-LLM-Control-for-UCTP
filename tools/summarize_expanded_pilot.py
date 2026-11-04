"""Audit a three-pipeline, multi-seed campaign without selecting runs by score."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_pipelines import compare
from tools.summarize_paired_pilot import audit_attempts, behavior, digest, discover

METHODS = ("baseline", "single", "multi")
DEFAULT_STATE_ROOT = ROOT / "tmp/phase1_pilot/state"
DEFAULT_REPORT = ROOT / "tmp/phase1_pilot/expanded_pilot_comparison.json"


def outcome(left: dict, right: dict) -> str:
    if left["hard"] != right["hard"]:
        return "right_better" if right["hard"] < left["hard"] else "left_better"
    if left["hard"]:
        return "infeasible_hard_tie"
    return ("tie" if left["internal_total"] == right["internal_total"] else
            "right_better" if right["internal_total"] < left["internal_total"] else "left_better")


def completed(folder: Path | None, method: str) -> bool:
    if folder is None:
        return False
    manifest = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return (manifest.get("completion_status") == "completed" and
            (method == "baseline" or all(row.get("controller", {}).get("comparison_eligible") for row in rows)))


def summarize(profile: dict, seeds: list[int], *, allow_no_time_limit: bool = False) -> dict:
    rows, missing, excluded, attempts = [], [], [], []
    for seed in seeds:
        attempts.extend({**record, "seed": seed} for record in audit_attempts(profile, seed, METHODS))
        for instance in profile["instances"]:
            folders = {method: discover(profile, instance, seed, method) for method in METHODS}
            unavailable = [method for method, folder in folders.items() if not completed(folder, method)]
            if unavailable:
                missing.append({"instance": instance, "seed": seed, "incomplete_or_missing": unavailable})
                continue
            try:
                result = (compare(folders, allow_generation_only=True) if allow_no_time_limit
                          else compare(folders))
                row = result["instances"][0]
                row.update(seed=seed, runs=result["runs"])
                row["controller_behavior"] = {method: behavior(folders[method], instance) for method in METHODS[1:]}
                row["evaluation_matched_triple"] = all(
                    record["evaluation_cap_reached"] for record in row["methods"].values())
                row["pairwise"] = {}
                for left, right in combinations(METHODS, 2):
                    a, b = row["methods"][left], row["methods"][right]
                    row["pairwise"][f"{right}_vs_{left}"] = {
                        "left": left, "right": right, "quality_outcome": outcome(a, b),
                        "evaluation_matched": bool(a["evaluation_cap_reached"] and b["evaluation_cap_reached"]),
                        "hard_change": b["hard"] - a["hard"],
                        "soft_change_percent_if_both_feasible": (
                            100 * (b["internal_total"] - a["internal_total"]) / a["internal_total"]
                            if a["hard"] == b["hard"] == 0 and a["internal_total"] else None)}
                rows.append(row)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                excluded.append({"instance": instance, "seed": seed, "reason": str(exc)})
    counts = {}
    for left, right in combinations(METHODS, 2):
        key = f"{right}_vs_{left}"
        counts[key] = dict(Counter(row["pairwise"][key]["quality_outcome"]
                                   for row in rows if row["pairwise"][key]["evaluation_matched"]))
    method_totals = {}
    for method in METHODS:
        method_totals[method] = {
            "feasible_internal": sum(row["methods"][method]["feasible_internal"] for row in rows),
            **{key: sum(row["methods"][method][key] for row in rows) for key in (
                "runtime_seconds", "evaluations", "reviews", "api_calls", "api_latency_seconds",
                "input_tokens", "output_tokens", "thinking_tokens", "total_native_tokens",
                "interventions", "selected_interventions", "fallback_no_op")}}
    seed_summaries = {}
    for seed in seeds:
        selected = [row for row in rows if row["seed"] == seed]
        seed_summaries[str(seed)] = {
            "completed_triples": len(selected),
            "feasible_internal": {method: sum(row["methods"][method]["feasible_internal"] for row in selected)
                                  for method in METHODS},
            "quality_counts_evaluation_matched_only": {
                key: dict(Counter(row["pairwise"][key]["quality_outcome"] for row in selected
                                  if row["pairwise"][key]["evaluation_matched"])) for key in counts}}
    controller_totals = {}
    for method in METHODS[1:]:
        totals = {}
        for key in ("operator_counts_valid", "next_review_interval_counts_valid", "execution_budget_counts_valid",
                    "trigger_counts", "validation_counts", "outcome_counts"):
            counter = Counter()
            for row in rows:
                counter.update(row["controller_behavior"][method].get(key, {}))
            totals[key] = dict(counter)
        for key in ("reviews_before_feasibility", "reviews_after_feasibility", "interventions_improving_incumbent",
                    "interventions_reducing_incumbent_hard", "interventions_improving_feasible_soft",
                    "interventions_lowering_infeasible_soft_only", "interventions_improving_primary_progress"):
            totals[key] = sum(row["controller_behavior"][method].get(key, 0) for row in rows)
        controller_totals[method] = totals
    return {
        "experiment": ("three_pipeline_no_timecap_followup" if allow_no_time_limit
                       else "nine_instance_three_pipeline_two_seed_check"), "seeds": seeds,
        "model": profile["llm"]["model"], "expected_instances": list(profile["instances"]),
        "expected_cells": len(seeds) * len(profile["instances"]) * len(METHODS),
        "completed_cells": sum(record["completion_status"] == "completed" for record in attempts),
        "completed_triples": len(rows),
        "evaluation_matched_triples": sum(row["evaluation_matched_triple"] for row in rows),
        "quality_counts_evaluation_matched_only": counts, "method_totals_completed_triples": method_totals,
        "per_seed": seed_summaries, "controller_behavior_totals_completed_triples": controller_totals,
        "instances": rows, "missing": missing, "excluded": excluded, "all_attempts": attempts,
        "all_attempt_totals": {
            "api_calls": sum(record["api_calls"] for record in attempts),
            "reported_native_tokens": sum(record["reported_native_tokens"] for record in attempts),
            "incomplete_runs": sum(record["completion_status"] != "completed" for record in attempts)},
        "interpretation": [
            "Sequential local jobs at shared caps; not an equal-runtime comparison.",
            "Original seed-17 baseline/single cells reused only with matching inputs, sources and settings.",
            "Quality counts use archived representative endpoints, not Pareto-front dominance or hypervolume.",
            "Do not rank soft scores at tied positive hard violations or average raw penalties across instances.",
            "Two GA seeds and non-deterministic API responses: descriptive, not a significance claim.",
            "Internal evaluator only; no official validator submission has been performed.",
            "Retain failed attempts; choose the unique completed attempt, never the best score.",
            "Reported tokens may omit usage from failed API requests."]}


def save_report(profile_path: Path, seeds: list[int], output: Path, *,
                state_root: Path | None = None, allow_no_time_limit: bool = False) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = (summarize(profile, seeds, allow_no_time_limit=True) if allow_no_time_limit
              else summarize(profile, seeds))
    report.update(profile_sha256=digest(profile_path), profile_snapshot=profile)
    state_root = state_root if state_root is not None else DEFAULT_STATE_ROOT
    failures_path = state_root / "storage_failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8"))["failures"] if failures_path.is_file() else []
    failures = [record for record in failures
                if record["seed"] in seeds and record["instance"] in profile["instances"]]
    report["unpublished_attempts"] = failures
    report["all_attempt_totals"]["unpublished_runs"] = len(failures)
    report["all_attempt_totals"]["usage_has_unknown_unpublished_component"] = bool(failures)
    recoveries_path = state_root / "human_recoveries.json"
    recoveries = (json.loads(recoveries_path.read_text(encoding="utf-8"))["recoveries"]
                  if recoveries_path.is_file() else [])
    report["human_authorized_recoveries"] = [record for record in recoveries
        if record["seed"] in seeds and record["instance"] in profile["instances"] and
        record.get("profile_sha256") == report["profile_sha256"]]
    if report["human_authorized_recoveries"]:
        report["interpretation"].append(
            "Each separately recorded human authorization permits one specific additional attempt after transient failures; the default automatic retry limit is unchanged. Retain all failed attempts and their unknown request usage.")
    if allow_no_time_limit:
        report["interpretation"].append(
            "Separate follow-up without a solver wall-clock cap; generation/evaluation, repair and API budgets remain bounded. Do not overwrite or pool the original time-capped campaign.")
    if failures:
        report["interpretation"].append(
            "Unpublished failures are retained separately, never counted as completed cells or quality losses. Their API calls/tokens may be unknown and are not included in the reported totals.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/expanded_pilot.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42])
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--allow-no-time-limit", action="store_true")
    args = parser.parse_args()
    if args.allow_no_time_limit and args.output.resolve() == DEFAULT_REPORT.resolve():
        parser.error("Use a separate output for a no-time-cap follow-up")
    report = save_report(args.profile, args.seeds, args.output, state_root=args.state_root,
                         allow_no_time_limit=args.allow_no_time_limit)
    print(json.dumps({key: report[key] for key in ("completed_cells", "completed_triples", "missing", "excluded")}))
