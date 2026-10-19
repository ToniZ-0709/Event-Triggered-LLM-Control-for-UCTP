"""Derive conservative local caps from measured calibration trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import median

if __package__:
    from .tune_solver import BASE_PARAMETERS, CANDIDATES
else:
    from tune_solver import BASE_PARAMETERS, CANDIDATES
from itc2019.repair_policy import RepairSettings


def horizon(record: dict) -> int:
    trace = record["trace"]
    feasible = [row for row in trace if row["best_observed_hard"] == 0]
    if feasible:
        initial = feasible[0]["best_observed_internal_total"]
        best = feasible[-1]["best_observed_internal_total"]
        threshold = best + 0.1 * (initial - best)
        knee = next(row["evaluations"] for row in feasible
                    if row["best_observed_internal_total"] <= threshold)
        desired = max(1200, feasible[0]["evaluations"] + 600, knee + 200)
    else:
        previous = float("inf")
        last_improvement = 0
        for row in trace:
            if row["best_observed_hard"] < previous:
                previous = row["best_observed_hard"]
                last_improvement = row["evaluations"]
        desired = max(1200, last_improvement + 600)
    measured_maximum = max(record["config"]["population"],
                           (trace[-1]["evaluations"] // 100) * 100)
    return min(math.ceil(desired / 100) * 100, measured_maximum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--controller-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding="utf-8")) for p in args.folder.glob("*.json")
               if p.name != "ranking.json"]
    records = [row for row in records if row.get("calibration") and row["candidate"] == args.candidate]
    if not records:
        parser.error("No calibration records found")
    if len({row["source_hash"] for row in records}) != 1:
        parser.error("Do not mix solver source versions in calibration")
    if len({json.dumps(row.get("environment"), sort_keys=True) for row in records}) != 1:
        parser.error("Do not mix environments in calibration")
    controller = json.loads(args.controller_settings.read_text(encoding="utf-8"))
    groups = {}
    for row in records:
        groups.setdefault(row["instance"], []).append(row)
    budgets = {}
    evidence = {}
    api_reserve = controller["max_reviews"] * 2 * controller["api_timeout_seconds"]
    for name, rows in groups.items():
        selected = max(horizon(row) for row in rows)
        observed_times = []
        generation_times = []
        cases = []
        for row in rows:
            trace = row["trace"]
            target = next((event for event in trace if event["evaluations"] >= selected), trace[-1])
            observed_times.append(target["elapsed_seconds"])
            generation_times.extend(trace[j]["elapsed_seconds"] - trace[j - 1]["elapsed_seconds"]
                                    for j in range(1, len(trace)))
            cases.append({"seed": row["seed"], "measured_evaluations": row["evaluations"],
                          "measured_generations": trace[-1]["generation"],
                          "measured_seconds": row["seconds"],
                          "hard": row["hard"], "stop_reason": row["stop_reason"],
                          "hard_components": {key: row[key] for key in
                                              ("room_conflicts", "room_unavailable", "hard_distributions")},
                          "internal_total": row["total"] if row["hard"] == 0 else None,
                          "concurrent_workers": row["concurrent_workers"],
                          "measured_cpu_seconds": row.get("cpu_seconds"),
                          "reached_selected_budget": target["evaluations"] >= selected,
                          "seconds_to_selected_budget": target["elapsed_seconds"]})
        generation_reserve = max(generation_times) if generation_times else observed_times[0]
        seconds = math.ceil((1.35 * max(observed_times) + 2 * generation_reserve + api_reserve) / 30) * 30
        budgets[name] = {"generations": 200, "evaluations": selected,
                         "time_limit_seconds": seconds}
        longest = max(rows, key=lambda row: row["evaluations"])
        budget_status = ("plateau_observed" if longest["stop_reason"].startswith("600_") else
                         "time_censored" if longest["stop_reason"] == "time_cap" else
                         "evaluation_censored")
        evidence[name] = {"cases": cases, "api_timeout_reserve_seconds": api_reserve,
                          "budget_status": budget_status,
                          "time_censored_cases": sum(row["stop_reason"] == "time_cap" for row in rows),
                          "median_generation_seconds": round(median(generation_times), 3)
                          if generation_times else None}
    profile = {"schema_version": 1, "name": "local-pilot-2026-10-17",
               "solver": {**BASE_PARAMETERS, **CANDIDATES[args.candidate]},
               "repair": asdict(RepairSettings()), "controller": controller,
               "instances": budgets,
               "calibration": {"candidate": args.candidate, "source_hash": records[0]["source_hash"],
                               "environment": records[0].get("environment"),
                               "timing_note": "Concurrent local engineering jobs; wall times select conservative caps, not a speed comparison",
                               "rule": "Hard progress, then 90% of measured soft gain; 600 evaluation grace after feasibility",
                               "budget_evidence": evidence}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(budgets, indent=2))
