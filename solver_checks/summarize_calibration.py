"""Summarize completed per-instance calibration runs and their checkpoint times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def summarize(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.glob("run_*/**/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace_path = result_path.parent / "anytime_trace.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        by_generation = {event["generation"]: event for event in trace}
        elapsed = {str(g): round(by_generation[g]["elapsed_seconds"], 1)
                   for g in (1, 5, 10) if g in by_generation}
        checkpoint_hard = {str(g): by_generation[g]["best_observed_hard"]
                           for g in (1, 5, 10) if g in by_generation}
        intervals = [trace[j]["elapsed_seconds"] - trace[j - 1]["elapsed_seconds"]
                     for j in range(1, len(trace))]
        rows.append({
            "instance": result["instance"],
            "seed": result["config"]["seed"],
            "generations": len(trace),
            "runtime_seconds": round(result["runtime_seconds"], 1),
            "checkpoint_seconds": elapsed,
            "checkpoint_hard": checkpoint_hard,
            "median_later_generation_seconds": round(median(intervals), 1) if intervals else None,
            "evaluations": result["evaluations"],
            "offspring_repair_steps": result["config"].get("offspring_repair_steps", 8),
            "offspring_pair_trials": result["config"].get("offspring_pair_trials", 2),
            "hard": result["score"]["hard"],
            "room_conflicts": result["score"]["room_conflicts"],
            "hard_distributions": result["score"]["hard_distributions"],
            "plateau_events": result["monitor"].get("room_plateau_events", []),
            "stage_seconds": result.get("stage_timing", {}).get("seconds"),
            "run_id": result["run_id"],
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
