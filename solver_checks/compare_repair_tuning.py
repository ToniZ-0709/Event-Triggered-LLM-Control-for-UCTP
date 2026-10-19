"""Print same-instance checkpoints for exploratory offspring-repair settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(roots: list[Path], generation: int) -> list[dict]:
    rows = []
    for root in roots:
        for result_path in root.glob("run_*/**/result.json"):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            trace_path = result_path.parent / "anytime_trace.jsonl"
            trace = (json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines())
            checkpoint = next((event for event in trace if event["generation"] == generation), None)
            if checkpoint is None:
                continue
            cfg = result["config"]
            rows.append({
                "instance": result["instance"],
                "seed": cfg["seed"],
                "repair": f"{cfg.get('offspring_repair_steps', 8)}/{cfg.get('offspring_pair_trials', 2)}",
                "generation": generation,
                "hard": checkpoint["best_observed_hard"],
                "elapsed_seconds": round(checkpoint["elapsed_seconds"], 1),
                "evaluations": checkpoint["evaluations"],
                "placement_diversity": checkpoint["placement_diversity"],
                "run_id": result["run_id"],
            })
    return sorted(rows, key=lambda row: (row["instance"], row["seed"], row["repair"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--generation", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(collect(args.roots, args.generation), indent=2))
