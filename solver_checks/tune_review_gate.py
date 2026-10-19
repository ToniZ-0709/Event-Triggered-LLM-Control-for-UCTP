"""Replay review frequencies on measured trajectories without making API calls."""

import argparse
from itertools import product
import json
from pathlib import Path
import tempfile

from replay_review_gate import replay

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    details = []
    with tempfile.TemporaryDirectory(prefix="review-replay-") as temporary:
        paths = []
        for trial in args.trials:
            record = json.loads(trial.read_text(encoding="utf-8"))
            trace = Path(temporary) / (trial.stem + ".jsonl")
            trace.write_text("".join(json.dumps(event) + "\n" for event in record["trace"]), encoding="utf-8")
            paths.append((record, trace, trial.stem))
        for interval, window, threshold in product((5, 10, 15), (3, 5, 8), (0.1, 0.2, 0.3)):
            events = []
            for record, trace, label in paths:
                triggered = replay(
                    trace, ROOT / "Dataset_ITC2019" / (record["instance"] + ".xml"),
                    interval=interval, stagnation_window=window, diversity_threshold=threshold,
                    min_review_interval=3, max_reviews=12)
                evals = {row["generation"]: row["evaluations"] for row in record["trace"]}
                triggered = [row for row in triggered
                             if evals[row["generation"]] + 4 < record["target_evaluations"]]
                events.extend(triggered)
                details.append({"trial": label, "interval": interval, "window": window,
                                "threshold": threshold, "events": triggered})
            rows.append({"interval": interval, "window": window, "threshold": threshold,
                         "reviews": len(events),
                         "periodic_only": sum(row["reasons"] == ["deadline"] for row in events),
                         "low_diversity_events": sum("low_diversity" in row["reasons"] for row in events),
                         "stagnation_events": sum("stagnation" in row["reasons"] for row in events)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": rows, "details": details}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([row for row in rows if row["window"] == 5 and row["threshold"] == 0.2], indent=2))
