"""Try the production room-plateau repair on one saved timetable without writing output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.instance import load_instance  # noqa: E402
from itc2019.room_search import room_plateau_repair  # noqa: E402
from itc2019.scoring import evaluate  # noqa: E402


def probe(instance_path: Path, choices_path: Path, *, seed: int, seconds: float,
          max_steps: int, max_restarts: int) -> dict:
    instance = load_instance(instance_path)
    with np.load(choices_path) as saved:
        source = np.concatenate([saved["time_indices"], saved["room_indices"]]).astype(np.int32)
    n = len(instance.classes)
    source_score, _ = evaluate(instance, source[:n], source[n:])
    started = time.perf_counter()
    candidate, stats = room_plateau_repair(
        instance, source, np.random.default_rng(seed),
        deadline=started + seconds, source_score=source_score,
        max_steps=max_steps, max_restarts=max_restarts)
    candidate_score = None
    if candidate is not None:
        candidate_score, _ = evaluate(instance, candidate[:n], candidate[n:])
    return {
        "instance": instance.name,
        "seed": seed,
        "source_hard": source_score.hard,
        "source_room_conflicts": source_score.room_conflicts,
        "candidate_hard": None if candidate_score is None else candidate_score.hard,
        "candidate_room_conflicts": None if candidate_score is None else
        candidate_score.room_conflicts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "search_stats": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("choices", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-restarts", type=int, default=100)
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_steps <= 0 or args.max_restarts <= 0:
        parser.error("--seconds, --max-steps and --max-restarts must be positive")
    print(json.dumps(probe(args.instance, args.choices,
                           seed=args.seed, seconds=args.seconds,
                           max_steps=args.max_steps, max_restarts=args.max_restarts), indent=2))
