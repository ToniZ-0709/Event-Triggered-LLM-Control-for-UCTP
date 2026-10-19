"""Replay the current review gate against saved baseline anytime traces.

This is a scheduling probe: it does not call an LLM or apply an operator.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.controller import ControllerCallback, ControllerSettings, _progress_signal  # noqa: E402
from itc2019.instance import load_instance  # noqa: E402


@lru_cache(maxsize=64)
def _cached_instance(path: Path):
    return load_instance(path)


def replay(trace_path: Path, instance_path: Path, *, interval: int,
           stagnation_window: int, diversity_threshold: float,
           min_review_interval: int, max_reviews: int,
           diversity_signal: str = "placement_diversity") -> list[dict]:
    settings = ControllerSettings(
        model="offline-gate-replay", review_interval=interval,
        warmup_generations=5, stagnation_window=stagnation_window,
        min_review_interval=min_review_interval,
        max_review_interval=max(10, interval),
        diversity_threshold=diversity_threshold,
        diversity_recovery_threshold=max(0.30, diversity_threshold), max_reviews=max_reviews)
    callback = ControllerCallback(_cached_instance(instance_path), settings, None, 17,
                                  time.perf_counter(), None)
    previous_signal = None
    reviews = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        metrics = json.loads(line)
        metrics["placement_diversity"] = metrics[diversity_signal]
        generation = metrics["generation"]
        signal = _progress_signal((metrics["best_observed_hard"],
                                   metrics["best_observed_internal_total"]))
        if signal != previous_signal:
            callback.last_progress_generation = generation
        previous_signal = signal
        metrics["stagnant_generations"] = generation - callback.last_progress_generation
        reasons = callback._trigger(metrics)
        if reasons and len(reviews) < max_reviews:
            callback._stagnation_event_handled |= "stagnation" in reasons
            callback._low_diversity_event_handled |= "low_diversity" in reasons
            callback.last_review = generation
            callback.next_review = generation + interval
            reviews.append({"generation": generation, "reasons": reasons,
                            "hard": metrics["best_observed_hard"],
                            "diversity_signal": diversity_signal,
                            "diversity": metrics["placement_diversity"]})
    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--stagnation-window", type=int, default=5)
    parser.add_argument("--diversity-threshold", type=float, default=0.20)
    parser.add_argument("--diversity-signal", choices=("decision_diversity", "placement_diversity"),
                        default="placement_diversity")
    parser.add_argument("--min-review-interval", type=int, default=3)
    parser.add_argument("--max-reviews", type=int, default=12)
    args = parser.parse_args()
    events = replay(args.trace, args.instance, interval=args.interval,
                    stagnation_window=args.stagnation_window,
                    diversity_threshold=args.diversity_threshold,
                    min_review_interval=args.min_review_interval,
                    max_reviews=args.max_reviews,
                    diversity_signal=args.diversity_signal)
    print(json.dumps({"trace": str(args.trace), "reviews": events}, indent=2))
