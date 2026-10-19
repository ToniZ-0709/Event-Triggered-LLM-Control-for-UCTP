"""Measure blocker-expanding repair on a saved ITC 2019 timetable."""

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
from itc2019.neighborhood import PlacementNeighborhood, room_neighborhood_repair  # noqa: E402
from itc2019.scoring import distribution_violations, evaluate  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("choices", type=Path)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--classes", type=int, default=24)
    parser.add_argument("--options", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--deterministic-limit", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--integrated", action="store_true", help="Execute the shared monitor repair")
    parser.add_argument("--distribution", action="store_true", help="Direct required-group probe with the explicit CP budget")
    args = parser.parse_args()
    if args.integrated and args.distribution:
        parser.error("--distribution is a direct post-run probe, not the monitor policy")
    instance = load_instance(args.instance)
    with np.load(args.choices) as saved:
        source = (saved["choices"].copy() if "choices" in saved else
                  np.concatenate([saved["time_indices"], saved["room_indices"]]))
    n = len(instance.classes)
    before, _ = evaluate(instance, source[:n], source[n:])
    started = time.perf_counter()
    if args.integrated:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.population import Population
        from pymoo.util.ref_dirs import get_reference_directions
        from itc2019.controller import BaselineMonitor
        from itc2019.nsga3 import PlacementProblem
        from itc2019.repair_policy import RepairSettings
        algo = NSGA3(ref_dirs=get_reference_directions("das-dennis", 4, n_partitions=4), pop_size=40)
        algo.setup(PlacementProblem(instance), seed=args.seed)
        algo.pop = Population.new("X", source.reshape(1, -1),
                                  "F", np.asarray([before.weighted]), "G", np.asarray([[before.hard]]))
        algo.n_gen = 5
        monitor = BaselineMonitor(instance, started, None, args.seed, RepairSettings(
            neighborhood_classes=args.classes, neighborhood_options=args.options,
            neighborhood_seconds=args.seconds,
            neighborhood_deterministic_limit=args.deterministic_limit))
        monitor._maybe_room_chain_repair(algo)
        candidate = monitor.best_choices
        stats = monitor.room_chain_events[-1] if monitor.room_chain_events else {"status": "skipped"}
    else:
        seeds = None
        if args.distribution:
            state = PlacementNeighborhood(instance, source)
            targets = [(-amount,len(d.classes),number,d) for number,d in enumerate(instance.distributions)
                       if d.required and (amount := distribution_violations(instance,d,state.times,state.rooms))]
            if not targets:
                parser.error("Saved solution has no required-distribution violations")
            seeds = min(targets,key=lambda item:item[:3])[3].classes
        candidate, stats = room_neighborhood_repair(
            instance, source, deadline=started + args.seconds, seed=args.seed,
            max_classes=args.classes, max_options=args.options, source_score=before,
            deterministic_limit=args.deterministic_limit, seed_classes=seeds,
            require_room_improvement=not args.distribution)
    result = {"instance": instance.name, "seed": args.seed,
              "target": "distribution" if args.distribution else "room",
              "source_hard": before.hard, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "stats": stats}
    if candidate is not None:
        score, _ = evaluate(instance, candidate[:n], candidate[n:])
        result["candidate_hard"] = score.hard
        result["changes"] = [instance.classes[i].id for i in range(n)
                             if candidate[i] != source[i] or candidate[n + i] != source[n + i]]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, time_indices=candidate[:n], room_indices=candidate[n:])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
