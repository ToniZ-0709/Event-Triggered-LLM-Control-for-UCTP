"""Measure region coverage and operator budgets on frozen solver populations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.population import Population

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.instance import load_instance  # noqa: E402
from itc2019.nsga3 import PlacementProblem  # noqa: E402
from itc2019.operators import apply_operator  # noqa: E402
from itc2019.regions import RegionBuilder  # noqa: E402


def algorithm(instance, snapshot: dict, seed: int):
    algo = NSGA3(ref_dirs=snapshot["ref_dirs"], pop_size=len(snapshot["X"]))
    algo.setup(PlacementProblem(instance), seed=seed)
    algo.pop = Population.new("X", snapshot["X"].copy(), "F", snapshot["F"].copy(),
                              "G", snapshot["G"].copy())
    algo.n_gen = int(snapshot["generation"])
    return algo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path, help="Tuning trial JSON files with paired NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--geometries", nargs="+", default=["3:3", "6:3", "6:5"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[8, 16, 24])
    args = parser.parse_args()
    rows = []
    geometries = []
    choices = [tuple(map(int, item.split(":"))) for item in args.geometries]
    for path in args.trials:
        trial = json.loads(path.read_text(encoding="utf-8"))
        instance = load_instance(ROOT / "Dataset_ITC2019" / (trial["instance"] + ".xml"))
        with np.load(path.with_suffix(".npz")) as saved:
            snapshot = {key: saved[key] for key in saved.files}
        for max_regions, region_size in choices:
            started = time.perf_counter()
            builder = RegionBuilder(instance)
            regions = builder.build(algorithm(instance, snapshot, 17), max_regions, region_size)
            geometries.append({"trial": path.stem, "max_regions": max_regions,
                               "region_size": region_size, "regions": len(regions),
                               "packet_region_bytes": len(json.dumps([r.public() for r in regions]).encode()),
                               "seconds": round(time.perf_counter() - started, 3),
                               "full_scoring_calls": builder.scoring_calls})
            for region in regions:
                for operator in region.available_operators:
                    for budget in args.budgets:
                        for seed in args.seeds:
                            outcome = apply_operator(
                                algorithm(instance, snapshot, seed), instance, region,
                                operator, budget, np.random.default_rng(seed), None)
                            rows.append({"trial": path.stem, "instance": instance.name,
                                         "max_regions": max_regions, "region_size": region_size,
                                         "region": region.region_id, "operator": operator,
                                         "budget": budget, "seed": seed, **asdict(outcome)})
            print(json.dumps({"trial": path.stem, "max_regions": max_regions,
                              "region_size": region_size, "trials_done": len(rows)}), flush=True)
    summary = []
    for max_regions, region_size in choices:
        for budget in args.budgets:
            selected = [row for row in rows if row["max_regions"] == max_regions and
                        row["region_size"] == region_size and row["budget"] == budget]
            gains = {}
            for row in selected:
                gain = (row["source_hard"] - row["candidate_hard"]
                        if row["candidate_hard"] is not None else 0)
                key = row["trial"], row["seed"]
                gains[key] = max(gains.get(key, 0), gain)
            summary.append({"max_regions": max_regions, "region_size": region_size,
                            "budget": budget, "states_improved": sum(value > 0 for value in gains.values()),
                            "best_hard_gains_sum": sum(gains.values()),
                            "successful_actions": sum(row["selected"] for row in selected),
                            "actions": len(selected),
                            "evaluations": sum(row["evaluations"] for row in selected)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "regions": geometries,
                                      "actions": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
