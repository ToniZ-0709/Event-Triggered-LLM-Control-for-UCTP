"""Bounded correctness probes for shared construction, repairs and accounting.

Print JSON only. This is a diagnostic, not a cross-solver quality benchmark.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.instance import load_instance
from itc2019.nsga3 import Config, run, evaluation_count
from itc2019.controller import BaselineMonitor
from itc2019.construction import ConstructionIndex
from itc2019.neighborhood import distribution_neighborhood_repair, adaptive_room_repair
from itc2019.scoring import evaluate, validate_sectioning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance")
    parser.add_argument("--mode", choices=("construction","run","repair"), default="construction")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--choices", type=Path)
    parser.add_argument("--output", type=Path, help="Save diagnostic JSON and final choices outside normal run outputs")
    args = parser.parse_args()
    instance = load_instance(ROOT / "Dataset_ITC2019" / (args.instance+".xml"))
    start = time.perf_counter()
    if args.mode == "construction":
        index = ConstructionIndex(instance)
        choices = index.build(np.random.default_rng(17), deadline=start+args.seconds)
        construction_seconds = time.perf_counter()-start
        score, assignments = evaluate(instance, choices[:len(instance.classes)],
                                      choices[len(instance.classes):],keep_assignments=True)
        result = {"construction_seconds":construction_seconds}
    elif args.mode == "repair":
        with np.load(args.choices) as saved:
            choices = saved["choices"].copy() if "choices" in saved else np.concatenate(
                [saved["time_indices"],saved["room_indices"]])
        n = len(instance.classes)
        before = evaluate(instance,choices[:n],choices[n:])[0]
        repair = distribution_neighborhood_repair if before.hard_distributions else adaptive_room_repair
        candidate, stats = repair(instance,choices,deadline=time.perf_counter()+args.seconds,
                                  source_score=before,max_full_scoring_calls=5)
        if candidate is not None:
            choices = candidate
        score, assignments = evaluate(instance,choices[:n],choices[n:],keep_assignments=True)
        result = {"source_hard":before.hard,"stats":stats}
    else:
        monitor = BaselineMonitor(instance,start,args.seconds,17)
        choices, score, result = run(instance,Config(population=4,partitions=1,generations=6,
                                                    seed=17,evaluations=120),callback=monitor,
                                     time_limit_seconds=args.seconds)
        engine = result.algorithm.problem.engine
        assert evaluation_count(result.algorithm) == engine.requests, (
            evaluation_count(result.algorithm),engine.report())
        checked, assignments = evaluate(instance,choices[:len(instance.classes)],
                                         choices[len(instance.classes):],keep_assignments=True)
        assert checked == score
        result = {"generations":len(monitor.anytime_events),"scoring":engine.report(),
                  "evaluations":evaluation_count(result.algorithm)}
    if score.unassigned_requests == 0:
        validate_sectioning(instance, assignments)
    result.update(instance=instance.name,hard=score.hard,total=score.total,
                  elapsed_seconds=time.perf_counter()-start,
                  unassigned=score.unassigned_requests)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"),choices=choices)
        args.output.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
