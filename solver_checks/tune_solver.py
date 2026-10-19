"""Run bounded engineering tuning with paired instances and full evaluation counts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOT = Path(os.environ.get("ITC_SOLVER_ROOT", str(ROOT / "NSGA-III")))
sys.path.insert(0, str(SOLVER_ROOT))
from itc2019.controller import BaselineMonitor  # noqa: E402
from itc2019.instance import load_instance  # noqa: E402
from itc2019.nsga3 import Config, evaluation_count, run  # noqa: E402

BASE_PARAMETERS = {
    "population": 40, "partitions": 4, "mutation_rate": 0.2,
    "mutation_fraction": 0.005, "crossover_rate": 0.9,
    "initial_repair_steps": 12, "initial_pair_trials": 3,
    "offspring_repair_steps": 8, "offspring_pair_trials": 2,
    "inner_partitions": None, "inner_scale": 0.5,
}

CANDIDATES = {
    "control": {},
    "compact": {"population": 20, "partitions": 3},
    "two_layer": {"population": 32, "partitions": 3, "inner_partitions": 2},
    "dense": {"population": 56, "partitions": 5},
    "mutation": {"mutation_rate": 0.4, "mutation_fraction": 0.01},
    "crossover": {"crossover_rate": 0.7},
    "initial_strong": {"initial_repair_steps": 16, "initial_pair_trials": 4},
    "initial_light": {"initial_repair_steps": 8, "initial_pair_trials": 2},
    "offspring_strong": {"offspring_repair_steps": 12, "offspring_pair_trials": 3},
    "compact_crossover": {"population": 20, "partitions": 3, "crossover_rate": 0.7},
    "compact_lowmutation": {"population": 20, "partitions": 3,
                            "mutation_rate": 0.1, "mutation_fraction": 0.002},
    "compact_repair": {"population": 20, "partitions": 3,
                       "offspring_repair_steps": 12, "offspring_pair_trials": 3},
}


class CalibrationMonitor(BaselineMonitor):
    """Measure a budget knee after 600 full evaluations without meaningful progress."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meaningful_quality = None
        self.last_meaningful_evaluation = 0
        self.stop_reason = "evaluation_cap"
        self.last_log_seconds = 0.0

    def notify(self, algorithm, **kwargs):
        super().notify(algorithm, **kwargs)
        quality = self.best_quality
        count = evaluation_count(algorithm)
        elapsed = time.perf_counter() - self.started
        self.anytime_events[-1]["cpu_seconds"] = round(
            time.process_time() - getattr(self, "cpu_started", time.process_time()), 3)
        if elapsed - self.last_log_seconds >= 60:
            print(json.dumps({"calibration_progress": getattr(self, "label", self.instance.name),
                              "generation": int(algorithm.n_gen), "evaluations": count,
                              "hard": quality[0], "seconds": round(elapsed, 1)}), flush=True)
            self.last_log_seconds = elapsed
        previous = self.meaningful_quality
        if previous is None or quality[0] < previous[0] or (
                quality[0] == 0 and previous[0] == 0 and quality[1] <= previous[1] * 0.99):
            self.meaningful_quality = quality
            self.last_meaningful_evaluation = count
        if count >= 1200 and count - self.last_meaningful_evaluation >= 600:
            self.stop_reason = "600_evaluations_without_hard_or_1pct_soft_progress"
            algorithm.termination.terminate()
            algorithm.termination.update(algorithm)


def source_hash() -> str:
    files = sorted((SOLVER_ROOT / "itc2019").glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": version("numpy"),
            "pymoo": version("pymoo"), "ortools": version("ortools"),
            "logical_cpus": os.cpu_count()}


def trial(job: dict) -> dict:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{job['candidate']}_{job['instance']}_s{job['seed']}_e{job['evaluations']}"
    if job.get("calibrate"):
        stem += "_cal"
    if job.get("time_limit_seconds") is not None:
        stem += f"_t{job['time_limit_seconds']:g}"
    path = output / (stem + ".json")
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["source_hash"] != job["source_hash"]:
            raise RuntimeError(f"Source changed since cached trial: {path}")
        if previous.get("environment", environment()) != environment():
            raise RuntimeError(f"Environment changed since cached trial: {path}")
        return previous
    instance = load_instance(ROOT / "Dataset_ITC2019" / (job["instance"] + ".xml"))
    config = Config(generations=1000, evaluations=job["evaluations"], seed=job["seed"],
                    **{**BASE_PARAMETERS, **CANDIDATES[job["candidate"]]})
    started = time.perf_counter()
    cpu_started = time.process_time()
    monitor_class = CalibrationMonitor if job.get("calibrate") else BaselineMonitor
    monitor = monitor_class(instance, started, job.get("time_limit_seconds"), seed=config.seed)
    monitor.label = stem
    monitor.cpu_started = cpu_started
    choices, score, result = run(instance, config, callback=monitor,
                                 time_limit_seconds=job.get("time_limit_seconds"))
    seconds = time.perf_counter() - started
    algorithm = result.algorithm
    record = {
        "candidate": job["candidate"], "instance": instance.name,
        "seed": config.seed, "target_evaluations": job["evaluations"],
        "evaluations": evaluation_count(algorithm), "seconds": round(seconds, 3),
        "cpu_seconds": round(time.process_time() - cpu_started, 3),
        "hard": score.hard, "total": score.total, "room_conflicts": score.room_conflicts,
        "room_unavailable": score.room_unavailable, "hard_distributions": score.hard_distributions,
        "config": asdict(config), "source_hash": job["source_hash"],
        "environment": environment(),
        "concurrent_workers": job["workers"], "trace": monitor.anytime_events,
        "monitor": monitor.report(),
        "stage_seconds": algorithm.problem.stage_seconds,
        "repair_scoring_calls": algorithm.problem.repair_scoring_calls,
        "calibration": bool(job.get("calibrate")),
        "stop_reason": getattr(monitor, "stop_reason", "evaluation_cap"),
        "time_limit_seconds": job.get("time_limit_seconds"),
    }
    if monitor.remaining() is not None and monitor.remaining() <= 0:
        record["stop_reason"] = "time_cap"
    np.savez_compressed(output / (stem + ".npz"),
                        X=algorithm.pop.get("X"), F=algorithm.pop.get("F"), G=algorithm.pop.get("G"),
                        choices=choices, ref_dirs=algorithm.ref_dirs,
                        generation=monitor.anytime_events[-1]["generation"])
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def rank(output: Path) -> dict:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in output.glob("*.json")
               if p.name != "ranking.json"]
    groups = {}
    for row in records:
        groups.setdefault((row["instance"], row["seed"], row["target_evaluations"]), []).append(row)
    ranks = {}
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: (row["hard"], row["total"] if row["hard"] == 0 else 0))
        for row in ordered:
            key = (row["hard"], row["total"] if row["hard"] == 0 else 0)
            position = sum((other["hard"], other["total"] if other["hard"] == 0 else 0) < key
                           for other in ordered)
            ranks.setdefault(row["candidate"], []).append(position)
    summary = {candidate: {"mean_rank": round(float(np.mean(values)), 3),
                           "worst_rank": max(values), "groups": len(values)}
               for candidate, values in ranks.items()}
    summary = dict(sorted(summary.items(), key=lambda item: (
        item[1]["mean_rank"], item[1]["worst_rank"], item[0])))
    (output / "ranking.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--instances", nargs="+", default=["muni-fsps-spr17", "nbi-spr18"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--cases", nargs="+", help="Explicit instance:seed pairs instead of a Cartesian grid")
    parser.add_argument("--evaluations", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fingerprint = source_hash()
    cases = ([(item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1])) for item in args.cases]
             if args.cases else [(instance, seed) for instance in args.instances for seed in args.seeds])
    jobs = [{"candidate": candidate, "instance": instance, "seed": seed,
             "evaluations": args.evaluations, "output": str(args.output.resolve()),
             "source_hash": fingerprint, "workers": args.workers}
            for candidate in args.candidates for instance, seed in cases]
    for job in jobs:
        job["calibrate"] = args.calibrate
        job["time_limit_seconds"] = args.time_limit_seconds
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(trial, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({key: row[key] for key in
                              ("candidate", "instance", "seed", "hard", "evaluations", "seconds")}),
                  flush=True)
    print(json.dumps(rank(args.output), indent=2), flush=True)
