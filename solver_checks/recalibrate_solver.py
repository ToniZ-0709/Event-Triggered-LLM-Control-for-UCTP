"""Paired engineering calibration of the frozen shared solver; no LLM calls.

Results are resumable JSON/NPZ artifacts, not a separate Markdown report.
Wall times under concurrent jobs are conservative local sizing measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.controller import BaselineMonitor
from itc2019.instance import load_instance
from itc2019.nsga3 import Config, evaluation_count, run
from itc2019.repair_policy import RepairSettings
from itc2019.scoring import distribution_violations, evaluate, overlaps, validate_sectioning
from tune_solver import source_hash, environment

VARIANTS = {
    "selected": ({}, {}),
    "current": ({}, {}),
    "small": ({"population": 12, "partitions": 2}, {}),
    "small_mutation": ({"population": 12, "partitions": 2,
                        "mutation_rate": 0.35, "mutation_fraction": 0.01}, {}),
    "small_cp": ({"population": 12, "partitions": 2},
                 {"neighborhood_seconds": 15.0, "neighborhood_deterministic_limit": 2.0}),
    "early_cp": ({"mutation_rate": 0.35, "mutation_fraction": 0.01},
                 {"near_feasible_hard": 40, "neighborhood_seconds": 15.0,
                  "neighborhood_deterministic_limit": 2.0}),
    "two_layer": ({"population": 32, "partitions": 3, "inner_partitions": 2}, {}),
    "mutation": ({"mutation_rate": 0.35, "mutation_fraction": 0.01}, {}),
    "strong_repair": ({"initial_repair_steps": 16, "initial_pair_trials": 4,
                       "offspring_repair_steps": 12, "offspring_pair_trials": 3}, {}),
    "wide_repair": ({}, {"neighborhood_classes": 48, "neighborhood_options": 48,
                         "neighborhood_deterministic_limit": 0.75}),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeasurementMonitor(BaselineMonitor):
    def __init__(self, *args, label, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.last_log = 0.0
        self.cpu_started = time.process_time()

    def notify(self, algorithm, **kwargs):
        super().notify(algorithm, **kwargs)
        event = self.anytime_events[-1]
        event["cpu_seconds"] = round(time.process_time()-self.cpu_started, 3)
        if event["elapsed_seconds"]-self.last_log >= 30:
            print(json.dumps({"progress":self.label,"generation":int(algorithm.n_gen),
                              "evaluations":evaluation_count(algorithm),
                              "hard":self.best_quality[0],
                              "seconds":round(event["elapsed_seconds"],1)}),flush=True)
            self.last_log = event["elapsed_seconds"]


def diagnose(instance, choices):
    n = len(instance.classes)
    times = [c.times[int(choices[i])] for i,c in enumerate(instance.classes)]
    rooms = [None if c.no_room else c.room_options[int(choices[n+i])][0]
             for i,c in enumerate(instance.classes)]
    groups = []
    for i,d in enumerate(instance.distributions):
        if d.required and (amount := distribution_violations(instance,d,times,rooms)):
            groups.append({"index":i,"kind":d.kind,"params":d.params,"violations":amount,
                           "class_ids":[instance.classes[j].id for j in d.classes]})
    collisions = []
    unavailable = []
    by_room = {}
    for i,room in enumerate(rooms):
        if room is None:
            continue
        if any(overlaps(times[i],u) for u in instance.rooms[room].unavailable):
            unavailable.append(instance.classes[i].id)
        for j in by_room.get(room,()):
            if overlaps(times[i],times[j]):
                collisions.append((room,instance.classes[j].id,instance.classes[i].id))
        by_room.setdefault(room,[]).append(i)
    return {"required_groups":groups,"room_collisions":collisions,
            "unavailable_class_ids":unavailable}


def job_identity(job):
    identity = {k:job[k] for k in ("variant","instance","seed","evaluations","generations",
                                  "seconds","solver","repair","source_hash","dataset_hash",
                                  "environment","workers")}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]


def measure(job):
    folder = Path(job["output"])
    folder.mkdir(parents=True,exist_ok=True)
    stem = f"{job['variant']}_{job['instance']}_s{job['seed']}_e{job['evaluations']}_{job_identity(job)}"
    path = folder/(stem+".json")
    if source_hash() != job["source_hash"] or environment() != job["environment"]:
        raise RuntimeError("Solver source/environment changed since measurement manifest")
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["identity"] != job_identity(job):
            raise RuntimeError(f"Cached measurement identity mismatch: {path}")
        return previous
    dataset = ROOT/"Dataset_ITC2019"/(job["instance"]+".xml")
    if digest(dataset) != job["dataset_hash"]:
        raise RuntimeError("Dataset changed during measurement")
    instance = load_instance(dataset)
    config = Config(**job["solver"],generations=job["generations"],seed=job["seed"],
                    evaluations=job["evaluations"])
    started = time.perf_counter()
    cpu_started = time.process_time()
    monitor = MeasurementMonitor(instance,started,job["seconds"],seed=job["seed"],
                                 repair_settings=RepairSettings(**job["repair"]),label=stem)
    choices,score,result = run(instance,config,callback=monitor,time_limit_seconds=job["seconds"])
    seconds = time.perf_counter()-started
    cpu_seconds = time.process_time()-cpu_started
    count = evaluation_count(result.algorithm)
    engine = result.algorithm.problem.engine
    if count != engine.requests or count > job["evaluations"]:
        raise RuntimeError(f"Invalid evaluation accounting: {count}, {engine.report()}")
    checked,assignments = evaluate(instance,choices[:len(instance.classes)],
                                    choices[len(instance.classes):],keep_assignments=True)
    if checked != score:
        raise RuntimeError("Final score differs from recorded incumbent")
    if score.unassigned_requests == 0:
        validate_sectioning(instance,assignments)
    trace = monitor.anytime_events
    stop = ("evaluation_cap" if count >= job["evaluations"] else
            "time_cap" if seconds >= job["seconds"] else "generation_cap")
    record = {"identity":job_identity(job),"variant":job["variant"],"instance":instance.name,
              "seed":job["seed"],"config":asdict(config),"repair":job["repair"],
              "target_evaluations":job["evaluations"],"evaluations":count,"seconds":round(seconds,3),
              "cpu_seconds":round(cpu_seconds,3),"time_limit_seconds":job["seconds"],
              "stop_reason":stop,"score":asdict(score),"trace":trace,"monitor":monitor.report(),
              "score_engine":engine.report(),"stage_seconds":result.algorithm.problem.stage_seconds,
              "stage_candidates":result.algorithm.problem.stage_candidates,
              "source_hash":job["source_hash"],"dataset_hash":job["dataset_hash"],
              "environment":job["environment"],"concurrent_workers":job["workers"],
              "diagnosis":diagnose(instance,choices),
              "sectioning_fallback":{key:{k:v for k,v in value.items() if k != "plan"}
                                      for key,value in instance.sectioning_cache.items()}}
    np.savez_compressed(folder/(stem+".npz"),choices=choices,X=result.algorithm.pop.get("X"),
                        F=result.algorithm.pop.get("F"),G=result.algorithm.pop.get("G"))
    path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return record


def ranking(folder):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")
            if not p.name.startswith("_")]
    cases = {}
    for row in rows:
        key = row["instance"],row["seed"],row["target_evaluations"]
        cases.setdefault(key,[]).append(row)
    # Compare the same measured evaluation horizon within each paired case.
    # Time-censored runs never count as if they reached the requested budget.
    summaries = {}
    for key,paired in cases.items():
        shared = min(r["evaluations"] for r in paired)
        case = []
        for row in paired:
            points = [p for p in row["trace"] if p["evaluations"] <= shared]
            if not points:
                continue
            point = points[-1]
            quality = (point["best_observed_hard"],
                       point["best_observed_internal_total"] if point["best_observed_hard"] == 0 else 0)
            case.append((row,point,quality))
        for row,point,quality in case:
            item = summaries.setdefault(row["variant"],{"ranks":[],"feasible":0,"cases":0,
                                                        "time_censored":0,"cpu_seconds":[]})
            item["ranks"].append(sum(q < quality for _,_,q in case))
            item["feasible"] += quality[0] == 0
            item["cases"] += 1
            item["time_censored"] += row["stop_reason"] == "time_cap"
            item["cpu_seconds"].append(row["cpu_seconds"])
    result = {key:{"mean_rank":round(float(np.mean(value["ranks"])),3),
                  "worst_rank":max(value["ranks"]),"feasible_cases":value["feasible"],
                  "cases":value["cases"],"time_censored":value["time_censored"],
                  "median_cpu_seconds":round(float(np.median(value["cpu_seconds"])),3)}
              for key,value in summaries.items()}
    return dict(sorted(result.items(),key=lambda pair:(pair[1]["mean_rank"],pair[1]["worst_rank"],pair[0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,default=ROOT/"profiles/pilot.json")
    parser.add_argument("--variants",nargs="+",choices=VARIANTS,default=["current"])
    parser.add_argument("--instances",nargs="+",default=["nbi-spr18","yach-fal17"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[17,42])
    parser.add_argument("--evaluations",type=int,default=240)
    parser.add_argument("--generations",type=int,default=2000)
    parser.add_argument("--seconds",type=float,default=150)
    parser.add_argument("--instance-evaluations",nargs="*",default=[],metavar="NAME:COUNT")
    parser.add_argument("--instance-seconds",nargs="*",default=[],metavar="NAME:SECONDS")
    parser.add_argument("--replay-manifest",type=Path,help="Replay frozen settings instead of reading the live profile")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--priority-instances",nargs="*",default=[])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--summarize",action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(ranking(args.output),indent=2))
    else:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        fingerprint,env = source_hash(),environment()
        jobs = []
        try:
            evaluation_limits = {name:int(value) for name,value in
                                 (item.rsplit(":",1) for item in args.instance_evaluations)}
            time_limits = {name:float(value) for name,value in
                           (item.rsplit(":",1) for item in args.instance_seconds)}
        except ValueError:
            parser.error("Instance limits must be NAME:NUMBER")
        for variant in args.variants:
            solver,repair = VARIANTS[variant]
            for instance in args.instances:
                per_instance = profile.get("instances",{}).get(instance,{})
                per_solver = {k:v for k,v in per_instance.items() if k in profile["solver"]}
                per_repair = {k.removeprefix("repair_"):v for k,v in per_instance.items()
                              if k.startswith("repair_")}
                for seed in args.seeds:
                    jobs.append({"variant":variant,"instance":instance,"seed":seed,
                                 "solver":{**profile["solver"],**per_solver,**solver},
                                 "repair":{**profile["repair"],**per_repair,**repair},
                                 "evaluations":evaluation_limits.get(instance,args.evaluations),
                                 "generations":args.generations,
                                 "seconds":time_limits.get(instance,args.seconds),"workers":args.workers,
                                 "source_hash":fingerprint,"environment":env,
                                 "dataset_hash":digest(ROOT/"Dataset_ITC2019"/(instance+".xml")),
                                 "output":str(args.output.resolve())})
        if args.replay_manifest:
            jobs = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
            for job in jobs:
                job["output"] = str(args.output.resolve())
                job["workers"] = args.workers
        # Interleave cases so no variant systematically occupies a different stage of the run.
        rng = np.random.default_rng(2026)
        rng.shuffle(jobs)
        priority = {name:i for i,name in enumerate(args.priority_instances)}
        jobs.sort(key=lambda job:priority.get(job["instance"],len(priority)))
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"_manifest.json").write_text(json.dumps(jobs,indent=2)+"\n",encoding="utf-8")
        done_count = 0
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
            pending = {pool.submit(measure,job) for job in jobs}
            while pending:
                done,pending = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    done_count += 1
                    print(json.dumps({"completed":done_count,"jobs":len(jobs),
                                      "variant":row["variant"],"instance":row["instance"],
                                      "seed":row["seed"],"hard":row["score"]["hard"],
                                      "evaluations":row["evaluations"],"seconds":row["seconds"],
                                      "stop":row["stop_reason"]}),flush=True)
                if not done:
                    print(json.dumps({"completed":done_count,"jobs":len(jobs),
                                      "elapsed_seconds":round(time.perf_counter()-started,1)}),flush=True)
        print(json.dumps(ranking(args.output),indent=2),flush=True)
