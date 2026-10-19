"""Summarize raw calibration trajectories and suggest measured local caps.

Never edit the live profile automatically. Incomplete/censored runs remain
explicit; a local plateau is not a certificate of solver or instance optimality.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def read_records(folders, variant=None):
    rows = []
    for folder in folders:
        paths = [folder] if folder.is_file() else folder.glob("*.json")
        for path in paths:
            if path.name.startswith("_"):
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if "score" in row and (variant is None or row["variant"] == variant):
                row["artifact"] = str(path)
                rows.append(row)
    return rows


def trajectory(row):
    points = row["trace"]
    hard_points = []
    best = math.inf
    for p in points:
        if p["best_observed_hard"] < best:
            best = p["best_observed_hard"]
            hard_points.append(p)
    feasible = [p for p in points if p["best_observed_hard"] == 0]
    if feasible:
        first, last = feasible[0], feasible[-1]
        threshold = last["best_observed_internal_total"]+0.1*(
            first["best_observed_internal_total"]-last["best_observed_internal_total"])
        knee = next(p for p in feasible if p["best_observed_internal_total"] <= threshold)
        desired = max(600,first["evaluations"]+400,knee["evaluations"]+100)
    else:
        first = knee = None
        desired = max(800,hard_points[-1]["evaluations"]+400)
    last_hard = hard_points[-1]
    events = row["monitor"]
    statuses = Counter()
    for name in ("room_chain_events","coordinated_repair_events","room_plateau_events"):
        for event in events.get(name,[]):
            statuses[name+":"+event["status"]] += 1
            neighborhood = event.get("neighborhood")
            if neighborhood:
                statuses["cp:"+neighborhood["status"]] += 1
    groups = Counter()
    for group in row["diagnosis"]["required_groups"]:
        groups[group["kind"]] += group["violations"]
    return {"instance":row["instance"],"variant":row["variant"],"seed":row["seed"],
            "hard":row["score"]["hard"],"total":row["score"]["total"],
            "components":{k:row["score"][k] for k in
                          ("room_conflicts","room_unavailable","hard_distributions","unassigned_requests")},
            "generations":points[-1]["generation"],"evaluations":row["evaluations"],
            "seconds":row["seconds"],"stop":row["stop_reason"],
            "first_feasible_evaluation":None if first is None else first["evaluations"],
            "last_hard_improvement_evaluation":last_hard["evaluations"],
            "evaluations_since_hard_improvement":row["evaluations"]-last_hard["evaluations"],
            "soft_gain_knee_evaluation":None if knee is None else knee["evaluations"],
            "desired_evaluations":math.ceil(desired/100)*100,
            "remaining_distribution_types":dict(groups),"repair_statuses":dict(statuses)}


def suggest(rows, profile):
    if not rows:
        raise ValueError("No calibration records")
    for key in ("source_hash","variant","environment"):
        if len({json.dumps(r.get(key),sort_keys=True) for r in rows}) != 1:
            raise ValueError(f"Mixed {key}")
    solvers = [{k:v for k,v in r["config"].items() if k not in
                {"generations","seed","evaluations"}} for r in rows]
    common_solver = solvers[0] if all(s == solvers[0] for s in solvers) else profile["solver"]
    common_repair = (rows[0]["repair"] if all(r["repair"] == rows[0]["repair"] for r in rows)
                     else profile["repair"])
    # For duplicate runs of a seed, use its longest measured trajectory.
    longest = {}
    for row in rows:
        key = row["instance"],row["seed"]
        if key not in longest or row["evaluations"] > longest[key]["evaluations"]:
            longest[key] = row
    groups = defaultdict(list)
    for row in longest.values():
        groups[row["instance"]].append(row)
    limits,statuses,evidence,endpoint_censoring = {},{},{},{}
    for name,cases in sorted(groups.items()):
        configs = [{k:v for k,v in r["config"].items() if k not in
                    {"generations","seed","evaluations"}} for r in cases]
        if any(c != configs[0] for c in configs) or any(r["repair"] != cases[0]["repair"] for r in cases):
            raise ValueError(f"Mixed settings within instance {name}")
        descriptions = [trajectory(r) for r in cases]
        desired = max(r["desired_evaluations"] for r in descriptions)
        common = min(r["evaluations"] for r in cases)
        measured_cap = max(cases[0]["config"]["population"],(common//100)*100)
        selected = min(desired,measured_cap)
        times,intervals = [],[]
        for row in cases:
            event = next(p for p in row["trace"] if p["evaluations"] >= selected)
            times.append(event["elapsed_seconds"])
            points = row["trace"]
            intervals.extend(points[i]["elapsed_seconds"]-points[i-1]["elapsed_seconds"]
                             for i in range(1,len(points)))
        # Shared cap has room for the current worst-case two-agent API allowance.
        api_reserve = profile["controller"]["max_reviews"]*2*profile["controller"]["api_timeout_seconds"]
        reserve = 2*max(intervals,default=times[0])+api_reserve
        seconds = math.ceil((1.35*max(times)+reserve)/30)*30
        censored = [r["seed"] for r in cases if r["stop_reason"] == "time_cap"]
        if censored:
            endpoint_censoring[name] = sorted(censored)
        statuses[name] = ("time_censored" if censored and desired > common
                          else "progressing_at_evaluation_cap" if desired > common
                          else "measured_local_knee" if all(r["score"]["hard"] == 0 for r in cases)
                          else "hard_plateau_observed")
        overrides = {k:v for k,v in configs[0].items() if common_solver.get(k) != v}
        overrides.update({"repair_"+k:v for k,v in cases[0]["repair"].items()
                          if common_repair.get(k) != v})
        limits[name] = {"generations":max(200,2*math.ceil(selected/configs[0]["population"])),
                        "evaluations":selected,"time_limit_seconds":seconds,**overrides}
        evidence[name] = {"seeds":sorted(r["seed"] for r in cases),
                          "measured_common_evaluations":common,"desired_evaluations":desired,
                          "selected_evaluations":selected,"status":statuses[name],
                          "endpoint_time_censored_seeds":sorted(censored),
                          "all_feasible":all(r["score"]["hard"] == 0 for r in cases),
                          "api_reserve_seconds":api_reserve}
    return {"solver":common_solver,"repair":common_repair,"instances":limits,
            "provenance":{"solver_source_sha256":rows[0]["source_hash"],
                          "candidate":rows[0]["variant"],
                          "recalibration_required":any(v in ("time_censored","progressing_at_evaluation_cap")
                                                       for v in statuses.values()),
                          "endpoint_time_censoring":endpoint_censoring,
                          "budget_status":statuses},"evidence":evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders",nargs="+",type=Path,help="Artifact folders or individual measurement JSON files")
    parser.add_argument("--variant")
    parser.add_argument("--suggest-profile",type=Path)
    args = parser.parse_args()
    rows = read_records(args.folders,args.variant)
    if args.suggest_profile:
        result = suggest(rows,json.loads(args.suggest_profile.read_text(encoding="utf-8")))
    else:
        result = [trajectory(r) for r in sorted(rows,key=lambda r:(r["instance"],r["seed"],r["variant"]))]
    print(json.dumps(result,indent=2))
