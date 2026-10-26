"""Read archived v1/v2 traces without changing scores, reports or attempts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

METHODS = ("baseline", "single", "multi")


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_anytime(path):
    points = []
    for raw in read_jsonl(path):
        # Population best can regress. Never substitute it for the archive.
        hard = raw.get("best_observed_hard")
        total = raw.get("best_observed_internal_total")
        point = {"generation": raw["generation"], "seconds": raw["elapsed_seconds"],
                 "evaluations": raw["evaluations"], "hard": hard, "internal_total": total,
                 "placement_diversity": raw.get("placement_diversity"),
                 "weighted_objectives": raw.get("incumbent_weighted_objectives")}
        if min(point["seconds"], point["evaluations"]) < 0:
            raise ValueError("Negative recorded progress")
        if points:
            previous = points[-1]
            if point["seconds"] < previous["seconds"] or point["evaluations"] < previous["evaluations"]:
                raise ValueError("Trace time/evaluations must be nondecreasing")
            if hard is not None and total is not None and previous["hard"] is not None:
                if (hard, total) > (previous["hard"], previous["internal_total"]):
                    raise ValueError("Recorded incumbent regresses")
        points.append(point)
    return points


def review_points(events, anytime):
    checkpoints = {p["generation"]: p for p in anytime}
    cumulative = calls = unknown = 0
    result = []
    for event in events:
        native = 0
        latency = 0.0
        review_calls = 0
        for role in event.get("roles", []):
            requests = role.get("api_requests", 1)
            review_calls += requests
            latency += role.get("latency_seconds") or 0.0  # already includes pacing
            if role.get("total_tokens") is None:
                if requests:
                    unknown += requests
            else:
                native += role["total_tokens"]
        cumulative += native
        calls += review_calls
        checkpoint = checkpoints.get(event["generation"])
        outcome = event.get("outcome") or {}
        state = event.get("state") or {}
        before = event.get("incumbent_before")
        if before is None:
            old = state.get("global", {})
            before = {"hard": old.get("incumbent_hard"), "internal_total": old.get("incumbent_internal_total")}
        after = event.get("incumbent_after", {})
        known_pair = all(d.get(k) is not None for d in (before, after) for k in ("hard", "internal_total"))
        source = (outcome.get("source_hard"), outcome.get("source_total"))
        candidate = (outcome.get("candidate_hard"), outcome.get("candidate_total"))
        source_improved = None if None in source+candidate else (
            candidate[0] < source[0] if source[0] else candidate[0] == 0 and candidate[1] < source[1])
        exact_time = event.get("review_completed_elapsed_seconds")
        result.append({"generation": event["generation"],
            "seconds": exact_time if exact_time is not None else None if checkpoint is None else checkpoint["seconds"],
            "evaluations": None if checkpoint is None else checkpoint["evaluations"],
            "hard": None if checkpoint is None else checkpoint["hard"],
            "timing_resolution": "review_completion" if exact_time is not None else "same_generation_post_review" if checkpoint else "unknown",
            "operator": event.get("action", {}).get("operator_id"),
            "decision_status": event.get("decision_status"), "validation": event.get("validation"),
            "status": outcome.get("status"), "selected": outcome.get("selected", False),
            "admitted": outcome.get("admitted", candidate[0] is not None and outcome.get("status") != "scoring_mismatch"),
            "source_improved": source_improved,
            "incumbent_improved": None if not known_pair else
                (after["hard"], after["internal_total"]) < (before["hard"], before["internal_total"]),
            "api_calls": review_calls, "cumulative_api_calls": calls,
            "reported_tokens": native, "cumulative_reported_tokens": cumulative,
            "usage_complete": unknown == 0, "unknown_usage_requests_cumulative": unknown,
            "latency_seconds": latency})
    return result


def load_run(run_folder, instance, method, endpoint):
    run_folder = Path(run_folder)
    folder = run_folder / instance
    if not folder.is_dir() and (run_folder / "anytime_trace.jsonl").is_file():
        folder = run_folder
    config_path = run_folder / "run_config.json"
    settings = json.loads(config_path.read_text(encoding="utf-8")).get("settings", {}) if config_path.is_file() else {}
    anytime_path, controller_path = folder / "anytime_trace.jsonl", folder / "controller_trace.jsonl"
    anytime = read_anytime(anytime_path)
    reviews = review_points(read_jsonl(controller_path), anytime)
    return {"method": method, "endpoint": endpoint, "anytime": anytime, "reviews": reviews,
        "settings": settings, "trace_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (anytime_path, controller_path, config_path) if p.is_file()},
        "warnings": (["anytime_trace_missing"] if not anytime else []) +
                    (["controller_trace_missing"] if method != "baseline" and not controller_path.is_file() else [])}


def shared_ranges(runs):
    result = {}
    for field in ("seconds", "evaluations"):
        ranges = [(r["anytime"][0][field], r["anytime"][-1][field]) for r in runs.values() if r["anytime"]]
        result[field] = None if len(ranges) != len(runs) else [max(a for a,b in ranges), min(b for a,b in ranges)]
        if result[field] is not None and result[field][0] > result[field][1]:
            result[field] = None
    return result


def campaign_costs(report, seed, instance):
    attempts = [a for a in report.get("all_attempts", []) if a.get("seed") == seed and a.get("instance") == instance]
    unpublished = [a for a in report.get("unpublished_attempts", []) if a.get("seed") == seed and a.get("instance") == instance]
    return {"scope": "whole_campaign_attempts_in_source_report", "published_attempts": attempts,
            "unpublished_attempts": unpublished,
            "usage_note": "Failed-request and interrupted-request usage remains unknown even when reported tokens are zero.",
            "recoveries": [a for a in report.get("human_authorized_recoveries", [])
                           if a.get("seed") == seed and a.get("instance") == instance]}
