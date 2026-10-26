"""Compare completed baseline/single runs, optionally including multi-agent.

This reports internal scores only. It does not claim official ITC validation or
statistical significance across seeds or instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = ("baseline", "single", "multi")


def _check_review_budget(single_controller: dict, multi_controller: dict) -> None:
    review_cap = single_controller.get("max_reviews")
    if (not isinstance(review_cap, int) or review_cap < 1 or
            single_controller["max_api_calls"] < review_cap or
            multi_controller["max_api_calls"] < 2 * review_cap):
        raise ValueError("API call limits must permit the same number of reviews "
                         "(one call per single review, up to two per multi review)")


def _load_run(folder: Path) -> tuple[dict, dict[str, dict]]:
    config_path = folder / "run_config.json"
    summary_path = folder / "summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise ValueError(f"Missing completed run files in {folder}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Invalid summary in {folder}")
    by_instance = {row["instance"]: row for row in rows}
    if len(by_instance) != len(rows):
        raise ValueError(f"Duplicate instance in {folder}")
    expected = {Path(item["file"]).stem for item in config["settings"]["instances"]}
    if set(by_instance) != expected:
        raise ValueError(f"Run summary does not match its input manifest: {folder}")
    for name, row in by_instance.items():
        receipt = folder / name / "official_validation.json"
        solution = folder / name / "solution.xml"
        if receipt.is_file():
            validation = json.loads(receipt.read_text(encoding="utf-8"))
            digest = hashlib.sha256(solution.read_bytes()).hexdigest() if solution.is_file() else None
            if (validation.get("instance") != name or
                    validation.get("solution_sha256") != digest or digest is None):
                raise ValueError(f"Official validation receipt does not match XML for {name}")
            if validation.get("official_status") == "passed" and (
                    validation.get("official_result") != "OK" or
                    validation.get("official_total") != row["score"]["total"]):
                raise ValueError(f"Official validation score does not match {name}")
            row["validation"] = {**row["validation"],
                "official_status": validation["official_status"]}
    return config, by_instance


def _paired_delta(left: dict, right: dict) -> dict:
    left_feasible = bool(left["feasible"])
    right_feasible = bool(right["feasible"])
    return {
        "hard_change": right["score"]["hard"] - left["score"]["hard"],
        "feasibility_change": int(right_feasible) - int(left_feasible),
        "internal_total_change_if_both_feasible": (
            right["score"]["total"] - left["score"]["total"]
            if left_feasible and right_feasible else None),
        "runtime_seconds_change": right["runtime_seconds"] - left["runtime_seconds"],
        "evaluations_change": right["evaluations"] - left["evaluations"],
    }


def compare(folders: dict[str, Path], *, allow_generation_only: bool = False) -> dict:
    if set(folders) not in ({"baseline", "single"}, set(METHODS)):
        raise ValueError("Provide baseline and single runs, optionally a multi run")
    methods = tuple(method for method in METHODS if method in folders)
    runs = {name: _load_run(folder) for name, folder in folders.items()}
    baseline = runs["baseline"][0]["settings"]
    if "pipeline" in baseline:
        raise ValueError("The baseline input is a controller run")
    shared_keys = ("instances", "config", "repair", "time_limit_seconds", "stop_on_feasible",
                   "solver_sources_sha256")
    for name in methods[1:]:
        settings = runs[name][0]["settings"]
        for key in shared_keys:
            if settings.get(key) != baseline.get(key):
                raise ValueError(f"{name} differs from baseline on {key}")
        for dependency in ("python", "numpy", "pymoo", "ortools"):
            if settings["environment"].get(dependency) != baseline["environment"].get(dependency):
                raise ValueError(f"{name} differs from baseline on {dependency}")
    if baseline.get("time_limit_seconds") is None and not allow_generation_only:
        raise ValueError("Wall-clock comparison requires --time-limit-seconds in all runs")
    single_model = runs["single"][0]["settings"]["controller"]["model"]
    single_controller = runs["single"][0]["settings"]["controller"]
    if "multi" in runs:
        multi_controller = runs["multi"][0]["settings"]["controller"]
        if single_model != multi_controller["model"]:
            raise ValueError("Single-agent and multi-agent runs use different models")
        if ({key: value for key, value in single_controller.items() if key != "max_api_calls"} !=
                {key: value for key, value in multi_controller.items() if key != "max_api_calls"}):
            raise ValueError("Controller settings differ between single and multi runs")
        _check_review_budget(single_controller, multi_controller)
    for method in methods[1:]:
        settings = runs[method][0]["settings"]
        if settings["controller"].get("provider") != "gemini":
            raise ValueError("Both controller runs must use the Gemini adapter")
        for row in runs[method][1].values():
            controller = row.get("controller", {})
            if not controller.get("comparison_eligible", False):
                raise ValueError(f"{method}/{row['instance']} has incomplete API coverage: "
                                 f"{controller.get('completion_status', 'unknown')}. "
                                 "Do not interpret quota/network failures as controller decisions.")
    if "multi" in runs and runs["single"][0]["settings"].get("llm") != runs["multi"][0]["settings"].get("llm"):
        raise ValueError("Gemini transport, model or pacing policies differ")
    if ("multi" in runs and runs["single"][0]["settings"]["environment"].get("google-genai") !=
            runs["multi"][0]["settings"]["environment"].get("google-genai")):
        raise ValueError("Google GenAI SDK versions differ between single and multi runs")
    def llm_source_hash(method: str) -> str:
        sources = runs[method][0]["settings"]["controller_sources_sha256"]
        matches = [value for key, value in sources.items()
                   if key.replace("\\", "/").endswith("NSGA-III/itc2019/llm.py")]
        if len(matches) != 1:
            raise ValueError(f"Missing shared Gemini adapter hash in {method} run")
        return matches[0]
    if "multi" in runs and llm_source_hash("single") != llm_source_hash("multi"):
        raise ValueError("Gemini adapter and prompts differ between single and multi runs")
    expected_names = {"single": "NSGA-III + Single-Agent",
                      "multi": "NSGA-III + Inspector-Planner"}
    for method, expected_name in expected_names.items():
        if method not in runs:
            continue
        if runs[method][0]["settings"]["pipeline"]["name"] != expected_name:
            raise ValueError(f"{method} input is not a {expected_name} run")
    if any(set(runs["baseline"][1]) != set(runs[method][1]) for method in methods[1:]):
        raise ValueError("Run summaries contain different instances")

    rows = []
    for instance in sorted(runs["baseline"][1]):
        records = {method: runs[method][1][instance] for method in methods}
        single_versions = records["single"].get("controller", {}).get("models_returned", [])
        multi_versions = records.get("multi", {}).get("controller", {}).get("models_returned", [])
        if single_versions and multi_versions and set(single_versions) != set(multi_versions):
            raise ValueError(f"Returned Gemini model versions differ for {instance}")
        rows.append({
            "instance": instance,
            "methods": {
                method: {
                    "feasible_internal": bool(row["feasible"]),
                    "hard": row["score"]["hard"],
                    "hard_components": {key: row["score"].get(key) for key in (
                        "room_conflicts", "room_unavailable", "hard_distributions", "unassigned_requests")},
                    "internal_total": row["score"]["total"],
                    "weighted_objectives": row["score"]["weighted"],
                    "official_status": row["validation"]["official_status"],
                    "runtime_seconds": row["runtime_seconds"],
                    "evaluations": row["evaluations"],
                    "physical_scoring_calls": row.get("score_engine", {}).get("physical_calls"),
                    "scoring_cache_hits": row.get("score_engine", {}).get("cache_hits"),
                    "generations": row.get("monitor", {}).get("generations_recorded"),
                    "evaluation_cap_reached": (row["evaluations"] >= baseline["config"]["evaluations"]
                        if baseline["config"].get("evaluations") is not None else None),
                    "time_cap_reached": (row["runtime_seconds"] >= baseline["time_limit_seconds"]
                        if baseline.get("time_limit_seconds") is not None else None),
                    "room_chain_scoring_calls": row.get("monitor", {}).get(
                        "room_chain_scoring_calls", 0),
                    "reviews": row.get("controller", {}).get("reviews", 0),
                    "api_calls": row.get("controller", {}).get("api_calls", 0),
                    "api_latency_seconds": row.get("controller", {}).get(
                        "api_latency_seconds", 0.0),
                    "provider": row.get("controller", {}).get("provider"),
                    "api_failures": row.get("controller", {}).get("api_failures", {}),
                    "input_tokens": row.get("controller", {}).get("input_tokens", 0),
                    "output_tokens": row.get("controller", {}).get("output_tokens", 0),
                    "thinking_tokens": row.get("controller", {}).get("thinking_tokens", 0),
                    "total_native_tokens": row.get("controller", {}).get("total_native_tokens", 0),
                    "interventions": row.get("controller", {}).get("interventions", 0),
                    "selected_interventions": row.get("controller", {}).get("selected_interventions", 0),
                    "fallback_no_op": row.get("controller", {}).get("fallback_no_op", 0),
                    "region_scoring_calls": row.get("controller", {}).get("region_scoring_calls", 0),
                    "operator_scoring_rechecks": row.get("controller", {}).get("operator_scoring_rechecks", 0),
                } for method, row in records.items()
            },
            "single_minus_baseline": _paired_delta(records["baseline"], records["single"]),
            **({"multi_minus_single": _paired_delta(records["single"], records["multi"])}
               if "multi" in records else {}),
        })
    return {
        "score_source": "internal_evaluator",
        "comparison_type": "paired_shared_caps",
        "budget_note": "Shared generation/evaluation/time ceilings; actual generations and runtime may differ. This is not an equal-wall-clock experiment.",
        "methods": list(methods),
        "seed": baseline["config"]["seed"],
        "time_limit_seconds": baseline.get("time_limit_seconds"),
        "model": single_model,
        "provider": "gemini",
        "runs": {method: str(folders[method].resolve()) for method in methods},
        "instances": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in METHODS:
        parser.add_argument(f"--{method}", type=Path, required=method != "multi",
                            help=f"Completed {method} run directory")
    parser.add_argument("--allow-generation-only", action="store_true",
                        help="Allow runs without a wall-clock limit")
    parser.add_argument("--output", type=Path, help="Write comparison JSON to this file")
    args = parser.parse_args()
    try:
        report = compare({method: getattr(args, method) for method in METHODS
                          if getattr(args, method) is not None},
                         allow_generation_only=args.allow_generation_only)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        try:
            args.output.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            parser.error(str(exc))
        print(args.output.resolve())


if __name__ == "__main__":
    main()
