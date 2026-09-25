"""Compare three already completed, paired ITC 2019 pipeline runs.

This reports internal scores only. It does not claim official ITC validation or
statistical significance across seeds or instances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("baseline", "single", "multi")


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
    return config, by_instance


def _paired_delta(left: dict, right: dict) -> dict:
    left_feasible = bool(left["feasible"])
    right_feasible = bool(right["feasible"])
    return {
        "feasibility_change": int(right_feasible) - int(left_feasible),
        "internal_total_change_if_both_feasible": (
            right["score"]["total"] - left["score"]["total"]
            if left_feasible and right_feasible else None),
        "runtime_seconds_change": right["runtime_seconds"] - left["runtime_seconds"],
        "pymoo_evaluations_change": right["evaluations"] - left["evaluations"],
    }


def compare(folders: dict[str, Path], *, allow_generation_only: bool = False) -> dict:
    runs = {name: _load_run(folder) for name, folder in folders.items()}
    baseline = runs["baseline"][0]["settings"]
    if "pipeline" in baseline:
        raise ValueError("The baseline input is a controller run")
    shared_keys = ("instances", "config", "time_limit_seconds", "solver_sources_sha256")
    for name in ("single", "multi"):
        settings = runs[name][0]["settings"]
        for key in shared_keys:
            if settings.get(key) != baseline.get(key):
                raise ValueError(f"{name} differs from baseline on {key}")
        for dependency in ("python", "numpy", "pymoo"):
            if settings["environment"].get(dependency) != baseline["environment"].get(dependency):
                raise ValueError(f"{name} differs from baseline on {dependency}")
    if baseline.get("time_limit_seconds") is None and not allow_generation_only:
        raise ValueError("Wall-clock comparison requires --time-limit-seconds in all runs")
    single_model = runs["single"][0]["settings"]["controller"]["model"]
    multi_model = runs["multi"][0]["settings"]["controller"]["model"]
    if single_model != multi_model:
        raise ValueError("Single-agent and multi-agent runs use different models")
    if (runs["single"][0]["settings"]["controller"] !=
            runs["multi"][0]["settings"]["controller"]):
        raise ValueError("Controller settings differ between single and multi runs")
    if (runs["single"][0]["settings"]["environment"].get("openai") !=
            runs["multi"][0]["settings"]["environment"].get("openai")):
        raise ValueError("OpenAI SDK versions differ between single and multi runs")
    def llm_source_hash(method: str) -> str:
        sources = runs[method][0]["settings"]["controller_sources_sha256"]
        matches = [value for key, value in sources.items()
                   if key.replace("\\", "/").endswith("NSGA-III/itc2019/llm.py")]
        if len(matches) != 1:
            raise ValueError(f"Missing shared OpenAI adapter hash in {method} run")
        return matches[0]
    if llm_source_hash("single") != llm_source_hash("multi"):
        raise ValueError("OpenAI adapter and prompts differ between single and multi runs")
    expected_names = {"single": "NSGA-III + Single-Agent",
                      "multi": "NSGA-III + Inspector-Planner"}
    for method, expected_name in expected_names.items():
        if runs[method][0]["settings"]["pipeline"]["name"] != expected_name:
            raise ValueError(f"{method} input is not a {expected_name} run")
    if set(runs["baseline"][1]) != set(runs["single"][1]) or (
            set(runs["baseline"][1]) != set(runs["multi"][1])):
        raise ValueError("Run summaries contain different instances")

    rows = []
    for instance in sorted(runs["baseline"][1]):
        records = {method: runs[method][1][instance] for method in METHODS}
        rows.append({
            "instance": instance,
            "methods": {
                method: {
                    "feasible_internal": bool(row["feasible"]),
                    "hard": row["score"]["hard"],
                    "internal_total": row["score"]["total"],
                    "weighted_objectives": row["score"]["weighted"],
                    "official_status": row["validation"]["official_status"],
                    "runtime_seconds": row["runtime_seconds"],
                    "pymoo_evaluations": row["evaluations"],
                    "reviews": row.get("controller", {}).get("reviews", 0),
                    "api_calls": row.get("controller", {}).get("api_calls", 0),
                    "api_latency_seconds": row.get("controller", {}).get(
                        "api_latency_seconds", 0.0),
                } for method, row in records.items()
            },
            "single_minus_baseline": _paired_delta(records["baseline"], records["single"]),
            "multi_minus_single": _paired_delta(records["single"], records["multi"]),
        })
    return {
        "score_source": "internal_evaluator",
        "comparison_type": "paired_wall_clock" if baseline.get("time_limit_seconds") is not None
        else "paired_generation_cap",
        "seed": baseline["config"]["seed"],
        "time_limit_seconds": baseline.get("time_limit_seconds"),
        "model": single_model,
        "runs": {method: str(folders[method].resolve()) for method in METHODS},
        "instances": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in METHODS:
        parser.add_argument(f"--{method}", type=Path, required=True,
                            help=f"Completed {method} run directory")
    parser.add_argument("--allow-generation-only", action="store_true",
                        help="Allow runs without a wall-clock limit")
    parser.add_argument("--output", type=Path, help="Write comparison JSON to this file")
    args = parser.parse_args()
    try:
        report = compare({method: getattr(args, method) for method in METHODS},
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
