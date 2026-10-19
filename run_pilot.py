"""Run the selected local pilot preset with a separate budget for each instance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENTRY = {"baseline": ROOT / "NSGA-III/main.py",
         "single": ROOT / "Single-Agent Controller/main.py",
         "multi": ROOT / "Multi-Agent Controller/main.py"}


def commands(profile: dict, *, pipeline: str, seed: int,
             instances: list[str], model: str | None, output: Path | None,
             confirm_free_tier: bool = False) -> list[list[str]]:
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported pilot profile schema")
    selected = instances or list(profile["instances"])
    result = []
    for instance in selected:
        if instance not in profile["instances"]:
            raise ValueError(f"No measured budget for {instance}")
        command = [sys.executable, str(ENTRY[pipeline]), "--instance", instance, "--seed", str(seed)]
        local = profile["instances"][instance]
        solver_and_limits = {key:value for key,value in local.items() if not key.startswith("repair_")}
        for key, value in {**profile["solver"], **solver_and_limits}.items():
            if value is not None:
                command.extend(("--" + key.replace("_", "-"), str(value)))
        local_repair = {key.removeprefix("repair_"):value for key,value in local.items()
                        if key.startswith("repair_")}
        for key, value in {**profile.get("repair", {}), **local_repair}.items():
            command.extend(("--repair-" + key.replace("_", "-"), str(value)))
        if pipeline != "baseline":
            controller = dict(profile["controller"])
            controller["max_api_calls"] = controller["max_reviews"] * (2 if pipeline == "multi" else 1)
            for key, value in controller.items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            llm = profile.get("llm", {})
            command.extend(("--provider", llm.get("provider", "gemini")))
            if model or llm.get("model"):
                command.extend(("--model", model or llm["model"]))
            command.extend(("--api-min-interval-seconds",
                            str(llm.get("min_request_interval_seconds", 10.0))))
            command.extend(("--api-max-input-characters",
                            str(llm.get("max_input_characters", 60000))))
            if confirm_free_tier:
                command.append("--confirm-free-tier")
        if output:
            command.extend(("--output", str(output.resolve() / pipeline)))
        result.append(command)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/pilot.json")
    parser.add_argument("--pipeline", choices=ENTRY, default="baseline")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--model")
    parser.add_argument("--confirm-free-tier", action="store_true",
                        help="Pass only after checking the API project's Free Tier in AI Studio")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        work = commands(profile, pipeline=args.pipeline, seed=args.seed,
                        instances=args.instance, model=args.model or os.environ.get("GEMINI_MODEL"), output=args.output,
                        confirm_free_tier=args.confirm_free_tier)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    for command in work:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
