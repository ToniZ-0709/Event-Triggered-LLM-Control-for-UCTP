"""Run the ITC 2019 NSGA-III baseline on selected independent XML instances."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
import numpy as np
import pymoo

from itc2019.controller import BaselineMonitor, ControllerSettings
from itc2019.instance import Instance, load_instance
from itc2019.nsga3 import Config, run
from itc2019.scoring import Score, evaluate, validate_sectioning


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "Dataset_ITC2019"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_settings(paths: list[Path], config: Config, args: argparse.Namespace) -> dict:
    source_root = Path(__file__).resolve().parent
    sources = ("main.py", "itc2019/instance.py", "itc2019/nsga3.py", "itc2019/scoring.py",
               "itc2019/controller.py", "itc2019/regions.py", "itc2019/operators.py")
    return {
        "instances": [{"file": path.name, "sha256": _sha256(path)}
                      for path in sorted(paths, key=lambda path: path.name)],
        "config": asdict(config),
        "solution_metadata": {"author": args.author, "institution": args.institution,
                              "country": args.country},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pymoo": pymoo.__version__},
        "solver_sources_sha256": {name: _sha256(source_root / name) for name in sources},
    }


def _run_id(settings: dict, config: Config) -> str:
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return (f"run_n{len(settings['instances'])}_p{config.population}_g{config.generations}"
            f"_r{config.partitions}_s{config.seed}_{fingerprint}")


def _check_existing_run(folder: Path, run_id: str, settings: dict) -> bool:
    if folder.is_symlink():
        raise FileExistsError(f"Refusing to replace a symlink: {folder}")
    if not folder.exists():
        return False
    manifest = folder / "run_config.json"
    if not folder.is_dir() or not manifest.is_file():
        raise FileExistsError(f"Existing output has no run_config.json; refusing to replace: {folder}")
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if previous.get("run_id") != run_id or previous.get("settings") != settings:
        raise FileExistsError(f"Existing output belongs to different settings: {folder}")
    return True


def _publish_run(staging: Path, folder: Path, run_id: str, settings: dict) -> bool:
    replacing = _check_existing_run(folder, run_id, settings)
    backup = folder.parent / f".{run_id}.previous-{uuid.uuid4().hex}" if replacing else None
    if backup is not None:
        folder.rename(backup)
    try:
        staging.rename(folder)
    except BaseException:
        if backup is not None:
            backup.rename(folder)
        raise
    if backup is not None:
        shutil.rmtree(backup)
    return replacing


def _paths(args: argparse.Namespace) -> list[Path]:
    if args.all and (args.instance or args.manifest):
        raise ValueError("Use --all or --instance/--manifest, not both")
    names: list[str] = list(args.instance)
    if args.manifest:
        names.extend(line.strip() for line in args.manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#"))
    if args.all:
        paths = sorted(args.dataset.glob("*.xml"))
    else:
        paths = []
        for name in dict.fromkeys(names):
            if Path(name).name != name:
                raise ValueError(f"Instance name must be a filename, not a path: {name}")
            path = args.dataset / (name if name.endswith(".xml") else name + ".xml")
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    if not paths:
        raise ValueError("Select --instance NAME, --manifest FILE, or --all")
    return paths


def _write_solution(path: Path, instance: Instance, choices, score: Score,
                    assignments: dict[int, list[str]] | None, runtime: float,
                    author: str, institution: str, country: str,
                    technique: str = "NSGA-III") -> None:
    if score.hard != 0 or assignments is None:
        raise ValueError("Only a fully feasible, sectioned solution can be exported")
    root = ET.Element("solution", name=instance.name, runtime=f"{runtime:.3f}",
                      cores="1", technique=technique, author=author,
                      institution=institution, country=country)
    n = len(instance.classes)
    for index, cls in enumerate(instance.classes):
        chosen = cls.times[int(choices[index])]
        attributes = {"id": cls.id, "days": chosen.days, "start": str(chosen.start),
                      "weeks": chosen.weeks}
        if not cls.no_room:
            attributes["room"] = cls.room_options[int(choices[n + index])][0]
        element = ET.SubElement(root, "class", attributes)
        for student_id in sorted(assignments.get(index, [])):
            ET.SubElement(element, "student", id=student_id)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main(*, controller_factory=None, pipeline_root: Path | None = None,
         pipeline_name: str | None = None,
         controller_sources: tuple[Path, ...] = ()) -> None:
    pipeline_root = pipeline_root or Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=(
        f"Run the ITC 2019 {pipeline_name} pipeline" if pipeline_name else __doc__))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--instance", action="append", default=[], help="XML stem or filename; repeatable")
    parser.add_argument("--manifest", type=Path, help="Text file with one XML stem or filename per line")
    parser.add_argument("--all", action="store_true", help="Run every XML file in --dataset")
    parser.add_argument("--inspect", action="store_true", help="Parse and summarize instances without optimizing")
    parser.add_argument("--output", type=Path, default=pipeline_root / "results_itc2019")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-seconds", type=float,
                        help="Per-instance wall-clock limit for solver and controller")
    parser.add_argument("--author", default=pipeline_name or "NSGA-III baseline")
    parser.add_argument("--institution", default="Ho Chi Minh City University of Technology")
    parser.add_argument("--country", default="Vietnam")
    if controller_factory is not None:
        parser.add_argument("--review-interval", type=int, default=5,
                            help="Default controller review interval in generations")
        parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"),
                            help="OpenAI model; can also be set via OPENAI_MODEL")
        parser.add_argument("--api-timeout-seconds", type=float, default=30.0)
        parser.add_argument("--max-output-tokens", type=int, default=300)
        parser.add_argument("--warmup-generations", type=int, default=5)
        parser.add_argument("--stagnation-window", type=int, default=5)
        parser.add_argument("--min-review-interval", type=int, default=2)
        parser.add_argument("--max-review-interval", type=int, default=10)
        parser.add_argument("--diversity-threshold", type=float, default=0.25)
        parser.add_argument("--max-regions", type=int, default=6)
        parser.add_argument("--region-size", type=int, default=3)
        parser.add_argument("--max-operator-evaluations", type=int, default=8)
        parser.add_argument("--max-api-calls", type=int, default=100)
    args = parser.parse_args()
    if args.time_limit_seconds is not None and args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    paths = _paths(args)
    config = Config(args.population, args.generations, args.partitions, args.seed)
    if args.inspect:
        for path in paths:
            instance = load_instance(path)
            print(json.dumps({"instance": instance.name, "path": str(path),
                              "classes": len(instance.classes), "rooms": len(instance.rooms),
                              "courses": len(instance.courses), "students": len(instance.students),
                              "distributions": len(instance.distributions)}))
        return

    controller_settings = None
    client = None
    if controller_factory is not None:
        try:
            controller_settings = ControllerSettings(
                model=args.model,
                review_interval=args.review_interval,
                warmup_generations=args.warmup_generations,
                stagnation_window=args.stagnation_window,
                min_review_interval=args.min_review_interval,
                max_review_interval=args.max_review_interval,
                diversity_threshold=args.diversity_threshold,
                max_regions=args.max_regions,
                region_size=args.region_size,
                max_operator_evaluations=args.max_operator_evaluations,
                api_timeout_seconds=args.api_timeout_seconds,
                max_output_tokens=args.max_output_tokens,
                max_api_calls=args.max_api_calls,
            )
            from itc2019.llm import OpenAIDecisionClient
            client = OpenAIDecisionClient(args.model, args.api_timeout_seconds,
                                          args.max_output_tokens)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    settings = _run_settings(paths, config, args)
    settings["time_limit_seconds"] = args.time_limit_seconds
    if controller_factory is not None:
        settings["pipeline"] = {"name": pipeline_name, "status": "active"}
        settings["controller"] = asdict(controller_settings)
        settings["environment"]["openai"] = package_version("openai")
        settings["controller_sources_sha256"] = {
            str(source.resolve().relative_to(pipeline_root.parent.resolve())): _sha256(source)
            for source in controller_sources
        }
    run_id = _run_id(settings, config)
    args.output.mkdir(parents=True, exist_ok=True)
    run_folder = args.output.resolve() / run_id
    _check_existing_run(run_folder, run_id, settings)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=run_folder.parent))
    try:
        summaries = []
        for path in paths:
            instance = load_instance(path)
            summary = {"instance": instance.name, "path": str(path), "classes": len(instance.classes),
                       "rooms": len(instance.rooms), "courses": len(instance.courses),
                       "students": len(instance.students), "distributions": len(instance.distributions)}
            started = time.perf_counter()
            monitor = (controller_factory(instance, controller_settings, client,
                                          config.seed, started, args.time_limit_seconds)
                       if controller_factory is not None else
                       BaselineMonitor(instance, started, args.time_limit_seconds))
            choices, score, result = run(instance, config, callback=monitor,
                                         time_limit_seconds=args.time_limit_seconds)
            runtime = time.perf_counter() - started
            n = len(instance.classes)
            checked_score, assignments = evaluate(instance, choices[:n], choices[n:], keep_assignments=True)
            if checked_score != score:
                raise RuntimeError(f"Inconsistent final scoring on {instance.name}")
            folder = staging / instance.name
            folder.mkdir(parents=True, exist_ok=True)
            report = monitor.report()
            summary.update({"runtime_seconds": runtime, "evaluations": int(result.algorithm.evaluator.n_eval),
                            "feasible": score.hard == 0, "score": asdict(score), "config": asdict(config),
                            "selection_policy": "best_observed_lexicographic_hard_then_total",
                            "validation": {"internal_feasible": score.hard == 0,
                                           "official_status": "not_run"},
                            "run_id": run_id, "monitor": {
                                key: report[key] for key in ("generations_recorded",
                                                         "time_limit_seconds", "elapsed_seconds",
                                                         "best_observed_hard",
                                                         "best_observed_internal_total")}})
            if controller_factory is not None:
                summary["controller"] = {key: value for key, value in report.items()
                                         if key not in summary["monitor"]}
            (folder / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            (folder / "anytime_trace.jsonl").write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n"
                        for event in monitor.anytime_events), encoding="utf-8")
            if controller_factory is not None:
                (folder / "controller_trace.jsonl").write_text(
                    "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in monitor.events),
                    encoding="utf-8")
            np.savez_compressed(folder / "choices.npz", time_indices=choices[:n], room_indices=choices[n:])
            if score.hard == 0:
                validate_sectioning(instance, assignments or {})
                _write_solution(folder / "solution.xml", instance, choices, score, assignments,
                                runtime, args.author, args.institution, args.country,
                                pipeline_name or "NSGA-III")
            print(json.dumps({"instance": instance.name, "hard": score.hard,
                              "weighted_total": score.total, "runtime_seconds": round(runtime, 2),
                              "solution_xml": score.hard == 0}))
            summaries.append(summary)

        (staging / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        (staging / "run_config.json").write_text(json.dumps({
            "run_id": run_id, "settings": settings,
            "dataset_directory": str(args.dataset.resolve()),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        replaced = _publish_run(staging, run_folder, run_id, settings)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"run_id": run_id, "results_dir": str(run_folder),
                      "overwrote_existing": replaced}))


if __name__ == "__main__":
    main()
