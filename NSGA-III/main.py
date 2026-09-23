"""Run the ITC 2019 NSGA-III baseline on selected independent XML instances."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np

from itc2019.instance import Instance, load_instance
from itc2019.nsga3 import Config, run
from itc2019.scoring import Score, evaluate, validate_sectioning


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "Dataset_ITC2019"


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
                    author: str, institution: str, country: str) -> None:
    if score.hard != 0 or assignments is None:
        raise ValueError("Only a fully feasible, sectioned solution can be exported")
    root = ET.Element("solution", name=instance.name, runtime=f"{runtime:.3f}",
                      cores="1", technique="NSGA-III", author=author,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--instance", action="append", default=[], help="XML stem or filename; repeatable")
    parser.add_argument("--manifest", type=Path, help="Text file with one XML stem or filename per line")
    parser.add_argument("--all", action="store_true", help="Run every XML file in --dataset")
    parser.add_argument("--inspect", action="store_true", help="Parse and summarize instances without optimizing")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results_itc2019")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--author", default="NSGA-III baseline")
    parser.add_argument("--institution", default="Ho Chi Minh City University of Technology")
    parser.add_argument("--country", default="Vietnam")
    args = parser.parse_args()
    paths = _paths(args)
    config = Config(args.population, args.generations, args.partitions, args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_seed{args.seed}"
    run_folder = args.output / run_id
    summaries = []
    for path in paths:
        instance = load_instance(path)
        summary = {"instance": instance.name, "path": str(path), "classes": len(instance.classes),
                   "rooms": len(instance.rooms), "courses": len(instance.courses),
                   "students": len(instance.students), "distributions": len(instance.distributions)}
        if args.inspect:
            print(json.dumps(summary))
            continue
        started = time.perf_counter()
        choices, score, result = run(instance, config)
        runtime = time.perf_counter() - started
        n = len(instance.classes)
        checked_score, assignments = evaluate(instance, choices[:n], choices[n:], keep_assignments=True)
        if checked_score != score:
            raise RuntimeError(f"Inconsistent final scoring on {instance.name}")
        folder = run_folder / instance.name
        folder.mkdir(parents=True, exist_ok=True)
        summary.update({"runtime_seconds": runtime, "evaluations": int(result.algorithm.evaluator.n_eval),
                        "feasible": score.hard == 0, "score": asdict(score), "config": asdict(config),
                        "run_id": run_id})
        (folder / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(folder / "choices.npz", time_indices=choices[:n], room_indices=choices[n:])
        if score.hard == 0:
            validate_sectioning(instance, assignments or {})
            _write_solution(folder / "solution.xml", instance, choices, score, assignments,
                            runtime, args.author, args.institution, args.country)
        print(json.dumps({"instance": instance.name, "hard": score.hard,
                          "weighted_total": score.total, "runtime_seconds": round(runtime, 2),
                          "solution_xml": score.hard == 0}))
        summaries.append(summary)
    if summaries:
        (run_folder / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
