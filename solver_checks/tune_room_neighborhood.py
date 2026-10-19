"""Small repeatable parameter sweep for the saved room bottleneck."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.instance import load_instance  # noqa: E402
from itc2019.neighborhood import room_neighborhood_repair  # noqa: E402
from itc2019.scoring import evaluate  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("choices", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instance = load_instance(args.instance)
    with np.load(args.choices) as saved:
        source = np.concatenate([saved["time_indices"], saved["room_indices"]])
    n = len(instance.classes)
    before, _ = evaluate(instance, source[:n], source[n:])
    rows = []
    for classes, options in ((16, 24), (24, 32), (32, 32), (32, 64)):
        for seed in (17, 42):
            started = time.perf_counter()
            candidate, stats = room_neighborhood_repair(
                instance, source, deadline=started + 5, seed=seed,
                max_classes=classes, max_options=options, source_score=before)
            after = None if candidate is None else evaluate(instance, candidate[:n], candidate[n:])[0].hard
            row = {"classes": classes, "options": options, "seed": seed,
                   "source_hard": before.hard, "candidate_hard": after,
                   "seconds": round(time.perf_counter() - started, 3),
                   "status": stats["status"], "active_classes": stats["active_classes"],
                   "changed_classes": stats["changed_classes"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "instance": instance.name, "source_choices_sha256": hashlib.sha256(args.choices.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((ROOT / "NSGA-III/itc2019/neighborhood.py").read_bytes()).hexdigest(),
        "rows": rows}, indent=2) + "\n", encoding="utf-8")
