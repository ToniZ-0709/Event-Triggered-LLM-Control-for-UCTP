"""Read the archived Phase 1 manifest and reports without modifying them."""
from __future__ import annotations

import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1] / "ExperimentResult" / "phase1"
METHODS = ("baseline", "single", "multi")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def archived_runs(root: Path = ARCHIVE):
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    result = {}
    for campaign in manifest["campaigns"]:
        for attempt in campaign["published_attempts"]:
            key = (campaign["name"], attempt["seed"], attempt["instance"], attempt["method"])
            if key in result:
                raise ValueError(f"Duplicate archive entry: {key}")
            path = (root / attempt["archive"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Archive path outside root: {path}")
            result[key] = (path, attempt)
    return result


def report_rows(root: Path = ARCHIVE):
    rows = []
    for campaign in ("expanded_pilot", "tg_no_timecap"):
        report = read_json(root / campaign / "reports" / f"{campaign}_comparison.json")
        rows.extend((campaign, row) for row in report["instances"])
    return rows
