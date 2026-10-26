"""Check saved experiment accounting and exports; not an official ITC validator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def audit_attempt(attempt: dict) -> dict:
    folder = Path(attempt["folder"])
    name = attempt["instance"]
    row = next(record for record in json.loads((folder / "summary.json").read_text(encoding="utf-8"))
               if record["instance"] == name)
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    score = row["score"]
    check(sum(score["weighted"]) == score["total"], "weighted objectives do not sum to total")
    check(sum(score[key] for key in ("room_conflicts", "room_unavailable", "hard_distributions",
                                    "unassigned_requests")) == score["hard"], "hard accounting mismatch")
    check(row["score_engine"]["logical_requests"] == row["evaluations"], "logical scoring mismatch")
    check(row["score_engine"]["physical_calls"] + row["score_engine"]["cache_hits"] == row["evaluations"],
          "physical/cache accounting mismatch")
    breakdown = row["evaluation_breakdown"]
    check(sum(value for key, value in breakdown.items() if key != "final_verification") == row["evaluations"],
          "evaluation breakdown mismatch")
    child = folder / name
    if "controller" in row:
        events = [json.loads(line) for line in (child / "controller_trace.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        calls = [record for event in events for record in event["roles"]]
        controller = row["controller"]
        check(len(events) == controller["reviews"], "review count mismatch")
        check(sum(record.get("api_requests", 1) for record in calls) == controller["api_calls"], "request count mismatch")
        for total_key, record_key in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                                      ("thinking_tokens", "thinking_tokens"), ("total_native_tokens", "total_tokens")):
            check(sum(record.get(record_key, 0) or 0 for record in calls) == controller[total_key], total_key + " mismatch")
        check(abs(sum(record["latency_seconds"] for record in calls) - controller["api_latency_seconds"]) < 0.01,
              "API latency mismatch")
        allowed_sequences = (["single_agent"],) if attempt["method"] == "single" else (["inspector"], ["inspector", "planner"])
        check(all([record["role"] for record in event["roles"]] in allowed_sequences or
                  (not event["roles"] and event.get("decision_status") in
                   {"context_limit_skipped", "deadline_after_diagnostics"}) for event in events),
              "unexpected role sequence")
        check(all(record["model_requested"] == "gemini-3.5-flash-lite" for record in calls), "wrong requested model")
        if attempt["completion_status"] == "completed":
            check(not controller["api_failures"] and controller["comparison_eligible"], "completed but API ineligible")
    solution = child / "solution.xml"
    if row["feasible"]:
        check(score["hard"] == 0, "feasible export has hard violations")
        expected = json.loads((child / "expected.json").read_text(encoding="utf-8"))
        check(hashlib.sha256(solution.read_bytes()).hexdigest() == expected["solution_sha256"], "XML hash mismatch")
        check(expected["score"] == score, "expected score mismatch")
        manifest = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        check(expected["dataset_sha256"] == manifest["settings"]["instances"][0]["sha256"], "dataset hash mismatch")
        classes = ET.parse(solution).getroot().findall("class")
        check(len(classes) == row["classes"] == len({item.attrib["id"] for item in classes}), "XML class count/IDs mismatch")
    else:
        check(not solution.exists(), "infeasible endpoint has stale solution.xml")
    return {"instance": name, "seed": attempt["seed"], "method": attempt["method"],
            "folder": str(folder), "passed": not errors, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    checks = []
    for attempt in report["all_attempts"]:
        try:
            checks.append(audit_attempt(attempt))
        except (KeyError, OSError, ValueError, TypeError, ET.ParseError) as exc:
            checks.append({"instance": attempt["instance"], "seed": attempt["seed"],
                           "method": attempt["method"], "passed": False, "errors": [str(exc)]})
    report["artifact_audit"] = {"checked_attempts": len(checks),
        "passed": all(item["passed"] for item in checks), "failures": [item for item in checks if not item["passed"]],
        "scope": "Internal accounting, role/token traces, export hash/class IDs; not official validation."}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["artifact_audit"], indent=2))
    if not report["artifact_audit"]["passed"]:
        raise SystemExit(1)
