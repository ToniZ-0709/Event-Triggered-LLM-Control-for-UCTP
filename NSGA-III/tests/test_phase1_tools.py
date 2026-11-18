"""Checks for Phase 1 archive mapping and audit integrity."""
import shutil
from pathlib import Path

from tools.audit_expanded_pilot import audit_attempt
from tools.phase1_data import archived_runs, report_rows
from tools.plot_phase1 import select_comparisons


def test_archive_covers_every_reported_run():
    runs = archived_runs()
    rows = report_rows()
    assert len(rows) == 16
    assert len(runs) == 48
    assert {campaign for campaign, _ in rows} == {"expanded_pilot"}
    assert "tg-spr18" not in {row["instance"] for _, row in rows}
    assert len(select_comparisons(Path(__file__).resolve().parents[2] / "ExperimentResult" / "phase1")) == 16
    for campaign, row in rows:
        for method in ("baseline", "single", "multi"):
            folder, _ = runs[(campaign, row["seed"], row["instance"], method)]
            assert (folder / row["instance"] / "anytime_trace.jsonl").is_file()


def test_audit_detects_changed_archive_file(tmp_path):
    runs = archived_runs()
    folder, attempt = runs[("expanded_pilot", 17, "muni-fi-spr16", "baseline")]
    copy = tmp_path / "run"
    shutil.copytree(folder, copy)
    assert audit_attempt(attempt, copy)["passed"]
    trace = copy / attempt["instance"] / "anytime_trace.jsonl"
    trace.write_bytes(trace.read_bytes() + b"\n")
    result = audit_attempt(attempt, copy)
    assert not result["passed"]
    assert any("hash mismatch" in error for error in result["errors"])
