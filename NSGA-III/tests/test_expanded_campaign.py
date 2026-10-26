"""Expanded comparison bookkeeping; all fixtures are offline."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_pilot import commands
from tools import run_expanded_pilot as runner
from tools.run_expanded_pilot import jobs, retry_allowed
from tools import summarize_expanded_pilot as expanded


def test_frozen_profile_preserves_original_six_and_pins_both_agent_models():
    original = json.loads((ROOT / "profiles/pilot.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "profiles/expanded_pilot.json").read_text(encoding="utf-8"))
    assert len(profile["instances"]) == 9
    for key in ("solver", "repair", "controller", "llm"):
        assert profile[key] == original[key]
    for name, settings in original["instances"].items():
        assert profile["instances"][name] == settings
    for method, multiplier in (("single", 1), ("multi", 2)):
        command = commands(profile, pipeline=method, seed=42, instances=["pu-cs-fal07"],
                           model=None, output=None, confirm_free_tier=True)[0]
        assert command[command.index("--model") + 1] == "gemini-3.5-flash-lite"
        assert int(command[command.index("--max-api-calls") + 1]) == 12 * multiplier


def test_plan_has_54_unique_cells_and_completes_seed17_before42():
    profile = json.loads((ROOT / "profiles/expanded_pilot.json").read_text(encoding="utf-8"))
    plan = jobs(profile)
    assert len(plan) == len(set(plan)) == 54
    assert all(seed == 17 for seed, _, _ in plan[:27])
    assert all(seed == 42 for seed, _, _ in plan[27:])
    assert {name for _, name, _ in plan[:9]} == {"pu-cs-fal07", "lums-spr18", "tg-spr18"}


def test_tg_followup_changes_only_wall_limit_and_has_six_jobs():
    original = json.loads((ROOT / "profiles/expanded_pilot.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "profiles/tg_no_timecap.json").read_text(encoding="utf-8"))
    for key in ("solver", "repair", "controller", "llm"):
        assert profile[key] == original[key]
    assert profile["instances"] == {"tg-spr18": {
        **original["instances"]["tg-spr18"], "time_limit_seconds": None}}
    assert jobs(profile) == [(seed, "tg-spr18", method)
                             for seed in (17, 42) for method in runner.METHODS]
    for method in runner.METHODS:
        command = commands(profile, pipeline=method, seed=17, instances=["tg-spr18"],
                           model=None, output=None, confirm_free_tier=True)[0]
        assert "--time-limit-seconds" not in command
        assert "--stop-on-feasible" not in command
        assert command[command.index("--evaluations") + 1] == "1200"
        assert command[command.index("--generations") + 1] == "200"
        if method != "baseline":
            assert command[command.index("--api-timeout-seconds") + 1] == "15"


def test_followup_output_is_separate_and_still_fits_windows():
    profile = json.loads((ROOT / "profiles/tg_no_timecap.json").read_text(encoding="utf-8"))
    assert runner.check_output_paths(profile, "t") <= 255
    assert runner.compact_output("multi", 17, 1, "t") != runner.compact_output("multi", 17, 1)
    with pytest.raises(ValueError, match="Output tag"):
        runner.compact_output("multi", 17, 1, "../x")


@pytest.mark.parametrize("failure", ["api_quota_exhausted", "api_billing_required",
                                     "api_configuration_error", "unknown", "api_input_limit"])
def test_campaign_does_not_auto_retry_non_transient_failures(failure):
    assert not retry_allowed([{"api_failures": {failure: 1}}])


def test_whole_run_retry_is_bounded_and_not_score_based():
    assert retry_allowed([])
    attempt = {"api_failures": {"api_service_error": 1}, "hard": 0, "internal_total": 1}
    assert retry_allowed([attempt])
    assert not retry_allowed([attempt, deepcopy(attempt)])
    assert not retry_allowed([{"api_failures": {}}])


def test_multi_summary_separates_censoring_and_infeasible_soft_ties(monkeypatch):
    methods = {method: {"hard": 2, "internal_total": 100,
        "feasible_internal": False, "evaluation_cap_reached": method != "multi",
        **{key: 0 for key in ("runtime_seconds", "evaluations", "reviews", "api_calls",
          "api_latency_seconds", "input_tokens", "output_tokens", "thinking_tokens",
          "total_native_tokens", "interventions", "selected_interventions", "fallback_no_op")}}
        for method in expanded.METHODS}
    methods["multi"]["internal_total"] = 1
    monkeypatch.setattr(expanded, "audit_attempts", lambda *args: [])
    monkeypatch.setattr(expanded, "discover", lambda p, n, s, m: Path(m))
    monkeypatch.setattr(expanded, "completed", lambda *args: True)
    monkeypatch.setattr(expanded, "behavior", lambda *args: {})
    monkeypatch.setattr(expanded, "compare", lambda *args: {
        "instances": [{"methods": deepcopy(methods)}], "runs": {}})
    report = expanded.summarize({"instances": {"X": {}}, "llm": {"model": "fake"}}, [17])
    assert report["completed_triples"] == 1
    assert report["evaluation_matched_triples"] == 0
    assert report["quality_counts_evaluation_matched_only"]["single_vs_baseline"] == {"infeasible_hard_tie": 1}
    assert report["quality_counts_evaluation_matched_only"]["multi_vs_baseline"] == {}
    assert report["instances"][0]["pairwise"]["multi_vs_single"]["quality_outcome"] == "infeasible_hard_tie"


def test_compact_layout_fits_windows_limit_for_all_campaign_cells():
    profile = json.loads((ROOT / "profiles/expanded_pilot.json").read_text(encoding="utf-8"))
    assert runner.check_output_paths(profile) <= 255
    path = runner.compact_output("multi", 17, 2)
    assert path.is_relative_to(ROOT / "Multi-Agent Controller/results_itc2019")
    assert "expanded_campaign" not in path.parts


def test_path_guard_stops_an_overlong_workspace_before_api_calls(monkeypatch, tmp_path):
    profile = json.loads((ROOT / "profiles/expanded_pilot.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(runner, "ROOT", tmp_path / ("x" * 180))
    with pytest.raises(ValueError, match="before any API call"):
        runner.check_output_paths(profile)


def test_unpublished_failure_needs_explicit_approval_and_counts_as_an_attempt():
    failure = {"attempt": 1, "retry_authorized": False}
    assert not runner.continuation_allowed([], [failure])
    failure["retry_authorized"] = True
    assert runner.continuation_allowed([], [failure])
    assert not runner.continuation_allowed([{"api_failures": {"api_service_error": 1}}], [failure])
    assert not runner.continuation_allowed([], [failure, deepcopy(failure)])


def test_attempt_number_preserves_orphan_failure_log(tmp_path):
    log = tmp_path / "s17_muni-fi-spr16_multi_attempt1.log"
    log.write_text("retained export error", encoding="utf-8")
    number = runner.next_attempt_number(tmp_path, 17, "muni-fi-spr16", "multi", [], [{"attempt": 1}])
    assert number == 2
    assert log.read_text(encoding="utf-8") == "retained export error"


def recovery_fixture(tmp_path):
    attempts = [{"folder": f"attempt-{number}", "completion_status": "incomplete_api_failure",
                 "api_failures": {"api_service_error": 1}} for number in (1, 2)]
    approval = {"authorized": True, "seed": 42, "instance": "tg-spr18", "method": "single",
                "attempt": 3, "profile_sha256": "frozen-profile",
                "previous_attempt_folders": [a["folder"] for a in attempts]}
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    return attempts, approval


def test_human_recovery_is_cell_profile_and_attempt_specific(tmp_path):
    attempts, approval = recovery_fixture(tmp_path)
    allowed = lambda **changes: runner.human_recovery_allowed(**{
        "state_root": tmp_path, "seed": 42, "instance": "tg-spr18", "method": "single",
        "number": 3, "attempts": attempts, "unpublished": [],
        "profile_sha256": "frozen-profile", **changes})
    assert allowed()
    for changes in ({"seed": 17}, {"method": "multi"}, {"instance": "other"},
                    {"number": 2}, {"number": 4}, {"profile_sha256": "changed"},
                    {"attempts": attempts[:1]}, {"attempts": attempts + attempts[:1]},
                    {"unpublished": [{"attempt": 2}]}):
        assert not allowed(**changes)
    approval["authorized"] = False
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    assert not allowed()
    (tmp_path / "human_recoveries.json").unlink()
    assert not allowed()
    assert not retry_allowed(attempts)


@pytest.mark.parametrize("failure", ["api_quota_exhausted", "api_configuration_error",
                                     "api_billing_required", "unknown"])
def test_human_recovery_never_bypasses_non_transient_failures(tmp_path, failure):
    attempts, _ = recovery_fixture(tmp_path)
    attempts[-1]["api_failures"] = {failure: 1}
    assert not runner.human_recovery_allowed(tmp_path, 42, "tg-spr18", "single", 3,
                                             attempts, [], "frozen-profile")


def test_started_recovery_cannot_be_replayed_from_an_orphan_log(tmp_path):
    attempts, _ = recovery_fixture(tmp_path)
    (tmp_path / "s42_tg-spr18_single_attempt3.log").write_text("retained", encoding="utf-8")
    number = runner.next_attempt_number(tmp_path, 42, "tg-spr18", "single", attempts, [])
    assert number == 4
    assert not runner.human_recovery_allowed(tmp_path, 42, "tg-spr18", "single", number,
                                             attempts, [], "frozen-profile")


def test_fourth_attempt_requires_fresh_exact_approval_and_cannot_allow_fifth(tmp_path):
    attempts, third_approval = recovery_fixture(tmp_path)
    attempts.append({**deepcopy(attempts[-1]), "folder": "attempt-3"})
    allowed = lambda number=4, prior=attempts: runner.human_recovery_allowed(
        tmp_path, 42, "tg-spr18", "single", number, prior, [], "frozen-profile")
    assert not allowed()
    fourth_approval = {**deepcopy(third_approval), "attempt": 4,
                       "previous_attempt_folders": [a["folder"] for a in attempts]}
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [third_approval, fourth_approval]}), encoding="utf-8")
    assert allowed()
    assert not retry_allowed(attempts)
    assert not allowed(prior=[attempts[0], attempts[1], attempts[1]])
    bad = deepcopy(attempts)
    bad[-1]["api_failures"] = {"api_quota_exhausted": 1}
    assert not allowed(prior=bad)
    after = attempts + [{**deepcopy(attempts[-1]), "folder": "attempt-4"}]
    assert not allowed(5, after)
    (tmp_path / "s42_tg-spr18_single_attempt4.log").write_text("retained", encoding="utf-8")
    assert runner.next_attempt_number(tmp_path, 42, "tg-spr18", "single", attempts, []) == 5
    assert not allowed(5)


def stopped_recovery_fixture(tmp_path):
    attempts, approval = recovery_fixture(tmp_path)
    attempts += [{**deepcopy(attempts[-1]), "folder": f"attempt-{n}"} for n in (3, 4)]
    log = tmp_path / "s42_tg-spr18_single_attempt5.log"
    log.write_text("retained user-interrupted attempt", encoding="utf-8")
    stopped = {"seed": 42, "instance": "tg-spr18", "method": "single", "attempt": 5,
               "status": "user_interrupted_unpublished_attempt", "log": str(log),
               "comparison_eligible": False, "retry_authorized": False,
               "api_calls": None, "reported_native_tokens": None}
    approval.update(attempt=6, previous_attempt_folders=[a["folder"] for a in attempts],
                    previous_unpublished_attempts=[deepcopy(stopped)])
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    return attempts, [stopped], approval


def test_user_stopped_attempt_needs_exact_fresh_recovery_without_enabling_auto_retry(tmp_path):
    attempts, stopped, approval = stopped_recovery_fixture(tmp_path)
    allowed = lambda n=6: runner.human_recovery_allowed(
        tmp_path, 42, "tg-spr18", "single", n, attempts, stopped, "frozen-profile")
    assert allowed()
    assert not allowed(5) and not allowed(7)
    assert not runner.continuation_allowed(attempts, stopped)
    assert not runner.retry_allowed(attempts)
    assert stopped[0]["api_calls"] is None
    assert stopped[0]["reported_native_tokens"] is None
    approval.pop("previous_unpublished_attempts")
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    assert not allowed()


@pytest.mark.parametrize("status", ["unpublished_execution_failure", "storage_failure", "unknown"])
def test_user_stop_recovery_never_bypasses_unclassified_unpublished_failure(tmp_path, status):
    attempts, stopped, approval = stopped_recovery_fixture(tmp_path)
    stopped[0]["status"] = status
    approval["previous_unpublished_attempts"] = deepcopy(stopped)
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    assert not runner.human_recovery_allowed(tmp_path, 42, "tg-spr18", "single", 6,
                                             attempts, stopped, "frozen-profile")


def test_user_stop_recovery_binds_ledger_and_requires_retained_exact_log(tmp_path):
    attempts, stopped, approval = stopped_recovery_fixture(tmp_path)
    allowed = lambda: runner.human_recovery_allowed(
        tmp_path, 42, "tg-spr18", "single", 6, attempts, stopped, "frozen-profile")
    original = deepcopy(stopped[0])
    for key, value in (("attempt", 4), ("attempt", True), ("seed", 17),
                       ("method", "multi"), ("retry_authorized", True),
                       ("comparison_eligible", True), ("log", "")):
        stopped[0] = {**deepcopy(original), key: value}
        approval["previous_unpublished_attempts"] = deepcopy(stopped)
        (tmp_path / "human_recoveries.json").write_text(
            json.dumps({"recoveries": [approval]}), encoding="utf-8")
        assert not allowed()
    stopped[0] = deepcopy(original)
    approval["previous_unpublished_attempts"] = deepcopy(stopped)
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    assert allowed()
    stopped[0]["api_calls"] = 0  # An edited ledger is not covered by the original approval.
    assert not allowed()
    stopped[0] = deepcopy(original)
    Path(original["log"]).unlink()
    assert not allowed()


def test_user_stop_recovery_keeps_quota_auth_billing_and_orphan_guards(tmp_path):
    attempts, stopped, _ = stopped_recovery_fixture(tmp_path)
    for failure in ("api_quota_exhausted", "api_configuration_error", "api_billing_required", "unknown"):
        bad = deepcopy(attempts)
        bad[-1]["api_failures"] = {failure: 1}
        assert not runner.human_recovery_allowed(tmp_path, 42, "tg-spr18", "single", 6,
                                                 bad, stopped, "frozen-profile")
    (tmp_path / "s42_tg-spr18_single_attempt6.log").write_text("orphan", encoding="utf-8")
    number = runner.next_attempt_number(tmp_path, 42, "tg-spr18", "single", attempts, stopped)
    assert number == 7
    assert not runner.human_recovery_allowed(tmp_path, 42, "tg-spr18", "single", number,
                                             attempts, stopped, "frozen-profile")


def test_report_keeps_unpublished_unknown_usage_out_of_completed_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(expanded, "ROOT", tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"instances": {"X": {}}}), encoding="utf-8")
    failures = tmp_path / "solver_checks/expanded_pilot/storage_failures.json"
    failures.parent.mkdir(parents=True)
    failures.write_text(json.dumps({"failures": [{"instance": "X", "seed": 17,
        "method": "multi", "api_calls": None, "reported_native_tokens": None}]}), encoding="utf-8")
    monkeypatch.setattr(expanded, "summarize", lambda *args: {
        "completed_cells": 3, "all_attempt_totals": {"api_calls": 5, "reported_native_tokens": 100},
        "interpretation": []})
    report = expanded.save_report(profile, [17], tmp_path / "report.json")
    assert report["completed_cells"] == 3
    assert report["all_attempt_totals"]["api_calls"] == 5
    assert report["all_attempt_totals"]["unpublished_runs"] == 1
    assert report["all_attempt_totals"]["usage_has_unknown_unpublished_component"] is True
    assert report["unpublished_attempts"][0]["api_calls"] is None


def test_report_retains_profile_bound_human_recovery(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"instances": {"tg-spr18": {}}}), encoding="utf-8")
    _, approval = recovery_fixture(tmp_path)
    approval["profile_sha256"] = expanded.digest(profile)
    (tmp_path / "human_recoveries.json").write_text(
        json.dumps({"recoveries": [approval]}), encoding="utf-8")
    monkeypatch.setattr(expanded, "summarize", lambda *args: {
        "all_attempt_totals": {}, "interpretation": []})
    report = expanded.save_report(profile, [42], tmp_path / "report.json", state_root=tmp_path)
    assert report["human_authorized_recoveries"] == [approval]
    assert any("specific additional attempt" in note for note in report["interpretation"])
    report = expanded.save_report(profile, [17], tmp_path / "report.json", state_root=tmp_path)
    assert report["human_authorized_recoveries"] == []


def test_plot_followup_replaces_only_requested_seed_without_mutating_sources():
    from tools.plot_expanded_pilot import apply_followup, seed_summary

    old = {"seed": 17, "instance": "tg-spr18", "evaluation_matched_triple": False,
           "methods": {method: {"hard": 28, "internal_total": 72126, "evaluations": 398,
                                 "evaluation_cap_reached": False, "api_failures": {}}
                       for method in runner.METHODS}}
    source = {"instances": [old, {**deepcopy(old), "seed": 42}]}
    replacement = {"seed": 17, "instance": "tg-spr18", "evaluation_matched_triple": True,
        "methods": {method: {"hard": hard, "internal_total": soft, "evaluations": 1200,
                              "evaluation_cap_reached": True, "api_failures": {}}
                    for method, hard, soft in zip(runner.METHODS, (17, 16, 15), (74300, 74070, 73870))}}
    before = deepcopy(source)
    followup = {"instances": [replacement]}
    view, replacements = apply_followup(source, followup, [17])
    assert source == before
    assert "display_source" not in replacement
    assert view["instances"][1] == before["instances"][1]
    assert view["instances"][0]["methods"]["multi"]["hard"] == 15
    assert replacements[0]["seed"] == 17
    assert seed_summary(view["instances"][:1])["quality_counts"]["multi_vs_single"]["better"] == 1
    with pytest.raises(ValueError, match="No complete"):
        apply_followup(source, followup, [42])
    replacement["evaluation_matched_triple"] = False
    with pytest.raises(ValueError, match="evaluation-matched"):
        apply_followup(source, followup, [17])


def test_partial_plot_redraw_retains_followup_source_provenance():
    from tools.plot_expanded_pilot import retained_followup_sources

    seed17 = {"sha256": "report-17", "replacements": [
        {"seed": 17, "instance": "tg-spr18", "evaluations": 1200, "runs": {}}]}
    seed42 = {"sha256": "report-42", "replacements": [
        {"seed": 42, "instance": "tg-spr18", "evaluations": 1200, "runs": {}}]}
    previous = {"followup_source": seed17}

    merged = retained_followup_sources(previous, seed42)
    assert [source["sha256"] for source in merged] == ["report-17", "report-42"]
    assert previous == {"followup_source": seed17}

    extended = retained_followup_sources(
        {"followup_source": seed42, "followup_sources": merged},
        {"sha256": "report-42", "replacements": [
            {"seed": 43, "instance": "tg-spr18", "evaluations": 1200, "runs": {}}]},
    )
    assert len(extended) == 2
    assert [item["seed"] for item in extended[1]["replacements"]] == [42, 43]


def test_plot_summary_uses_soft_only_to_break_exact_hard_ties():
    from tools.plot_expanded_pilot import seed_summary

    row = {"evaluation_matched_triple": True, "methods": {
        method: {"hard": hard, "internal_total": soft}
        for method, hard, soft in zip(runner.METHODS, (2, 2, 3), (100, 90, 1))}}
    summary = seed_summary([row])
    assert summary["quality_counts"]["single_vs_baseline"] == {"better": 1, "worse": 0, "equal": 0}
    assert summary["quality_counts"]["multi_vs_single"] == {"better": 0, "worse": 1, "equal": 0}
    assert summary["method_totals_completed_endpoints"]["single"]["feasible_internal"] == 0
