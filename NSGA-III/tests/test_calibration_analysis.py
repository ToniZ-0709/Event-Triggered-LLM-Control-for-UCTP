"""Budget recommendations must not imply unmeasured search coverage."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from solver_checks.analyze_recalibration import suggest, trajectory


def record(hard=0,maximum=1200):
    trace = [{"generation":i+1,"evaluations":e,"elapsed_seconds":e/10,
              "best_observed_hard":h,"best_observed_internal_total":total}
             for i,(e,h,total) in enumerate(((100,8,1200),(400,hard,1000),
                                            (800,hard,900),(maximum,hard,900)))]
    return {"instance":"X","variant":"current","seed":17,
            "source_hash":"frozen","config":{"population":20,"partitions":3,
            "generations":2000,"seed":17,"evaluations":maximum},"repair":{"interval":5},
            "evaluations":maximum,"seconds":maximum/10,"stop_reason":"evaluation_cap",
            "trace":trace,"monitor":{},"diagnosis":{"required_groups":[]},
            "score":{"hard":hard,"total":900,"room_conflicts":hard,"room_unavailable":0,
                     "hard_distributions":0,"unassigned_requests":0}}


PROFILE = {"controller":{"max_reviews":12,"api_timeout_seconds":15}}


def test_profile_uses_measured_horizon_and_reserves_agent_time():
    row = record()
    result = suggest([row],PROFILE)
    assert result["instances"]["X"]["evaluations"] == 900
    assert result["instances"]["X"]["time_limit_seconds"] > 360
    assert not result["provenance"]["recalibration_required"]


def test_newly_feasible_or_censored_case_is_not_called_calibrated():
    row = record(maximum=800)
    result = suggest([row],PROFILE)
    assert result["instances"]["X"]["evaluations"] <= 800
    assert result["provenance"]["recalibration_required"]
    row["stop_reason"] = "time_cap"
    assert suggest([row],PROFILE)["provenance"]["budget_status"]["X"] == "time_censored"


def test_hard_plateau_is_reported_without_claiming_feasibility():
    row = record(hard=2)
    result = suggest([row],PROFILE)
    assert result["provenance"]["budget_status"]["X"] == "hard_plateau_observed"
    assert not result["evidence"]["X"]["all_feasible"]
    assert trajectory(row)["evaluations_since_hard_improvement"] == 800


def test_endpoint_censoring_does_not_erase_a_fully_measured_earlier_budget():
    row = record()
    row["stop_reason"] = "time_cap"
    result = suggest([row],PROFILE)
    assert result["instances"]["X"]["evaluations"] == 900
    assert result["provenance"]["budget_status"]["X"] == "measured_local_knee"
    assert result["provenance"]["endpoint_time_censoring"]["X"] == [17]


def test_profile_rejects_mixed_solver_versions():
    row = record()
    other = deepcopy(row)
    other["source_hash"] = "changed"
    with pytest.raises(ValueError,match="source_hash"):
        suggest([row,other],PROFILE)


def test_runner_applies_instance_overrides_once_for_every_pipeline():
    from run_pilot import commands
    profile = {"schema_version":1,"solver":{"population":20,"partitions":3},
               "repair":{"near_feasible_hard":12},"controller":{"max_reviews":4},
               "instances":{"X":{"population":12,"partitions":2,"evaluations":1000,
                                  "repair_near_feasible_hard":40}}}
    for method in ("baseline","single","multi"):
        command = commands(profile,pipeline=method,seed=17,instances=["X"],model="fake",output=None)[0]
        assert command[command.index("--population")+1] == "12"
        assert command[command.index("--partitions")+1] == "2"
        assert command.count("--repair-near-feasible-hard") == 1
        assert command[command.index("--repair-near-feasible-hard")+1] == "40"


def test_profile_allows_explicit_instance_settings_but_not_seed_drift():
    first = record()
    second = deepcopy(first)
    second["instance"] = "Y"
    second["config"]["population"] = 12
    second["config"]["partitions"] = 2
    profile = {**PROFILE,"solver":{"population":20,"partitions":3},"repair":{"interval":5}}
    result = suggest([first,second],profile)
    assert result["instances"]["Y"]["population"] == 12
    third = deepcopy(second)
    third["seed"] = 42
    third["config"]["seed"] = 42
    third["config"]["population"] = 16
    with pytest.raises(ValueError,match="within instance"):
        suggest([first,second,third],profile)
