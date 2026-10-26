"""Plot data tests cover historical traces, uncertainty and censored budgets."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.search_result_data import read_anytime, review_points, shared_ranges, campaign_costs


def point(generation=5,seconds=20,evaluations=100,hard=2,total=300):
    return {"generation":generation,"elapsed_seconds":seconds,"evaluations":evaluations,
            "best_observed_hard":hard,"best_observed_internal_total":total,
            "best_hard":0,"best_internal_total":1}


def test_anytime_keeps_real_start_endpoint_and_incumbent_instead_of_population(tmp_path):
    trace = tmp_path/"trace.jsonl"
    trace.write_text("\n".join(map(json.dumps,[point(),point(6,40,200,1,400)])),encoding="utf-8")
    data = read_anytime(trace)
    assert data[0]["seconds"] == 20 and data[-1]["seconds"] == 40
    assert [p["hard"] for p in data] == [2,1]
    assert data[-1]["internal_total"] == 400  # soft can rise when hard falls
    assert data[0]["weighted_objectives"] is None


def test_no_population_fallback_for_unrecorded_incumbent(tmp_path):
    trace = tmp_path/"trace.jsonl"
    trace.write_text(json.dumps({"generation":1,"elapsed_seconds":1,"evaluations":1,"best_hard":0}),encoding="utf-8")
    assert read_anytime(trace)[0]["hard"] is None


def test_usage_uses_native_total_once_and_pacing_is_not_double_counted():
    event = {"generation":5,"roles":[{"total_tokens":10,"thinking_tokens":4,"cached_input_tokens":3,
              "latency_seconds":5,"pacing_wait_seconds":2,"api_requests":1}],
             "action":{"operator_id":"move"},"outcome":{"status":"selected","selected":True,
              "source_hard":2,"source_total":300,"candidate_hard":1,"candidate_total":400},
             "incumbent_before":{"hard":1,"internal_total":200},"incumbent_after":{"hard":1,"internal_total":200}}
    checkpoint = {"generation":5,"seconds":30,"evaluations":110,"hard":1}
    data = review_points([event],[checkpoint])[0]
    assert data["reported_tokens"] == 10 and data["latency_seconds"] == 5
    assert data["source_improved"] and not data["incumbent_improved"]
    assert data["seconds"] == 30 and data["timing_resolution"] == "same_generation_post_review"


def test_missing_usage_is_lower_bound_and_absent_checkpoint_not_interpolated():
    events = [{"generation":g,"roles":[{"api_requests":1,"total_tokens":tokens}],"action":{}}
              for g,tokens in [(5,None),(6,20),(8,5)]]
    data = review_points(events,[{"generation":6,"seconds":40,"evaluations":100,"hard":3}])
    assert data[0]["seconds"] is None and data[2]["seconds"] is None
    assert data[1]["cumulative_reported_tokens"] == 20
    assert not data[1]["usage_complete"] and data[1]["unknown_usage_requests_cumulative"] == 1
    assert data[2]["cumulative_reported_tokens"] == 25


def test_new_review_timestamp_and_intentional_no_call_skip():
    event = {"generation":5,"review_completed_elapsed_seconds":19,"roles":[],"action":{"operator_id":"no_op"},
             "decision_status":"context_limit_skipped"}
    data = review_points([event],[])[0]
    assert data["seconds"] == 19 and data["usage_complete"] and data["api_calls"] == 0


def test_shared_ranges_keep_early_stop_visible_and_unknown_missing_trace():
    runs = {"a":{"anytime":[{"seconds":10,"evaluations":20},{"seconds":100,"evaluations":200}]},
            "b":{"anytime":[{"seconds":20,"evaluations":40},{"seconds":80,"evaluations":120}]}}
    assert shared_ranges(runs) == {"seconds":[20,80],"evaluations":[40,120]}
    runs["c"] = {"anytime":[]}
    assert shared_ranges(runs)["evaluations"] is None


def test_attempt_costs_preserve_failures_and_unpublished_unknown_usage():
    report = {"all_attempts":[{"seed":42,"instance":"tg","completion_status":"incomplete_api_failure","reported_native_tokens":0}],
              "unpublished_attempts":[{"seed":42,"instance":"tg","usage":None}],"human_authorized_recoveries":[]}
    costs = campaign_costs(report,42,"tg")
    assert len(costs["published_attempts"]) == 1 and costs["unpublished_attempts"][0]["usage"] is None
    assert "unknown" in costs["usage_note"]


def test_regressing_incumbent_and_backward_trace_time_fail_closed(tmp_path):
    trace = tmp_path/"trace.jsonl"
    for next_point in (point(6,21,101,3,300),point(6,19,101,1,300)):
        trace.write_text("\n".join(map(json.dumps,[point(),next_point])),encoding="utf-8")
        with pytest.raises(ValueError):
            read_anytime(trace)
