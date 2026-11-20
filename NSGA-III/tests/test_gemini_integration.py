"""Gemini integration tests; no real API requests or credentials."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

genai=pytest.importorskip("google.genai",reason="Install controller requirements for Gemini SDK tests")
httpx=pytest.importorskip("httpx")
from google.genai import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"NSGA-III"))
from itc2019 import llm
from itc2019.controller import ControllerCallback,ControllerSettings
from run_pilot import commands
from compare_pipelines import compare

NO_OP={"solution_id":None,"region_id":None,"boundary_id":None,"operator_id":"no_op",
       "execution_budget":0,"next_review_interval":10}


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","offline-test-key-never-send")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    calls=[]
    response=NS(response_id="fake-id",model_version="gemini-4-hydrogen-v1",
                text=json.dumps(NO_OP),candidates=[NS(finish_reason="STOP")],prompt_feedback=None,
                usage_metadata=NS(prompt_token_count=20,candidates_token_count=30,
                                  thoughts_token_count=12,total_token_count=62,cached_content_token_count=0))
    def generate(**kwargs):
        calls.append(kwargs)
        return response
    fake=NS(models=NS(generate_content=generate),close=lambda:None)
    monkeypatch.setattr(genai,"Client",lambda **kw:fake)
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,free_tier_confirmed=True,
                                  min_request_interval_seconds=0)
    return client,response,calls


def test_key_and_billing_confirmation_are_required_before_any_sdk_call(monkeypatch):
    monkeypatch.delenv("GEMINI_FREE_TIER_CONFIRMED",raising=False)
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    with pytest.raises(RuntimeError,match="Free Tier"):
        llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15)
    with pytest.raises(RuntimeError,match="GEMINI_API_KEY"):
        llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,free_tier_confirmed=True)
    with pytest.raises(ValueError,match="permits"):
        llm.GeminiDecisionClient("gemini-4-argon",60,free_tier_confirmed=True)
    with pytest.raises(ValueError,match="permits"):
        llm.GeminiDecisionClient("gemini-3.6-flash",60,free_tier_confirmed=True)
    with pytest.raises(ValueError,match="permits"):
        llm.GeminiDecisionClient("gemini-3.8-flash",60,free_tier_confirmed=True)
    with pytest.raises(ValueError,match="thinking_level"):
        llm.GeminiDecisionClient(llm.DEFAULT_MODEL,60,free_tier_confirmed=True,
                                 thinking_level="turbo")


def test_allowlist_defaults_and_configurable_thinking_level(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","offline-test-key-never-send")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    calls=[]
    response=NS(response_id="fake-id",model_version="gemini-4-hydrogen-v1",
                text=json.dumps(NO_OP),candidates=[NS(finish_reason="STOP")],prompt_feedback=None,
                usage_metadata=NS(prompt_token_count=10,candidates_token_count=10,
                                  thoughts_token_count=0,total_token_count=20,cached_content_token_count=0))
    fake=NS(models=NS(generate_content=lambda **kw:(calls.append(kw),response)[1]),close=lambda:None)
    monkeypatch.setattr(genai,"Client",lambda **kw:fake)
    assert llm.DEFAULT_MODEL=="gemini-4-hydrogen"
    assert llm.FREE_TIER_MODELS==("gemini-4-hydrogen","gemini-3.5-flash-lite")
    assert llm.DEFAULT_TIMEOUT_SECONDS==60.0
    for model_name in llm.FREE_TIER_MODELS:
        client=llm.GeminiDecisionClient(model_name,free_tier_confirmed=True,
                                        min_request_interval_seconds=0,thinking_level="medium")
        assert client.timeout_seconds==60.0
        assert client.metadata()["thinking_level"]=="medium"
        assert client.metadata()["sdk_attempts_per_call"]==3
        decision,record=client.request("single_agent",{"candidate_regions":[]})
        assert decision==NO_OP and record["status"]=="ok"
        assert calls[-1]["config"].thinking_config.thinking_level==types.ThinkingLevel.MEDIUM


def test_valid_json_usage_and_bounded_native_config(adapter):
    client,response,calls=adapter
    decision,record=client.request("single_agent",{"candidate_regions":[]},12)
    assert decision==NO_OP and record["status"]=="ok"
    assert record["api_requests"]==1 and record["thinking_tokens"]==12
    assert record["total_tokens"]==62 and record["provider"]=="gemini"
    config=calls[0]["config"]
    assert config.response_json_schema==llm.SCHEMAS["single_agent"]
    assert config.http_options.timeout<=12000
    assert config.http_options.retry_options.attempts==3
    assert 429 not in config.http_options.retry_options.http_status_codes
    assert {500,503,504}<=set(config.http_options.retry_options.http_status_codes)
    assert config.max_output_tokens==4096 and config.candidate_count==1
    assert config.automatic_continuation is False and config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level==types.ThinkingLevel.LOW
    assert config.tools is None


@pytest.mark.parametrize("text,status",[
    ('{"execution_budget":NaN}',"invalid_json"),
    ('{"x":1,"x":2}',"invalid_json"),
    ('not JSON',"invalid_json"),
    ('[]',"invalid_schema"),
    (json.dumps({**NO_OP,"execution_budget":True}),"invalid_schema"),
    (json.dumps({**NO_OP,"extra":"not allowed"}),"invalid_schema"),
    (json.dumps({**NO_OP,"next_review_interval":"10"}),"invalid_schema"),
    (json.dumps({**NO_OP,"operator_id":"run_python"}),"invalid_schema"),
])
def test_malformed_responses_never_reach_solver(adapter,text,status):
    client,response,_=adapter
    response.text=text
    decision,record=client.request("single_agent",{})
    assert decision is None and record["status"]==status


def test_truncated_response_is_not_success_even_if_partial_text_parses(adapter):
    client,response,_=adapter
    response.candidates[0].finish_reason="MAX_TOKENS"
    decision,record=client.request("single_agent",{})
    assert decision is None and record["status"]=="incomplete_response"


def test_input_limit_and_expired_deadline_make_zero_requests(adapter):
    client,_,calls=adapter
    client.max_input_characters=1
    assert client.request("single_agent",{})[1]["status"]=="api_input_limit"
    assert calls==[]
    client.max_input_characters=60000
    assert client.request("single_agent",{},0)[1]["status"]=="api_deadline_skipped"
    assert calls==[]


def test_pacing_respects_remaining_deadline_and_records_wait(adapter,monkeypatch):
    client,_,calls=adapter
    clock=[100.0]
    monkeypatch.setattr(llm.time,"perf_counter",lambda:clock[0])
    monkeypatch.setattr(llm.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    client.min_request_interval_seconds=10
    client._last_request_started=95
    assert client.request("single_agent",{},3)[1]["status"]=="api_deadline_skipped"
    assert calls==[] and clock==[100]
    _,record=client.request("single_agent",{},8)
    assert record["pacing_wait_seconds"]==5 and calls[0]["config"].http_options.timeout==3000


@pytest.mark.parametrize("code,status",[(429,"api_quota_exhausted"),(402,"api_billing_required"),
                                      (401,"api_configuration_error"),(503,"api_service_error")])
def test_api_failures_are_explicit_no_retry_no_provider_fallback(adapter,code,status):
    client,_,calls=adapter
    class Failure(Exception):
        pass
    def fail(**kwargs):
        calls.append(kwargs)
        exc=Failure("offline-test-key-never-send failed")
        exc.code=code
        raise exc
    client.client.models.generate_content=fail
    decision,record=client.request("single_agent",{})
    assert decision is None and record["status"]==status and len(calls)==1
    assert "offline-test-key-never-send" not in json.dumps(record)
    if code in (429,402,401):
        assert client.request("single_agent",{})[1]["api_requests"]==0
        assert len(calls)==1


def test_timeout_is_recorded(adapter):
    client,_,_=adapter
    def fail(**kwargs):
        raise httpx.ReadTimeout("slow")
    client.client.models.generate_content=fail
    assert client.request("single_agent",{})[1]["status"]=="api_timeout"


@pytest.mark.parametrize("finish_reason",["STOP","CONTINUATION"])
def test_native_sdk_serializes_schema_without_network(monkeypatch,finish_reason):
    monkeypatch.setenv("GEMINI_API_KEY","offline-placeholder")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.host=="generativelanguage.googleapis.com"
        return httpx.Response(200,json={
            "candidates":[{"content":{"role":"model","parts":[{"text":json.dumps(NO_OP)}]},
                           "finishReason":finish_reason}],"modelVersion":"gemini-4-hydrogen-v1",
            "continuationToken":"must-not-be-followed",
            "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":30,"totalTokenCount":50}})
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,free_tier_confirmed=True)
    client.close()
    client.client=genai.Client(api_key="offline-placeholder",vertexai=False,
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                      retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        decision,record=client.request("single_agent",{})
        if finish_reason=="STOP":
            assert decision==NO_OP and record["status"]=="ok"
        else:
            assert decision is None and record["status"]=="incomplete_response"
        assert len(seen)==1 and seen[0]["generationConfig"]["responseMimeType"]=="application/json"
        assert seen[0]["generationConfig"]["responseJsonSchema"]==llm.SCHEMAS["single_agent"]
    finally:
        client.close()


def test_native_sdk_does_not_retry_429(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","offline-placeholder")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(429,json={"error":{"code":429,"status":"RESOURCE_EXHAUSTED",
                                                 "message":"fake test quota"}})
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,60,free_tier_confirmed=True)
    client.close()
    client.client=genai.Client(api_key="offline-placeholder",vertexai=False,
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                      retry_options=client.retry_options))
    try:
        decision,record=client.request("single_agent",{})
        assert decision is None and record["status"]=="api_quota_exhausted"
        assert len(requests)==1
    finally:
        client.close()


def test_native_sdk_retries_transient_504_up_to_3_attempts(monkeypatch):
    import tenacity.nap
    monkeypatch.setenv("GEMINI_API_KEY","offline-placeholder")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    monkeypatch.setattr(tenacity.nap.time,"sleep",lambda _s:None)
    requests=[]
    def handler(request):
        requests.append(request)
        if len(requests)<3:
            return httpx.Response(504,json={"error":{"code":504,"status":"DEADLINE_EXCEEDED",
                                                     "message":"transient test timeout"}})
        return httpx.Response(200,json={
            "candidates":[{"content":{"role":"model","parts":[{"text":json.dumps(NO_OP)}]},
                           "finishReason":"STOP"}],
            "modelVersion":"gemini-4-hydrogen-v1",
            "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":30,"totalTokenCount":50}})
    client=llm.GeminiDecisionClient("gemini-4-hydrogen",60,free_tier_confirmed=True)
    client.close()
    client.client=genai.Client(api_key="offline-placeholder",vertexai=False,
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                      retry_options=client.retry_options))
    try:
        decision,record=client.request("single_agent",{})
        assert len(requests)==3
        assert decision==NO_OP and record["status"]=="ok"
    finally:
        client.close()


def test_wrapper_applies_gemini_policy_only_to_controllers():
    profile=json.loads((ROOT/"profiles/pilot.json").read_text(encoding="utf-8"))
    for mode in ("baseline","single","multi","random","greedy"):
        command=commands(profile,pipeline=mode,seed=17,instances=["nbi-spr18"],model=None,
                         output=None,confirm_free_tier=True,thinking_level="high")[0]
        if mode in ("single","multi"):
            assert "--confirm-free-tier" in command
            assert command[command.index("--model")+1]==profile["llm"]["model"]
            assert command[command.index("--api-timeout-seconds")+1]=="60"
            assert command[command.index("--thinking-level")+1]=="high"
            assert command.count("--provider")==1
        else:
            assert "--confirm-free-tier" not in command
            assert "--thinking-level" not in command
            assert "--api-timeout-seconds" not in command


def test_accounting_counts_real_requests_not_deadline_skips():
    from itc2019.instance import Instance
    import time
    instance=Instance("fake",1,1,1,(1,1,1,1),{},(),{},(),(),{})
    callback=ControllerCallback(instance,ControllerSettings(model="fake"),None,17,time.perf_counter(),None)
    callback._account_calls([{"api_requests":0,"status":"api_deadline_skipped"},
        {"api_requests":1,"status":"api_quota_exhausted","thinking_tokens":5,"total_tokens":15}])
    report=callback.report()
    assert report["api_calls"]==1 and report["thinking_tokens"]==5
    assert report["completion_status"]=="incomplete_api_failure"
    assert report["comparison_eligible"] is False


def make_runs(tmp_path,incomplete=False):
    shared={"instances":[{"file":"X.xml","sha256":"data"}],"config":{"seed":17},
            "repair":{},"time_limit_seconds":90,"stop_on_feasible":False,
            "solver_sources_sha256":{"core":"same"},"environment":{
                "python":"same","numpy":"same","pymoo":"same","ortools":"same"}}
    folders={}
    for mode in ("baseline","single","multi"):
        folder=tmp_path/mode
        folder.mkdir()
        settings=deepcopy(shared)
        row={"instance":"X","feasible":False,"score":{"hard":3,"total":100,"weighted":[1,2,3,4]},
             "runtime_seconds":12,"evaluations":120,"validation":{"official_status":"not_run"}}
        if mode!="baseline":
            settings.update(pipeline={"name":"NSGA-III + Single-Agent" if mode=="single" else
                                     "NSGA-III + Inspector-Planner"},
                controller={"model":llm.DEFAULT_MODEL,"provider":"gemini","max_reviews":1,
                            "max_api_calls":1 if mode=="single" else 2},
                controller_sources_sha256={"NSGA-III/itc2019/llm.py":"same"},llm={"provider":"gemini"})
            settings["environment"]["google-genai"]="same"
            row["controller"]={"comparison_eligible":not(incomplete and mode=="multi"),
                               "completion_status":"incomplete_api_failure" if incomplete else "completed"}
        (folder/"run_config.json").write_text(json.dumps({"settings":settings}))
        (folder/"summary.json").write_text(json.dumps([row]))
        folders[mode]=folder
    return folders


def test_comparison_rejects_infrastructure_failure(tmp_path):
    with pytest.raises(ValueError,match="incomplete API coverage"):
        compare(make_runs(tmp_path,incomplete=True))


def test_comparison_allows_infeasible_results_and_reports_hard_delta(tmp_path):
    result=compare(make_runs(tmp_path))
    assert result["provider"]=="gemini"
    assert result["instances"][0]["single_minus_baseline"]["hard_change"]==0
    assert result["instances"][0]["single_minus_baseline"]["internal_total_change_if_both_feasible"] is None


def test_comparison_accepts_local_policies_alongside_llm_and_rejects_different_review_caps(tmp_path):
    folders = make_runs(tmp_path)
    baseline = json.loads((folders["baseline"] / "run_config.json").read_text())
    summary = json.loads((folders["baseline"] / "summary.json").read_text())
    for method in ("random", "greedy"):
        folder = tmp_path / method
        folder.mkdir()
        folders[method] = folder
        config = deepcopy(baseline)
        config["completion_status"] = "completed"
        config["settings"].update(
            pipeline={"name": "NSGA-III + " + method.capitalize()},
            controller={"provider": "none", "model": None, "max_api_calls": 0,
                        "fixed_execution_budget": 10, "max_reviews": 1},
        )
        rows = deepcopy(summary)
        rows[0]["controller"] = {
            "mode": method, "completion_status": "completed", "comparison_eligible": True,
            "api_calls": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
            "total_native_tokens": 0, "api_latency_seconds": 0.0, "probe_evaluations": 7,
        }
        (folder / "run_config.json").write_text(json.dumps(config))
        (folder / "summary.json").write_text(json.dumps(rows))
    row, = compare(folders)["instances"]
    assert set(row["methods"]) == set(folders)
    assert row["methods"]["greedy"]["probe_evaluations"] == 7
    assert row["greedy_minus_random"]["evaluations_change"] == 0
    config["settings"]["controller"]["max_reviews"] = 2
    (folders["greedy"] / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Shared review settings differ"):
        compare(folders)


def test_quota_failure_ends_solver_early_and_preserves_incomplete_report():
    import time
    from itc2019.instance import Instance,Class,Room,Time
    from itc2019.nsga3 import Config,run
    slots=(Time("1","1",0,1),Time("1","1",1,1))
    classes=tuple(Class(str(i),str(i),"1","1",None,None,slots,
                        (("1",0),("2",0)),False) for i in range(2))
    instance=Instance("quota-fixture",1,1,4,(1,1,1,1),
        {str(i):Room(str(i),None,(),{}) for i in (1,2)},classes,{},(),(),
        {c.id:i for i,c in enumerate(classes)})
    class QuotaController(ControllerCallback):
        def decide(self,packet):
            return None,[{"status":"api_quota_exhausted","api_requests":1}],"api_quota_exhausted"
    controller=QuotaController(instance,ControllerSettings(model="fake",warmup_generations=2,
        review_interval=2,min_review_interval=1),None,17,time.perf_counter(),None)
    _,_,result=run(instance,Config(population=4,partitions=1,generations=20,seed=17,
        initial_repair_steps=0,offspring_repair_steps=0),callback=controller)
    assert len(controller.anytime_events)<20 and controller.report()["api_calls"]==1
    assert controller.report()["completion_status"]=="incomplete_api_failure"
    assert controller.report()["model_no_op"]==0 and controller.report()["fallback_no_op"]==1


def test_comparison_rejects_stale_official_receipt(tmp_path):
    folders=make_runs(tmp_path)
    folder=folders["baseline"]/"X"
    folder.mkdir()
    (folder/"solution.xml").write_text('<solution name="X"/>')
    (folder/"official_validation.json").write_text(json.dumps({
        "instance":"X","solution_sha256":"old-hash","official_status":"passed"}))
    with pytest.raises(ValueError,match="does not match XML"):
        compare(folders)


def test_state_packet_constraint_details_and_payload_truncation():
    import time
    from itc2019.controller import ActionPlan
    from itc2019.instance import Class, Instance, Room, Time
    from itc2019.nsga3 import Config, run
    from itc2019.regions import ConstraintObservation, RegionBuilder

    slots = (Time("1", "1", 0, 1), Time("1", "1", 1, 1))
    classes = tuple(
        Class(f"C{i}", f"C{i}", "1", "1", None, None, slots, (("1", 0), ("2", 0)), False)
        for i in range(4)
    )
    instance = Instance(
        "packet-test", 1, 1, 4, (1, 1, 1, 1),
        {"1": Room("1", None, (), {}), "2": Room("2", None, (), {})},
        classes, {}, (), (), {c.id: i for i, c in enumerate(classes)},
    )
    builder = RegionBuilder(instance)
    observations = {
        "distribution:0": ConstraintObservation(
            "NotOverlap", False, 5, 0, (0, 1), violation_groups={},
        ),
        "distribution:1": ConstraintObservation(
            "DifferentTime", False, 2, 2, (0, 1, 2),
            violation_groups={(1, 2): 1, (0, 1): 1},
        ),
        "distribution:2": ConstraintObservation(
            "SameStart", True, 0, 1, (0, 1), violation_groups={(0, 1): 1},
        ),
    }
    details = builder._constraint_details(
        ("distribution:0", "distribution:1", "distribution:2"),
        {0, 1}, observations, {},
    )
    assert [item["constraint_id"] for item in details] == ["distribution:2", "distribution:1"]
    assert details[1]["violation_groups"][0]["outside_region_class_ids"] == ()
    assert details[1]["violation_groups"][1]["outside_region_class_ids"] == ("C2",)

    region_payload = {
        "solution_id": "0", "region_id": "0:0", "boundary_id": "b0",
        "constraint_details": [
            {"constraint_id": "soft_cross", "required": False, "cross_boundary": True,
             "current_violations": 4, "region_violation_units": 4},
            {"constraint_id": "hard_cross", "required": True, "cross_boundary": True,
             "current_violations": 2, "region_violation_units": 2},
            {"constraint_id": "hard_in", "required": True, "cross_boundary": False,
             "current_violations": 1, "region_violation_units": 1},
            {"constraint_id": "diagnosed_soft", "required": False, "cross_boundary": True,
             "current_violations": 1, "region_violation_units": 1},
        ],
    }
    payload = {
        "state": {"candidate_regions": [region_payload]},
        "diagnosis": {
            "solution_id": "0", "region_id": "0:0", "boundary_id": "b0",
            "relevant_constraint_ids": ["diagnosed_soft"],
        },
    }
    one_detail = {
        "state": {"candidate_regions": [{
            "solution_id": "0", "region_id": "0:0", "boundary_id": "b0",
            "constraint_details": [
                region_payload["constraint_details"][3],
                region_payload["constraint_details"][2],
            ],
            "constraint_details_omitted": 2,
        }]},
        "diagnosis": payload["diagnosis"],
    }
    max_chars = len(json.dumps(one_detail, separators=(",", ":")))
    encoded, omitted = llm._encode_payload(payload, max_chars)
    assert omitted == 2
    kept = json.loads(encoded)["state"]["candidate_regions"][0]["constraint_details"]
    assert [item["constraint_id"] for item in kept] == ["diagnosed_soft", "hard_in"]

    class OutcomeController(ControllerCallback):
        def decide(self, packet):
            region = packet["candidate_regions"][0]
            plan = ActionPlan(
                region["solution_id"], region["region_id"], region["boundary_id"],
                "move", 1, 2,
            )
            return plan, [{"role": "single_agent", "status": "ok", "api_requests": 1}], "decision_received"

    callback = OutcomeController(
        instance, ControllerSettings(model="fake", warmup_generations=2,
                                     review_interval=2, min_review_interval=1),
        None, 17, time.perf_counter(), None,
    )
    run(instance, Config(population=4, partitions=1, generations=3, seed=17,
                         initial_repair_steps=0, offspring_repair_steps=0), callback=callback)
    outcome_row = callback.outcomes[0]
    for key in ("source_hard", "source_total", "best_trial_hard", "best_trial_total"):
        assert key in outcome_row
    assert isinstance(outcome_row["source_hard"], int)
    assert isinstance(outcome_row["source_total"], int)


def test_state_packet_omits_redundant_fields_and_multi_agent_uses_constraint_details():
    import importlib.util
    import time
    from itc2019.instance import Class, Instance, Room, Time
    from itc2019.nsga3 import Config, run
    from itc2019.regions import ConstraintObservation, RegionBuilder

    slots = (Time("1", "1", 0, 1), Time("1", "1", 1, 1))
    classes = (
        Class("C0", "C0", "1", "1", None, None, slots, (), True),
        Class("C1", "C1", "1", "1", None, None, slots, (("1", 0), ("2", 1)), False),
        Class("C2", "C2", "1", "1", None, None, slots, (("1", 0), ("2", 1)), False),
    )
    instance = Instance(
        "clean-packet", 1, 1, 4, (1, 1, 1, 1),
        {"1": Room("1", None, (), {}), "2": Room("2", None, (), {})},
        classes, {}, (), (), {c.id: i for i, c in enumerate(classes)},
    )
    builder = RegionBuilder(instance)
    observations = {
        "distribution:0": ConstraintObservation(
            "SameStart", True, 0, 1, (0, 1), violation_groups={(0, 1): 1},
        ),
        "distribution:1": ConstraintObservation(
            "DifferentTime", False, 2, 3, (0, 1, 2), params=(1,),
            violation_groups={(1, 2): 1, (0, 1): 1},
        ),
    }
    details = builder._constraint_details(("distribution:0", "distribution:1"), {0, 1}, observations, {})
    assert "penalty" not in details[0]
    assert "params" not in details[0]
    assert "current_violations" not in details[0]
    assert details[1]["penalty"] == 2
    assert details[1]["params"] == (1,)
    assert details[1]["current_violations"] == 3
    group = details[1]["violation_groups"][1]
    assert "class_ids" not in group
    assert group["in_region_class_ids"] == ("C1",)
    assert group["outside_region_class_ids"] == ("C2",)

    # Verify _encode_payload works when current_violations is omitted.
    payload = {
        "state": {"candidate_regions": [{
            "solution_id": "0", "region_id": "0:0", "boundary_id": "b0",
            "constraint_details": details,
        }]}
    }
    encoded, omitted = llm._encode_payload(payload, 320)
    assert omitted >= 1
    assert json.loads(encoded)["state"]["candidate_regions"][0]["constraint_details_omitted"] == omitted

    class PacketRecorder(ControllerCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.packets = []

        def decide(self, packet):
            self.packets.append(packet)
            plan = self._fallback()
            return plan, [{"role": "single_agent", "status": "ok", "api_requests": 1}], "decision_received"

    recorder = PacketRecorder(
        instance, ControllerSettings(model="fake", warmup_generations=1,
                                     review_interval=1, min_review_interval=1),
        None, 17, time.perf_counter(), None,
    )
    run(instance, Config(population=4, partitions=1, generations=3, evaluations=80, seed=17,
                         initial_repair_steps=0, offspring_repair_steps=0), callback=recorder)
    assert len(recorder.packets) >= 2
    packet = recorder.packets[1]
    for removed in ("schema_version", "recent_best_quality"):
        assert removed not in packet
    g = packet["global"]
    for removed in ("elapsed_seconds", "remaining_seconds", "evaluations",
                    "best_hard", "best_internal_total", "best_feasible_internal_total",
                    "best_feasible_official_score", "metric_source", "progress_mode"):
        assert removed not in g
    assert "evaluation_budget" in g and "remaining_evaluations" in g
    assert "operators" not in packet["allowed_actions"]
    assert packet["recent_outcomes"]
    no_op_outcome = packet["recent_outcomes"][0]
    for omitted_key in ("elapsed_seconds", "region_id", "boundary_id", "class_ids", "outcome",
                        "attempts", "source_hard", "source_total", "best_trial_hard",
                        "best_trial_total", "best_hard_change", "best_internal_total_change"):
        assert omitted_key not in no_op_outcome
    hard_neutral = recorder._public_outcome({**recorder.outcomes[0], "outcome": "selected",
                                             "source_hard": 3, "best_hard_change": 0})
    assert hard_neutral["best_hard_change"] == 0
    region = packet["candidate_regions"][0]
    for removed in ("constraint_ids", "class_summaries", "solution_hard", "solution_total",
                    "is_incumbent", "unlocalized_penalties", "movable_class_count",
                    "compatible_swap_pairs", "boundary_class_count", "boundary_truncated"):
        assert removed not in region
    sol = packet["solutions"][region["solution_id"]]
    assert isinstance(sol["hard"], int) and isinstance(sol["total"], int)
    c0_summary = sol["classes"]["C0"]
    assert "class_id" not in c0_summary
    assert c0_summary["room_options"] == 0
    assert "current_room_id" not in c0_summary and "current_room_penalty" not in c0_summary
    assert "current_room_id" in sol["classes"]["C1"] and "current_room_penalty" in sol["classes"]["C1"]

    spec = importlib.util.spec_from_file_location("multi_ctrl", ROOT / "Multi-Agent Controller/controller.py")
    assert spec is not None and spec.loader is not None
    multi_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_mod)

    class DiagnosisClient:
        def __init__(self, cid: str):
            self.cid = cid

        def request(self, role: str, req_payload: dict, remaining_seconds=None):
            if role == "inspector":
                reg = req_payload["candidate_regions"][0]
                return ({"solution_id": reg["solution_id"], "region_id": reg["region_id"],
                         "boundary_id": reg["boundary_id"], "relevant_constraint_ids": [self.cid]},
                        {"role": role, "status": "ok", "api_requests": 1, "latency_seconds": 0.0})
            return ({"operator_id": "no_op", "execution_budget": 0, "next_review_interval": 1},
                    {"role": role, "status": "ok", "api_requests": 1, "latency_seconds": 0.0})

    multi_packet = {
        "candidate_regions": [{
            "solution_id": "0", "region_id": "0:0", "boundary_id": "b0",
            "available_operators": ("move",), "constraint_details": list(details),
        }]
    }
    valid_multi = multi_mod.MultiAgentController(
        instance, ControllerSettings(model="fake", review_interval=1, min_review_interval=1),
        DiagnosisClient("distribution:0"), 17, time.perf_counter(), None,
    )
    _, roles_ok, status_ok = valid_multi.decide(multi_packet)
    assert status_ok == "decision_received" and len(roles_ok) == 2

    bad_multi = multi_mod.MultiAgentController(
        instance, ControllerSettings(model="fake", review_interval=1, min_review_interval=1),
        DiagnosisClient("not_in_constraint_details"), 17, time.perf_counter(), None,
    )
    _, roles_bad, status_bad = bad_multi.decide(multi_packet)
    assert status_bad == "invalid_inspector_constraints" and len(roles_bad) == 1


def test_prompts_give_balanced_operator_and_repeat_avoidance_guidance():
    from itc2019.llm import PROMPTS

    for key in ("single_agent", "planner", "inspector"):
        assert "Prefer the smallest change" not in PROMPTS[key]
        assert "depend mainly on outside_region classes" not in PROMPTS[key]
        assert "search window" in PROMPTS[key]
        assert "best_hard_change" in PROMPTS[key]
    for key in ("single_agent", "planner"):
        prompt = PROMPTS[key]
        assert "randomly resamples" not in prompt
        assert "eviction" in prompt
        assert "3 to 5 in-region classes" not in prompt
        for token in ("move", "swap", "destroy_repair", "no_op", "recent_outcomes"):
            assert token in prompt
    assert "recent_outcomes" in PROMPTS["inspector"]
    assert "no_improving_candidate" in PROMPTS["inspector"]
