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
    response=NS(response_id="fake-id",model_version=llm.DEFAULT_MODEL+"-v1",
                text=json.dumps(NO_OP),candidates=[NS(finish_reason="STOP")],prompt_feedback=None,
                usage_metadata=NS(prompt_token_count=20,candidates_token_count=30,
                                  thoughts_token_count=12,total_token_count=62,cached_content_token_count=0))
    def generate(**kwargs):
        calls.append(kwargs)
        return response
    fake=NS(models=NS(generate_content=generate),close=lambda:None)
    monkeypatch.setattr(genai,"Client",lambda **kw:fake)
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,1024,free_tier_confirmed=True,
                                  min_request_interval_seconds=0)
    return client,response,calls


def test_default_model_matches_selected_live_connection_model():
    assert llm.DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert len(llm.FREE_TIER_MODELS) == len(set(llm.FREE_TIER_MODELS))


def test_key_and_billing_confirmation_are_required_before_any_sdk_call(monkeypatch):
    monkeypatch.delenv("GEMINI_FREE_TIER_CONFIRMED",raising=False)
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    with pytest.raises(RuntimeError,match="Free Tier"):
        llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,1024)
    with pytest.raises(RuntimeError,match="GEMINI_API_KEY"):
        llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,1024,free_tier_confirmed=True)
    with pytest.raises(ValueError,match="permits"):
        llm.GeminiDecisionClient("gemini-4-argon",15,1024,free_tier_confirmed=True)


def test_valid_json_usage_and_bounded_native_config(adapter):
    client,response,calls=adapter
    decision,record=client.request("single_agent",{"candidate_regions":[]},12)
    assert decision==NO_OP and record["status"]=="ok"
    assert record["api_requests"]==1 and record["thinking_tokens"]==12
    assert record["total_tokens"]==62 and record["provider"]=="gemini"
    config=calls[0]["config"]
    assert config.response_json_schema==llm.SCHEMAS["single_agent"]
    assert config.http_options.timeout<=12000
    assert config.http_options.retry_options.attempts==1
    assert config.max_output_tokens==1024 and config.candidate_count==1
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
                           "finishReason":finish_reason}],"modelVersion":llm.DEFAULT_MODEL+"-v1",
            "continuationToken":"must-not-be-followed",
            "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":30,"totalTokenCount":50}})
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,1024,free_tier_confirmed=True)
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
    client=llm.GeminiDecisionClient(llm.DEFAULT_MODEL,15,1024,free_tier_confirmed=True)
    client.close()
    client.client=genai.Client(api_key="offline-placeholder",vertexai=False,
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                      retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        decision,record=client.request("single_agent",{})
        assert decision is None and record["status"]=="api_quota_exhausted"
        assert len(requests)==1
    finally:
        client.close()


def test_wrapper_applies_gemini_policy_only_to_controllers():
    profile=json.loads((ROOT/"profiles/pilot.json").read_text(encoding="utf-8"))
    for mode in ("baseline","single","multi"):
        command=commands(profile,pipeline=mode,seed=17,instances=["nbi-spr18"],model=None,
                         output=None,confirm_free_tier=True)[0]
        assert ("--confirm-free-tier" in command)==(mode!="baseline")
        if mode!="baseline":
            assert command[command.index("--model")+1]==llm.DEFAULT_MODEL
            assert command.count("--provider")==1


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


def test_comparison_accepts_baseline_and_single_without_multi(tmp_path):
    folders=make_runs(tmp_path)
    result=compare({key:value for key,value in folders.items() if key!="multi"})
    assert result["methods"]==["baseline","single"]
    assert set(result["instances"][0]["methods"])=={"baseline","single"}
    assert "multi_minus_single" not in result["instances"][0]
    assert result["comparison_type"]=="paired_shared_caps"


def test_two_pipeline_comparison_still_rejects_changed_search_settings(tmp_path):
    folders=make_runs(tmp_path)
    folders.pop("multi")
    path=folders["single"]/"run_config.json"
    manifest=json.loads(path.read_text())
    manifest["settings"]["config"]["seed"]=42
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="differs from baseline on config"):
        compare(folders)


def _summary_fixture(tmp_path, monkeypatch):
    from tools import summarize_paired_pilot as collector
    folders=make_runs(tmp_path)
    for method in ("baseline","single"):
        path=folders[method]/"run_config.json"
        manifest=json.loads(path.read_text())
        manifest["settings"]["config"]["evaluations"]=120
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(collector,"discover",lambda profile,instance,seed,method:folders[method])
    monkeypatch.setattr(collector,"audit_attempts",lambda profile,seed:[])
    profile={"instances":{"X":{}},"llm":{"model":llm.DEFAULT_MODEL}}
    return collector,folders,profile


def test_summary_does_not_count_censored_pair_as_equal_budget_win(tmp_path,monkeypatch):
    collector,folders,profile=_summary_fixture(tmp_path,monkeypatch)
    path=folders["single"]/"summary.json"
    rows=json.loads(path.read_text())
    rows[0]["evaluations"]=100
    rows[0]["score"]["hard"]=2
    path.write_text(json.dumps(rows))
    report=collector.summarize(profile,17)
    assert report["paired_instances"]==1 and report["evaluation_matched_pairs"]==0
    assert report["instances"][0]["quality_outcome"]=="single_better"
    assert report["quality_counts_matched_only"]["single_better"]==0


def test_summary_excludes_failed_single_run_not_as_a_quality_loss(tmp_path,monkeypatch):
    collector,folders,profile=_summary_fixture(tmp_path,monkeypatch)
    path=folders["single"]/"summary.json"
    rows=json.loads(path.read_text())
    rows[0]["controller"].update(comparison_eligible=False,completion_status="incomplete_api_failure")
    path.write_text(json.dumps(rows))
    report=collector.summarize(profile,17)
    assert report["paired_instances"]==0 and len(report["excluded"])==1
    assert report["quality_counts_matched_only"]["baseline_better"]==0


def test_summary_does_not_rank_soft_totals_for_equal_infeasible_hard(tmp_path,monkeypatch):
    collector,folders,profile=_summary_fixture(tmp_path,monkeypatch)
    path=folders["single"]/"summary.json"
    rows=json.loads(path.read_text())
    rows[0]["score"]["total"]=50
    path.write_text(json.dumps(rows))
    report=collector.summarize(profile,17)
    assert report["evaluation_matched_pairs"]==1
    assert report["instances"][0]["quality_outcome"]=="infeasible_hard_tie"
    assert report["instances"][0]["soft_change_percent_if_both_feasible"] is None


def test_discovery_keeps_failed_attempt_but_selects_completed_retry(tmp_path,monkeypatch):
    from tools import summarize_paired_pilot as collector
    folders=[]
    for label,complete in (("failed",False),("retry",True)):
        folder=tmp_path/label
        folder.mkdir()
        (folder/"run_config.json").write_text(json.dumps({"completion_status":
            "completed" if complete else "incomplete_api_failure"}))
        (folder/"summary.json").write_text(json.dumps([{"controller":{"comparison_eligible":complete}}]))
        folders.append(folder)
    monkeypatch.setattr(collector,"matching_runs",lambda *args:folders)
    assert collector.discover({},"X",17,"single")==folders[1]
    assert (folders[0]/"run_config.json").is_file()


def test_discovery_rejects_multiple_completed_retries_instead_of_best_score(tmp_path,monkeypatch):
    from tools import summarize_paired_pilot as collector
    folders=[]
    for label in ("a","b"):
        folder=tmp_path/label
        folder.mkdir()
        (folder/"run_config.json").write_text(json.dumps({"completion_status":"completed"}))
        (folder/"summary.json").write_text(json.dumps([{"controller":{"comparison_eligible":True}}]))
        folders.append(folder)
    monkeypatch.setattr(collector,"matching_runs",lambda *args:folders)
    with pytest.raises(ValueError,match="do not select by score"):
        collector.discover({},"X",17,"single")


def test_behavior_does_not_count_infeasible_soft_drop_as_primary_progress(tmp_path):
    from tools import summarize_paired_pilot as collector
    folder=tmp_path/"X"
    folder.mkdir()
    events=[]
    for generation,(before,after) in enumerate((((4,100),(4,90)),((4,90),(3,200)),((0,300),(0,250))),1):
        events.append({"generation":generation,"validation":"valid_action","trigger":["deadline"],
            "state":{"global":{"incumbent_hard":before[0],"incumbent_internal_total":before[1]}},
            "incumbent_after":{"hard":after[0],"internal_total":after[1]},
            "action":{"operator_id":"move","next_review_interval":10,"execution_budget":10},
            "outcome":{"status":"no_improving_candidate","evaluations":1}})
    (folder/"controller_trace.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    result=collector.behavior(tmp_path,"X")
    assert result["interventions_improving_incumbent"]==3
    assert result["interventions_improving_primary_progress"]==2
    assert result["interventions_reducing_incumbent_hard"]==1
    assert result["interventions_improving_feasible_soft"]==1
    assert result["interventions_lowering_infeasible_soft_only"]==1


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
