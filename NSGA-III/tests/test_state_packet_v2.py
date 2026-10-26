"""Offline state/repair tests use full solver semantics without API calls."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import sys
import time
import numpy as np
import pytest
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itc2019.instance import Class, Distribution, Instance, Room, Student, Time
from itc2019.nsga3 import PlacementProblem
from itc2019.regions import RegionBuilder, RegionCandidate, population_metrics, expand_blocker_region
from itc2019.diagnostics import PlacementDiagnostics, source_hash, placement
from itc2019.controller import ControllerSettings
from itc2019.operators import apply_operator, OPERATORS
from itc2019.state_packet import make_packet, fit_packet
from itc2019.llm import SCHEMAS, validate_shape


def instance_fixture():
    early, late = Time("1","1",0,1,10), Time("1","1",1,1,0)
    classes = (Class("A","A","1","1",None,None,(early,late),(("1",0),),False),
               Class("B","B","1","1",None,None,(early,),(("1",0),),False))
    return Instance("packet-test",1,1,4,(3,1,10,5), {"1":Room("1",None,(),{})},
                    classes,{},(),(),{"A":0,"B":1})


def algorithm(instance, source, reject=False, budget=50):
    problem = PlacementProblem(instance, budget)
    pop = Population.new("X", np.asarray([source], dtype=np.int32))
    evaluator = Evaluator()
    evaluator.eval(problem,pop)
    return SimpleNamespace(problem=problem,pop=pop,evaluator=evaluator,n_gen=5,
        ref_dirs=np.eye(4),random_state=np.random.default_rng(17),
        survival=SimpleNamespace(do=lambda problem,pop,**kw: pop[:1] if reject else pop[-1:]),
        _set_optimum=lambda:None)


def region_for(instance, algo, active=(0,)):
    source = algo.pop[0].X
    n = len(instance.classes)
    summaries = tuple({"class_id":instance.classes[i].id,
        "current_time":placement(instance,source,i)["time"], "current_room_id":placement(instance,source,i)["room_id"],
        "time_options":len(instance.classes[i].times), "room_options":len(instance.classes[i].room_options),
        "current_time_penalty":instance.classes[i].times[int(source[i])].penalty,
        "current_room_penalty":0} for i in active)
    ids = tuple(f"distribution:{j}" for j,d in enumerate(instance.distributions) if set(active).intersection(d.classes))
    region = RegionCandidate("0","0:r","boundary",active,tuple(instance.classes[i].id for i in active),
        tuple(instance.classes[i].id for i in range(n) if i not in active),ids,1,0,(),summaries,
        len(active),0,OPERATORS,source_quality=(int(algo.pop[0].G[0]),int(sum(algo.pop[0].F))),
        source_weighted=tuple(map(int,algo.pop[0].F)))
    return replace(region,screening=PlacementDiagnostics(instance,source).extract(region,region.source_quality[0]))


def test_diagnostics_are_read_only_deterministic_without_full_scoring_or_rng():
    instance = instance_fixture()
    algo = algorithm(instance,[0,0,0,0])
    original = algo.pop[0].X.copy()
    requests = algo.problem.engine.requests
    cache = list(algo.problem.engine.cache)
    rng_state = copy.deepcopy(algo.random_state.bit_generator.state)
    first = RegionBuilder(instance).build(algo,2,1)
    second = RegionBuilder(instance).build(algo,2,1)
    assert first == second
    assert algo.problem.engine.requests == requests
    assert cache == list(algo.problem.engine.cache)
    assert rng_state == algo.random_state.bit_generator.state
    np.testing.assert_array_equal(original,algo.pop[0].X)
    assert first[0].screening["move_proposals"]
    assert first[0].screening["boundary_context"]["outside_classes"]["items"][0]["class_id"] == "B"


def test_bounded_coverage_and_full_group_rule_check_before_truncation():
    instance = instance_fixture()
    times = tuple(Time("1","1",i,1) for i in range(50))
    classes = tuple(replace(instance.classes[0],id=str(i),no_room=True,room_options=(),times=times if i == 0 else (times[0],)) for i in range(40))
    instance = replace(instance,classes=classes,class_index={c.id:i for i,c in enumerate(classes)},
        distributions=(Distribution("SameStart",(),True,0,tuple(range(40))),))
    algo = algorithm(instance,[0]*40+[-1]*40)
    region = region_for(instance,algo)
    diagnostic = region.screening["diagnostics"][0]
    assert diagnostic["coverage"] == {"candidates_total":49,"checked":24,"complete":False}
    assert sum(diagnostic["counts"].values()) == 24
    assert diagnostic["counts"]["fail"] == 24
    delta = diagnostic["witnesses"][0]["evidence"]["hard_components_delta"]
    assert delta["hard_distributions"] == 39
    context = region.screening["boundary_context"]
    assert context["outside_classes"]["count"] == 39
    assert context["outside_classes"]["returned_count"] == 32
    assert context["outside_classes"]["truncated"]
    assert len(context["constraints"]["details"][0]["class_ids"]) == 40


def test_guided_move_uses_screened_proposals_and_charges_recheck_inside_budget():
    instance = instance_fixture()
    algo = algorithm(instance,[0,0,0,0])
    region = region_for(instance,algo)
    before = algo.problem.engine.requests
    outcome = apply_operator(algo,instance,region,"guided_move",2,np.random.default_rng(17),None)
    assert outcome.admitted and outcome.selected and outcome.candidate_hard == 0
    assert outcome.changed_classes == ("A",)
    assert outcome.evaluations == 2 and outcome.scoring_rechecks == 1
    assert algo.problem.engine.requests-before == 2
    assert algo.problem.engine.requests == algo.evaluator.n_eval
    assert algo.pop[0].X[1] == 0 and algo.pop[0].X[3] == 0


@pytest.mark.parametrize("budget",[1,2,3])
def test_global_remaining_allowance_is_never_exceeded(budget):
    instance = instance_fixture()
    algo = algorithm(instance,[0,0,0,0],budget=budget)
    region = region_for(instance,algo)
    outcome = apply_operator(algo,instance,region,"guided_move",24,np.random.default_rng(17),None)
    assert algo.problem.engine.requests <= budget
    assert outcome.evaluations <= max(0,budget-1)


def test_stale_shortlist_cannot_be_reused_on_changed_source():
    instance = instance_fixture()
    algo = algorithm(instance,[0,0,0,0])
    region = region_for(instance,algo)
    algo.pop[0].X[0] = 1
    before = algo.problem.engine.requests
    outcome = apply_operator(algo,instance,region,"guided_move",5,np.random.default_rng(17),None)
    assert outcome.status == "stale_region" and algo.problem.engine.requests == before


def test_irrelevant_room_freedom_does_not_advertise_time_rule_repair():
    instance = instance_fixture()
    classes = (replace(instance.classes[0],times=(Time("1","1",1,1),),room_options=(("1",0),("2",0))),instance.classes[1])
    instance = replace(instance,classes=classes,rooms={**instance.rooms,"2":Room("2",None,(),{})},
        distributions=(Distribution("SameStart",(),True,0,(0,1)),))
    algo = algorithm(instance,[0,0,0,0])
    assert region_for(instance,algo).screening["distribution_target"] is None
    assert "distribution_repair" not in RegionBuilder(instance).build(algo,1,1)[0].available_operators


def test_distribution_repair_targets_exact_active_participants_and_keeps_outside():
    instance = instance_fixture()
    classes = tuple(replace(c,no_room=True,room_options=()) for c in instance.classes)
    instance = replace(instance,classes=classes,distributions=(Distribution("DifferentTime",(),True,0,(0,1)),))
    algo = algorithm(instance,[0,0,-1,-1],reject=True)
    region = region_for(instance,algo)
    result = apply_operator(algo,instance,region,"distribution_repair",4,np.random.default_rng(17),None)
    assert result.targeted_rule == "distribution:0"
    assert result.candidate_hard == 0 and result.admitted and not result.selected
    assert result.changed_classes == ("A",)
    assert algo.problem.incumbent.choices[1] == 0
    assert algo.problem.engine.requests == algo.evaluator.n_eval


def test_fixing_target_without_full_hard_progress_is_rejected():
    instance = instance_fixture()
    # A's alternative repairs DifferentTime with B, but conflicts with fixed C.
    classes = tuple(replace(c,no_room=True,room_options=()) for c in
        (instance.classes[0],instance.classes[1],replace(instance.classes[1],id="C",times=(Time("1","1",1,1),))))
    instance = replace(instance,classes=classes,class_index={c.id:i for i,c in enumerate(classes)},
        distributions=(Distribution("DifferentTime",(),True,0,(0,1)),Distribution("NotOverlap",(),True,0,(0,2))))
    algo = algorithm(instance,[0,0,0,-1,-1,-1])
    result = apply_operator(algo,instance,region_for(instance,algo),"distribution_repair",5,np.random.default_rng(17),None)
    assert result.source_hard == 1 and not result.admitted
    np.testing.assert_array_equal(algo.pop[0].X,[0,0,0,-1,-1,-1])


@pytest.mark.parametrize("kind",["MaxDays","MaxDayLoad"])
def test_targeted_global_distribution_checks_the_fixed_group(kind):
    instance = instance_fixture()
    times = ((Time("10","1",0,1),Time("01","1",0,1)) if kind == "MaxDays" else
             (Time("10","1",0,2),Time("10","1",2,1)))
    fixed = Time("01" if kind == "MaxDays" else "10","1",0,1)
    classes = (replace(instance.classes[0],no_room=True,room_options=(),times=times),
               replace(instance.classes[1],no_room=True,room_options=(),times=(fixed,)))
    instance = replace(instance,classes=classes,nr_days=2,
        distributions=(Distribution(kind,(1 if kind == "MaxDays" else 2,),True,0,(0,1)),))
    algo = algorithm(instance,[0,0,-1,-1])
    result = apply_operator(algo,instance,region_for(instance,algo),"distribution_repair",5,np.random.default_rng(17),None)
    assert result.source_hard == 1 and result.candidate_hard == 0
    assert result.targeted_rule == "distribution:0" and algo.pop[0].X[1] == 0


def test_packet_has_six_blocks_and_trim_preserves_validation_ids():
    instance = instance_fixture()
    algo = algorithm(instance,[0,0,0,0])
    regions = RegionBuilder(instance).build(algo,2,1)
    metrics = population_metrics(algo,10,None)
    metrics.update(incumbent_hard=1,incumbent_internal_total=60,trigger_reasons=["deadline"])
    packet = make_packet(metrics,regions,[{"generation":5,"hard":1,"internal_total":60}],[],ControllerSettings("fake"),4,OPERATORS)
    assert set(packet) == {"global_search_state","recent_quality","candidate_regions","boundary_context","recent_outcomes","action_limits"}
    assert packet["global_search_state"]["quality"]["best_feasible"] is None
    fitted,status = fit_packet(packet,60000)
    assert fitted == packet and status["status"] == "ready"
    assert fit_packet(packet,1)[0] is None
    assert fitted["boundary_context"][0]["constraints"]["ids"] == list(regions[0].constraint_ids)
    assert packet["recent_quality"]["history"][0]["generation"] == 5
    unknown = make_packet(metrics,regions,[],[{"generation":1}],ControllerSettings("fake"),4,OPERATORS)
    assert unknown["recent_outcomes"]["reviews"][0]["comparable_to_current"] is None


def test_no_room_is_null_and_student_ids_never_enter_packet():
    instance = instance_fixture()
    instance = replace(instance,classes=tuple(replace(c,no_room=True,room_options=()) for c in instance.classes))
    algo = algorithm(instance,[0,0,-1,-1])
    # Diagnostics must not touch a decoder even when assignment details aren't cached.
    instance.students = (Student("PRIVATE_STUDENT_ID",()),)
    algo.problem.engine.cache.clear()
    regions = RegionBuilder(instance).build(algo,1,1)
    from itc2019.state_packet import region_packet
    packet = region_packet(regions[0])
    assert packet["active_classes"][0]["placement"]["room_id"] is None
    assert packet["active_classes"][0]["domains"]["no_room"]
    assert "PRIVATE_STUDENT_ID" not in json.dumps(packet)


def test_response_schemas_accept_new_actions_without_changing_action_contract():
    for operator in ("guided_move","distribution_repair"):
        assert operator in SCHEMAS["planner"]["properties"]["operator_id"]["enum"]
        assert validate_shape("planner",{"operator_id":operator,"execution_budget":3,"next_review_interval":5})


def test_blocker_chain_requires_an_explicit_expanded_target_and_keeps_other_classes_fixed():
    instance = instance_fixture()
    slot = Time("1","1",0,1)
    classes = tuple(replace(instance.classes[0],id=str(i),times=(slot,),room_options=rooms)
                    for i,rooms in enumerate(((("1",0),),(("1",0),("2",0)),(("2",0),("3",0)),(("4",0),))))
    instance = replace(instance,classes=classes,class_index={c.id:i for i,c in enumerate(classes)},
        rooms={str(i):Room(str(i),None,(),{}) for i in range(1,5)})
    algo = algorithm(instance,[0]*8)
    original = region_for(instance,algo,active=(1,))
    refused = apply_operator(algo,instance,original,"blocker_chain",12,np.random.default_rng(17),None)
    assert refused.status == "operator_unavailable_in_region"
    expanded = expand_blocker_region(instance,algo.pop[0].X,original,max_classes=3)
    assert expanded is not None and expanded.expanded_region
    assert expanded.region_id != original.region_id and expanded.boundary_id != original.boundary_id
    assert expanded.class_indices == (0,1,2) and original.class_indices == (1,)
    result = apply_operator(algo,instance,expanded,"blocker_chain",12,np.random.default_rng(17),None)
    assert result.admitted and result.candidate_hard == 0
    assert algo.pop[0].X[3] == 0 and algo.pop[0].X[7] == 0
    assert result.evaluations <= 12 and algo.problem.engine.requests == algo.evaluator.n_eval


def test_infeasible_soft_only_trial_updates_archive_but_is_not_admitted():
    instance = instance_fixture()
    classes = (replace(instance.classes[0],room_options=(("2",0),)),instance.classes[1],
               replace(instance.classes[1],id="C"))
    instance = replace(instance,classes=classes,class_index={c.id:i for i,c in enumerate(classes)},
                       rooms={**instance.rooms,"2":Room("2",None,(),{})})
    algo = algorithm(instance,[0]*6)
    result = apply_operator(algo,instance,region_for(instance,algo),"guided_move",3,np.random.default_rng(17),None)
    assert result.source_hard == result.best_trial_hard == 1
    assert result.best_trial_total < result.source_total
    assert not result.admitted and result.candidate_hard is None
    assert algo.problem.incumbent.quality == (1,result.best_trial_total)
    assert algo.pop[0].X[0] == 0


@pytest.mark.parametrize("mode",["single","multi"])
def test_both_controllers_execute_v2_with_offline_client(mode):
    from itc2019.nsga3 import Config, run, evaluation_count
    root = Path(__file__).resolve().parents[2]
    folder, name = (("Single-Agent Controller","SingleAgentController") if mode == "single" else
                    ("Multi-Agent Controller","MultiAgentController"))
    spec = importlib.util.spec_from_file_location("v2_"+mode,root/folder/"controller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class OfflineClient:
        def request(self,role,payload,remaining_seconds=None):
            packet = payload["state"] if role == "planner" else payload
            assert "global_search_state" in packet and "schema_version" not in packet
            region = next(r for r in packet["candidate_regions"] if "guided_move" in r["available_actions"]["operators"])
            decision = {"operator_id":"guided_move","execution_budget":4,"next_review_interval":2}
            if role == "inspector":
                decision = {**region["target"],"relevant_constraint_ids":[]}
            elif role == "single_agent":
                decision.update(region["target"])
            return decision,{"role":role,"status":"ok","api_requests":1,"input_tokens":8,
                             "output_tokens":2,"total_tokens":10,"latency_seconds":0}
    controller = getattr(module,name)(instance_fixture(),ControllerSettings("fake",warmup_generations=1,
        review_interval=2,min_review_interval=1,max_reviews=1),OfflineClient(),17,time.perf_counter(),None)
    _,_,result = run(controller.instance,Config(population=4,partitions=1,generations=2,seed=17,
        initial_repair_steps=0,offspring_repair_steps=0),callback=controller)
    event = controller.events[0]
    assert event["validation"] == "valid_action"
    assert event["metadata"]["schema_version"] == 2
    assert event["metadata"]["diagnostics"]["full_evaluations"] == 0
    assert event["review_started_elapsed_seconds"] <= event["review_completed_elapsed_seconds"]
    assert evaluation_count(result.algorithm) == result.algorithm.problem.engine.requests
    assert controller.report()["native_token_usage_complete"]
