"""Shared-solver correctness and neighborhood regression tests."""
from dataclasses import replace
from types import SimpleNamespace
import sys
from pathlib import Path
import time

import numpy as np
from pymoo.core.population import Population

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itc2019.instance import Class, Instance, Room, Time, Distribution, Course, Config, Student
from itc2019.nsga3 import PlacementProblem
from itc2019.controller import BaselineMonitor
from itc2019.scoring import evaluate
from itc2019.neighborhood import distribution_neighborhood_repair
from itc2019.construction import ConstructionIndex
from itc2019.scoring import ScoreEngine
from itc2019.nsga3 import _improve_hard
from itc2019.nsga3 import Config as SolverConfig, run, evaluation_count
from itc2019.scoring import _greedy_section_students, validate_sectioning
from itc2019.operators import apply_operator
from itc2019.regions import RegionCandidate
from pymoo.core.evaluator import Evaluator
from itertools import product
from itc2019.neighborhood import PlacementNeighborhood, _global_costs
from itc2019.scoring import distribution_violations
from itc2019.regions import RegionBuilder


def fixture():
    slot = Time("1", "1", 0, 1)
    classes = tuple(Class(str(i), str(i), "1", "1", None, None, (slot,),
                          (("1", 0), ("2", 0)), False) for i in range(2))
    return Instance("strengthening", 1, 1, 4, (1, 1, 1, 1),
                    {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                    classes, {}, (), (), {c.id: i for i, c in enumerate(classes)})


def test_archive_survives_rejected_candidate_and_tracks_progress():
    instance = fixture()
    source = np.zeros(4, dtype=np.int32)
    improved = source.copy()
    improved[3] = 1
    problem = PlacementProblem(instance)
    problem._evaluate(source, {})
    problem._evaluate(improved, {})
    source_score = evaluate(instance, source[:2], source[2:])[0]
    algo = SimpleNamespace(problem=problem, pop=Population.new(
        "X", source[None], "F", np.asarray([source_score.weighted]), "G", np.asarray([[1]])))
    monitor = BaselineMonitor(instance, time.perf_counter(), None)
    assert monitor._sync_incumbent(algo)[0] == 0
    np.testing.assert_array_equal(monitor.best_choices, improved)
    improved[:] = 0
    assert problem.incumbent.quality[0] == 0


def test_distribution_joint_neighborhood_without_room_targets():
    instance = fixture()
    classes = tuple(replace(instance.classes[0], id=str(i), no_room=True, room_options=(),
                            times=tuple(Time("1", "1", start, 1) for start in range(3)))
                    for i in range(3))
    instance = replace(instance, classes=classes, class_index={c.id: i for i, c in enumerate(classes)},
                       distributions=(Distribution("SameAttendees", (), True, 0, (0, 1, 2)),))
    source = np.asarray([0, 0, 0, -1, -1, -1])
    candidate, stats = distribution_neighborhood_repair(
        instance, source, deadline=time.perf_counter()+3)
    assert candidate is not None and stats["target_constraint"] == "SameAttendees"
    assert evaluate(instance, candidate[:3], candidate[3:])[0].hard == 0
    np.testing.assert_array_equal(source, [0, 0, 0, -1, -1, -1])


def test_construction_propagates_common_start_and_placed_pair_constraints():
    instance = fixture()
    classes = tuple(replace(instance.classes[0], id=str(i), no_room=True, room_options=(),
                            times=(Time("1", "1", i, 1), Time("1", "1", 3, 1))) for i in range(3))
    instance = replace(instance, classes=classes, class_index={c.id: i for i, c in enumerate(classes)},
                       distributions=(Distribution("SameStart", (), True, 0, (0, 1, 2)),))
    builder = ConstructionIndex(instance)
    for seed in range(3):
        choices = builder.build(np.random.default_rng(seed))
        assert evaluate(instance, choices[:3], choices[3:])[0].hard == 0


def test_block_and_break_models_repair_without_room_conflicts():
    for kind, starts, params in [("MaxBreaks", (0, 2, 4), (0, 0)),
                                 ("MaxBlock", (0, 1, 2), (2, 0))]:
        instance = fixture()
        options = ((0,), (1, 2), (2, 4)) if kind == "MaxBreaks" else ((0,), (1,), (2, 4))
        classes = tuple(replace(instance.classes[0], id=str(i), no_room=True, room_options=(),
                                times=tuple(Time("1", "1", t, 1) for t in values))
                        for i, values in enumerate(options))
        instance = replace(instance, classes=classes, class_index={c.id:i for i,c in enumerate(classes)},
                           distributions=(Distribution(kind, params, True, 0, (0,1,2)),))
        source = np.asarray([0, 1 if kind == "MaxBreaks" else 0, 1 if kind == "MaxBreaks" else 0,
                             -1, -1, -1])
        candidate, stats = distribution_neighborhood_repair(instance, source, deadline=time.perf_counter()+5)
        assert candidate is not None, stats
        assert evaluate(instance, candidate[:3], candidate[3:])[0].hard == 0
        assert stats["modeled_global_constraints"] == 1


def test_cache_is_per_instance_and_hits_are_charged_logically():
    instance = fixture()
    engine = ScoreEngine(instance, capacity=1)
    source = np.zeros(4, dtype=int)
    first = engine.score(source)
    assert engine.score(source) == first
    assert engine.requests == 2 and engine.physical_calls == 1 and engine.cache_hits == 1
    changed = source.copy()
    changed[3] = 1
    assert engine.score(changed).hard == 0
    assert engine.score(source) == first
    assert engine.physical_calls == 3 and len(engine.cache) == 1


def test_routine_repair_respects_an_expired_deadline():
    instance = fixture()
    source = np.zeros(4, dtype=int)
    calls = [0]
    result = _improve_hard(instance, source.copy(), np.random.default_rng(17), 100, 10,
                           calls, deadline=time.perf_counter()-1)
    np.testing.assert_array_equal(result, source)
    assert calls[0] == 0


def sectioning_fixture(beam_failure=False):
    base = fixture()
    roots = 16 if beam_failure else 2
    parents = [None] * roots + ([str(roots-1)] if beam_failure else [None, "0"])
    classes = tuple(replace(base.classes[0], id=str(i), course_id="C", config_id="K",
                            subpart_id="root" if i < roots else "child", parent_id=parent,
                            no_room=True, room_options=(), limit=1)
                    for i, parent in enumerate(parents))
    course = Course("C", (Config("K", (tuple(range(roots)), tuple(range(roots, len(classes))))),))
    return replace(base, classes=classes, class_index={c.id:i for i,c in enumerate(classes)},
                   courses={"C":course}, students=tuple(Student(str(i), ("C",))
                                                     for i in range(1 if beam_failure else 2)))


def test_sectioning_reassigns_earlier_students_to_release_capacity():
    instance = sectioning_fixture()
    times = [c.times[0] for c in instance.classes]
    assert _greedy_section_students(instance, times, [None]*4, keep_assignments=True)[0] == 1
    score, assignments = evaluate(instance, [0]*4, [-1]*4, keep_assignments=True)
    assert score.unassigned_requests == 0
    validate_sectioning(instance, assignments)
    repeated = evaluate(instance, [0]*4, [-1]*4, keep_assignments=True)
    assert repeated == (score, assignments)
    assert instance.sectioning_cache["C"]["plan"]
    assert not replace(instance).sectioning_cache


def test_sectioning_recovers_a_valid_bundle_outside_greedy_beam():
    instance = sectioning_fixture(beam_failure=True)
    count = len(instance.classes)
    assert _greedy_section_students(instance, [c.times[0] for c in instance.classes],
                                   [None]*count, keep_assignments=True)[0] == 1
    score, assignments = evaluate(instance, [0]*count, [-1]*count, keep_assignments=True)
    assert score.unassigned_requests == 0
    validate_sectioning(instance, assignments)
    assert set(assignments) == {15,16}


def test_sectioning_does_not_hide_insufficient_capacity():
    instance = sectioning_fixture()
    instance = replace(instance, students=instance.students+(Student("extra", ("C",)),))
    score, _ = evaluate(instance, [0]*4, [-1]*4)
    assert score.unassigned_requests > 0
    assert not instance.sectioning_cache["C"]["plan"]


def test_joint_operator_crosses_four_class_barrier_and_respects_boundary():
    base = fixture()
    classes = tuple(replace(base.classes[0], id=str(i), no_room=True, room_options=(),
                            times=(Time("1","1",0,1,10),Time("1","1",1,1,0))) for i in range(5))
    instance = replace(base, classes=classes, class_index={c.id:i for i,c in enumerate(classes)},
                       distributions=(Distribution("SameStart", (), True, 0, (0,1,2,3)),))
    problem = PlacementProblem(instance)
    source = np.asarray([0]*5+[-1]*5)
    population = Population.new("X", source[None])
    evaluator = Evaluator()
    evaluator.eval(problem, population)
    # Deliberately reject the improved candidate: the shared archive must still retain it.
    survival = SimpleNamespace(do=lambda problem,pop,**kw: pop[:1])
    algorithm = SimpleNamespace(problem=problem,pop=population,evaluator=evaluator,survival=survival,
                                random_state=np.random.default_rng(17),_set_optimum=lambda:None)
    region = RegionCandidate("0","r","b",(0,1,2,3),tuple(map(str,range(4))), ("4",),
                             ("0",),0,40,(),(),4,0,("destroy_repair",))
    outcome = apply_operator(algorithm,instance,region,"destroy_repair",12,
                             np.random.default_rng(17),time.perf_counter()+5)
    assert outcome.candidate_hard == 0 and outcome.candidate_total == 10, outcome
    assert outcome.status == "rejected_by_selection"
    assert problem.incumbent.quality == (0,10)
    assert problem.incumbent.choices[4] == 0 and problem.incumbent.choices[9] == -1
    assert problem.engine.requests == evaluator.n_eval
    assert outcome.evaluations <= 12  # final recheck is inside the requested allowance


def test_cp_global_costs_match_scorer_with_fixed_boundary_and_duplicate_intervals():
    from ortools.sat.python import cp_model
    base = fixture()
    options = ((Time("11","11",0,2),Time("10","01",4,1)),
               (Time("11","11",0,2),Time("01","10",2,1)),
               (Time("11","11",2,2),Time("10","11",5,1)),
               (Time("11","11",6,1),))
    classes = tuple(replace(base.classes[0],id=str(i),times=ts,no_room=True,room_options=())
                    for i,ts in enumerate(options))
    for kind,params in (("MaxBreaks",(0,0)),("MaxBlock",(3,0)),
                        ("MaxDayLoad",(3,)),("MaxDays",(1,))):
        d = Distribution(kind,params,True,0,(0,1,2,3))
        instance = replace(base,nr_days=2,nr_weeks=2,classes=classes,
                           class_index={c.id:i for i,c in enumerate(classes)},distributions=(d,))
        for selection in product(range(2),repeat=3):
            source = np.asarray([*selection,0,-1,-1,-1,-1])
            state = PlacementNeighborhood(instance,source)
            domains = {i:[(j,-1) for j in range(len(classes[i].times))] for i in range(3)}
            model = cp_model.CpModel()
            variables = {i:model.new_int_var(0,1,f"x{i}") for i in range(3)}
            for i in range(3):
                model.add(variables[i] == selection[i])
            costs = []
            before, count = _global_costs(model,variables,domains,dict(enumerate(selection)),
                                          set(range(3)),state,costs,time.perf_counter()+5)
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            assert solver.solve(model) == cp_model.OPTIMAL
            expected = distribution_violations(instance,d,state.times,state.rooms)
            assert count == 1 and before == expected
            assert sum(solver.value(cost) for cost in costs) == expected, (kind,selection)


def test_regions_attribute_only_violating_pairs_and_keep_satisfied_boundary_edges():
    base = fixture()
    classes = tuple(replace(base.classes[0],id=str(i),no_room=True,room_options=(),
                            times=(Time("1","1",start,1),)) for i,start in enumerate((0,0,1,3)))
    instance = replace(base,classes=classes,class_index={c.id:i for i,c in enumerate(classes)},
                       distributions=(Distribution("NotOverlap",(),True,0,(0,1,2)),
                                      Distribution("DifferentTime",(),True,0,(2,3))))
    builder = RegionBuilder(instance)
    hard,soft,neighbors,ids,_ = builder._contributions(np.asarray([0]*4+[-1]*4))
    assert hard == [1,1,0,0]
    assert 3 in neighbors[2] and "distribution:1" in ids[2]


def test_partial_last_offspring_batch_does_not_overshoot_logical_budget():
    for budget in (20,21,39,41):
        _,_,result = run(fixture(),SolverConfig(population=20,partitions=3,generations=20,
                                                evaluations=budget,seed=17))
        assert evaluation_count(result.algorithm) == budget
        assert result.algorithm.problem.engine.requests == budget
