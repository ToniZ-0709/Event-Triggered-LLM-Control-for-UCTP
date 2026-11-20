"""Regression checks for bounded repair and controller acceptance."""

from pathlib import Path
from dataclasses import asdict, replace
import importlib.util
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itc2019.instance import Class, Config as ITCConfig, Course, Distribution, Instance, Room, Student, Time
from itc2019.controller import BaselineMonitor, ControllerCallback, ControllerSettings, _progress_signal
from itc2019.nsga3 import (Config, PlacementProblem, TimetableSurvival, _improve_hard,
                           _joint_improve_hard, _room_chain_repair, _seeded_cv_tournament,
                           evaluation_count, reference_directions, run)
from itc2019.neighborhood import room_neighborhood_repair
from itc2019.repair_policy import FailureBackoff
from itc2019.operators import _admissible_candidate, _admit_repair, inject_operator_candidate, probe_operator
from itc2019.regions import RegionBuilder, RegionCandidate, placement_diversity
from itc2019.room_search import room_plateau_repair
from itc2019.scoring import evaluate


def test_offspring_repair_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        Config(offspring_repair_steps=-1)
    with pytest.raises(ValueError, match="cannot be negative"):
        Config(offspring_pair_trials=-1)
    assert Config(offspring_repair_steps=4, offspring_pair_trials=1).offspring_repair_steps == 4


def _chain_instance() -> Instance:
    slot = Time("1", "1", 0, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (slot,), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None, (slot,), (("1", 0), ("2", 0)), False),
        Class("C", "CC", "1", "1", None, None, (slot,), (("2", 0), ("3", 0)), False),
    )
    return Instance("room-chain", 1, 1, 1, (1, 1, 1, 1),
                    {str(j): Room(str(j), None, (), {}) for j in range(1, 4)},
                    classes, {}, (), (), {cls.id: j for j, cls in enumerate(classes)})


def test_room_chain_can_use_neutral_intermediate_move() -> None:
    instance = _chain_instance()
    source = np.zeros(2 * len(instance.classes), dtype=np.int32)
    original = source.copy()
    before, _ = evaluate(instance, source[:3], source[3:])
    candidate, calls = _room_chain_repair(instance, source,
                                          max_evaluations=100, max_depth=3)
    assert before.hard == 1
    assert candidate is not None
    after, _ = evaluate(instance, candidate[:3], candidate[3:])
    assert after.hard == 0
    assert calls <= 100
    np.testing.assert_array_equal(source, original)


def test_room_chain_repairs_unavailable_room_without_a_clash() -> None:
    slot = Time("1", "1", 0, 1)
    instance = _chain_instance()
    cls = replace(instance.classes[0], times=(slot, Time("1", "1", 1, 1)))
    instance = replace(instance, classes=(cls,), class_index={"A": 0},
                       rooms={"1": Room("1", None, (slot,), {})}, slots_per_day=2)
    source = np.zeros(2, dtype=np.int32)
    before, _ = evaluate(instance, source[:1], source[1:])
    assert before.room_conflicts == 0 and before.room_unavailable == 1
    candidate, calls = _room_chain_repair(instance, source, max_evaluations=10)
    assert candidate is not None and calls <= 10
    assert evaluate(instance, candidate[:1], candidate[1:])[0].hard == 0
    np.testing.assert_array_equal(source, np.zeros(2, dtype=np.int32))


def test_neighborhood_repair_releases_nonviolating_blocker() -> None:
    instance = _chain_instance()
    source = np.zeros(6, dtype=np.int32)
    original = source.copy()
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3,
        max_classes=3, max_options=8, source_score=evaluate(instance, source[:3], source[3:])[0])
    assert candidate is not None
    assert evaluate(instance, candidate[:3], candidate[3:])[0].hard == 0
    assert stats["changed_classes"] == 2
    assert stats["full_scoring_calls"] == 1
    np.testing.assert_array_equal(source, original)


def test_neighborhood_repair_rejects_a_global_constraint_regression() -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (
        Class("A", "A", "1", "1", None, None, (first,), (("1", 0),), False),
        Class("B", "B", "1", "1", None, None, (first, second), (("1", 0),), False),
    )
    instance = Instance("global-boundary", 1, 1, 2, (1, 1, 1, 1),
                        {"1": Room("1", None, (), {})}, classes, {},
                        (Distribution("MaxBlock", (1, 0), True, 0, (0, 1)),), (), {"A": 0, "B": 1})
    source = np.zeros(4, dtype=np.int32)
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3, max_classes=2, max_options=2)
    assert candidate is None
    assert stats["modeled_global_constraints"] == 1
    assert stats["full_scoring_calls"] == 1


def test_neighborhood_models_days_and_load_with_fixed_outside_classes() -> None:
    early = Time("10", "11", 0, 2)
    later = Time("10", "11", 2, 2)
    other_day = Time("01", "11", 0, 2)
    fixed = Time("10", "11", 4, 1)
    classes = (
        Class("A", "A", "1", "1", None, None, (early,), (("1", 0),), False),
        Class("B", "B", "1", "1", None, None, (early, later, other_day), (("1", 0),), False),
        Class("C", "C", "1", "1", None, None, (fixed,), (("2", 0),), False),
    )
    instance = Instance("global-load", 2, 2, 5, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)}, classes, {},
                        (Distribution("MaxDays", (1,), True, 0, (0, 1, 2)),
                         Distribution("MaxDayLoad", (4,), True, 0, (0, 1, 2))),
                        (), {cls.id: i for i, cls in enumerate(classes)})
    source = np.zeros(6, dtype=np.int32)
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3, max_classes=2, max_options=3)
    assert candidate is not None
    score, _ = evaluate(instance, candidate[:3], candidate[3:])
    assert 0 < score.hard < 3
    assert stats["modeled_global_constraints"] == 2
    assert stats["local_hard_after"] == score.hard
    assert candidate[2] == source[2] and candidate[5] == source[5]


def test_controller_neighborhood_can_improve_soft_at_the_same_positive_hard() -> None:
    instance = _chain_instance()
    cls = replace(instance.classes[2], times=(Time("1", "1", 0, 1, 10),
                                              Time("1", "1", 1, 1)),
                  room_options=(("2", 0),))
    instance = replace(instance, classes=(*instance.classes[:2], cls), slots_per_day=2)
    source = np.zeros(6, dtype=np.int32)
    before, _ = evaluate(instance, source[:3], source[3:])
    args = dict(source_score=before, allowed_classes=(2,), require_room_improvement=False)
    assert room_neighborhood_repair(instance, source, deadline=time.perf_counter() + 3,
                                    **args)[0] is None
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3, admit_candidate=_admit_repair, **args)
    assert candidate is not None and stats["full_scoring_calls"] == 1
    after, _ = evaluate(instance, candidate[:3], candidate[3:])
    assert before.hard == after.hard == 1 and after.total < before.total


@pytest.mark.parametrize("distribution,expected", [
    (Distribution("Precedence", (), False, 10, (1, 0)), 0),
    (Distribution("MaxDayLoad", (1,), False, 10, (0, 1)), 5),
])
def test_controller_neighborhood_optimizes_soft_distribution_at_fixed_boundary(
        distribution, expected) -> None:
    first = Time("1", "10", 0, 2)
    if distribution.kind == "Precedence":
        fixed, bad, good = (Time("1", "10", 0, 1), Time("1", "10", 0, 1),
                            Time("1", "10", 2, 1, 1))
    else:
        fixed, bad, good = Time("1", "01", 0, 2), Time("1", "10", 1, 2), Time("1", "10", 0, 1, 1)
    base = _chain_instance()
    classes = (replace(base.classes[0], times=(first, bad, good)),
               replace(base.classes[2], times=(fixed,), room_options=(("2", 0),)))
    instance = replace(base, classes=classes, class_index={"A": 0, "C": 1}, nr_weeks=2,
                       slots_per_day=5, distributions=(distribution,))
    source = np.zeros(4, dtype=np.int32)
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3, allowed_classes=(0,),
        source_score=evaluate(instance, source[:2], source[2:])[0],
        require_room_improvement=False, admit_candidate=_admit_repair,
        max_full_scoring_calls=1)
    assert candidate is not None and stats["full_scoring_calls"] == 1
    assert candidate.tolist() == [2, 0, 0, 0]
    assert evaluate(instance, candidate[:2], candidate[2:])[0].distribution == expected


def test_controller_neighborhood_models_soft_pairs_between_moving_classes() -> None:
    base = _chain_instance()
    early, late = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (replace(base.classes[0], times=(early, late), room_options=(("1", 0), ("2", 1))),
               replace(base.classes[1], times=(late, early), room_options=(("2", 0), ("1", 1))))
    instance = replace(base, classes=classes, class_index={"A": 0, "B": 1},
                       slots_per_day=2, distributions=(Distribution("SameRoom", (), False, 10, (0, 1)),))
    source = np.zeros(4, dtype=np.int32)
    before, _ = evaluate(instance, source[:2], source[2:])
    candidate, stats = room_neighborhood_repair(
        instance, source, deadline=time.perf_counter() + 3, allowed_classes=(0, 1),
        source_score=before, require_room_improvement=False, admit_candidate=_admit_repair,
        max_full_scoring_calls=1)
    assert candidate is not None and stats["full_scoring_calls"] == 1
    after, _ = evaluate(instance, candidate[:2], candidate[2:])
    assert before.hard == after.hard == 0
    assert before.distribution == 10 and after.distribution == 0 and after.total == 1


def test_repair_failure_backoff_allows_progress_and_caps_cooldown() -> None:
    policy = FailureBackoff(5, 20)
    assert policy.ready("same-conflict", 5)
    assert policy.record("same-conflict", 5, False) == 5
    assert not policy.ready("same-conflict", 9)
    assert policy.ready("same-conflict", 10)
    assert policy.record("same-conflict", 10, False) == 10
    assert policy.record("same-conflict", 20, False) == 20
    assert policy.record("same-conflict", 40, False) == 20
    assert policy.ready("hard-improved", 41)


def test_joint_full_scores_are_counted_inside_routine_repair() -> None:
    early, late = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = tuple(Class(str(i), str(i), "1", "1", None, None, (slot,),
                          (("1", 0), ("2", 0)), False)
                    for i, slot in enumerate((early, late)))
    instance = Instance("count-joint", 1, 1, 2, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, {}, (Distribution("SameRoom", (), True, 0, (0, 1)),),
                        (), {cls.id: i for i, cls in enumerate(classes)})
    calls = [0]
    candidate = _improve_hard(instance, np.asarray([0, 0, 0, 1], dtype=np.int32),
                               np.random.default_rng(17), 0, 2, calls)
    assert calls[0] == 2
    assert evaluate(instance, candidate[:2], candidate[2:])[0].hard == 0


def test_two_reference_layers_and_evaluation_stop() -> None:
    config = Config(population=32, partitions=3, inner_partitions=2,
                    generations=20, evaluations=64, seed=17)
    directions = reference_directions(config)
    assert len(directions) == 30
    np.testing.assert_allclose(directions.sum(axis=1), 1)
    _, _, result = run(_chain_instance(), config)
    assert evaluation_count(result.algorithm) >= 64
    assert result.algorithm.n_gen <= 4


def test_default_run_keeps_evolving_after_feasibility() -> None:
    instance = _chain_instance()
    monitor = BaselineMonitor(instance, time.perf_counter(), None)
    _, score, _ = run(instance, Config(population=20, partitions=3, generations=5, seed=17),
                      callback=monitor)
    assert score.hard == 0
    assert len(monitor.anytime_events) == 5


def test_room_plateau_repair_finds_hard_decrease_without_mutating_source() -> None:
    instance = _chain_instance()
    source = np.zeros(2 * len(instance.classes), dtype=np.int32)
    original = source.copy()
    candidate, stats = room_plateau_repair(
        instance, source, np.random.default_rng(17),
        deadline=time.perf_counter() + 2, max_steps=500,
        max_changes=2, hard_slack=1)
    assert candidate is not None
    assert evaluate(instance, candidate[:3], candidate[3:])[0].hard == 0
    assert stats["full_scoring_calls"] >= 2
    np.testing.assert_array_equal(source, original)


def test_room_plateau_repair_handles_unavailable_room_without_a_clash() -> None:
    slot = Time("1", "1", 0, 1)
    cls = Class("A", "CA", "1", "1", None, None, (slot,),
                (("1", 0), ("2", 0)), False)
    instance = Instance("unavailable-room", 1, 1, 1, (1, 1, 1, 1),
                        {"1": Room("1", None, (slot,), {}),
                         "2": Room("2", None, (), {})},
                        (cls,), {}, (), (), {"A": 0})
    source = np.zeros(2, dtype=np.int32)
    before, _ = evaluate(instance, source[:1], source[1:])
    candidate, _ = room_plateau_repair(
        instance, source, np.random.default_rng(17),
        deadline=time.perf_counter() + 2)
    assert before.hard == before.room_unavailable == 1
    assert candidate is not None
    after, _ = evaluate(instance, candidate[:1], candidate[1:])
    assert after.hard == after.room_unavailable == 0


def test_room_chain_can_reduce_clash_alongside_distribution_violation() -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (first,), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None,
              (first, second), (("1", 0), ("2", 0)), False),
        Class("C", "CC", "1", "1", None, None, (first,), (("2", 0),), False),
    )
    instance = Instance("mixed-room-chain", 1, 1, 2, (1, 1, 1, 1),
                        {str(j): Room(str(j), None, (), {}) for j in range(1, 3)},
                        classes, {}, (Distribution("SameRoom", (), True, 0, (1, 2)),),
                        (), {cls.id: j for j, cls in enumerate(classes)})
    source = np.zeros(2 * len(classes), dtype=np.int32)
    before, _ = evaluate(instance, source[:3], source[3:])
    candidate, calls = _room_chain_repair(instance, source, max_evaluations=100,
                                          max_depth=3)
    assert before.hard == 2 and before.room_conflicts == 1
    assert candidate is not None and calls <= 100
    after, _ = evaluate(instance, candidate[:3], candidate[3:])
    assert after.hard == 1 and after.room_conflicts == 0


def test_joint_repair_respects_expired_deadline() -> None:
    instance = _chain_instance()
    choices = np.zeros(2 * len(instance.classes), dtype=np.int32)
    original = choices.copy()
    assert not _joint_improve_hard(instance, choices, np.random.default_rng(17), 8,
                                   deadline=time.perf_counter() - 1)
    np.testing.assert_array_equal(choices, original)


def test_placement_diversity_distinguishes_unique_but_similar_schedules() -> None:
    chromosomes = np.zeros((2, 200), dtype=np.int32)
    chromosomes[1, 0] = 1
    assert placement_diversity(chromosomes) == 0.01


def test_region_builder_backfills_duplicate_centers_and_solutions() -> None:
    slot = Time("1", "1", 0, 1)
    classes = tuple(Class(name, name, "1", "1", None, None, (slot,),
                          ((room, 0),), False)
                    for name, room in (("A", "1"), ("B", "1"), ("C", "2"), ("D", "3")))
    instance = Instance("duplicate-regions", 1, 1, 1, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in range(1, 4)},
                        classes, {}, (), (), {cls.id: i for i, cls in enumerate(classes)})
    pop = Population.new("X", np.zeros((2, 8), dtype=np.int32))
    pop.set("G", np.ones((2, 1)))
    pop.set("F", np.zeros((2, 4)))
    regions = RegionBuilder(instance).build(SimpleNamespace(pop=pop), 3, 2)
    assert len(regions) == 3
    assert {region.solution_id for region in regions} == {"0"}
    assert len({region.class_indices for region in regions}) == 3


def test_region_builder_skips_four_of_five_overlap_on_a_source(monkeypatch) -> None:
    base = _chain_instance()
    classes = tuple(replace(base.classes[0], id=str(i)) for i in range(9))
    instance = replace(base, classes=classes, class_index={cls.id: i for i, cls in enumerate(classes)})
    pop = Population.new("X", np.zeros((1, 18), dtype=np.int32),
                         "G", np.zeros((1, 1)), "F", np.zeros((1, 4)))
    # Two partial neighborhoods share classes 0..3. The third is disjoint.
    neighbors = [{1, 2, 3, 4, 5}, {0}, {0}, {0}, {0}, {0}, {7, 8}, {6}, {6}]
    builder = RegionBuilder(instance)
    monkeypatch.setattr(builder, "_contributions", lambda *a, **k: (
        list(range(9, 0, -1)), [0] * 9, neighbors, [set() for _ in classes], ()))
    regions = builder.build(SimpleNamespace(pop=pop), max_regions=3, region_size=5)
    assert [r.class_indices for r in regions] == [(0, 1, 2, 3, 4), (6, 7, 8)]


def test_stagnation_uses_hard_until_feasibility() -> None:
    assert _progress_signal((5, 100)) == _progress_signal((5, 80))
    assert _progress_signal((0, 100)) != _progress_signal((0, 80))
    settings = ControllerSettings(model="fake", stagnation_window=3)
    callback = ControllerCallback(_chain_instance(), settings, None, 17,
                                  time.perf_counter(), None)
    callback.next_review = 99
    callback.last_review = 2
    metrics = {"generation": 5, "decision_diversity": 1.0,
               "placement_diversity": 0.8,
               "stagnant_generations": 3}
    assert callback._trigger(metrics) == ["stagnation"]
    callback._stagnation_event_handled = True
    assert callback._trigger(metrics) == []
    assert callback._trigger({**metrics, "stagnant_generations": 0}) == []
    assert callback._trigger(metrics) == ["stagnation"]


def test_low_diversity_event_fires_once_per_episode() -> None:
    settings = ControllerSettings(model="fake", diversity_threshold=0.20)
    callback = ControllerCallback(_chain_instance(), settings, None, 17,
                                  time.perf_counter(), None)
    callback.next_review = 99
    metrics = {"generation": 5, "decision_diversity": 1.0,
               "placement_diversity": 0.05,
               "stagnant_generations": 0}
    assert callback._trigger(metrics) == ["low_diversity"]
    callback._low_diversity_event_handled = True
    assert callback._trigger(metrics) == []
    assert callback._trigger({**metrics, "placement_diversity": 0.25}) == []
    assert callback._trigger(metrics) == []
    assert callback._trigger({**metrics, "decision_diversity": 0.2,
                              "placement_diversity": 0.5}) == []
    assert callback._trigger(metrics) == ["low_diversity"]


@pytest.mark.parametrize("active", [(1,), (0, 1)])
def test_destroy_repair_cp_sat_models_student_conflicts_when_feasible(active) -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (first,), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None, (first, second), (("2", 0),), False),
    )
    courses = {"CA": Course("CA", (ITCConfig("1", ((0,),)),)),
               "CB": Course("CB", (ITCConfig("1", ((1,),)),))}
    students = (Student("S1", ("CA", "CB")), Student("S2", ("CA", "CB")))
    instance = Instance("student-pairs", 1, 1, 2, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, courses, (), students,
                        {cls.id: i for i, cls in enumerate(classes)})
    clashing, assignments = evaluate(instance, [0, 0], [0, 0], keep_assignments=True)
    separated, _ = evaluate(instance, [0, 1], [0, 0])
    assert clashing.hard == separated.hard == 0
    assert clashing.student == 2 and separated.student == 0
    assert assignments == {0: ["S1", "S2"], 1: ["S1", "S2"]}

    # One full score suffices only if CP-SAT favors resolving the conflict
    # over the penalty-free option that leaves both classes at the same time.
    options = (first, first, replace(second, penalty=1))
    instance = replace(instance, classes=(classes[0], replace(classes[1], times=options)),
                       weights=(1, 1, 1, 10))
    source = Population.new("X", np.zeros((1, 4), dtype=np.int32))
    problem = PlacementProblem(instance)
    evaluator = Evaluator()
    evaluator.eval(problem, source)
    details = ({"violation_groups": ({"in_region_class_ids": ("B",), "violation_units": 2},)},)
    class_ids = tuple(classes[i].id for i in active)
    boundary = ("A",) if len(active) == 1 else ()
    region = RegionCandidate("0", "r", "b", active, class_ids, boundary, ("student_conflict",),
                             0, 20, (), (), 1, 0, ("destroy_repair",), constraint_details=details)
    probe = probe_operator(SimpleNamespace(pop=source, evaluator=evaluator, problem=problem),
                           instance, region, "destroy_repair", 1, np.random.default_rng(17), None)
    assert probe.candidate_quality == (0, 1)
    assert probe.candidate_origin == "cp_sat"
    assert probe.evaluations == 1 and evaluator.n_eval == problem.engine.requests == 2


def test_controller_acceptance_preserves_hard_priority() -> None:
    source = (50, 50, 0, 0)
    assert _admissible_candidate(2, source, 1, (10000, 0, 0, 0))
    assert _admissible_candidate(2, source, 2, (1, 0, 0, 0))
    assert not _admissible_candidate(2, source, 2, (25, 100, 0, 0))
    assert not _admissible_candidate(2, source, 3, (1, 0, 0, 0))
    assert not _admissible_candidate(0, source, 1, (1, 0, 0, 0))
    assert not _admissible_candidate(0, source, 0, source)
    assert not _admissible_candidate(0, source, 0, (50, 51, 0, 0))
    assert _admissible_candidate(0, source, 0, (49, 50, 0, 0))
    assert _admissible_candidate(0, source, 0, (25, 100, 0, 0))


@pytest.mark.parametrize("status", ["selected", "rejected_by_selection", "scoring_mismatch"])
def test_operator_outcome_serializes_candidate_origin(status) -> None:
    instance = _chain_instance()
    source = Population.new("X", np.asarray([[0, 0, 0, 0, 1, 0]], dtype=np.int32))
    problem, evaluator = PlacementProblem(instance), Evaluator()
    evaluator.eval(problem, source)
    algorithm = SimpleNamespace(pop=source, evaluator=evaluator, problem=problem,
                                random_state=np.random.default_rng(17), _set_optimum=lambda: None)
    region = RegionCandidate("0", "r", "b", (2,), ("C",), ("B",), (), 0, 0, (), (), 1, 0, ("move",))
    probe = probe_operator(algorithm, instance, region, "move", 1, np.random.default_rng(17), None)
    assert probe.candidate_origin == "move"

    if status == "scoring_mismatch":
        probe = replace(probe, candidate_quality=(probe.source_hard, probe.source_total))
    algorithm.survival = SimpleNamespace(
        do=lambda _problem, population, **_: population[-1:] if status == "selected" else population[:1])

    outcome = asdict(inject_operator_candidate(algorithm, probe))

    assert outcome["status"] == status
    assert outcome["candidate_origin"] == "move"


def test_coordinated_repair_is_shared_and_accounted_for() -> None:
    early, late = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (early,), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None, (late,), (("2", 0),), False),
    )
    instance = Instance("fixed-distribution", 1, 1, 2, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, {}, (Distribution("SameRoom", (), True, 0, (0, 1)),),
                        (), {cls.id: i for i, cls in enumerate(classes)})
    monitor = BaselineMonitor(instance, time.perf_counter(), None, seed=17)
    _, score, result = run(instance, Config(population=40, generations=5,
                                            partitions=4, seed=17), callback=monitor)
    assert score.hard == 1
    assert len(monitor.coordinated_repair_events) == 1
    assert monitor.coordinated_repair_events[0]["status"] == "no_improvement"
    assert monitor.coordinated_repair_scoring_calls == 1
    assert result.algorithm.evaluator.n_eval > 200


def test_infeasible_tournament_ties_use_seeded_rng() -> None:
    population = Population.new("CV", np.asarray([[1.0], [1.0], [0.0]]))
    pairs = np.asarray([[0, 1]] * 12 + [[0, 2]])
    left = _seeded_cv_tournament(population, pairs, np.random.default_rng(17))
    right = _seeded_cv_tournament(population, pairs, np.random.default_rng(17))
    np.testing.assert_array_equal(left, right)
    assert left[-1, 0] == 2


def _agent_class(folder: str, class_name: str):
    path = Path(__file__).resolve().parents[2] / folder / "controller.py"
    spec = importlib.util.spec_from_file_location(folder.replace(" ", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class _FakeDecisionClient:
    def request(self, role: str, payload: dict, remaining_seconds=None):
        regions = (payload["state"] if role == "planner" else payload)["candidate_regions"]
        region = next((item for item in regions if item["available_operators"]), None)
        if role == "inspector":
            decision = ({"solution_id": region["solution_id"],
                         "region_id": region["region_id"],
                         "boundary_id": region["boundary_id"],
                         "relevant_constraint_ids": []} if region else
                        {"solution_id": None, "region_id": None,
                         "boundary_id": None, "relevant_constraint_ids": []})
        elif role == "planner":
            decision = {"operator_id": "move" if region else "no_op",
                        "execution_budget": 1 if region else 0,
                        "next_review_interval": 2}
        else:
            decision = {"solution_id": region["solution_id"] if region else None,
                        "region_id": region["region_id"] if region else None,
                        "boundary_id": region["boundary_id"] if region else None,
                        "operator_id": "move" if region else "no_op",
                        "execution_budget": 1 if region else 0,
                        "next_review_interval": 2}
        return decision, {"role": role, "status": "ok", "input_tokens": 0,
                          "output_tokens": 0, "latency_seconds": 0.0}


def test_single_and_inspector_planner_share_solver_and_action_rules() -> None:
    instance = _chain_instance()
    settings = ControllerSettings(model="fake", review_interval=2, min_review_interval=2,
                                  warmup_generations=2)
    for folder, class_name, calls in (
        ("Single-Agent Controller", "SingleAgentController", 1),
        ("Multi-Agent Controller", "MultiAgentController", 2),
    ):
        controller = _agent_class(folder, class_name)(
            instance, settings, _FakeDecisionClient(), 17, time.perf_counter(), None)
        _, _, result = run(instance, Config(population=40, generations=3,
                                            partitions=4, seed=17), callback=controller)
        assert controller.report()["api_calls"] == calls
        assert controller.events[0]["validation"] == "valid_action"
        assert result.algorithm.evaluator.n_eval >= 120
        assert evaluation_count(result.algorithm) == result.algorithm.problem.engine.requests


def test_review_cap_is_shared_even_with_different_api_call_counts() -> None:
    instance = _chain_instance()
    for folder, class_name, max_calls in (
        ("Single-Agent Controller", "SingleAgentController", 1),
        ("Multi-Agent Controller", "MultiAgentController", 2),
    ):
        settings = ControllerSettings(model="fake", review_interval=2, min_review_interval=2,
                                      warmup_generations=2, max_reviews=1,
                                      max_api_calls=max_calls)
        controller = _agent_class(folder, class_name)(
            instance, settings, _FakeDecisionClient(), 17, time.perf_counter(), None)
        run(instance, Config(population=40, generations=5, partitions=4, seed=17),
            callback=controller)
        assert controller.report()["reviews"] == 1
        assert controller.report()["api_calls"] == max_calls


def test_comparison_review_budget_checks_role_call_multiplicity() -> None:
    from compare_pipelines import _check_review_budget
    _check_review_budget({"max_reviews": 8, "max_api_calls": 8},
                         {"max_reviews": 8, "max_api_calls": 16})
    with pytest.raises(ValueError, match="API call limits"):
        _check_review_budget({"max_reviews": 8, "max_api_calls": 8},
                             {"max_reviews": 8, "max_api_calls": 8})


def test_controller_does_not_call_api_after_search_budget_is_spent() -> None:
    instance = _chain_instance()
    settings = ControllerSettings(model="fake", warmup_generations=2,
                                  review_interval=3, min_review_interval=1)
    controller = _agent_class("Single-Agent Controller", "SingleAgentController")(
        instance, settings, _FakeDecisionClient(), 17, time.perf_counter(), None)
    run(instance, Config(population=20, partitions=3, generations=20, evaluations=40), callback=controller)
    assert controller.api_calls == 0
    assert controller.budget_skipped_reviews >= 1


def test_pilot_runner_keeps_per_instance_budgets_and_role_call_caps() -> None:
    from run_pilot import commands
    profile = {"schema_version": 1, "solver": {"population": 20, "partitions": 3},
               "instances": {"A": {"evaluations": 600, "generations": 100},
                             "B": {"evaluations": 1200, "generations": 200}},
               "repair": {"interval": 5}, "controller": {"max_reviews": 4}}
    baseline = commands(profile, pipeline="baseline", seed=17, instances=[], model=None, output=None)
    multi = commands(profile, pipeline="multi", seed=17, instances=["B"], model="fake", output=None)
    assert baseline[0][baseline[0].index("--evaluations") + 1] == "600"
    assert baseline[1][baseline[1].index("--evaluations") + 1] == "1200"
    assert multi[0][multi[0].index("--max-api-calls") + 1] == "8"
    assert multi[0][multi[0].index("--evaluations") + 1] == "1200"
    with pytest.raises(ValueError, match="No measured budget"):
        commands(profile, pipeline="baseline", seed=17, instances=["C"], model=None, output=None)


def test_destroy_repair_rebuild_prefers_hard_safe_options_over_cheap_clashes() -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    # C and D outside the region already clash at (first, room 1), making source_hard=1
    # while leaving B locally free (baseline_local=0) so Stage 1 CP-SAT hands off to rebuild.
    # For B, the cheapest placement uses room 1 when it is unavailable.
    # Moving to room 3 in the second time slot preserves hard and improves soft.
    classes = (
        Class("B", "CB", "1", "1", None, None,
              (replace(first, penalty=10), replace(second, penalty=1)),
              (("1", 0), ("2", 10), ("3", 1)), False),
        Class("C", "CC", "1", "1", None, None, (first,), (("1", 0),), False),
        Class("D", "CD", "1", "1", None, None, (first,), (("1", 0),), False),
    )
    instance = Instance("rebuild-hard-safety", 1, 1, 2, (1, 1, 1, 1),
                        {"1": Room("1", None, (second,), {}), "2": Room("2", None, (), {}),
                         "3": Room("3", None, (), {})},
                        classes, {}, (), (), {cls.id: i for i, cls in enumerate(classes)})
    source = Population.new("X", np.asarray([[0, 0, 0, 1, 0, 0]], dtype=np.int32))
    problem = PlacementProblem(instance)
    evaluator = Evaluator()
    evaluator.eval(problem, source)
    assert int(source[0].G[0]) == 1

    region = RegionCandidate("0", "r", "b", (0,), ("B",), (), (),
                             0, 20, (), (), 1, 0, ("destroy_repair",))
    probe = probe_operator(SimpleNamespace(pop=source, evaluator=evaluator, problem=problem),
                           instance, region, "destroy_repair", 2, np.random.default_rng(17), None)
    assert probe.evaluations == 2
    assert probe.candidate_quality == (1, 2)


def test_placement_neighborhood_move_updates_room_occupants_for_sequential_steps() -> None:
    from itc2019.neighborhood import PlacementNeighborhood

    first = Time("1", "1", 0, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (first,), (("1", 0), ("2", 0)), False),
        Class("B", "CB", "1", "1", None, None, (first,), (("1", 0), ("2", 0)), False),
    )
    instance = Instance("seq-move", 1, 1, 2, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, {}, (), (), {cls.id: i for i, cls in enumerate(classes)})
    hn = PlacementNeighborhood(instance, np.asarray([0, 0, 0, 1], dtype=np.int32))
    assert hn.external_cost(1, (0, 0), set())[0] == 1  # A currently occupies room 1
    assert hn.external_cost(1, (0, 1), set())[0] == 0  # room 2 is empty
    hn.move(0, 0, 1)  # move A into room 2
    assert hn.external_cost(1, (0, 0), set())[0] == 0  # room 1 is now free
    assert hn.external_cost(1, (0, 1), set())[0] == 1  # room 2 now has A

def test_guided_move_can_change_time_and_room_with_one_evaluation() -> None:
    first = Time("1", "1", 0, 1)
    second = Time("1", "1", 1, 1, penalty=1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (first,), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None, (first, second), (("2", 5), ("1", 0)), False),
    )
    courses = {"CA": Course("CA", (ITCConfig("1", ((0,),)),)),
               "CB": Course("CB", (ITCConfig("1", ((1,),)),))}
    instance = Instance("guided-move", 1, 1, 2, (1, 1, 1, 10),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, courses, (), (Student("S", ("CA", "CB")),), {"A": 0, "B": 1})
    source = Population.new("X", np.zeros((1, 4), dtype=np.int32))
    problem, evaluator = PlacementProblem(instance), Evaluator()
    evaluator.eval(problem, source)
    region = RegionCandidate("0", "r", "b", (1,), ("B",), ("A",), (), 0, 15, (), (), 1, 0, ("move",))

    probe = probe_operator(SimpleNamespace(pop=source, evaluator=evaluator, problem=problem),
                           instance, region, "move", 1, np.random.default_rng(17), None)

    assert probe.evaluations == 1 and evaluator.n_eval == 2
    assert probe.candidate_quality == (0, 1)
    assert probe.candidate[0].X.tolist() == [0, 1, 0, 1]
    assert source[0].X.tolist() == [0, 0, 0, 0]


def test_guided_move_ranks_improvement_across_classes_with_one_evaluation() -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    classes = (
        Class("A", "CA", "1", "1", None, None, (first, replace(second, penalty=1)), (("1", 0),), False),
        Class("B", "CB", "1", "1", None, None,
              (replace(first, penalty=100), replace(second, penalty=50)), (("2", 0),), False),
    )
    instance = Instance("guided-move-delta", 1, 1, 2, (1, 1, 1, 1),
                        {str(i): Room(str(i), None, (), {}) for i in (1, 2)},
                        classes, {}, (), (), {"A": 0, "B": 1})
    source = Population.new("X", np.zeros((1, 4), dtype=np.int32))
    problem, evaluator = PlacementProblem(instance), Evaluator()
    evaluator.eval(problem, source)
    region = RegionCandidate("0", "r", "b", (0, 1), ("A", "B"), (), (), 0, 100, (), (), 2, 0, ("move",))

    probe = probe_operator(SimpleNamespace(pop=source, evaluator=evaluator, problem=problem),
                           instance, region, "move", 1, np.random.default_rng(17), None)

    assert probe.evaluations == 1 and evaluator.n_eval == 2
    assert probe.candidate_quality == (0, 50)
    assert probe.candidate[0].X.tolist() == [0, 1, 0, 0]


@pytest.mark.parametrize("exchange", ["room", "time", "full"])
def test_guided_swap_accepts_partial_exchange_and_preserves_outside(exchange) -> None:
    first, second = Time("1", "1", 0, 1), Time("1", "1", 1, 1)
    if exchange == "room":
        classes = (
            Class("A", "CA", "1", "1", None, None, (first,), (("1", 10), ("2", 0)), False),
            Class("B", "CB", "1", "1", None, None, (second,), (("2", 10), ("1", 0)), False),
        )
        target = [0, 0, 0, 1, 1, 0]
    elif exchange == "time":
        classes = (
            Class("A", "CA", "1", "1", None, None,
                  (replace(first, penalty=10), second), (("1", 0),), False),
            Class("B", "CB", "1", "1", None, None,
                  (replace(second, penalty=10), first), (("2", 0),), False),
        )
        target = [1, 1, 0, 0, 0, 0]
    else:
        classes = (
            Class("A", "CA", "1", "1", None, None,
                  (replace(first, penalty=10), second), (("1", 10), ("2", 0)), False),
            Class("B", "CB", "1", "1", None, None,
                  (replace(second, penalty=10), first), (("2", 10), ("1", 0)), False),
        )
        target = [1, 1, 0, 1, 1, 0]
    classes += (Class("C", "CC", "1", "1", None, None, (first,), (("3", 0),), False),)
    rooms = {str(i): Room(str(i), None, (), {}) for i in (1, 2, 3)}
    if exchange == "full":
        # Both partial swaps would place a class in the unavailable room.
        rooms["1"] = replace(rooms["1"], unavailable=(second,))
    instance = Instance("guided-swap", 1, 1, 2, (1, 1, 1, 1),
                        rooms, classes, {}, (), (), {"A": 0, "B": 1, "C": 2})
    source = Population.new("X", np.zeros((1, 6), dtype=np.int32))
    problem, evaluator = PlacementProblem(instance), Evaluator()
    evaluator.eval(problem, source)
    algorithm = SimpleNamespace(pop=source, evaluator=evaluator, problem=problem)
    if exchange == "time":
        region = RegionCandidate("0", "r", "b", (0, 1), ("A", "B"), ("C",), (),
                                 0, 20, (), (), 2, 1, ("swap",))
    else:
        region, = RegionBuilder(instance).build(algorithm, max_regions=1, region_size=2)
        assert region.class_ids == ("A", "B") and "swap" in region.available_operators

    probe = probe_operator(algorithm, instance, region, "swap", 1, np.random.default_rng(17), None)

    assert probe.evaluations == 1 and evaluator.n_eval == 2
    assert probe.candidate_quality == (0, 0)
    assert probe.candidate[0].X.tolist() == target
    assert source[0].X.tolist() == [0, 0, 0, 0, 0, 0]


def test_guided_swap_resolves_two_class_eviction_handoff() -> None:
    first = Time("1", "1", 0, 1)
    # Class A collides with fixed class C in room 1 and can only move into room 2,
    # which is currently occupied by class B; B cannot use room 1, so symmetric
    # direct swap_options() is empty and single-class moves cannot reduce hard.
    classes = (
        Class("A", "CA", "1", "1", None, None, (first,), (("1", 0), ("2", 0)), False),
        Class("B", "CB", "1", "1", None, None, (first,), (("2", 0), ("3", 0)), False),
        Class("C", "CC", "1", "1", None, None, (first,), (("1", 0),), False),
    )
    rooms = {str(i): Room(str(i), None, (), {}) for i in (1, 2, 3)}
    instance = Instance("swap-eviction", 1, 1, 1, (1, 1, 1, 1),
                        rooms, classes, {}, (), (), {"A": 0, "B": 1, "C": 2})
    source = Population.new("X", np.zeros((1, 6), dtype=np.int32))
    problem, evaluator = PlacementProblem(instance), Evaluator()
    evaluator.eval(problem, source)
    algorithm = SimpleNamespace(pop=source, evaluator=evaluator, problem=problem)

    region, = RegionBuilder(instance).build(algorithm, max_regions=1, region_size=3)
    assert region.class_ids == ("A", "B", "C")
    assert "swap" in region.available_operators

    move_probe = probe_operator(algorithm, instance, region, "move", 4, np.random.default_rng(17), None)
    assert move_probe.candidate_quality is None

    swap_probe = probe_operator(algorithm, instance, region, "swap", 1, np.random.default_rng(17), None)
    assert swap_probe.evaluations == 1
    assert swap_probe.candidate_quality == (0, 0)
    assert swap_probe.candidate[0].X.tolist() == [0, 0, 0, 1, 1, 0]


def test_seeded_cv_tournament_breaks_infeasible_cv_ties_by_soft_total() -> None:
    pop = Population.new(
        "CV", np.asarray([[5.0], [5.0]]),
        "F", np.asarray([[10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]]),
    )
    pairs = np.asarray([[0, 1]] * 10 + [[1, 0]] * 10)
    winners = _seeded_cv_tournament(pop, pairs, np.random.default_rng(17))
    assert np.all(winners == 1)


def test_timetable_survival_keeps_same_hard_soft_improvement_and_unique_clones() -> None:
    instance = _chain_instance()
    problem = PlacementProblem(instance)
    survival = TimetableSurvival(reference_directions(Config(population=4, partitions=1)))

    # 4 identical infeasible clones at H=2, total=40 plus 1 candidate at H=2, total=20
    clones = np.zeros((4, 6), dtype=np.int32)
    cand = np.asarray([[0, 1, 0, 0, 0, 0]], dtype=np.int32)
    pop = Population.new("X", clones, "G", np.full((4, 1), 2.0), "F", np.full((4, 4), 10.0))
    cand_pop = Population.new("X", cand, "G", np.array([[2.0]]), "F", np.full((1, 4), 5.0))
    survived = survival.do(problem, Population.merge(pop, cand_pop), n_survive=4,
                           random_state=np.random.default_rng(17))
    assert any(ind is cand_pop[0] for ind in survived)
    assert survived[0] is cand_pop[0]

    # Feasible stage: unique chromosomes should survive before identical lower/equal clones.
    x_feas = np.asarray([[0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1]], dtype=np.int32)
    f_feas = np.asarray([[10, 0, 0, 0], [10, 0, 0, 0], [10, 0, 0, 0],
                         [20, 0, 0, 0], [30, 0, 0, 0]], dtype=float)
    feas_pop = Population.new("X", x_feas, "G", np.zeros((5, 1)), "F", f_feas)
    kept_feas = survival.do(problem, feas_pop, n_survive=3,
                            random_state=np.random.default_rng(17))
    unique_chroms = {tuple(map(int, ind.X)) for ind in kept_feas}
    assert len(unique_chroms) == 3
