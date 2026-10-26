"""Regression checks for bounded repair and controller acceptance."""

from pathlib import Path
from dataclasses import replace
import importlib.util
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.population import Population

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itc2019.instance import Class, Config as ITCConfig, Course, Distribution, Instance, Room, Student, Time
from itc2019.controller import BaselineMonitor, ControllerCallback, ControllerSettings, _progress_signal
from itc2019.nsga3 import (Config, _improve_hard, _joint_improve_hard, _room_chain_repair,
                          _seeded_cv_tournament, evaluation_count, reference_directions, run)
from itc2019.neighborhood import room_neighborhood_repair
from itc2019.repair_policy import FailureBackoff
from itc2019.operators import _admissible_candidate
from itc2019.regions import RegionBuilder, placement_diversity
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


def test_student_pair_cache_preserves_conflict_and_non_conflict_scores() -> None:
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


def test_controller_acceptance_preserves_hard_priority() -> None:
    assert _admissible_candidate(2, 100, 1, 10000)
    assert not _admissible_candidate(2, 100, 2, 1)
    assert not _admissible_candidate(0, 100, 1, 1)
    assert not _admissible_candidate(0, 100, 0, 100)
    assert _admissible_candidate(0, 100, 0, 99)


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
        region = next((item for item in regions if item["available_actions"]["operators"]), None)
        target = {} if region is None else region["target"]
        if role == "inspector":
            decision = ({"solution_id": target["solution_id"],
                         "region_id": target["region_id"],
                         "boundary_id": target["boundary_id"],
                         "relevant_constraint_ids": []} if region else
                        {"solution_id": None, "region_id": None,
                         "boundary_id": None, "relevant_constraint_ids": []})
        elif role == "planner":
            decision = {"operator_id": "move" if region else "no_op",
                        "execution_budget": 1 if region else 0,
                        "next_review_interval": 2}
        else:
            decision = {"solution_id": target.get("solution_id"),
                        "region_id": target.get("region_id"),
                        "boundary_id": target.get("boundary_id"),
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


def test_budget_horizon_never_claims_an_unmeasured_1200_evaluations() -> None:
    from solver_checks.select_pilot_budget import horizon
    record = {"config": {"population": 20}, "trace": [
        {"evaluations": 100, "best_observed_hard": 10},
        {"evaluations": 500, "best_observed_hard": 4},
        {"evaluations": 850, "best_observed_hard": 4},
    ]}
    assert horizon(record) == 800


def test_budget_horizon_keeps_search_after_first_feasibility() -> None:
    from solver_checks.select_pilot_budget import horizon
    record = {"config": {"population": 20}, "trace": [
        {"evaluations": 100, "best_observed_hard": 2},
        {"evaluations": 500, "best_observed_hard": 0, "best_observed_internal_total": 1000},
        {"evaluations": 1000, "best_observed_hard": 0, "best_observed_internal_total": 500},
        {"evaluations": 2000, "best_observed_hard": 0, "best_observed_internal_total": 400},
    ]}
    assert horizon(record) == 2000
