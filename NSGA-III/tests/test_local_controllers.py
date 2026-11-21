"""Exercise local policy decisions, scoring budgets and offline CLI exports."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "NSGA-III"))
sys.path.insert(0, str(ROOT))

from itc2019 import llm
from itc2019.controller import ControllerSettings
from itc2019.instance import load_instance
from itc2019.nsga3 import PlacementProblem
from itc2019.regions import RegionCandidate
from run_experiment import commands
from tools.audit_runs import audit_attempt


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.parent.name + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RandomController = load_module(ROOT / "Random Controller/controller.py").RandomController
GreedyController = load_module(ROOT / "Greedy Controller/controller.py").GreedyController
SETTINGS = ControllerSettings(model=None, provider="none", max_api_calls=0)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "tiny.xml"
    options = ('<room id="1" penalty="0"/><room id="2" penalty="1"/>'
               '<time days="1" weeks="1" start="0" length="1" penalty="5"/>'
               '<time days="1" weeks="1" start="1" length="1" penalty="0"/>')
    path.write_text(
        '<problem name="tiny" nrDays="1" nrWeeks="1" slotsPerDay="4">'
        '<optimization time="1" room="1" distribution="1" student="1"/>'
        '<rooms><room id="1"/><room id="2"/></rooms><courses>' +
        ''.join(f'<course id="{i}"><config id="{i}"><subpart id="{i}">'
                f'<class id="{i}" limit="10">{options}</class>'
                '</subpart></config></course>' for i in (1, 2)) +
        '</courses><distributions/><students/></problem>', encoding="utf-8")
    return path


def region(ops=("move", "destroy_repair", "swap")):
    return RegionCandidate("0", "r", "b", (0, 1), ("1", "2"), (), (), 1, 10,
                           (), (), 2, 1, ops)


def test_random_draws_over_pairs_plus_one_no_op_and_leaves_operator_rng_alone(dataset):
    instance = load_instance(dataset)
    first, second = (RandomController(instance, SETTINGS, None, 17, time.perf_counter(), None)
                     for _ in range(2))
    regions = [region(("move", "swap")), replace(region(("move",)), region_id="other")]
    packet = {"candidate_regions": [r.public() for r in regions]}
    rng_before = first.rng.bit_generator.state
    first.current_operator_limit = second.current_operator_limit = 3
    sampled = [first.decide(packet) for _ in range(50)]
    assert sampled == [second.decide(packet) for _ in range(50)]
    expected = [("r", "move"), ("r", "swap"), ("other", "move"), (None, "no_op")]
    assert {roles[0]["chosen_index"] for _, roles, _ in sampled} == set(range(4))
    for plan, roles, _ in sampled:
        assert roles[0]["valid_pairs"] == 3
        assert (plan.region_id, plan.operator_id) == expected[roles[0]["chosen_index"]]
        assert plan.execution_budget == (0 if plan.operator_id == "no_op" else 3)
        assert first._validate(plan, regions)[2] in {"valid_action", "valid_no_op"}
    assert first.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("improvable", [True, False])
@pytest.mark.parametrize("evaluation_budget,region_calls,quotas", [
    (5, 0, [("move", 1), ("swap", 1), ("destroy_repair", 1)]),
    (5, 2, [("move", 1)]),
    (20, 0, [("move", 4), ("swap", 3), ("destroy_repair", 3)]),
])
def test_greedy_uses_one_budget_and_injects_at_most_once(
        dataset, improvable, evaluation_budget, region_calls, quotas):
    instance = load_instance(dataset)
    if not improvable:
        # Start with a zero vector so no trade-off can improve an objective.
        instance = replace(instance, classes=tuple(
            replace(cls, room_options=tuple((room, 0) for room, _ in cls.room_options))
            for cls in instance.classes))
    start = [0, 0, 0, 0] if improvable else [1, 1, 0, 1]
    pop = Population.new("X", np.array([start]))
    problem = PlacementProblem(instance, evaluation_budget=evaluation_budget)
    evaluator = Evaluator()
    evaluator.eval(problem, pop)
    survival_calls = []

    def survival(problem, population, **kwargs):
        survival_calls.append(population)
        return population[-1:]

    algorithm = SimpleNamespace(
        problem=problem, pop=pop, evaluator=evaluator,
        survival=SimpleNamespace(do=survival), random_state=np.random.default_rng(42),
        _set_optimum=lambda: None, n_gen=5, ref_dirs=np.empty((1, 4)),
    )
    controller = GreedyController(instance, SETTINGS, None, 17, time.perf_counter(), None)
    # Isolate controller scoring from routine solver repairs; the operators are real.
    controller._maybe_coordinated_repair = lambda algorithm: None
    controller._maybe_room_plateau_repair = lambda algorithm: None
    controller._maybe_room_chain_repair = lambda algorithm: None
    selected = region()

    def build_regions(*args):
        for _ in range(region_calls):
            problem.score_state(pop[0].X)
            controller.region_builder.scoring_calls += 1
        return [replace(selected, region_id="soft", hard_contribution=0, soft_contribution=100),
                selected]

    controller.region_builder.build = build_regions
    controller.notify(algorithm)
    event, = controller.events
    role, = event["roles"]
    assert role["region_id"] == selected.region_id
    assert [(p["operator_id"], p["quota"]) for p in role["probes"]] == quotas
    used = event["probe_evaluations"]
    assert 0 < used <= sum(quota for _, quota in quotas)
    assert evaluator.n_eval == problem.engine.requests <= problem.evaluation_budget
    assert evaluator.n_eval == 1 + region_calls + used + int(improvable)
    assert len(survival_calls) == int(improvable)
    if improvable:
        best = min((p for p in role["probes"] if p["candidate_quality"] is not None),
                   key=lambda p: p["candidate_quality"])
        assert event["action"]["operator_id"] == best["operator_id"]
        outcome = event["outcome"]
        assert (outcome["candidate_hard"], outcome["candidate_total"]) == best["candidate_quality"]
        assert outcome["candidate_hard"] == 0 and outcome["scoring_rechecks"] == 1
    else:
        assert event["decision_status"] == "no_improving_candidate"
        assert event["action"]["operator_id"] == "no_op" and event["outcome"] is None
        assert algorithm.pop is pop
        assert controller.report()["fallback_no_op"] == 0
    assert controller.report()["probe_evaluations"] == used


@pytest.mark.parametrize("worse", [[0, 0, 0, 1], [0, 0, 0, 0]])
def test_greedy_prioritizes_the_better_source_over_a_more_expensive_region(dataset, worse):
    instance = load_instance(dataset)
    pop = Population.new("X", np.array([[1, 0, 0, 0], worse]))
    problem = PlacementProblem(instance)
    evaluator = Evaluator()
    evaluator.eval(problem, pop)
    algorithm = SimpleNamespace(problem=problem, pop=pop, evaluator=evaluator)
    controller = GreedyController(instance, SETTINGS, None, 17, time.perf_counter(), None)
    controller.current_operator_limit = 4
    best = replace(region(("move",)), hard_contribution=0, soft_contribution=5)
    other = replace(best, solution_id="1", region_id="r-worse", soft_contribution=10,
                    hard_contribution=int(pop[1].G[0]))

    decision = controller._review(algorithm, {}, [other, best])

    assert decision.plan.solution_id == "0"
    assert decision.probe.candidate_quality == (0, 1)


def test_greedy_proposes_a_feasible_pareto_tradeoff(dataset):
    instance = load_instance(dataset)
    first, second = instance.classes
    first = replace(first, times=(replace(first.times[0], penalty=1), first.times[1]),
                    room_options=(("1", 0), ("2", 7)))
    instance = replace(instance, classes=(first, second))
    pop = Population.new("X", np.array([[0, 1, 0, 0]]))
    problem = PlacementProblem(instance)
    evaluator = Evaluator()
    evaluator.eval(problem, pop)
    algorithm = SimpleNamespace(problem=problem, pop=pop, evaluator=evaluator)
    controller = GreedyController(instance, SETTINGS, None, 17, time.perf_counter(), None)
    controller.current_operator_limit = 4
    target = replace(region(("destroy_repair",)), class_indices=(0,), class_ids=("1",),
                     hard_contribution=0)
    decision = controller._review(algorithm, {}, [target])
    assert decision.plan.operator_id == "destroy_repair"
    assert decision.probe.source_hard == 0 and decision.probe.source_total == 1
    assert decision.probe.candidate_quality == (0, 7)
    assert decision.probe_evaluations <= 4 and algorithm.pop is pop


@pytest.mark.parametrize("method, factory", [("random", RandomController), ("greedy", GreedyController)])
def test_local_cli_exports_auditable_zero_api_run(dataset, tmp_path, monkeypatch, method, factory):
    def unexpected_api(*args, **kwargs):
        pytest.fail("Offline controller attempted to create a Gemini client")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "GeminiDecisionClient", unexpected_api)
    monkeypatch.setattr(llm, "create_decision_client", unexpected_api)
    profile = {"schema_version": 1,
               "instances": {"tiny": {"evaluations": 60}},
               "solver": {"population": 4, "partitions": 1, "generations": 8,
                          "initial_repair_steps": 0, "offspring_repair_steps": 0},
               "controller": {"max_reviews": 2, "api_timeout_seconds": 5, "max_output_tokens": 512},
               "llm": {"model": "ignored"}}
    command, = commands(profile, pipeline=method, seed=17, instances=["tiny"],
                        model=None, output=tmp_path / "runs")
    assert "--model" not in command and "--api-timeout-seconds" not in command
    monkeypatch.setattr(sys, "argv", command[1:] + ["--dataset", str(dataset.parent)])
    entry = Path(command[1])
    assert entry.is_file()
    cli = load_module(ROOT / "NSGA-III/main.py")
    cli.main(controller_factory=factory, pipeline_name="NSGA-III + " + method.capitalize())
    folder, = (tmp_path / "runs" / method).glob("run_*")
    manifest = json.loads((folder / "run_config.json").read_text())
    record, = json.loads((folder / "summary.json").read_text())
    report = record["controller"]
    assert manifest["completion_status"] == report["completion_status"] == "completed"
    assert "llm" not in manifest["settings"] and report["comparison_eligible"]
    assert report["reviews"] > 0 and report["api_calls"] == report["total_native_tokens"] == 0
    check = audit_attempt({"instance": "tiny", "seed": 17, "method": method, "files": []}, folder)
    assert check["passed"], check
