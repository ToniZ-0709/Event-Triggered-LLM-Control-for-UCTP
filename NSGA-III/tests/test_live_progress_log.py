"""Unit and CLI tests for live terminal/file progress logging across pipelines."""

from __future__ import annotations

import importlib.util
import io
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
BASE = ROOT / "NSGA-III"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from itc2019.controller import (
    ActionPlan, BaselineMonitor, ControllerCallback, ControllerSettings,
)
from itc2019.instance import load_instance
from itc2019.nsga3 import Config, PlacementProblem, run
from itc2019.regions import RegionCandidate


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.parent.name + "_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RandomController = _load_module(ROOT / "Random Controller/controller.py").RandomController
GreedyController = _load_module(ROOT / "Greedy Controller/controller.py").GreedyController
LOCAL_SETTINGS = ControllerSettings(model=None, provider="none", max_api_calls=0,
                                    warmup_generations=2, review_interval=2, min_review_interval=1)


@pytest.fixture
def tiny_dataset(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    options = ('<room id="1" penalty="0"/><room id="2" penalty="1"/>'
               '<time days="1" weeks="1" start="0" length="1" penalty="5"/>'
               '<time days="1" weeks="1" start="1" length="1" penalty="0"/>')
    (dataset / "tiny.xml").write_text(
        '<problem name="tiny" nrDays="1" nrWeeks="1" slotsPerDay="4">'
        '<optimization time="1" room="1" distribution="1" student="1"/>'
        '<rooms><room id="1"/><room id="2"/></rooms><courses>' +
        ''.join(f'<course id="{i}"><config id="{i}"><subpart id="{i}">'
                f'<class id="{i}" limit="10">{options}</class>'
                '</subpart></config></course>' for i in (1, 2)) +
        '</courses><distributions/><students/></problem>',
        encoding="utf-8",
    )
    return dataset


def test_baseline_progress_interval_stream_and_file_mirroring(tiny_dataset, tmp_path):
    instance = load_instance(tiny_dataset / "tiny.xml")
    stream = io.StringIO()
    log_file = tmp_path / "logs" / "baseline.log"
    monitor = BaselineMonitor(instance, time.perf_counter(), None, seed=17)
    with pytest.raises(ValueError, match="non-negative"):
        monitor.configure_progress(interval=-1)
    monitor.configure_progress(stream=stream, log_paths=[log_file], interval=2, max_generations=5)
    run(instance, Config(population=4, partitions=1, generations=5, seed=17,
                         initial_repair_steps=0, offspring_repair_steps=0),
        callback=monitor)
    text = stream.getvalue()
    assert text == log_file.read_text(encoding="utf-8")
    gen_lines = [line for line in text.splitlines() if line.startswith("[gen ")]
    assert [line.split("]")[0] for line in gen_lines] == [
        "[gen 1/5", "[gen 2/5", "[gen 4/5", "[gen 5/5",
    ]
    assert all("eval=" in line and "best=" in line and "div=" in line for line in gen_lines)


def test_interval_zero_emits_only_events_not_periodic_generations(tiny_dataset, tmp_path):
    instance = load_instance(tiny_dataset / "tiny.xml")
    stream = io.StringIO()
    controller = RandomController(instance, LOCAL_SETTINGS, None, 17, time.perf_counter(), None)
    controller.configure_progress(stream=stream, interval=0, max_generations=5)
    run(instance, Config(population=4, partitions=1, generations=5, seed=17,
                         initial_repair_steps=0, offspring_repair_steps=0),
        callback=controller)
    lines = stream.getvalue().splitlines()
    assert not any(line.startswith("[gen ") for line in lines)
    assert any(line.startswith("[review #1 @ gen ") and "| random |" in line for line in lines)
    assert any(line.strip().startswith("decision:") and "random[pairs=" in line for line in lines)


@pytest.mark.parametrize("role_mode, roles, expected_tokens", [
    ("single_agent",
     [{"role": "single_agent", "status": "ok", "total_tokens": 3210,
       "pacing_wait_seconds": 1.5, "latency_seconds": 0.82}],
     ["[review #1 @ gen 5 | single_agent |", "op=swap budget=8 next_when=6",
      "single_agent[ok tok=3210 wait=1.5s t=0.82s]"]),
    ("multi_agent",
     [{"role": "inspector", "status": "ok", "total_tokens": 2100, "latency_seconds": 0.50},
      {"role": "planner", "status": "ok", "total_tokens": 1800, "latency_seconds": 0.65}],
     ["[review #1 @ gen 5 | multi_agent |", "op=swap budget=8 next_when=6",
      "inspector[ok tok=2100 t=0.50s] planner[ok tok=1800 t=0.65s]"]),
])
def test_llm_review_lines_summarize_roles_and_survive_partial_records(
        tiny_dataset, role_mode, roles, expected_tokens):
    instance = load_instance(tiny_dataset / "tiny.xml")
    settings = ControllerSettings(model="gemini-3.8-flash", warmup_generations=5,
                                  review_interval=5, min_review_interval=1)

    class FakeLLMController(ControllerCallback):
        mode = role_mode

        def decide(self, packet):
            region = packet["candidate_regions"][0]
            plan = ActionPlan(region["solution_id"], region["region_id"], region["boundary_id"],
                              "swap", 8, 6)
            return plan, roles, "decision_received"

    pop = Population.new("X", np.array([[0, 0, 0, 0], [1, 1, 0, 1]]))
    problem = PlacementProblem(instance, evaluation_budget=50)
    evaluator = Evaluator()
    evaluator.eval(problem, pop)
    algorithm = SimpleNamespace(
        problem=problem, pop=pop, evaluator=evaluator,
        survival=SimpleNamespace(do=lambda p, m, n_survive, **kw: m[:n_survive]),
        random_state=np.random.default_rng(42),
        _set_optimum=lambda: None, n_gen=5, ref_dirs=np.empty((1, 4)),
        termination=SimpleNamespace(force_termination=False),
    )
    stream = io.StringIO()
    controller = FakeLLMController(instance, settings, None, 17, time.perf_counter(), None)
    controller._maybe_coordinated_repair = lambda alg: None
    controller._maybe_room_plateau_repair = lambda alg: None
    controller._maybe_room_chain_repair = lambda alg: None
    controller.configure_progress(stream=stream, interval=1, max_generations=10)
    controller.notify(algorithm)
    text = stream.getvalue()
    for fragment in expected_tokens:
        assert fragment in text
    assert "[gen 5/10]" in text and "stag=" in text

    class PartialRoleController(ControllerCallback):
        mode = "partial"

        def decide(self, packet):
            return None, [{"api_requests": 1, "status": "api_quota_exhausted"}], "api_quota_exhausted"

    partial_stream = io.StringIO()
    partial = PartialRoleController(instance, settings, None, 17, time.perf_counter(), None)
    partial._maybe_coordinated_repair = lambda alg: None
    partial._maybe_room_plateau_repair = lambda alg: None
    partial._maybe_room_chain_repair = lambda alg: None
    partial.configure_progress(stream=partial_stream, interval=1, max_generations=10)
    partial.notify(algorithm)
    assert "llm[api_quota_exhausted]" in partial_stream.getvalue()


@pytest.mark.parametrize("improvable", [True, False])
def test_greedy_progress_logs_probes_and_no_op_outcome(tiny_dataset, improvable):
    instance = load_instance(tiny_dataset / "tiny.xml")
    start = [0, 0, 0, 0] if improvable else [1, 1, 0, 1]
    pop = Population.new("X", np.array([start]))
    problem = PlacementProblem(instance, evaluation_budget=25)
    evaluator = Evaluator()
    evaluator.eval(problem, pop)
    algorithm = SimpleNamespace(
        problem=problem, pop=pop, evaluator=evaluator,
        survival=SimpleNamespace(do=lambda p, m, **kw: m[-1:]),
        random_state=np.random.default_rng(42),
        _set_optimum=lambda: None, n_gen=5, ref_dirs=np.empty((1, 4)),
        termination=SimpleNamespace(force_termination=False),
    )
    controller = GreedyController(
        instance,
        ControllerSettings(model=None, provider="none", max_api_calls=0),
        None, 17, time.perf_counter(), None,
    )
    controller._maybe_coordinated_repair = lambda alg: None
    controller._maybe_room_plateau_repair = lambda alg: None
    controller._maybe_room_chain_repair = lambda alg: None
    controller.region_builder.build = lambda *args: [
        RegionCandidate("0", "0:0", "b", (0, 1), ("1", "2"), (), (), 1, 10, (), (), 2, 1,
                        ("move", "swap", "destroy_repair"))
    ]
    stream = io.StringIO()
    controller.configure_progress(stream=stream, interval=1, max_generations=10)
    controller.notify(algorithm)
    text = stream.getvalue()
    assert "[review #1 @ gen 5 | greedy |" in text
    assert "greedy[region=0:0 probes=" in text
    if improvable:
        assert "outcome: selected best=" in text and "rechecks=1" in text
    else:
        assert "op=no_op" in text and "no_improving_candidate" in text
        assert "probe_evals=" in text


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_keeps_stdout_json_clean_and_writes_live_run_log(
        tiny_dataset, tmp_path, monkeypatch, capsys, quiet):
    cli = _load_module(BASE / "main.py")
    output = tmp_path / "runs"
    extra_log = tmp_path / "live" / "custom.log"
    args = [
        "main.py", "--dataset", str(tiny_dataset), "--instance", "tiny",
        "--output", str(output), "--population", "4", "--partitions", "1",
        "--generations", "4", "--initial-repair-steps", "0", "--offspring-repair-steps", "0",
        "--progress-interval", "2", "--progress-log", str(extra_log),
    ]
    if quiet:
        args.append("--quiet")
    monkeypatch.setattr(sys, "argv", args)
    cli.main(controller_factory=GreedyController, pipeline_name="NSGA-III + Greedy")
    captured = capsys.readouterr()

    stdout_lines = [json.loads(line) for line in captured.out.strip().splitlines()]
    assert len(stdout_lines) == 2
    assert stdout_lines[0]["instance"] == "tiny"
    assert "run_id" in stdout_lines[1]

    run_folder, = output.glob("run_*")
    run_log = (run_folder / "tiny" / "run.log").read_text(encoding="utf-8")
    assert run_log == extra_log.read_text(encoding="utf-8")
    assert "[start] pipeline=NSGA-III + Greedy instance=tiny" in run_log
    assert "[done] instance=tiny" in run_log
    assert "[gen 4/4]" in run_log
    if quiet:
        assert captured.err == ""
    else:
        assert captured.err == run_log
