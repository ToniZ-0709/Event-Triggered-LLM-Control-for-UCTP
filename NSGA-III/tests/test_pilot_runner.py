"""Regression checks for the pilot command construction."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_pilot import commands
from itc2019.nsga3 import Config


def test_runner_applies_instance_overrides_once_for_every_pipeline():
    profile = {"schema_version": 1, "solver": {"population": 20, "partitions": 3},
               "repair": {"near_feasible_hard": 12}, "controller": {"max_reviews": 4},
               "instances": {"X": {"population": 12, "partitions": 2, "evaluations": 1000,
                                    "repair_near_feasible_hard": 40}}}
    for method in ("baseline", "single", "multi"):
        command = commands(profile, pipeline=method, seed=17, instances=["X"], model="fake", output=None)[0]
        assert command[command.index("--population") + 1] == "12"
        assert command[command.index("--partitions") + 1] == "2"
        assert command.count("--repair-near-feasible-hard") == 1
        assert command[command.index("--repair-near-feasible-hard") + 1] == "40"


def test_pilot_source_inventory_matches_solver_run_metadata(tmp_path):
    solver_main = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("phase1_solver_main", solver_main)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = tmp_path / "instance.xml"
    instance.write_text("<problem/>", encoding="utf-8")
    args = SimpleNamespace(author="test", institution="test", country="test")
    sources = module._run_settings([instance], Config(), args)["solver_sources_sha256"]
    assert "main.py" in sources
    assert "itc2019/controller.py" in sources
    assert all((solver_main.parent / name).is_file() for name in sources)
