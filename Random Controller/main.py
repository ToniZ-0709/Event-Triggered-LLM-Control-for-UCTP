"""Run ITC 2019 NSGA-III with the local Random Controller."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "NSGA-III"
sys.path.insert(0, str(BASELINE))

from controller import RandomController  # noqa: E402


def main() -> None:
    spec = importlib.util.spec_from_file_location("baseline_cli", BASELINE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the shared NSGA-III CLI")
    baseline_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_cli)
    baseline_cli.main(
        controller_factory=RandomController,
        pipeline_root=ROOT,
        pipeline_name="NSGA-III + Random",
        controller_sources=(ROOT / "main.py", ROOT / "controller.py",
                            BASELINE / "itc2019" / "controller.py",
                            BASELINE / "itc2019" / "regions.py",
                            BASELINE / "itc2019" / "operators.py"),
    )


if __name__ == "__main__":
    main()
