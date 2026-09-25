"""One OpenAI decision chooses WHERE, HOW and WHEN at each review."""

from __future__ import annotations

from pathlib import Path
import sys


_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "NSGA-III"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from itc2019.controller import ActionPlan, ControllerCallback


class SingleAgentController(ControllerCallback):
    mode = "single_agent"

    def decide(self, packet: dict) -> tuple[ActionPlan | None, list[dict], str]:
        raw, record = self.client.request("single_agent", packet, self.remaining())
        if raw is None:
            return None, [record], record["status"]
        try:
            plan = ActionPlan(**raw)
        except (TypeError, ValueError):
            return None, [record], "malformed_action"
        return plan, [record], "decision_received"
