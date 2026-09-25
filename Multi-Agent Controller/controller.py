"""Inspector chooses WHERE; Planner chooses HOW and WHEN on the same packet."""

from __future__ import annotations

from pathlib import Path
import sys


_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "NSGA-III"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from itc2019.controller import ActionPlan, ControllerCallback


class MultiAgentController(ControllerCallback):
    mode = "multi_agent"

    def decide(self, packet: dict) -> tuple[ActionPlan | None, list[dict], str]:
        raw, inspector_record = self.client.request("inspector", packet, self.remaining())
        records = [inspector_record]
        if raw is None:
            return None, records, "inspector_" + inspector_record["status"]
        target = (raw.get("solution_id"), raw.get("region_id"), raw.get("boundary_id"))
        if target == (None, None, None):
            return self._fallback(), records, "inspector_no_target"
        matching = next((region for region in packet["candidate_regions"]
                         if target == (region["solution_id"], region["region_id"],
                                       region["boundary_id"])), None)
        if matching is None:
            return None, records, "invalid_inspector_target"
        constraint_ids = raw.get("relevant_constraint_ids")
        if (not isinstance(constraint_ids, list) or
                any(not isinstance(value, str) or value not in matching["constraint_ids"]
                    for value in constraint_ids)):
            return None, records, "invalid_inspector_constraints"
        if self.remaining() is not None and self.remaining() <= 0.1:
            return self._fallback(), records, "deadline_before_planner"
        if self.api_calls + len(records) >= self.settings.max_api_calls:
            return self._fallback(), records, "call_budget_before_planner"
        planner_payload = {"state": packet, "diagnosis": raw}
        planned, planner_record = self.client.request("planner", planner_payload,
                                                      self.remaining())
        records.append(planner_record)
        if planned is None:
            return None, records, "planner_" + planner_record["status"]
        try:
            operator_id = planned["operator_id"]
            ids = (None, None, None) if operator_id == "no_op" else target
            plan = ActionPlan(*ids, operator_id, planned["execution_budget"],
                              planned["next_review_interval"])
        except (KeyError, TypeError, ValueError):
            return None, records, "malformed_planner_action"
        return plan, records, "decision_received"
