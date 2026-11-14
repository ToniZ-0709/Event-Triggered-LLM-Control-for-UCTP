"""Uniform random choice among legal region/operator pairs and one no-op."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

BASELINE = Path(__file__).resolve().parents[1] / "NSGA-III"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from itc2019.controller import ActionPlan, ControllerCallback, FIXED_EXECUTION_BUDGET


class RandomController(ControllerCallback):
    mode = "random"
    requires_llm = False

    def __init__(self, instance, settings, client, seed, started, time_limit_seconds,
                 repair_settings=None):
        super().__init__(instance, settings, client, seed, started, time_limit_seconds,
                         repair_settings)
        digest = hashlib.sha256(f"{instance.name}:{seed}:controller-policy".encode()).digest()
        self.policy_rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def decide(self, packet: dict) -> tuple[ActionPlan, list[dict], str]:
        pairs = [(region, operator) for region in packet["candidate_regions"]
                 for operator in region["available_operators"]]
        budget = min(FIXED_EXECUTION_BUDGET, self.current_operator_limit)
        chosen = int(self.policy_rng.integers(len(pairs) + 1)) if pairs and budget > 0 else None
        plan = self._fallback()
        if chosen is not None and chosen < len(pairs):
            region, operator = pairs[chosen]
            plan = ActionPlan(region["solution_id"], region["region_id"], region["boundary_id"],
                              operator, budget, self.settings.review_interval)
        status = "policy_decision" if chosen is not None else "no_legal_action"
        roles = [{"role": "random", "policy": "uniform_random", "valid_pairs": len(pairs),
                  "chosen_index": chosen, "api_requests": 0, "latency_seconds": 0.0,
                  "status": status}]
        return plan, roles, status
