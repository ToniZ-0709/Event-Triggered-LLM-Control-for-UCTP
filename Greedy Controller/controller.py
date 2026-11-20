"""Rank regions by penalty, then spend one shared budget probing legal operators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

BASELINE = Path(__file__).resolve().parents[1] / "NSGA-III"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from itc2019.controller import (
    ActionPlan, ControllerCallback, FIXED_EXECUTION_BUDGET, ReviewDecision,
)
from itc2019.operators import OPERATORS, probe_operator
from itc2019.regions import RegionCandidate


class GreedyController(ControllerCallback):
    mode = "greedy"
    requires_llm = False

    def _review(self, algorithm, packet: dict, regions: list[RegionCandidate]) -> ReviewDecision:
        budget = min(FIXED_EXECUTION_BUDGET, self.current_operator_limit)
        role = {"role": "greedy", "policy": "greedy", "api_requests": 0,
                "latency_seconds": 0.0, "probes": []}

        def finish(status, evaluations=0, plan=None, probe=None):
            role["status"] = status
            return ReviewDecision(plan or self._fallback(), [role], status, probe, evaluations)

        eligible = [r for r in regions if r.available_operators]
        if not eligible or budget < 1:
            return finish("no_legal_action")
        penalized = [r for r in eligible if r.hard_contribution or r.soft_contribution
                     or r.unlocalized_penalties]
        if not penalized:
            return finish("no_region_penalty")

        def rank(region):
            individual = algorithm.pop[int(region.solution_id)]
            hard = int(np.asarray(individual.G).reshape(-1)[0])
            total = int(np.asarray(individual.F).sum())
            return (hard, total, -region.hard_contribution, -region.soft_contribution,
                    region.region_id)

        region = min(penalized, key=rank)
        role.update(solution_id=region.solution_id, region_id=region.region_id,
                    boundary_id=region.boundary_id, probe_budget=budget)
        ops = [op for op in OPERATORS if op in region.available_operators]
        base, extra = divmod(budget, len(ops))
        started = time.perf_counter()
        best = None
        best_quality = None
        best_trial = None
        attempts = evaluations = 0
        for i, op in enumerate(ops):
            quota = base + int(i < extra)
            if quota == 0 or (self.deadline is not None and time.perf_counter() >= self.deadline):
                break
            probe = probe_operator(algorithm, self.instance, region, op, quota, self.rng,
                                   self.deadline)
            attempts += probe.attempts
            evaluations += probe.evaluations
            role["probes"].append({"operator_id": op, "quota": quota,
                                   "evaluations": probe.evaluations,
                                   "best_trial": probe.best_trial,
                                   "candidate_quality": probe.candidate_quality})
            if probe.best_trial is not None and (best_trial is None or probe.best_trial < best_trial):
                best_trial = probe.best_trial
            # All operators share a source, so minimizing (H, S) maximizes its improvement.
            # On exact ties the first operator in OPERATORS wins.
            if probe.candidate_quality is not None and (
                    best_quality is None or probe.candidate_quality < best_quality):
                best_quality = probe.candidate_quality
                best = op, probe
        if best is None:
            return finish("no_improving_candidate", evaluations)
        best_op, best_probe = best
        plan = ActionPlan(region.solution_id, region.region_id, region.boundary_id,
                          best_op, budget, self.settings.review_interval)
        combined = replace(best_probe, started=started, attempts=attempts,
                           evaluations=evaluations, best_trial=best_trial)
        return finish("policy_decision", evaluations, plan, combined)

    def report(self) -> dict:
        return {**super().report(),
                "probe_evaluations": sum(event["probe_evaluations"] for event in self.events)}
