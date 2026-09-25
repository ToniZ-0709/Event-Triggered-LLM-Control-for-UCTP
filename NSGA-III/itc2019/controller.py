"""Shared review gate, action validation and monitoring for all ITC 2019 pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any

import numpy as np
from pymoo.core.callback import Callback

from .instance import Instance
from .operators import OPERATORS, apply_operator
from .regions import RegionBuilder, RegionCandidate, population_metrics


@dataclass(frozen=True, slots=True)
class ControllerSettings:
    model: str
    review_interval: int = 5
    warmup_generations: int = 5
    stagnation_window: int = 5
    min_review_interval: int = 2
    max_review_interval: int = 10
    diversity_threshold: float = 0.25
    max_regions: int = 6
    region_size: int = 3
    max_operator_evaluations: int = 8
    recent_outcomes: int = 5
    api_timeout_seconds: float = 30.0
    max_output_tokens: int = 300
    max_api_calls: int = 100

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("A model is required for controller pipelines")
        positive = (self.review_interval, self.warmup_generations,
                    self.stagnation_window, self.min_review_interval,
                    self.max_review_interval, self.max_regions, self.region_size,
                    self.max_operator_evaluations, self.recent_outcomes,
                    self.max_output_tokens, self.max_api_calls)
        if any(value < 1 for value in positive) or self.api_timeout_seconds <= 0:
            raise ValueError("Controller settings must be positive")
        if not self.min_review_interval <= self.review_interval <= self.max_review_interval:
            raise ValueError("Review interval must lie within min/max interval")
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("Diversity threshold must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ActionPlan:
    solution_id: str | None
    region_id: str | None
    boundary_id: str | None
    operator_id: str
    execution_budget: int
    next_review_interval: int


class BaselineMonitor(Callback):
    """Measure the same internal anytime signal in all three methods."""

    def __init__(self, instance: Instance, started: float,
                 time_limit_seconds: float | None) -> None:
        super().__init__()
        self.instance = instance
        self.started = started
        self.deadline = None if time_limit_seconds is None else started + time_limit_seconds
        self.anytime_events: list[dict[str, Any]] = []
        self.best_choices: np.ndarray | None = None
        self.best_quality: tuple[int, int] | None = None

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.perf_counter()

    def notify(self, algorithm) -> None:
        metrics = population_metrics(
            algorithm, time.perf_counter() - self.started, self.remaining())
        hard = np.asarray(algorithm.pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(algorithm.pop.get("F"), dtype=float).sum(axis=1)
        best_index = min(range(len(algorithm.pop)), key=lambda i: (hard[i], totals[i], i))
        quality = (int(hard[best_index]), int(totals[best_index]))
        if self.best_quality is None or quality < self.best_quality:
            self.best_quality = quality
            self.best_choices = np.asarray(algorithm.pop[best_index].X, dtype=np.int32).copy()
        metrics["best_observed_hard"] = self.best_quality[0]
        metrics["best_observed_internal_total"] = self.best_quality[1]
        self.anytime_events.append(metrics)
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            algorithm.termination.terminate()
            algorithm.termination.update(algorithm)

    def report(self) -> dict[str, Any]:
        return {"generations_recorded": len(self.anytime_events),
                "time_limit_seconds": None if self.deadline is None else
                round(self.deadline - self.started, 3),
                "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                "best_observed_hard": None if self.best_quality is None else self.best_quality[0],
                "best_observed_internal_total": None if self.best_quality is None else self.best_quality[1]}


class ControllerCallback(BaselineMonitor):
    """One reviewed intervention at a time; the solver owns every state change."""

    mode = "unspecified"

    def __init__(self, instance: Instance, settings: ControllerSettings, client,
                 seed: int, started: float, time_limit_seconds: float | None) -> None:
        super().__init__(instance, started, time_limit_seconds)
        self.settings = settings
        self.client = client
        self.region_builder = RegionBuilder(instance)
        seed_bytes = hashlib.sha256(f"{instance.name}:{seed}:operators".encode()).digest()
        self.rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
        self.next_review = settings.warmup_generations
        self.last_review = -settings.min_review_interval
        self.events: list[dict[str, Any]] = []
        self.outcomes: list[dict[str, Any]] = []
        self.quality_history: list[tuple[int, int]] = []
        self.best_observed: tuple[int, int] | None = None
        self.api_calls = self.input_tokens = self.output_tokens = 0
        self.api_latency_seconds = 0.0
        self.api_disabled = False
        self.interventions = self.selected_interventions = 0
        self.operator_scoring_rechecks = 0

    def decide(self, packet: dict) -> tuple[ActionPlan | None, list[dict], str]:
        raise NotImplementedError

    def _fallback(self) -> ActionPlan:
        return ActionPlan(None, None, None, "no_op", 0, self.settings.review_interval)

    def _validate(self, plan: ActionPlan | None,
                  regions: list[RegionCandidate]) -> tuple[ActionPlan, RegionCandidate | None, str]:
        if plan is None:
            return self._fallback(), None, "missing_decision"
        s = self.settings
        if type(plan.next_review_interval) is not int or not (
                s.min_review_interval <= plan.next_review_interval <= s.max_review_interval):
            return self._fallback(), None, "invalid_review_interval"
        if plan.operator_id == "no_op":
            if (plan.solution_id is not None or plan.region_id is not None or
                    plan.boundary_id is not None or type(plan.execution_budget) is not int or
                    plan.execution_budget != 0):
                return self._fallback(), None, "invalid_no_op"
            return plan, None, "valid_no_op"
        if plan.operator_id not in OPERATORS:
            return self._fallback(), None, "unknown_operator"
        if type(plan.execution_budget) is not int or not (
                1 <= plan.execution_budget <= s.max_operator_evaluations):
            return self._fallback(), None, "invalid_operator_budget"
        for region in regions:
            if (plan.solution_id, plan.region_id, plan.boundary_id) == (
                    region.solution_id, region.region_id, region.boundary_id):
                if plan.operator_id not in region.available_operators:
                    return self._fallback(), None, "operator_unavailable_in_region"
                return plan, region, "valid_action"
        return self._fallback(), None, "unknown_region_or_boundary"

    def _trigger(self, metrics: dict) -> list[str]:
        generation = metrics["generation"]
        if generation < self.settings.warmup_generations or (
                generation - self.last_review < self.settings.min_review_interval):
            return []
        reasons = []
        if generation >= self.next_review:
            reasons.append("deadline")
        if (len(self.quality_history) >= self.settings.stagnation_window and
                self.quality_history[-1] ==
                self.quality_history[-self.settings.stagnation_window]):
            reasons.append("stagnation")
        if metrics["decision_diversity"] <= self.settings.diversity_threshold:
            reasons.append("low_diversity")
        return reasons

    def _packet(self, metrics: dict, regions: list[RegionCandidate]) -> dict:
        s = self.settings
        return {
            "schema_version": 1,
            "instance": self.instance.name,
            "global": metrics,
            "recent_best_quality": self.quality_history[-s.stagnation_window:],
            "candidate_regions": [region.public() for region in regions],
            "recent_outcomes": self.outcomes[-s.recent_outcomes:],
            "allowed_actions": {
                "operators": [*OPERATORS, "no_op"],
                "execution_budget": [1, s.max_operator_evaluations],
                "review_interval": [s.min_review_interval, s.max_review_interval],
            },
        }

    def _account_calls(self, role_records: list[dict]) -> None:
        for record in role_records:
            self.api_calls += 1
            self.input_tokens += record.get("input_tokens") or 0
            self.output_tokens += record.get("output_tokens") or 0
            self.api_latency_seconds += record.get("latency_seconds") or 0.0
            if record.get("status") == "api_configuration_error":
                self.api_disabled = True

    def notify(self, algorithm) -> None:
        metrics = population_metrics(algorithm, time.perf_counter() - self.started,
                                     self.remaining())
        quality = (metrics["best_hard"], metrics["best_internal_total"])
        self.best_observed = min(self.best_observed, quality) if self.best_observed else quality
        self.quality_history.append(self.best_observed)
        reasons = self._trigger(metrics)
        if reasons and (self.remaining() is None or self.remaining() > 0.1) and (
                self.api_calls < self.settings.max_api_calls) and not self.api_disabled:
            regions = self.region_builder.build(
                algorithm, self.settings.max_regions, self.settings.region_size)
            packet = self._packet(metrics, regions)
            decision, role_records, decision_status = self.decide(packet)
            self._account_calls(role_records)
            plan, region, validation = self._validate(decision, regions)
            outcome = None
            if region is not None and plan.operator_id != "no_op":
                self.interventions += 1
                outcome = apply_operator(
                    algorithm, self.instance, region, plan.operator_id,
                    plan.execution_budget, self.rng, self.deadline)
                self.selected_interventions += int(outcome.selected)
                self.operator_scoring_rechecks += outcome.scoring_rechecks
            after = population_metrics(algorithm, time.perf_counter() - self.started,
                                       self.remaining())
            after_quality = (after["best_hard"], after["best_internal_total"])
            self.best_observed = min(self.best_observed, after_quality)
            self.quality_history[-1] = self.best_observed
            event = {
                "generation": metrics["generation"], "trigger": reasons,
                "state": packet, "roles": role_records,
                "decision_status": decision_status, "validation": validation,
                "action": asdict(plan),
                "outcome": None if outcome is None else asdict(outcome),
                "population_after": {"best_hard": after["best_hard"],
                                     "best_internal_total": after["best_internal_total"]},
            }
            self.events.append(event)
            self.outcomes.append({"generation": metrics["generation"],
                                  "region_id": plan.region_id,
                                  "operator_id": plan.operator_id,
                                  "validation": validation,
                                  "outcome": None if outcome is None else outcome.status,
                                  "attempts": 0 if outcome is None else outcome.attempts,
                                  "evaluations": 0 if outcome is None else outcome.evaluations,
                                  "elapsed_seconds": 0 if outcome is None else
                                  round(outcome.elapsed_seconds, 3),
                                  "best_hard_change": after["best_hard"] - metrics["best_hard"],
                                  "best_internal_total_change": (
                                      after["best_internal_total"] - metrics["best_internal_total"])})
            self.last_review = metrics["generation"]
            self.next_review = metrics["generation"] + plan.next_review_interval
        super().notify(algorithm)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base.update({"mode": self.mode,
                     "status": "api_disabled" if self.api_disabled else "active",
                     "model": self.settings.model,
                     "reviews": len(self.events),
                     "no_op": sum(event["action"]["operator_id"] == "no_op"
                                  for event in self.events),
                     "api_calls": self.api_calls,
                     "input_tokens": self.input_tokens,
                     "output_tokens": self.output_tokens,
                     "api_latency_seconds": round(self.api_latency_seconds, 3),
                     "api_disabled_after_configuration_error": self.api_disabled,
                     "interventions": self.interventions,
                     "selected_interventions": self.selected_interventions})
        base["region_scoring_calls"] = self.region_builder.scoring_calls
        base["operator_scoring_rechecks"] = self.operator_scoring_rechecks
        return base
