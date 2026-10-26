"""Shared review gate, action validation and monitoring for all ITC 2019 pipelines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any

import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.population import Population

from .instance import Instance
from .nsga3 import _joint_improve_hard, _room_chain_repair, evaluation_count
from .operators import OPERATORS, apply_operator
from .neighborhood import adaptive_room_repair, distribution_neighborhood_repair
from .repair_policy import FailureBackoff, RepairSettings
from .regions import RegionBuilder, RegionCandidate, population_metrics, expand_blocker_region
from .room_search import room_plateau_repair
from .scoring import evaluate, overlaps
from .state_packet import SCHEMA_VERSION, make_packet, fit_packet, packet_hash


_COORDINATED_INTERVAL = 5
_COORDINATED_PAIR_TRIALS = 8
_COORDINATED_ROUNDS = 3


@dataclass(frozen=True, slots=True)
class ControllerSettings:
    model: str
    review_interval: int = 10
    warmup_generations: int = 5
    stagnation_window: int = 5
    min_review_interval: int = 3
    max_review_interval: int = 20
    diversity_threshold: float = 0.20
    diversity_recovery_threshold: float = 0.30
    max_regions: int = 6
    region_size: int = 5
    max_operator_evaluations: int = 24
    recent_outcomes: int = 5
    api_timeout_seconds: float = 15.0
    max_output_tokens: int = 300
    max_api_calls: int = 100
    max_reviews: int = 12
    provider: str = "gemini"
    enable_blocker_chain: bool = False

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("A model is required for controller pipelines")
        if self.provider != "gemini":
            raise ValueError("Only the Gemini free-tier controller is enabled")
        positive = (self.review_interval, self.warmup_generations,
                    self.stagnation_window, self.min_review_interval,
                    self.max_review_interval, self.max_regions, self.region_size,
                    self.max_operator_evaluations, self.recent_outcomes,
                    self.max_output_tokens, self.max_api_calls, self.max_reviews)
        if any(value < 1 for value in positive) or self.api_timeout_seconds <= 0:
            raise ValueError("Controller settings must be positive")
        if not self.min_review_interval <= self.review_interval <= self.max_review_interval:
            raise ValueError("Review interval must lie within min/max interval")
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("Diversity threshold must lie in [0,1]")
        if not self.diversity_threshold <= self.diversity_recovery_threshold <= 1:
            raise ValueError("Diversity recovery must be at least the trigger threshold")


@dataclass(frozen=True, slots=True)
class ActionPlan:
    solution_id: str | None
    region_id: str | None
    boundary_id: str | None
    operator_id: str
    execution_budget: int
    next_review_interval: int


def _progress_signal(quality: tuple[int, int]) -> tuple[str, int]:
    return ("hard", quality[0]) if quality[0] > 0 else ("feasible_total", quality[1])


class BaselineMonitor(Callback):
    """Measure the same internal anytime signal in all three methods."""

    def __init__(self, instance: Instance, started: float,
                 time_limit_seconds: float | None, seed: int = 42,
                 repair_settings: RepairSettings | None = None) -> None:
        super().__init__()
        self.instance = instance
        self.started = started
        self.deadline = None if time_limit_seconds is None else started + time_limit_seconds
        self.repair_settings = repair_settings or RepairSettings()
        self.repair_backoff = FailureBackoff(self.repair_settings.failure_cooldown,
                                             self.repair_settings.max_failure_cooldown)
        self.room_repair_backoff_skips = 0
        self.seed = seed
        self.anytime_events: list[dict[str, Any]] = []
        self.best_choices: np.ndarray | None = None
        self.best_quality: tuple[int, int] | None = None
        self.room_chain_events: list[dict[str, Any]] = []
        self.room_chain_scoring_calls = 0
        self._last_room_chain_generation = -2
        self.coordinated_repair_events: list[dict[str, Any]] = []
        self.coordinated_repair_scoring_calls = 0
        self._coordinated_sources: set[str] = set()
        seed_bytes = hashlib.sha256(f"{instance.name}:{seed}:coordinated-repair".encode()).digest()
        self._coordinated_rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
        self.room_plateau_events: list[dict[str, Any]] = []
        self.room_plateau_scoring_calls = 0
        self._room_plateau_sources: set[str] = set()
        seed_bytes = hashlib.sha256(f"{instance.name}:{seed}:room-plateau".encode()).digest()
        self._room_plateau_rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.perf_counter()

    def remaining_evaluations(self, algorithm) -> int | None:
        limit = getattr(algorithm.problem, "evaluation_budget", None)
        return None if limit is None else max(0, limit - evaluation_count(algorithm))

    def _repair_budget_available(self, algorithm) -> bool:
        remaining = self.remaining_evaluations(algorithm)
        return remaining is None or remaining >= 8

    def _sync_incumbent(self, algorithm) -> tuple[int, int]:
        archive = algorithm.problem.incumbent
        if self.best_choices is not None:
            archive.observe(self.best_choices, self.best_quality)
        for individual in algorithm.pop:
            archive.observe(individual.X, (np.asarray(individual.G).reshape(-1)[0],
                                            np.asarray(individual.F).sum()))
        self.best_quality = archive.quality
        self.best_choices = archive.choices.copy()
        return self.best_quality

    def _maybe_coordinated_repair(self, algorithm) -> None:
        """Intensify the best incumbent around linked required constraints."""
        if not self._repair_budget_available(algorithm):
            return
        generation = int(algorithm.n_gen)
        if (generation < _COORDINATED_INTERVAL or
                generation % _COORDINATED_INTERVAL != 0 or
                (self.remaining() is not None and self.remaining() <= 10)):
            return
        hard = np.asarray(algorithm.pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(algorithm.pop.get("F"), dtype=float).sum(axis=1)
        best_index = min(range(len(algorithm.pop)), key=lambda i: (hard[i], totals[i], i))
        source = np.asarray(algorithm.pop[best_index].X, dtype=np.int32).copy()
        if (self.best_choices is not None and self.best_quality is not None and
                self.best_quality < (int(hard[best_index]), int(totals[best_index]))):
            source = self.best_choices.copy()
        if self.best_quality is not None and self.best_quality[0] == 0:
            return
        source_key = hashlib.sha256(source.tobytes()).hexdigest()
        if source_key in self._coordinated_sources:
            return
        self._coordinated_sources.add(source_key)
        n = len(self.instance.classes)
        source_score = algorithm.problem.score_state(source)
        calls = [1]
        if (source_score.hard == 0 or source_score.hard_distributions == 0 or
                source_score.hard_distributions < source_score.room_conflicts):
            algorithm.evaluator.n_eval += calls[0]
            self.coordinated_repair_scoring_calls += calls[0]
            return
        started = time.perf_counter()
        repair_deadline = started + 5.0
        if self.deadline is not None:
            repair_deadline = min(repair_deadline, self.deadline - 3.0)
        working = source.copy()
        rounds = 0
        remaining_calls = self.remaining_evaluations(algorithm)
        candidate, group_stats = distribution_neighborhood_repair(
            self.instance, source, deadline=repair_deadline, seed=self.seed + generation,
            max_classes=self.repair_settings.neighborhood_classes,
            max_options=self.repair_settings.neighborhood_options, source_score=source_score,
            max_full_scoring_calls=min(4, remaining_calls - 2) if remaining_calls is not None else 4,
            score_fn=algorithm.problem.score_state)
        calls[0] += group_stats["full_scoring_calls"]
        if candidate is not None:
            working, rounds = candidate, 1
        for _ in range(0 if candidate is not None else _COORDINATED_ROUNDS):
            if time.perf_counter() >= repair_deadline:
                break
            if not _joint_improve_hard(self.instance, working, self._coordinated_rng,
                                       _COORDINATED_PAIR_TRIALS, calls,
                                       deadline=repair_deadline, score_fn=algorithm.problem.score_state,
                                       max_full_scoring_calls=algorithm.problem.remaining_scores(reserve=1)):
                break
            rounds += 1
        algorithm.evaluator.n_eval += calls[0]
        self.coordinated_repair_scoring_calls += calls[0]
        event = {"generation": generation, "source_hard": source_score.hard,
                 "source_hard_distributions": source_score.hard_distributions,
                 "source_room_conflicts": source_score.room_conflicts,
                 "rounds": rounds, "scoring_calls": calls[0], "neighborhood": group_stats,
                 "elapsed_seconds": round(time.perf_counter() - started, 3),
                 "status": "time_budget_exhausted" if time.perf_counter() >= repair_deadline
                 else "no_improvement"}
        if rounds:
            trial_pop = Population.new("X", working.reshape(1, -1))
            algorithm.evaluator.eval(algorithm.problem, trial_pop, algorithm=algorithm)
            candidate_hard = int(np.asarray(trial_pop.get("G")).reshape(-1)[0])
            event["candidate_hard"] = candidate_hard
            if candidate_hard < source_score.hard:
                candidate_quality = (candidate_hard, int(np.asarray(trial_pop.get("F")).sum()))
                if self.best_quality is None or candidate_quality < self.best_quality:
                    self.best_quality, self.best_choices = candidate_quality, working.copy()
                population_size = len(algorithm.pop)
                algorithm.pop = algorithm.survival.do(
                    algorithm.problem, Population.merge(algorithm.pop, trial_pop),
                    n_survive=population_size, random_state=algorithm.random_state)
                algorithm._set_optimum()
                selected = any(individual is trial_pop[0] for individual in algorithm.pop)
                event["status"] = "selected" if selected else "rejected_by_selection"
            else:
                event["status"] = "scoring_mismatch"
        self.coordinated_repair_events.append(event)

    def _maybe_room_plateau_repair(self, algorithm) -> None:
        """Search bounded neutral moves around room clashes or blocked rooms."""
        if not self._repair_budget_available(algorithm):
            return
        generation = int(algorithm.n_gen)
        if (generation < 5 or generation % 5 != 0 or
                (self.remaining() is not None and self.remaining() <= 10)):
            return
        hard = np.asarray(algorithm.pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(algorithm.pop.get("F"), dtype=float).sum(axis=1)
        best_index = min(range(len(algorithm.pop)), key=lambda i: (hard[i], totals[i], i))
        source = np.asarray(algorithm.pop[best_index].X, dtype=np.int32).copy()
        if (self.best_choices is not None and self.best_quality is not None and
                self.best_quality < (int(hard[best_index]), int(totals[best_index]))):
            source = self.best_choices.copy()
        if (hard[best_index] == 0 or
                self.best_quality is not None and self.best_quality[0] == 0):
            return
        source_key = hashlib.sha256(source.tobytes()).hexdigest()
        if source_key in self._room_plateau_sources:
            return
        self._room_plateau_sources.add(source_key)
        started = time.perf_counter()
        n = len(self.instance.classes)
        source_score = algorithm.problem.score_state(source)
        self.room_plateau_scoring_calls += 1
        algorithm.evaluator.n_eval += 1
        if (source_score.room_conflicts + source_score.room_unavailable < 4 or
                source_score.hard <= self.repair_settings.near_feasible_hard):
            return
        repair_deadline = started + 5.0
        if self.deadline is not None:
            repair_deadline = min(repair_deadline, self.deadline - 3.0)
        candidate, stats = room_plateau_repair(
            self.instance, source, self._room_plateau_rng,
            deadline=repair_deadline, source_score=source_score,
            max_full_scoring_calls=min(16, (self.remaining_evaluations(algorithm) or 18) - 2),
            score_fn=algorithm.problem.score_state)
        self.room_plateau_scoring_calls += stats["full_scoring_calls"]
        algorithm.evaluator.n_eval += stats["full_scoring_calls"]
        event = {"generation": generation, "source_hard": source_score.hard,
                 "source_room_conflicts": source_score.room_conflicts,
                 "source_room_unavailable": source_score.room_unavailable,
                 "source_hard_distributions": source_score.hard_distributions,
                 "search_steps": stats["steps"], "restarts": stats["restarts"],
                 "scoring_calls": 1 + stats["full_scoring_calls"],
                 "elapsed_seconds": round(time.perf_counter() - started, 3),
                 "status": "no_improvement"}
        if candidate is not None:
            trial_pop = Population.new("X", candidate.reshape(1, -1))
            algorithm.evaluator.eval(algorithm.problem, trial_pop, algorithm=algorithm)
            candidate_hard = int(np.asarray(trial_pop.get("G")).reshape(-1)[0])
            event["candidate_hard"] = candidate_hard
            if candidate_hard < source_score.hard:
                candidate_quality = (candidate_hard, int(np.asarray(trial_pop.get("F")).sum()))
                if self.best_quality is None or candidate_quality < self.best_quality:
                    self.best_quality, self.best_choices = candidate_quality, candidate.copy()
                population_size = len(algorithm.pop)
                algorithm.pop = algorithm.survival.do(
                    algorithm.problem, Population.merge(algorithm.pop, trial_pop),
                    n_survive=population_size, random_state=algorithm.random_state)
                algorithm._set_optimum()
                selected = any(individual is trial_pop[0] for individual in algorithm.pop)
                event["status"] = "selected" if selected else "rejected_by_selection"
            else:
                event["status"] = "scoring_mismatch"
        self.room_plateau_events.append(event)

    def _maybe_room_chain_repair(self, algorithm) -> None:
        """Release blocking classes, then use a small beam fallback with cooldown."""
        if not self._repair_budget_available(algorithm):
            return
        settings = self.repair_settings
        generation = int(algorithm.n_gen)
        if (generation < settings.interval or
                generation - self._last_room_chain_generation < settings.interval or
                (self.remaining() is not None and self.remaining() <= 0.5)):
            return
        hard = np.asarray(algorithm.pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(algorithm.pop.get("F"), dtype=float).sum(axis=1)
        best_index = min(range(len(algorithm.pop)), key=lambda i: (hard[i], totals[i], i))
        source = np.asarray(algorithm.pop[best_index].X, dtype=np.int32).copy()
        source_hard = int(hard[best_index])
        if self.best_choices is not None and self.best_quality is not None and (
                self.best_quality < (int(hard[best_index]), int(totals[best_index]))):
            source = self.best_choices.copy()
            source_hard = self.best_quality[0]
        n = len(self.instance.classes)
        selected_times = [cls.times[int(source[index])]
                          for index, cls in enumerate(self.instance.classes)]
        selected_rooms = [None if cls.no_room else
                          cls.room_options[int(source[n + index])][0]
                          for index, cls in enumerate(self.instance.classes)]
        signatures = []
        room_conflicts = 0
        by_room: dict[str, list[int]] = defaultdict(list)
        for right, room_id in enumerate(selected_rooms):
            if room_id is None:
                continue
            if any(overlaps(selected_times[right], block)
                   for block in self.instance.rooms[room_id].unavailable):
                signatures.append(("unavailable", right))
            for left in by_room[room_id]:
                if overlaps(selected_times[left], selected_times[right]):
                    signatures.append(("room", left, right))
                    room_conflicts += 1
            by_room[room_id].append(right)
        if not signatures or source_hard > settings.near_feasible_hard:
            return
        source_key = hashlib.sha256(repr((source_hard, sorted(signatures))).encode()).hexdigest()
        if not self.repair_backoff.ready(source_key, generation):
            self.room_repair_backoff_skips += 1
            return
        self._last_room_chain_generation = generation
        search_started = time.perf_counter()
        local_deadline = search_started + settings.neighborhood_seconds
        if self.deadline is not None:
            local_deadline = min(local_deadline, self.deadline - 0.1)
        candidate, stats = adaptive_room_repair(
            self.instance, source, deadline=local_deadline, seed=self.seed,
            max_classes=settings.neighborhood_classes,
            max_options=settings.neighborhood_options,
            deterministic_limit=settings.neighborhood_deterministic_limit,
            score_fn=algorithm.problem.score_state)
        calls = stats["full_scoring_calls"]
        fallback_calls = 0
        remaining = self.remaining_evaluations(algorithm)
        fallback_budget = settings.fallback_evaluations if remaining is None else min(
            settings.fallback_evaluations, remaining - calls - 1)
        if candidate is None and fallback_budget >= 2:
            fallback_deadline = time.perf_counter() + settings.fallback_seconds
            if self.deadline is not None:
                fallback_deadline = min(fallback_deadline, self.deadline - 0.1)
            candidate, fallback_calls = _room_chain_repair(
                self.instance, source, max_evaluations=fallback_budget,
                max_depth=6, beam_width=12, max_placements_per_class=24,
                hard_slack=2, deadline=fallback_deadline, score_fn=algorithm.problem.score_state)
            calls += fallback_calls
        self.room_chain_scoring_calls += calls
        algorithm.evaluator.n_eval += calls
        event = {"generation": generation, "source_hard": source_hard,
                 "source_room_conflicts": room_conflicts,
                 "scoring_calls": calls,
                 "elapsed_seconds": round(time.perf_counter() - search_started, 3),
                 "neighborhood": stats, "fallback_scoring_calls": fallback_calls,
                 "status": "no_improvement"}
        if candidate is not None:
            trial_pop = Population.new("X", candidate.reshape(1, -1))
            algorithm.evaluator.eval(algorithm.problem, trial_pop, algorithm=algorithm)
            candidate_hard = int(np.asarray(trial_pop.get("G")).reshape(-1)[0])
            event["candidate_hard"] = candidate_hard
            if candidate_hard < source_hard:
                candidate_quality = (candidate_hard, int(np.asarray(trial_pop.get("F")).sum()))
                if self.best_quality is None or candidate_quality < self.best_quality:
                    self.best_quality, self.best_choices = candidate_quality, candidate.copy()
                population_size = len(algorithm.pop)
                algorithm.pop = algorithm.survival.do(
                    algorithm.problem, Population.merge(algorithm.pop, trial_pop),
                    n_survive=population_size, random_state=algorithm.random_state)
                algorithm._set_optimum()
                selected = any(individual is trial_pop[0] for individual in algorithm.pop)
                event["status"] = "selected" if selected else "rejected_by_selection"
            else:
                event["status"] = "scoring_mismatch"
        event["retry_after_generations"] = self.repair_backoff.record(
            source_key, generation, event["status"] in ("selected", "rejected_by_selection"))
        self.room_chain_events.append(event)

    def notify(self, algorithm, *, room_chain_checked: bool = False,
               room_plateau_checked: bool = False,
               coordinated_checked: bool = False) -> None:
        self._sync_incumbent(algorithm)
        if not coordinated_checked:
            self._maybe_coordinated_repair(algorithm)
        if not room_plateau_checked:
            self._maybe_room_plateau_repair(algorithm)
        if not room_chain_checked:
            self._maybe_room_chain_repair(algorithm)
        self._sync_incumbent(algorithm)
        metrics = population_metrics(
            algorithm, time.perf_counter() - self.started, self.remaining())
        metrics["stage_seconds_cumulative"] = {
            name: round(seconds, 3)
            for name, seconds in algorithm.problem.stage_seconds.items()
        }
        metrics["stage_candidates_cumulative"] = dict(algorithm.problem.stage_candidates)
        hard = np.asarray(algorithm.pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(algorithm.pop.get("F"), dtype=float).sum(axis=1)
        best_index = min(range(len(algorithm.pop)), key=lambda i: (hard[i], totals[i], i))
        quality = (int(hard[best_index]), int(totals[best_index]))
        if self.best_quality is None or quality < self.best_quality:
            self.best_quality = quality
            self.best_choices = np.asarray(algorithm.pop[best_index].X, dtype=np.int32).copy()
        metrics["best_observed_hard"] = self.best_quality[0]
        metrics["best_observed_internal_total"] = self.best_quality[1]
        metrics["incumbent_weighted_objectives"] = algorithm.problem.incumbent.weighted
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
                "best_observed_internal_total": None if self.best_quality is None else self.best_quality[1],
                "coordinated_repair_scoring_calls": self.coordinated_repair_scoring_calls,
                "coordinated_repair_events": self.coordinated_repair_events,
                "room_plateau_scoring_calls": self.room_plateau_scoring_calls,
                "room_plateau_events": self.room_plateau_events,
                "room_chain_scoring_calls": self.room_chain_scoring_calls,
                "room_repair_backoff_skips": self.room_repair_backoff_skips,
                "room_chain_events": self.room_chain_events}


class ControllerCallback(BaselineMonitor):
    """One reviewed intervention at a time; the solver owns every state change."""

    mode = "unspecified"

    def __init__(self, instance: Instance, settings: ControllerSettings, client,
                 seed: int, started: float, time_limit_seconds: float | None,
                 repair_settings: RepairSettings | None = None) -> None:
        super().__init__(instance, started, time_limit_seconds, seed, repair_settings)
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
        self.quality_records: list[dict] = []
        self.progress_history: list[tuple[str, int]] = []
        self.last_progress_generation = 0
        self._stagnation_event_handled = False
        self._low_diversity_event_handled = False
        self.current_operator_limit = settings.max_operator_evaluations
        self.budget_skipped_reviews = 0
        self.best_observed: tuple[int, int] | None = None
        self.api_calls = self.input_tokens = self.output_tokens = 0
        self.api_latency_seconds = 0.0
        self.api_disabled = False
        self.api_disabled_reason: str | None = None
        self.api_failures: Counter = Counter()
        self.thinking_tokens = self.total_native_tokens = self.cached_input_tokens = 0
        self.api_pacing_wait_seconds = 0.0
        self.models_returned: set[str] = set()
        self.interventions = self.selected_interventions = 0
        self.operator_scoring_rechecks = 0
        self.diagnostic_seconds = 0.0
        self.diagnostic_local_checks = 0
        self.context_skipped_reviews = 0
        self.unknown_usage_requests = 0

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
                1 <= plan.execution_budget <= self.current_operator_limit):
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
        if metrics["stagnant_generations"] < self.settings.stagnation_window:
            self._stagnation_event_handled = False
        if metrics["placement_diversity"] > self.settings.diversity_recovery_threshold:
            self._low_diversity_event_handled = False
        if generation < self.settings.warmup_generations or (
                generation - self.last_review < self.settings.min_review_interval):
            return []
        reasons = []
        if generation >= self.next_review:
            reasons.append("deadline")
        if (metrics["stagnant_generations"] >= self.settings.stagnation_window and
                not self._stagnation_event_handled):
            reasons.append("stagnation")
        if (metrics["placement_diversity"] <= self.settings.diversity_threshold and
                not self._low_diversity_event_handled):
            reasons.append("low_diversity")
        return reasons

    def _packet(self, metrics: dict, regions: list[RegionCandidate]) -> dict:
        return make_packet(metrics, regions, self.quality_records, self.outcomes,
                           self.settings, self.current_operator_limit, OPERATORS)

    def _account_calls(self, role_records: list[dict]) -> None:
        for record in role_records:
            self.api_calls += record.get("api_requests", 1)
            if record.get("api_requests", 1) and record.get("total_tokens") is None:
                self.unknown_usage_requests += record.get("api_requests", 1)
            if isinstance(record.get("model_returned"), str):
                self.models_returned.add(record["model_returned"])
            self.input_tokens += record.get("input_tokens") or 0
            self.output_tokens += record.get("output_tokens") or 0
            self.thinking_tokens += record.get("thinking_tokens") or 0
            self.total_native_tokens += record.get("total_tokens") or 0
            self.cached_input_tokens += record.get("cached_input_tokens") or 0
            self.api_pacing_wait_seconds += record.get("pacing_wait_seconds") or 0.0
            self.api_latency_seconds += record.get("latency_seconds") or 0.0
            status = record.get("status", "")
            if status.startswith("api_") and status != "api_deadline_skipped":
                self.api_failures[status] += 1
                self.api_disabled = True
                self.api_disabled_reason = status

    def notify(self, algorithm) -> None:
        self._sync_incumbent(algorithm)
        self._maybe_coordinated_repair(algorithm)
        self._maybe_room_plateau_repair(algorithm)
        self._maybe_room_chain_repair(algorithm)
        metrics = population_metrics(algorithm, time.perf_counter() - self.started,
                                     self.remaining())
        self.best_observed = self._sync_incumbent(algorithm)
        metrics["incumbent_hard"], metrics["incumbent_internal_total"] = self.best_observed
        metrics["incumbent_weighted_objectives"] = algorithm.problem.incumbent.weighted
        self.quality_history.append(self.best_observed)
        self.quality_records.append({"generation": metrics["generation"], "hard": self.best_observed[0],
                                     "internal_total": self.best_observed[1]})
        progress_signal = _progress_signal(self.best_observed)
        if not self.progress_history or progress_signal != self.progress_history[-1]:
            self.last_progress_generation = metrics["generation"]
        self.progress_history.append(progress_signal)
        metrics["progress_mode"] = progress_signal[0]
        metrics["stagnant_generations"] = metrics["generation"] - self.last_progress_generation
        reasons = self._trigger(metrics)
        available = metrics.get("remaining_evaluations")
        if reasons and available is not None and available < 4:
            self.budget_skipped_reviews += 1
            reasons = []
        if reasons and (self.remaining() is None or self.remaining() > 0.1) and (
                self.api_calls < self.settings.max_api_calls) and (
                len(self.events) < self.settings.max_reviews) and not self.api_disabled:
            self._stagnation_event_handled |= "stagnation" in reasons
            self._low_diversity_event_handled |= "low_diversity" in reasons
            prior_scoring_calls = self.region_builder.scoring_calls
            review_started = time.perf_counter()-self.started
            extraction_started = time.perf_counter()
            regions = self.region_builder.build(
                algorithm, self.settings.max_regions, self.settings.region_size)
            if self.settings.enable_blocker_chain and len(regions) > 1:
                for original in regions:
                    expanded = expand_blocker_region(self.instance,
                        algorithm.pop[int(original.solution_id)].X, original,
                        max_classes=min(10,self.settings.region_size*2))
                    if expanded is not None:
                        regions = regions[:self.settings.max_regions-1]+[expanded]
                        break
            extraction_seconds = time.perf_counter()-extraction_started
            local_checks = sum(r.screening["local_checks"] for r in regions if r.screening)
            self.diagnostic_seconds += extraction_seconds
            self.diagnostic_local_checks += local_checks
            region_calls = self.region_builder.scoring_calls - prior_scoring_calls
            algorithm.evaluator.n_eval += region_calls
            metrics["evaluations"] += region_calls
            remaining_evaluations = metrics.get("remaining_evaluations")
            self.current_operator_limit = self.settings.max_operator_evaluations
            if remaining_evaluations is not None:
                remaining_evaluations -= region_calls
                self.current_operator_limit = min(self.current_operator_limit, remaining_evaluations)
                metrics["remaining_evaluations"] = max(0, remaining_evaluations)
            metrics["trigger_reasons"] = reasons
            packet = self._packet(metrics, regions)
            # Bound the largest valid Inspector diagnosis and Planner wrapper.
            from .llm import PROMPTS
            import json
            diagnosis_sizes = [len(json.dumps({**r["target"], "relevant_constraint_ids": b["constraints"]["ids"]},
                                              ensure_ascii=False))
                for r in packet["candidate_regions"] for b in packet["boundary_context"]
                if r["target"]["region_id"] == b["region_id"]]
            envelope = max(map(len,PROMPTS.values())) + max(diagnosis_sizes,default=0) + 64
            packet, context_status = fit_packet(packet, max(0, getattr(self.client, "max_input_characters", 60000)-envelope))
            context_status["reserved_envelope_characters"] = envelope
            if packet is None:
                self.context_skipped_reviews += 1
                decision, role_records, decision_status = None, [], "context_limit_skipped"
                regions = []
            else:
                published = {r["target"]["region_id"] for r in packet["candidate_regions"]}
                regions = [r for r in regions if r.region_id in published]
                if self.remaining() is not None and self.remaining() <= 0.1:
                    decision, role_records, decision_status = None, [], "deadline_after_diagnostics"
                else:
                    decision, role_records, decision_status = self.decide(packet)
            self._account_calls(role_records)
            if self.api_disabled:
                # Export an incomplete run rather than keep evolving silently
                # after infrastructure has disabled its controller.
                algorithm.termination.force_termination = True
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
            self.best_observed = self._sync_incumbent(algorithm)
            self.quality_history[-1] = self.best_observed
            self.quality_records[-1].update(hard=self.best_observed[0], internal_total=self.best_observed[1])
            after_progress = _progress_signal(self.best_observed)
            if after_progress != self.progress_history[-1]:
                self.last_progress_generation = metrics["generation"]
                self._stagnation_event_handled = False
            self.progress_history[-1] = after_progress
            event = {
                "generation": metrics["generation"], "trigger": reasons,
                "state": packet, "roles": role_records,
                "metadata": {"schema_version": SCHEMA_VERSION, "instance": self.instance.name,
                    "seed": self.seed, "run_id": getattr(self, "run_id", None),
                    "packet_sha256": None if packet is None else packet_hash(packet),
                    "controller_settings_sha256": packet_hash(asdict(self.settings)),
                    "source_fingerprints": {r.region_id: r.screening["source_hash"] for r in regions if r.screening},
                    "diagnostics": {"elapsed_seconds": extraction_seconds, "local_checks": local_checks,
                                    "full_evaluations": region_calls, "context": context_status}},
                "review_started_elapsed_seconds": review_started,
                "review_completed_elapsed_seconds": time.perf_counter()-self.started,
                "decision_status": decision_status, "validation": validation,
                "action": asdict(plan),
                "outcome": None if outcome is None else asdict(outcome),
                "population_after": {"best_hard": after["best_hard"],
                                     "best_internal_total": after["best_internal_total"]},
                "incumbent_after": {"hard": self.best_observed[0], "internal_total": self.best_observed[1]},
                "incumbent_before": {"hard": metrics["incumbent_hard"], "internal_total": metrics["incumbent_internal_total"]},
            }
            self.events.append(event)
            def q(hard, total, weighted=None):
                return {"hard": hard, "internal_total": total, "weighted_objectives": weighted}
            self.outcomes.append({"generation": metrics["generation"],
                "source_hash": None if region is None or region.screening is None else region.screening["source_hash"],
                "target": {k: getattr(plan,k) for k in ("solution_id", "region_id", "boundary_id")},
                "action": {k: getattr(plan,k) for k in ("operator_id", "execution_budget", "next_review_interval")},
                "execution": {"attempts": 0 if outcome is None else outcome.attempts,
                    "full_evaluations": 0 if outcome is None else outcome.evaluations,
                    "scoring_rechecks": 0 if outcome is None else outcome.scoring_rechecks,
                    "elapsed_seconds": 0 if outcome is None else outcome.elapsed_seconds,
                    "stop_reason": decision_status if outcome is None else outcome.stop_reason},
                "result": {"decision_status": decision_status, "validation": validation,
                    "operator_status": None if outcome is None else outcome.status,
                    "selected": False if outcome is None else outcome.selected,
                    "source_quality": None if outcome is None else q(outcome.source_hard, outcome.source_total, outcome.source_weighted),
                    "best_trial_quality": None if outcome is None or outcome.best_trial_hard is None else
                        q(outcome.best_trial_hard, outcome.best_trial_total, outcome.best_trial_weighted),
                    "candidate_quality": None if outcome is None or outcome.candidate_hard is None else
                        q(outcome.candidate_hard, outcome.candidate_total, outcome.candidate_weighted),
                    "incumbent_before": event["incumbent_before"], "incumbent_after": event["incumbent_after"],
                    "incumbent_change": {"hard": self.best_observed[0]-metrics["incumbent_hard"],
                        "internal_total": self.best_observed[1]-metrics["incumbent_internal_total"]}}})
            self.last_review = metrics["generation"]
            self.next_review = metrics["generation"] + plan.next_review_interval
        super().notify(algorithm, room_chain_checked=True, room_plateau_checked=True,
                       coordinated_checked=True)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base.update({"mode": self.mode,
                     "status": "api_disabled" if self.api_disabled else "active",
                     "completion_status": ("incomplete_api_failure" if self.api_failures else
                                           "no_api_calls" if self.api_calls == 0 else "completed"),
                     "comparison_eligible": not self.api_failures and self.api_calls > 0,
                     "provider": self.settings.provider,
                     "model": self.settings.model,
                     "models_returned": sorted(self.models_returned),
                     "reviews": len(self.events),
                     "budget_skipped_reviews": self.budget_skipped_reviews,
                     "no_op": sum(event["action"]["operator_id"] == "no_op"
                                  for event in self.events),
                     "model_no_op": sum(event["validation"] == "valid_no_op" and
                         event["decision_status"] in {"decision_received", "inspector_no_target"}
                         for event in self.events),
                     "fallback_no_op": sum(event["action"]["operator_id"] == "no_op" and
                         not (event["validation"] == "valid_no_op" and
                              event["decision_status"] in {"decision_received", "inspector_no_target"})
                         for event in self.events),
                     "api_calls": self.api_calls,
                     "input_tokens": self.input_tokens,
                     "output_tokens": self.output_tokens,
                     "thinking_tokens": self.thinking_tokens,
                     "total_native_tokens": self.total_native_tokens,
                     "cached_input_tokens": self.cached_input_tokens,
                     "api_failures": dict(self.api_failures),
                     "api_disabled_reason": self.api_disabled_reason,
                     "api_pacing_wait_seconds": round(self.api_pacing_wait_seconds, 3),
                     "api_latency_seconds": round(self.api_latency_seconds, 3),
                     "api_disabled_after_configuration_error": (
                         self.api_disabled_reason == "api_configuration_error"),
                     "interventions": self.interventions,
                     "selected_interventions": self.selected_interventions})
        base["region_scoring_calls"] = self.region_builder.scoring_calls
        base["operator_scoring_rechecks"] = self.operator_scoring_rechecks
        base["state_packet_schema_version"] = SCHEMA_VERSION
        base["diagnostics"] = {"elapsed_seconds": self.diagnostic_seconds,
            "local_checks": self.diagnostic_local_checks, "full_evaluations": self.region_builder.scoring_calls,
            "context_skipped_reviews": self.context_skipped_reviews}
        base["unknown_usage_requests"] = self.unknown_usage_requests
        base["native_token_usage_complete"] = self.unknown_usage_requests == 0
        return base
