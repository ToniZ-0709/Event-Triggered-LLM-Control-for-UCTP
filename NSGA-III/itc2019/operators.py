"""Bounded improvement operators acting only on a selected class region."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np
from pymoo.core.population import Population

from .instance import Instance
from .regions import RegionCandidate, swap_targets
from .scoring import evaluate
from .neighborhood import room_neighborhood_repair
from .diagnostics import PlacementDiagnostics, source_hash


OPERATORS = ("move", "swap", "destroy_repair", "guided_move", "distribution_repair", "blocker_chain")


@dataclass(frozen=True, slots=True)
class OperatorOutcome:
    status: str
    attempts: int
    evaluations: int
    selected: bool
    source_hard: int
    source_total: int
    best_trial_hard: int | None
    best_trial_total: int | None
    candidate_hard: int | None
    candidate_total: int | None
    candidate_weighted: tuple[int, int, int, int] | None
    scoring_rechecks: int
    elapsed_seconds: float
    changed_classes: tuple[str, ...] = ()
    targeted_rule: str | None = None
    admitted: bool = False
    stop_reason: str | None = None
    source_weighted: tuple[int, ...] | None = None
    best_trial_weighted: tuple[int, ...] | None = None


def _move(instance: Instance, source: np.ndarray, indices: tuple[int, ...],
          rng: np.random.Generator) -> np.ndarray | None:
    n = len(instance.classes)
    movable = [i for i in indices if len(instance.classes[i].times) > 1 or
               len(instance.classes[i].room_options) > 1]
    if not movable:
        return None
    index = int(rng.choice(movable))
    cls = instance.classes[index]
    candidate = source.copy()
    genes = []
    if len(cls.times) > 1:
        genes.append(index)
    if len(cls.room_options) > 1:
        genes.append(n + index)
    gene = int(rng.choice(genes))
    count = len(cls.times) if gene == index else len(cls.room_options)
    current = int(source[gene])
    draw = int(rng.integers(count - 1))
    candidate[gene] = draw + (draw >= current)
    return candidate


def _swap(instance: Instance, source: np.ndarray, indices: tuple[int, ...],
          rng: np.random.Generator) -> np.ndarray | None:
    if len(indices) < 2:
        return None
    n = len(instance.classes)
    pairs = [(left, right, target) for left, right in combinations(indices, 2)
             if (target := swap_targets(instance, source, left, right)) is not None]
    if not pairs:
        return None
    left, right, (a_time, a_room, b_time, b_room) = pairs[int(rng.integers(len(pairs)))]
    candidate = source.copy()
    candidate[left], candidate[n + left] = a_time, a_room
    candidate[right], candidate[n + right] = b_time, b_room
    return candidate


def _repair_options(instance: Instance, current: np.ndarray, index: int,
                    rng: np.random.Generator) -> list[tuple[int, int]]:
    """Bounded class placements, ordered by cheap domain options and random ties."""
    n = len(instance.classes)
    cls = instance.classes[index]
    old_t, old_r = int(current[index]), int(current[n + index])
    times = sorted(range(len(cls.times)), key=lambda i: (cls.times[i].penalty, i))[:6]
    times = list(dict.fromkeys([old_t, *times]))
    if cls.no_room:
        rooms = [-1]
    else:
        rooms = sorted(range(len(cls.room_options)),
                       key=lambda i: (cls.room_options[i][1], i))[:4]
        rooms = list(dict.fromkeys([old_r, *rooms]))
    options = [(ti, ri) for ti in times for ri in rooms if (ti, ri) != (old_t, old_r)]
    rng.shuffle(options)
    options.sort(key=lambda pair: (cls.times[pair[0]].penalty +
                                    (0 if pair[1] == -1 else cls.room_options[pair[1]][1])))
    return options


_PROPOSALS = {"move": _move, "swap": _swap}


def _admissible_candidate(source_hard: int, source_total: int,
                          trial_hard: int, trial_total: int) -> bool:
    """Require hard progress before feasibility, then preserve it and improve soft."""
    if source_hard > 0:
        return trial_hard < source_hard
    return trial_hard == 0 and trial_total < source_total


def apply_operator(algorithm, instance: Instance, region: RegionCandidate,
                   operator_id: str, budget: int, rng: np.random.Generator,
                   deadline: float | None) -> OperatorOutcome:
    """Share the allowance across source/trials/final recheck, including cache hits."""
    if operator_id not in OPERATORS or budget < 1:
        raise ValueError("Invalid improvement operator or budget")
    started = time.perf_counter()
    source = np.asarray(algorithm.pop[int(region.solution_id)].X, dtype=np.int32)
    source_hard = int(np.asarray(algorithm.pop[int(region.solution_id)].G).reshape(-1)[0])
    source_total = int(np.asarray(algorithm.pop[int(region.solution_id)].F).sum())
    source_weighted = tuple(map(int, algorithm.pop[int(region.solution_id)].F))
    best_pop = best_key = best_trial = best_trial_weighted = None
    attempts = evaluations = rechecks = 0
    targeted_rule = None
    reason = "proposal_pool_exhausted"
    evaluated: set[str] = set()
    remaining = getattr(algorithm.problem, "evaluation_budget", None)
    if remaining is not None:
        used = algorithm.evaluator.n_eval + getattr(algorithm.problem, "repair_scoring_calls", 0)
        budget = min(budget, max(0, remaining-used))
    local_deadline = started + 3.0 if deadline is None else min(deadline, started + 3.0)

    def room_for_trial():
        return evaluations < budget-1 and time.perf_counter() < local_deadline

    def charged_score(trial):
        nonlocal evaluations
        result = algorithm.problem.score_state(trial)
        algorithm.evaluator.n_eval += 1
        evaluations += 1
        return result

    def observe(trial, score):
        nonlocal best_pop, best_key, best_trial, best_trial_weighted
        rank = (score.hard, score.total)
        if best_trial is None or rank < best_trial:
            best_trial, best_trial_weighted = rank, score.weighted
        if not _admissible_candidate(source_hard, source_total, *rank):
            return rank
        key = (*rank, tuple(map(int, trial)))
        if best_key is None or key < best_key:
            best_key = key
            best_pop = Population.new("X", trial[None].copy(), "F", np.asarray([score.weighted]),
                                      "G", np.asarray([[score.hard]]))
        return rank

    def consider(trial):
        nonlocal attempts
        if not room_for_trial():
            return None
        attempts += 1
        if trial is None or np.array_equal(trial, source):
            return None
        identity = source_hash(trial)
        if identity in evaluated:
            return None
        evaluated.add(identity)
        return observe(trial, charged_score(trial))

    def outcome(status, selected=False, score=None):
        changed = () if best_pop is None else tuple(instance.classes[i].id
            for i in region.class_indices if source[i] != best_pop[0].X[i] or
            source[len(instance.classes)+i] != best_pop[0].X[len(instance.classes)+i])
        return OperatorOutcome(status, attempts, evaluations, selected, source_hard, source_total,
            *(best_trial or (None, None)), None if score is None else score.hard,
            None if score is None else score.total, None if score is None else score.weighted,
            rechecks, time.perf_counter()-started, changed, targeted_rule,
            score is not None and status in {"selected", "rejected_by_selection"}, reason,
            source_weighted, best_trial_weighted)

    screening = region.screening
    if screening is not None and screening["source_hash"] != source_hash(source):
        reason = "source_identity_changed"
        return outcome("stale_region")
    if not room_for_trial():
        reason = "time_budget_exhausted" if time.perf_counter() >= local_deadline else "evaluation_budget_exhausted"
        return outcome("no_candidate_evaluated")
    if operator_id == "blocker_chain":
        if not region.expanded_region:
            reason = "expanded_region_required"
            return outcome("operator_unavailable_in_region")
        beam = [(source.copy(), (source_hard, source_total))]
        for depth in range(3):
            next_beam = []
            for state, rank in beam:
                proposals, _, _ = PlacementDiagnostics(instance,state).moves(region.class_indices,rank[0])
                quota = max(1,(budget-evaluations-1)//max(1,(3-depth)*len(beam)))
                for gene,value in proposals[:quota]:
                    if not room_for_trial():
                        break
                    trial = state.copy()
                    trial[gene] = value
                    trial_rank = consider(trial)
                    if trial_rank is not None and trial_rank[0] <= source_hard+1:
                        next_beam.append((trial,trial_rank))
            next_beam.sort(key=lambda item:(item[1],tuple(map(int,item[0]))))
            beam = next_beam[:2]
            if not beam or not room_for_trial():
                break
    elif operator_id == "guided_move":
        if screening is None:
            reason = "screening_not_available"
            return outcome("no_candidate_evaluated")
        for gene, value in screening["move_proposals"]:
            if not room_for_trial():
                break
            trial = source.copy()
            trial[gene] = value
            consider(trial)
    elif operator_id in {"destroy_repair", "distribution_repair"}:
        active = region.class_indices
        target_index = None
        if operator_id == "distribution_repair":
            selected_target = PlacementDiagnostics(instance, source).distribution_target(active)
            if selected_target is None:
                reason = "no_actionable_violated_required_rule"
                return outcome("no_candidate_evaluated")
            target_index, active = selected_target
            if screening is not None and selected_target != screening["distribution_target"]:
                reason = "target_rule_changed"
                return outcome("stale_region")
            targeted_rule = f"distribution:{target_index}"
        engine = algorithm.problem.engine
        cached = engine.cache.get(engine.key(source))
        source_score = cached[0] if cached else charged_score(source)
        if room_for_trial():
            candidate, stats = room_neighborhood_repair(
                instance, source, deadline=local_deadline, seed=int(rng.integers(2147483647)),
                max_classes=len(active), max_options=32, max_table_cells=50000,
                deterministic_limit=0.25, allowed_classes=active,
                require_room_improvement=False, allow_soft_improvement=operator_id == "destroy_repair",
                target_distribution_index=target_index, source_score=source_score,
                max_full_scoring_calls=max(0, budget-evaluations-1), score_fn=charged_score)
            attempts += stats["full_scoring_calls"]
            reason = stats["status"]
            if stats.get("best_trial_quality") is not None:
                best_trial = tuple(stats["best_trial_quality"])
                best_trial_weighted = tuple(stats["best_trial_weighted"])
            if candidate is not None:
                observe(candidate, engine.cache[engine.key(candidate)][0])
        if operator_id == "destroy_repair":
            mutable = [i for i in active if len(instance.classes[i].times) > 1 or
                       len(instance.classes[i].room_options) > 1]
            selected = list(map(int, rng.choice(mutable, size=min(3, len(mutable)), replace=False))) if mutable else []
            working, working_rank = source.copy(), (source_hard, source_total)
            n = len(instance.classes)
            for position, index in enumerate(selected):
                quota = max(0, (budget-evaluations-1)//(len(selected)-position))
                local_best, local_choice = working_rank, working
                for ti, ri in _repair_options(instance, working, index, rng)[:quota]:
                    trial = working.copy()
                    trial[index], trial[n+index] = ti, ri
                    rank = consider(trial)
                    if rank is not None and rank < local_best:
                        local_best, local_choice = rank, trial
                working, working_rank = local_choice, local_best
    else:
        for _ in range(budget * 5):
            if not room_for_trial():
                break
            consider(_PROPOSALS[operator_id](instance, source, region.class_indices, rng))
    if time.perf_counter() >= local_deadline:
        reason = "time_budget_exhausted"
    elif evaluations >= budget-1:
        reason = "evaluation_budget_exhausted"
    if best_pop is None:
        return outcome("no_candidate_evaluated" if best_trial is None else "no_improving_candidate")
    if time.perf_counter() >= local_deadline:
        return outcome("recheck_deadline_exhausted")
    choices = np.asarray(best_pop[0].X, dtype=np.int32)
    score = charged_score(choices)
    rechecks = 1
    if (score.hard, score.total) != best_key[:2]:
        return outcome("scoring_mismatch", score=score)
    population_size = len(algorithm.pop)
    algorithm.pop = algorithm.survival.do(
        algorithm.problem, Population.merge(algorithm.pop, best_pop),
        n_survive=population_size, random_state=algorithm.random_state)
    algorithm._set_optimum()
    selected = any(individual is best_pop[0] for individual in algorithm.pop)
    return outcome("selected" if selected else "rejected_by_selection", selected, score)
