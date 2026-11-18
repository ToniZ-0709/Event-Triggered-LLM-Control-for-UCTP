"""Bounded improvement operators acting only on a selected class region."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np
from pymoo.core.population import Population

from .instance import Instance
from .regions import RegionCandidate, swap_targets
from .scoring import Score
from .neighborhood import LocalSoftCosts, room_neighborhood_repair


OPERATORS = ("move", "swap", "destroy_repair")


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


@dataclass(frozen=True, slots=True)
class OperatorProbe:
    """Results of a bounded search before any population change."""

    started: float
    attempts: int
    evaluations: int
    source_hard: int
    source_total: int
    best_trial: tuple[int, int] | None
    candidate: Population | None
    candidate_quality: tuple[int, int] | None


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


def _repair_options(instance: Instance, current: np.ndarray, index: int, local_soft: LocalSoftCosts,
                    times, rooms, rng: np.random.Generator, deadline: float | None
                    ) -> list[tuple[int, int]]:
    """Rank placements by affected soft costs without re-running student sectioning."""
    n = len(instance.classes)
    cls = instance.classes[index]
    old_t, old_r = int(current[index]), int(current[n + index])
    room_options = ((-1, None, 0),) if cls.no_room else tuple(
        (ri, room, penalty) for ri, (room, penalty) in enumerate(cls.room_options))
    ranked = []
    for ti, chosen in enumerate(cls.times):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        time_cost = (chosen.penalty * instance.weights[0] +
                     local_soft.global_option_cost(index, chosen, times, rooms))
        for ri, room, penalty in room_options:
            if (ti, ri) == (old_t, old_r):
                continue
            amount = (time_cost + penalty * instance.weights[1] +
                      local_soft.option_cost(index, chosen, room, times, rooms))
            ranked.append((amount, (ti, ri)))
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: item[0])
    return [option for _, option in ranked]


def _rebuild_order(region: RegionCandidate, rng: np.random.Generator) -> list[int]:
    violations = Counter()
    for detail in region.constraint_details:
        for group in detail.get("violation_groups", ()):
            for class_id in group["in_region_class_ids"]:
                violations[class_id] += group["violation_units"]
    order = list(zip(region.class_indices, region.class_ids))
    rng.shuffle(order)
    order.sort(key=lambda entry: -violations[entry[1]])
    return [index for index, _ in order]


_PROPOSALS = {"move": _move, "swap": _swap}


def _admissible_candidate(source_hard: int, source_weighted: tuple[int, ...],
                          trial_hard: int, trial_weighted: tuple[int, ...]) -> bool:
    """Admit lexicographic gains before feasibility and Pareto trades after it."""
    if source_hard > 0:
        return (trial_hard, sum(trial_weighted)) < (source_hard, sum(source_weighted))
    return trial_hard == 0 and any(after < before
                                   for before, after in zip(source_weighted, trial_weighted))


def _admit_repair(source: Score, trial: Score) -> bool:
    return _admissible_candidate(source.hard, source.weighted, trial.hard, trial.weighted)


def probe_operator(algorithm, instance: Instance, region: RegionCandidate,
                   operator_id: str, budget: int, rng: np.random.Generator,
                   deadline: float | None) -> OperatorProbe:
    """Charge all scoring to the evaluator; leave the source population intact."""
    if operator_id not in OPERATORS or budget < 1:
        raise ValueError("Invalid improvement operator or budget")
    started = time.perf_counter()
    source = np.asarray(algorithm.pop[int(region.solution_id)].X, dtype=np.int32)
    source_hard = int(np.asarray(algorithm.pop[int(region.solution_id)].G).reshape(-1)[0])
    source_weighted = tuple(map(int, np.asarray(algorithm.pop[int(region.solution_id)].F).reshape(-1)))
    source_total = sum(source_weighted)
    best_pop = None
    best_key = None
    best_trial: tuple[int, int] | None = None
    attempts = evaluations = 0
    evaluated: set[tuple[int, ...]] = set()

    def result() -> OperatorProbe:
        return OperatorProbe(started, attempts, evaluations, source_hard, source_total,
                             best_trial, best_pop, None if best_key is None else best_key[:2])

    def consider(trial: np.ndarray) -> tuple[int, int] | None:
        nonlocal attempts, evaluations, best_pop, best_key, best_trial
        if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
            return None
        attempts += 1
        if trial is None or np.array_equal(trial, source):
            return None
        identity = tuple(map(int, trial))
        if identity in evaluated:
            return None
        evaluated.add(identity)
        candidate_pop = Population.new("X", trial.reshape(1, -1))
        algorithm.evaluator.eval(algorithm.problem, candidate_pop, algorithm=algorithm)
        evaluations += 1
        hard = int(np.asarray(candidate_pop.get("G")).reshape(-1)[0])
        weighted = tuple(map(int, np.asarray(candidate_pop.get("F")).reshape(-1)))
        total = sum(weighted)
        rank = (hard, total)
        if best_trial is None or rank < best_trial:
            best_trial = rank
        if not _admissible_candidate(source_hard, source_weighted, hard, weighted):
            return rank
        key = (hard, total, identity)
        if best_key is None or key < best_key:
            best_pop, best_key = candidate_pop, key
        return rank

    if deadline is not None and time.perf_counter() >= deadline:
        return result()
    if operator_id == "destroy_repair":
        # Solve the entire selected region jointly before the cheap sequential fallback.
        # The outside chromosome is immutable; all solver checks are charged to this budget.
        engine = algorithm.problem.engine
        cached = engine.cache.get(engine.key(source))
        source_score = cached[0] if cached else algorithm.problem.score_state(source)
        if cached is None:
            evaluations += 1
            algorithm.evaluator.n_eval += 1
        assignments = engine.cache[engine.key(source)][1]
        local_deadline = time.perf_counter()+3.0
        if deadline is not None:
            local_deadline = min(local_deadline, deadline)
        remaining = max(0, budget - evaluations)
        candidate, stats = room_neighborhood_repair(
            instance, source, deadline=local_deadline, seed=int(rng.integers(2147483647)),
            max_classes=len(region.class_indices), max_options=32, max_table_cells=50000,
            deterministic_limit=0.25, allowed_classes=region.class_indices,
            require_room_improvement=False, source_score=source_score, admit_candidate=_admit_repair,
            # Leave one full evaluation for the student-aware rebuild when possible.
            max_full_scoring_calls=remaining - int(remaining > 1), score_fn=algorithm.problem.score_state)
        used = stats["full_scoring_calls"]
        evaluations += used
        attempts += used
        algorithm.evaluator.n_eval += used
        if candidate is not None:
            score = engine.cache[engine.key(candidate)][0]
            best_trial = score.hard, score.total
            if _admissible_candidate(source_hard, source_weighted, score.hard, score.weighted):
                best_key = (score.hard, score.total, tuple(map(int, candidate)))
                best_pop = Population.new("X", candidate[None], "F", np.asarray([score.weighted]),
                                          "G", np.asarray([[score.hard]]))
        # Rebuild up to three classes sequentially. Each proposed placement is
        # evaluated on the complete timetable and charged to the same budget.
        if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
            return result()
        mutable = [i for i in _rebuild_order(region, rng) if len(instance.classes[i].times) > 1 or
                   len(instance.classes[i].room_options) > 1]
        selected = mutable[:3]
        working = source.copy()
        working_rank = (source_hard, source_total)
        n = len(instance.classes)
        local_soft = LocalSoftCosts(instance, set(region.class_indices), assignments)
        times = [cls.times[int(working[i])] for i, cls in enumerate(instance.classes)]
        rooms = [None if cls.no_room else cls.room_options[int(working[n+i])][0]
                 for i, cls in enumerate(instance.classes)]
        for position, index in enumerate(selected):
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            remaining_classes = len(selected) - position
            quota = max(1, (budget - evaluations) // remaining_classes)
            local_best = working_rank
            local_choice = working
            for ti, ri in _repair_options(instance, working, index, local_soft, times, rooms,
                                          rng, deadline)[:quota]:
                trial = working.copy()
                trial[index], trial[n + index] = ti, ri
                rank = consider(trial)
                if rank is not None and rank < local_best:
                    local_best, local_choice = rank, trial
            working, working_rank = local_choice, local_best
            cls = instance.classes[index]
            times[index] = cls.times[int(working[index])]
            rooms[index] = None if cls.no_room else cls.room_options[int(working[n+index])][0]
    else:
        for _ in range(budget * 5):
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            consider(_PROPOSALS[operator_id](instance, source, region.class_indices, rng))

    return result()


def inject_operator_candidate(algorithm, probe: OperatorProbe) -> OperatorOutcome:
    """Recheck and pass at most one already evaluated candidate to survival."""
    attempts, evaluations = probe.attempts, probe.evaluations
    source_hard, source_total = probe.source_hard, probe.source_total
    best_pop, started = probe.candidate, probe.started
    trial_hard, trial_total = probe.best_trial or (None, None)
    if best_pop is None:
        status = "no_candidate_evaluated" if evaluations == 0 else "no_improving_candidate"
        return OperatorOutcome(status, attempts, evaluations, False, source_hard,
                               source_total, trial_hard, trial_total,
                               None, None, None, 0, time.perf_counter() - started)
    choices = np.asarray(best_pop[0].X, dtype=np.int32)
    score = algorithm.problem.score_state(choices)
    if (score.hard, score.total) != probe.candidate_quality:
        return OperatorOutcome("scoring_mismatch", attempts, evaluations, False,
                               source_hard, source_total, trial_hard, trial_total,
                               score.hard, score.total, score.weighted, 1,
                               time.perf_counter() - started)
    population_size = len(algorithm.pop)
    algorithm.pop = algorithm.survival.do(
        algorithm.problem, Population.merge(algorithm.pop, best_pop),
        n_survive=population_size, random_state=algorithm.random_state)
    algorithm._set_optimum()
    selected = any(individual is best_pop[0] for individual in algorithm.pop)
    return OperatorOutcome("selected" if selected else "rejected_by_selection",
                           attempts, evaluations, selected, source_hard, source_total,
                           trial_hard, trial_total,
                           score.hard, score.total, score.weighted, 1,
                           time.perf_counter() - started)


def apply_operator(algorithm, instance: Instance, region: RegionCandidate,
                   operator_id: str, budget: int, rng: np.random.Generator,
                   deadline: float | None) -> OperatorOutcome:
    """Search within the budget and inject at most one improving candidate."""
    probe = probe_operator(algorithm, instance, region, operator_id, budget, rng, deadline)
    return inject_operator_candidate(algorithm, probe)
