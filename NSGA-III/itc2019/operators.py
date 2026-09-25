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


def apply_operator(algorithm, instance: Instance, region: RegionCandidate,
                   operator_id: str, budget: int, rng: np.random.Generator,
                   deadline: float | None) -> OperatorOutcome:
    """Try at most ``budget`` full evaluations and inject at most one feasible candidate."""
    if operator_id not in OPERATORS or budget < 1:
        raise ValueError("Invalid improvement operator or budget")
    started = time.perf_counter()
    source = np.asarray(algorithm.pop[int(region.solution_id)].X, dtype=np.int32)
    source_hard = int(np.asarray(algorithm.pop[int(region.solution_id)].G).reshape(-1)[0])
    source_total = int(np.asarray(algorithm.pop[int(region.solution_id)].F).sum())
    best_pop = None
    best_key = None
    best_trial: tuple[int, int] | None = None
    attempts = evaluations = 0
    evaluated: set[tuple[int, ...]] = set()

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
        total = int(np.asarray(candidate_pop.get("F")).reshape(1, -1).sum())
        rank = (hard, total)
        if best_trial is None or rank < best_trial:
            best_trial = rank
        if hard != 0:
            return rank
        key = (total, identity)
        if best_key is None or key < best_key:
            best_pop, best_key = candidate_pop, key
        return rank

    if operator_id == "destroy_repair":
        # Rebuild up to three classes sequentially. Each proposed placement is
        # evaluated on the complete timetable and charged to the same budget.
        mutable = [i for i in region.class_indices if len(instance.classes[i].times) > 1 or
                   len(instance.classes[i].room_options) > 1]
        selected = list(map(int, rng.choice(mutable, size=min(3, len(mutable)),
                                            replace=False))) if mutable else []
        working = source.copy()
        working_rank = (source_hard, source_total)
        n = len(instance.classes)
        for position, index in enumerate(selected):
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            remaining_classes = len(selected) - position
            quota = max(1, (budget - evaluations) // remaining_classes)
            local_best = working_rank
            local_choice = working
            for ti, ri in _repair_options(instance, working, index, rng)[:quota]:
                trial = working.copy()
                trial[index], trial[n + index] = ti, ri
                rank = consider(trial)
                if rank is not None and rank < local_best:
                    local_best, local_choice = rank, trial
            working, working_rank = local_choice, local_best
    else:
        for _ in range(budget * 5):
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            consider(_PROPOSALS[operator_id](instance, source, region.class_indices, rng))

    trial_hard, trial_total = best_trial if best_trial is not None else (None, None)
    if best_pop is None:
        status = "no_candidate_evaluated" if evaluations == 0 else "no_feasible_candidate"
        return OperatorOutcome(status, attempts, evaluations, False, source_hard,
                               source_total, trial_hard, trial_total,
                               None, None, None, 0, time.perf_counter() - started)
    n = len(instance.classes)
    choices = np.asarray(best_pop[0].X, dtype=np.int32)
    score, _ = evaluate(instance, choices[:n], choices[n:])
    if score.hard != 0 or score.total != best_key[0]:
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
