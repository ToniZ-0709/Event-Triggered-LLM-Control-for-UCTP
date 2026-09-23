"""NSGA-III placement baseline for one ITC 2019 XML instance."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .instance import Instance, Time
from .scoring import Score, distribution_violations, evaluate, overlaps


@dataclass(frozen=True, slots=True)
class Config:
    population: int = 40
    generations: int = 20
    partitions: int = 4
    seed: int = 42
    mutation_rate: float = 0.2
    crossover_rate: float = 0.9

    def __post_init__(self) -> None:
        if self.population < 2 or self.generations < 1 or self.partitions < 1:
            raise ValueError("Population, generations and partitions must be positive")
        if not 0 <= self.mutation_rate <= 1 or not 0 <= self.crossover_rate <= 1:
            raise ValueError("Mutation/crossover rates must lie in [0,1]")


def _candidate_options(penalties: list[int], rng: np.random.Generator, limit: int) -> list[int]:
    if len(penalties) <= limit:
        return list(range(len(penalties)))
    cheapest = sorted(range(len(penalties)), key=lambda i: penalties[i])[:max(1, limit // 2)]
    sampled = rng.choice(len(penalties), size=limit - len(cheapest), replace=False)
    return list(dict.fromkeys([*cheapest, *map(int, sampled)]))


def _place(instance: Instance, rng: np.random.Generator,
           start: np.ndarray | None = None) -> np.ndarray:
    """Greedy room packing; all choices remain inside each class's XML domain."""
    n = len(instance.classes)
    choices = np.empty(2 * n, dtype=np.int32)
    room_usage: dict[str, list[Time]] = defaultdict(list)
    order = sorted(range(n), key=lambda i: (len(instance.classes[i].times) *
                                           max(1, len(instance.classes[i].room_options)),
                                           int(rng.integers(0, 1000000))))
    for index in order:
        cls = instance.classes[index]
        time_options = _candidate_options([t.penalty for t in cls.times], rng, 8)
        room_options = [-1] if cls.no_room else _candidate_options(
            [penalty for _, penalty in cls.room_options], rng, 6)
        if start is not None:
            time_options = list(dict.fromkeys([int(start[index]), *time_options]))
            room_options = list(dict.fromkeys([int(start[n + index]), *room_options]))
        best: tuple[float, int, int] | None = None
        for ti in time_options:
            time = cls.times[ti]
            for ri in room_options:
                room_id = None if ri == -1 else cls.room_options[ri][0]
                clashes = sum(overlaps(time, other) for other in room_usage[room_id]) if room_id else 0
                unavailable = (any(overlaps(time, block) for block in instance.rooms[room_id].unavailable)
                               if room_id else False)
                room_penalty = 0 if room_id is None else cls.room_options[ri][1]
                penalty = (clashes + unavailable) * 1000000 + time.penalty * instance.weights[0] + \
                          room_penalty * instance.weights[1] + float(rng.random())
                candidate = (penalty, ti, ri)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        choices[index], choices[n + index] = best[1], best[2]
        chosen_room = None if best[2] == -1 else cls.room_options[best[2]][0]
        if chosen_room is not None:
            room_usage[chosen_room].append(cls.times[best[1]])
    return choices


def _improve_hard(instance: Instance, choices: np.ndarray,
                  rng: np.random.Generator, steps: int) -> np.ndarray:
    """Target room clashes and required distributions with bounded local moves."""
    n = len(instance.classes)
    times = [cls.times[int(choices[i])] for i, cls in enumerate(instance.classes)]
    rooms = [None if cls.no_room else cls.room_options[int(choices[n + i])][0]
             for i, cls in enumerate(instance.classes)]
    hard_by_class: list[list] = [[] for _ in instance.classes]
    required = [d for d in instance.distributions if d.required]
    room_members: dict[str, set[int]] = defaultdict(set)
    for index, room_id in enumerate(rooms):
        if room_id is not None:
            room_members[room_id].add(index)
    for distribution in required:
        for index in distribution.classes:
            hard_by_class[index].append(distribution)

    def local_hard(index: int) -> int:
        room_id = rooms[index]
        conflicts = 0
        if room_id is not None:
            conflicts += any(overlaps(times[index], unavailable)
                             for unavailable in instance.rooms[room_id].unavailable)
            conflicts += sum(overlaps(times[index], times[j])
                             for j in room_members[room_id] if j != index)
        return conflicts + sum(distribution_violations(instance, d, times, rooms)
                               for d in hard_by_class[index])

    for _ in range(steps):
        targets: set[int] = set()
        by_room: dict[str, list[int]] = defaultdict(list)
        for index, room_id in enumerate(rooms):
            if room_id is None:
                continue
            if any(overlaps(times[index], u) for u in instance.rooms[room_id].unavailable):
                targets.add(index)
            for other in by_room[room_id]:
                if overlaps(times[index], times[other]):
                    targets.update((index, other))
            by_room[room_id].append(index)
        for distribution in required:
            if distribution_violations(instance, distribution, times, rooms):
                targets.update(distribution.classes)
        if not targets:
            break
        target_list = sorted(targets)
        rng.shuffle(target_list)
        changed = False
        for index in target_list[:min(12, len(target_list))]:
            cls = instance.classes[index]
            old_t, old_r = int(choices[index]), int(choices[n + index])
            old_hard = local_hard(index)
            if old_hard == 0:
                continue
            best = (old_hard, old_t, old_r)
            if n <= 500:
                time_options = list(range(len(cls.times)))
                room_options = [-1] if cls.no_room else list(range(len(cls.room_options)))
            else:
                time_options = list(dict.fromkeys([old_t, *_candidate_options(
                    [t.penalty for t in cls.times], rng, 8)]))
                room_options = ([-1] if cls.no_room else list(dict.fromkeys([old_r, *_candidate_options(
                    [p for _, p in cls.room_options], rng, 5)])))
            for ti in time_options:
                times[index] = cls.times[ti]
                for ri in room_options:
                    rooms[index] = None if ri == -1 else cls.room_options[ri][0]
                    hard = local_hard(index)
                    if hard < best[0]:
                        best = (hard, ti, ri)
            choices[index], choices[n + index] = best[1], best[2]
            times[index] = cls.times[best[1]]
            new_room = None if best[2] == -1 else cls.room_options[best[2]][0]
            if new_room != (None if old_r == -1 else cls.room_options[old_r][0]):
                if old_r != -1:
                    room_members[cls.room_options[old_r][0]].remove(index)
                if new_room is not None:
                    room_members[new_room].add(index)
            rooms[index] = new_room
            if best[0] < old_hard:
                changed = True
                break
        if not changed:
            break
    return choices


class PlacementProblem(ElementwiseProblem):
    def __init__(self, instance: Instance):
        self.instance = instance
        n = len(instance.classes)
        upper = np.asarray([len(c.times) - 1 for c in instance.classes] +
                           [len(c.room_options) - 1 if not c.no_room else -1 for c in instance.classes],
                           dtype=int)
        lower = np.asarray([0] * n + [-1 if c.no_room else 0 for c in instance.classes], dtype=int)
        super().__init__(n_var=2 * n, n_obj=4, n_ieq_constr=1, xl=lower, xu=upper, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        n = len(self.instance.classes)
        score, _ = evaluate(self.instance, x[:n], x[n:])
        out["F"] = np.asarray(score.weighted, dtype=float)
        out["G"] = np.asarray([score.hard], dtype=float)


class PlacementSampling(Sampling):
    def _do(self, problem: PlacementProblem, n_samples: int, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        return np.stack([_improve_hard(problem.instance, _place(problem.instance, random_state),
                                       random_state, steps=12) for _ in range(n_samples)])


class ClassCrossover(Crossover):
    def __init__(self, probability: float):
        super().__init__(2, 2, prob=probability)

    def _do(self, problem: PlacementProblem, X, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        n = len(problem.instance.classes)
        offspring = np.empty_like(X)
        for mating in range(X.shape[1]):
            swap = random_state.random(n) < 0.5
            for child in range(2):
                other = 1 - child
                offspring[child, mating] = X[child, mating]
                offspring[child, mating, :n][swap] = X[other, mating, :n][swap]
                offspring[child, mating, n:][swap] = X[other, mating, n:][swap]
        return offspring


class ClassMutation(Mutation):
    def __init__(self, probability: float):
        super().__init__(prob=probability)

    def _do(self, problem: PlacementProblem, X, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        result = X.copy()
        n = len(problem.instance.classes)
        mutable = [i for i, cls in enumerate(problem.instance.classes)
                   if len(cls.times) > 1 or len(cls.room_options) > 1]
        if not mutable:
            return result
        changes = min(len(mutable), max(1, round(n * 0.005)))
        for row in result:
            for index in random_state.choice(mutable, size=changes, replace=False):
                cls = problem.instance.classes[int(index)]
                if len(cls.times) > 1:
                    row[index] = int(random_state.integers(len(cls.times)))
                if len(cls.room_options) > 1:
                    row[n + index] = int(random_state.integers(len(cls.room_options)))
        return result


class RoomRepair(Repair):
    def _do(self, problem: PlacementProblem, X, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        return np.stack([_improve_hard(problem.instance, _place(problem.instance, random_state, row),
                                       random_state, steps=4) for row in X])


def run(instance: Instance, config: Config) -> tuple[np.ndarray, Score, object]:
    directions = get_reference_directions("das-dennis", n_dim=4, n_partitions=config.partitions)
    if config.population < len(directions):
        raise ValueError(f"Population must be >= {len(directions)} reference directions")
    algorithm = NSGA3(ref_dirs=directions, pop_size=config.population,
                      sampling=PlacementSampling(), crossover=ClassCrossover(config.crossover_rate),
                      mutation=ClassMutation(config.mutation_rate), repair=RoomRepair(),
                      eliminate_duplicates=False)
    result = minimize(PlacementProblem(instance), algorithm,
                      termination=("n_gen", config.generations), seed=config.seed,
                      verbose=False, save_history=False)
    if result.pop is None or len(result.pop) == 0:
        raise RuntimeError("NSGA-III returned an empty population")
    n = len(instance.classes)
    candidates = []
    for individual in result.pop:
        choices = np.asarray(individual.X, dtype=np.int32)
        score, _ = evaluate(instance, choices[:n], choices[n:])
        candidates.append((score.hard, score.total, choices, score))
    _, _, best_choices, best_score = min(candidates, key=lambda item: (item[0], item[1]))
    return best_choices, best_score, result
