"""NSGA-III placement baseline for one ITC 2019 XML instance."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.ref_dirs import get_reference_directions

from .instance import Distribution, Instance, Time
from .scoring import Score, ScoreEngine, distribution_violations, evaluate, overlaps


@dataclass(frozen=True, slots=True)
class Config:
    population: int = 20
    generations: int = 20
    partitions: int = 3
    seed: int = 42
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    offspring_repair_steps: int = 8
    offspring_pair_trials: int = 2
    initial_repair_steps: int = 12
    initial_pair_trials: int = 3
    mutation_fraction: float = 0.005
    inner_partitions: int | None = None
    inner_scale: float = 0.5
    evaluations: int | None = None

    def __post_init__(self) -> None:
        if self.population < 2 or self.generations < 1 or self.partitions < 1:
            raise ValueError("Population, generations and partitions must be positive")
        if not 0 <= self.mutation_rate <= 1 or not 0 <= self.crossover_rate <= 1:
            raise ValueError("Mutation/crossover rates must lie in [0,1]")
        if min(self.offspring_repair_steps, self.offspring_pair_trials,
               self.initial_repair_steps, self.initial_pair_trials) < 0:
            raise ValueError("Repair steps and pair trials cannot be negative")
        if not 0 < self.mutation_fraction <= 1:
            raise ValueError("Mutation fraction must lie in (0,1]")
        if self.inner_partitions is not None and self.inner_partitions < 1:
            raise ValueError("Inner partitions must be positive")
        if not 0 < self.inner_scale < 1:
            raise ValueError("Inner reference scale must lie in (0,1)")
        if self.evaluations is not None and self.evaluations < self.population:
            raise ValueError("Evaluation budget must be at least the population size")


def reference_directions(config: Config) -> np.ndarray:
    outer = get_reference_directions("das-dennis", n_dim=4, n_partitions=config.partitions)
    if config.inner_partitions is None:
        return outer
    inner = get_reference_directions("das-dennis", n_dim=4, n_partitions=config.inner_partitions)
    inner = config.inner_scale * inner + (1 - config.inner_scale) / 4
    return np.unique(np.vstack((outer, inner)), axis=0)


def evaluation_count(algorithm) -> int:
    return int(algorithm.evaluator.n_eval) + getattr(algorithm.problem, "repair_scoring_calls", 0)


class FullEvaluationTermination(Termination):
    def __init__(self, budget: int) -> None:
        super().__init__()
        self.budget = budget

    def _update(self, algorithm) -> float:
        return min(1.0, evaluation_count(algorithm) / self.budget)


class BudgetedNSGA3(NSGA3):
    """Shrink the last offspring batch rather than overshoot a scoring cap."""
    def _infill(self):
        remaining = self.problem.remaining_scores()
        if remaining is not None and remaining <= 0:
            self.termination.force_termination = True
            return None
        usual = self.n_offsprings
        if remaining is not None:
            self.n_offsprings = min(usual, remaining)
        try:
            return super()._infill()
        finally:
            self.n_offsprings = usual


def _candidate_options(penalties: list[int], rng: np.random.Generator, limit: int) -> list[int]:
    if len(penalties) <= limit:
        return list(range(len(penalties)))
    cheapest = sorted(range(len(penalties)), key=lambda i: penalties[i])[:max(1, limit // 2)]
    sampled = rng.choice(len(penalties), size=limit - len(cheapest), replace=False)
    return list(dict.fromkeys([*cheapest, *map(int, sampled)]))


def _place(instance: Instance, rng: np.random.Generator, start=None, *, index=None, deadline=None) -> np.ndarray:
    from .construction import ConstructionIndex
    return (index or ConstructionIndex(instance)).build(rng, start=start, deadline=deadline)


def _improve_hard(instance: Instance, choices: np.ndarray,
                  rng: np.random.Generator, steps: int,
                  pair_trials: int = 0,
                  full_score_calls: list[int] | None = None, *, deadline=None,
                  score_fn=None, max_full_scoring_calls=None) -> np.ndarray:
    """Repair hard violations with single moves, then bounded joint moves."""
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
        if deadline is not None and time.perf_counter() >= deadline:
            return choices
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
            return choices
        target_list = sorted(targets)
        rng.shuffle(target_list)
        changed = False
        for index in target_list[:min(12, len(target_list))]:
            if deadline is not None and time.perf_counter() >= deadline:
                return choices
            cls = instance.classes[index]
            old_t, old_r = int(choices[index]), int(choices[n + index])
            old_hard = local_hard(index)
            if old_hard == 0:
                continue
            best = (old_hard, old_t, old_r)
            if len(cls.times) * max(1, len(cls.room_options)) <= 96:
                time_options = list(range(len(cls.times)))
                room_options = [-1] if cls.no_room else list(range(len(cls.room_options)))
            else:
                time_options = list(dict.fromkeys([old_t, *_candidate_options(
                    [t.penalty for t in cls.times], rng, 8)]))
                room_options = ([-1] if cls.no_room else list(dict.fromkeys([old_r, *_candidate_options(
                    [p for _, p in cls.room_options], rng, 5)])))
            for ti in time_options:
                if deadline is not None and time.perf_counter() >= deadline:
                    return choices
                times[index] = cls.times[ti]
                for ri in room_options:
                    if deadline is not None and time.perf_counter() >= deadline:
                        return choices
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
    if pair_trials and _joint_improve_hard(instance, choices, rng, pair_trials, full_score_calls,
                                          deadline=deadline, score_fn=score_fn,
                                          max_full_scoring_calls=max_full_scoring_calls):
        return _improve_hard(instance, choices, rng, steps=min(4, steps),
                             full_score_calls=full_score_calls, deadline=deadline, score_fn=score_fn)
    return choices


def _joint_improve_hard(instance: Instance, choices: np.ndarray,
                        rng: np.random.Generator, max_pairs: int,
                        full_score_calls: list[int] | None = None,
                        deadline: float | None = None, score_fn=None,
                        max_full_scoring_calls=None) -> bool:
    """Try bounded two-class changes rooted in room clashes or distributions."""
    if max_full_scoring_calls is not None and max_full_scoring_calls < 2:
        return False
    if deadline is not None and time.perf_counter() >= deadline:
        return False
    n = len(instance.classes)
    times = [cls.times[int(choices[i])] for i, cls in enumerate(instance.classes)]
    rooms = [None if cls.no_room else cls.room_options[int(choices[n + i])][0]
             for i, cls in enumerate(instance.classes)]
    room_members: dict[str, set[int]] = defaultdict(set)
    required_by_class: list[list[Distribution]] = [[] for _ in instance.classes]
    target_pairs: list[tuple[int, int]] = []
    for index, room_id in enumerate(rooms):
        if room_id is not None:
            for other in sorted(room_members[room_id]):
                if overlaps(times[index], times[other]):
                    target_pairs.append((other, index))
            room_members[room_id].add(index)

    violated: list[tuple[Distribution, int]] = []
    for distribution in instance.distributions:
        if not distribution.required:
            continue
        for index in distribution.classes:
            required_by_class[index].append(distribution)
        violations = distribution_violations(instance, distribution, times, rooms)
        if violations and len(distribution.classes) >= 2:
            violated.append((distribution, violations))
    rng.shuffle(violated)
    violated.sort(key=lambda item: item[1], reverse=True)
    special = {"MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock"}
    for distribution, _ in violated:
        candidates = list(combinations(distribution.classes, 2))
        if distribution.kind not in special:
            candidates = [pair for pair in candidates if distribution_violations(
                instance, Distribution(distribution.kind, distribution.params, True,
                                       distribution.penalty, pair), times, rooms)]
        rng.shuffle(candidates)
        for pair in candidates[:2]:
            if pair not in target_pairs:
                target_pairs.append(pair)
            if len(target_pairs) >= max_pairs:
                break
        if len(target_pairs) >= max_pairs:
            break
    target_pairs = target_pairs[:max_pairs]
    if not target_pairs:
        return False

    def options(index: int) -> list[tuple[int, int]]:
        cls = instance.classes[index]
        current = (int(choices[index]), int(choices[n + index]))
        time_options = list(dict.fromkeys([current[0], *_candidate_options(
            [time.penalty for time in cls.times], rng, 5)]))
        if cls.no_room:
            room_options = [-1]
        else:
            preferred_rooms = []
            for distribution in required_by_class[index]:
                if distribution.kind == "SameRoom":
                    for other in distribution.classes:
                        if other != index and rooms[other] is not None:
                            preferred_rooms.extend(j for j, (room_id, _) in
                                                   enumerate(cls.room_options)
                                                   if room_id == rooms[other])
            room_options = list(dict.fromkeys([
                current[1], *preferred_rooms,
                *_candidate_options([penalty for _, penalty in cls.room_options], rng, 6)]))
        alternatives = [(ti, ri) for ti in time_options for ri in room_options
                        if (ti, ri) != current]
        rng.shuffle(alternatives)
        alternatives.sort(key=lambda pair: (cls.times[pair[0]].penalty * instance.weights[0] +
                                            (0 if pair[1] == -1 else
                                             cls.room_options[pair[1]][1] * instance.weights[1])))
        return [current, *alternatives[:15]]

    def scored_hard(state: np.ndarray) -> int:
        if full_score_calls is not None:
            full_score_calls[0] += 1
        return (score_fn(state) if score_fn else evaluate(instance, state[:n], state[n:])[0]).hard

    best_gain = 0
    best_soft = float("inf")
    best_move: tuple[int, int, tuple[int, int], tuple[int, int]] | None = None
    for left, right in target_pairs:
        if deadline is not None and time.perf_counter() >= deadline:
            return False
        touched = {left, right}
        seen: set[int] = set()
        affected: list[Distribution] = []
        for distribution in (*required_by_class[left], *required_by_class[right]):
            if id(distribution) not in seen:
                seen.add(id(distribution))
                affected.append(distribution)

        def pair_hard() -> int:
            hard = 0
            for index in (left, right):
                room_id = rooms[index]
                if room_id is None:
                    continue
                hard += any(overlaps(times[index], unavailable)
                            for unavailable in instance.rooms[room_id].unavailable)
                hard += sum(overlaps(times[index], times[other])
                            for other in room_members[room_id] if other not in touched)
            if rooms[left] is not None and rooms[left] == rooms[right]:
                hard += overlaps(times[left], times[right])
            return hard + sum(distribution_violations(instance, distribution, times, rooms)
                              for distribution in affected)

        old_times = (times[left], times[right])
        old_rooms = (rooms[left], rooms[right])
        old_local_hard = pair_hard()
        left_options, right_options = options(left), options(right)
        for left_t, left_r in left_options:
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            times[left] = instance.classes[left].times[left_t]
            rooms[left] = (None if left_r == -1 else
                           instance.classes[left].room_options[left_r][0])
            for right_t, right_r in right_options:
                if deadline is not None and time.perf_counter() >= deadline:
                    return False
                times[right] = instance.classes[right].times[right_t]
                rooms[right] = (None if right_r == -1 else
                                instance.classes[right].room_options[right_r][0])
                gain = old_local_hard - pair_hard()
                if gain <= 0:
                    continue
                soft = sum(instance.classes[index].times[ti].penalty * instance.weights[0] +
                           (0 if ri == -1 else
                            instance.classes[index].room_options[ri][1] * instance.weights[1])
                           for index, ti, ri in ((left, left_t, left_r),
                                                 (right, right_t, right_r)))
                if gain > best_gain or (gain == best_gain and soft < best_soft):
                    best_gain, best_soft = gain, soft
                    best_move = (left, right, (left_t, left_r), (right_t, right_r))
            times[right], rooms[right] = old_times[1], old_rooms[1]
        times[left], rooms[left] = old_times[0], old_rooms[0]

    if best_move is None:
        return False
    if max_full_scoring_calls is not None and max_full_scoring_calls < 2:
        return False
    old_hard = scored_hard(choices)
    left, right, (left_t, left_r), (right_t, right_r) = best_move
    trial = choices.copy()
    trial[left], trial[n + left] = left_t, left_r
    trial[right], trial[n + right] = right_t, right_r
    if scored_hard(trial) >= old_hard:
        return False
    choices[:] = trial
    return True


def _room_chain_repair(instance: Instance, choices: np.ndarray, *,
                       max_evaluations: int = 4000, max_depth: int = 6,
                       beam_width: int = 12,
                       max_placements_per_class: int = 24,
                       hard_slack: int = 1,
                       deadline: float | None = None, score_fn=None) -> tuple[np.ndarray | None, int]:
    """Follow room/distribution blockers through bounded neutral moves.

    Only a final strict hard decrease is returned. Intermediate schedules stay
    private to this search and may have up to ``hard_slack`` extra hard violations.
    """
    n = len(instance.classes)
    source_score = score_fn(choices) if score_fn else evaluate(instance, choices[:n], choices[n:])[0]
    evaluations = 1
    if deadline is not None and time.perf_counter() >= deadline:
        return None, evaluations
    if source_score.room_conflicts + source_score.room_unavailable == 0:
        return None, evaluations

    def violations(state: np.ndarray) -> tuple[list[int], tuple]:
        times = [cls.times[int(state[i])] for i, cls in enumerate(instance.classes)]
        rooms = [None if cls.no_room else cls.room_options[int(state[n + i])][0]
                 for i, cls in enumerate(instance.classes)]
        by_room: dict[str, list[int]] = defaultdict(list)
        targets: list[int] = []
        signatures: list[tuple] = []
        for index, room_id in enumerate(rooms):
            if room_id is None:
                continue
            if any(overlaps(times[index], block)
                   for block in instance.rooms[room_id].unavailable):
                targets.append(index)
                signatures.append(("unavailable", index))
            for other in by_room[room_id]:
                if overlaps(times[index], times[other]):
                    targets.extend((other, index))
                    signatures.append(("room", other, index))
            by_room[room_id].append(index)
        for distribution_index, distribution in enumerate(instance.distributions):
            if distribution.required and distribution_violations(
                    instance, distribution, times, rooms):
                targets.extend(distribution.classes)
                signatures.append(("distribution", distribution_index))
        return list(dict.fromkeys(targets))[:8], tuple(signatures)

    def placements(state: np.ndarray, index: int) -> list[tuple[int, int]]:
        cls = instance.classes[index]
        current = (int(state[index]), int(state[n + index]))
        selected_times = [c.times[int(state[j])] for j, c in enumerate(instance.classes)]
        selected_rooms = [None if c.no_room else c.room_options[int(state[n + j])][0]
                          for j, c in enumerate(instance.classes)]
        per_time: list[list[tuple[int, int, int, int]]] = []
        for ti, time in enumerate(cls.times):
            ranked = []
            for ri in ([-1] if cls.no_room else range(len(cls.room_options))):
                if (ti, ri) == current:
                    continue
                room_id = None if ri == -1 else cls.room_options[ri][0]
                conflicts = (0 if room_id is None else
                             sum(room_id == selected_rooms[j] and overlaps(time, other)
                                 for j, other in enumerate(selected_times) if j != index))
                unavailable = (0 if room_id is None else
                               any(overlaps(time, block)
                                   for block in instance.rooms[room_id].unavailable))
                soft = time.penalty * instance.weights[0] + (
                    0 if ri == -1 else cls.room_options[ri][1] * instance.weights[1])
                ranked.append((int(conflicts + unavailable), soft, ti, ri))
            ranked.sort()
            per_time.append(ranked[:max(3, max_placements_per_class // len(cls.times))])
        candidates = [item for group in per_time for item in group]
        candidates.sort()
        return [(ti, ri) for _, _, ti, ri in candidates[:max_placements_per_class]]

    source = np.asarray(choices, dtype=np.int32).copy()
    beam: list[tuple[np.ndarray, int, int, int, tuple]] = [
        (source, source_score.hard, source_score.room_conflicts,
         source_score.total, violations(source)[1])]
    seen = {tuple(map(int, source))}
    for _ in range(max_depth):
        if deadline is not None and time.perf_counter() >= deadline:
            return None, evaluations
        next_states: list[tuple[np.ndarray, int, int, int, tuple]] = []
        for state, _, _, _, _ in beam:
            targets, _ = violations(state)
            for index in targets:
                for ti, ri in placements(state, index):
                    if (evaluations >= max_evaluations or
                            (deadline is not None and time.perf_counter() >= deadline)):
                        return None, evaluations
                    trial = state.copy()
                    trial[index], trial[n + index] = ti, ri
                    identity = tuple(map(int, trial))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    score = score_fn(trial) if score_fn else evaluate(instance, trial[:n], trial[n:])[0]
                    evaluations += 1
                    if (score.hard < source_score.hard and
                            score.room_conflicts + score.room_unavailable <
                            source_score.room_conflicts + source_score.room_unavailable):
                        return trial, evaluations
                    if score.hard <= source_score.hard + hard_slack:
                        next_states.append((trial, score.hard, score.room_conflicts,
                                            score.total,
                                            violations(trial)[1]))
        if not next_states:
            break
        if source_score.hard == source_score.room_conflicts:
            next_states.sort(key=lambda item: (item[1], item[3]))
        else:
            next_states.sort(key=lambda item: (item[2], item[1], item[3]))
        beam = []
        per_signature: dict[tuple, int] = defaultdict(int)
        for state in next_states:
            signature = state[4]
            if per_signature[signature] < 4:
                beam.append(state)
                per_signature[signature] += 1
            if len(beam) >= beam_width:
                break
    return None, evaluations




class Incumbent:
    """One archive for every full-scored candidate, independent of survival."""

    def __init__(self):
        self.quality = None
        self.choices = None

    def observe(self, choices, quality):
        quality = tuple(map(int, quality))
        if self.quality is None or quality < self.quality:
            self.quality = quality
            self.choices = np.asarray(choices, dtype=np.int32).copy()


class PlacementProblem(ElementwiseProblem):
    def __init__(self, instance: Instance, evaluation_budget: int | None = None, deadline=None):
        self.instance = instance
        self.incumbent = Incumbent()
        self.engine = ScoreEngine(instance)
        self.deadline = deadline
        self.skip_initial_repair = False
        from .construction import ConstructionIndex
        self.construction = ConstructionIndex(instance)
        self.evaluation_budget = evaluation_budget
        self.repair_scoring_calls = 0
        self.stage_seconds = {"initial_sampling": 0.0, "offspring_repair": 0.0,
                              "objective_evaluation": 0.0}
        self.stage_candidates = {"initial_sampling": 0, "offspring_repair": 0,
                                 "objective_evaluation": 0}
        n = len(instance.classes)
        upper = np.asarray([len(c.times) - 1 for c in instance.classes] +
                           [len(c.room_options) - 1 if not c.no_room else -1 for c in instance.classes],
                           dtype=int)
        lower = np.asarray([0] * n + [-1 if c.no_room else 0 for c in instance.classes], dtype=int)
        super().__init__(n_var=2 * n, n_obj=4, n_ieq_constr=1, xl=lower, xu=upper, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        started = time.perf_counter()
        n = len(self.instance.classes)
        score = self.score_state(x)
        out["F"] = np.asarray(score.weighted, dtype=float)
        out["G"] = np.asarray([score.hard], dtype=float)
        self.stage_seconds["objective_evaluation"] += time.perf_counter() - started
        self.stage_candidates["objective_evaluation"] += 1

    def score_state(self, choices):
        score = self.engine.score(choices)
        self.incumbent.observe(choices, (score.hard, score.total))
        return score

    def remaining_scores(self, reserve=0):
        return None if self.evaluation_budget is None else max(
            0, self.evaluation_budget-self.engine.requests-reserve)


class PlacementSampling(Sampling):
    def __init__(self, steps: int = 12, pair_trials: int = 3) -> None:
        super().__init__()
        self.steps = steps
        self.pair_trials = pair_trials

    def _do(self, problem: PlacementProblem, n_samples: int, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        started = time.perf_counter()
        calls = [0]
        samples = []
        for _ in range(n_samples):
            placed = _place(problem.instance, random_state, index=problem.construction,
                            deadline=problem.deadline)
            samples.append(_improve_hard(problem.instance, placed, random_state,
                steps=self.steps, pair_trials=self.pair_trials, full_score_calls=calls,
                deadline=problem.deadline, score_fn=problem.score_state,
                max_full_scoring_calls=problem.remaining_scores(reserve=n_samples)))
        samples = np.stack(samples)
        problem.skip_initial_repair = True
        problem.repair_scoring_calls += calls[0]
        problem.stage_seconds["initial_sampling"] += time.perf_counter() - started
        problem.stage_candidates["initial_sampling"] += n_samples
        return samples


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
    def __init__(self, probability: float, fraction: float = 0.005):
        super().__init__(prob=probability)
        self.fraction = fraction

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
        changes = min(len(mutable), max(1, round(n * self.fraction)))
        for row in result:
            for index in random_state.choice(mutable, size=changes, replace=False):
                cls = problem.instance.classes[int(index)]
                if len(cls.times) > 1:
                    row[index] = int(random_state.integers(len(cls.times)))
                if len(cls.room_options) > 1:
                    row[n + index] = int(random_state.integers(len(cls.room_options)))
        return result


class RoomRepair(Repair):
    def __init__(self, steps: int = 8, pair_trials: int = 2) -> None:
        super().__init__()
        self.steps = steps
        self.pair_trials = pair_trials

    def _do(self, problem: PlacementProblem, X, *args,
            random_state: np.random.Generator | None = None, **kwargs):
        if random_state is None:
            raise RuntimeError("pymoo did not provide random_state")
        # Preserve inherited placements; repair only classes around violations.
        if problem.skip_initial_repair:
            problem.skip_initial_repair = False
            return X.copy()
        started = time.perf_counter()
        calls = [0]
        repaired = np.stack([_improve_hard(problem.instance, row.copy(), random_state,
                                           steps=self.steps, pair_trials=self.pair_trials,
                                           full_score_calls=calls, deadline=problem.deadline,
                                           score_fn=problem.score_state,
                                           max_full_scoring_calls=problem.remaining_scores(reserve=len(X)))
                              for row in X])
        problem.repair_scoring_calls += calls[0]
        problem.stage_seconds["offspring_repair"] += time.perf_counter() - started
        problem.stage_candidates["offspring_repair"] += len(X)
        return repaired


class FeasibleSolutionTermination(Termination):
    """Stop after a completed generation contains an internally feasible candidate."""

    def _update(self, algorithm) -> float:
        population = getattr(algorithm, "pop", None)
        if population is None or len(population) == 0:
            return 0.0
        constraints = population.get("G")
        if constraints is None:
            return 0.0
        values = np.asarray(constraints, dtype=float)
        if values.size == 0:
            return 0.0
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        feasible = np.all(values <= 0.0, axis=1)
        return 1.0 if np.any(feasible) else 0.0


def _seeded_cv_tournament(pop, pairs: np.ndarray,
                          random_state: np.random.Generator | None = None,
                          **kwargs) -> np.ndarray:
    """Use the solver RNG for every tournament tie, including infeasible ties.

    pymoo 0.6.2's default NSGA-III comparator does not pass its RNG into the
    infeasible-tie branch, so two runs with the same seed can diverge.
    """
    if random_state is None:
        raise RuntimeError("Tournament selection requires the solver random state")
    winners = np.empty(len(pairs), dtype=int)
    for index, (left, right) in enumerate(pairs):
        a, b = int(left), int(right)
        a_cv = float(np.asarray(pop[a].CV).reshape(-1)[0])
        b_cv = float(np.asarray(pop[b].CV).reshape(-1)[0])
        if a_cv > 0 or b_cv > 0:
            if a_cv < b_cv:
                winners[index] = a
                continue
            if b_cv < a_cv:
                winners[index] = b
                continue
        winners[index] = int(random_state.choice([a, b]))
    return winners[:, None]


def run(instance: Instance, config: Config, *, callback=None,
        time_limit_seconds: float | None = None,
        stop_on_feasible: bool = False) -> tuple[np.ndarray, Score, object]:
    directions = reference_directions(config)
    if config.population < len(directions):
        raise ValueError(f"Population must be >= {len(directions)} reference directions")
    algorithm = BudgetedNSGA3(ref_dirs=directions, pop_size=config.population,
                      sampling=PlacementSampling(config.initial_repair_steps, config.initial_pair_trials),
                      crossover=ClassCrossover(config.crossover_rate),
                      mutation=ClassMutation(config.mutation_rate, config.mutation_fraction),
                      repair=RoomRepair(config.offspring_repair_steps,
                                        config.offspring_pair_trials),
                      selection=TournamentSelection(func_comp=_seeded_cv_tournament),
                      eliminate_duplicates=False)
    terminations = [MaximumGenerationTermination(config.generations)]
    if time_limit_seconds is not None:
        terminations.append(TimeBasedTermination(time_limit_seconds))
    if config.evaluations is not None:
        terminations.append(FullEvaluationTermination(config.evaluations))
    if stop_on_feasible:
        terminations.append(FeasibleSolutionTermination())
    termination = (terminations[0] if len(terminations) == 1 else
                   TerminationCollection(*terminations))
    options = {"termination": termination, "seed": config.seed,
               "verbose": False, "save_history": False}
    if callback is not None:
        options["callback"] = callback
    deadline = getattr(callback, "deadline", None)
    if deadline is None and time_limit_seconds is not None:
        deadline = time.perf_counter()+time_limit_seconds
    result = minimize(PlacementProblem(instance, config.evaluations, deadline), algorithm, **options)
    if result.pop is None or len(result.pop) == 0:
        raise RuntimeError("NSGA-III returned an empty population")
    n = len(instance.classes)
    hard = np.asarray(result.pop.get("G"), dtype=float).reshape(-1)
    totals = np.asarray(result.pop.get("F"), dtype=float).sum(axis=1)
    best_index = min(range(len(result.pop)), key=lambda i: (hard[i], totals[i], i))
    choices = np.asarray(result.pop[best_index].X, dtype=np.int32).copy()
    best_quality = (int(hard[best_index]), int(totals[best_index]))
    # Selection can discard a low-total trade-off even when it is the best
    # feasible solution seen so far. Preserve the same anytime incumbent for
    # all three pipelines, including candidates injected by a controller.
    observed = getattr(callback, "best_choices", None)
    observed_quality = getattr(callback, "best_quality", None)
    if observed is not None and observed_quality is not None and observed_quality < best_quality:
        choices = np.asarray(observed, dtype=np.int32).copy()
        best_quality = observed_quality
    archived = result.algorithm.problem.incumbent
    if archived.quality is not None and archived.quality < best_quality:
        choices, best_quality = archived.choices.copy(), archived.quality
    best_score, _ = evaluate(instance, choices[:n], choices[n:])
    if (best_score.hard, best_score.total) != best_quality:
        raise RuntimeError("Stored incumbent score differs from final verification")
    return choices, best_score, result
