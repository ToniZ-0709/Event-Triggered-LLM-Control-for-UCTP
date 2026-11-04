"""Deterministic search summaries and candidate regions for controller reviews."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .instance import Instance
from .scoring import distribution_violations, evaluate, overlaps, attendee_conflict


@dataclass(frozen=True, slots=True)
class RegionCandidate:
    solution_id: str
    region_id: str
    boundary_id: str
    class_indices: tuple[int, ...]
    class_ids: tuple[str, ...]
    boundary_class_ids: tuple[str, ...]
    constraint_ids: tuple[str, ...]
    hard_contribution: float
    soft_contribution: float
    unlocalized_penalties: tuple[str, ...]
    class_summaries: tuple[dict, ...]
    movable_class_count: int
    compatible_swap_pairs: int
    available_operators: tuple[str, ...]

    def public(self) -> dict:
        result = asdict(self)
        result.pop("class_indices")
        result["boundary_class_count"] = len(self.boundary_class_ids)
        result["boundary_truncated"] = len(self.boundary_class_ids) > 32
        result["boundary_class_ids"] = self.boundary_class_ids[:32]
        return result


def swap_targets(instance: Instance, choices: np.ndarray, left: int,
                 right: int) -> tuple[int, int, int, int] | None:
    """Return legal target option indices for exchanging two placements."""
    n = len(instance.classes)
    a, b = instance.classes[left], instance.classes[right]
    a_time, b_time = a.times[int(choices[left])], b.times[int(choices[right])]
    a_room = None if a.no_room else a.room_options[int(choices[n + left])][0]
    b_room = None if b.no_room else b.room_options[int(choices[n + right])][0]

    def target(cls, other_time, other_room):
        ti = next((i for i, option in enumerate(cls.times)
                   if (option.days, option.weeks, option.start, option.length) ==
                   (other_time.days, other_time.weeks, other_time.start,
                    other_time.length)), None)
        if cls.no_room:
            ri = -1 if other_room is None else None
        else:
            ri = next((i for i, (room_id, _) in enumerate(cls.room_options)
                       if room_id == other_room), None)
        return ti, ri

    new_a = target(a, b_time, b_room)
    new_b = target(b, a_time, a_room)
    if None in new_a or None in new_b:
        return None
    if (new_a == (int(choices[left]), int(choices[n + left])) and
            new_b == (int(choices[right]), int(choices[n + right]))):
        return None
    return new_a[0], new_a[1], new_b[0], new_b[1]


def placement_diversity(chromosomes: np.ndarray) -> float:
    """Mean pairwise fraction of classes with a different time or room."""
    chromosomes = np.asarray(chromosomes, dtype=np.int32)
    n_classes = chromosomes.shape[1] // 2
    if len(chromosomes) < 2 or n_classes == 0:
        return 0.0
    left, right = np.triu_indices(len(chromosomes), k=1)
    different = ((chromosomes[left, :n_classes] !=
                  chromosomes[right, :n_classes]) |
                 (chromosomes[left, n_classes:] !=
                  chromosomes[right, n_classes:]))
    return float(np.mean(np.count_nonzero(different, axis=1) / n_classes))


def population_metrics(algorithm, elapsed_seconds: float,
                       remaining_seconds: float | None) -> dict:
    pop = algorithm.pop
    hard = np.asarray(pop.get("G"), dtype=float).reshape(-1)
    weighted = np.asarray(pop.get("F"), dtype=float)
    feasible = hard == 0
    totals = weighted.sum(axis=1)
    best_index = min(range(len(pop)), key=lambda i: (hard[i], totals[i], i))
    feasible_front = 0
    if feasible.any():
        feasible_front = len(NonDominatedSorting().do(weighted[feasible], only_non_dominated_front=True))
    chromosomes = np.asarray(pop.get("X"), dtype=np.int32)
    unique = len({tuple(map(int, row)) for row in chromosomes})
    niches = pop.get("niche")
    valid_niches = [] if niches is None else [int(value) for value in niches
                                             if value is not None and int(value) >= 0]
    direction_count = len(algorithm.ref_dirs)
    objective_names = ("time", "room", "distribution", "student")
    evaluations = int(algorithm.evaluator.n_eval) + getattr(algorithm.problem, "repair_scoring_calls", 0)
    evaluation_budget = getattr(algorithm.problem, "evaluation_budget", None)
    return {
        "generation": int(algorithm.n_gen),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "remaining_seconds": None if remaining_seconds is None else round(max(0.0, remaining_seconds), 3),
        "evaluations": evaluations,
        "evaluation_budget": evaluation_budget,
        "remaining_evaluations": None if evaluation_budget is None else max(0, evaluation_budget - evaluations),
        "population_size": len(pop),
        "feasible_count": int(np.count_nonzero(feasible)),
        "best_hard": int(hard[best_index]),
        "best_internal_total": int(totals[best_index]),
        "best_feasible_internal_total": int(np.min(totals[feasible])) if feasible.any() else None,
        "best_feasible_official_score": None,
        "metric_source": "internal_evaluator",
        "feasible_nondominated_count": feasible_front,
        "decision_diversity": round(unique / len(pop), 4),
        "placement_diversity": round(placement_diversity(chromosomes), 4),
        "reference_direction_coverage": (
            round(len(set(valid_niches)) / direction_count, 4)
            if valid_niches and direction_count else None),
        "objective_distribution": {
            name: {"min": float(np.min(weighted[:, j])),
                   "median": float(np.median(weighted[:, j])),
                   "max": float(np.max(weighted[:, j]))}
            for j, name in enumerate(objective_names)
        },
    }


class RegionBuilder:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.scoring_calls = 0
        self.by_class: list[list] = [[] for _ in instance.classes]
        for distribution in instance.distributions:
            for index in distribution.classes:
                self.by_class[index].append(distribution)

    def _contributions(self, choices: np.ndarray) -> tuple[
            list[float], list[float], list[set[int]], list[set[str]], tuple[str, ...]]:
        instance = self.instance
        n = len(instance.classes)
        times = [cls.times[int(choices[i])] for i, cls in enumerate(instance.classes)]
        rooms = [None if cls.no_room else cls.room_options[int(choices[n + i])][0]
                 for i, cls in enumerate(instance.classes)]
        hard = [0] * n
        soft = [0] * n
        neighbors: list[set[int]] = [set() for _ in range(n)]
        constraint_ids: list[set[str]] = [set() for _ in range(n)]
        by_room: dict[str, list[int]] = defaultdict(list)
        for i, cls in enumerate(instance.classes):
            soft[i] += cls.times[int(choices[i])].penalty * instance.weights[0]
            if rooms[i] is not None:
                soft[i] += cls.room_options[int(choices[n + i])][1] * instance.weights[1]
                by_room[rooms[i]].append(i)
                if any(overlaps(times[i], blocked) for blocked in instance.rooms[rooms[i]].unavailable):
                    hard[i] += 1
                    constraint_ids[i].add(f"room_unavailable:{cls.id}")
        for indices in by_room.values():
            for offset, left in enumerate(indices):
                for right in indices[offset + 1:]:
                    if overlaps(times[left], times[right]):
                        hard[left] += 1
                        hard[right] += 1
                        neighbors[left].add(right)
                        neighbors[right].add(left)
                        conflict_id = f"room_conflict:{instance.classes[left].id}:{instance.classes[right].id}"
                        constraint_ids[left].add(conflict_id)
                        constraint_ids[right].add(conflict_id)
        for distribution_index, distribution in enumerate(instance.distributions):
            for i in distribution.classes:
                neighbors[i].update(j for j in distribution.classes if j != i)
                constraint_ids[i].add(f"distribution:{distribution_index}")
            violations = distribution_violations(instance, distribution, times, rooms)
            if not violations:
                continue
            contribution = (violations if distribution.required else
                            violations * distribution.penalty * instance.weights[2])
            target = hard if distribution.required else soft
            if distribution.kind not in {"MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock"}:
                for a, b in combinations(distribution.classes, 2):
                    amount = distribution_violations(instance, replace(distribution, classes=(a,b)), times, rooms)
                    if amount:
                        cost = amount if distribution.required else amount*distribution.penalty*instance.weights[2]
                        target[a] += cost
                        target[b] += cost
            else:
                for i in distribution.classes:
                    target[i] += contribution/max(1, len(distribution.classes))
        # A legal alternative room can be occupied by a currently nonviolating blocker.
        for i, cls in enumerate(instance.classes):
            for room, _ in cls.room_options:
                neighbors[i].update(j for j in by_room.get(room, ()) if j != i)
        fn = getattr(self, "score_fn", None)
        score = fn(choices) if fn else evaluate(instance, choices[:n], choices[n:])[0]
        self.scoring_calls += 1
        unlocalized = []
        if score.student:
            engine = getattr(self, "engine", None)
            if engine is None:
                unlocalized.append("student_conflict")
            else:
                attending = defaultdict(list)
                for i, students in engine.details(choices).items():
                    for student in students:
                        attending[student].append(i)
                for members in attending.values():
                    for a, b in combinations(sorted(members), 2):
                        if attendee_conflict(instance, times[a], rooms[a], times[b], rooms[b]):
                            soft[a] += instance.weights[3]
                            soft[b] += instance.weights[3]
                            neighbors[a].add(b)
                            neighbors[b].add(a)
                            constraint_ids[a].add("student_conflict")
                            constraint_ids[b].add("student_conflict")
        if score.unassigned_requests:
            unlocalized.append("unassigned_requests")
        return hard, soft, neighbors, constraint_ids, tuple(unlocalized)

    def build(self, algorithm, max_regions: int, region_size: int) -> list[RegionCandidate]:
        self.score_fn = getattr(getattr(algorithm, "problem", None), "score_state", None)
        self.engine = getattr(getattr(algorithm, "problem", None), "engine", None)
        pop = algorithm.pop
        hard = np.asarray(pop.get("G"), dtype=float).reshape(-1)
        totals = np.asarray(pop.get("F"), dtype=float).sum(axis=1)
        ranked_solutions = sorted(range(len(pop)), key=lambda i: (hard[i], totals[i], i))
        representatives = []
        seen_chromosomes: set[bytes] = set()
        for index in ranked_solutions:
            chromosome = np.asarray(pop[index].X, dtype=np.int32).tobytes()
            if chromosome in seen_chromosomes:
                continue
            seen_chromosomes.add(chromosome)
            representatives.append(index)
            if len(representatives) == 2:
                break
        if len(representatives) == 2:
            first = representatives[0]
            weighted = np.asarray(pop.get("F"), dtype=float)
            norm = (weighted-weighted.min(axis=0))/(np.ptp(weighted, axis=0)+1)
            eligible = [i for i in ranked_solutions if i != first and
                        hard[i] <= hard[first]+(0 if hard[first] == 0 else 1) and
                        not np.array_equal(pop[i].X, pop[first].X)]
            if eligible:
                representatives[1] = max(eligible, key=lambda i: (
                    float(np.linalg.norm(norm[i]-norm[first])) +
                    float(np.mean(np.asarray(pop[i].X) != np.asarray(pop[first].X))), -i))
        per_solution = math.ceil(max_regions / len(representatives))
        candidates: list[RegionCandidate] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for position, solution_index in enumerate(representatives):
            target_count = (max_regions if position == len(representatives) - 1 else
                            min(max_regions, per_solution * (position + 1)))
            choices = np.asarray(pop[solution_index].X, dtype=np.int32)
            hard_by_class, soft_by_class, neighbors, by_class_constraints, unlocalized = (
                self._contributions(choices))
            ranked = sorted(range(len(self.instance.classes)),
                            key=lambda i: (-hard_by_class[i], -soft_by_class[i], i))
            for center in ranked:
                if len(candidates) >= target_count:
                    break
                component = {center}
                while len(component) < region_size:
                    frontier = set().union(*(neighbors[i] for i in component))-component
                    if not frontier:
                        break
                    component.add(min(frontier, key=lambda i: (-hard_by_class[i], -soft_by_class[i], i)))
                active = tuple(sorted(component))
                key = (solution_index, active)
                if key in seen:
                    continue
                seen.add(key)
                boundary = sorted(set().union(*(neighbors[i] for i in active)) - set(active))
                boundary_ids = tuple(self.instance.classes[i].id for i in boundary)
                stamp = hashlib.sha256(repr((active, boundary)).encode()).hexdigest()[:12]
                n = len(self.instance.classes)
                summaries = []
                movable = 0
                for index in active:
                    cls = self.instance.classes[index]
                    time = cls.times[int(choices[index])]
                    room_index = int(choices[n + index])
                    room_id = None if cls.no_room else cls.room_options[room_index][0]
                    movable += int(len(cls.times) > 1 or len(cls.room_options) > 1)
                    summaries.append({
                        "class_id": cls.id,
                        "current_time": {"days": time.days, "weeks": time.weeks,
                                         "start": time.start, "length": time.length},
                        "current_room_id": room_id,
                        "time_options": len(cls.times),
                        "room_options": len(cls.room_options),
                        "current_time_penalty": time.penalty,
                        "current_room_penalty": (0 if cls.no_room else
                                                 cls.room_options[room_index][1]),
                    })
                swap_count = sum(swap_targets(self.instance, choices, left, right) is not None
                                 for left, right in combinations(active, 2))
                available = []
                if movable:
                    available.extend(("move", "destroy_repair"))
                if swap_count:
                    available.append("swap")
                candidates.append(RegionCandidate(
                    solution_id=str(solution_index),
                    region_id=f"{solution_index}:{len(candidates)}",
                    boundary_id=stamp,
                    class_indices=active,
                    class_ids=tuple(self.instance.classes[i].id for i in active),
                    boundary_class_ids=boundary_ids,
                    constraint_ids=tuple(sorted(set().union(
                        *(by_class_constraints[i] for i in active)))),
                    hard_contribution=sum(hard_by_class[i] for i in active),
                    soft_contribution=sum(soft_by_class[i] for i in active),
                    unlocalized_penalties=unlocalized,
                    class_summaries=tuple(summaries),
                    movable_class_count=movable,
                    compatible_swap_pairs=swap_count,
                    available_operators=tuple(available),
                ))
                if len(candidates) >= max_regions:
                    return candidates
        return candidates
