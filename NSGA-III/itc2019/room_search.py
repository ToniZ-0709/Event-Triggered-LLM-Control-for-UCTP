"""Bounded plateau search for room clashes and unavailable-room assignments."""

from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from .instance import Instance
from .scoring import Score, distribution_violations, evaluate, overlaps


def room_plateau_repair(
    instance: Instance,
    choices: np.ndarray,
    rng: np.random.Generator,
    *,
    deadline: float,
    max_steps: int = 2000,
    max_restarts: int = 100,
    max_changes: int = 8,
    hard_slack: int = 2,
    max_full_scoring_calls: int = 16,
    source_score: Score | None = None,
    score_fn=None,
) -> tuple[np.ndarray | None, dict[str, int]]:
    """Explore neutral/worse local moves, but return only a strict hard improvement.

    Cheap local deltas are exact for room, unavailable-room, and required-distribution
    hard penalties. Student sectioning is checked again with the full evaluator before
    returning a candidate. The input chromosome is never modified.
    """
    n = len(instance.classes)
    source = np.asarray(choices, dtype=np.int32).copy()
    stats = {"full_scoring_calls": 0, "steps": 0, "restarts": 0}
    if source_score is None:
        source_score = score_fn(source) if score_fn else evaluate(instance, source[:n], source[n:])[0]
        stats["full_scoring_calls"] = 1
    source_room_hard = source_score.room_conflicts + source_score.room_unavailable
    if source_room_hard == 0 or time.perf_counter() >= deadline:
        return None, stats

    source_core = source_score.hard - source_score.unassigned_requests
    required_by_class = [[] for _ in instance.classes]
    for distribution in instance.distributions:
        if distribution.required:
            for index in distribution.classes:
                required_by_class[index].append(distribution)

    while (time.perf_counter() < deadline and stats["steps"] < max_steps and
           stats["restarts"] < max_restarts):
        state = source.copy()
        times = [cls.times[int(state[i])] for i, cls in enumerate(instance.classes)]
        rooms = [None if cls.no_room else cls.room_options[int(state[n + i])][0]
                 for i, cls in enumerate(instance.classes)]
        by_room: dict[str, set[int]] = defaultdict(set)
        for index, room_id in enumerate(rooms):
            if room_id is not None:
                by_room[room_id].add(index)
        core = source_core
        changed: set[int] = set()
        tabu: list[tuple[int, int, int]] = []
        stats["restarts"] += 1

        for _ in range(100):
            if time.perf_counter() >= deadline or stats["steps"] >= max_steps:
                break
            room_targets: list[int] = []
            for members in by_room.values():
                order = sorted(members)
                for pos, left in enumerate(order):
                    for right in order[pos + 1:]:
                        if overlaps(times[left], times[right]):
                            room_targets.extend((left, right))
            for index, room_id in enumerate(rooms):
                if room_id is not None and any(
                        overlaps(times[index], blocked)
                        for blocked in instance.rooms[room_id].unavailable):
                    room_targets.append(index)
            if not room_targets:
                break
            target = int(rng.choice(room_targets))
            cls = instance.classes[target]

            def local_parts() -> tuple[int, int]:
                room_id = rooms[target]
                room_hard = 0
                if room_id is not None:
                    room_hard += sum(overlaps(times[target], times[other])
                                     for other in by_room[room_id] if other != target)
                    room_hard += any(overlaps(times[target], block)
                                     for block in instance.rooms[room_id].unavailable)
                distribution_hard = sum(
                    distribution_violations(instance, distribution, times, rooms)
                    for distribution in required_by_class[target])
                return int(room_hard), int(distribution_hard)

            old_room_hard, old_distribution_hard = local_parts()
            old_time, old_room = times[target], rooms[target]
            old_ti, old_ri = int(state[target]), int(state[n + target])
            options: list[tuple[int, int, int, int, int]] = []
            for ti, option_time in enumerate(cls.times):
                if time.perf_counter() >= deadline:
                    break
                times[target] = option_time
                for ri in ([-1] if cls.no_room else range(len(cls.room_options))):
                    if time.perf_counter() >= deadline:
                        break
                    if (ti, ri) == (old_ti, old_ri):
                        continue
                    if target not in changed and len(changed) >= max_changes:
                        continue
                    rooms[target] = None if ri == -1 else cls.room_options[ri][0]
                    room_hard, distribution_hard = local_parts()
                    delta_room = room_hard - old_room_hard
                    delta = delta_room + distribution_hard - old_distribution_hard
                    if core + delta <= source_core + hard_slack:
                        reverse_penalty = int((target, ti, ri) in tabu)
                        options.append((delta, delta_room, reverse_penalty, ti, ri))
            times[target], rooms[target] = old_time, old_room
            if not options:
                break
            options.sort(key=lambda item: (item[0], item[1], item[2]))
            best_options = options[:min(8, len(options))]
            delta, _, _, ti, ri = best_options[int(rng.integers(len(best_options)))]
            state[target], state[n + target] = ti, ri
            times[target] = cls.times[ti]
            new_room = None if ri == -1 else cls.room_options[ri][0]
            if old_room != new_room:
                if old_room is not None:
                    by_room[old_room].remove(target)
                if new_room is not None:
                    by_room[new_room].add(target)
            rooms[target] = new_room
            core += delta
            if (ti, ri) == (int(source[target]), int(source[n + target])):
                changed.discard(target)
            else:
                changed.add(target)
            tabu.append((target, old_ti, old_ri))
            if len(tabu) > 12:
                tabu.pop(0)
            stats["steps"] += 1

            if core < source_core:
                if stats["full_scoring_calls"] >= max_full_scoring_calls:
                    return None, stats
                score = score_fn(state) if score_fn else evaluate(instance, state[:n], state[n:])[0]
                stats["full_scoring_calls"] += 1
                if (score.hard < source_score.hard and
                        score.room_conflicts + score.room_unavailable < source_room_hard):
                    return state, stats
    return None, stats
