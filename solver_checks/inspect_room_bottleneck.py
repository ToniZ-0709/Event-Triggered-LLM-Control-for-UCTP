"""Inspect room-conflict clusters in one saved ITC 2019 chromosome."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NSGA-III"))
from itc2019.instance import load_instance  # noqa: E402
from itc2019.scoring import distribution_violations, evaluate, overlaps  # noqa: E402


def inspect(instance_path: Path, choices_path: Path) -> None:
    instance = load_instance(instance_path)
    data = np.load(choices_path)
    times_choice = data["time_indices"]
    rooms_choice = data["room_indices"]
    score, _ = evaluate(instance, times_choice, rooms_choice)
    times = [cls.times[int(times_choice[i])] for i, cls in enumerate(instance.classes)]
    rooms = [None if cls.no_room else cls.room_options[int(rooms_choice[i])][0]
             for i, cls in enumerate(instance.classes)]
    by_room: dict[str, list[int]] = defaultdict(list)
    for index, room_id in enumerate(rooms):
        if room_id is not None:
            by_room[room_id].append(index)
    clashes: list[tuple[str, int, int]] = []
    for room_id, indices in by_room.items():
        for pos, left in enumerate(indices):
            for right in indices[pos + 1:]:
                if overlaps(times[left], times[right]):
                    clashes.append((room_id, left, right))
    by_class = Counter(index for _, left, right in clashes for index in (left, right))
    required_by_class = defaultdict(list)
    for distribution in instance.distributions:
        if distribution.required:
            for index in distribution.classes:
                required_by_class[index].append(distribution)

    def local_hard(index: int) -> int:
        room_id = rooms[index]
        collisions = (sum(overlaps(times[index], times[other])
                          for other in by_room[room_id] if other != index)
                      if room_id is not None else 0)
        unavailable = (any(overlaps(times[index], block)
                           for block in instance.rooms[room_id].unavailable)
                       if room_id is not None else False)
        distributions = sum(distribution_violations(instance, distribution, times, rooms)
                            for distribution in required_by_class[index])
        return collisions + int(unavailable) + distributions
    print("instance", instance.name, "classes", len(instance.classes), "students", len(instance.students))
    print("hard", score.hard, "room", score.room_conflicts, "distribution", score.hard_distributions)
    print("room_clashes", dict(Counter(room for room, _, _ in clashes)))
    print("clash_pairs", [(room, instance.classes[left].id, instance.classes[right].id)
                          for room, left, right in clashes])
    print("room_unavailable", [
        (instance.classes[index].id, room_id)
        for index, room_id in enumerate(rooms)
        if room_id is not None and any(overlaps(times[index], blocked)
                                       for blocked in instance.rooms[room_id].unavailable)
    ])
    for index, count in by_class.most_common():
        cls = instance.classes[index]
        original_time, original_room = times[index], rooms[index]
        old_local = local_hard(index)
        option_deltas = []
        for time_option in cls.times:
            times[index] = time_option
            for room_id, _ in cls.room_options:
                rooms[index] = room_id
                option_deltas.append(local_hard(index) - old_local)
        times[index], rooms[index] = original_time, original_room
        required = [(distribution.kind, tuple(instance.classes[j].id for j in distribution.classes))
                    for distribution in instance.distributions
                    if distribution.required and index in distribution.classes and
                    distribution_violations(instance, distribution, times, rooms)]
        print("class", cls.id, "clashes", count, "time_options", len(cls.times),
              "room_options", len(cls.room_options), "room", rooms[index],
              "best_local_delta", min(option_deltas),
              "neutral_options", sum(value == 0 for value in option_deltas),
              "violated_required", required[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("choices", type=Path)
    args = parser.parse_args()
    inspect(args.instance, args.choices)
