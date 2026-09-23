"""ITC 2019 penalties and feasibility for a complete class placement.

Distribution formulas follow Muller, Rudova and Mullerova (PATAT 2018),
https://www.itc2019.org/format . Student sectioning is a deterministic greedy
decoder; failure to section a requested course is reported as a hard violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict

from .instance import Distribution, Instance, Time


@dataclass(frozen=True, slots=True)
class Score:
    hard: int
    time: int
    room: int
    distribution: int
    student: int
    weighted: tuple[int, int, int, int]
    total: int
    room_conflicts: int
    room_unavailable: int
    hard_distributions: int
    unassigned_requests: int


def overlaps(a: Time, b: Time) -> bool:
    return bool(a.day_mask & b.day_mask and a.week_mask & b.week_mask and
                a.start < b.end and b.start < a.end)


def attendee_conflict(instance: Instance, a: Time, room_a: str | None,
                      b: Time, room_b: str | None) -> bool:
    if not (a.day_mask & b.day_mask and a.week_mask & b.week_mask):
        return False
    travel = instance.travel(room_a, room_b)
    return not (a.end + travel <= b.start or b.end + travel <= a.start)


def _blocks(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    blocks = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = blocks[-1]
        if start <= previous_end + gap:
            blocks[-1] = (previous_start, max(previous_end, end))
        else:
            blocks.append((start, end))
    return blocks


def distribution_violations(instance: Instance, distribution: Distribution,
                            times: list[Time], room_ids: list[str | None]) -> int:
    """Number of pair violations, or excess units for a special constraint."""
    kind = distribution.kind
    indices = distribution.classes
    if kind == "MaxDays":
        days = 0
        for index in indices:
            days |= times[index].day_mask
        return max(0, days.bit_count() - distribution.params[0])

    if kind in {"MaxDayLoad", "MaxBreaks", "MaxBlock"}:
        excess = 0
        for week in range(instance.nr_weeks):
            week_bit = 1 << (instance.nr_weeks - 1 - week)
            for day in range(instance.nr_days):
                day_bit = 1 << (instance.nr_days - 1 - day)
                meetings = [times[i] for i in indices if
                            times[i].day_mask & day_bit and times[i].week_mask & week_bit]
                if not meetings:
                    continue
                if kind == "MaxDayLoad":
                    excess += max(0, sum(t.length for t in meetings) - distribution.params[0])
                else:
                    blocks = _blocks([(t.start, t.end) for t in meetings], distribution.params[1])
                    if kind == "MaxBreaks":
                        excess += max(0, len(blocks) - distribution.params[0] - 1)
                    else:
                        excess += sum(end - start > distribution.params[0] for start, end in blocks)
        return excess

    count = 0
    for left, right in combinations(indices, 2):
        a, b = times[left], times[right]
        room_a, room_b = room_ids[left], room_ids[right]
        shared = bool(a.day_mask & b.day_mask and a.week_mask & b.week_mask)
        if kind == "SameStart":
            valid = a.start == b.start
        elif kind == "SameTime":
            valid = (a.start <= b.start and b.end <= a.end) or (b.start <= a.start and a.end <= b.end)
        elif kind == "DifferentTime":
            valid = a.end <= b.start or b.end <= a.start
        elif kind == "SameDays":
            valid = (a.day_mask | b.day_mask) in (a.day_mask, b.day_mask)
        elif kind == "DifferentDays":
            valid = not (a.day_mask & b.day_mask)
        elif kind == "SameWeeks":
            valid = (a.week_mask | b.week_mask) in (a.week_mask, b.week_mask)
        elif kind == "DifferentWeeks":
            valid = not (a.week_mask & b.week_mask)
        elif kind == "SameRoom":
            valid = room_a == room_b
        elif kind == "DifferentRoom":
            valid = room_a != room_b
        elif kind == "Overlap":
            valid = overlaps(a, b)
        elif kind == "NotOverlap":
            valid = not overlaps(a, b)
        elif kind == "SameAttendees":
            valid = not attendee_conflict(instance, a, room_a, b, room_b)
        elif kind == "Precedence":
            first_a = (a.weeks.index("1"), a.days.index("1"))
            first_b = (b.weeks.index("1"), b.days.index("1"))
            valid = first_a < first_b or (first_a == first_b and a.end <= b.start)
        elif kind == "WorkDay":
            valid = not shared or max(a.end, b.end) - min(a.start, b.start) <= distribution.params[0]
        elif kind == "MinGap":
            gap = distribution.params[0]
            valid = not shared or a.end + gap <= b.start or b.end + gap <= a.start
        else:
            raise ValueError(f"Unsupported distribution type {kind}")
        count += not valid
    return count


def _section_students(instance: Instance, times: list[Time], room_ids: list[str | None],
                      *, keep_assignments: bool) -> tuple[int, int, dict[int, list[str]] | None]:
    """Greedy, capacity aware sectioning with a small beam for course choices.

    It is a baseline decoder, not an exact sectioning solver. The caller must
    never export a solution if any requested course could not be assigned.
    """
    if not instance.students:
        return 0, 0, {} if keep_assignments else None
    counts = [0] * len(instance.classes)
    assignments: dict[int, list[str]] | None = defaultdict(list) if keep_assignments else None
    unassigned = 0
    conflicts = 0
    # Harder students first reduces, but does not eliminate, greedy failures.
    students = sorted(instance.students, key=lambda s: (-len(s.courses), s.id))
    for student in students:
        attending: list[int] = []
        for course_id in student.courses:
            best: tuple[int, tuple[int, ...]] | None = None
            for config in instance.courses[course_id].configs:
                beam: list[tuple[int, tuple[int, ...]]] = [(0, ())]
                for subpart in config.subparts:
                    next_beam: list[tuple[int, tuple[int, ...]]] = []
                    for cost, selected in beam:
                        selected_ids = {instance.classes[i].id for i in selected}
                        for index in subpart:
                            cls = instance.classes[index]
                            if cls.limit is not None and counts[index] >= cls.limit:
                                continue
                            if cls.parent_id is not None:
                                parent_index = instance.class_index[cls.parent_id]
                                if parent_index not in selected and any(parent_index in s for s in config.subparts[:len(selected)]):
                                    continue
                            extra = sum(attendee_conflict(instance, times[index], room_ids[index],
                                                          times[other], room_ids[other])
                                        for other in (*attending, *selected))
                            next_beam.append((cost + extra, (*selected, index)))
                    if not next_beam:
                        beam = []
                        break
                    next_beam.sort(key=lambda item: (item[0], item[1]))
                    beam = next_beam[:12]
                for candidate in beam:
                    selected_ids = {instance.classes[i].id for i in candidate[1]}
                    if all(instance.classes[i].parent_id in (None, *selected_ids) for i in candidate[1]):
                        if best is None or candidate < best:
                            best = candidate
            if best is None:
                unassigned += 1
                continue
            conflicts += best[0]
            for index in best[1]:
                counts[index] += 1
                attending.append(index)
                if assignments is not None:
                    assignments[index].append(student.id)
    return unassigned, conflicts, assignments


def validate_sectioning(instance: Instance, assignments: dict[int, list[str]]) -> None:
    """Check decoder output against requests, configs, subparts, parents, limits."""
    requested = {student.id: set(student.courses) for student in instance.students}
    by_student: dict[str, set[int]] = defaultdict(set)
    for index, ids in assignments.items():
        if not 0 <= index < len(instance.classes):
            raise ValueError(f"Unknown class index {index} in student assignment")
        cls = instance.classes[index]
        if cls.limit is not None and len(ids) > cls.limit:
            raise ValueError(f"Class {cls.id} exceeds its student limit")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Class {cls.id} has duplicate students")
        for student_id in ids:
            if student_id not in requested or cls.course_id not in requested[student_id]:
                raise ValueError(f"Unexpected enrollment: student {student_id}, class {cls.id}")
            by_student[student_id].add(index)
    for student in instance.students:
        attended = by_student[student.id]
        for course_id in student.courses:
            selected = {i for i in attended if instance.classes[i].course_id == course_id}
            valid = False
            for config in instance.courses[course_id].configs:
                all_classes = set().union(*map(set, config.subparts))
                if not selected <= all_classes:
                    continue
                if any(len(selected.intersection(subpart)) != 1 for subpart in config.subparts):
                    continue
                selected_ids = {instance.classes[i].id for i in selected}
                if all(instance.classes[i].parent_id in (None, *selected_ids) for i in selected):
                    valid = True
                    break
            if not valid:
                raise ValueError(f"Student {student.id} is not correctly sectioned into course {course_id}")


def evaluate(instance: Instance, time_choices: list[int], room_choices: list[int],
             *, keep_assignments: bool = False) -> tuple[Score, dict[int, list[str]] | None]:
    if len(time_choices) != len(instance.classes) or len(room_choices) != len(instance.classes):
        raise ValueError("Chromosome length differs from number of classes")
    selected_times: list[Time] = []
    selected_rooms: list[str | None] = []
    time_cost = room_cost = 0
    by_room: dict[str, list[int]] = defaultdict(list)
    room_unavailable = 0
    for index, cls in enumerate(instance.classes):
        ti, ri = int(time_choices[index]), int(room_choices[index])
        if not 0 <= ti < len(cls.times):
            raise ValueError(f"Class {cls.id}: time index outside domain")
        if cls.no_room:
            if ri != -1:
                raise ValueError(f"Class {cls.id}: room index must be -1")
            room_id = None
        else:
            if not 0 <= ri < len(cls.room_options):
                raise ValueError(f"Class {cls.id}: room index outside domain")
            room_id, room_penalty = cls.room_options[ri]
            room_cost += room_penalty
            by_room[room_id].append(index)
            room_unavailable += any(overlaps(cls.times[ti], block)
                                    for block in instance.rooms[room_id].unavailable)
        selected_times.append(cls.times[ti])
        selected_rooms.append(room_id)
        time_cost += cls.times[ti].penalty

    room_conflicts = 0
    for indices in by_room.values():
        for left, right in combinations(indices, 2):
            room_conflicts += overlaps(selected_times[left], selected_times[right])

    hard_distributions = soft_cost = 0
    for distribution in instance.distributions:
        violations = distribution_violations(instance, distribution, selected_times, selected_rooms)
        if distribution.required:
            hard_distributions += violations
        elif distribution.kind in {"MaxDayLoad", "MaxBreaks", "MaxBlock"}:
            soft_cost += distribution.penalty * violations // instance.nr_weeks
        else:
            soft_cost += distribution.penalty * violations

    unassigned, student_conflicts, assignments = _section_students(
        instance, selected_times, selected_rooms, keep_assignments=keep_assignments)
    weighted = tuple(raw * weight for raw, weight in
                     zip((time_cost, room_cost, soft_cost, student_conflicts), instance.weights))
    hard = room_conflicts + room_unavailable + hard_distributions + unassigned
    score = Score(hard, time_cost, room_cost, soft_cost, student_conflicts,
                  weighted, sum(weighted), room_conflicts, room_unavailable,
                  hard_distributions, unassigned)
    return score, assignments
