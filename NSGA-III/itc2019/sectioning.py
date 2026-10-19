"""Deterministic bounded course reallocation when greedy sectioning fails.

All students requesting a course share its structural configuration choices in
ITC 2019. A capacity-feasible bundle-count plan is therefore independent of the
timetable and is safely cached per instance/course; only soft matching varies.
"""
from collections import defaultdict
from itertools import combinations


def course_plan(instance, course_id, max_bundles=4096, max_nodes=200000):
    if course_id in instance.sectioning_cache:
        return instance.sectioning_cache[course_id]
    from ortools.sat.python import cp_model
    requested = sum(course_id in s.courses for s in instance.students)
    bundles = []
    complete = True
    nodes = 0
    for config in instance.courses[course_id].configs:
        owner = {i: k for k, part in enumerate(config.subparts) for i in part}
        dependencies = {k: {owner[instance.class_index[instance.classes[i].parent_id]]
                           for i in part if instance.classes[i].parent_id is not None and
                           instance.class_index[instance.classes[i].parent_id] in owner}
                        for k, part in enumerate(config.subparts)}
        order = []
        while len(order) < len(config.subparts):
            ready = [k for k in dependencies if k not in order and dependencies[k] <= set(order)]
            if not ready:
                break
            order.append(min(ready, key=lambda k: (len(config.subparts[k]), k)))
        if len(order) != len(config.subparts):
            continue
        beam = [()]
        for part_index in order:
            expanded = []
            for selected in beam:
                ids = {instance.classes[i].id for i in selected}
                for i in config.subparts[part_index]:
                    nodes += 1
                    if nodes > max_nodes:
                        complete = False
                        break
                    cls = instance.classes[i]
                    if cls.limit == 0 or cls.parent_id is not None and cls.parent_id not in ids:
                        continue
                    expanded.append((*selected, i))
                if nodes > max_nodes:
                    break
            if len(expanded) > max_bundles:
                complete = False
                # Preserve coverage of root choices rather than a DFS prefix only.
                groups = defaultdict(list)
                for row in expanded:
                    groups[row[0]].append(row)
                expanded = [row[j] for j in range(max(map(len, groups.values())))
                            for row in groups.values() if j < len(row)][:max_bundles]
            beam = expanded
            if not beam:
                break
        if beam and all(len(row) == len(config.subparts) for row in beam):
            bundles.extend(beam)
    bundles = list(dict.fromkeys(bundles))
    if len(bundles) > max_bundles:
        complete = False
        bundles = bundles[:max_bundles]
    result = {"status": "no_bundles", "enumeration_complete": complete, "plan": ()}
    if bundles:
        model = cp_model.CpModel()
        counts = [model.new_int_var(0, requested, f"bundle_{i}") for i in range(len(bundles))]
        model.add(sum(counts) == requested)
        memberships = defaultdict(list)
        for i, bundle in enumerate(bundles):
            for cls in bundle:
                memberships[cls].append(i)
        for cls, rows in memberships.items():
            cap = instance.classes[cls].limit
            if cap is not None:
                model.add(sum(counts[i] for i in rows) <= cap)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_deterministic_time = 0.5
        solver.parameters.stop_after_first_solution = True
        status = solver.solve(model)
        result["status"] = solver.status_name(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result["plan"] = tuple((bundle, solver.value(count)) for bundle, count in zip(bundles, counts)
                                    if solver.value(count))
    instance.sectioning_cache[course_id] = result
    return result


def repair_sectioning(instance, times, rooms, assignments):
    from .scoring import attendee_conflict
    ledger = {s.id: {} for s in instance.students}
    for cls, students in assignments.items():
        course = instance.classes[cls].course_id
        for student in students:
            ledger[student].setdefault(course, []).append(cls)
    failed = sorted({course for s in instance.students for course in s.courses
                     if course not in ledger[s.id]})
    pair_cache = {}

    def conflict(a, b):
        key = min(a, b), max(a, b)
        if key not in pair_cache:
            pair_cache[key] = int(attendee_conflict(instance, times[a], rooms[a], times[b], rooms[b]))
        return pair_cache[key]

    for course in failed:
        structural = course_plan(instance, course)
        if not structural["plan"]:
            continue
        options = [list(row) for row in structural["plan"]]
        students = sorted((s for s in instance.students if course in s.courses),
                          key=lambda s: (-len(s.courses), s.id))
        for student in students:
            other = [cls for c, members in ledger[student.id].items() if c != course for cls in members]
            available = [(sum(conflict(a, b) for a in bundle for b in other) +
                          sum(conflict(a, b) for a, b in combinations(bundle, 2)), bundle, j)
                         for j, (bundle, count) in enumerate(options) if count]
            _, bundle, j = min(available)
            options[j][1] -= 1
            ledger[student.id][course] = list(bundle)
    rebuilt = defaultdict(list)
    unassigned = conflicts = 0
    for student in instance.students:
        for course in student.courses:
            unassigned += course not in ledger[student.id]
        attending = sorted(cls for members in ledger[student.id].values() for cls in members)
        conflicts += sum(conflict(a, b) for a, b in combinations(attending, 2))
        for cls in attending:
            rebuilt[cls].append(student.id)
    return unassigned, conflicts, rebuilt
