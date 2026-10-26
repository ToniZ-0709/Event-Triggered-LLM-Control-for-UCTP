"""Bounded finite-domain repair that releases classes blocking room alternatives."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
import time

import numpy as np

from .instance import Distribution, Instance
from .scoring import Score, distribution_violations, evaluate, overlaps, _blocks


class _BuildDeadline(Exception):
    pass

_GLOBAL = {"MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock"}


class PlacementNeighborhood:
    """Pairwise hard costs against an incumbent, with explicit fixed boundaries."""

    def __init__(self, instance: Instance, source: np.ndarray) -> None:
        self.instance = instance
        self.n = len(instance.classes)
        self.source = np.asarray(source, dtype=np.int32).copy()
        self.times = [cls.times[int(source[i])] for i, cls in enumerate(instance.classes)]
        self.rooms = [None if cls.no_room else cls.room_options[int(source[self.n + i])][0]
                      for i, cls in enumerate(instance.classes)]
        self.by_room: dict[str, list[int]] = defaultdict(list)
        for i, room in enumerate(self.rooms):
            if room is not None:
                self.by_room[room].append(i)
        self.pairs: dict[tuple[int, int], list[Distribution]] = defaultdict(list)
        self.related: list[set[int]] = [set() for _ in instance.classes]
        self.globals = []
        for distribution in instance.distributions:
            if not distribution.required:
                continue
            if distribution.kind in _GLOBAL:
                self.globals.append(distribution)
                continue
            for left, right in combinations(distribution.classes, 2):
                pair = (min(left, right), max(left, right))
                self.pairs[pair].append(replace(distribution, classes=(left, right)))
                self.related[left].add(right)
                self.related[right].add(left)

    def placement(self, index: int, option: tuple[int, int]):
        cls = self.instance.classes[index]
        ti, ri = option
        return cls.times[ti], None if cls.no_room else cls.room_options[ri][0]

    def current(self, index: int) -> tuple[int, int]:
        return int(self.source[index]), int(self.source[self.n + index])

    def pair_cost(self, left: int, a: tuple[int, int], right: int,
                  b: tuple[int, int]) -> int:
        ta, ra = self.placement(left, a)
        tb, rb = self.placement(right, b)
        cost = int(ra is not None and ra == rb and overlaps(ta, tb))
        distributions = self.pairs.get((min(left, right), max(left, right)), ())
        if distributions:
            old_a, old_b = self.times[left], self.times[right]
            old_ra, old_rb = self.rooms[left], self.rooms[right]
            self.times[left], self.times[right] = ta, tb
            self.rooms[left], self.rooms[right] = ra, rb
            cost += sum(distribution_violations(self.instance, d, self.times, self.rooms)
                        for d in distributions)
            self.times[left], self.times[right] = old_a, old_b
            self.rooms[left], self.rooms[right] = old_ra, old_rb
        return cost

    def external_cost(self, index: int, option: tuple[int, int], active: set[int]
                      ) -> tuple[int, set[int]]:
        chosen, room = self.placement(index, option)
        unavailable = int(room is not None and any(
            overlaps(chosen, block) for block in self.instance.rooms[room].unavailable))
        others = set(self.related[index])
        if room is not None:
            others.update(self.by_room.get(room, ()))
        blockers = set()
        cost = unavailable
        for other in sorted(others - active - {index}):
            amount = self.pair_cost(index, option, other, self.current(other))
            cost += amount
            if amount:
                blockers.add(other)
        return cost, blockers

    def options(self, index: int) -> list[tuple[int, int]]:
        cls = self.instance.classes[index]
        return [(ti, ri) for ti in range(len(cls.times))
                for ri in ([-1] if cls.no_room else range(len(cls.room_options)))]

    def seeds(self) -> list[int]:
        seeds = Counter()
        for index, room in enumerate(self.rooms):
            if room is not None and any(overlaps(self.times[index], block)
                                        for block in self.instance.rooms[room].unavailable):
                seeds[index] += 1
        for members in self.by_room.values():
            for left, right in combinations(members, 2):
                if overlaps(self.times[left], self.times[right]):
                    seeds[left] += 1
                    seeds[right] += 1
        return sorted(seeds, key=lambda i: (-seeds[i], i))

    def expand(self, max_classes: int, deadline: float, seeds=None) -> list[int]:
        active = set((self.seeds() if seeds is None else list(dict.fromkeys(seeds)))[:max_classes])
        expanded = set()
        while len(active) < max_classes and active - expanded and time.perf_counter() < deadline:
            additions = Counter()
            for index in sorted(active - expanded):
                expanded.add(index)
                ranked = []
                for option in self.options(index):
                    if time.perf_counter() >= deadline:
                        break
                    if option == self.current(index):
                        continue
                    cost, blockers = self.external_cost(index, option, active)
                    ranked.append((cost, len(blockers), option, blockers))
                ranked.sort(key=lambda item: item[:3])
                distinct = []
                signatures = set()
                for item in ranked:
                    signature = tuple(sorted(item[3]))
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    distinct.append(item)
                    if len(distinct) == 4:
                        break
                for _, _, _, blockers in distinct:
                    for blocker in blockers:
                        additions[blocker] += 1
            for index in sorted(additions, key=lambda i: (-additions[i], i)):
                active.add(index)
                if len(active) == max_classes:
                    break
        return sorted(active)


def room_neighborhood_repair(instance: Instance, choices: np.ndarray, *,
                             deadline: float, seed: int = 42, max_classes: int = 32,
                             max_options: int = 32, deterministic_limit: float = 0.5,
                             max_table_cells: int = 200000,
                             source_score: Score | None = None,
                             seed_classes=None, require_room_improvement: bool = True,
                             allowed_classes=None, allow_soft_improvement: bool = False,
                             target_distribution_index: int | None = None,
                             max_full_scoring_calls: int = 5, score_fn=None
                             ) -> tuple[np.ndarray | None, dict]:
    """Solve a small hard-cost model, then check the entire modified timetable.

    The local model includes pairwise/global required distributions and room
    conflicts against the fixed boundary. Student sectioning and the complete
    timetable are checked by the full scorer before returning an improvement.
    """
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    source = np.asarray(choices, dtype=np.int32).copy()
    n = len(instance.classes)
    stats = {"status": "no_room_targets", "full_scoring_calls": 0,
             "active_classes": 0, "options": 0, "pair_tables": 0,
             "changed_classes": 0, "solver_deterministic_time": 0.0}
    if max_full_scoring_calls <= 0:
        stats["status"] = "evaluation_budget_exhausted"
        return None, stats
    if time.perf_counter() >= deadline:
        stats["status"] = "time_budget_exhausted"
        return None, stats
    if source_score is None:
        source_score = score_fn(source) if score_fn else evaluate(instance, source[:n], source[n:])[0]
        stats["full_scoring_calls"] += 1
    neighborhood = PlacementNeighborhood(instance, source)
    active = (sorted(set(allowed_classes)) if allowed_classes is not None else
              neighborhood.expand(max_classes, deadline, seed_classes))
    active = active[:max_classes]
    stats["active_classes"] = len(active)
    stats["active_class_ids"] = [instance.classes[i].id for i in active]
    if not active:
        return None, stats
    active_set = set(active)
    domains = {}
    unary = {}
    originals = {}
    for index in active:
        ranked = []
        current = neighborhood.current(index)
        for option in neighborhood.options(index):
            if time.perf_counter() >= deadline:
                stats["status"] = "time_budget_exhausted"
                return None, stats
            cost, _ = neighborhood.external_cost(index, option, active_set)
            chosen, room = neighborhood.placement(index, option)
            soft = chosen.penalty + (0 if room is None else
                                      instance.classes[index].room_options[option[1]][1])
            ranked.append((cost, option != current, soft, option))
        ranked.sort()
        selected = ranked[:max_options]
        if not any(item[3] == current for item in selected):
            selected[-1] = next(item for item in ranked if item[3] == current)
        domains[index] = [item[3] for item in selected]
        unary[index] = [item[0] for item in selected]
        originals[index] = domains[index].index(current)
    stats["options"] = sum(map(len, domains.values()))
    stats["external_zero_options"] = {instance.classes[i].id: unary[i].count(0) for i in active}
    room_domains = {i: {neighborhood.placement(i, option)[1] for option in domains[i]} - {None}
                    for i in active}
    pairs = [(a, b) for a, b in combinations(active, 2)
             if room_domains[a].intersection(room_domains[b]) or (a, b) in neighborhood.pairs]
    cells = sum(len(domains[a]) * len(domains[b]) for a, b in pairs)
    if cells > max_table_cells:
        cap = max(4, int((max_table_cells / max(1, len(pairs))) ** 0.5))
        for index in active:
            kept = list(range(min(cap, len(domains[index]))))
            if originals[index] not in kept:
                kept[-1] = originals[index]
            domains[index] = [domains[index][j] for j in kept]
            unary[index] = [unary[index][j] for j in kept]
            originals[index] = kept.index(originals[index])
        stats["options"] = sum(map(len, domains.values()))
        cells = sum(len(domains[a]) * len(domains[b]) for a, b in pairs)
        stats["resource_domain_cap"] = cap
    stats["table_cells"] = cells
    stats["external_zero_options"] = {instance.classes[i].id: unary[i].count(0) for i in active}
    model = cp_model.CpModel()
    variables = {i: model.new_int_var(0, len(domains[i]) - 1, f"placement_{i}") for i in active}
    costs = []
    changed = []
    baseline_local = 0
    for index in active:
        cost = model.new_int_var(0, max(unary[index]), f"outside_{index}")
        model.add_element(variables[index], unary[index], cost)
        costs.append(cost)
        baseline_local += unary[index][originals[index]]
        moved = model.new_bool_var(f"moved_{index}")
        model.add(variables[index] != originals[index]).only_enforce_if(moved)
        model.add(variables[index] == originals[index]).only_enforce_if(moved.Not())
        changed.append(moved)
        model.add_hint(variables[index], originals[index])
    for left, right in pairs:
        if time.perf_counter() >= deadline:
            stats["status"] = "time_budget_exhausted"
            return None, stats
        table = []
        for a, option_a in enumerate(domains[left]):
            if time.perf_counter() >= deadline:
                stats["status"] = "time_budget_exhausted"
                return None, stats
            table.extend((a, b, neighborhood.pair_cost(left, option_a, right, option_b))
                         for b, option_b in enumerate(domains[right]))
        maximum = max(row[2] for row in table)
        baseline_local += next(row[2] for row in table
                               if row[:2] == (originals[left], originals[right]))
        if maximum:
            cost = model.new_int_var(0, maximum, f"pair_{left}_{right}")
            model.add_allowed_assignments([variables[left], variables[right], cost], table)
            costs.append(cost)
            stats["pair_tables"] += 1
    try:
        global_before, global_count = _global_costs(
            model, variables, domains, originals, active_set, neighborhood, costs, deadline)
    except _BuildDeadline:
        stats["status"] = "global_model_budget_exhausted"
        return None, stats
    baseline_local += global_before
    stats["modeled_global_constraints"] = global_count
    if target_distribution_index is not None:
        rule = instance.distributions[target_distribution_index]
        target_before = distribution_violations(instance, rule, neighborhood.times, neighborhood.rooms)
        target_costs = []
        try:
            if rule.kind in _GLOBAL:
                original_globals = neighborhood.globals
                neighborhood.globals = [rule]
                try:
                    _global_costs(model, variables, domains, originals, active_set,
                                  neighborhood, target_costs, deadline)
                finally:
                    neighborhood.globals = original_globals
            else:
                for left, right in combinations(rule.classes, 2):
                    if time.perf_counter() >= deadline:
                        raise _BuildDeadline
                    if left not in active_set and right not in active_set:
                        fixed = distribution_violations(instance, replace(rule, classes=(left,right)),
                                                        neighborhood.times, neighborhood.rooms)
                        if fixed:
                            target_costs.append(fixed)
                        continue
                    a_values = range(len(domains[left])) if left in active_set else [None]
                    b_values = range(len(domains[right])) if right in active_set else [None]
                    table = []
                    for a in a_values:
                        for b in b_values:
                            times, rooms = neighborhood.times.copy(), neighborhood.rooms.copy()
                            for i, value in ((left,a), (right,b)):
                                if value is not None:
                                    times[i], rooms[i] = neighborhood.placement(i, domains[i][value])
                            value = distribution_violations(instance, replace(rule, classes=(left,right)), times, rooms)
                            table.append(tuple(v for v in (a,b) if v is not None) + (value,))
                    cost = model.new_int_var(0, max(row[-1] for row in table), f"target_{left}_{right}")
                    model.add_allowed_assignments([variables[i] for i in (left,right) if i in active_set] + [cost], table)
                    target_costs.append(cost)
            model.add(sum(target_costs) < target_before)
            stats["targeted_rule"] = f"distribution:{target_distribution_index}"
        except _BuildDeadline:
            stats["status"] = "target_model_budget_exhausted"
            return None, stats
    if baseline_local == 0 and not (allow_soft_improvement and source_score.hard == 0):
        stats["status"] = "no_modeled_violation"
        return None, stats
    model.add(sum(costs) <= baseline_local if allow_soft_improvement and source_score.hard == 0
              else sum(costs) < baseline_local)
    if allow_soft_improvement and source_score.hard == 0:
        soft_costs = []
        for i in active:
            cls = instance.classes[i]
            values = [cls.times[ti].penalty * instance.weights[0] +
                      (0 if ri == -1 else cls.room_options[ri][1] * instance.weights[1])
                      for ti, ri in domains[i]]
            value = model.new_int_var(0, max(values), f"soft_{i}")
            model.add_element(variables[i], values, value)
            soft_costs.append(value)
        model.minimize(sum(soft_costs) * (len(active) + 1) + sum(changed))
    else:
        model.minimize(sum(costs) * (len(active) + 1) + sum(changed))
    zero_cost = model.new_bool_var("try_without_local_violations")
    model.add(sum(costs) == 0).only_enforce_if(zero_cost)
    model.add_assumption(zero_cost)
    stats["local_hard_before"] = baseline_local
    stats["build_seconds"] = round(time.perf_counter() - started, 3)
    for attempt in range(4):
        if stats["full_scoring_calls"] >= max_full_scoring_calls:
            stats["status"] = "evaluation_budget_exhausted"
            break
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            stats["status"] = "time_budget_exhausted"
            break
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed % 2147483647
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.max_deterministic_time = deterministic_limit
        solver.parameters.max_memory_in_mb = 512
        solver.parameters.stop_after_first_solution = not (allow_soft_improvement and source_score.hard == 0)
        status = solver.solve(model)
        stats["status"] = solver.status_name(status)
        stats["solver_deterministic_time"] += solver.response_proto.deterministic_time
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if attempt == 0:
                stats["zero_cost_status"] = stats["status"]
                model.clear_assumptions()
                continue
            break
        candidate = source.copy()
        selected = [solver.value(variables[i]) for i in active]
        for index, choice in zip(active, selected):
            candidate[index], candidate[n + index] = domains[index][choice]
        score = score_fn(candidate) if score_fn else evaluate(instance, candidate[:n], candidate[n:])[0]
        stats["full_scoring_calls"] += 1
        if (stats.get("best_trial_quality") is None or
                (score.hard, score.total) < tuple(stats["best_trial_quality"])):
            stats["best_trial_quality"] = [score.hard, score.total]
            stats["best_trial_weighted"] = list(score.weighted)
        stats["local_hard_after"] = sum(solver.value(cost) for cost in costs)
        hard_improved = score.hard < source_score.hard
        soft_improved = (allow_soft_improvement and source_score.hard == score.hard == 0
                         and score.total < source_score.total)
        target_improved = True
        if target_distribution_index is not None:
            times = [c.times[int(candidate[i])] for i,c in enumerate(instance.classes)]
            rooms = [None if c.no_room else c.room_options[int(candidate[n+i])][0]
                     for i,c in enumerate(instance.classes)]
            target_improved = distribution_violations(instance, instance.distributions[target_distribution_index],
                                                      times, rooms) < target_before
        if ((hard_improved or soft_improved) and target_improved and (not require_room_improvement or
                score.room_conflicts + score.room_unavailable <
                source_score.room_conflicts + source_score.room_unavailable)):
            stats["changed_classes"] = int(sum(
                candidate[i] != source[i] or candidate[n + i] != source[n + i] for i in active))
            stats["status"] = "improved"
            stats["candidate_hard"] = score.hard
            return candidate, stats
        model.add_forbidden_assignments([variables[i] for i in active], [selected])
        model.clear_assumptions()
    return None, stats


def distribution_neighborhood_repair(instance, choices, *, deadline, seed=42,
                                      max_classes=32, max_options=32, source_score=None,
                                      max_full_scoring_calls=5, score_fn=None):
    """Seed a joint neighborhood from the most violated required constraint."""
    state = PlacementNeighborhood(instance, choices)
    ranked = []
    for number, distribution in enumerate(instance.distributions):
        if time.perf_counter() >= deadline:
            break
        if distribution.required:
            amount = distribution_violations(instance, distribution, state.times, state.rooms)
            if amount:
                ranked.append((-amount, len(distribution.classes), number, distribution))
    if not ranked:
        return None, {"status": "no_distribution_targets", "full_scoring_calls": 0}
    distribution = min(ranked, key=lambda row: row[:3])[3]
    candidate, stats = room_neighborhood_repair(
        instance, choices, deadline=deadline, seed=seed, max_classes=max_classes,
        max_options=max_options, source_score=source_score,
        seed_classes=distribution.classes, require_room_improvement=False,
        max_full_scoring_calls=max_full_scoring_calls, score_fn=score_fn)
    stats["target_constraint"] = distribution.kind
    return candidate, stats


def _global_costs(model, variables, domains, originals, active, neighborhood, costs, deadline):
    """Encode required global distributions with the outside timetable fixed."""
    before = count = 0
    load_cache = {}

    def element(index, values):
        if len(set(values)) == 1:
            return values[0]
        key = index, values
        if key not in load_cache:
            value = model.new_int_var(min(values), max(values), f"load_{len(load_cache)}")
            model.add_element(variables[index], list(values), value)
            load_cache[key] = value
        return load_cache[key]

    for number, distribution in enumerate(neighborhood.globals):
        if time.perf_counter() >= deadline:
            raise _BuildDeadline()
        indices = sorted(active.intersection(distribution.classes))
        if not indices:
            continue
        outside = [i for i in distribution.classes if i not in active]
        if distribution.kind == "MaxDays":
            outside_mask = 0
            for i in outside:
                outside_mask |= neighborhood.times[i].day_mask
            days = []
            for day in range(neighborhood.instance.nr_days):
                bit = 1 << day
                if outside_mask & bit:
                    days.append(1)
                    continue
                presences = [element(i, tuple(int(neighborhood.placement(i, option)[0].day_mask & bit != 0)
                                              for option in domains[i])) for i in indices]
                present = model.new_bool_var(f"day_{number}_{day}")
                model.add_max_equality(present, presences)
                days.append(present)
            maximum = max(0, neighborhood.instance.nr_days - distribution.params[0])
            excess = model.new_int_var(0, maximum, f"days_excess_{number}")
            model.add_max_equality(excess, [sum(days) - distribution.params[0], 0])
            costs.append(excess)
            before += distribution_violations(neighborhood.instance, distribution,
                                               neighborhood.times, neighborhood.rooms)
        elif distribution.kind == "MaxDayLoad":
            patterns = Counter()
            for week in range(neighborhood.instance.nr_weeks):
                for day in range(neighborhood.instance.nr_days):
                    week_bit, day_bit = 1 << week, 1 << day
                    fixed = sum(neighborhood.times[i].length for i in outside
                                if neighborhood.times[i].week_mask & week_bit and
                                neighborhood.times[i].day_mask & day_bit)
                    vectors = tuple(tuple(chosen.length if chosen.week_mask & week_bit and
                                          chosen.day_mask & day_bit else 0
                                          for chosen in (neighborhood.placement(i, option)[0]
                                                         for option in domains[i])) for i in indices)
                    if any(any(values) for values in vectors):
                        patterns[fixed, vectors] += 1
            for pattern_index, ((fixed, vectors), multiplicity) in enumerate(patterns.items()):
                maximum = max(0, fixed + sum(max(values) for values in vectors) - distribution.params[0])
                if not maximum:
                    continue
                loads = [element(i, values) for i, values in zip(indices, vectors)]
                excess = model.new_int_var(0, maximum, f"load_excess_{number}_{pattern_index}")
                model.add_max_equality(excess, [fixed + sum(loads) - distribution.params[0], 0])
                costs.append(excess * multiplicity)
                current = fixed + sum(values[originals[i]] for i, values in zip(indices, vectors))
                before += max(0, current - distribution.params[0]) * multiplicity
        else:
            before += _interval_costs(model, variables, domains, originals, indices,
                                      outside, neighborhood, distribution, costs, deadline, number)
        count += 1
    return before, count


def _interval_costs(model, variables, domains, originals, indices, outside, state,
                    distribution, costs, deadline, number):
    """Exact optional-interval block scan; duplicate week/day patterns share a model."""
    patterns = Counter()
    for week in range(state.instance.nr_weeks):
        for day in range(state.instance.nr_days):
            if time.perf_counter() >= deadline:
                raise _BuildDeadline()
            fixed = Counter((state.times[i].start, state.times[i].end) for i in outside
                            if state.times[i].week_mask & (1 << week) and
                            state.times[i].day_mask & (1 << day))
            moving = tuple((i, tuple((j, chosen.start, chosen.end) for j, option in enumerate(domains[i])
                                     if (chosen := state.placement(i, option)[0]).week_mask & (1 << week)
                                     and chosen.day_mask & (1 << day))) for i in indices)
            if any(entries for _, entries in moving):
                patterns[tuple(sorted(fixed.items())), moving] += 1
    before = 0
    nodes = 0

    def conjunction(a, b, name):
        v = model.new_bool_var(name)
        model.add(v <= a)
        model.add(v <= b)
        model.add(v >= a+b-1)
        return v

    for pattern, ((fixed_items, moving), multiplicity) in enumerate(patterns.items()):
        fixed = dict(fixed_items)
        memberships = defaultdict(lambda: defaultdict(list))
        for i, entries in moving:
            for j, start, end in entries:
                memberships[start, end][i].append(j)
        intervals = sorted(set(fixed) | set(memberships))
        nodes += len(intervals)
        if nodes > 2000:
            raise _BuildDeadline()
        prefix = f"blocks_{number}_{pattern}"
        cap = len(distribution.classes)
        horizon = max(end for _, end in intervals)
        prev_start, prev_end, prev_count = 0, -1, 0
        opens, closed_bad = [], []
        current_intervals = []
        for k, (start, end) in enumerate(intervals):
            if time.perf_counter() >= deadline:
                raise _BuildDeadline()
            terms = [fixed.get((start, end), 0)]
            current_count = terms[0]
            for i, option_ids in memberships[start, end].items():
                values = [int(j in option_ids) for j in range(len(domains[i]))]
                present = model.new_bool_var(f"{prefix}_member_{k}_{i}")
                model.add_element(variables[i], values, present)
                terms.append(present)
                current_count += values[originals[i]]
            current_intervals.extend([(start, end)] * current_count)
            amount = model.new_int_var(0, cap, f"{prefix}_amount_{k}")
            model.add(amount == sum(terms))
            present = model.new_bool_var(f"{prefix}_present_{k}")
            model.add(amount >= 1).only_enforce_if(present)
            model.add(amount == 0).only_enforce_if(present.Not())
            separate = model.new_bool_var(f"{prefix}_separate_{k}")
            model.add(start > prev_end + distribution.params[1]).only_enforce_if(separate)
            model.add(start <= prev_end + distribution.params[1]).only_enforce_if(separate.Not())
            empty = model.new_bool_var(f"{prefix}_empty_{k}")
            model.add(prev_count == 0).only_enforce_if(empty)
            model.add(prev_count >= 1).only_enforce_if(empty.Not())
            boundary = model.new_bool_var(f"{prefix}_boundary_{k}")
            model.add_max_equality(boundary, [empty, separate])
            new = conjunction(present, boundary, f"{prefix}_new_{k}")
            same = conjunction(present, boundary.Not(), f"{prefix}_same_{k}")
            opens.append(new)

            def bad_block(start_var, end_var, count_var, suffix):
                multiple = model.new_bool_var(f"{prefix}_multiple_{suffix}")
                model.add(count_var > 1).only_enforce_if(multiple)
                model.add(count_var <= 1).only_enforce_if(multiple.Not())
                long = model.new_bool_var(f"{prefix}_long_{suffix}")
                model.add(end_var-start_var > distribution.params[0]).only_enforce_if(long)
                model.add(end_var-start_var <= distribution.params[0]).only_enforce_if(long.Not())
                return conjunction(multiple, long, f"{prefix}_bad_{suffix}")

            if distribution.kind == "MaxBlock":
                closed_bad.append(conjunction(new, bad_block(prev_start, prev_end, prev_count, k),
                                              f"{prefix}_closed_{k}"))
            nxt_start = model.new_int_var(0, horizon, f"{prefix}_start_{k}")
            nxt_end = model.new_int_var(-1, horizon, f"{prefix}_end_{k}")
            nxt_count = model.new_int_var(0, cap, f"{prefix}_count_{k}")
            maximum_end = model.new_int_var(-1, horizon, f"{prefix}_maxend_{k}")
            model.add_max_equality(maximum_end, [prev_end, end])
            for variable, old, first, continuing in (
                (nxt_start, prev_start, start, prev_start), (nxt_end, prev_end, end, maximum_end),
                (nxt_count, prev_count, amount, prev_count+amount)):
                model.add(variable == old).only_enforce_if(present.Not())
                model.add(variable == first).only_enforce_if(new)
                model.add(variable == continuing).only_enforce_if(same)
            prev_start, prev_end, prev_count = nxt_start, nxt_end, nxt_count
        blocks = _blocks(current_intervals, distribution.params[1])
        if distribution.kind == "MaxBreaks":
            excess = model.new_int_var(0, len(intervals), f"{prefix}_excess")
            model.add_max_equality(excess, [sum(opens)-distribution.params[0]-1, 0])
            costs.append(excess*multiplicity)
            before += max(0, len(blocks)-distribution.params[0]-1)*multiplicity
        else:
            closed_bad.append(bad_block(prev_start, prev_end, prev_count, "last"))
            costs.append(sum(closed_bad)*multiplicity)
            before += sum(count > 1 and end-start > distribution.params[0]
                          for start, end, count in blocks)*multiplicity
    return before


def adaptive_room_repair(instance, choices, *, deadline, seed=42, max_classes=32,
                          max_options=32, deterministic_limit=0.5, source_score=None,
                          max_full_scoring_calls=5, score_fn=None):
    """Widen an unsuccessful neighborhood within one shared deadline/check budget."""
    calls = 0
    if max_full_scoring_calls <= 0 or time.perf_counter() >= deadline:
        return None, {"status": "budget_exhausted", "full_scoring_calls": 0, "attempts": []}
    if source_score is None:
        n = len(instance.classes)
        source_score = score_fn(choices) if score_fn else evaluate(instance, choices[:n], choices[n:])[0]
        calls = 1
    attempts = []
    sizes = list(dict.fromkeys([(min(16, max_classes), min(16, max_options)),
                                (max_classes, max_options),
                                (min(64, max_classes*2), min(64, max_options*2))]))
    for position, (classes, options) in enumerate(sizes):
        remaining = deadline-time.perf_counter()
        if remaining <= 0 or calls >= max_full_scoring_calls:
            break
        attempt_deadline = deadline if position == len(sizes)-1 else time.perf_counter()+remaining*0.45
        candidate, stats = room_neighborhood_repair(
            instance, choices, deadline=attempt_deadline, seed=seed, max_classes=classes,
            max_options=options, deterministic_limit=deterministic_limit, source_score=source_score,
            max_full_scoring_calls=max_full_scoring_calls-calls, score_fn=score_fn)
        calls += stats["full_scoring_calls"]
        attempts.append(stats)
        if candidate is not None:
            break
    else:
        candidate = None
    if not attempts:
        return None, {"status": "budget_exhausted", "full_scoring_calls": calls, "attempts": []}
    result = dict(attempts[-1])
    result.update(full_scoring_calls=calls, attempts=attempts)
    return candidate, result
