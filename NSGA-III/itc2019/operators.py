"""Bounded improvement operators acting only on a selected class region."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np
from pymoo.core.population import Population

from .instance import Instance
from .regions import RegionCandidate, swap_options
from .scoring import Score, overlaps
from .neighborhood import LocalSoftCosts, PlacementNeighborhood, room_neighborhood_repair


OPERATORS = ("move", "swap", "destroy_repair")
_GUIDED_RANKING_SECONDS = 1.0


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
    candidate_origin: str | None = None


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
    candidate_origin: str | None = None


def _placement_options(instance: Instance, index: int, hard: PlacementNeighborhood,
                       soft: LocalSoftCosts, deadline: float | None, *, by_delta: bool = False):
    """Yield local costs and placements with the rest of the timetable fixed."""
    cls = instance.classes[index]
    old_t, old_r = hard.current(index)
    room_options = ((-1, None, 0),) if cls.no_room else tuple(
        (ri, room, penalty) for ri, (room, penalty) in enumerate(cls.room_options))
    active = set()

    def time_cost(chosen):
        return (hard.global_hard_cost(index, chosen), chosen.penalty * instance.weights[0] +
                soft.global_option_cost(index, chosen, hard.times, hard.rooms))

    def placement_cost(ti, ri, room, penalty, time_hard, time_soft):
        pair_hard, _ = hard.external_cost(index, (ti, ri), active)
        soft_amount = (time_soft + penalty * instance.weights[1] +
                       soft.option_cost(index, cls.times[ti], room, hard.times, hard.rooms))
        return time_hard + pair_hard, soft_amount

    source_hard = source_soft = 0
    if by_delta:
        _, room, penalty = room_options[old_r]
        source_hard, source_soft = placement_cost(old_t, old_r, room, penalty, *time_cost(cls.times[old_t]))

    for ti, chosen in enumerate(cls.times):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        time_hard, time_soft = time_cost(chosen)
        for ri, room, penalty in room_options:
            if (ti, ri) == (old_t, old_r):
                continue
            new_hard, new_soft = placement_cost(ti, ri, room, penalty, time_hard, time_soft)
            yield (new_hard - source_hard, new_soft - source_soft), (ti, ri)


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


def _rank_move_options(instance, _source, region, hard, soft, rng, deadline):
    order = [i for i in _rebuild_order(region, rng)
             if len(instance.classes[i].times) > 1 or len(instance.classes[i].room_options) > 1]
    ranked = []
    for index in order:
        if time.perf_counter() >= deadline:
            break
        ranked.extend((delta, (index, ti, ri)) for delta, (ti, ri) in
                      _placement_options(instance, index, hard, soft, deadline, by_delta=True))
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: item[0])
    return ((change,) for _, change in ranked)


def _swap_cost(instance, hard, soft, left, right):
    pair = (left, right)
    active = set(pair)
    hard_cost = sum(hard.external_cost(i, hard.current(i), active)[0] for i in pair)
    hard_cost += hard.pair_cost(left, hard.current(left), right, hard.current(right))
    hard_cost += hard.global_hard_cost_current(pair)
    soft_cost = soft.global_cost(pair, hard.times, hard.rooms)
    for i in pair:
        time_option, room = hard.times[i], hard.rooms[i]
        room_index = hard.current(i)[1]
        room_penalty = 0 if room is None else instance.classes[i].room_options[room_index][1]
        soft_cost += time_option.penalty * instance.weights[0] + room_penalty * instance.weights[1]
        soft_cost += soft.option_cost(i, time_option, room, hard.times, hard.rooms,
                                      exclude=left if i == right else None)
    return hard_cost, soft_cost


def _top_single_options(instance: Instance, region: RegionCandidate,
                        hard: PlacementNeighborhood, soft: LocalSoftCosts,
                        rng: np.random.Generator, deadline: float | None,
                        *, limit: int = 4, per_class_seconds: float = 0.10
                        ) -> dict[int, list[tuple[int, int]]]:
    """Collect the top locally ranked single-class moves for each mutable region class."""
    order = [i for i in _rebuild_order(region, rng)
             if len(instance.classes[i].times) > 1 or len(instance.classes[i].room_options) > 1]
    top_single: dict[int, list[tuple[int, int]]] = {}
    for index in order:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        cls_deadline = time.perf_counter() + per_class_seconds
        if deadline is not None:
            cls_deadline = min(cls_deadline, deadline)
        opts = list(_placement_options(instance, index, hard, soft, cls_deadline, by_delta=True))
        if opts:
            rng.shuffle(opts)
            opts.sort(key=lambda item: item[0])
            top_single[index] = [opt for _, opt in opts[:limit]]
    return top_single


def _coupled_pair_set(instance: Instance, region: RegionCandidate) -> set[tuple[int, int]]:
    """Return sorted index pairs that appear together in a violated regional constraint."""
    pairs: set[tuple[int, int]] = set()
    id_to_index = instance.class_index
    for detail in region.constraint_details:
        for group in detail.get("violation_groups", ()):
            in_region = sorted(id_to_index[cid] for cid in group.get("in_region_class_ids", ())
                               if cid in id_to_index)
            for a, b in combinations(in_region, 2):
                pairs.add((a, b))
    return pairs


def _class_room_id(instance: Instance, index: int, room_index: int) -> str | None:
    cls = instance.classes[index]
    return None if cls.no_room or room_index < 0 else cls.room_options[room_index][0]


def _eviction_options(instance: Instance, index: int, target_room: str | None,
                      target_time, current_opt: tuple[int, int],
                      top_opts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Build placement options for `index` moving into `target_room` vacated by its partner."""
    cls = instance.classes[index]
    if target_room is None or cls.no_room:
        return []
    room_idx = next((i for i, (room_id, _) in enumerate(cls.room_options) if room_id == target_room), None)
    if room_idx is None:
        return []
    options: list[tuple[int, int]] = []
    target_key = (target_time.days, target_time.weeks, target_time.start, target_time.length)
    candidates = [(current_opt[0], room_idx)]
    matching_ti = next((ti for ti, opt_time in enumerate(cls.times)
                        if (opt_time.days, opt_time.weeks, opt_time.start, opt_time.length) == target_key), None)
    if matching_ti is not None:
        candidates.append((matching_ti, room_idx))
    candidates.extend((ti, ri) for ti, ri in top_opts if ri == room_idx)
    for cand in candidates:
        if cand != current_opt and cand not in options:
            options.append(cand)
    return options


def _pair_swap_candidates(instance: Instance, source: np.ndarray,
                          hard: PlacementNeighborhood, left: int, right: int,
                          top_single: dict[int, list[tuple[int, int]]],
                          coupled_pairs: set[tuple[int, int]]) -> list[tuple[int, int, int, int, int]]:
    """Enumerate direct exchanges, room-eviction handoffs, and coupled two-class moves."""
    original_left, original_right = hard.current(left), hard.current(right)
    l_cls, r_cls = instance.classes[left], instance.classes[right]
    l_old_time, r_old_time = hard.times[left], hard.times[right]
    l_old_room, r_old_room = hard.rooms[left], hard.rooms[right]
    pair_key = (min(left, right), max(left, right))
    has_pair_conflict = (pair_key in coupled_pairs or
                         hard.pair_cost(left, original_left, right, original_right) > 0)

    seen = set(swap_options(instance, source, left, right))
    candidates = [(*quad, 0) for quad in seen]
    left_top = top_single.get(left, ())
    right_top = top_single.get(right, ())

    for a_opt in _eviction_options(instance, left, r_old_room, r_old_time, original_left, left_top):
        for b_opt in right_top:
            quad = (*a_opt, *b_opt)
            if quad not in seen:
                seen.add(quad)
                candidates.append((*quad, 0))

    for b_opt in _eviction_options(instance, right, l_old_room, l_old_time, original_right, right_top):
        for a_opt in left_top:
            quad = (*a_opt, *b_opt)
            if quad not in seen:
                seen.add(quad)
                candidates.append((*quad, 0))

    for a_t, a_r in left_top:
        a_time = l_cls.times[a_t]
        a_room = _class_room_id(instance, left, a_r)
        for b_t, b_r in right_top:
            quad = (a_t, a_r, b_t, b_r)
            if quad in seen:
                continue
            b_time = r_cls.times[b_t]
            b_room = _class_room_id(instance, right, b_r)
            room_handoff = ((a_room is not None and a_room == r_old_room) or
                            (b_room is not None and b_room == l_old_room))
            time_handoff = overlaps(a_time, r_old_time) or overlaps(b_time, l_old_time)
            if room_handoff or time_handoff or has_pair_conflict:
                seen.add(quad)
                candidates.append((*quad, 0 if (room_handoff or has_pair_conflict) else 1))
    return candidates


def _rank_swap_options(instance, source, region, hard, soft, rng, deadline):
    started = time.perf_counter()
    single_deadline = (started + 0.55 * max(0.05, deadline - started)
                       if deadline is not None else None)
    top_single = _top_single_options(instance, region, hard, soft, rng, single_deadline)
    coupled_pairs = _coupled_pair_set(instance, region)
    ranked = []
    for left, right in combinations(region.class_indices, 2):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        candidates = _pair_swap_candidates(
            instance, source, hard, left, right, top_single, coupled_pairs)
        if not candidates:
            continue
        original_left, original_right = hard.current(left), hard.current(right)
        before_hard, before_soft = _swap_cost(instance, hard, soft, left, right)
        for a_time, a_room, b_time, b_room, tier in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            hard.move(left, a_time, a_room)
            hard.move(right, b_time, b_room)
            try:
                after_hard, after_soft = _swap_cost(instance, hard, soft, left, right)
            finally:
                hard.move(left, *original_left)
                hard.move(right, *original_right)
            ranked.append(((after_hard - before_hard, after_soft - before_soft, tier),
                           ((left, a_time, a_room), (right, b_time, b_room))))
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: item[0])
    return (changes for _, changes in ranked)


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
    candidate_origin = None
    best_trial: tuple[int, int] | None = None
    attempts = evaluations = 0
    evaluated: set[tuple[int, ...]] = set()

    def result() -> OperatorProbe:
        return OperatorProbe(started, attempts, evaluations, source_hard, source_total,
                             best_trial, best_pop, None if best_key is None else best_key[:2],
                             candidate_origin)

    def consider(trial: np.ndarray, origin: str) -> tuple[int, int] | None:
        nonlocal attempts, evaluations, best_pop, best_key, best_trial, candidate_origin
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
            candidate_origin = origin
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
                candidate_origin = "cp_sat"
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
        hard_neighborhood = PlacementNeighborhood(instance, working)
        local_soft = LocalSoftCosts(instance, set(region.class_indices), assignments)
        for position, index in enumerate(selected):
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            remaining_classes = len(selected) - position
            quota = max(1, (budget - evaluations) // remaining_classes)
            local_best = working_rank
            local_choice = working
            options = list(_placement_options(instance, index, hard_neighborhood, local_soft, deadline))
            rng.shuffle(options)
            options.sort(key=lambda item: item[0])
            for _, (ti, ri) in options[:quota]:
                trial = working.copy()
                trial[index], trial[n + index] = ti, ri
                rank = consider(trial, "sequential_rebuild")
                if rank is not None and rank < local_best:
                    local_best, local_choice = rank, trial
            working, working_rank = local_choice, local_best
            new_ti, new_ri = int(working[index]), int(working[n + index])
            hard_neighborhood.move(index, new_ti, new_ri)
    else:
        ranking_deadline = started + _GUIDED_RANKING_SECONDS
        if deadline is not None:
            ranking_deadline = min(ranking_deadline, deadline)
        engine = algorithm.problem.engine
        cached = engine.cache.get(engine.key(source))
        # Region scoring normally has this sectioning cached. Ranking itself never
        # spends a full evaluation; the complete scorer re-sections each trial.
        assignments = cached[1] if cached else None
        hard = PlacementNeighborhood(instance, source)
        soft = LocalSoftCosts(instance, set(region.class_indices), assignments)
        rank_options = _rank_move_options if operator_id == "move" else _rank_swap_options
        options = rank_options(instance, source, region, hard, soft, rng, ranking_deadline)
        n = len(instance.classes)
        for changes in options:
            if evaluations >= budget or (deadline is not None and time.perf_counter() >= deadline):
                break
            trial = source.copy()
            for i, ti, ri in changes:
                trial[i], trial[n + i] = ti, ri
            consider(trial, operator_id)

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
                               time.perf_counter() - started, candidate_origin=probe.candidate_origin)
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
                           time.perf_counter() - started, candidate_origin=probe.candidate_origin)


def apply_operator(algorithm, instance: Instance, region: RegionCandidate,
                   operator_id: str, budget: int, rng: np.random.Generator,
                   deadline: float | None) -> OperatorOutcome:
    """Search within the budget and inject at most one improving candidate."""
    probe = probe_operator(algorithm, instance, region, operator_id, budget, rng, deadline)
    return inject_operator_candidate(algorithm, probe)
