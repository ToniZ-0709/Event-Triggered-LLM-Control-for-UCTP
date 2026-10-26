"""Deterministic, read-only placement checks shared by packets and guided actions.

These checks never decode students, call a full scorer, run repair or draw RNG.
Their costs describe affected constraints, not a prediction of a full score.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import numpy as np

from .scoring import distribution_violations, overlaps

ALTERNATIVES_PER_CLASS = 24
OUTSIDE_LIMIT = 32
RULE_LIMIT = 12
WITNESS_LIMIT = 2
LIMITATIONS = ["student_resectioning_not_checked", "full_score_not_evaluated"]
ROOM_RULES = {"SameRoom", "DifferentRoom"}
TIME_ROOM_RULES = {"SameAttendees"}


def source_hash(choices) -> str:
    return hashlib.sha256(np.asarray(choices, dtype=np.int32).tobytes()).hexdigest()


def placement(instance, choices, index):
    cls = instance.classes[index]
    t = cls.times[int(choices[index])]
    room = None if cls.no_room else cls.room_options[int(choices[len(instance.classes) + index])][0]
    return {"time": {"days": t.days, "weeks": t.weeks, "start": t.start,
                     "length": t.length}, "room_id": room}


def relevant_movable(instance, rule, active):
    """Participants with freedom in an attribute actually used by this rule."""
    result = []
    for i in sorted(set(rule.classes).intersection(active)):
        cls = instance.classes[i]
        fields = ({"SameStart": ("start",), "SameTime": ("start","length"),
                   "DifferentTime": ("start","length"), "SameDays": ("days",),
                   "DifferentDays": ("days",), "MaxDays": ("days",),
                   "SameWeeks": ("weeks",), "DifferentWeeks": ("weeks",),
                   "MaxDayLoad": ("days","weeks","length")}.get(rule.kind,
                    ("days","weeks","start","length")))
        time_keys = {tuple(getattr(t,f) for f in fields) for t in cls.times}
        room_keys = {r for r, _ in cls.room_options}
        time_free, room_free = len(time_keys) > 1, len(room_keys) > 1
        if ((rule.kind in ROOM_RULES and room_free) or
                (rule.kind in TIME_ROOM_RULES and (time_free or room_free)) or
                (rule.kind not in ROOM_RULES | TIME_ROOM_RULES and time_free)):
            result.append(i)
    return tuple(result)


class PlacementDiagnostics:
    def __init__(self, instance, choices):
        self.instance = instance
        self.source = np.asarray(choices, dtype=np.int32).copy()
        self.n = len(instance.classes)
        self.times = [c.times[int(self.source[i])] for i, c in enumerate(instance.classes)]
        self.rooms = [None if c.no_room else c.room_options[int(self.source[self.n+i])][0]
                      for i, c in enumerate(instance.classes)]
        self.work = 0

    def distribution_target(self, active):
        ranked = []
        for number, rule in enumerate(self.instance.distributions):
            mutable = relevant_movable(self.instance, rule, active)
            if not rule.required or not mutable:
                continue
            amount = distribution_violations(self.instance, rule, self.times, self.rooms)
            if amount:
                ranked.append((-amount, len(mutable), number, mutable))
        if not ranked:
            return None
        _, _, number, mutable = min(ranked)
        return number, mutable

    def _observations(self, choices, changed):
        instance = self.instance
        times = self.times.copy()
        rooms = self.rooms.copy()
        result = {}
        for i in changed:
            cls = instance.classes[i]
            times[i] = cls.times[int(choices[i])]
            rooms[i] = None if cls.no_room else cls.room_options[int(choices[self.n+i])][0]
        for i in changed:
            if rooms[i] is not None:
                result[f"room_unavailable:{instance.classes[i].id}"] = (
                    "room_unavailable", int(any(overlaps(times[i], t) for t in
                                                 instance.rooms[rooms[i]].unavailable)), 1, (i,))
            for j in range(self.n):
                if i == j or (j in changed and j < i):
                    continue
                a, b = sorted((i, j))
                self.work += 1
                if rooms[a] is not None and rooms[a] == rooms[b] and overlaps(times[a], times[b]):
                    result[f"room_conflict:{instance.classes[a].id}:{instance.classes[b].id}"] = (
                        "room_conflicts", 1, 1, (a, b))
        for number, rule in enumerate(instance.distributions):
            if not changed.intersection(rule.classes):
                continue
            self.work += 1
            amount = distribution_violations(instance, rule, times, rooms)
            result[f"distribution:{number}"] = (
                "hard_distributions" if rule.required else "soft_distribution",
                amount, 1 if rule.required else rule.penalty * instance.weights[2], rule.classes)
        return result

    def check(self, candidate, changed):
        changed = set(changed)
        before = self._observations(self.source, changed)
        after = self._observations(candidate, changed)
        hard_delta = Counter({"room_conflicts": 0, "room_unavailable": 0,
                              "hard_distributions": 0})
        resolved, created, blocked, blockers = [], [], [], set()
        for key in sorted(set(before) | set(after)):
            old = before.get(key)
            new = after.get(key)
            kind, _, weight, members = new or old
            a, b = (0 if old is None else old[1]), (0 if new is None else new[1])
            if kind != "soft_distribution":
                hard_delta[kind] += b - a
                if b:
                    blocked.append(key)
                    blockers.update(members)
            if a > b:
                resolved.append(key)
            if b > a:
                created.append(key)
        delta = {"time": 0, "room": 0}
        for i in changed:
            cls = self.instance.classes[i]
            delta["time"] += (cls.times[int(candidate[i])].penalty -
                              cls.times[int(self.source[i])].penalty) * self.instance.weights[0]
            if not cls.no_room:
                delta["room"] += (cls.room_options[int(candidate[self.n+i])][1] -
                                  cls.room_options[int(self.source[self.n+i])][1]) * self.instance.weights[1]
        return {"level": "local_constraints", "resolved_constraint_ids": resolved,
                "created_constraint_ids": created, "hard_components_delta": dict(hard_delta),
                "weighted_time_room_delta": delta, "full_score": None,
                "limitations": LIMITATIONS.copy()}, blocked, blockers

    def moves(self, active, source_hard):
        """Return ranked one-gene proposals plus bounded screening evidence."""
        proposals, witnesses, total, checked = [], [], 0, 0
        counts, reasons, blockers = Counter(pass_=0, fail=0, unknown=0), Counter(), set()
        for i in active:
            cls = self.instance.classes[i]
            pool = [(i, j) for j in range(len(cls.times)) if j != self.source[i]]
            pool += [(self.n+i, j) for j in range(len(cls.room_options))
                     if not cls.no_room and j != self.source[self.n+i]]
            total += len(pool)
            # Evenly spaced deterministic coverage, exhaustive for small pools.
            indices = (list(range(len(pool))) if len(pool) <= ALTERNATIVES_PER_CLASS else
                       np.linspace(0, len(pool)-1, ALTERNATIVES_PER_CLASS, dtype=int).tolist())
            for position in indices:
                gene, value = pool[position]
                trial = self.source.copy()
                trial[gene] = value
                evidence, blocked, outside = self.check(trial, {i})
                checked += 1
                counts["fail" if blocked else "pass_"] += 1
                reasons.update(blocked)
                blockers.update(outside)
                hard = sum(evidence["hard_components_delta"].values())
                cost = sum(evidence["weighted_time_room_delta"].values())
                rank = (hard, cost, gene, value) if source_hard else (cost, hard, gene, value)
                proposals.append((rank, gene, value))
                witnesses.append((rank, {"class_ids": [cls.id],
                    "placements": [placement(self.instance, trial, i)], "evidence": evidence}))
        proposals.sort()
        witnesses.sort(key=lambda item: item[0])
        diagnostic = {"operator_id": "move", "class_ids": [self.instance.classes[i].id for i in active],
            "scope": "one_time_or_room_gene_with_all_other_placements_fixed",
            "selection_policy": "exhaustive_if_small_else_evenly_spaced_per_class",
            "checks": ["room_conflicts", "room_unavailable", "full_distribution_groups"],
            "coverage": {"candidates_total": total, "checked": checked, "complete": total == checked},
            "counts": {"pass": counts["pass_"], "fail": counts["fail"], "unknown": counts["unknown"]},
            "blocking_reasons": [{"constraint_id": k, "count": v} for k, v in sorted(reasons.items())],
            "witnesses": [w for _, w in witnesses[:WITNESS_LIMIT]]}
        return tuple((gene, value) for _, gene, value in proposals), diagnostic, blockers

    def constraint_detail(self, key):
        instance = self.instance
        if key.startswith("distribution:"):
            number = int(key.split(":")[1])
            rule = instance.distributions[number]
            amount = distribution_violations(instance, rule, self.times, self.rooms)
            return {"constraint_id": key, "type": rule.kind, "required": rule.required,
                "parameters": list(rule.params), "class_ids": [instance.classes[i].id for i in rule.classes],
                "satisfaction": {"status": "violated" if amount else "satisfied", "violation_count": amount},
                "cost": {"raw_units": amount, "penalty": None if rule.required else rule.penalty,
                         "weight": None if rule.required else instance.weights[2],
                         "weighted_cost": None if rule.required else amount * rule.penalty * instance.weights[2]}}
        if key.startswith("room_conflict:"):
            members = [instance.class_index[v] for v in key.split(":")[1:]]
            a, b = members
            amount = int(self.rooms[a] is not None and self.rooms[a] == self.rooms[b]
                         and overlaps(self.times[a], self.times[b]))
            kind = "RoomConflict"
        elif key.startswith("room_unavailable:"):
            members = [instance.class_index[key.split(":")[1]]]
            i = members[0]
            amount = int(self.rooms[i] is not None and any(overlaps(self.times[i], t)
                         for t in instance.rooms[self.rooms[i]].unavailable))
            kind = "RoomUnavailable"
        else:
            return {"constraint_id": key, "type": key, "required": False,
                    "parameters": [], "class_ids": [], "satisfaction": None, "cost": None}
        return {"constraint_id": key, "type": kind, "required": True, "parameters": [],
                "class_ids": [instance.classes[i].id for i in members],
                "satisfaction": {"status": "violated" if amount else "satisfied", "violation_count": amount},
                "cost": {"raw_units": amount, "penalty": None, "weight": None, "weighted_cost": None}}

    def extract(self, region, source_hard):
        from .regions import swap_targets
        active = set(region.class_indices)
        shortlist, move, blockers = self.moves(region.class_indices, source_hard)
        swap = {"operator_id": "swap", "scope": "two_compatible_placements",
                "class_ids": list(region.class_ids), "selection_policy": "exhaustive_region_pairs",
                "checks": ["domain_compatibility", "room_conflicts", "full_distribution_groups"],
                "coverage": {"candidates_total": len(active)*(len(active)-1)//2, "checked": 0, "complete": True},
                "counts": {"pass": 0, "fail": 0, "unknown": 0}, "domain_compatible_pairs": 0,
                "domain_incompatible_pairs": 0,
                "blocking_reasons": [], "witnesses": []}
        for a, b in combinations(sorted(active), 2):
            swap["coverage"]["checked"] += 1
            target = swap_targets(self.instance, self.source, a, b)
            if target is None:
                swap["counts"]["fail"] += 1
                swap["domain_incompatible_pairs"] += 1
                continue
            swap["domain_compatible_pairs"] += 1
            trial = self.source.copy()
            trial[a], trial[self.n+a], trial[b], trial[self.n+b] = target
            evidence, blocked, outside = self.check(trial, {a, b})
            blockers.update(outside)
            swap["counts"]["fail" if blocked else "pass"] += 1
            if len(swap["witnesses"]) < WITNESS_LIMIT:
                swap["witnesses"].append({"class_ids": [self.instance.classes[i].id for i in (a,b)],
                    "placements": [placement(self.instance, trial, i) for i in (a,b)], "evidence": evidence})
        diagnostics = [move, swap, {"operator_id": "destroy_repair", "class_ids": list(region.class_ids),
            "scope": "joint_active_region_then_bounded_sequential_fallback",
            "selection_policy": "execution_only", "checks": [], "blocking_reasons": [],
            "coverage": {"candidates_total": None, "checked": 0, "complete": False},
            "counts": {"pass": 0, "fail": 0, "unknown": 0}, "witnesses": []}]
        if shortlist:
            diagnostics.append({**move, "operator_id": "guided_move",
                "selection_policy": "rank_checked_hard_delta_then_known_time_room_cost" if source_hard else
                                    "rank_known_time_room_cost_then_checked_hard_delta"})
        target = self.distribution_target(active)
        if target is not None:
            number, mutable = target
            diagnostics.append({"operator_id": "distribution_repair", "selected_rule_id": f"distribution:{number}",
                "class_ids": [self.instance.classes[i].id for i in mutable],
                "scope": "movable_active_participants_only_all_other_classes_fixed",
                "selection_policy": "largest_violation_then_smallest_mutable_group_then_stable_id",
                "checks": [], "blocking_reasons": [],
                "coverage": {"candidates_total": None, "checked": 0, "complete": False},
                "counts": {"pass": 0, "fail": 0, "unknown": 0}, "witnesses": []})
        remaining_witnesses = WITNESS_LIMIT
        for diagnostic in diagnostics:
            witnesses = diagnostic.get("witnesses", [])
            diagnostic["witnesses_total"] = len(witnesses)
            diagnostic["witnesses"] = witnesses[:remaining_witnesses]
            diagnostic["witnesses_returned"] = len(diagnostic["witnesses"])
            diagnostic["witnesses_truncated"] = len(witnesses) > remaining_witnesses
            remaining_witnesses -= diagnostic["witnesses_returned"]
            diagnostic["minimum_full_evaluations_for_trial"] = 2
        current = self._observations(self.source, active)
        for kind,amount,_,members in current.values():
            if amount and kind != "soft_distribution":
                blockers.update(members)
        keys = sorted(set(region.constraint_ids) | {k for k,v in current.items() if v[1]} |
                      {r["constraint_id"] for r in move["blocking_reasons"]})
        details = [self.constraint_detail(k) for k in keys]
        details.sort(key=lambda d: (not (d.get("satisfaction") or {}).get("violation_count", 0), d["constraint_id"]))
        boundary = {self.instance.class_index[c] for c in region.boundary_class_ids}
        boundary.update(blockers-active)
        outside = sorted(boundary-active, key=lambda i: (i not in blockers, i))
        context = {"region_id": region.region_id,
            "outside_classes": {"count": len(outside), "returned_count": min(OUTSIDE_LIMIT, len(outside)),
                "truncated": len(outside) > OUTSIDE_LIMIT,
                "items": [{"class_id": self.instance.classes[i].id,
                    "placement": placement(self.instance, self.source, i), "mutable": False,
                    "relation": "checked_blocker" if i in blockers else "related_fixed_class"}
                    for i in outside[:OUTSIDE_LIMIT]]},
            "constraints": {"ids": keys, "detail_count": len(details),
                "returned_count": min(RULE_LIMIT, len(details)), "truncated": len(details) > RULE_LIMIT,
                "details": details[:RULE_LIMIT]}}
        return {"source_hash": source_hash(self.source), "move_proposals": shortlist,
                "distribution_target": target, "diagnostics": diagnostics,
                "boundary_context": context, "local_checks": self.work}
