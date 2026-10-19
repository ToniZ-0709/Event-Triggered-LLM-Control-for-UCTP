"""Bounded forward-checking construction, reused across initial individuals."""
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from itertools import combinations
import time

import numpy as np
from .scoring import distribution_violations, overlaps


class ConstructionIndex:
    def __init__(self, instance):
        self.instance = instance
        self.n = len(instance.classes)
        self.pairs = defaultdict(list)
        self.neighbors = [set() for _ in instance.classes]
        self.globals = [[] for _ in instance.classes]
        self.room_domains = [set(r for r, _ in c.room_options) for c in instance.classes]
        self.start_domains = [set(t.start for t in c.times) for c in instance.classes]
        for d in instance.distributions:
            if not d.required:
                continue
            if d.kind in {"MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock"}:
                for i in d.classes:
                    self.globals[i].append(d)
                continue
            if d.kind in {"SameRoom", "SameStart"}:
                domains = self.room_domains if d.kind == "SameRoom" else self.start_domains
                common = set.intersection(*(domains[i] for i in d.classes)) if d.classes else set()
                if common:
                    for i in d.classes:
                        domains[i] &= common
            for a, b in combinations(d.classes, 2):
                self.pairs[min(a, b), max(a, b)].append(replace(d, classes=(a, b)))
                self.neighbors[a].add(b)
                self.neighbors[b].add(a)
        self.times = [c.times[0] for c in instance.classes]
        self.rooms = [None if c.no_room else c.room_options[0][0] for c in instance.classes]

    @lru_cache(maxsize=100000)
    def pair_cost(self, a, ta, ra, b, tb, rb):
        ca, cb = self.instance.classes[a], self.instance.classes[b]
        at, bt = ca.times[ta], cb.times[tb]
        ar = None if ca.no_room else ca.room_options[ra][0]
        br = None if cb.no_room else cb.room_options[rb][0]
        old = self.times[a], self.times[b], self.rooms[a], self.rooms[b]
        self.times[a], self.times[b], self.rooms[a], self.rooms[b] = at, bt, ar, br
        cost = sum(distribution_violations(self.instance, d, self.times, self.rooms)
                   for d in self.pairs.get((min(a, b), max(a, b)), ()))
        self.times[a], self.times[b], self.rooms[a], self.rooms[b] = old
        return cost

    def build(self, rng, start=None, deadline=None):
        domains = []
        for i, c in enumerate(self.instance.classes):
            ts = [j for j, t in enumerate(c.times) if t.start in self.start_domains[i]]
            rs = [-1] if c.no_room else [j for j, (r, _) in enumerate(c.room_options)
                                         if r in self.room_domains[i]]
            # Preserve low-cost options and diversity without constructing a huge product.
            def sample(values, key, cap):
                ordered = sorted(values, key=key)
                if len(ordered) <= cap:
                    return ordered
                return list(dict.fromkeys(ordered[:cap//2] +
                    list(map(int, rng.choice(ordered[cap//2:], cap-cap//2, replace=False)))))
            ts = sample(ts, lambda j: c.times[j].penalty, 32)
            rs = sample(rs, lambda j: 0 if j == -1 else c.room_options[j][1], 16)
            options = [(t, r) for t in ts for r in rs]
            valid = [(t, r) for t, r in options if r == -1 or not any(
                overlaps(c.times[t], block) for block in self.instance.rooms[c.room_options[r][0]].unavailable)]
            options = valid or options
            rng.shuffle(options)
            # Round-robin over time choices so a time with many rooms does not consume the domain.
            grouped = [[o for o in options if o[0] == t] for t in ts]
            options = [row[k] for k in range(max(map(len, grouped), default=0))
                       for row in grouped if k < len(row)][:64]
            if start is not None:
                old = (int(start[i]), int(start[self.n+i]))
                if old in options:
                    options.remove(old)
                    options.insert(0, old)
            domains.append(options)
        placed = {}
        usage = defaultdict(list)
        choices = np.empty(2*self.n, dtype=np.int32)
        tie = rng.random(self.n)
        while len(placed) < self.n:
            if deadline is not None and time.perf_counter() >= deadline:
                for j in range(self.n):
                    if j not in placed:
                        choices[j], choices[self.n+j] = domains[j][int(rng.integers(len(domains[j])))]
                break
            i = min((j for j in range(self.n) if j not in placed),
                    key=lambda j: (len(domains[j]), -len(self.neighbors[j]), tie[j]))
            c = self.instance.classes[i]
            ranked = []
            for t, r in domains[i]:
                chosen = c.times[t]
                room = None if r == -1 else c.room_options[r][0]
                room_hard = (0 if room is None else sum(overlaps(chosen, other) for other in usage[room]) +
                             int(any(overlaps(chosen, b) for b in self.instance.rooms[room].unavailable)))
                hard = room_hard + sum(self.pair_cost(i, t, r, j, *placed[j])
                                      for j in self.neighbors[i] if j in placed)
                self.times[i], self.rooms[i] = chosen, room
                for d in self.globals[i]:
                    members = tuple(j for j in d.classes if j in placed or j == i)
                    hard += distribution_violations(self.instance, replace(d, classes=members),
                                                   self.times, self.rooms)
                blocked = 0
                if deadline is None or time.perf_counter() < deadline:
                    lookahead = sorted(self.neighbors[i]-placed.keys(), key=lambda j: len(domains[j]))[:4]
                    for j in lookahead:
                        blocked += not any(self.pair_cost(i, t, r, j, *other) == 0
                                           for other in domains[j])
                soft = chosen.penalty*self.instance.weights[0] + (0 if r == -1 else
                        c.room_options[r][1]*self.instance.weights[1])
                ranked.append(((hard, blocked, soft, float(rng.random())), (t, r)))
            _, (t, r) = min(ranked)
            choices[i], choices[self.n+i] = t, r
            placed[i] = t, r
            self.times[i] = c.times[t]
            self.rooms[i] = None if r == -1 else c.room_options[r][0]
            if r != -1:
                usage[self.rooms[i]].append(self.times[i])
            if deadline is None or time.perf_counter() < deadline:
                for j in self.neighbors[i]-placed.keys():
                    filtered = [o for o in domains[j] if self.pair_cost(i, t, r, j, *o) == 0]
                    if filtered:
                        domains[j] = filtered
        return choices
