"""Budgets and failure cooldowns for the repair shared by all pipelines."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepairSettings:
    interval: int = 5
    near_feasible_hard: int = 12
    neighborhood_classes: int = 32
    neighborhood_options: int = 32
    neighborhood_seconds: float = 8.0
    neighborhood_deterministic_limit: float = 0.5
    fallback_seconds: float = 2.0
    fallback_evaluations: int = 128
    failure_cooldown: int = 5
    max_failure_cooldown: int = 20

    def __post_init__(self) -> None:
        if min(self.interval, self.near_feasible_hard, self.neighborhood_classes,
               self.neighborhood_options, self.fallback_evaluations,
               self.failure_cooldown, self.max_failure_cooldown) < 1:
            raise ValueError("Repair budgets must be positive")
        if min(self.neighborhood_seconds, self.neighborhood_deterministic_limit,
               self.fallback_seconds) <= 0:
            raise ValueError("Repair time budgets must be positive")
        if self.max_failure_cooldown < self.failure_cooldown:
            raise ValueError("Maximum repair cooldown cannot be smaller than the initial cooldown")


class FailureBackoff:
    def __init__(self, initial: int = 5, maximum: int = 20) -> None:
        self.initial = initial
        self.maximum = maximum
        self.attempts: dict[str, tuple[int, int]] = {}

    def ready(self, signature: str, generation: int) -> bool:
        return generation >= self.attempts.get(signature, (0, 0))[1]

    def record(self, signature: str, generation: int, improved: bool) -> int:
        failures = 0 if improved else self.attempts.get(signature, (0, 0))[0] + 1
        cooldown = self.initial if improved else min(
            self.maximum, self.initial * 2 ** min(failures - 1, 10))
        self.attempts[signature] = failures, generation + cooldown
        return cooldown
