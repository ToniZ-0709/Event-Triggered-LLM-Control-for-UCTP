"""Shared controller schemas, balanced prompts, payload sizing, and client bindings."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from .operators import OPERATORS

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from providers.gemini import GEMINI_MODELS, THINKING_LEVELS, GeminiDecisionClient  # noqa: E402
from providers.settings import (  # noqa: E402
    create_decision_client,
    load_experiment_settings,
    resolve_model_profile,
)

DEFAULT_MODEL = GEMINI_MODELS[0]
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_THINKING_LEVEL = THINKING_LEVELS[0]
DEFAULT_MAX_OUTPUT_TOKENS = 4096
VALID_THINKING_LEVELS = THINKING_LEVELS
FREE_TIER_MODELS = GEMINI_MODELS


def _object(properties: dict[str, dict]) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_NULLABLE_ID = {"type": ["string", "null"]}
SCHEMAS = {
    "single_agent": _object({
        "solution_id": _NULLABLE_ID, "region_id": _NULLABLE_ID,
        "boundary_id": _NULLABLE_ID,
        "operator_id": {"type": "string", "enum": ["no_op", *OPERATORS]},
        "execution_budget": {"type": "integer"},
        "next_review_interval": {"type": "integer"},
    }),
    "inspector": _object({
        "solution_id": _NULLABLE_ID, "region_id": _NULLABLE_ID,
        "boundary_id": _NULLABLE_ID,
        "relevant_constraint_ids": {"type": "array", "items": {"type": "string"}},
    }),
    "planner": _object({
        "operator_id": {"type": "string", "enum": ["no_op", *OPERATORS]},
        "execution_budget": {"type": "integer"},
        "next_review_interval": {"type": "integer"},
    }),
}

_STATE_GUIDANCE = (
    "Look up each candidate region's solution_id in solutions to inspect its schedule hard "
    "violations (hard), internal soft total (total), optional unlocalized_penalties, and class "
    "assignments (classes), and compare hard then total against global.incumbent_hard and "
    "global.incumbent_internal_total. A large regional penalty alone is weak evidence if its "
    "source is far behind the incumbent. Work on a different source only if its repair "
    "opportunity plausibly closes the gap or offers a distinct useful placement. "
    "Each candidate region is a search window. A five-class window can contain a one- or two-class bottleneck, "
    "while pairwise constraints (such as room_conflict or student_conflict) list one or two in_region_class_ids "
    "per group even when several region classes interact across groups. "
    "Use violation groups and shared classes to identify which placements need to change together. "
    "constraint_details lists active violations touching the region, ordered with hard and in-region "
    "violations first; if it is empty, regional soft cost comes from unary current_time_penalty or "
    "current_room_penalty in solutions[solution_id].classes. Use required to separate hard from "
    "soft constraints. "
    "region_violation_units covers violating groups touching the region; when violations extend "
    "outside the region, current_violations also reports the constraint's full violation total. "
    "Read params (when present) in the constraint type's order (e.g. MinGap's first parameter is "
    "the minimum gap in time slots). In violation_groups, in_region_class_ids lists classes "
    "inside the region that operators can modify together, while outside_region_class_ids lists "
    "fixed boundary classes held constant as reference context; when non-empty, inspect "
    "solutions[solution_id].classes (time_options, room_options) to judge whether moving one or "
    "more in-region classes can separate them from the boundary. cross_boundary indicates whether any "
    "part of the constraint lies outside the region; group lists can be incomplete. Check "
    "boundary_id, overlapping class_ids, source_hard, source_total, best_trial_hard, and "
    "best_trial_total in recent_outcomes when comparing similar interventions. A best_hard_change "
    "of 0 means incumbent hard violations stayed unchanged even if the candidate was selected. "
    "Revisit a stalled cluster when the state changed or a different intervention fits better. "
)
_ACCEPTANCE_GUIDANCE = (
    "For source hard > 0, reducing hard violations (trial_hard < source_hard) is the primary goal; "
    "lowering total at the same hard count is also a valid secondary gain that enters survival "
    "selection. For source hard = 0, a trial must stay feasible (hard = 0) and improve at least "
    "one weighted objective (time, room, distribution, student), including Pareto trade-offs where "
    "total increases. NSGA-III decides final survival while keeping unique chromosomes first. "
    "Use these rules to judge whether a target and operator are worth the evaluations. "
)
_BUDGET_GUIDANCE = (
    "Region scoring, operator trials and any recheck share the entire search evaluation budget "
    "(1,800 in the standard campaign; use global.evaluation_budget and remaining_evaluations "
    "for this run). Even no_op consumes a review and region scoring has already spent evaluations. "
    "Use reviews_used and max_reviews to preserve review opportunities and leave evaluations "
    "for NSGA-III generations. "
)
_ACTION_GUIDANCE = (
    "Use an operator in the target's available_operators and respect allowed_actions ranges. "
    "All operators act only inside the selected region, rank hard conflicts and soft costs "
    "(time, room, distribution, and estimated student conflicts from the source sectioning), "
    "and charge full scoring on trial evaluations. "
    "Match the operator to the structural size of the bottleneck rather than defaulting to one choice: "
    "move reassigns 1 class (time and/or room, typically 2 to 5 evaluations); choose it when a single class "
    "dominates the penalty or can vacate a hard, soft, or student conflict independently. "
    "swap coordinates 2 classes in the region, covering direct room/time exchanges, 2-class eviction "
    "hand-offs (one class takes the other's room or overlapping slot while the partner moves to its "
    "best free option), and coupled conflict-pair moves across hard, soft, or student conflicts "
    "(typically 4 to 8 evaluations); choose it when a 1-class move is blocked by another in-region class, "
    "a violation group pairs two in-region classes, or two classes compete for slots or rooms. "
    "destroy_repair uses bounded joint CP-SAT optimization (minimizing local hard violations when "
    "hard > 0, or weighted time, room, soft-distribution, and student-conflict costs when hard = 0) "
    "followed by sequential rebuild (typically 6 to 10 evaluations); choose it when hard, soft, or "
    "student conflicts span three or more linked classes across violation groups, or when 1- or 2-class "
    "steps are insufficient. "
    "Check recent_outcomes for any operator that returned no_improving_candidate, repeatedly "
    "left best_hard_change at 0 while the incumbent was infeasible, or made only small repeated "
    "1-class gains on a multi-class conflict region. For a similar source and "
    "region, choose another suitable operator, retarget if possible, or use no_op. "
    "For no_op, set execution_budget=0 and next_review_interval to 10-20 within the allowed range, "
    "or the nearest allowed value. Choose 3-5 only when the current state shows a concrete "
    "opportunity that needs an early follow-up. "
    "next_review_interval sets the scheduled review. Stagnation or low diversity can trigger "
    "an earlier review once the minimum review interval has elapsed. "
)

PROMPTS = {
    "single_agent": (
        "You control one bounded NSGA-III improvement step for ITC 2019. "
        "Select WHERE from candidate_regions, HOW from allowed_actions, and WHEN. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE + _ACTION_GUIDANCE
        + "Use only listed IDs. For no_op, return null solution_id, region_id and boundary_id. "
        "Never output code or a timetable. "
        "The solver validates and executes your decision."
    ),
    "inspector": (
        "You are the Inspector for ITC 2019 NSGA-III. Choose only WHERE: one "
        "candidate region's solution_id, region_id, and boundary_id. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE
        + "Check recent_outcomes first: if an overlapping cluster already returned "
        "no_improving_candidate on a similar source state, avoid repeating the same target when another "
        "candidate_region or solution_id offers a distinct repair opportunity, or return null target IDs "
        "and an empty list if no target is promising. "
        "Return only a short list of relevant_constraint_ids from the selected region's "
        "constraint_details that explains why this region is worth repairing (or an empty list "
        "if the region is selected for class time/room penalties). "
        "Do not choose an operator or schedule."
    ),
    "planner": (
        "You are the Planner for ITC 2019 NSGA-III. The Inspector's target is fixed. "
        "Choose only HOW and WHEN. Use the diagnosis IDs to look up the target's constraints. "
        + _STATE_GUIDANCE + _ACCEPTANCE_GUIDANCE + _BUDGET_GUIDANCE + _ACTION_GUIDANCE
        + "Never output code or a timetable."
    ),
}


def _encode_payload(payload: dict[str, Any], max_characters: int) -> tuple[str, int]:
    def encode(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    body = encode(payload)
    if len(body) <= max_characters:
        return body, 0

    state = payload.get("state", payload)
    regions = state.get("candidate_regions") if isinstance(state, dict) else None
    if not regions:
        return body, 0
    diagnosis = payload.get("diagnosis", {})
    diagnosed_ids = set(diagnosis.get("relevant_constraint_ids", ()))
    target_keys = ("solution_id", "region_id", "boundary_id")
    ranked_details = []
    for region in regions:
        is_target = diagnosed_ids and all(region.get(key) == diagnosis.get(key) for key in target_keys)
        required_ids = diagnosed_ids if is_target else set()
        def _detail_units(item: dict[str, Any]) -> int:
            if "region_violation_units" in item:
                return int(item["region_violation_units"])
            return int(item.get("current_violations", 0))

        details = sorted(region.get("constraint_details", ()), key=lambda item: (
            item["constraint_id"] not in required_ids,
            not bool(_detail_units(item)),
            not item["required"],
            item["cross_boundary"],
            -_detail_units(item),
        ))
        keep = sum(item["constraint_id"] in required_ids for item in details)
        ranked_details.append((region, details, keep))

    def limited(limit: int) -> tuple[str, int]:
        public_regions = []
        omitted = 0
        for region, details, keep in ranked_details:
            count = max(limit, keep)
            dropped = max(0, len(details) - count)
            if dropped:
                region = {**region, "constraint_details": details[:count],
                          "constraint_details_omitted": dropped}
            public_regions.append(region)
            omitted += dropped
        public_state = {**state, "candidate_regions": public_regions}
        public_payload = public_state if state is payload else {**payload, "state": public_state}
        return encode(public_payload), omitted

    # Fit as many details per region as possible without changing IDs or the response schema.
    best = limited(0)
    low, high = 1, max(len(details) for _, details, _ in ranked_details)
    while low <= high:
        limit = (low + high) // 2
        reduced = limited(limit)
        if len(reduced[0]) <= max_characters:
            best = reduced
            low = limit + 1
        else:
            high = limit - 1
    return best


def validate_shape(role: str, value: Any) -> bool:
    """Validate JSON types/keys; the solver separately validates stateful IDs."""
    schema = SCHEMAS[role]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return False
    for key, spec in schema["properties"].items():
        item = value[key]
        kind = spec["type"]
        if isinstance(kind, list):
            if item is not None and not isinstance(item, str):
                return False
        elif kind == "integer" and type(item) is not int:
            return False
        elif kind == "string" and not isinstance(item, str):
            return False
        elif kind == "array" and (not isinstance(item, list) or
                                  any(not isinstance(x, str) for x in item)):
            return False
        if "enum" in spec and item not in spec["enum"]:
            return False
    return True


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "FREE_TIER_MODELS",
    "GEMINI_MODELS",
    "PROMPTS",
    "SCHEMAS",
    "VALID_THINKING_LEVELS",
    "GeminiDecisionClient",
    "create_decision_client",
    "load_experiment_settings",
    "resolve_model_profile",
    "validate_shape",
]
