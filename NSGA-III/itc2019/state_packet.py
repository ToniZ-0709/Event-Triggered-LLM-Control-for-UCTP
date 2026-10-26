"""StatePacket v2 serialization and input-size reduction, without solver work."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json

SCHEMA_VERSION = 2


def packet_hash(packet):
    return hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()


def region_packet(region):
    return {"target": {k: getattr(region, k) for k in ("solution_id", "region_id", "boundary_id")},
        "source_quality": None if region.source_quality is None else {
            "hard": region.source_quality[0], "internal_total": region.source_quality[1],
            "weighted_objectives": region.source_weighted},
        "active_classes": [{"class_id": c["class_id"],
            "placement": {"time": c["current_time"], "room_id": c["current_room_id"]},
            "domains": {"time_options": c["time_options"], "room_options": c["room_options"],
                        "no_room": c["current_room_id"] is None},
            "penalties": {"current_time": c["current_time_penalty"], "current_room": c["current_room_penalty"]}}
            for c in region.class_summaries],
        "penalty_summary": {"hard_contribution": region.hard_contribution,
            "soft_contribution": region.soft_contribution, "attribution": "heuristic_nonadditive",
            "unlocalized_penalties": list(region.unlocalized_penalties)},
        "available_actions": {"operators": list(region.available_operators),
            "movement": {"movable_class_count": region.movable_class_count,
                         "compatible_swap_pairs": region.compatible_swap_pairs},
            "diagnostics": None if region.screening is None else deepcopy(region.screening["diagnostics"])}}


def make_packet(metrics, regions, history, outcomes, settings, operator_limit, operators):
    def quality(hard, total, weighted=None):
        return {"hard": hard, "internal_total": total, "weighted_objectives": weighted}
    incumbent = quality(metrics.get("incumbent_hard"), metrics.get("incumbent_internal_total"),
                        metrics.get("incumbent_weighted_objectives"))
    recent = deepcopy(outcomes[-settings.recent_outcomes:])
    current_hashes = {r.screening["source_hash"] for r in regions if r.screening}
    for review in recent:
        fingerprint = review.pop("source_hash", None)
        review["comparable_to_current"] = False if fingerprint is not None and fingerprint not in current_hashes else None
        review["comparability_reason"] = ("source_identity_unavailable" if fingerprint is None else
            "source_placements_differ" if fingerprint not in current_hashes else
            "current_action_scope_and_budget_not_yet_selected")
    feasible = metrics.get("best_feasible_internal_total")
    return {"global_search_state": {
        "progress": {k: metrics.get(k) for k in ("generation", "elapsed_seconds", "remaining_seconds")},
        "evaluations": {"used": metrics.get("evaluations"), "limit": metrics.get("evaluation_budget"),
                        "remaining": metrics.get("remaining_evaluations")},
        "population": {"size": metrics.get("population_size"), "feasible_count": metrics.get("feasible_count"),
                       "feasible_nondominated_count": metrics.get("feasible_nondominated_count")},
        "quality": {"population_best": quality(metrics.get("best_hard"), metrics.get("best_internal_total"),
                                               metrics.get("population_best_weighted_objectives")),
            "incumbent": incumbent, "best_feasible": None if feasible is None else {
                "hard": 0, "internal_total": feasible, "official_score": metrics.get("best_feasible_official_score")},
            "objective_distribution": metrics.get("objective_distribution"), "metric_source": metrics.get("metric_source")},
        "diversity": {"decision": metrics.get("decision_diversity"), "placement": metrics.get("placement_diversity"),
                      "reference_direction_coverage": metrics.get("reference_direction_coverage")}},
        "recent_quality": {"progress": {"mode": metrics.get("progress_mode"),
            "stagnant_generations": metrics.get("stagnant_generations"), "trigger_reasons": metrics.get("trigger_reasons", [])},
            "history": deepcopy(history[-settings.stagnation_window:])},
        "candidate_regions": [region_packet(r) for r in regions],
        "boundary_context": [deepcopy(r.screening["boundary_context"]) for r in regions if r.screening],
        "recent_outcomes": {"reviews": recent},
        "action_limits": {"operators": [*operators, "no_op"],
            "execution_budget": {"min": 1, "max": max(0, operator_limit)},
            "review_interval": {"min": settings.min_review_interval, "max": settings.max_review_interval}}}


def fit_packet(packet, max_characters):
    """Trim optional examples first, then lower-ranked regions, or skip the review.

    All validation IDs and the hard-blocker context of retained regions survive.
    Counts on a trimmed list explicitly describe returned versus total items.
    """
    packet = deepcopy(packet)
    size = lambda: len(json.dumps(packet, ensure_ascii=False, allow_nan=False))
    trims = []
    if size() > max_characters:
        for region in packet["candidate_regions"]:
            for diagnostic in region["available_actions"]["diagnostics"] or []:
                diagnostic.setdefault("witnesses_total", len(diagnostic["witnesses"]))
                diagnostic["witnesses"] = []
                diagnostic["witnesses_returned"] = 0
                diagnostic["witnesses_truncated"] = diagnostic["witnesses_total"] > 0
        trims.append("optional_witnesses")
    if size() > max_characters:
        for boundary in packet["boundary_context"]:
            rules = boundary["constraints"]
            rules["details"] = [d for d in rules["details"] if d.get("required") and
                                (d.get("satisfaction") is None or d["satisfaction"].get("violation_count", 0))]
            rules["returned_count"] = len(rules["details"])
            rules["truncated"] = rules["returned_count"] < rules["detail_count"]
            outside = boundary["outside_classes"]
            outside["items"] = [c for c in outside["items"] if c["relation"] == "checked_blocker"]
            outside["returned_count"] = len(outside["items"])
            outside["truncated"] = outside["returned_count"] < outside["count"]
        trims.append("optional_nonblocker_context")
    while size() > max_characters and len(packet["candidate_regions"]) > 1:
        dropped = packet["candidate_regions"].pop()["target"]["region_id"]
        packet["boundary_context"] = [b for b in packet["boundary_context"] if b["region_id"] != dropped]
        trims.append("region:" + dropped)
    if size() > max_characters:
        return None, {"status": "essential_context_exceeds_input_limit", "characters": size(), "trimmed": trims}
    return packet, {"status": "ready", "characters": size(), "trimmed": trims}
