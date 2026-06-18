"""Composite confidence for semantic frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from eva_agent.domain.frame import PlanningFrame
from eva_agent.nlu.preprocess import NluFeatures
from eva_agent.planner.compile import ProtocolRank

WEIGHT_TARGET_KNOWN = 0.25
WEIGHT_SLOTS_COVERED = 0.25
WEIGHT_DRAFT_AGREEMENT = 0.20
WEIGHT_RANK_MARGIN = 0.15
WEIGHT_NLU_SIGNAL = 0.15


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    factors: dict[str, float]


def composite_confidence(
    frame: PlanningFrame,
    *,
    draft: PlanningFrame,
    nlu: NluFeatures,
    ranked: list[ProtocolRank],
    domain_map: dict[str, Any],
) -> ConfidenceBreakdown:
    raw = {
        "target_known": _target_known(frame, domain_map),
        "slots_covered": _slots_covered(ranked),
        "draft_agreement": _draft_agreement(frame, draft),
        "rank_margin": _rank_margin(ranked),
        "nlu_signal": _nlu_signal(nlu),
    }
    factors = {
        "target_known": raw["target_known"] * WEIGHT_TARGET_KNOWN,
        "slots_covered": raw["slots_covered"] * WEIGHT_SLOTS_COVERED,
        "draft_agreement": raw["draft_agreement"] * WEIGHT_DRAFT_AGREEMENT,
        "rank_margin": raw["rank_margin"] * WEIGHT_RANK_MARGIN,
        "nlu_signal": raw["nlu_signal"] * WEIGHT_NLU_SIGNAL,
        "llm_reported": _clamp(float(frame.confidence)),
    }
    score = _clamp(sum(value for key, value in factors.items() if key != "llm_reported"))
    return ConfidenceBreakdown(score=score, factors=factors)


def _target_known(frame: PlanningFrame, domain_map: dict[str, Any]) -> float:
    raw = domain_map.get("entities", {})
    entities = set(cast(dict[str, Any], raw)) if isinstance(raw, dict) else set()
    return 1.0 if frame.target in entities else 0.0


def _slots_covered(ranked: list[ProtocolRank]) -> float:
    if not ranked:
        return 0.0
    top = ranked[0]
    total = len(top.covered_slots) + len(top.missing_slots)
    if total == 0:
        return 1.0
    return len(top.covered_slots) / total


def _draft_agreement(frame: PlanningFrame, draft: PlanningFrame) -> float:
    fields = ("operation", "target", "relation", "cardinality")
    matches = sum(int(getattr(frame, field) == getattr(draft, field)) for field in fields)
    return matches / len(fields)


def _rank_margin(ranked: list[ProtocolRank]) -> float:
    if not ranked:
        return 0.0
    top = ranked[0].score
    if len(ranked) == 1:
        return 1.0 if top > 0 else 0.0
    gap = max(0, top - ranked[1].score)
    return _clamp(gap / 40.0)


def _nlu_signal(nlu: NluFeatures) -> float:
    groups = (
        any(bool(values) for values in nlu.entity_ids.values()),
        bool(nlu.roles),
        bool(nlu.statuses),
        bool(nlu.action_verbs),
    )
    return sum(int(group) for group in groups) / len(groups)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["ConfidenceBreakdown", "composite_confidence"]
