from __future__ import annotations

from eva_agent.domain.confidence import composite_confidence
from eva_agent.domain.frame import PlanningFrame
from eva_agent.nlu.preprocess import preprocess
from eva_agent.planner.compile import rank_protocol_cards
from eva_agent.tools.build_domain_map import load_domain_map


def _frame(**updates) -> PlanningFrame:
    payload = {
        "operation": "read",
        "target": "Contract",
        "relation": None,
        "fields": [],
        "filters": {"date_hint": "none", "status": []},
        "cardinality": "one",
        "selector": {"contract_id": "CT-1"},
        "output": "summary",
        "subtasks": [],
        "needs_clarification": False,
        "clarify_reason": "",
        "confidence": 0.1,
    }
    payload.update(updates)
    return PlanningFrame.model_validate(payload)


def _score(frame: PlanningFrame, draft: PlanningFrame | None = None) -> float:
    domain_map = load_domain_map()
    nlu = preprocess("статус договора CT-1")
    return composite_confidence(
        frame,
        draft=draft or frame,
        nlu=nlu,
        ranked=rank_protocol_cards(frame),
        domain_map=domain_map,
    ).score


def test_composite_confidence_ignores_llm_self_report() -> None:
    low_report = _frame(confidence=0.1)
    high_report = _frame(confidence=0.95)

    assert _score(low_report) == _score(high_report)


def test_composite_confidence_rises_with_target_and_slots() -> None:
    no_slot = _frame(selector={})
    with_slot = _frame(selector={"contract_id": "CT-1"})
    unknown_target = _frame(target="")

    assert _score(with_slot) > _score(no_slot)
    assert _score(with_slot) > _score(unknown_target, draft=_frame(target="Contract"))
