from __future__ import annotations

from eva_agent.domain.frame import PlanningFrame
from eva_agent.domain.frame_validate import validate_frame
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
        "confidence": 0.9,
    }
    payload.update(updates)
    return PlanningFrame.model_validate(payload)


def test_validate_frame_reports_syntax_semantics_and_compile_levels() -> None:
    domain_map = load_domain_map()

    syntax_frame = _frame().model_copy(update={"operation": "search"})
    syntax = validate_frame(syntax_frame, domain_map=domain_map)
    assert any(issue.level == "syntax" and issue.code == "bad_operation" for issue in syntax.issues)
    assert syntax.ok is False

    semantics = validate_frame(_frame(relation="parties"), domain_map=domain_map)
    assert any(issue.level == "semantics" and issue.code == "bad_relation" for issue in semantics.issues)
    assert semantics.ok is False

    compile_check = validate_frame(_frame(target="CreativeMedia"), domain_map=domain_map)
    assert any(issue.level == "compile" and issue.code == "no_protocol" for issue in compile_check.issues)
    assert compile_check.ok is False


def test_validate_frame_keeps_unknown_field_soft() -> None:
    result = validate_frame(_frame(fields=["status"]), domain_map=load_domain_map())

    assert result.ok is True
    assert any(issue.code == "bad_field" for issue in result.issues)
    assert result.signature == validate_frame(_frame(fields=["other"]), domain_map=load_domain_map()).signature
