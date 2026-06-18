from __future__ import annotations

import json
from typing import Any

import eva_agent.nodes.frame_parser as frame_parser
from eva_agent.llm.base import LLMResponse
from eva_agent.nlu.preprocess import preprocess
from eva_agent.state import AgentState, Intent


class _FrameClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "json_mode": json_mode,
                "schema": schema,
            }
        )
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return LLMResponse(
            text=json.dumps(self.payloads[index], ensure_ascii=False),
            model="fake",
            backend="fake",
        )


def _base_frame(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": "read",
        "target": "Contract",
        "relation": None,
        "fields": [],
        "filters": {"date_hint": "none", "status": []},
        "cardinality": "one",
        "selector": {},
        "output": "summary",
        "subtasks": [],
        "needs_clarification": False,
        "clarify_reason": "",
        "confidence": 0.9,
    }
    payload.update(updates)
    return payload


def test_intent_frame_parser_uses_schema_and_deterministic_hints(monkeypatch: Any) -> None:
    client = _FrameClient(
        [
            _base_frame(
                target="ContractParty",
                relation="parties",
                selector={"role": "customer"},
                output="card",
            )
        ]
    )
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    state = AgentState(
        user_input_raw="кто заказчик по договору CT-1",
        intent=Intent(kind="mixed_diagnostic", confidence=0.9),
        nlu=preprocess("кто заказчик по договору CT-1"),
    )

    result = frame_parser.intent_frame_parser(state)

    frame = result["frame"]
    assert client.calls[0]["schema"]["$defs"]["PlanningFrame"]["properties"]["target"]["type"] == "string"
    assert client.calls[0]["temperature"] == 0.0
    assert frame.operation == "read"
    assert frame.target == "ContractParty"
    assert frame.relation == "parties"
    assert frame.cardinality == "one"
    assert frame.selector == {"role": "customer", "contract_id": "CT-1"}
    assert result["checklist"].entities == ["ContractParty", "Contract", "Counterparty"]


def test_intent_frame_parser_relation_dominates_llm_contract_target(monkeypatch: Any) -> None:
    client = _FrameClient(
        [
            _base_frame(
                target="Contract",
                relation="parties",
                output="card",
            )
        ]
    )
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(
        AgentState(
            user_input_raw="Покажи стороны по договору CT-1",
            intent=Intent(kind="mixed_diagnostic", confidence=0.9),
            nlu=preprocess("Покажи стороны по договору CT-1"),
        )
    )

    frame = result["frame"]
    assert frame.target == "ContractParty"
    assert frame.relation == "parties"
    assert frame.selector["contract_id"] == "CT-1"
    assert result["debug"]["frame"]["normalized"]["target"] == "ContractParty"


def test_intent_frame_parser_infers_party_relation_from_nlu(monkeypatch: Any) -> None:
    client = _FrameClient([_base_frame(target="Contract", relation=None)])
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(
        AgentState(
            user_input_raw="С кем заключен договор CT-1, выведи контрагентов",
            intent=Intent(kind="mixed_diagnostic", confidence=0.9),
            nlu=preprocess("С кем заключен договор CT-1, выведи контрагентов"),
        )
    )

    frame = result["frame"]
    assert frame.target == "ContractParty"
    assert frame.relation == "parties"
    assert frame.selector["contract_id"] == "CT-1"


def test_intent_frame_parser_infers_documents_relation_from_nlu(monkeypatch: Any) -> None:
    client = _FrameClient([_base_frame(target="Contract", relation=None)])
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(
        AgentState(
            user_input_raw="Покажи чего не хватает в договоре CT-2",
            intent=Intent(kind="mixed_diagnostic", confidence=0.9),
            nlu=preprocess("Покажи чего не хватает в договоре CT-2"),
        )
    )

    frame = result["frame"]
    assert frame.target == "Document"
    assert frame.relation == "documents"
    assert frame.selector["contract_id"] == "CT-2"


def test_intent_frame_parser_retries_once_and_returns_clarify_frame(monkeypatch: Any) -> None:
    class BrokenClient:
        calls = 0

        def invoke(
            self,
            system: str,
            user: str,
            *,
            temperature: float | None = None,
            json_mode: bool = False,
            schema: dict[str, Any] | None = None,
        ) -> LLMResponse:
            del system, user, temperature, json_mode, schema
            self.calls += 1
            return LLMResponse(text="{not-json", model="fake", backend="fake")

    client = BrokenClient()
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(AgentState(user_input_raw="помоги"))

    assert client.calls == 2
    assert result["frame"].needs_clarification is True
    assert result["checklist"].resolution == "clarify"


def test_intent_frame_parser_common_queries(monkeypatch: Any) -> None:
    cases = [
        (
            "все стороны договора CT-1",
            _base_frame(
                operation="list",
                target="ContractParty",
                relation="parties",
                cardinality="all",
                output="list",
            ),
            ("list", "ContractParty", "parties", "all", {"contract_id": "CT-1"}),
        ),
        (
            "почему креатив CR-2 не готов",
            _base_frame(
                operation="diagnose",
                target="Creative",
                fields=["status"],
                output="summary",
            ),
            ("diagnose", "Creative", None, "one", {"creative_id": "CR-2"}),
        ),
        (
            "каких документов не хватает по договору CT-2",
            _base_frame(
                operation="diagnose",
                target="Document",
                relation="documents",
                fields=["missing"],
                cardinality="all",
                output="list",
            ),
            ("diagnose", "Document", "documents", "all", {"contract_id": "CT-2"}),
        ),
        (
            "где размещения по договору CT-1",
            _base_frame(
                operation="list",
                target="Placement",
                relation="placements",
                cardinality="all",
                output="list",
            ),
            ("list", "Placement", "placements", "all", {"contract_id": "CT-1"}),
        ),
        (
            "скачай документ DOC-1 по договору CT-1",
            _base_frame(
                operation="download",
                target="Document",
                relation="documents",
                selector={"doc_id": "DOC-1"},
                output="value",
            ),
            ("download", "Document", "documents", "one", {"contract_id": "CT-1", "doc_id": "DOC-1"}),
        ),
        (
            "креативы по договору CT-2",
            _base_frame(
                operation="list",
                target="Creative",
                relation="placements",
                cardinality="all",
                output="list",
            ),
            ("list", "Placement", "placements", "all", {"contract_id": "CT-2"}),
        ),
        (
            "статус креатива CR-1 и документы по договору CT-1",
            _base_frame(
                operation="read",
                target="Creative",
                fields=["status"],
                selector={"creative_id": "CR-1"},
                subtasks=[
                    _base_frame(
                        operation="list",
                        target="Document",
                        relation="documents",
                        selector={"contract_id": "CT-1"},
                        output="list",
                    )
                ],
            ),
            ("read", "Creative", None, "one", {"creative_id": "CR-1", "contract_id": "CT-1"}),
        ),
    ]
    client = _FrameClient([payload for _, payload, _ in cases])
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    for query, _, expected in cases:
        result = frame_parser.intent_frame_parser(
            AgentState(
                user_input_raw=query,
                intent=Intent(kind="mixed_diagnostic", confidence=0.9),
                nlu=preprocess(query),
            )
        )
        operation, target, relation, cardinality, selector = expected
        frame = result["frame"]
        assert (frame.operation, frame.target, frame.relation, frame.cardinality) == (
            operation,
            target,
            relation,
            cardinality,
        )
        for key, value in selector.items():
            assert frame.selector[key] == value


def test_intent_frame_parser_repeated_validation_signature_stops_at_fallback(
    monkeypatch: Any,
) -> None:
    client = _FrameClient(
        [
            _base_frame(target="UnknownThing"),
            _base_frame(target="UnknownThing"),
            _base_frame(target="Contract"),
        ]
    )
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(AgentState(user_input_raw="помоги"))

    assert len(client.calls) == 2
    assert result["frame"].needs_clarification is True
    assert result["frame"].clarify_reason


def test_intent_frame_parser_uses_deterministic_draft_fallback(monkeypatch: Any) -> None:
    client = _FrameClient(
        [
            _base_frame(target="Contract", relation="bad_relation"),
            _base_frame(target="Contract", relation="bad_relation"),
        ]
    )
    monkeypatch.setattr(frame_parser, "get_client", lambda role: client)
    monkeypatch.setattr(frame_parser, "retrieve_examples", lambda query, k=5: [])

    result = frame_parser.intent_frame_parser(
        AgentState(
            user_input_raw="статус договора CT-1",
            intent=Intent(kind="mixed_diagnostic", confidence=0.9),
            nlu=preprocess("статус договора CT-1"),
        )
    )

    frame = result["frame"]
    assert len(client.calls) == 2
    assert frame.needs_clarification is False
    assert frame.target == "Contract"
    assert frame.relation is None
    assert "frame fallback: deterministic draft" in frame.trace


def test_decompose_query_is_conservative() -> None:
    domain_map = frame_parser.load_domain_map()
    composite = frame_parser.decompose_query(
        "статус креатива CR-1 и каких документов не хватает по договору CT-1",
        nlu=preprocess("статус креатива CR-1 и каких документов не хватает по договору CT-1"),
        domain_map=domain_map,
    )
    single = frame_parser.decompose_query(
        "статус креатива CR-1",
        nlu=preprocess("статус креатива CR-1"),
        domain_map=domain_map,
    )

    assert [(frame.operation, frame.target) for frame in composite] == [
        ("read", "Creative"),
        ("diagnose", "Document"),
    ]
    assert single == []
