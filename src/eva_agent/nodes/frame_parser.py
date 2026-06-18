"""Semantic frame parser node."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, cast

from eva_agent.domain.checklist import EntityCount, PlanningChecklist
from eva_agent.domain.confidence import composite_confidence
from eva_agent.domain.frame import FrameFilters, PlanningFrame
from eva_agent.domain.frame_validate import FrameValidation, validate_frame
from eva_agent.llm.config import get_client
from eva_agent.nlu.fewshot import Example, retrieve_examples
from eva_agent.nlu.preprocess import NluFeatures, preprocess
from eva_agent.planner.compile import rank_protocol_cards
from eva_agent.state import AgentState
from eva_agent.tools.build_domain_map import load_domain_map

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|,?\s+(?:а также|затем|потом|и еще|и)\s+)\s*",
    re.IGNORECASE,
)
_CARD_OUTPUT_TARGETS = frozenset({"Contract", "Counterparty", "Creative"})
MAX_FRAME_REFINES = 2
_MAX_FRAME_CALLS = MAX_FRAME_REFINES + 1
_MAX_PARSE_ERRORS = 2

_FRAME_SYS = (
    "Ты заполняешь типизированный семантический фрейм для планировщика. "
    "Верни только JSON по схеме. Используй имена сущностей, связей, ролей и статусов из онтологии. "
    "Не придумывай идентификаторы: бери их из NLU_HINTS. Если не хватает входа, поставь "
    "needs_clarification=true и кратко заполни clarify_reason."
)


def intent_frame_parser(state: AgentState) -> dict[str, Any]:
    """Parse user query into a typed PlanningFrame and seed checklist fields."""

    query = state.user_input_clean or state.user_input_raw
    nlu = state.nlu or preprocess(query)
    domain_map = load_domain_map()
    draft = _draft_frame(query, nlu, domain_map)
    examples = _safe_examples(query)
    user = _user_prompt(query, nlu, draft, examples, domain_map)
    schema = PlanningFrame.model_json_schema()

    client = get_client("domain")
    last_error = ""
    parse_errors = 0
    seen_signatures: set[str] = set()
    last_validation: FrameValidation | None = None
    frame: PlanningFrame | None = None
    prompt = user

    for call_index in range(_MAX_FRAME_CALLS):
        try:
            response = client.invoke(
                _FRAME_SYS,
                prompt,
                temperature=0.0,
                schema=schema,
            )
            frame = _parse_frame(response.text)
            frame = _merge_deterministic_hints(frame, draft, domain_map)
            frame = _apply_decomposition(query, nlu, frame, domain_map)
            validation = validate_frame(frame, domain_map=domain_map, draft=draft)
            last_validation = validation
            if validation.ok:
                frame = _with_validation_trace(frame, validation)
                frame = _with_composite_confidence(frame, draft, nlu, domain_map)
                return {"frame": frame, "checklist": _checklist_from_frame(frame, state, domain_map)}

            signature = validation.signature
            if signature in seen_signatures:
                frame = _with_trace(frame, f"frame validation repeated: {signature}")
                break
            seen_signatures.add(signature)
            if call_index >= MAX_FRAME_REFINES:
                break
            prompt = _repair_prompt(user, validation)
        except Exception as exc:
            last_error = exc.__class__.__name__
            parse_errors += 1
            if parse_errors >= _MAX_PARSE_ERRORS or call_index >= _MAX_FRAME_CALLS - 1:
                break
            prompt = _parse_repair_prompt(user, last_error)

    frame = _fallback_frame(
        draft,
        domain_map=domain_map,
        validation=last_validation,
        last_error=last_error,
    )
    frame = _apply_decomposition(query, nlu, frame, domain_map)
    frame = _with_composite_confidence(frame, draft, nlu, domain_map)
    return {"frame": frame, "checklist": _checklist_from_frame(frame, state, domain_map)}


def _user_prompt(
    query: str,
    nlu: NluFeatures,
    draft: PlanningFrame,
    examples: list[Example],
    domain_map: dict[str, Any],
) -> str:
    payload = {
        "query": query,
        "nlu_hints": _nlu_payload(nlu),
        "deterministic_draft": draft.model_dump(mode="json"),
        "ontology": _ontology_index(domain_map),
        "few_shot": _fewshot_pairs(examples, domain_map),
    }
    return json.dumps(payload, ensure_ascii=False)


def _repair_prompt(base_user: str, validation: FrameValidation) -> str:
    payload = {
        "request": _safe_json(base_user),
        "validation_issues": [
            {"level": issue.level, "code": issue.code, "hint": issue.hint}
            for issue in validation.issues
        ],
        "instruction": "Верните JSON по той же схеме. Исправьте только поля из validation_issues.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_repair_prompt(base_user: str, error: str) -> str:
    payload = {
        "request": _safe_json(base_user),
        "parse_error": error,
        "instruction": "Верните один JSON-объект по схеме без текста вокруг.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _with_validation_trace(frame: PlanningFrame, validation: FrameValidation) -> PlanningFrame:
    if not validation.issues:
        return frame
    codes = ",".join(issue.code for issue in validation.issues)
    return _with_trace(frame, f"frame validation ok with issues: {codes}")


def _with_trace(frame: PlanningFrame, event: str) -> PlanningFrame:
    trace = list(frame.trace)
    if event not in trace:
        trace.append(event)
    return frame.model_copy(update={"trace": trace})


def _fallback_frame(
    draft: PlanningFrame,
    *,
    domain_map: dict[str, Any],
    validation: FrameValidation | None,
    last_error: str,
) -> PlanningFrame:
    trace: list[str] = []
    if validation is not None:
        trace.append(f"frame validation failed: {validation.signature}")
    if last_error:
        trace.append(f"frame parse failed: {last_error}")
    if _frame_can_compile(draft, domain_map):
        trace.append("frame fallback: deterministic draft")
        return draft.model_copy(
            update={
                "needs_clarification": False,
                "clarify_reason": "",
                "trace": _unique([*draft.trace, *trace]),
            }
        )
    reason = _clarify_hint(validation, last_error)
    return draft.model_copy(
        update={
            "needs_clarification": True,
            "clarify_reason": reason,
            "confidence": 0.0,
            "trace": _unique([*draft.trace, *trace]),
        }
    )


def _clarify_hint(validation: FrameValidation | None, last_error: str) -> str:
    if validation is not None:
        for issue in validation.issues:
            if issue.hint:
                return issue.hint
        if validation.issues:
            return validation.issues[0].message
    if last_error:
        return "уточните, с какой сущностью работать"
    return "уточните цель запроса"


def _frame_can_compile(frame: PlanningFrame, domain_map: dict[str, Any]) -> bool:
    if frame.target not in _entity_names(domain_map):
        return False
    ranked = rank_protocol_cards(frame, domain_slice=None)
    return bool(ranked and ranked[0].score > 0)


def _with_composite_confidence(
    frame: PlanningFrame,
    draft: PlanningFrame,
    nlu: NluFeatures,
    domain_map: dict[str, Any],
) -> PlanningFrame:
    ranked = rank_protocol_cards(frame, domain_slice=None)
    breakdown = composite_confidence(
        frame,
        draft=draft,
        nlu=nlu,
        ranked=ranked,
        domain_map=domain_map,
    )
    trace = list(frame.trace)
    trace.append(f"frame confidence: {breakdown.score:.2f}")
    return frame.model_copy(
        update={
            "confidence": breakdown.score,
            "confidence_factors": breakdown.factors,
            "trace": _unique(trace),
        }
    )


def _apply_decomposition(
    query: str,
    nlu: NluFeatures,
    frame: PlanningFrame,
    domain_map: dict[str, Any],
) -> PlanningFrame:
    subtasks = decompose_query(query, nlu=nlu, domain_map=domain_map)
    if not subtasks:
        return frame
    return frame.model_copy(
        update={
            "subtasks": subtasks,
            "trace": _unique([*frame.trace, f"frame decomposed: {len(subtasks)}"]),
        }
    )


def decompose_query(
    query: str,
    *,
    nlu: NluFeatures | None = None,
    domain_map: dict[str, Any] | None = None,
) -> list[PlanningFrame]:
    del nlu
    domain_map = domain_map or load_domain_map()
    clauses = _split_clauses(query)
    if len(clauses) < 2 or len(clauses) > 3:
        return []
    frames: list[PlanningFrame] = []
    for clause in clauses:
        clause_nlu = preprocess(clause)
        clause_frame = _draft_frame(clause, clause_nlu, domain_map)
        if not clause_frame.target or not _frame_can_compile(clause_frame, domain_map):
            return []
        frames.append(clause_frame)
    signatures = {(item.target, item.operation) for item in frames}
    return frames if len(signatures) > 1 else []


def _split_clauses(query: str) -> list[str]:
    normalized = query.lower().replace("\u0451", "е")
    parts: list[str] = []
    start = 0
    for match in _CLAUSE_SPLIT_RE.finditer(normalized):
        part = query[start : match.start()].strip(" ,;")
        if part:
            parts.append(part)
        start = match.end()
    tail = query[start:].strip(" ,;")
    if tail:
        parts.append(tail)
    return parts


def _nlu_payload(nlu: NluFeatures) -> dict[str, Any]:
    return {
        "lemmas": nlu.lemmas,
        "entity_ids": nlu.entity_ids,
        "entities": nlu.entities,
        "roles": nlu.roles,
        "statuses": nlu.statuses,
        "dates": [date.model_dump(mode="json") for date in nlu.dates],
        "date_hint": nlu.date_hint,
        "action_verbs": nlu.action_verbs,
    }


def _ontology_index(domain_map: dict[str, Any]) -> list[dict[str, Any]]:
    entities = cast(dict[str, dict[str, Any]], domain_map.get("entities", {}))
    return [
        {
            "name": name,
            "description": entity.get("description", ""),
            "fields": entity.get("fields", []),
            "operations": entity.get("operations", []),
            "roles": entity.get("roles", []),
            "statuses": entity.get("statuses", []),
        }
        for name, entity in entities.items()
    ]


def _fewshot_pairs(examples: list[Example], domain_map: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for example in examples:
        features = preprocess(example.query)
        frame = _draft_frame(example.query, features, domain_map)
        pairs.append({"query": example.query, "frame": frame.model_dump(mode="json")})
    return pairs


def _safe_examples(query: str) -> list[Example]:
    try:
        return retrieve_examples(query, k=5)
    except Exception:
        return []


def _parse_frame(text: str) -> PlanningFrame:
    raw = _safe_json(text)
    if not raw:
        raise ValueError("empty frame")
    return PlanningFrame.model_validate(raw)


def _safe_json(text: str) -> dict[str, Any]:
    clean = _strip_fences(text)
    try:
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        parsed = _load_json_object(clean)
    return parsed if isinstance(parsed, dict) else {}


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _load_json_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}


def _draft_frame(query: str, nlu: NluFeatures, domain_map: dict[str, Any]) -> PlanningFrame:
    selector = _selector_from_nlu(nlu)
    target = _target_from_nlu(nlu, selector)
    operation = _operation_from_nlu(query, nlu, target, selector)
    cardinality = _cardinality_from_nlu(query, nlu, selector)
    if cardinality == "all" and operation == "read":
        operation = "list"
    relation = _relation_from_nlu(target, selector)
    fields = _fields_from_nlu(query, nlu)
    return PlanningFrame(
        operation=operation,
        target=target,
        relation=relation,
        fields=fields,
        filters={"date_hint": nlu.date_hint, "status": nlu.statuses},
        cardinality=cardinality,
        selector=selector,
        output=_output(operation, target, cardinality, fields),
        needs_clarification=target not in _entity_names(domain_map),
        clarify_reason="" if target in _entity_names(domain_map) else "не определена целевая сущность",
        confidence=0.65 if target else 0.2,
    )


def _selector_from_nlu(nlu: NluFeatures) -> dict[str, str]:
    ids = nlu.entity_ids
    selector: dict[str, str] = {}
    _put_first(selector, "contract_id", ids.get("contracts"))
    _put_first(selector, "creative_id", ids.get("creatives"))
    _put_first(selector, "counterparty_id", ids.get("counterparties"))
    _put_first(selector, "doc_id", ids.get("documents"))
    _put_first(selector, "placement_id", ids.get("placements"))
    _put_first(selector, "contract_id", ids.get("contract_numbers"))
    _put_first(selector, "counterparty_hint", ids.get("counterparty_hints"))
    _put_first(selector, "role", nlu.roles)
    return selector


def _put_first(out: dict[str, str], key: str, values: list[str] | None) -> None:
    if key in out or not values:
        return
    value = str(values[0]).strip()
    if value:
        out[key] = value


def _target_from_nlu(nlu: NluFeatures, selector: dict[str, str]) -> str:
    if nlu.roles:
        return "ContractParty"
    if nlu.entities:
        if "Creative" in nlu.entities and "contract_id" in selector and "creative_id" not in selector:
            return "Creative"
        return nlu.entities[0]
    if "creative_id" in selector:
        return "Creative"
    if "counterparty_id" in selector or "counterparty_hint" in selector:
        return "Counterparty"
    if "doc_id" in selector:
        return "Document"
    if "placement_id" in selector:
        return "Placement"
    if "contract_id" in selector:
        return "Contract"
    return ""


def _operation_from_nlu(
    query: str,
    nlu: NluFeatures,
    target: str,
    selector: dict[str, str],
) -> str:
    actions = set(nlu.action_verbs)
    if "compare" in actions:
        return "compare"
    if "download" in actions:
        return "download"
    if "attach" in actions:
        return "attach"
    if "open" in actions:
        return "open"
    lowered = query.lower()
    if target == "Document" and _has_any(lowered, ("не хватает", "каких документов")):
        return "diagnose"
    if _has_any(lowered, ("почему", "мешает", "готов", "провер")):
        return "diagnose"
    if nlu.statuses and "contract_id" not in selector:
        return "list"
    if "list" in actions or "search" in actions:
        return "list"
    if _has_any(lowered, ("список", "какие ", "каких ", "по каким", "сколько", "все ")):
        return "list"
    return "read"


def _cardinality_from_nlu(query: str, nlu: NluFeatures, selector: dict[str, str]) -> str:
    lowered = query.lower()
    if _has_any(
        lowered,
        (
            "все",
            "какие",
            "каких",
            "какими",
            "список",
            "перечис",
            "сколько",
            "неоформлен",
            "незаверш",
            "неподпис",
            "остались",
            "кажд",
        ),
    ):
        return "all"
    if "list" in nlu.action_verbs and not any(
        key in selector for key in ("creative_id", "counterparty_id", "doc_id", "placement_id")
    ):
        return "all"
    return "one"


def _relation_from_nlu(target: str, selector: dict[str, str]) -> str | None:
    if target == "ContractParty" or (target == "Counterparty" and "contract_id" in selector):
        return "parties"
    if target == "Document":
        return "documents"
    if target == "Placement":
        return "placements"
    if target == "Creative" and "contract_id" in selector and "creative_id" not in selector:
        return "placements"
    if target == "Contract" and "creative_id" in selector and "contract_id" not in selector:
        return "creative"
    return None


def _fields_from_nlu(query: str, nlu: NluFeatures) -> list[str]:
    lowered = query.lower()
    fields: list[str] = []
    if nlu.statuses or "статус" in lowered:
        fields.append("status")
    if "реквизит" in lowered:
        fields.extend(["name", "inn", "legal_type"])
    if "инн" in lowered:
        fields.append("inn")
    if "не хватает" in lowered:
        fields.append("missing")
    if "форма" in lowered:
        fields.append("distribution_form")
    return _unique(fields)


def _output(operation: str, target: str, cardinality: str, fields: list[str]) -> str:
    if operation == "list" or cardinality == "all":
        return "list"
    if len(fields) == 1:
        return "value"
    if target in _CARD_OUTPUT_TARGETS:
        return "card"
    return "summary"


def _merge_deterministic_hints(
    frame: PlanningFrame,
    draft: PlanningFrame,
    domain_map: dict[str, Any],
) -> PlanningFrame:
    selector = dict(frame.selector)
    selector.update(draft.selector)
    statuses = _unique([*frame.filters.status, *draft.filters.status])
    date_hint = draft.filters.date_hint if draft.filters.date_hint != "none" else frame.filters.date_hint
    target = frame.target if frame.target in _entity_names(domain_map) else draft.target
    operation = frame.operation
    if frame.operation == "read" and draft.operation != "read":
        operation = draft.operation
    cardinality = "all" if draft.cardinality == "all" else frame.cardinality
    relation = frame.relation or draft.relation
    output = frame.output if frame.output != "summary" else draft.output
    needs_clarification = frame.needs_clarification or target not in _entity_names(domain_map)
    clarify_reason = frame.clarify_reason
    if needs_clarification and not clarify_reason:
        clarify_reason = "не определена целевая сущность"
    return frame.model_copy(
        update={
            "operation": operation,
            "target": target,
            "relation": relation,
            "fields": _unique([*frame.fields, *draft.fields]),
            "filters": FrameFilters(date_hint=date_hint, status=statuses),
            "cardinality": cardinality,
            "selector": selector,
            "output": output,
            "needs_clarification": needs_clarification,
            "clarify_reason": clarify_reason,
        }
    )


def _checklist_from_frame(
    frame: PlanningFrame,
    state: AgentState,
    domain_map: dict[str, Any],
) -> PlanningChecklist:
    entities = _entities_from_frame(frame, domain_map)
    access = _tools_for_entities(entities, domain_map)
    intent = state.intent.kind if state.intent else ""
    return PlanningChecklist(
        intent=intent,
        entities=entities,
        cardinality=[
            EntityCount(
                entity=entity,
                intent_count=1 if frame.cardinality == "one" else None,
            )
            for entity in entities
        ],
        access=access,
        needs_chain=bool(frame.relation or frame.subtasks),
        resolution="clarify" if frame.needs_clarification else "proceed",
        clarify_reason=frame.clarify_reason,
    )


def _entities_from_frame(frame: PlanningFrame, domain_map: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    known = _entity_names(domain_map)
    _append_known(entities, frame.target, known)
    if frame.relation == "parties":
        _append_known(entities, "Contract", known)
        _append_known(entities, "ContractParty", known)
        _append_known(entities, "Counterparty", known)
    elif frame.relation == "placements":
        _append_known(entities, "Contract", known)
        _append_known(entities, "Placement", known)
        if frame.target == "Creative":
            _append_known(entities, "Creative", known)
    elif frame.relation == "documents":
        _append_known(entities, "Contract", known)
        _append_known(entities, "Document", known)
    elif frame.relation == "creative":
        _append_known(entities, "Creative", known)
        _append_known(entities, "Contract", known)
    for subtask in frame.subtasks:
        for entity in _entities_from_frame(subtask, domain_map):
            _append_known(entities, entity, known)
    return entities


def _tools_for_entities(entities: list[str], domain_map: dict[str, Any]) -> list[str]:
    all_entities = cast(dict[str, dict[str, Any]], domain_map.get("entities", {}))
    tools: list[str] = []
    for entity in entities:
        for tool in cast(list[str], all_entities.get(entity, {}).get("tools", [])):
            if tool not in tools:
                tools.append(tool)
    return tools


def _entity_names(domain_map: dict[str, Any]) -> set[str]:
    entities = domain_map.get("entities", {})
    return set(entities) if isinstance(entities, dict) else set()


def _append_known(out: list[str], entity: str, known: set[str]) -> None:
    if entity in known and entity not in out:
        out.append(entity)


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


__all__ = ["MAX_FRAME_REFINES", "decompose_query", "intent_frame_parser"]
