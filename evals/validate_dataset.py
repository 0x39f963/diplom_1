"""Static validation for benchmark JSONL files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eva_agent.planner.catalog import CATALOG
from eva_agent.planner.protocols import PROTOCOLS
from eva_agent.tools.entity_ref import EntityRefs, extract_refs
from eva_agent.tools.selector import EXECUTION_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARKS = (ROOT / "bench" / "benchmark_big.jsonl", ROOT / "bench" / "benchmark.jsonl")
DEFAULT_GOLD = ROOT / "bench" / "audit_gold.jsonl"
FIXTURES = ROOT / "src" / "eva_agent" / "mock" / "fixtures" / "scenarios.json"

INTENTS = {
    "legal_consult",
    "interface_consult",
    "mixed_diagnostic",
    "need_clarification",
    "out_of_scope",
    "blocked",
}
ASSERT_TYPES = {
    "intent",
    "blocked",
    "not_blocked",
    "has_citation",
    "used_tool",
    "expected_tools",
    "expected_entities",
    "mutating_tool",
    "must_clarify",
}
TOOL_ASSERTS = {"used_tool", "expected_tools", "mutating_tool"}
ENTITY_TYPES = {"contract", "party", "counterparty", "creative", "placement", "document"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        errors.append(f"{path}: file does not exist")
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"{path}:{line_no}: row is not an object")
            continue
        rows.append(raw)
    return rows


def _tool_names() -> set[str]:
    return set(EXECUTION_REGISTRY)


def _collect_known_ids() -> set[str]:
    raw = _load_json(FIXTURES)
    if not isinstance(raw, dict):
        return set()
    known: set[str] = set()
    _collect_contracts(raw, known)
    _collect_mapping_keys(raw, known, "creatives", skip={"default"})
    _collect_mapping_keys(raw, known, "counterparties")
    _collect_nested_items(raw, known, "placements", "id")
    _collect_nested_items(raw, known, "documents", "id")
    return known


def _collect_mapping_keys(
    raw: dict[str, Any],
    known: set[str],
    section_name: str,
    *,
    skip: set[str] | None = None,
) -> None:
    section = raw.get(section_name)
    if not isinstance(section, dict):
        return
    skip_values = skip or set()
    for key in section:
        if isinstance(key, str) and key not in skip_values:
            known.add(key)


def _collect_contracts(raw: dict[str, Any], known: set[str]) -> None:
    contracts = raw.get("contracts")
    if not isinstance(contracts, dict):
        return
    for contract_id, contract in contracts.items():
        if isinstance(contract_id, str):
            known.add(contract_id)
        if not isinstance(contract, dict):
            continue
        for key in ("contract_number", "number"):
            value = contract.get(key)
            if isinstance(value, str):
                known.add(value)


def _collect_nested_items(
    raw: dict[str, Any],
    known: set[str],
    section_name: str,
    id_key: str,
) -> None:
    section = raw.get(section_name)
    if not isinstance(section, dict):
        return
    for items in section.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            value = item.get(id_key)
            if isinstance(value, str):
                known.add(value)


def _refs_values(refs: EntityRefs) -> list[str]:
    return [
        *refs.contract_ids,
        *refs.creative_ids,
        *refs.counterparty_ids,
        *refs.document_ids,
        *refs.placement_ids,
        *refs.contract_numbers,
    ]


def _case_label(path: Path, case: dict[str, Any], index: int) -> str:
    case_id = case.get("id")
    if isinstance(case_id, str):
        return f"{path}:{case_id}"
    return f"{path}:row-{index}"


def validate_benchmark(path: Path, tools: set[str], known_ids: set[str]) -> list[str]:
    errors: list[str] = []
    rows = _load_jsonl(path, errors)
    seen: set[str] = set()
    for index, case in enumerate(rows, start=1):
        label = _case_label(path, case, index)
        _validate_required_case_fields(case, label, errors)
        case_id = case.get("id")
        if isinstance(case_id, str):
            if case_id in seen:
                errors.append(f"{label}: duplicate id")
            seen.add(case_id)
        expected_intent = case.get("expected_intent")
        if expected_intent not in INTENTS:
            errors.append(f"{label}: unknown expected_intent: {expected_intent!r}")
        asserts = case.get("asserts")
        if isinstance(asserts, list):
            _validate_asserts(asserts, expected_intent, label, tools, errors)
            _validate_mixed_has_tool_or_policy(case, asserts, label, errors)
        _validate_case_metadata(case, label, errors)
        _validate_known_refs(str(case.get("input", "")), known_ids, label, errors)
    return errors


def _validate_required_case_fields(case: dict[str, Any], label: str, errors: list[str]) -> None:
    for key in ("id", "input", "expected_intent", "asserts"):
        if key not in case:
            errors.append(f"{label}: missing field {key}")
    if not isinstance(case.get("id"), str):
        errors.append(f"{label}: id must be a string")
    if not isinstance(case.get("input"), str):
        errors.append(f"{label}: input must be a string")
    if not isinstance(case.get("asserts"), list):
        errors.append(f"{label}: asserts must be a list")


def _validate_asserts(
    asserts: list[Any],
    expected_intent: object,
    label: str,
    tools: set[str],
    errors: list[str],
) -> None:
    for index, assertion in enumerate(asserts, start=1):
        if not isinstance(assertion, dict):
            errors.append(f"{label}: assert #{index} is not an object")
            continue
        kind = assertion.get("type")
        if kind not in ASSERT_TYPES:
            errors.append(f"{label}: unknown assert type: {kind!r}")
            continue
        if kind == "intent" and assertion.get("value") != expected_intent:
            errors.append(f"{label}: intent assert does not match expected_intent")
        if kind in TOOL_ASSERTS:
            _validate_tool_assert(assertion, kind, label, tools, errors)
        if kind == "expected_entities":
            _validate_expected_entities(assertion, label, errors)
        if kind == "must_clarify" and not isinstance(assertion.get("value", True), bool):
            errors.append(f"{label}: must_clarify assert value must be bool")


def _validate_tool_assert(
    assertion: dict[str, Any],
    kind: object,
    label: str,
    tools: set[str],
    errors: list[str],
) -> None:
    value = assertion.get("value")
    values = value if isinstance(value, list) else [value]
    for tool in values:
        if not isinstance(tool, str):
            errors.append(f"{label}: {kind} value must contain strings")
            continue
        if tool not in tools:
            errors.append(f"{label}: unknown tool: {tool}")


def _validate_expected_entities(
    assertion: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    value = assertion.get("value")
    if not isinstance(value, list):
        errors.append(f"{label}: expected_entities value must be a list")
        return
    for entity_type in value:
        if entity_type not in ENTITY_TYPES:
            errors.append(f"{label}: unknown entity type: {entity_type!r}")


def _validate_mixed_has_tool_or_policy(
    case: dict[str, Any],
    asserts: list[Any],
    label: str,
    errors: list[str],
) -> None:
    if case.get("expected_intent") != "mixed_diagnostic":
        return
    has_tool_assert = any(
        isinstance(assertion, dict) and assertion.get("type") in TOOL_ASSERTS
        for assertion in asserts
    )
    has_must_clarify = bool(case.get("must_clarify")) or any(
        isinstance(assertion, dict) and assertion.get("type") == "must_clarify"
        for assertion in asserts
    )
    notes = str(case.get("notes", "")).lower()
    if not has_tool_assert and not has_must_clarify and "чисто-юрид" not in notes:
        errors.append(f"{label}: mixed_diagnostic has no tool assert or must_clarify policy")


def _validate_case_metadata(case: dict[str, Any], label: str, errors: list[str]) -> None:
    if "must_clarify" in case and not isinstance(case["must_clarify"], bool):
        errors.append(f"{label}: must_clarify must be bool")
    known_fail = case.get("known_fail")
    if known_fail is None:
        return
    if isinstance(known_fail, bool):
        return
    if not isinstance(known_fail, dict):
        errors.append(f"{label}: known_fail must be bool or object")
        return
    if not isinstance(known_fail.get("value", True), bool):
        errors.append(f"{label}: known_fail.value must be bool")
    if known_fail.get("value", True) and not isinstance(known_fail.get("reason"), str):
        errors.append(f"{label}: known_fail.reason must be string")


def _validate_known_refs(text: str, known_ids: set[str], label: str, errors: list[str]) -> None:
    refs = extract_refs(text)
    for entity_id in _refs_values(refs):
        if entity_id not in known_ids:
            errors.append(f"{label}: unknown entity id: {entity_id}")


def validate_gold(path: Path, known_ids: set[str], benchmark_ids: set[str]) -> list[str]:
    errors: list[str] = []
    rows = _load_jsonl(path, errors)
    seen: set[str] = set()
    protocols_seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"{path}:row-{index}"
        _validate_gold_row(row, label, known_ids, benchmark_ids, protocols_seen, errors)
        case_id = row.get("case_id")
        if isinstance(case_id, str):
            if case_id in seen:
                errors.append(f"{path}:{case_id}: duplicate gold case_id")
            seen.add(case_id)
    if rows and not 30 <= len(rows) <= 50:
        errors.append(f"{path}: gold size must be 30-50, got {len(rows)}")
    missing_protocols = set(PROTOCOLS) - protocols_seen
    if missing_protocols:
        errors.append(f"{path}: missing protocols in gold: {sorted(missing_protocols)}")
    return errors


def _validate_gold_row(
    row: dict[str, Any],
    label: str,
    known_ids: set[str],
    benchmark_ids: set[str],
    protocols_seen: set[str],
    errors: list[str],
) -> None:
    for key in (
        "case_id",
        "input",
        "expected_protocol",
        "expected_todos_ordered",
        "acceptable_alternatives",
        "must_clarify",
    ):
        if key not in row:
            errors.append(f"{label}: missing field {key}")
    case_id = row.get("case_id")
    if not isinstance(case_id, str):
        errors.append(f"{label}: case_id must be a string")
    elif case_id not in benchmark_ids:
        errors.append(f"{label}: case_id is not present in benchmark_big: {case_id}")
    if not isinstance(row.get("input"), str):
        errors.append(f"{label}: input must be a string")
    protocol = row.get("expected_protocol")
    if protocol not in PROTOCOLS:
        errors.append(f"{label}: unknown expected_protocol: {protocol!r}")
    elif isinstance(protocol, str):
        protocols_seen.add(protocol)
    _validate_todo_list(row.get("expected_todos_ordered"), f"{label}:expected_todos_ordered", errors)
    _validate_alternatives(row.get("acceptable_alternatives"), label, errors)
    if not isinstance(row.get("must_clarify"), bool):
        errors.append(f"{label}: must_clarify must be bool")
    _validate_known_refs(str(row.get("input", "")), known_ids, label, errors)


def _validate_todo_list(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{label}: must be a list")
        return
    for todo_id in value:
        if not isinstance(todo_id, str):
            errors.append(f"{label}: todo ids must be strings")
        elif todo_id not in CATALOG:
            errors.append(f"{label}: unknown todo id: {todo_id}")


def _validate_alternatives(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{label}: acceptable_alternatives must be a list")
        return
    for index, alternative in enumerate(value, start=1):
        _validate_todo_list(alternative, f"{label}:acceptable_alternatives[{index}]", errors)


def _benchmark_ids(path: Path) -> set[str]:
    errors: list[str] = []
    rows = _load_jsonl(path, errors)
    return {row["id"] for row in rows if isinstance(row.get("id"), str)}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    benchmark_paths = tuple(Path(arg) for arg in args) if args else DEFAULT_BENCHMARKS
    tools = _tool_names()
    known_ids = _collect_known_ids()
    errors: list[str] = []
    for path in benchmark_paths:
        errors.extend(validate_benchmark(path, tools, known_ids))
    benchmark_ids = _benchmark_ids(ROOT / "bench" / "benchmark_big.jsonl")
    errors.extend(validate_gold(DEFAULT_GOLD, known_ids, benchmark_ids))

    if errors:
        print(f"Dataset validation failed: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Dataset validation ok: "
        f"tools={len(tools)}, known_ids={len(known_ids)}, "
        f"benchmarks={len(benchmark_paths)}, gold={DEFAULT_GOLD.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
