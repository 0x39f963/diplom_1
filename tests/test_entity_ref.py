from __future__ import annotations

from eva_agent.tools.entity_ref import extract_refs, has_domain_signal, resolve_contract_ref


def test_extract_refs_all_synthetic_ids_and_contract_number() -> None:
    refs = extract_refs("Проверь ct-2, CR-1, cp-3, DOC-4, pl-5 и договор D-2025/249.")

    assert refs.contract_ids == ["CT-2"]
    assert refs.creative_ids == ["CR-1"]
    assert refs.counterparty_ids == ["CP-3"]
    assert refs.document_ids == ["DOC-4"]
    assert refs.placement_ids == ["PL-5"]
    assert refs.contract_numbers == ["Д-2025/249"]
    assert refs.primary_contract == "CT-2"


def test_extract_refs_dedupes_cyrillic_and_latin_contract_number() -> None:
    refs = extract_refs("Д-2025/249 и d-2025/249 относятся к одному договору.")

    assert refs.contract_numbers == ["Д-2025/249"]


def test_extract_refs_counterparty_hints_from_quotes() -> None:
    refs = extract_refs('Контрагент ООО "Площадка" и АО «Медиа» требуют проверки.')

    assert refs.counterparty_hints == ["ООО Площадка", "АО Медиа"]


def test_has_domain_signal_from_id_or_keyword() -> None:
    assert has_domain_signal("Проверь договор CT-1.")
    assert has_domain_signal("Покажи список незакрытых договоров.")
    assert not has_domain_signal("Как сварить кофе?")


def test_resolve_contract_ref_supports_synthetic_numeric_and_number() -> None:
    assert resolve_contract_ref("CT-2") == 2
    assert resolve_contract_ref("77") == 77
    assert resolve_contract_ref("D-2025/249") == "CT-249"
