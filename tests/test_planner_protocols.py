from __future__ import annotations

from eva_agent.planner.catalog import CATALOG
from eva_agent.planner.protocols import PROTOCOL_CARDS, PROTOCOLS, render_protocol, select_protocol


def test_protocol_todo_ids_exist_in_catalog() -> None:
    catalog_ids = set(CATALOG)

    for spec in PROTOCOLS.values():
        for todo_id in spec.mandatory + spec.optional:
            assert todo_id in catalog_ids

    for card in PROTOCOL_CARDS:
        assert card.protocol_id in PROTOCOLS
        for todo_id in card.todo_template:
            assert todo_id in catalog_ids


def test_select_protocol_typical_phrases() -> None:
    cases = [
        ("что говорит закон о маркировке рекламы", False, "legal_consult", "legal_only"),
        ("кто заказчик по договору CT-1", True, None, "party_lookup"),
        ("покажи карточку договора CT-1", True, None, "contract_card"),
        ("что мешает выпустить креатив CR-2", True, None, "creative_status"),
        ("карточка контрагента CP-1", True, None, "counterparty_card"),
        ("где идут размещения по договору CT-1", True, None, "placement_list"),
        ("каких документов не хватает по договору CT-2", True, None, "document_list"),
        ("дай общий обзор готовности", False, None, "overview"),
        ("можно ли по закону выпустить креатив CR-1", True, None, "mixed_legal_data"),
        ("помоги", False, "need_clarification", "clarify_first"),
    ]

    for query, has_entity, intent_kind, expected in cases:
        assert select_protocol(query, has_entity=has_entity, intent_kind=intent_kind) == expected


def test_mixed_diagnostic_does_not_force_strict_protocol() -> None:
    # mixed_diagnostic - это запрос данных: data-протокол, а не строгий mixed_legal_data.
    assert (
        select_protocol("какие договоры еще не оформлены", has_entity=False, intent_kind="mixed_diagnostic")
        == "overview"
    )
    assert (
        select_protocol("кто заказчик по договору CT-1", has_entity=True, intent_kind="mixed_diagnostic")
        == "party_lookup"
    )
    # строгий протокол - только при реальном пересечении нормы и данных.
    assert (
        select_protocol("можно ли по закону выпустить креатив CR-1", has_entity=True, intent_kind="mixed_diagnostic")
        == "mixed_legal_data"
    )


def test_render_protocol_is_not_empty() -> None:
    for protocol_id in PROTOCOLS:
        text = render_protocol(protocol_id)

        assert protocol_id in text
        assert "ОБЯЗАТЕЛЬНЫЕ TODO" in text
