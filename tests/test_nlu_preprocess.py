from __future__ import annotations

from eva_agent.nlu.preprocess import is_read_only_domain_command, preprocess


def test_preprocess_extracts_lemmas_and_entities() -> None:
    features = preprocess("контрагентом по договору")

    assert "контрагент" in features.lemmas
    assert "договор" in features.lemmas
    assert set(features.entities) >= {"Counterparty", "Contract"}


def test_preprocess_extracts_date_and_status() -> None:
    features = preprocess("вчерашние неподписанные договоры")

    assert features.date_hint == "yesterday"
    assert features.dates
    assert features.dates[0].start is not None
    assert "unsigned" in features.statuses


def test_preprocess_extracts_action_and_entity_id() -> None:
    features = preprocess("открой DOC-1")

    assert "open" in features.action_verbs
    assert features.entity_ids["documents"] == ["DOC-1"]


def test_gazetteer_matches_client_and_party_role_by_lemmas() -> None:
    client = preprocess("клиент")
    customer = preprocess("заказчик")

    assert "Counterparty" in client.entities
    assert "ContractParty" in customer.entities
    assert "customer" in customer.roles


def test_read_only_domain_command_allows_read_verbs() -> None:
    assert is_read_only_domain_command("Скачай документ DOC-1")
    assert is_read_only_domain_command("Открой карточку контрагента CP-1")
    assert is_read_only_domain_command("Выведи все мои незакрытые договоры")


def test_read_only_domain_command_rejects_write_verbs() -> None:
    assert not is_read_only_domain_command("Приложи документ DOC-3 к договору CT-2")
    assert not is_read_only_domain_command("Удали документ DOC-1")
    assert not is_read_only_domain_command("Отправь договор CT-1 наружу")
