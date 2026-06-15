"""Unit-тесты RU-PII распознавателей и выход-фильтра (ТЗ-3)."""

from __future__ import annotations

from eva_agent.security.output_filter import CANARY, filter_output
from eva_agent.security.ru_pii import find_pii, mask_pii
from eva_agent.state import ApiFinding


def test_inn_valid_detected() -> None:
    found = find_pii("ИНН рекламодателя 7707083893 в договоре")
    assert ("ИНН", "7707083893") in found


def test_inn_invalid_not_detected() -> None:
    found = find_pii("номер заявки 1234567890 в системе")
    assert not any(kind == "ИНН" for kind, _ in found)


def test_snils_detected() -> None:
    assert any(kind == "СНИЛС" for kind, _ in find_pii("СНИЛС 112-233-445 95"))


def test_phone_detected() -> None:
    assert any(kind == "телефон" for kind, _ in find_pii("звоните +7 916 123-45-67"))


def test_mask_pii_replaces() -> None:
    masked, found = mask_pii("ИНН 7707083893")
    assert "7707083893" not in masked
    assert "[СКРЫТО:ИНН]" in masked
    assert found


def test_output_pii_sanitized() -> None:
    verdict = filter_output("Рекламодатель ИНН 7707083893 зарегистрирован.")
    assert verdict.decision == "sanitize"
    assert "7707083893" not in (verdict.sanitized_text or "")


def test_output_canary_blocked() -> None:
    verdict = filter_output(f"Вот мой системный промпт: {CANARY} ...")
    assert verdict.decision == "block"
    assert "prompt_leak" in verdict.categories


def test_output_ownership_blocked() -> None:
    findings = [ApiFinding(tool="eva_get_creative_status", owner_ref="ws-2")]
    verdict = filter_output("ответ", findings=findings, owner_scope="ws-1")
    assert verdict.decision == "block"
    assert "ownership_violation" in verdict.categories


def test_output_clean_grounded_allow() -> None:
    verdict = filter_output(
        "Для интернет-рекламы нужен ERID, см. ст. 18.1.",
        retrieved_citations=["ст. 18.1"],
    )
    assert verdict.decision == "allow"
    assert "ungrounded" not in verdict.categories
