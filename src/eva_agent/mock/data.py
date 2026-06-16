"""Read-only данные системы: внешний backend (если EVA_API_BASE=http) с авто-fallback на мок.

Прокладка: каждый eva_* tool сначала пробует внешний backend (через tools/eva_client, read-only),
а если сервиса нет / нет сессии / эндпоинт упал - отдает мок (fixtures или random). Контракт ответа
(ApiFinding.data) одинаков, поэтому узлы графа не зависят от источника. Боевой бэк не мутируем (только GET).
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from eva_agent.settings import settings
from eva_agent.state import ApiFinding
from eva_agent.tools.entity_ref import resolve_contract_ref
from eva_agent.tools.eva_client import real_get

OWNER_SCOPE = "ws-demo"
_FIXTURES: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "scenarios.json").read_text("utf-8")
)

_ORD_STATUSES = ["draft", "pending", "sent", "registered"]
_FORMS = ["banner", "text_block", "video", "html5"]
_KKTU = [
    {"code": "30.01.01", "title": "Реклама товаров"},
    {"code": "30.02.05", "title": "Реклама услуг"},
    {"code": "30.09.09", "title": "Прочая реклама"},
]


def _finding(tool: str, args: dict[str, Any], data: dict[str, Any]) -> ApiFinding:
    return ApiFinding(tool=tool, args=args, data=data, owner_ref=OWNER_SCOPE)


def _items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items") or payload.get("content") or payload.get("parties") or []
    return []


def _section(name: str) -> dict[str, Any]:
    data = _FIXTURES.get(name, {})
    return data if isinstance(data, dict) else {}


def _synthetic_key(prefix: str, value: str | int) -> str:
    if isinstance(value, int):
        return f"{prefix}-{value}"
    clean = value.strip()
    upper = clean.upper()
    if upper.startswith(f"{prefix}-"):
        return upper
    if clean.isdigit():
        return f"{prefix}-{clean}"
    if prefix == "CT":
        resolved = resolve_contract_ref(clean)
        if isinstance(resolved, int):
            return f"CT-{resolved}"
        if resolved.upper().startswith("CT-"):
            return resolved.upper()
    return clean


def _backend_id(prefix: str, value: str | int) -> str | int:
    if prefix == "CT":
        resolved = resolve_contract_ref(value)
        if isinstance(resolved, int):
            return resolved
        synthetic = _synthetic_key("CT", resolved)
    else:
        synthetic = _synthetic_key(prefix, value)
    marker = f"{prefix}-"
    if synthetic.startswith(marker) and synthetic.removeprefix(marker).isdigit():
        return int(synthetic.removeprefix(marker))
    return synthetic


def _fixture_contract(contract_id: str | int) -> tuple[str, dict[str, Any]]:
    contracts = _section("contracts")
    key = _synthetic_key("CT", contract_id)
    contract = contracts.get(key)
    if not isinstance(contract, dict):
        key = "CT-1"
        contract = contracts.get(key, {})
    return key, contract if isinstance(contract, dict) else {}


def _fixture_counterparty(counterparty_id: str | int) -> tuple[str, dict[str, Any]]:
    counterparties = _section("counterparties")
    key = _synthetic_key("CP", counterparty_id)
    counterparty = counterparties.get(key)
    if not isinstance(counterparty, dict):
        key = "CP-1"
        counterparty = counterparties.get(key, {})
    return key, counterparty if isinstance(counterparty, dict) else {}


# --------------------- маппинг реальных ответов backend ---------------------

def _map_creative(creative: dict[str, Any]) -> dict[str, Any]:
    ord_status = creative.get("ord_status", "draft")
    erid = creative.get("erid_token") or creative.get("erid")
    reasons: list[str] = []
    if ord_status != "registered" and not erid:
        reasons.append("Объявление еще не зарегистрировано во внутренней системе")
    return {
        "id": creative.get("id"),
        "title": creative.get("title", ""),
        "distribution_form": creative.get("distribution_form", ""),
        "ord_status": ord_status,
        "erid_token": erid,
        "contract_id": creative.get("original_contract_id") or creative.get("contract_id"),
        "blocking_reasons": reasons,
        "source": "backend",
    }


def _map_contract(contract: dict[str, Any], contract_id: str | int, source: str) -> dict[str, Any]:
    return {
        "id": contract.get("id", contract_id),
        "number": contract.get("contract_number") or contract.get("number"),
        "contract_date": contract.get("contract_date"),
        "contract_type": contract.get("contract_type") or contract.get("type"),
        "chain_role": contract.get("chain_role"),
        "ord_status": contract.get("ord_status"),
        "price": contract.get("price"),
        "price_not_stipulated": contract.get("price_not_stipulated"),
        "comment": contract.get("comment", ""),
        "source": source,
    }


def _map_counterparty(
    counterparty: dict[str, Any], counterparty_id: str | int, source: str
) -> dict[str, Any]:
    return {
        "id": counterparty.get("id", counterparty_id),
        "name": counterparty.get("name", ""),
        "inn": counterparty.get("inn"),
        "legal_type": counterparty.get("legal_type"),
        "ord_status": counterparty.get("ord_status") or counterparty.get("ord_registration_status"),
        "source": source,
    }


def _platform_name(platform: object) -> object:
    if isinstance(platform, dict):
        return platform.get("name") or platform.get("title") or platform.get("id")
    return platform


def _map_placement(placement: dict[str, Any], source: str) -> dict[str, Any]:
    period = placement.get("period")
    if not isinstance(period, dict):
        period = {
            "start": placement.get("pub_planned_start"),
            "end": placement.get("pub_planned_end"),
        }
    return {
        "id": placement.get("id"),
        "creative_id": placement.get("creative_id"),
        "creative_title": placement.get("creative_title"),
        "platform": _platform_name(placement.get("platform")),
        "status": placement.get("effective_status") or placement.get("status"),
        "period": period,
        "erid": placement.get("erid"),
        "cost": placement.get("cost"),
        "source": source,
    }


def _map_document(document: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "id": document.get("id"),
        "doc_type": document.get("doc_type"),
        "file_name": document.get("file_name"),
        "mime_type": document.get("mime_type"),
        "size": document.get("size"),
        "status": document.get("status", "attached"),
        "source": source,
    }


def _normalized_contract_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"CT-{value}"
    clean = str(value).strip()
    if clean.isdigit():
        return f"CT-{clean}"
    upper = clean.upper()
    if upper.startswith("CT-"):
        return upper
    return None


def _matches_contract(value: object, contract_id: str | int) -> bool:
    expected = _synthetic_key("CT", contract_id)
    return _normalized_contract_value(value) == expected


# --------------------- fallback-мок ---------------------

def _random_creative(creative_id: str) -> dict[str, Any]:
    registered = random.random() < 0.5
    return {
        "title": f"Объявление {creative_id}",
        "distribution_form": random.choice(_FORMS),
        "ord_status": "registered" if registered else random.choice(["draft", "pending_contract"]),
        "erid_token": "ID" + str(random.randint(10**6, 10**7)) if registered else None,
        "contract_id": "CT-1",
        "contract_ord_status": random.choice(_ORD_STATUSES),
        "blocking_reasons": [] if registered else ["Объявление не отправлено на регистрацию"],
    }


# --------------------- read-only tools (real-first) ---------------------

def eva_get_creative_status(creative_id: str) -> ApiFinding:
    real = real_get(f"/api/creatives/{creative_id}")
    if isinstance(real, dict):
        return _finding("eva_get_creative_status", {"creative_id": creative_id}, _map_creative(real))
    if settings.eva_mock_mode == "fixtures":
        data = _FIXTURES["creatives"].get(creative_id) or _FIXTURES["creatives"]["default"]
    else:
        data = _random_creative(creative_id)
    return _finding("eva_get_creative_status", {"creative_id": creative_id}, {"id": creative_id, **data})


def eva_get_contract(contract_id: str | int) -> ApiFinding:
    real_id = _backend_id("CT", contract_id)
    real = real_get(f"/api/contracts/{real_id}")
    if isinstance(real, dict):
        return _finding(
            "eva_get_contract",
            {"contract_id": contract_id},
            _map_contract(real, contract_id, "backend"),
        )
    fixture_id, contract = _fixture_contract(contract_id)
    return _finding(
        "eva_get_contract",
        {"contract_id": contract_id},
        _map_contract({"id": fixture_id, **contract}, fixture_id, "mock"),
    )


def eva_get_counterparty(counterparty_id: str | int) -> ApiFinding:
    real_id = _backend_id("CP", counterparty_id)
    real = real_get(f"/api/counterparties/{real_id}")
    if isinstance(real, dict):
        return _finding(
            "eva_get_counterparty",
            {"counterparty_id": counterparty_id},
            _map_counterparty(real, counterparty_id, "backend"),
        )
    fixture_id, counterparty = _fixture_counterparty(counterparty_id)
    return _finding(
        "eva_get_counterparty",
        {"counterparty_id": counterparty_id},
        _map_counterparty({"id": fixture_id, **counterparty}, fixture_id, "mock"),
    )


def eva_get_contract_parties(contract_id: str | int) -> ApiFinding:
    real_id = _backend_id("CT", contract_id)
    real = real_get(f"/api/contracts/{real_id}/parties")
    if real is not None:
        return _finding(
            "eva_get_contract_parties",
            {"contract_id": contract_id},
            {"id": contract_id, "parties": _items(real), "source": "backend"},
        )
    fixture_id, contract = _fixture_contract(contract_id)
    return _finding(
        "eva_get_contract_parties",
        {"contract_id": contract_id},
        {
            "id": fixture_id,
            "number": contract.get("contract_number") or contract.get("number"),
            "parties": contract.get("parties", []),
            "source": "mock",
        },
    )


def eva_list_unsigned_contracts() -> ApiFinding:
    """Договоры, еще не зарегистрированные во внутренней системе."""
    real = real_get("/api/contracts")
    if real is not None:
        unsigned = [
            {
                "id": c.get("id"),
                "number": c.get("contract_number"),
                "type": c.get("contract_type"),
                "ord_status": c.get("ord_status"),
            }
            for c in _items(real)
            if c.get("ord_status") != "registered"
        ]
        return _finding("eva_list_unsigned_contracts", {}, {"contracts": unsigned, "source": "backend"})
    unsigned = []
    for contract_id, contract in _section("contracts").items():
        if not isinstance(contract, dict):
            continue
        parties = contract.get("parties", [])
        pending_parties = [
            p.get("role")
            for p in parties
            if isinstance(p, dict) and p.get("ord_status") != "registered"
        ]
        if contract.get("ord_status") != "registered" or pending_parties:
            unsigned.append(
                {
                    "id": contract_id,
                    "number": contract.get("contract_number") or contract.get("number"),
                    "pending_parties": pending_parties,
                }
            )
    return _finding("eva_list_unsigned_contracts", {}, {"contracts": unsigned, "source": "mock"})


def eva_list_placements(contract_id: str | int) -> ApiFinding:
    real_id = _backend_id("CT", contract_id)
    real = real_get("/api/placements", params={"contract_id": real_id})
    if real is not None:
        placements = [
            _map_placement(item, "backend")
            for item in _items(real)
            if _matches_contract(item.get("contract_id"), contract_id)
        ]
        if placements:
            return _finding(
                "eva_list_placements",
                {"contract_id": contract_id},
                {
                    "contract_id": _synthetic_key("CT", contract_id),
                    "placements": placements,
                    "count": len(placements),
                    "source": "backend",
                },
            )

    fixture_id = _synthetic_key("CT", contract_id)
    placements = [
        _map_placement(item, "mock")
        for item in _section("placements").get(fixture_id, [])
        if isinstance(item, dict)
    ]
    return _finding(
        "eva_list_placements",
        {"contract_id": contract_id},
        {
            "contract_id": fixture_id,
            "placements": placements,
            "count": len(placements),
            "source": "mock",
        },
    )


def eva_list_contract_documents(contract_id: str | int) -> ApiFinding:
    real_id = _backend_id("CT", contract_id)
    real = real_get(f"/api/contracts/{real_id}/documents")
    if real is not None:
        documents = [_map_document(item, "backend") for item in _items(real)]
        missing = [doc["doc_type"] for doc in documents if doc.get("status") == "missing"]
        return _finding(
            "eva_list_contract_documents",
            {"contract_id": contract_id},
            {
                "contract_id": _synthetic_key("CT", contract_id),
                "documents": documents,
                "missing": missing,
                "count": len(documents),
                "source": "backend",
            },
        )

    fixture_id = _synthetic_key("CT", contract_id)
    documents = [
        _map_document(item, "mock")
        for item in _section("documents").get(fixture_id, [])
        if isinstance(item, dict)
    ]
    missing = [doc["doc_type"] for doc in documents if doc.get("status") == "missing"]
    return _finding(
        "eva_list_contract_documents",
        {"contract_id": contract_id},
        {
            "contract_id": fixture_id,
            "documents": documents,
            "missing": missing,
            "count": len(documents),
            "source": "mock",
        },
    )


def _contract_unsigned(contract: dict[str, Any]) -> bool:
    parties = contract.get("parties", [])
    party_pending = any(
        isinstance(party, dict) and party.get("ord_status") != "registered" for party in parties
    )
    return contract.get("ord_status") != "registered" or party_pending


def _status_matches(contract: dict[str, Any], status_hint: str | None) -> bool:
    if not status_hint:
        return True
    low = status_hint.lower()
    if any(marker in low for marker in ("неподпис", "незарегистр", "unsigned", "draft")):
        return _contract_unsigned(contract)
    status = str(contract.get("ord_status", "")).lower()
    return low in status or status in low


def _date_matches(contract: dict[str, Any], date_hint: str | None) -> bool:
    if not date_hint:
        return True
    low = date_hint.lower()
    contract_date = str(contract.get("contract_date") or "")
    if "вчера" in low:
        return contract_date == (date.today() - timedelta(days=1)).isoformat()
    return date_hint in contract_date


def _contract_text(contract_id: str, contract: dict[str, Any]) -> str:
    parties = contract.get("parties", [])
    party_text = " ".join(str(party.get("name", "")) for party in parties if isinstance(party, dict))
    return " ".join(
        (
            contract_id,
            str(contract.get("number", "")),
            str(contract.get("contract_number", "")),
            str(contract.get("ord_status", "")),
            party_text,
        )
    ).lower()


def eva_search_contracts(
    q: str | None = None, date_hint: str | None = None, status_hint: str | None = None
) -> ApiFinding:
    query = q or ""
    low = query.lower()
    date_filter = date_hint or ("вчера" if "вчера" in low else None)
    status_filter = status_hint or (
        "неподписанные" if any(marker in low for marker in ("неподпис", "незарегистр")) else None
    )
    text_filter = ""
    if query and not any(marker in low for marker in ("последн", "вчера", "неподпис", "незарегистр")):
        text_filter = low

    matches: list[dict[str, Any]] = []
    for contract_id, contract in _section("contracts").items():
        if not isinstance(contract_id, str) or not isinstance(contract, dict):
            continue
        if text_filter and text_filter not in _contract_text(contract_id, contract):
            continue
        if not _date_matches(contract, date_filter):
            continue
        if not _status_matches(contract, status_filter):
            continue
        matches.append(_map_contract({"id": contract_id, **contract}, contract_id, "mock"))

    if "последн" in low:
        matches = sorted(matches, key=lambda item: str(item.get("contract_date") or ""), reverse=True)[:1]

    return _finding(
        "eva_search_contracts",
        {"q": q, "date_hint": date_hint, "status_hint": status_hint},
        {"contracts": matches, "count": len(matches), "source": "mock"},
    )


def eva_kktu_suggest(description: str) -> ApiFinding:
    return _finding("eva_kktu_suggest", {"description": description}, {"suggestions": _KKTU})
