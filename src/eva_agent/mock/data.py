"""Read-only данные системы: внешний backend (если EVA_API_BASE=http) с авто-fallback на мок.

Прокладка: каждый eva_* tool сначала пробует внешний backend (через tools/eva_client, read-only),
а если сервиса нет / нет сессии / эндпоинт упал - отдает мок (fixtures или random). Контракт ответа
(ApiFinding.data) одинаков, поэтому узлы графа не зависят от источника. Боевой бэк не мутируем (только GET).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from eva_agent.settings import settings
from eva_agent.state import ApiFinding
from eva_agent.tools.eva_client import real_get

OWNER_SCOPE = "ws-demo"
_FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "scenarios.json").read_text("utf-8"))

_ORD_STATUSES = ["draft", "pending", "sent", "registered"]
_FORMS = ["banner", "text_block", "video", "html5"]
_KKTU = [
    {"code": "30.01.01", "title": "Реклама товаров"},
    {"code": "30.02.05", "title": "Реклама услуг"},
    {"code": "30.09.09", "title": "Прочая реклама"},
]


def _finding(tool: str, args: dict, data: dict) -> ApiFinding:
    return ApiFinding(tool=tool, args=args, data=data, owner_ref=OWNER_SCOPE)


def _items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items") or payload.get("content") or payload.get("parties") or []
    return []


# --------------------- маппинг реальных ответов backend ---------------------

def _map_creative(creative: dict) -> dict:
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


# --------------------- fallback-мок ---------------------

def _random_creative(creative_id: str) -> dict:
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


def eva_get_contract_parties(contract_id: str) -> ApiFinding:
    real = real_get(f"/api/contracts/{contract_id}/parties")
    if real is not None:
        return _finding(
            "eva_get_contract_parties",
            {"contract_id": contract_id},
            {"id": contract_id, "parties": _items(real), "source": "backend"},
        )
    contract = _FIXTURES["contracts"].get(contract_id, _FIXTURES["contracts"]["CT-1"])
    return _finding(
        "eva_get_contract_parties",
        {"contract_id": contract_id},
        {"id": contract_id, "number": contract["number"], "parties": contract["parties"]},
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
    unsigned = [
        {"id": cid, "number": c["number"],
         "pending_parties": [p["role"] for p in c["parties"] if p["ord_status"] != "registered"]}
        for cid, c in _FIXTURES["contracts"].items()
        if any(p["ord_status"] != "registered" for p in c["parties"])
    ]
    return _finding("eva_list_unsigned_contracts", {}, {"contracts": unsigned})


def eva_kktu_suggest(description: str) -> ApiFinding:
    return _finding("eva_kktu_suggest", {"description": description}, {"suggestions": _KKTU})
