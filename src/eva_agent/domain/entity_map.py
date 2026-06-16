"""Карта бизнес-сущностей для планировщика."""

from __future__ import annotations

from dataclasses import dataclass

CONTRACT_NUMBER_EXAMPLE = "Д-2025/249"

SYNTHETIC_ID_CONVENTIONS: dict[str, str] = {
    "contract": "CT-N",
    "creative": "CR-N",
    "counterparty": "CP-N",
    "document": "DOC-N",
    "placement": "PL-N",
}

REAL_ID_CONVENTIONS: dict[str, str] = {
    "backend_id": "numeric",
    "contract_number": CONTRACT_NUMBER_EXAMPLE,
}


@dataclass(frozen=True)
class EntitySpec:
    name: str
    description: str
    key_fields: tuple[str, ...]
    relations: tuple[str, ...]
    endpoints: tuple[str, ...]


ENTITY_MAP: tuple[EntitySpec, ...] = (
    EntitySpec(
        name="Contract",
        description="договор",
        key_fields=("id", "contract_number", "contract_date", "contract_type", "chain_role", "ord_status"),
        relations=("parties/creatives/placements/docs по contract_id",),
        endpoints=("GET /api/contracts", "GET /api/contracts/{id}"),
    ),
    EntitySpec(
        name="ContractParty",
        description="сторона договора",
        key_fields=("id", "contract_id", "counterparty_id", "role", "ord_status"),
        relations=("role customer/executor -> Counterparty",),
        endpoints=("GET /api/contracts/{id}/parties",),
    ),
    EntitySpec(
        name="Counterparty",
        description="контрагент",
        key_fields=("id", "name", "inn", "legal_type", "ord_registration_status"),
        relations=("ContractParty; dedup counterparty_id",),
        endpoints=("GET /api/counterparties/{id}",),
    ),
    EntitySpec(
        name="Creative",
        description="креатив",
        key_fields=("id", "title", "distribution_form", "ord_status", "erid_token", "contract_id"),
        relations=("Contract, CreativeMedia, Placement",),
        endpoints=("GET /api/creatives/{id}",),
    ),
    EntitySpec(
        name="CreativeMedia",
        description="файл креатива",
        key_fields=("id", "creative_id", "file_name", "mime_type", "size", "url"),
        relations=("Creative -> media/doc",),
        endpoints=("GET /api/creatives/{id}/media",),
    ),
    EntitySpec(
        name="Placement",
        description="размещение",
        key_fields=("id", "contract_id", "creative_id", "status", "period", "platform"),
        relations=("Contract -> Placement -> Creative",),
        endpoints=("GET /api/contracts/{id}/placements", "GET /api/placements?contract_id="),
    ),
    EntitySpec(
        name="Document",
        description="документ",
        key_fields=("id", "entity_type", "entity_id", "doc_type", "file_name", "status", "url"),
        relations=("Contract/Creative -> Document; missing",),
        endpoints=("GET /api/contracts/{id}/documents", "POST /api/contracts/{id}/documents"),
    ),
)

CHAIN_HINTS: tuple[str, ...] = (
    "стороны Contract->ContractParty(role)->Counterparty",
    "контрагенты Contract->ContractParty->Counterparty dedup",
    "размещения Contract->Placement(+Creative)",
    "документы Contract->Document missing",
    "вчера неподписанные Contract date=вчера,status!=registered",
)


def render_entity_map() -> str:
    lines = [
        "ID: "
        + ", ".join(f"{entity}={pattern}" for entity, pattern in SYNTHETIC_ID_CONVENTIONS.items())
        + f"; real numeric id, number={CONTRACT_NUMBER_EXAMPLE}.",
    ]
    for entity in ENTITY_MAP:
        endpoint_text = ",".join(endpoint.removeprefix("GET ") for endpoint in entity.endpoints)
        if entity.name == "Document":
            endpoint_text = "GET/POST /api/contracts/{id}/documents"
        lines.append(
            f"{entity.name}: {entity.description}; f {','.join(entity.key_fields)}; "
            f"r {','.join(entity.relations)}; api {endpoint_text}."
        )
    lines.append("Chains: " + "; ".join(CHAIN_HINTS) + ".")
    return "\n".join(lines)
