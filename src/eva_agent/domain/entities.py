"""Pydantic-модели сущностей для инструментов данных."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

EntityId = int | str
DatePeriod = tuple[date, date]


class OrdStatus(StrEnum):
    draft = "draft"
    sent = "sent"
    pending = "pending"
    registered = "registered"
    error = "error"
    needs_fix = "needs_fix"


class ContractType(StrEnum):
    service = "service"
    mediation = "mediation"
    amendment = "amendment"


class ChainRole(StrEnum):
    initial = "initial"
    intermediate = "intermediate"


class PartyRole(StrEnum):
    customer = "customer"
    executor = "executor"


class DistributionForm(StrEnum):
    banner = "banner"
    text_block = "text_block"
    video = "video"
    html5 = "html5"


class DocumentEntityType(StrEnum):
    contract = "contract"
    creative = "creative"


class DocumentStatus(StrEnum):
    attached = "attached"
    missing = "missing"


class Contract(BaseModel):
    id: EntityId
    contract_number: str
    contract_date: date
    contract_type: ContractType
    chain_role: ChainRole
    price: Decimal | None = None
    price_not_stipulated: bool = False
    executor_reports_creatives: bool = False
    ord_status: OrdStatus
    ord_submitted_at: datetime | None = None
    comment: str = ""


class ContractParty(BaseModel):
    id: EntityId
    contract_id: EntityId
    counterparty_id: EntityId
    role: PartyRole
    ord_status: OrdStatus


class Counterparty(BaseModel):
    id: EntityId
    name: str
    inn: str
    legal_type: str
    ord_registration_status: OrdStatus | bool


class Creative(BaseModel):
    id: EntityId
    title: str
    distribution_form: DistributionForm
    ord_status: OrdStatus
    erid_token: str | None = None
    contract_id: EntityId
    blocking_reasons: list[str] = Field(default_factory=list)


class CreativeMedia(BaseModel):
    id: EntityId
    creative_id: EntityId
    file_name: str
    mime_type: str
    size: int
    url: str


class Placement(BaseModel):
    id: EntityId
    contract_id: EntityId
    creative_id: EntityId
    status: str
    period: DatePeriod
    platform: str


class Document(BaseModel):
    id: EntityId
    entity_type: DocumentEntityType
    entity_id: EntityId
    doc_type: str
    file_name: str
    mime_type: str
    size: int
    status: DocumentStatus
    url: str | None = None
