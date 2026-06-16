"""Pydantic contracts for document MCP tools and local adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.security.verdict import GuardVerdict

EntityType = Literal["contract", "creative"]
DocumentType = Literal["contract_file", "act", "annex", "creative_media"]
AttachDocumentType = Literal["contract_file", "act", "annex"]
DocumentStatus = Literal["attached", "missing"]
DocumentSource = Literal["backend", "mock"]
DownloadEncoding = Literal["base64", "path", "locator"]
AttachStatus = Literal["attached", "blocked", "disabled"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentRef(_StrictModel):
    doc_id: str = Field(min_length=1)
    entity_type: EntityType
    entity_id: str = Field(min_length=1)
    doc_type: DocumentType
    file_name: str
    mime_type: str
    size: int = Field(ge=0)
    status: DocumentStatus
    source: DocumentSource
    locator: str | None = None


class ListDocumentsInput(_StrictModel):
    entity_type: EntityType
    entity_id: str = Field(min_length=1)


class DocumentIdInput(_StrictModel):
    doc_id: str = Field(min_length=1)


class ListDocumentsResult(_StrictModel):
    entity_type: EntityType
    entity_id: str
    source: DocumentSource
    documents: list[DocumentRef]


class ReadDocumentResult(_StrictModel):
    document: DocumentRef
    text_preview: str
    truncated: bool


class DownloadDocumentResult(_StrictModel):
    doc_id: str
    file_name: str
    mime_type: str
    size: int = Field(ge=0)
    encoding: DownloadEncoding
    content: str


class AttachmentFile(_StrictModel):
    file_name: str = Field(min_length=1, max_length=256)
    content_b64: str
    mime_type: str = "application/octet-stream"
    doc_type: AttachDocumentType = "annex"


class AttachDocumentInput(_StrictModel):
    contract_id: str = Field(min_length=1)
    file: AttachmentFile


class GuardResult(_StrictModel):
    decision: str
    risk_score: float = Field(ge=0.0, le=1.0)
    categories: list[str] = Field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_verdict(cls, verdict: GuardVerdict) -> GuardResult:
        return cls(
            decision=verdict.decision,
            risk_score=verdict.risk_score,
            categories=verdict.categories,
            reason=verdict.reason,
        )


class AttachDocumentResult(_StrictModel):
    status: AttachStatus
    guard: GuardResult
    document: DocumentRef | None = None
