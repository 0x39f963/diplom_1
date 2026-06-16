"""Shared document access for MCP tools and local eva_doc adapters."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from eva_agent.mcp_docs.schemas import (
    AttachDocumentResult,
    AttachmentFile,
    DocumentRef,
    DocumentSource,
    DownloadDocumentResult,
    EntityType,
    GuardResult,
    ListDocumentsResult,
    ReadDocumentResult,
)
from eva_agent.mcp_docs.security import guard_attachment_file
from eva_agent.security.verdict import GuardVerdict
from eva_agent.tools.eva_client import real_get

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CATALOG_PATH = _FIXTURES_DIR / "documents.json"
_BLOBS_DIR = _FIXTURES_DIR / "blobs"
_PREVIEW_CHARS = 2000
_DEFAULT_MAX_BYTES = 1_048_576

_RUNTIME_ATTACHMENTS: dict[str, list[dict[str, Any]]] = {}
_RUNTIME_BLOBS: dict[str, bytes] = {}


class DocumentLookupError(LookupError):
    """Raised when a document id cannot be resolved."""


class DocumentUnavailableError(LookupError):
    """Raised when a known document has no downloadable artifact."""


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_CATALOG_PATH.read_text("utf-8")))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _max_download_bytes() -> int:
    raw = os.environ.get("MCP_DOC_MAX_BYTES")
    if raw is None:
        return _DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BYTES
    return max(value, 0)


def _items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "content", "media", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _mime_from_media(media_type: object) -> str:
    mapping = {
        "image": "image/*",
        "video": "video/*",
        "audio": "audio/*",
        "zip": "application/zip",
        "other": "application/octet-stream",
    }
    return mapping.get(str(media_type or "other").lower(), str(media_type or "application/octet-stream"))


def _blob_path(entry: Mapping[str, Any]) -> Path | None:
    blob_name = entry.get("blob_name")
    if not isinstance(blob_name, str) or not blob_name:
        return None
    return _BLOBS_DIR / blob_name


def _entry_size(entry: Mapping[str, Any]) -> int:
    path = _blob_path(entry)
    if path is not None and path.exists():
        return path.stat().st_size
    size = entry.get("size", 0)
    return size if isinstance(size, int) and size >= 0 else 0


def _entry_to_ref(entry: Mapping[str, Any], source: DocumentSource) -> DocumentRef:
    locator = entry.get("locator")
    path = _blob_path(entry)
    if locator is None and path is not None:
        locator = str(path.resolve())
    return DocumentRef(
        doc_id=str(entry["doc_id"]),
        entity_type=cast(EntityType, entry["entity_type"]),
        entity_id=str(entry["entity_id"]),
        doc_type=entry["doc_type"],
        file_name=str(entry.get("file_name") or ""),
        mime_type=str(entry.get("mime_type") or "application/octet-stream"),
        size=_entry_size(entry),
        status=entry.get("status", "attached"),
        source=source,
        locator=locator if isinstance(locator, str) else None,
    )


def _mock_entries(entity_type: EntityType, entity_id: str) -> list[dict[str, Any]]:
    documents = _catalog().get("documents", {})
    if not isinstance(documents, dict):
        return []
    by_type = documents.get(entity_type, {})
    if not isinstance(by_type, dict):
        return []
    entries = by_type.get(entity_id, [])
    if not isinstance(entries, list):
        return []
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _runtime_entries(entity_type: EntityType, entity_id: str) -> list[dict[str, Any]]:
    if entity_type != "contract":
        return []
    return [dict(entry) for entry in _RUNTIME_ATTACHMENTS.get(entity_id, [])]


def _backend_creative_documents(creative_id: str) -> ListDocumentsResult | None:
    real = real_get(f"/api/creatives/{creative_id}/media")
    if real is None:
        return None
    documents: list[DocumentRef] = []
    for item in _items(real):
        media_id = item.get("id", len(documents) + 1)
        locator = item.get("file_ref") or item.get("url")
        documents.append(
            DocumentRef(
                doc_id=f"cr-{creative_id}-{media_id}",
                entity_type="creative",
                entity_id=creative_id,
                doc_type="creative_media",
                file_name=str(item.get("name") or item.get("file_name") or f"media-{media_id}"),
                mime_type=str(item.get("mime_type") or _mime_from_media(item.get("media_type"))),
                size=0,
                status="attached",
                source="backend",
                locator=str(locator) if locator else None,
            )
        )
    return ListDocumentsResult(
        entity_type="creative",
        entity_id=creative_id,
        source="backend",
        documents=documents,
    )


def list_documents(entity_type: EntityType, entity_id: str) -> ListDocumentsResult:
    if entity_type == "creative":
        backend = _backend_creative_documents(entity_id)
        if backend is not None:
            return backend

    entries = [*_mock_entries(entity_type, entity_id), *_runtime_entries(entity_type, entity_id)]
    return ListDocumentsResult(
        entity_type=entity_type,
        entity_id=entity_id,
        source="mock",
        documents=[_entry_to_ref(entry, "mock") for entry in entries],
    )


def _entity_from_doc_id(doc_id: str) -> tuple[EntityType, str]:
    if doc_id.startswith("co-"):
        rest = doc_id.removeprefix("co-")
        if "-" in rest:
            return "contract", rest.rsplit("-", 1)[0]
    if doc_id.startswith("cr-"):
        rest = doc_id.removeprefix("cr-")
        if "-" in rest:
            return "creative", rest.rsplit("-", 1)[0]
    msg = f"cannot resolve document id: {doc_id}"
    raise DocumentLookupError(msg)


def _lookup(doc_id: str) -> tuple[DocumentRef, dict[str, Any]]:
    entity_type, entity_id = _entity_from_doc_id(doc_id)
    result = list_documents(entity_type, entity_id)
    entries = [*_mock_entries(entity_type, entity_id), *_runtime_entries(entity_type, entity_id)]
    by_id = {str(entry.get("doc_id")): entry for entry in entries}
    for document in result.documents:
        if document.doc_id == doc_id:
            return document, by_id.get(doc_id, {})
    msg = f"document not found: {doc_id}"
    raise DocumentLookupError(msg)


def _is_binary_preview(mime_type: str) -> bool:
    clean = mime_type.lower().split(";", 1)[0].strip()
    return clean.startswith(("image/", "video/", "audio/")) or clean in {
        "application/zip",
        "application/octet-stream",
    }


def _entry_bytes(document: DocumentRef, entry: Mapping[str, Any]) -> bytes | None:
    if document.doc_id in _RUNTIME_BLOBS:
        return _RUNTIME_BLOBS[document.doc_id]
    path = _blob_path(entry)
    if path is not None and path.exists():
        return path.read_bytes()
    return None


def read_document(doc_id: str) -> ReadDocumentResult:
    document, entry = _lookup(doc_id)
    if document.status == "missing":
        preview = "document is missing, preview not available"
        return ReadDocumentResult(document=document, text_preview=preview, truncated=False)
    if _is_binary_preview(document.mime_type):
        preview = f"binary media, preview not available; size={document.size}"
        return ReadDocumentResult(document=document, text_preview=preview, truncated=False)

    raw = _entry_bytes(document, entry)
    if raw is None:
        preview = document.locator or "preview not available"
    else:
        preview = raw.decode("utf-8", "ignore")
    truncated = len(preview) > _PREVIEW_CHARS
    return ReadDocumentResult(
        document=document,
        text_preview=preview[:_PREVIEW_CHARS],
        truncated=truncated,
    )


def download_document(doc_id: str) -> DownloadDocumentResult:
    document, entry = _lookup(doc_id)
    if document.status == "missing":
        msg = f"document is missing: {doc_id}"
        raise DocumentUnavailableError(msg)

    raw = _entry_bytes(document, entry)
    if raw is not None:
        if len(raw) <= _max_download_bytes():
            return DownloadDocumentResult(
                doc_id=document.doc_id,
                file_name=document.file_name,
                mime_type=document.mime_type,
                size=len(raw),
                encoding="base64",
                content=base64.b64encode(raw).decode("ascii"),
            )
        path = _blob_path(entry)
        if path is not None:
            return DownloadDocumentResult(
                doc_id=document.doc_id,
                file_name=document.file_name,
                mime_type=document.mime_type,
                size=len(raw),
                encoding="path",
                content=str(path.resolve()),
            )

    if document.locator:
        return DownloadDocumentResult(
            doc_id=document.doc_id,
            file_name=document.file_name,
            mime_type=document.mime_type,
            size=document.size,
            encoding="locator",
            content=document.locator,
        )

    msg = f"document artifact unavailable: {doc_id}"
    raise DocumentUnavailableError(msg)


def _disabled_result() -> AttachDocumentResult:
    guard = GuardVerdict(
        decision="allow",
        risk_score=0.0,
        categories=["mutation_disabled"],
        reason="mutating tools disabled",
    )
    return AttachDocumentResult(status="disabled", guard=GuardResult.from_verdict(guard))


def _coerce_file(file: AttachmentFile | Mapping[str, Any]) -> AttachmentFile:
    if isinstance(file, AttachmentFile):
        return file
    return AttachmentFile.model_validate(dict(file))


def _next_doc_id(contract_id: str) -> str:
    existing = {document.doc_id for document in list_documents("contract", contract_id).documents}
    seq = len(existing) + 1
    while f"co-{contract_id}-{seq}" in existing:
        seq += 1
    return f"co-{contract_id}-{seq}"


def _store_mock_attachment(contract_id: str, file: AttachmentFile) -> DocumentRef:
    raw = base64.b64decode(file.content_b64, validate=True)
    doc_id = _next_doc_id(contract_id)
    entry: dict[str, Any] = {
        "doc_id": doc_id,
        "entity_type": "contract",
        "entity_id": contract_id,
        "doc_type": file.doc_type,
        "file_name": file.file_name,
        "mime_type": file.mime_type,
        "size": len(raw),
        "status": "attached",
        "locator": f"memory://{doc_id}",
    }
    _RUNTIME_ATTACHMENTS.setdefault(contract_id, []).append(entry)
    _RUNTIME_BLOBS[doc_id] = raw
    return _entry_to_ref(entry, "mock")


def attach_document(contract_id: str, file: AttachmentFile | Mapping[str, Any]) -> AttachDocumentResult:
    payload = _coerce_file(file)
    if not _env_bool("EVA_MCP_ALLOW_MUTATIONS", default=False):
        return _disabled_result()

    guard = guard_attachment_file(payload)
    guard_result = GuardResult.from_verdict(guard)
    if not guard.passed:
        return AttachDocumentResult(status="blocked", guard=guard_result)

    document = _store_mock_attachment(contract_id, payload)
    return AttachDocumentResult(status="attached", guard=guard_result, document=document)

