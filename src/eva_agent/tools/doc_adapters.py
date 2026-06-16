"""Agent-facing document adapters over the shared document layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eva_agent.mcp_docs import documents
from eva_agent.mcp_docs.schemas import AttachmentFile
from eva_agent.mock.data import OWNER_SCOPE
from eva_agent.state import ApiFinding


def _finding(tool: str, args: dict[str, Any], data: dict[str, Any]) -> ApiFinding:
    return ApiFinding(tool=tool, args=args, data=data, owner_ref=OWNER_SCOPE)


def eva_doc_read(doc_id: str) -> ApiFinding:
    result = documents.read_document(doc_id)
    return _finding("eva_doc_read", {"doc_id": doc_id}, result.model_dump())


def eva_doc_download(doc_id: str) -> ApiFinding:
    result = documents.download_document(doc_id)
    return _finding("eva_doc_download", {"doc_id": doc_id}, result.model_dump())


def eva_doc_attach(contract_id: str, file: AttachmentFile | Mapping[str, Any]) -> ApiFinding:
    payload = file if isinstance(file, AttachmentFile) else AttachmentFile.model_validate(dict(file))
    result = documents.attach_document(contract_id, payload)
    args = {"contract_id": contract_id, "file_name": payload.file_name}
    return _finding("eva_doc_attach", args, result.model_dump())

