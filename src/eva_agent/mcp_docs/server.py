"""FastMCP stdio server for document tools."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from eva_agent.mcp_docs import documents
from eva_agent.mcp_docs.schemas import (
    AttachDocumentResult,
    AttachmentFile,
    DownloadDocumentResult,
    EntityType,
    ListDocumentsResult,
    ReadDocumentResult,
)

mcp = FastMCP("eva-documents")
NonEmptyStr = Annotated[str, Field(min_length=1)]


@mcp.tool()
def list_documents(entity_type: EntityType, entity_id: NonEmptyStr) -> ListDocumentsResult:
    return documents.list_documents(entity_type, entity_id)


@mcp.tool()
def read_document(doc_id: NonEmptyStr) -> ReadDocumentResult:
    return documents.read_document(doc_id)


@mcp.tool()
def download_document(doc_id: NonEmptyStr) -> DownloadDocumentResult:
    return documents.download_document(doc_id)


@mcp.tool()
def attach_document(contract_id: NonEmptyStr, file: AttachmentFile) -> AttachDocumentResult:
    return documents.attach_document(contract_id, file)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
