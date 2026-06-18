from __future__ import annotations

from eva_agent.planner.catalog import AVAILABLE_TOOLS_DEFAULT, CATALOG, available_todo_ids
from eva_agent.tools.selector import EXECUTION_REGISTRY


def test_catalog_tools_exist_in_execution_registry() -> None:
    registry_tools = set(EXECUTION_REGISTRY)

    for spec in CATALOG.values():
        for tool in spec.tools:
            assert tool in registry_tools


def test_available_todo_ids_is_not_empty() -> None:
    assert available_todo_ids()


def test_attach_document_is_enabled_for_write_gate() -> None:
    assert "eva_doc_attach" in AVAILABLE_TOOLS_DEFAULT
    assert "eva_doc_read" in AVAILABLE_TOOLS_DEFAULT
    assert "eva_doc_download" in AVAILABLE_TOOLS_DEFAULT
