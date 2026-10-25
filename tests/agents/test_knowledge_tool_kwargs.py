"""Knowledge tools enabled via config must be bound to the synced store."""

from __future__ import annotations

import logging

from openjarvis.agents.tool_resolver import (
    instantiate_registered_tool,
    knowledge_tool_kwargs,
)
from openjarvis.connectors.store import KnowledgeStore
from openjarvis.tools.knowledge_search import KnowledgeSearchTool
from openjarvis.tools.knowledge_sql import KnowledgeSQLTool


def _seeded_db(tmp_path):
    path = tmp_path / "knowledge.db"
    store = KnowledgeStore(str(path))
    store.store(
        "Your invoice for October is attached",
        source="gmail",
        doc_type="email",
        doc_id="msg-1",
        title="Invoice October",
        author="billing@example.com",
        timestamp="2026-10-20T09:00:00+00:00",
    )
    store.close()
    return path


def test_other_tools_get_no_store(tmp_path):
    assert knowledge_tool_kwargs("web_search", _seeded_db(tmp_path)) == {}


def test_search_tool_reads_synced_emails(tmp_path):
    tool = KnowledgeSearchTool(
        **knowledge_tool_kwargs("knowledge_search", _seeded_db(tmp_path))
    )
    result = tool.execute(query="invoice", source="gmail")
    assert result.success
    assert "Invoice October" in result.content or "invoice" in result.content


def test_sql_tool_can_order_by_date(tmp_path):
    tool = KnowledgeSQLTool(
        **knowledge_tool_kwargs("knowledge_sql", _seeded_db(tmp_path))
    )
    result = tool.execute(
        query="SELECT title FROM knowledge_chunks ORDER BY timestamp DESC LIMIT 1"
    )
    assert result.success
    assert "Invoice October" in result.content


def test_missing_db_keeps_storeless_behaviour(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        kwargs = knowledge_tool_kwargs("knowledge_search", tmp_path / "none.db")
    assert kwargs == {}
    assert "does not exist yet" in caplog.text
    result = KnowledgeSearchTool(**kwargs).execute(query="x")
    assert not result.success
    assert "No knowledge store configured" in result.content


def test_default_path_is_the_config_dir(tmp_path, monkeypatch):
    import openjarvis.core.config as config_mod

    _seeded_db(tmp_path)
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)
    kwargs = knowledge_tool_kwargs("knowledge_search")
    assert isinstance(kwargs["store"], KnowledgeStore)
    kwargs["store"].close()


def test_instantiate_registered_tool_binds_store(tmp_path, monkeypatch):
    import openjarvis.core.config as config_mod

    _seeded_db(tmp_path)
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)
    tool = instantiate_registered_tool(
        KnowledgeSearchTool, "knowledge_search", engine=None, model="m"
    )
    assert tool.execute(query="invoice").success
