"""Knowledge tool results say how current the index is."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.tools._knowledge_freshness import freshness_note
from openjarvis.tools.knowledge_search import KnowledgeSearchTool
from openjarvis.tools.knowledge_sql import KnowledgeSQLTool

_NOW = datetime(2026, 11, 6, 17, 30, tzinfo=timezone.utc)


def _store(tmp_path, *, synced=True):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    # Mixed offsets on purpose: the +05:00 one is the OLDER instant even
    # though its string sorts higher.
    for doc_id, ts in (
        ("gmail:1", "2026-10-25T10:02:54+00:00"),
        ("gmail:2", "2026-11-06T14:33:58+00:00"),
        ("gmail:3", "2026-11-06T18:00:00+05:00"),
    ):
        store.store(
            f"invoice {doc_id}",
            source="gmail",
            doc_type="email",
            doc_id=doc_id,
            title=doc_id,
            timestamp=ts,
        )
    if synced:
        conn = sqlite3.connect(tmp_path / "sync_state.db")
        conn.execute(
            "CREATE TABLE sync_state (connector_id TEXT PRIMARY KEY, "
            "items_synced INTEGER, cursor TEXT, last_sync TEXT, error TEXT)"
        )
        conn.execute(
            "INSERT INTO sync_state VALUES "
            "('gmail_imap', 3, NULL, '2026-11-06T16:15:56+00:00', NULL)"
        )
        conn.commit()
        conn.close()
    return store


def test_note_reports_newest_item_and_last_sync(tmp_path):
    store = _store(tmp_path)
    note = freshness_note(store, now=lambda: _NOW)
    store.close()
    newest = datetime(2026, 11, 6, 14, 33, tzinfo=timezone.utc).astimezone()
    synced = datetime(2026, 11, 6, 16, 15, tzinfo=timezone.utc).astimezone()
    assert note.startswith("[Index freshness]")
    assert f"gmail {newest:%Y-%m-%d %H:%M}" in note
    assert f"gmail_imap {synced:%Y-%m-%d %H:%M}" in note
    assert "Current time: 2026-11-06 17:30" in note
    assert "not indexed yet" in note


def test_source_filter_limits_the_note(tmp_path):
    store = _store(tmp_path)
    store.store(
        "standup notes",
        source="slack",
        doc_type="message",
        doc_id="slack:1",
        title="standup",
        timestamp="2026-11-05T09:00:00+00:00",
    )
    try:
        both = freshness_note(store, now=lambda: _NOW)
        assert "slack" in both and "gmail" in both
        gmail_only = freshness_note(store, source="gmail", now=lambda: _NOW)
        assert "gmail_imap" in gmail_only and "slack" not in gmail_only
        result = KnowledgeSearchTool(store=store).execute(
            query="invoice", source="gmail"
        )
        assert "slack" not in result.content
    finally:
        store.close()


def test_note_without_completed_sync_warns_partial(tmp_path):
    store = _store(tmp_path, synced=False)
    note = freshness_note(store, now=lambda: _NOW)
    store.close()
    assert "No sync has completed yet" in note


def test_empty_or_missing_store_gives_no_note(tmp_path):
    empty = KnowledgeStore(str(tmp_path / "empty.db"))
    assert freshness_note(empty) == ""
    empty.close()
    assert freshness_note(None) == ""


def test_tools_append_note_on_success_only(tmp_path):
    store = _store(tmp_path)
    try:
        ok = KnowledgeSQLTool(store=store).execute(
            query="SELECT count(*) AS n FROM knowledge_chunks"
        )
        assert ok.success and "[Index freshness]" in ok.content
        bad = KnowledgeSQLTool(store=store).execute(query="DELETE FROM x")
        assert not bad.success and "[Index freshness]" not in bad.content

        found = KnowledgeSearchTool(store=store).execute(query="invoice")
        assert found.success and "[Index freshness]" in found.content
        # "Nothing found" is exactly when the model must not assume the
        # index is complete.
        none = KnowledgeSearchTool(store=store).execute(query="zzzunmatched")
        assert none.success and "[Index freshness]" in none.content
    finally:
        store.close()


def test_storeless_tool_has_no_note():
    result = KnowledgeSearchTool().execute(query="x")
    assert not result.success
    assert "[Index freshness]" not in result.content
