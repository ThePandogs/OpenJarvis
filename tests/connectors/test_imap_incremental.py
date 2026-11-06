"""IMAP syncs must only re-scan recent mail once a full sync has completed."""

from __future__ import annotations

import locale
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from openjarvis.connectors.gmail_imap import _imap_search_criteria
from openjarvis.connectors.imap import IMAPConnector
from openjarvis.connectors.pipeline import IngestionPipeline
from openjarvis.connectors.store import KnowledgeStore
from openjarvis.connectors.sync_engine import SyncEngine
from tests.connectors.test_imap import _sync_client


def test_no_watermark_scans_everything():
    assert _imap_search_criteria(None) == ("ALL",)


def test_watermark_searches_from_the_previous_day():
    since = datetime(2026, 11, 6, 16, 15, tzinfo=timezone.utc)
    assert _imap_search_criteria(since) == ("SINCE", "05-Nov-2026")


def test_watermark_is_converted_to_utc_first():
    # 00:30 on 1 March in UTC+2 is still 28 Feb in UTC.
    since = datetime(2026, 3, 1, 0, 30, tzinfo=timezone(timedelta(hours=2)))
    assert _imap_search_criteria(since) == ("SINCE", "27-Feb-2026")


def test_month_names_ignore_the_process_locale():
    previous = locale.setlocale(locale.LC_TIME)
    try:
        try:
            locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")
        except locale.Error:
            try:
                locale.setlocale(locale.LC_TIME, "Spanish_Spain")
            except locale.Error:
                pytest.skip("no Spanish locale available")
        since = datetime(2026, 1, 2, tzinfo=timezone.utc)
        assert _imap_search_criteria(since) == ("SINCE", "01-Jan-2026")
    finally:
        locale.setlocale(locale.LC_TIME, previous)


def _connector(tmp_path: Path) -> IMAPConnector:
    return IMAPConnector(
        email_address="user@example.org",
        app_password="secret",
        credentials_path=str(tmp_path / "imap.json"),
    )


def test_sync_passes_since_to_imap_search(tmp_path):
    conn = _connector(tmp_path)
    client = _sync_client()
    since = datetime(2026, 11, 6, 9, 0, tzinfo=timezone.utc)
    with patch.object(conn, "_make_imap_client", return_value=client):
        docs = list(conn.sync(since=since))
    assert client.uid.call_args_list[0].args == ("SEARCH", None, "SINCE", "05-Nov-2026")
    assert len(docs) == 1


def test_first_sync_full_then_incremental_through_sync_engine(tmp_path):
    """The watermark only exists after a completed sync, so the first run is a
    full backfill and the next one is incremental."""
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    engine = SyncEngine(
        pipeline=IngestionPipeline(store=store),
        state_db=str(tmp_path / "sync_state.db"),
    )
    conn = _connector(tmp_path)
    try:
        first, second = _sync_client(), _sync_client()
        with patch.object(conn, "_make_imap_client", side_effect=[first, second]):
            engine.sync(conn)
            engine.sync(conn)
        assert first.uid.call_args_list[0].args == ("SEARCH", None, "ALL")
        search = second.uid.call_args_list[0].args
        assert search[:3] == ("SEARCH", None, "SINCE")
        # Re-fetching the overlap must not duplicate the document.
        rows = store._conn.execute(
            "SELECT count(DISTINCT doc_id) FROM knowledge_chunks"
        ).fetchone()
        assert rows[0] == 1
    finally:
        engine.close()
        store.close()
