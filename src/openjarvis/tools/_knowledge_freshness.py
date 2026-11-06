"""How fresh the knowledge index is, phrased for an LLM.

Knowledge tools only see what connector syncs have ingested. Without saying
so, a model asked "what is my latest email?" treats the newest *indexed* item
as the newest one in existence and confidently answers with something days
old. Appending this note to knowledge tool results lets it say "nothing newer
than <time> is indexed" instead.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _parse(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone() if dt.tzinfo else dt


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M (UTC%z)")


def _newest_per_source(conn: sqlite3.Connection) -> dict[str, datetime]:
    # Timestamps carry mixed UTC offsets, so a SQL max() over the strings is
    # not chronological; compare parsed values instead.
    newest: dict[str, datetime] = {}
    rows = conn.execute(
        "SELECT source, timestamp FROM knowledge_chunks "
        "WHERE deleted_at IS NULL AND timestamp IS NOT NULL AND timestamp != '' "
        "GROUP BY source, doc_id"
    )
    for source, ts in rows:
        dt = _parse(ts)
        if dt is None or not source:
            continue
        if dt.tzinfo is None:
            dt = dt.astimezone()
        if source not in newest or dt > newest[source]:
            newest[source] = dt
    return newest


def _last_syncs(sync_db: Path) -> dict[str, datetime]:
    if not sync_db.exists():
        return {}
    conn = sqlite3.connect(f"file:{sync_db}?mode=ro", uri=True)
    try:
        out = {}
        for connector_id, last_sync in conn.execute(
            "SELECT connector_id, last_sync FROM sync_state WHERE last_sync IS NOT NULL"
        ):
            dt = _parse(last_sync)
            if dt is not None:
                out[connector_id] = dt
        return out
    finally:
        conn.close()


def _belongs(connector_id: str, source: str) -> bool:
    # Connector ids extend their source name (gmail -> gmail_imap).
    return connector_id == source or connector_id.startswith(f"{source}_")


def freshness_note(
    store: Any,
    *,
    source: str = "",
    now: Optional[Callable[[], datetime]] = None,
) -> str:
    """Return a short "index is current up to …" note, or ``""``.

    With *source*, only that source (and the connectors feeding it) is
    reported, matching a search that was filtered to it.
    """
    conn = getattr(store, "_conn", None)
    if conn is None:
        return ""
    try:
        newest = _newest_per_source(conn)
        db_path = str(getattr(store, "_db_path", "") or "")
        syncs = (
            _last_syncs(Path(db_path).parent / "sync_state.db")
            if db_path and db_path != ":memory:"
            else {}
        )
    except sqlite3.Error as exc:
        logger.debug("Knowledge freshness lookup failed: %s", exc)
        return ""
    if source:
        newest = {s: dt for s, dt in newest.items() if s == source}
        syncs = {c: dt for c, dt in syncs.items() if _belongs(c, source)}
    if not newest and not syncs:
        return ""

    current = (now or (lambda: datetime.now().astimezone()))()
    parts = [f"[Index freshness] Current time: {_fmt(current)}."]
    if newest:
        parts.append(
            "Newest indexed item per source: "
            + "; ".join(f"{s} {_fmt(dt)}" for s, dt in sorted(newest.items()))
            + "."
        )
    if syncs:
        parts.append(
            "Last completed sync: "
            + "; ".join(f"{c} {_fmt(dt)}" for c, dt in sorted(syncs.items()))
            + "."
        )
    else:
        parts.append("No sync has completed yet, so the index may be partial.")
    parts.append(
        "Anything that arrived after the last sync is not indexed yet: say so "
        "instead of presenting the newest indexed item as the newest one."
    )
    return " ".join(parts)


__all__ = ["freshness_note"]
